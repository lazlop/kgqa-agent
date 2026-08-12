#!/usr/bin/env python3
"""
parallel_benchmark.py

Runs every benchmark config in a directory (same config files benchmark.py already
accepts) as its own `benchmark.py` subprocess, spreading them across a pool of API keys
so several configs run at once instead of queuing behind each other.

Each key gets a fixed number of worker slots (--agents-per-key). Total concurrency is
`len(api-keys) * agents-per-key`; that many long-lived worker threads pull configs off a
shared queue and run them one at a time on their assigned key, so no key ever has more
than --agents-per-key requests in flight.

A worker never mutates the original config file: it writes a temp copy with the
"api-key" field swapped in, passes that to benchmark.py, and deletes the temp copy the
moment the subprocess exits (success or failure) so the key doesn't sit around on disk
longer than it has to.

Usage
-----
python parallel_benchmark.py \\
    --config-dir ../configs/rerun_batch \\
    --keys-config ../configs/api_keys.json \\
    --agents-per-key 2

keys-config format:
    {"api-keys": ["sk-...", "sk-...", "sk-..."]}
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Queue

SCRIPT_DIR = Path(__file__).parent


def load_keys(keys_config_path: Path) -> list[str]:
    with open(keys_config_path) as f:
        data = json.load(f)
    keys = data.get("api-keys")
    if not keys:
        print(f"No 'api-keys' list found in {keys_config_path}", file=sys.stderr)
        sys.exit(1)
    return keys


def make_patched_config(original_path: Path, api_key: str, tmp_dir: Path) -> Path:
    """Write a temp copy of a config with its api-key field overridden."""
    with open(original_path) as f:
        config = json.load(f)
    config["api-key"] = api_key

    tmp_path = tmp_dir / f"{original_path.stem}_{threading.get_ident()}_{int(time.time() * 1000)}.json"
    with open(tmp_path, "w") as f:
        json.dump(config, f, indent=2)
    return tmp_path


def worker(
    key_index: int,
    api_key: str,
    task_queue: "Queue[Path]",
    tmp_dir: Path,
    log_dir: Path,
    results: list[dict],
    results_lock: threading.Lock,
) -> None:
    while True:
        try:
            config_path = task_queue.get_nowait()
        except Exception:
            return

        label = f"key#{key_index} <- {config_path.name}"
        patched_path = make_patched_config(config_path, api_key, tmp_dir)
        log_path = log_dir / f"{config_path.stem}.log"

        print(f"▶ starting {label} (log: {log_path})")
        start = time.time()
        try:
            with open(log_path, "w") as log_file:
                proc = subprocess.run(
                    [sys.executable, "benchmark.py", "--config", str(patched_path)],
                    cwd=SCRIPT_DIR,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            elapsed = time.time() - start
            status = "ok" if proc.returncode == 0 else f"failed (exit {proc.returncode})"
            print(f"{'✅' if proc.returncode == 0 else '❌'} finished {label} in {elapsed / 60:.1f}m — {status}")
            with results_lock:
                results.append({
                    "config": config_path.name,
                    "key_index": key_index,
                    "status": status,
                    "elapsed_min": round(elapsed / 60, 1),
                    "log": str(log_path),
                })
        except Exception as e:
            print(f"❌ {label} crashed before/while launching subprocess: {e}")
            with results_lock:
                results.append({"config": config_path.name, "key_index": key_index, "status": f"error: {e}"})
        finally:
            patched_path.unlink(missing_ok=True)
            task_queue.task_done()


def main():
    parser = argparse.ArgumentParser(
        description="Run every config in a directory through benchmark.py, spread across a pool of API keys."
    )
    parser.add_argument("--config-dir", required=True, help="Directory of benchmark config JSON files.")
    parser.add_argument("--keys-config", required=True, help='JSON file: {"api-keys": ["sk-...", ...]}')
    parser.add_argument("--agents-per-key", type=int, default=1, help="Concurrent workers per key (default: 1).")
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Where to write per-config subprocess logs (default: ../logs/parallel_benchmark_<timestamp>/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the config -> key assignment plan without launching any subprocesses.",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    if not config_dir.is_dir():
        print(f"Not a directory: {config_dir}", file=sys.stderr)
        sys.exit(1)

    config_paths = sorted(config_dir.glob("*.json"))
    if not config_paths:
        print(f"No config files found in {config_dir}", file=sys.stderr)
        sys.exit(1)

    keys = load_keys(Path(args.keys_config))
    total_slots = len(keys) * args.agents_per_key

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) if args.log_dir else SCRIPT_DIR / ".." / "logs" / f"parallel_benchmark_{timestamp}"
    tmp_dir = log_dir / "tmp_configs"
    print(f"Found {len(config_paths)} config(s) in {config_dir}")
    print(f"{len(keys)} key(s) x {args.agents_per_key} agent(s)/key = {total_slots} concurrent worker(s)")
    print()

    if args.dry_run:
        # No threads here on purpose: real assignment depends on which subprocess
        # finishes first, which a dry run can't predict anyway. This just previews the
        # concurrency structure (how configs round-robin across slots/keys), deterministically.
        print("(dry run — no subprocesses will be launched)")
        print()
        for i, config_path in enumerate(config_paths):
            slot = i % total_slots
            key_index = slot % len(keys)
            print(f"  slot#{slot} (key#{key_index}) <- {config_path.name}")
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logs: {log_dir}")
    print()

    task_queue: "Queue[Path]" = Queue()
    for p in config_paths:
        task_queue.put(p)

    results: list[dict] = []
    results_lock = threading.Lock()

    threads = []
    for slot in range(total_slots):
        key_index = slot % len(keys)
        t = threading.Thread(
            target=worker,
            args=(key_index, keys[key_index], task_queue, tmp_dir, log_dir, results, results_lock),
            daemon=True,
        )
        threads.append(t)
        t.start()

    task_queue.join()
    for t in threads:
        t.join()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for r in results:
        extra = f" ({r['elapsed_min']}m, log: {r['log']})" if "elapsed_min" in r else ""
        print(f"  key#{r['key_index']}  {r['config']:<45} {r['status']}{extra}")

    failures = [r for r in results if r["status"] not in ("ok",)]
    if failures:
        print(f"\n{len(failures)}/{len(results)} config(s) did not finish cleanly — check logs above.")
        sys.exit(1)

    try:
        tmp_dir.rmdir()
    except OSError:
        pass  # leftover temp files from a crashed worker; harmless, left for inspection


if __name__ == "__main__":
    main()
