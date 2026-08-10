#!/usr/bin/env python3
"""
Add a UNIQUE_ID column to benchmark result CSVs, derived from the English-language
question text rather than `query_id`.

Why: `query_id` (e.g. "LBNL_001") identifies a *group* of paraphrased questions, not a
single question -- each group holds several distinct English phrasings of the same
underlying query. Comparing two CSVs (e.g. an old run vs a rerun) by filtering on
`query_id` and taking the first match silently assumes both files iterate that group's
questions in the same order. That assumption held for the specific pair checked during
this investigation (zero mismatches across all 27 groups), but nothing guarantees it in
general -- a resumed/appended run, a reordered QA JSON, or a different config could
desync row order between two files sharing the same `query_id` values, silently pairing
up unrelated questions.

UNIQUE_ID is a stable hash of the (whitespace-normalized) `question` column, so two rows
from different CSVs with the same UNIQUE_ID are guaranteed to be the same question,
independent of row order or which `query_id` group they were logged under.
"""

import argparse
import csv
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

csv.field_size_limit(10 * 1024 * 1024)


def normalize_question(question: str) -> str:
    """Collapse whitespace so two questions differing only in formatting hash equal."""
    return " ".join(question.split())


def question_to_unique_id(question: str, length: int = 16) -> str:
    """Deterministic, collision-resistant id for a question's text.

    sha256 truncated to `length` hex chars (default 16 -> 64 bits) -- ample collision
    resistance for benchmark sizes in the hundreds to low thousands of questions.
    """
    normalized = normalize_question(question)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"Q_{digest[:length]}"


def add_unique_id_column(input_path: Path, output_path: Path) -> list[dict]:
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = ["UNIQUE_ID"] + list(reader.fieldnames)
        rows = list(reader)

    for row in rows:
        row["UNIQUE_ID"] = question_to_unique_id(row["question"])

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def verify(rows: list[dict], label: str) -> Counter:
    """Check for hash collisions (same UNIQUE_ID, different normalized question text)."""
    id_to_questions = {}
    collisions = 0
    for row in rows:
        uid = row["UNIQUE_ID"]
        norm_q = normalize_question(row["question"])
        if uid in id_to_questions and id_to_questions[uid] != norm_q:
            collisions += 1
            print(f"  ⚠️  COLLISION in {label}: {uid} maps to two different questions:")
            print(f"      {id_to_questions[uid][:80]!r}")
            print(f"      {norm_q[:80]!r}")
        id_to_questions[uid] = norm_q

    print(f"  {label}: {len(rows)} rows, {len(id_to_questions)} distinct UNIQUE_IDs, {collisions} collisions")
    return Counter(row["UNIQUE_ID"] for row in rows)


def main():
    parser = argparse.ArgumentParser(
        description="Add a question-text-hash UNIQUE_ID column to one or more benchmark CSVs."
    )
    parser.add_argument("input_csvs", nargs="+", help="Path(s) to input CSV file(s)")
    parser.add_argument(
        "--suffix",
        default="_with_uid",
        help="Suffix inserted before .csv in output filenames (default: _with_uid)",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("CSV_A", "CSV_B"),
        default=None,
        help="After processing, report UNIQUE_IDs present in one file but not the other",
    )
    args = parser.parse_args()

    all_rows = {}
    for input_str in args.input_csvs:
        input_path = Path(input_str)
        output_path = input_path.with_name(input_path.stem + args.suffix + input_path.suffix)
        print(f"Processing {input_path} -> {output_path}")
        rows = add_unique_id_column(input_path, output_path)
        verify(rows, str(input_path.name))
        all_rows[str(input_path)] = rows
        print()

    if args.compare:
        a_path, b_path = args.compare
        a_rows = all_rows.get(a_path) or list(csv.DictReader(open(a_path, newline="", encoding="utf-8")))
        b_rows = all_rows.get(b_path) or list(csv.DictReader(open(b_path, newline="", encoding="utf-8")))
        a_ids = {r["UNIQUE_ID"] if "UNIQUE_ID" in r else question_to_unique_id(r["question"]) for r in a_rows}
        b_ids = {r["UNIQUE_ID"] if "UNIQUE_ID" in r else question_to_unique_id(r["question"]) for r in b_rows}
        print(f"Comparing {a_path} vs {b_path}:")
        print(f"  only in A: {len(a_ids - b_ids)}")
        print(f"  only in B: {len(b_ids - a_ids)}")
        print(f"  in both:   {len(a_ids & b_ids)}")


if __name__ == "__main__":
    main()
