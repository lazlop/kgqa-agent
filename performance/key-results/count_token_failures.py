import os
import pandas as pd

TOKEN_LIMIT = 200_000

csv_files = [f for f in os.listdir(".") if f.endswith(".csv")]

print(f"Token cap failure report (>{TOKEN_LIMIT:,} total tokens)\n")
print(f"{'Condition':<60} {'Exceeded':>8} {'Total':>7} {'% Failed':>9}")
print("-" * 88)

for fname in sorted(csv_files):
    df = pd.read_csv(fname, usecols=["total_tokens"])
    exceeded = (df["total_tokens"] > TOKEN_LIMIT).sum()
    total = len(df)
    pct = 100 * exceeded / total if total > 0 else 0
    label = fname.replace(".csv", "")
    print(f"{label:<60} {exceeded:>8} {total:>7} {pct:>8.1f}%")

print()
