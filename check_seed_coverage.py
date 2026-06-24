#!/usr/bin/env python3
"""
check_seed_coverage.py

Checks that all 5 seeds (42-46) are present for every model-dataset
combination defined in RUN_PLAN. Prints a summary table of missing seeds.

NOTE: reads results/csv/per_seed_runs.csv — run aggregate_results.py first
to ensure it reflects the latest completed experiments.
"""

import csv
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# RUN_PLAN — mirrors run_all_experiments.sh exactly
# ---------------------------------------------------------------------------
RUN_PLAN = {
    "ASTD":           ["marbert", "arabert", "egybert"],
    "ArSAS":          ["marbert", "arabert", "egybert"],
    "AfriSenti_ARY":  ["marbert", "arabert", "darijabert"],
    "AfriSenti_ARQ":  ["marbert", "arabert", "dziribert"],
    "LABR":           ["marbert", "arabert"],
    "MACcorpus":      ["marbert", "arabert", "darijabert"],
    "HARD":           ["marbert", "arabert"],
}

EXPECTED_SEEDS = set(range(42, 47))

# ---------------------------------------------------------------------------
# Load per-seed CSV
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CSV  = ROOT / "results" / "csv" / "per_seed_runs.csv"

if not CSV.exists():
    raise SystemExit(f"ERROR: file not found: {CSV}")

present = defaultdict(lambda: defaultdict(set))
with open(CSV, newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        present[row["dataset"]][row["model"]].add(int(row["seed"]))

# ---------------------------------------------------------------------------
# Check coverage
# ---------------------------------------------------------------------------
missing_rows = []
ok_count     = 0

for dataset, models in RUN_PLAN.items():
    for model in models:
        seeds_found   = present[dataset].get(model, set())
        seeds_missing = sorted(EXPECTED_SEEDS - seeds_found)
        seeds_found_s = sorted(seeds_found)
        if seeds_missing:
            missing_rows.append((dataset, model, seeds_missing, seeds_found_s))
        else:
            ok_count += 1

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
total = sum(len(m) for m in RUN_PLAN.values())

print()
print(f"Seed coverage check  —  expected seeds: {sorted(EXPECTED_SEEDS)}")
print(f"Combinations checked: {total}   OK: {ok_count}   Missing: {len(missing_rows)}")
print()

if not missing_rows:
    print("All combinations have all 5 seeds. Nothing missing.")
else:
    col_ds    = max(len(r[0]) for r in missing_rows)
    col_model = max(len(r[1]) for r in missing_rows)
    col_ds    = max(col_ds,    len("Dataset"))
    col_model = max(col_model, len("Model"))

    header = (f"{'Dataset':<{col_ds}}  {'Model':<{col_model}}  "
              f"{'Missing Seeds':<25}  Present Seeds")
    print(header)
    print("-" * len(header))
    for dataset, model, miss, found in missing_rows:
        label = "ALL MISSING" if not found else str(miss)
        print(f"{dataset:<{col_ds}}  {model:<{col_model}}  {label:<25}  {found}")

print()