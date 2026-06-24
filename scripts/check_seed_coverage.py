#!/usr/bin/env python3
"""
scripts/check_seed_coverage.py

Checks that all 5 seeds (42-46) are present for every model-dataset
combination defined in RUN_PLAN by inspecting the checkpoints directory
directly — does not rely on any aggregated CSV.

A seed is considered complete when its directory contains at least one
checkpoint subdirectory (checkpoint-*).
"""

from pathlib import Path

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

DATASET_KEY = {
    "ASTD":          "astd",
    "ArSAS":         "arsas",
    "AfriSenti_ARY": "afrisenti_ary",
    "AfriSenti_ARQ": "afrisenti_arq",
    "LABR":          "labr",
    "MACcorpus":     "maccorpus",
    "HARD":          "hard",
}

EXPECTED_SEEDS = list(range(42, 47))

# ---------------------------------------------------------------------------
# Check checkpoints
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parent.parent
CHECKPOINTS  = ROOT / "checkpoints"

missing_rows = []
ok_count     = 0

for dataset, models in RUN_PLAN.items():
    ds_key = DATASET_KEY[dataset]
    for model in models:
        run_dir = CHECKPOINTS / f"{model}_{ds_key}_split_42"
        seeds_found   = []
        seeds_missing = []
        for seed in EXPECTED_SEEDS:
            seed_dir = run_dir / f"seed_{seed}"
            has_checkpoint = seed_dir.is_dir() and any(seed_dir.glob("checkpoint-*"))
            if has_checkpoint:
                seeds_found.append(seed)
            else:
                seeds_missing.append(seed)

        if seeds_missing:
            missing_rows.append((dataset, model, seeds_missing, seeds_found))
        else:
            ok_count += 1

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
total = sum(len(m) for m in RUN_PLAN.values())

print()
print(f"Seed coverage check  —  expected seeds: {EXPECTED_SEEDS}")
print(f"Source: {CHECKPOINTS}")
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