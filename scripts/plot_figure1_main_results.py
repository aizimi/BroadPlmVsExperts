#!/usr/bin/env python3
"""
Figure 1: Main results — grouped bar chart of Macro-F1 per model per dataset.

Source files:
  - results/csv/results_summary.csv  → f1_mean and f1_ci95 (t-based 95% CI over 5 seeds)
  - results/csv/pivot_table.csv      → dataset order (Table 2 row order)

Mirrors Table 2: mean Macro-F1 with t-based 95% CIs. Not ensemble, not bootstrap.
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(SCRIPT_DIR)

SUMMARY_CSV = os.path.join(BASE, "results", "csv", "results_summary.csv")
PIVOT_CSV   = os.path.join(BASE, "results", "csv", "pivot_table.csv")
OUT_DIR     = os.path.join(BASE, "outputs", "paper_figures")

# ── Model configuration ───────────────────────────────────────────────────────
# Internal names as they appear in results_summary.csv → paper display names
MODEL_MAP = {
    "marbert":    "MARBERTv2",
    "arabert":    "AraBERTv2",
    "egybert":    "EgyBERT",
    "darijabert": "DarijaBERT",
    "dziribert":  "DziriBERT",
}
# Fixed model order for the legend and bar grouping
MODEL_ORDER = ["marbert", "arabert", "egybert", "darijabert", "dziribert"]

# Visual style per model: color + hatch for grayscale compatibility
MODEL_STYLE = {
    "marbert":    {"color": "#2b2b2b", "hatch": ""},
    "arabert":    {"color": "#666666", "hatch": "//"},
    "egybert":    {"color": "#999999", "hatch": "\\\\"},
    "darijabert": {"color": "#bbbbbb", "hatch": "xx"},
    "dziribert":  {"color": "#dedede", "hatch": ".."},
}

# ── Load dataset order from pivot_table.csv ──────────────────────────────────
def load_dataset_order(path):
    order = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row["Dataset"].strip()
            if ds:
                order.append(ds)
    return order

# ── Load results from results_summary.csv ────────────────────────────────────
def load_results(path):
    """Return dict: (dataset, model_internal) -> {"mean": float, "ci95": float}"""
    results = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["dataset"].strip(), row["model"].strip())
            results[key] = {
                "mean": float(row["f1_mean"]),
                "ci95": float(row["f1_ci95"]),
            }
    return results

# ── Dataset display labels ────────────────────────────────────────────────────
# Use underscores replaced with hyphens for cleaner axis tick appearance
def display_label(ds):
    return ds.replace("_", "-")

# ── Build figure ──────────────────────────────────────────────────────────────
def make_figure(dataset_order, results):
    n_datasets = len(dataset_order)
    n_models   = len(MODEL_ORDER)

    bar_w      = 0.14          # width of each individual bar
    intra_gap  = 0.01          # gap between bars within a group
    inter_gap  = 0.30          # gap between groups

    # Space groups based on the largest actual group, not the global model count.
    # This removes the empty reserved slots for absent models.
    max_present = max(
        sum(1 for m in MODEL_ORDER if (ds, m) in results)
        for ds in dataset_order
    )
    group_span = max_present * bar_w + (max_present - 1) * intra_gap
    group_step = group_span + inter_gap

    # Centre x-position of each dataset group
    x_centers = np.arange(n_datasets) * group_step

    fig, ax = plt.subplots(figsize=(13, 5.2), facecolor="white")
    ax.set_facecolor("white")

    # Pass 1 — compute bar positions dataset-by-dataset so bars are packed
    # with no empty slots, then collect per model for legend-aware bar() calls.
    model_bars = {m: {"x": [], "means": [], "ci95s": []} for m in MODEL_ORDER}

    for di, dataset in enumerate(dataset_order):
        present = [m for m in MODEL_ORDER if (dataset, m) in results]
        k = len(present)
        if k == 0:
            continue
        actual_span = k * bar_w + (k - 1) * intra_gap
        left_centre = x_centers[di] - actual_span / 2 + bar_w / 2
        for bi, model_key in enumerate(present):
            x = left_centre + bi * (bar_w + intra_gap)
            model_bars[model_key]["x"].append(x)
            model_bars[model_key]["means"].append(results[(dataset, model_key)]["mean"])
            model_bars[model_key]["ci95s"].append(results[(dataset, model_key)]["ci95"])

    # Pass 2 — one bar() call per model so each gets exactly one legend entry
    for model_key in MODEL_ORDER:
        bd = model_bars[model_key]
        if not bd["x"]:
            continue
        style = MODEL_STYLE[model_key]
        ax.bar(
            bd["x"], bd["means"],
            width=bar_w,
            yerr=bd["ci95s"],
            capsize=2.5,
            color=style["color"],
            hatch=style["hatch"],
            edgecolor="black",
            linewidth=0.55,
            error_kw={
                "elinewidth": 0.75,
                "capthick":   0.75,
                "ecolor":     "#333333",
                "zorder":     4,
            },
            label=MODEL_MAP[model_key],
            zorder=3,
        )

    # ── Axes styling ──────────────────────────────────────────────────────────
    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [display_label(ds) for ds in dataset_order],
        fontsize=10,
    )

    # y-axis: fixed lower bound of 0.4 (lowest CI lower bound is ~0.39, so 0.4 captures
    # all bars without compressing the scale; 0.5 would hide the AraBERTv2 AfriSenti-ARY bar)
    all_means = [v["mean"] for v in results.values()]
    all_ci95  = [v["ci95"] for v in results.values()]
    y_max_data = max(m + c for m, c in zip(all_means, all_ci95))
    y_lo = 0.4
    y_hi = min(1.0, np.ceil((y_max_data + 0.04) * 20) / 20)

    ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel("Macro-F1", fontsize=11)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.tick_params(axis="y", which="minor", length=2.5, width=0.6)
    ax.tick_params(axis="both", labelsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="both", width=0.7)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.45, color="#cccccc", zorder=0)

    # ── Legend — single row above the plot, visually receded ─────────────────
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=n_models,
        fontsize=8.5,
        frameon=True,
        framealpha=0.7,
        edgecolor="#dddddd",
        handlelength=1.2,
        handleheight=0.8,
        columnspacing=0.9,
        handletextpad=0.5,
    )

    # Tight layout with room for the legend above
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Verify input files exist
    for p in (SUMMARY_CSV, PIVOT_CSV):
        if not os.path.isfile(p):
            print(f"ERROR: required file not found: {p}", file=sys.stderr)
            sys.exit(1)

    dataset_order = load_dataset_order(PIVOT_CSV)
    results       = load_results(SUMMARY_CSV)

    # Confirm every dataset in order has at least one result
    covered = {ds for (ds, _) in results}
    for ds in dataset_order:
        if ds not in covered:
            print(f"WARNING: dataset '{ds}' in pivot_table.csv has no rows in results_summary.csv",
                  file=sys.stderr)

    fig = make_figure(dataset_order, results)

    os.makedirs(OUT_DIR, exist_ok=True)
    png_path = os.path.join(OUT_DIR, "figure1_main_results.png")
    pdf_path = os.path.join(OUT_DIR, "figure1_main_results.pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    # ── Verification summary ──────────────────────────────────────────────────
    print("=" * 60)
    print("Figure 1 — Verification Summary")
    print("=" * 60)
    print()
    print("Source files used:")
    print(f"  Numerical values (f1_mean, f1_ci95): {SUMMARY_CSV}")
    print(f"  Dataset order (Table 2 rows):         {PIVOT_CSV}")
    print()
    print(f"Dataset order ({len(dataset_order)} datasets):")
    for i, ds in enumerate(dataset_order, 1):
        print(f"  {i}. {ds}")
    print()
    print(f"Model order ({len(MODEL_ORDER)} models):")
    for i, m in enumerate(MODEL_ORDER, 1):
        print(f"  {i}. {MODEL_MAP[m]}  (internal key: {m!r})")
    print()
    print("Table 2 alignment:")
    print("  Scores   = mean Macro-F1 over 5 seeds (f1_mean)")
    print("  Error bars = +/-t-based 95% CI (f1_ci95), same quantity as Table 2")
    print("  Missing model-dataset pairs: bar omitted (no fabrication)")
    print("  Ensemble predictions: NOT included")
    print("  Bootstrap CIs:        NOT used")
    print()
    print("Model–dataset coverage:")
    for ds in dataset_order:
        models_present = [MODEL_MAP[m] for m in MODEL_ORDER if (ds, m) in results]
        print(f"  {ds:<20}: {', '.join(models_present)}")
    print()
    print("Output files:")
    print(f"  PNG (300 dpi): {png_path}")
    print(f"  PDF (vector):  {pdf_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
