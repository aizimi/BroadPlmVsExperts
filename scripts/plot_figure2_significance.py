#!/usr/bin/env python3
"""
Figure 2: Statistical significance — forest plot of pairwise bootstrap comparisons vs MARBERTv2.

Source: results/csv/significance_summary_table.csv
Mirrors Table 3:
  - delta_f1 = comparator Macro-F1 minus MARBERTv2 Macro-F1 (on ensemble predictions)
  - CI = bootstrap 95% confidence interval
  - significance = Holm-corrected bootstrap p-value < 0.05
Do NOT use results_summary.csv or t-based CIs for this figure.
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE       = os.path.dirname(SCRIPT_DIR)

SIG_CSV  = os.path.join(BASE, "results", "csv", "significance_summary_table.csv")
OUT_DIR  = os.path.join(BASE, "outputs", "paper_figures")

# ── Dataset order (same as Table 2 / Table 3) ─────────────────────────────────
DATASET_ORDER = [
    "ASTD",
    "ArSAS",
    "AfriSenti_ARQ",
    "MACcorpus",
    "AfriSenti_ARY",
    "LABR",
    "HARD",
]

# Within each dataset AraBERTv2 appears first (as stored in the source file);
# any dialect-specialist comparator follows.  We preserve that order.
COMPARATOR_PRIORITY = ["AraBERTv2", "EgyBERT", "DziriBERT", "DarijaBERT"]

def comparator_rank(name):
    try:
        return COMPARATOR_PRIORITY.index(name)
    except ValueError:
        return len(COMPARATOR_PRIORITY)

def display_dataset(ds):
    return ds.replace("_", "-")

# ── Load data ─────────────────────────────────────────────────────────────────
def load_sig(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "dataset":   row["Dataset"].strip(),
                "model":     row["Model"].strip(),
                "delta":     float(row["delta_f1"]),
                "ci_low":    float(row["ci_low"]),
                "ci_high":   float(row["ci_high"]),
                "p_holm":    float(row["p_holm"]),
                "sig":       row["sig"].strip() == "✓",  # ✓ character
            })
    return rows

# ── Order rows for the plot ───────────────────────────────────────────────────
def order_rows(rows):
    # Build a lookup: (dataset, model) -> row
    lookup = {(r["dataset"], r["model"]): r for r in rows}
    ordered = []
    for ds in DATASET_ORDER:
        # Collect all comparators for this dataset in COMPARATOR_PRIORITY order
        ds_rows = [r for r in rows if r["dataset"] == ds]
        ds_rows.sort(key=lambda r: comparator_rank(r["model"]))
        ordered.extend(ds_rows)
    return ordered

# ── Build figure ──────────────────────────────────────────────────────────────
def make_figure(rows):
    # Assign y positions (0 = top after axis inversion).
    # Add 0.5 extra gap between dataset groups for visual separation.
    y_vals   = []
    y_labels = []
    y_group_boundaries = []   # y midpoints between groups for separator lines

    y = 0.0
    prev_ds = None
    for i, row in enumerate(rows):
        if prev_ds is not None and row["dataset"] != prev_ds:
            y_group_boundaries.append(y - 0.25)   # midpoint of the gap
            y += 0.5
        y_vals.append(y)
        y_labels.append(f"{display_dataset(row['dataset'])} — {row['model']}")
        y += 1.0
        prev_ds = row["dataset"]

    n_rows = len(rows)
    y_max  = max(y_vals)

    # Determine x limits dynamically with some padding
    all_x = [r["delta"] for r in rows] + [r["ci_low"] for r in rows] + [r["ci_high"] for r in rows]
    x_pad = 0.01
    x_lo  = min(all_x) - x_pad
    x_hi  = max(all_x) + x_pad
    # Always include zero with a small visible margin
    x_lo  = min(x_lo, -0.005)
    x_hi  = max(x_hi,  0.005)

    fig, ax = plt.subplots(figsize=(9, 8.2), facecolor="white")
    ax.set_facecolor("white")

    # ── Draw rows ─────────────────────────────────────────────────────────────
    for row, y in zip(rows, y_vals):
        delta   = row["delta"]
        ci_low  = row["ci_low"]
        ci_high = row["ci_high"]
        sig     = row["sig"]

        # Horizontal CI line
        ax.plot(
            [ci_low, ci_high], [y, y],
            color="black", linewidth=1.0, solid_capstyle="round", zorder=2,
        )
        # CI cap ticks
        cap_h = 0.12
        for cx in (ci_low, ci_high):
            ax.plot([cx, cx], [y - cap_h, y + cap_h],
                    color="black", linewidth=0.8, zorder=2)

        # Point estimate marker
        if sig:
            # Significant: filled circle
            ax.plot(delta, y, marker="o", markersize=7,
                    color="black", markerfacecolor="black",
                    markeredgecolor="black", markeredgewidth=0.8,
                    zorder=3, linestyle="none")
        else:
            # Non-significant: open circle
            ax.plot(delta, y, marker="o", markersize=7,
                    color="black", markerfacecolor="white",
                    markeredgecolor="black", markeredgewidth=0.9,
                    zorder=3, linestyle="none")

    # ── Zero reference line ───────────────────────────────────────────────────
    ax.axvline(x=0, color="#555555", linewidth=0.9,
               linestyle="--", zorder=1, dashes=(4, 3))

    # ── Dataset group separator lines ─────────────────────────────────────────
    for yb in y_group_boundaries:
        ax.axhline(y=yb, color="#ebebeb", linewidth=0.5,
                   linestyle="-", zorder=0)

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_yticks(y_vals)
    ax.set_yticklabels(y_labels, fontsize=9.5)
    ax.set_ylim(-0.65, y_max + 0.65)
    ax.invert_yaxis()   # first row at top

    ax.set_xlabel(
        "Delta Macro-F1 (Comparator − MARBERTv2)",
        fontsize=10.5,
    )
    ax.set_xlim(x_lo, x_hi)
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.01))
    ax.tick_params(axis="x", which="minor", length=2.5, width=0.6)
    ax.tick_params(axis="both", labelsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="y", which="both", length=0)   # hide y tick marks

    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)

    # ── Legend ────────────────────────────────────────────────────────────────
    sig_marker   = mlines.Line2D([], [], marker="o", color="black",
                                 markerfacecolor="black", markersize=6,
                                 linestyle="none",
                                 label="Holm-significant (p < 0.05)")
    nosig_marker = mlines.Line2D([], [], marker="o", color="black",
                                 markerfacecolor="white", markersize=6,
                                 linestyle="none",
                                 label="Not significant")
    ax.legend(
        handles=[sig_marker, nosig_marker],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        fontsize=8.5,
        frameon=True,
        framealpha=0.85,
        edgecolor="#cccccc",
        handletextpad=0.5,
        borderpad=0.6,
        columnspacing=1.2,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.isfile(SIG_CSV):
        print(f"ERROR: required file not found: {SIG_CSV}", file=sys.stderr)
        sys.exit(1)

    rows     = load_sig(SIG_CSV)
    ordered  = order_rows(rows)

    fig = make_figure(ordered)

    os.makedirs(OUT_DIR, exist_ok=True)
    png_path = os.path.join(OUT_DIR, "figure2_significance.png")
    pdf_path = os.path.join(OUT_DIR, "figure2_significance.pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    # ── Verification summary ──────────────────────────────────────────────────
    sig_count   = sum(1 for r in ordered if r["sig"])
    nosig_count = sum(1 for r in ordered if not r["sig"])

    print("=" * 60)
    print("Figure 2 -- Verification Summary")
    print("=" * 60)
    print()
    print("Source file used:")
    print(f"  {SIG_CSV}")
    print()
    print(f"Dataset order ({len(DATASET_ORDER)} datasets):")
    for i, ds in enumerate(DATASET_ORDER, 1):
        print(f"  {i}. {ds}")
    print()
    print(f"Comparison rows ({len(ordered)} total):")
    for r in ordered:
        flag = "[sig]" if r["sig"] else "[   ]"
        print(f"  {flag}  {r['dataset']:<20} {r['model']:<14}  "
              f"delta={r['delta']:+.3f}  "
              f"CI [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  "
              f"p_holm={r['p_holm']:.4f}")
    print()
    print("Significance encoding:")
    print("  Filled circle  = significant (Holm-corrected p < 0.05)")
    print("  Open circle    = not significant")
    print(f"  {sig_count} significant, {nosig_count} not significant")
    print()
    print("Table 3 alignment:")
    print("  delta_f1  = bootstrap Macro-F1 comparator minus MARBERTv2")
    print("  CI        = bootstrap 95% confidence interval")
    print("  p_holm    = Holm-corrected bootstrap p-value")
    print("  Ensemble predictions used (NOT mean-over-seeds)")
    print("  t-based CIs: NOT used")
    print()
    print("Output files:")
    print(f"  PNG (300 dpi): {png_path}")
    print(f"  PDF (vector):  {pdf_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
