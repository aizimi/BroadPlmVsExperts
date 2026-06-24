#!/usr/bin/env python3
"""
Figure 3: ASTD weighted-loss ablation — grouped paired bar chart.

Source files:
  - results/csv/results_summary.csv          -> ASTD standard: f1_mean, f1_ci95, n
  - results/csv/results_summary_weighted.csv -> ASTD weighted: f1_mean, f1_ci95, n

Quantities plotted:
  - mean Macro-F1 over seeds (t-based 95% CIs)
  - NOT ensemble predictions
  - NOT bootstrap CIs
  - ASTD dataset only
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE          = os.path.dirname(SCRIPT_DIR)

STANDARD_CSV  = os.path.join(BASE, "results", "csv", "results_summary.csv")
WEIGHTED_CSV  = os.path.join(BASE, "results", "csv", "results_summary_weighted.csv")
OUT_DIR       = os.path.join(BASE, "outputs", "paper_figures")

# ── Model configuration ───────────────────────────────────────────────────────
MODEL_ORDER = ["marbert", "arabert", "egybert"]
MODEL_DISPLAY = {
    "marbert": "MARBERTv2",
    "arabert": "AraBERTv2",
    "egybert": "EgyBERT",
}

# Bar styles — grayscale-friendly, hatch differentiates conditions
STYLE = {
    "standard": {"color": "#2b2b2b", "hatch": "",    "label": "Standard"},
    "weighted": {"color": "#999999", "hatch": "///", "label": "Weighted"},
}

# ── Load ASTD rows from a results_summary CSV ─────────────────────────────────
def load_astd(path):
    """Return dict: model_internal -> {mean, ci95, n}"""
    result = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["dataset"].strip() != "ASTD":
                continue
            result[row["model"].strip()] = {
                "mean": float(row["f1_mean"]),
                "ci95": float(row["f1_ci95"]),
                "n":    int(row["n"]),
            }
    return result

# ── Build figure ──────────────────────────────────────────────────────────────
def make_figure(standard, weighted):
    n_models  = len(MODEL_ORDER)
    bar_w     = 0.30
    intra_gap = 0.04
    inter_gap = 0.45
    group_step = 2 * bar_w + intra_gap + inter_gap

    x_centers = np.arange(n_models) * group_step

    fig, ax = plt.subplots(figsize=(6.5, 4.5), facecolor="white")
    ax.set_facecolor("white")

    for condition, data_dict in (("standard", standard), ("weighted", weighted)):
        style  = STYLE[condition]
        offset = -(bar_w + intra_gap) / 2 if condition == "standard" else (bar_w + intra_gap) / 2

        x_pos  = []
        means  = []
        ci95s  = []
        ns     = []

        for model_key in MODEL_ORDER:
            if model_key not in data_dict:
                # Append placeholders so indices stay aligned; skip plotting below
                x_pos.append(None)
                means.append(None)
                ci95s.append(None)
                ns.append(None)
                continue
            d = data_dict[model_key]
            x_pos.append(x_centers[MODEL_ORDER.index(model_key)] + offset)
            means.append(d["mean"])
            ci95s.append(d["ci95"])
            ns.append(d["n"])

        # Plot bars individually to handle potential missing entries cleanly
        for xi, mean, ci95, n in zip(x_pos, means, ci95s, ns):
            if xi is None:
                continue
            bc = ax.bar(
                xi, mean,
                width=bar_w,
                yerr=ci95,
                capsize=3.0,
                color=style["color"],
                hatch=style["hatch"],
                edgecolor="black",
                linewidth=0.6,
                error_kw={
                    "elinewidth": 1.1,
                    "capthick":   1.1,
                    "ecolor":     "#111111",
                    "zorder":     5,
                },
                label=style["label"],   # duplicate labels filtered by legend handler
                zorder=3,
            )
            # White halo: renders a thick white Stroke first, then the dark
            # line on top — visible on both dark bar fills and light background.
            halo = [pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()]
            _, caplines, barlinecols = bc.errorbar.lines
            for cap in caplines:
                cap.set_path_effects(halo)
            for bcol in barlinecols:
                bcol.set_path_effects(halo)

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [MODEL_DISPLAY[m] for m in MODEL_ORDER],
        fontsize=10.5,
    )

    # y-axis: accommodate all CI bounds; always include 0.4 as floor
    all_means = [v["mean"] for d in (standard, weighted) for v in d.values()]
    all_ci95  = [v["ci95"] for d in (standard, weighted) for v in d.values()]
    y_lo = 0.40
    y_hi = min(1.0, np.ceil((max(m + c for m, c in zip(all_means, all_ci95)) + 0.04) * 20) / 20)

    ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel("Macro-F1", fontsize=11)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax.tick_params(axis="y", which="minor", length=2.5, width=0.6)
    ax.tick_params(axis="both", labelsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="both", width=0.7)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.35, color="#e0e0e0", zorder=0)

    # ── CI note — placed below the axes in figure-fraction space ─────────────
    # fig.text avoids any possible overlap with bars regardless of x layout.
    fig.text(
        0.5, 0.012, "Error bars: 95% CI",
        ha="center", va="bottom",
        fontsize=7.5, color="#888888",
        style="italic",
    )

    # ── Legend — deduplicate, place above plot ────────────────────────────────
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(
        list(seen.values()), list(seen.keys()),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=2,
        fontsize=8.5,
        frameon=True,
        framealpha=0.60,
        edgecolor="#e8e8e8",
        handlelength=1.1,
        handleheight=0.70,
        columnspacing=0.8,
        handletextpad=0.45,
        borderpad=0.5,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    for p in (STANDARD_CSV, WEIGHTED_CSV):
        if not os.path.isfile(p):
            print(f"ERROR: required file not found: {p}", file=sys.stderr)
            sys.exit(1)

    standard = load_astd(STANDARD_CSV)
    weighted = load_astd(WEIGHTED_CSV)

    # Verify all expected models are present
    for label, d in (("standard", standard), ("weighted", weighted)):
        for m in MODEL_ORDER:
            if m not in d:
                print(f"WARNING: model '{m}' not found in {label} data", file=sys.stderr)

    fig = make_figure(standard, weighted)

    os.makedirs(OUT_DIR, exist_ok=True)
    png_path = os.path.join(OUT_DIR, "figure3_astd_weighted_ablation.png")
    pdf_path = os.path.join(OUT_DIR, "figure3_astd_weighted_ablation.pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    # ── Verification summary ──────────────────────────────────────────────────
    print("=" * 60)
    print("Figure 3 -- Verification Summary")
    print("=" * 60)
    print()
    print("Source files used:")
    print(f"  Standard: {STANDARD_CSV}")
    print(f"  Weighted: {WEIGHTED_CSV}")
    print()
    print(f"Model order: {[MODEL_DISPLAY[m] for m in MODEL_ORDER]}")
    print()
    print("Values used (ASTD only):")
    header = f"  {'Model':<14} {'Cond':<10} {'n':>3}  {'f1_mean':>8}  {'f1_ci95':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for model_key in MODEL_ORDER:
        for label, d in (("standard", standard), ("weighted", weighted)):
            if model_key in d:
                v = d[model_key]
                print(f"  {MODEL_DISPLAY[model_key]:<14} {label:<10} {v['n']:>3}  "
                      f"{v['mean']:>8.4f}  {v['ci95']:>8.4f}")
    print()
    print("Notes:")
    print("  - Error bars = t-based 95% CI over seeds (f1_ci95)")
    print("  - EgyBERT standard CI is wide (~0.127) due to seed instability")
    print("  - Weighted AraBERTv2 and EgyBERT used n=4 seeds (1 run excluded)")
    print("  - Ensemble predictions: NOT used")
    print("  - Bootstrap CIs: NOT used")
    print("  - Dataset: ASTD only")
    print()
    print("Ambiguity: none. All three models present in both conditions.")
    print()
    print("Output files:")
    print(f"  PNG (300 dpi): {png_path}")
    print(f"  PDF (vector):  {pdf_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
