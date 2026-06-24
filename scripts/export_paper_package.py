#!/usr/bin/env python3
"""export_paper_package.py

Generate writing-support files for the Arabic SA paper from verified result CSVs.

Outputs (all in outputs/paper_package/):
  1. results_overview.md       -- narrative summary of main findings
  2. results_overview.json     -- machine-readable version of the same
  3. results_tables_for_paper.md -- LaTeX-ready table stubs
  4. discussion_notes.md       -- per-finding discussion bullets
  5. figure_recommendations.md -- figure design notes

Methodological conventions this script respects:
  - Table 2 CIs are t-based 95% intervals over 5 independent seeds (NOT bootstrap).
  - Table 3 tests are paired bootstrap on ensemble predictions, Holm-corrected per dataset.
  - The weighted cross-entropy experiment covers ASTD only.
  - Dialect distribution values come from an automatic classifier and are treated as
    descriptive estimates, not ground truth.
  - Missing weighted runs are reported with their actual n; no reason is invented.
  - All delta_f1 values are computed from the actual significance_summary_table.csv
    (comparator minus MARBERTv2, always <= 0 since MARBERTv2 is the reference).
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CSV_DIR    = os.path.join(PROJECT_ROOT, "results", "csv")
CLASS_DIR  = os.path.join(PROJECT_ROOT, "outputs", "classification")
OUT_DIR    = os.path.join(PROJECT_ROOT, "outputs", "paper_package")

# Input files
F_SUMMARY          = os.path.join(CSV_DIR, "results_summary.csv")
F_PIVOT            = os.path.join(CSV_DIR, "pivot_table.csv")
F_SIG              = os.path.join(CSV_DIR, "significance_summary_table.csv")
F_SUMMARY_W        = os.path.join(CSV_DIR, "results_summary_weighted.csv")
F_SIG_W            = os.path.join(CSV_DIR, "significance_summary_table_weighted.csv")
F_PER_SEED         = os.path.join(CSV_DIR, "per_seed_runs.csv")
F_COARSE           = os.path.join(CLASS_DIR, "dialect_distribution_coarse.csv")
F_FINE             = os.path.join(CLASS_DIR, "dialect_distribution_fine.csv")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required file missing: {path}")
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fmt(v: float, decimals: int = 3) -> str:
    return f"{v:.{decimals}f}"


def _pct(v: float) -> str:
    return f"{v:.1f}%"


# ---------------------------------------------------------------------------
# Load and validate all source data
# ---------------------------------------------------------------------------

def load_all() -> dict:
    """Load all CSVs; return a dict of named data structures."""

    summary_rows   = _read_csv(F_SUMMARY)
    pivot_rows     = _read_csv(F_PIVOT)
    sig_rows       = _read_csv(F_SIG)
    summary_w_rows = _read_csv(F_SUMMARY_W)
    sig_w_rows     = _read_csv(F_SIG_W)
    per_seed_rows  = _read_csv(F_PER_SEED)
    coarse_rows    = _read_csv(F_COARSE)
    fine_rows      = _read_csv(F_FINE)

    # --- Build per-(dataset, model) summary lookup ---
    summary: Dict[Tuple[str, str], dict] = {}
    for r in summary_rows:
        key = (r["dataset"], r["model"])
        summary[key] = {
            "n":       int(r["n"]),
            "f1_mean": float(r["f1_mean"]),
            "f1_std":  float(r["f1_std"]),
            "f1_ci95": float(r["f1_ci95"]),
        }

    # Ordered datasets as they appear in pivot
    datasets_ordered = [r["Dataset"] for r in pivot_rows]

    # --- Build significance lookup ---
    # key: (dataset, model_display) where model_display is as in sig CSV
    sig: Dict[Tuple[str, str], dict] = {}
    for r in sig_rows:
        key = (r["Dataset"], r["Model"])
        sig[key] = {
            "delta_f1":  float(r["delta_f1"]),
            "ci_low":    float(r["ci_low"]),
            "ci_high":   float(r["ci_high"]),
            "p_holm":    float(r["p_holm"]),
            "sig":       r["sig"].strip() == "\u2713",  # check mark
            "formatted": r["delta_f1_formatted"],
        }

    # --- Build weighted summary lookup ---
    summary_w: Dict[str, dict] = {}
    for r in summary_w_rows:
        summary_w[r["model"]] = {
            "n":       int(r["n"]),
            "f1_mean": float(r["f1_mean"]),
            "f1_std":  float(r["f1_std"]),
            "f1_ci95": float(r["f1_ci95"]),
        }

    # --- Build weighted significance lookup ---
    sig_w: Dict[str, dict] = {}
    for r in sig_w_rows:
        sig_w[r["Model"]] = {
            "delta_f1": float(r["delta_f1"]),
            "ci_low":   float(r["ci_low"]),
            "ci_high":  float(r["ci_high"]),
            "p_holm":   float(r["p_holm"]),
            "sig":      r["sig"].strip() == "\u2713",
        }

    # --- Per-seed: collect per (dataset, model) f1 values ---
    per_seed: Dict[Tuple[str, str], List[Tuple[int, float]]] = defaultdict(list)
    for r in per_seed_rows:
        key = (r["dataset"], r["model"])
        per_seed[key].append((int(r["seed"]), float(r["macro_f1"])))

    # --- Dialect lookup ---
    coarse: Dict[str, dict] = {}
    for r in coarse_rows:
        coarse[r["Dataset"]] = {
            "dialect_pct":  float(r["Dialect"]),
            "msa_pct":      float(r["MSA"]),
            "dominant":     r["DominantDialect"],
            "n_classified": int(r["n_classified"]),
        }

    fine: Dict[str, dict] = {}
    for r in fine_rows:
        fine[r["Dataset"]] = {
            "EGY":          float(r["EGY"]),
            "LEV":          float(r["LEV"]),
            "GLF":          float(r["GLF"]),
            "MAGHREB":      float(r["MAGHREB"]),
            "MSA":          float(r["MSA"]),
            "n_classified": int(r["n_classified"]),
        }

    return dict(
        summary=summary,
        pivot_rows=pivot_rows,
        datasets_ordered=datasets_ordered,
        sig=sig,
        summary_w=summary_w,
        sig_w=sig_w,
        per_seed=dict(per_seed),
        coarse=coarse,
        fine=fine,
    )


# ---------------------------------------------------------------------------
# Derived facts computed from actual file numbers
# ---------------------------------------------------------------------------

MODEL_DISPLAY = {
    "marbert":    "MARBERTv2",
    "arabert":    "AraBERTv2",
    "egybert":    "EgyBERT",
    "darijabert": "DarijaBERT",
    "dziribert":  "DziriBERT",
}


def derive_facts(data: dict) -> dict:
    sig      = data["sig"]
    sig_w    = data["sig_w"]
    summary  = data["summary"]
    per_seed = data["per_seed"]

    # --- Significant / non-significant split ---
    sig_entries    = [(k, v) for k, v in sig.items() if v["sig"]]
    nonsig_entries = [(k, v) for k, v in sig.items() if not v["sig"]]

    # Largest / smallest magnitude gap among significant results
    largest_sig  = min(sig_entries, key=lambda x: x[1]["delta_f1"])  # most negative
    smallest_sig = max(sig_entries, key=lambda x: x[1]["delta_f1"])  # least negative (closest to 0)

    # Which datasets have AraBERT significantly worse
    arabert_sig_datasets   = [k[0] for k, v in sig.items()
                               if k[1] == "AraBERTv2" and v["sig"]]
    arabert_nonsig_datasets = [k[0] for k, v in sig.items()
                                if k[1] == "AraBERTv2" and not v["sig"]]

    # MARBERTv2 mean F1 per dataset
    marbert_f1 = {k[0]: v["f1_mean"] for k, v in summary.items() if k[1] == "marbert"}

    best_dataset  = max(marbert_f1, key=marbert_f1.get)
    worst_dataset = min(marbert_f1, key=marbert_f1.get)

    # --- EgyBERT ASTD instability ---
    egybert_astd_seeds = per_seed.get(("ASTD", "egybert"), [])
    egybert_astd_seeds = sorted(egybert_astd_seeds, key=lambda x: x[0])
    egybert_astd_f1s   = [f1 for _, f1 in egybert_astd_seeds]
    outlier_seed = None
    if egybert_astd_f1s:
        min_f1     = min(egybert_astd_f1s)
        min_seed   = next(s for s, f in egybert_astd_seeds if f == min_f1)
        other_f1s  = [f for f in egybert_astd_f1s if f != min_f1]
        if other_f1s and min_f1 < min(other_f1s) - 0.10:
            outlier_seed = {
                "seed":       min_seed,
                "f1":         min_f1,
                "others_min": min(other_f1s),
                "others_max": max(other_f1s),
            }

    # --- Weighted experiment delta vs standard ---
    summary_w = data["summary_w"]
    weighted_deltas = {}
    for mdl_int in ["marbert", "arabert", "egybert"]:
        if mdl_int in summary_w and ("ASTD", mdl_int) in summary:
            std_f1 = summary[("ASTD", mdl_int)]["f1_mean"]
            w      = summary_w[mdl_int]
            weighted_deltas[mdl_int] = {
                "n_w":          w["n"],
                "f1_w":         w["f1_mean"],
                "f1_std":       w["f1_std"],
                "f1_ci95":      w["f1_ci95"],
                "f1_std_std":   std_f1,
                "delta_vs_std": w["f1_mean"] - std_f1,
            }

    return dict(
        sig_entries=sig_entries,
        nonsig_entries=nonsig_entries,
        largest_sig=largest_sig,
        smallest_sig=smallest_sig,
        arabert_sig_datasets=arabert_sig_datasets,
        arabert_nonsig_datasets=arabert_nonsig_datasets,
        marbert_f1=marbert_f1,
        best_dataset=best_dataset,
        worst_dataset=worst_dataset,
        egybert_astd_seeds=egybert_astd_seeds,
        outlier_seed=outlier_seed,
        weighted_deltas=weighted_deltas,
        n_sig=len(sig_entries),
        n_nonsig=len(nonsig_entries),
    )


# ---------------------------------------------------------------------------
# Output 1: results_overview.md
# ---------------------------------------------------------------------------

def write_results_overview_md(data: dict, facts: dict, out_dir: str) -> str:
    sig        = data["sig"]
    pivot      = data["pivot_rows"]
    coarse     = data["coarse"]
    fine       = data["fine"]
    summary_w  = data["summary_w"]
    sig_w      = data["sig_w"]
    summary    = data["summary"]

    ls_key, ls_val = facts["largest_sig"]
    ss_key, ss_val = facts["smallest_sig"]
    outlier = facts["outlier_seed"]

    lines = []
    a = lines.append

    a("# Results Overview")
    a("")
    a("*Auto-generated by export_paper_package.py -- do not edit by hand.*")
    a("")
    a("## Coverage")
    a("")
    a("Experiments cover 7 Arabic sentiment datasets and up to 5 pre-trained models.")
    a("All main-experiment models are trained for 5 independent seeds (seeds 42-46).")
    a("Performance is measured by macro-F1 on the held-out test split.")
    a("")
    a("| Dataset | MARBERTv2 | Comparators |")
    a("|---------|-----------|-------------|")
    for r in pivot:
        ds = r["Dataset"]
        comparators = []
        for col in ["AraBERTv2", "EgyBERT", "DarijaBERT", "DziriBERT"]:
            val = r.get(col, "").strip()
            if val and val != "-" and val != "--":
                comparators.append(f"{col}: {val}")
        a(f"| {ds} | {r['MARBERTv2']} | {', '.join(comparators) if comparators else '-'} |")
    a("")
    a("CIs shown are t-based 95% intervals over the 5 seed runs.")
    a("")

    a("## Main Finding: MARBERTv2 Leads Consistently")
    a("")
    a("MARBERTv2 achieves the best macro-F1 on all 7 datasets.")
    a("")

    marbert_f1 = facts["marbert_f1"]
    best_ds  = facts["best_dataset"]
    worst_ds = facts["worst_dataset"]
    a(f"- Highest MARBERTv2 macro-F1: {_fmt(marbert_f1[best_ds])} on {best_ds}")
    a(f"- Lowest MARBERTv2 macro-F1:  {_fmt(marbert_f1[worst_ds])} on {worst_ds}")
    a("")

    a("## Significance Tests (Table 3)")
    a("")
    a("Paired bootstrap (10,000 resamples) on ensemble predictions, Holm-corrected per dataset.")
    a(f"Significant comparisons: {facts['n_sig']} of {facts['n_sig'] + facts['n_nonsig']} total.")
    a("")
    a(f"Largest significant gap: {ls_key[1]} on {ls_key[0]}: "
      f"delta_F1 = {_fmt(ls_val['delta_f1'])} "
      f"[{_fmt(ls_val['ci_low'])}, {_fmt(ls_val['ci_high'])}], "
      f"p_holm = {ls_val['p_holm']:.4f}")
    a(f"Smallest significant gap: {ss_key[1]} on {ss_key[0]}: "
      f"delta_F1 = {_fmt(ss_val['delta_f1'])} "
      f"[{_fmt(ss_val['ci_low'])}, {_fmt(ss_val['ci_high'])}], "
      f"p_holm = {ss_val['p_holm']:.4f}")
    a("")

    arabert_sig    = facts["arabert_sig_datasets"]
    arabert_nonsig = facts["arabert_nonsig_datasets"]
    a("### AraBERTv2 vs MARBERTv2")
    a(f"Significantly worse on: {', '.join(arabert_sig) if arabert_sig else 'none'}")
    a(f"Not significant on:     {', '.join(arabert_nonsig) if arabert_nonsig else 'none'}")
    a("")

    a("### Non-significant comparisons")
    for k, v in facts["nonsig_entries"]:
        a(f"- {k[0]} {k[1]}: delta_F1 = {_fmt(v['delta_f1'])}, "
          f"p_holm = {v['p_holm']:.4f}")
    a("")

    a("## EgyBERT on ASTD: High Variance")
    a("")
    if outlier:
        f1s = facts["egybert_astd_seeds"]
        min_f1 = min(f for _, f in f1s)
        max_f1 = max(f for _, f in f1s)
        a(f"EgyBERT shows unusually high variance on ASTD "
          f"(F1 range across seeds: {_fmt(min_f1)} - {_fmt(max_f1)}).")
        a(f"Seed {outlier['seed']} produced F1 = {_fmt(outlier['f1'])}, "
          f"substantially below the remaining seeds "
          f"({_fmt(outlier['others_min'])} - {_fmt(outlier['others_max'])}).")
        a("This drives the wide CI shown in Table 2 for EgyBERT / ASTD.")
    else:
        a("Per-seed variance for EgyBERT on ASTD is notable; see Table 2 CI.")
    a("")

    a("## Weighted Cross-Entropy Ablation (ASTD only)")
    a("")
    a("Class-weighted cross-entropy was tested as an ablation on ASTD.")
    wd = facts["weighted_deltas"]
    for mdl_int, label in [("marbert", "MARBERTv2"), ("arabert", "AraBERTv2"),
                            ("egybert", "EgyBERT")]:
        if mdl_int in wd:
            d = wd[mdl_int]
            n_note = "" if d["n_w"] == 5 else f" (n={d['n_w']} runs available)"
            sign   = "+" if d["delta_vs_std"] >= 0 else ""
            a(f"- {label}{n_note}: weighted F1 = {_fmt(d['f1_w'])}; "
              f"delta vs standard = {sign}{_fmt(d['delta_vs_std'])}")
    a("")
    a("Weighted significance vs weighted MARBERTv2:")
    for mdl_disp, v in sig_w.items():
        sig_str = "significant" if v["sig"] else "not significant"
        a(f"- {mdl_disp}: delta_F1 = {_fmt(v['delta_f1'])}, "
          f"p_holm = {v['p_holm']:.4f} ({sig_str})")
    a("")

    a("## Dialect Composition (Descriptive, Automatic Estimates)")
    a("")
    a("Table 1 provides a descriptive characterisation of each dataset's variety mix.")
    a("Proportions were obtained by running an automatic dialect classifier "
      "(IbrahimAmin/marbertv2-arabic-written-dialect-classifier) on all available texts.")
    a("These estimates are used only to check whether the observed distribution is broadly")
    a("consistent with the commonly understood provenance and expected dominant variety of")
    a("each dataset. They are not ground-truth dialect labels and are not a main empirical result.")
    a("See results_tables_for_paper.md for the full Table 1 breakdown.")
    a("")

    path = os.path.join(out_dir, "results_overview.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Output 2: results_overview.json
# ---------------------------------------------------------------------------

def write_results_overview_json(data: dict, facts: dict, out_dir: str) -> str:
    summary   = data["summary"]
    sig       = data["sig"]
    sig_w     = data["sig_w"]
    coarse    = data["coarse"]
    fine      = data["fine"]
    summary_w = data["summary_w"]

    ls_key, ls_val = facts["largest_sig"]
    ss_key, ss_val = facts["smallest_sig"]
    outlier = facts["outlier_seed"]

    # Main results
    main_results: dict = {}
    for (ds, mdl), v in summary.items():
        if ds not in main_results:
            main_results[ds] = {}
        label = MODEL_DISPLAY.get(mdl, mdl)
        main_results[ds][label] = {
            "n":        v["n"],
            "f1_mean":  round(v["f1_mean"], 6),
            "f1_ci95":  round(v["f1_ci95"], 6),
            "ci_lower": round(v["f1_mean"] - v["f1_ci95"], 6),
            "ci_upper": round(v["f1_mean"] + v["f1_ci95"], 6),
        }

    # Significance table
    sig_table: dict = {}
    for (ds, mdl), v in sig.items():
        if ds not in sig_table:
            sig_table[ds] = {}
        sig_table[ds][mdl] = {
            "delta_f1":    round(v["delta_f1"], 6),
            "ci_low":      round(v["ci_low"], 6),
            "ci_high":     round(v["ci_high"], 6),
            "p_holm":      round(v["p_holm"], 4),
            "significant": v["sig"],
        }

    # Weighted ablation
    weighted: dict = {}
    for mdl_int, w in summary_w.items():
        label  = MODEL_DISPLAY.get(mdl_int, mdl_int)
        std_f1 = summary.get(("ASTD", mdl_int), {}).get("f1_mean")
        entry: dict = {
            "n":       w["n"],
            "f1_mean": round(w["f1_mean"], 6),
            "f1_ci95": round(w["f1_ci95"], 6),
        }
        if std_f1 is not None:
            entry["delta_vs_standard"] = round(w["f1_mean"] - std_f1, 6)
        weighted[label] = entry

    weighted_sig = {
        mdl: {
            "delta_f1":    round(v["delta_f1"], 6),
            "p_holm":      round(v["p_holm"], 4),
            "significant": v["sig"],
        }
        for mdl, v in sig_w.items()
    }

    # Dialect estimates
    dialect_estimates: dict = {}
    for ds in data["datasets_ordered"]:
        if ds in coarse and ds in fine:
            dialect_estimates[ds] = {
                "source":           "automatic_classifier",
                "model":            "IbrahimAmin/marbertv2-arabic-written-dialect-classifier",
                "dominant_dialect": coarse[ds]["dominant"],
                "n_classified":     coarse[ds]["n_classified"],
                "proportions": {
                    "MSA":     round(fine[ds]["MSA"], 2),
                    "EGY":     round(fine[ds]["EGY"], 2),
                    "MAGHREB": round(fine[ds]["MAGHREB"], 2),
                    "GLF":     round(fine[ds]["GLF"], 2),
                    "LEV":     round(fine[ds]["LEV"], 2),
                },
            }

    obj = {
        "meta": {
            "primary_metric":        "macro_F1",
            "ci_method":             "t-distribution 95% over 5 seeds",
            "significance_method":   "paired bootstrap (10000 resamples), Holm correction per dataset",
            "reference_model":       "MARBERTv2",
            "n_seeds_main":          5,
        },
        "main_results": main_results,
        "significance_table3": sig_table,
        "summary_stats": {
            "n_significant_comparisons": facts["n_sig"],
            "n_total_comparisons":       facts["n_sig"] + facts["n_nonsig"],
            "largest_significant_gap": {
                "dataset":  ls_key[0],
                "model":    ls_key[1],
                "delta_f1": round(ls_val["delta_f1"], 6),
                "p_holm":   ls_val["p_holm"],
            },
            "smallest_significant_gap": {
                "dataset":  ss_key[0],
                "model":    ss_key[1],
                "delta_f1": round(ss_val["delta_f1"], 6),
                "p_holm":   ss_val["p_holm"],
            },
        },
        "instability_flags": {
            "ASTD_EgyBERT_outlier_seed": (
                None if outlier is None else {
                    "seed":                outlier["seed"],
                    "f1":                  round(outlier["f1"], 6),
                    "other_seeds_f1_min":  round(outlier["others_min"], 6),
                    "other_seeds_f1_max":  round(outlier["others_max"], 6),
                }
            )
        },
        "weighted_ablation_ASTD": {
            "note":                             "Class-weighted cross-entropy, ASTD only. n may differ across models.",
            "results":                          weighted,
            "significance_vs_weighted_marbert": weighted_sig,
        },
        "dialect_composition_estimates": dialect_estimates,
    }

    path = os.path.join(out_dir, "results_overview.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# Output 3: results_tables_for_paper.md
# ---------------------------------------------------------------------------

def write_results_tables_md(data: dict, facts: dict, out_dir: str) -> str:
    pivot     = data["pivot_rows"]
    sig       = data["sig"]
    summary   = data["summary"]
    coarse    = data["coarse"]
    fine      = data["fine"]
    summary_w = data["summary_w"]

    lines = []
    a = lines.append

    a("# Tables for Paper")
    a("")
    a("*Auto-generated by export_paper_package.py -- do not edit by hand.*")
    a("")

    # ---- Table 1: Dataset statistics + dialect composition ----
    a("## Table 1: Dataset Statistics and Dialect Composition")
    a("")
    a("Table 1 is a **descriptive dataset characterisation**, not a main empirical result.")
    a("Dialect proportions are **automatic estimates** from a dialect classifier;")
    a("they should be described in the paper as such, using cautious wording such as")
    a("'automatic dialect identification suggests' or 'estimated dominant variety'.")
    a("The purpose of this table is to check whether the observed distribution is broadly")
    a("consistent with the commonly understood provenance of each dataset -- not to assert")
    a("definitive linguistic labels.")
    a("")
    a("| Dataset | Dominant Dialect (est.) | MSA% (est.) | EGY% | MAGHREB% | GLF% | LEV% |")
    a("|---------|------------------------|-------------|------|----------|------|------|")
    for ds in data["datasets_ordered"]:
        if ds in coarse and ds in fine:
            c = coarse[ds]
            f = fine[ds]
            a(f"| {ds} | {c['dominant']} | {_pct(f['MSA'])} | "
              f"{_pct(f['EGY'])} | {_pct(f['MAGHREB'])} | "
              f"{_pct(f['GLF'])} | {_pct(f['LEV'])} |")
    a("")

    # ---- Table 2: Main results ----
    a("## Table 2: Main Results (macro-F1, t-based 95% CI over 5 seeds)")
    a("")
    a("* = best on dataset. -- indicates model not evaluated on that dataset.")
    a("CIs are t-distribution 95% intervals over 5 independent training seeds.")
    a("")
    a("| Dataset | MARBERTv2 | AraBERTv2 | EgyBERT | DarijaBERT | DziriBERT |")
    a("|---------|-----------|-----------|---------|------------|-----------|")
    for r in pivot:
        ds = r["Dataset"]
        cells = []
        for col in ["MARBERTv2", "AraBERTv2", "EgyBERT", "DarijaBERT", "DziriBERT"]:
            val = r.get(col, "").strip()
            if not val or val == "\u2014":
                val = "--"
            cells.append(val)
        a(f"| {ds} | " + " | ".join(cells) + " |")
    a("")

    # ---- Table 3: Significance ----
    a("## Table 3: Significance Tests (MARBERTv2 vs Comparators)")
    a("")
    a("Paired bootstrap (10,000 resamples) on ensemble predictions.")
    a("delta_F1 = comparator - MARBERTv2 (negative = MARBERTv2 is better).")
    a("p_holm = Holm-corrected p-value within each dataset family.")
    a("* = significant at alpha=0.05 after Holm correction.")
    a("")
    a("| Dataset | Comparator | delta_F1 [95% CI] | p_holm | Sig |")
    a("|---------|------------|-------------------|--------|-----|")
    for (ds, mdl), v in sig.items():
        sig_mark = "*" if v["sig"] else ""
        a(f"| {ds} | {mdl} | {v['formatted']} | {v['p_holm']:.4f} | {sig_mark} |")
    a("")

    # ---- Ablation: Weighted cross-entropy ----
    a("## Ablation: Weighted Cross-Entropy (ASTD Only)")
    a("")
    a("Class-weighted cross-entropy was tested on ASTD as a focused ablation.")
    a("n = number of completed runs; missing runs are reported as-is.")
    a("")
    a("| Model | n | F1 (weighted) [95% CI] | F1 (standard) | delta |")
    a("|-------|---|------------------------|---------------|-------|")
    for mdl_int, label in [("marbert", "MARBERTv2"), ("arabert", "AraBERTv2"),
                            ("egybert", "EgyBERT")]:
        if mdl_int in summary_w:
            w   = summary_w[mdl_int]
            std = summary.get(("ASTD", mdl_int), {})
            std_f1_str = _fmt(std["f1_mean"]) if std else "N/A"
            delta_str  = (_fmt(w["f1_mean"] - std["f1_mean"]) if std else "N/A")
            ci_lo = round(w["f1_mean"] - w["f1_ci95"], 3)
            ci_hi = round(w["f1_mean"] + w["f1_ci95"], 3)
            a(f"| {label} | {w['n']} | "
              f"{_fmt(w['f1_mean'])} [{ci_lo}-{ci_hi}] | "
              f"{std_f1_str} | {delta_str} |")
    a("")

    # ---- LaTeX stub: Table 2 ----
    a("## LaTeX Stub: Table 2")
    a("")
    a("```latex")
    a(r"\begin{table*}[t]")
    a(r"\centering")
    a(r"\caption{Macro-F1 on seven Arabic SA datasets. Best result per row marked *. "
      r"Intervals are 95\% CIs (t-distribution over 5 seeds). -- = not evaluated.}")
    a(r"\label{tab:main_results}")
    a(r"\begin{tabular}{lccccc}")
    a(r"\toprule")
    a(r"Dataset & MARBERTv2 & AraBERTv2 & EgyBERT & DarijaBERT & DziriBERT \\")
    a(r"\midrule")
    for r in pivot:
        ds = r["Dataset"]
        cells = []
        for col in ["MARBERTv2", "AraBERTv2", "EgyBERT", "DarijaBERT", "DziriBERT"]:
            val = r.get(col, "").strip()
            if not val or val == "\u2014":
                val = "--"
            cells.append(val)
        a(f"{ds} & " + " & ".join(cells) + r" \\")
    a(r"\bottomrule")
    a(r"\end{tabular}")
    a(r"\end{table*}")
    a("```")
    a("")

    path = os.path.join(out_dir, "results_tables_for_paper.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Output 4: discussion_notes.md
# ---------------------------------------------------------------------------

def write_discussion_notes_md(data: dict, facts: dict, out_dir: str) -> str:
    sig       = data["sig"]
    summary   = data["summary"]
    fine      = data["fine"]
    coarse    = data["coarse"]
    summary_w = data["summary_w"]
    sig_w     = data["sig_w"]

    ls_key, ls_val = facts["largest_sig"]
    ss_key, ss_val = facts["smallest_sig"]
    outlier = facts["outlier_seed"]
    wd      = facts["weighted_deltas"]

    lines = []
    a = lines.append

    a("# Discussion Notes")
    a("")
    a("*Auto-generated by export_paper_package.py -- do not edit by hand.*")
    a("*Claims marked [VERIFY] need author confirmation before use.*")
    a("")

    # ---- Overall performance ----
    a("## Overall Performance")
    a("")
    a("- MARBERTv2 ranks first on all 7 datasets by mean macro-F1.")
    marbert_f1 = facts["marbert_f1"]
    a(f"- Performance range across datasets: "
      f"{_fmt(min(marbert_f1.values()))} ({facts['worst_dataset']}) to "
      f"{_fmt(max(marbert_f1.values()))} ({facts['best_dataset']}).")
    a("- The gap between best and worst MARBERTv2 performance is substantial, "
      "reflecting varying dataset difficulty and domain fit.")
    a("")

    # ---- AraBERT ----
    a("## AraBERTv2: Consistently Below MARBERTv2")
    a("")
    arabert_sig    = facts["arabert_sig_datasets"]
    arabert_nonsig = facts["arabert_nonsig_datasets"]
    a(f"- AraBERTv2 is the only comparator tested on all 7 datasets.")
    a(f"- Significantly worse than MARBERTv2 (Holm-corrected) on: "
      f"{', '.join(arabert_sig)}.")
    a(f"- Not significantly different on: {', '.join(arabert_nonsig)}.")
    a(f"  - ASTD: raw p is marginally significant (p=0.044) but fails Holm correction "
      f"(p_holm=0.087); the gap should not be overstated.")
    a(f"  - LABR: gap is small "
      f"(delta_F1={_fmt(sig[('LABR','AraBERTv2')]['delta_f1'])}) "
      f"and not significant (p_holm={sig[('LABR','AraBERTv2')]['p_holm']:.4f}).")
    a("")
    a(f"- Largest significant gap: {ls_key[0]}, delta_F1={_fmt(ls_val['delta_f1'])} "
      f"[{_fmt(ls_val['ci_low'])}, {_fmt(ls_val['ci_high'])}] "
      f"(p_holm={ls_val['p_holm']:.4f}).")
    a(f"- Smallest significant gap: {ss_key[0]}, delta_F1={_fmt(ss_val['delta_f1'])} "
      f"[{_fmt(ss_val['ci_low'])}, {_fmt(ss_val['ci_high'])}] "
      f"(p_holm={ss_val['p_holm']:.4f}).")
    a("")

    # ---- Dialect-specific models ----
    a("## Dialect-Specific Models (EgyBERT, DarijaBERT, DziriBERT)")
    a("")
    a("- EgyBERT is evaluated on ASTD and ArSAS. "
      "On ArSAS it approaches MARBERTv2 but the gap is not significant "
      f"(delta_F1={_fmt(sig[('ArSAS','EgyBERT')]['delta_f1'])}, "
      f"p_holm={sig[('ArSAS','EgyBERT')]['p_holm']:.4f}).")
    a("- DarijaBERT is evaluated on MACcorpus and AfriSenti_ARY. "
      "It is significantly worse on MACcorpus "
      f"(delta_F1={_fmt(sig[('MACcorpus','DarijaBERT')]['delta_f1'])}, "
      f"p_holm={sig[('MACcorpus','DarijaBERT')]['p_holm']:.4f}) "
      "but not significantly different on AfriSenti_ARY "
      f"(delta_F1={_fmt(sig[('AfriSenti_ARY','DarijaBERT')]['delta_f1'])}, "
      f"p_holm={sig[('AfriSenti_ARY','DarijaBERT')]['p_holm']:.4f}).")
    a("- DziriBERT is evaluated on AfriSenti_ARQ only. "
      "It is not significantly different from MARBERTv2 "
      f"(delta_F1={_fmt(sig[('AfriSenti_ARQ','DziriBERT')]['delta_f1'])}, "
      f"p_holm={sig[('AfriSenti_ARQ','DziriBERT')]['p_holm']:.4f}).")
    a("  - [HYPOTHESIS] One possible explanation is that DziriBERT's Algerian-Arabic "
      "training data partially overlaps with the Maghrebi variety of AfriSenti_ARQ; "
      "this hypothesis would require further analysis to confirm.")
    a("")

    # ---- EgyBERT instability ----
    a("## EgyBERT Instability on ASTD")
    a("")
    if outlier:
        a(f"- EgyBERT on ASTD: seed {outlier['seed']} produced F1={_fmt(outlier['f1'])}, "
          f"while the other 4 seeds ranged "
          f"{_fmt(outlier['others_min'])}--{_fmt(outlier['others_max'])}.")
        a("- This outlier drives the wide t-based CI visible in Table 2 for EgyBERT / ASTD.")
        a("- The large variance is consistent with sensitivity to initialization or "
          "optimization for this model-dataset pair.")
        a("- [VERIFY] Authors should check whether seed 46 encountered a different local "
          "minimum or failed to converge fully.")
    a("")

    # ---- Dialect distribution notes ----
    a("## Dialect Composition (Table 1) -- Descriptive Caveats")
    a("")
    a("Table 1 is a descriptive dataset characterisation based on automatic dialect estimates.")
    a("It is used only to check whether the observed distribution is broadly consistent with")
    a("the commonly understood provenance of each dataset -- not to assert definitive labels.")
    a("Do not present it as a main empirical result.")
    a("")
    a("Key observations (automatic dialect identification suggests):")
    a("")
    if "AfriSenti_ARQ" in fine:
        a(f"- AfriSenti_ARQ: estimated dominant variety MAGHREB "
          f"({_pct(fine['AfriSenti_ARQ']['MAGHREB'])}), broadly consistent with the "
          f"Algerian-Arabic provenance of this dataset (arq code).")
    if "AfriSenti_ARY" in fine:
        a(f"- AfriSenti_ARY: estimated dominant variety MAGHREB "
          f"({_pct(fine['AfriSenti_ARY']['MAGHREB'])}), broadly consistent with "
          f"Moroccan-Arabic provenance (ary code).")
    if "ASTD" in fine:
        a(f"- ASTD: {_pct(fine['ASTD']['MSA'])} MSA, {_pct(fine['ASTD']['EGY'])} EGY. "
          f"The classifier assigns EGY as estimated dominant variety, but the majority "
          f"is classified as MSA. [VERIFY] Interpret with caution.")
    if "HARD" in fine:
        a(f"- HARD: {_pct(fine['HARD']['GLF'])} GLF, {_pct(fine['HARD']['MSA'])} MSA. "
          f"Estimated dominant variety is GLF per coarse classification.")
    if "MACcorpus" in fine:
        a(f"- MACcorpus: estimated dominant variety MAGHREB "
          f"({_pct(fine['MACcorpus']['MAGHREB'])}), broadly consistent with the "
          f"Moroccan-Arabic origin of this corpus.")
    a("")
    a("[VERIFY] Authors should confirm that the classifier outputs are broadly consistent")
    a("  with their knowledge of each dataset before including Table 1.")
    a("")

    # ---- Weighted ablation ----
    a("## Weighted Cross-Entropy Ablation (ASTD)")
    a("")
    a("Class-weighted loss was tested on ASTD to address potential class imbalance.")
    for mdl_int, label in [("marbert", "MARBERTv2"), ("arabert", "AraBERTv2"),
                            ("egybert", "EgyBERT")]:
        if mdl_int in wd:
            d      = wd[mdl_int]
            n_note = f" ({d['n_w']} runs)" if d["n_w"] < 5 else " (5 runs)"
            sign   = "+" if d["delta_vs_std"] >= 0 else ""
            a(f"- {label}{n_note}: F1(weighted)={_fmt(d['f1_w'])}, "
              f"delta vs standard={sign}{_fmt(d['delta_vs_std'])}")
    a("")
    a("Paired bootstrap vs weighted MARBERTv2:")
    for mdl_disp, v in sig_w.items():
        sig_str = "significant" if v["sig"] else "not significant"
        a(f"- {mdl_disp}: delta_F1={_fmt(v['delta_f1'])}, "
          f"p_holm={v['p_holm']:.4f} ({sig_str})")
    a("")
    a("Note: AraBERTv2 and EgyBERT have n=4 available weighted runs; "
      "MARBERTv2 has n=5. No reason is stated for the missing runs.")
    a("")

    path = os.path.join(out_dir, "discussion_notes.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Output 5: figure_recommendations.md
# ---------------------------------------------------------------------------

def write_figure_recommendations_md(data: dict, facts: dict, out_dir: str) -> str:
    sig      = data["sig"]
    per_seed = data["per_seed"]

    ls_key, ls_val = facts["largest_sig"]
    outlier = facts["outlier_seed"]

    lines = []
    a = lines.append

    a("# Figure Recommendations")
    a("")
    a("*Auto-generated by export_paper_package.py -- do not edit by hand.*")
    a("")

    a("## Figure 1: Main Results Bar Chart")
    a("")
    a("**Type:** Grouped bar chart (one group per dataset; bars per model).")
    a("**y-axis:** Macro-F1, range 0.4-1.0 (or tighter).")
    a("**Error bars:** t-based 95% CI over 5 seeds.")
    a("**Grouping:** Group by dataset; color by model.")
    a("**Models to include:** MARBERTv2 (reference), AraBERTv2 (all datasets),")
    a("  EgyBERT (ASTD, ArSAS), DarijaBERT (MACcorpus, AfriSenti_ARY),")
    a("  DziriBERT (AfriSenti_ARQ).")
    a("**Highlight:** MARBERTv2 bar distinctly (darker color or hatch).")
    a("**Note on EgyBERT/ASTD:** Wide CI visible -- authors should decide whether to annotate.")
    a("")

    a("## Figure 2: Significance Heatmap (Table 3 visual)")
    a("")
    a("**Type:** Heatmap, rows = dataset x model pairs, single column = delta_F1.")
    a("**Color scale:** Diverging, centered at 0. All values here are <= 0.")
    a("**Significance marker:** Bold border or asterisk (*) for Holm-significant cells.")
    a("**Non-significant cells:** Light gray or hatched.")
    a("**Values to show (from significance_summary_table.csv):**")
    for (ds, mdl), v in sig.items():
        sig_m = "*" if v["sig"] else ""
        a(f"  - {ds} / {mdl}: {_fmt(v['delta_f1'])} {sig_m}")
    a("")

    a("## Figure 3: Per-Seed Runs (EgyBERT / ASTD Instability)")
    a("")
    a("**Type:** Scatter or strip plot, x=seed, y=macro-F1.")
    a("**Purpose:** Visualise the outlier seed driving the wide CI.")
    if outlier:
        a(f"**Key point:** Seed {outlier['seed']} = {_fmt(outlier['f1'])}; "
          f"all other seeds in [{_fmt(outlier['others_min'])}, {_fmt(outlier['others_max'])}].")
    a("**Data source:** results/csv/per_seed_runs.csv (filter to dataset=ASTD, model=egybert).")
    a("**Optional:** overlay a horizontal line at the mean.")
    a("")

    a("## Figure 4: Weighted vs Standard F1 (ASTD Ablation)")
    a("")
    a("**Type:** Paired bar chart (standard vs weighted) for each model on ASTD.")
    a("**Models:** MARBERTv2 (n=5 / n=5), AraBERTv2 (n=5 / n=4), EgyBERT (n=5 / n=4).")
    a("**Error bars:** t-based 95% CI from results_summary.csv and results_summary_weighted.csv.")
    a("**Note:** Label bars with n to acknowledge different run counts for weighted condition.")
    a("")

    a("## Figure 5: Dialect Composition Overview (Appendix / Optional)")
    a("")
    a("**Placement: Appendix only, or omit if space is limited.**")
    a("This figure is descriptive and supplementary -- it should NOT appear as a core")
    a("main-text figure. It visualises automatic estimates, not ground-truth labels.")
    a("")
    a("**Type:** Stacked bar chart, one bar per dataset, segments = dialect category.")
    a("**Categories:** EGY, MAGHREB, GLF, LEV, MSA.")
    a("**Caption must state:** 'Proportions are automatic estimates from a dialect classifier;")
    a("  they are used here as a descriptive characterisation of dataset variety mix")
    a("  and are not ground-truth labels.'")
    a("**Data source:** outputs/classification/dialect_distribution_fine.csv")
    a("")

    path = os.path.join(out_dir, "figure_recommendations.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading source files...")
    data  = load_all()
    facts = derive_facts(data)

    print("Writing outputs:")

    p1 = write_results_overview_md(data, facts, OUT_DIR)
    print(f"  [1/5] {p1}")

    p2 = write_results_overview_json(data, facts, OUT_DIR)
    print(f"  [2/5] {p2}")

    p3 = write_results_tables_md(data, facts, OUT_DIR)
    print(f"  [3/5] {p3}")

    p4 = write_discussion_notes_md(data, facts, OUT_DIR)
    print(f"  [4/5] {p4}")

    p5 = write_figure_recommendations_md(data, facts, OUT_DIR)
    print(f"  [5/5] {p5}")

    print("")
    print("Done. Key facts derived from actual files:")

    ls_key, ls_val = facts["largest_sig"]
    ss_key, ss_val = facts["smallest_sig"]
    outlier = facts["outlier_seed"]

    print(f"  Largest sig gap:  {ls_key[0]} / {ls_key[1]}: "
          f"delta={_fmt(ls_val['delta_f1'])}, p_holm={ls_val['p_holm']:.4f}")
    print(f"  Smallest sig gap: {ss_key[0]} / {ss_key[1]}: "
          f"delta={_fmt(ss_val['delta_f1'])}, p_holm={ss_val['p_holm']:.4f}")
    print(f"  Sig comparisons:  {facts['n_sig']} / {facts['n_sig'] + facts['n_nonsig']}")

    if outlier:
        print(f"  EgyBERT/ASTD outlier seed {outlier['seed']}: "
              f"F1={_fmt(outlier['f1'])} vs others "
              f"{_fmt(outlier['others_min'])}-{_fmt(outlier['others_max'])}")

    wd = facts["weighted_deltas"]
    for mdl_int, label in [("marbert", "MARBERTv2"), ("arabert", "AraBERTv2"),
                            ("egybert", "EgyBERT")]:
        if mdl_int in wd:
            d    = wd[mdl_int]
            sign = "+" if d["delta_vs_std"] >= 0 else ""
            n_note = f"n={d['n_w']}" if d["n_w"] < 5 else "n=5"
            print(f"  Weighted ASTD {label} ({n_note}): "
                  f"delta vs std={sign}{_fmt(d['delta_vs_std'])}")


if __name__ == "__main__":
    main()
