# Broad-Coverage versus Dialect-Specialized Arabic PLMs for Sentiment Analysis

> **Paper:** Broad-Coverage versus Dialect-Specialized Arabic PLMs for Sentiment Analysis:
> A Controlled Multi-Dataset Evaluation with Paired Statistical Testing

This repository contains the code, data splits, result tables, and figures associated
with the paper above. It is structured to support reproducibility of the reported
experiments.

---

## Overview

We compare five Arabic pre-trained language models — two broad-coverage models
(MARBERTv2, AraBERTv2) and three dialect-specialized models (EgyBERT, DarijaBERT,
DziriBERT) — for three-class Arabic sentiment analysis across seven dialectal
datasets. All models are fine-tuned under identical hyperparameter conditions.
Statistical significance is assessed using paired bootstrap resampling on ensemble
predictions across five random seeds, with Holm–Bonferroni correction for multiple
comparisons.

---

## Models

| Alias | HuggingFace checkpoint | Type |
|---|---|---|
| `marbert` | `UBC-NLP/MARBERTv2` | Broad-coverage |
| `arabert` | `aubmindlab/bert-base-arabertv2` | Broad-coverage |
| `egybert` | `faisalq/EgyBERT` | Dialect-specialized (Egyptian) |
| `darijabert` | `SI2M-Lab/DarijaBERT` | Dialect-specialized (Moroccan) |
| `dziribert` | `alger-ia/dziribert` | Dialect-specialized (Algerian) |

A full HuggingFace model ID can also be passed directly to `--model`.

---

## Datasets

| Alias | Dataset | Variety | Source |
|---|---|---|---|
| `astd` | ASTD | Egyptian | HuggingFace `arbml/ASTD` |
| `arsas` | ArSAS | Multi-dialectal | HuggingFace `arbml/ArSAS` |
| `mac` / `maccorpus` | MACcorpus | Moroccan | `data/MACcorpus.csv` (included) |
| `afrisenti_ary` | AfriSenti_ARY | Moroccan | HuggingFace `masakhane/afrisenti` (ary) |
| `afrisenti_arq` | AfriSenti_ARQ | Algerian | HuggingFace `masakhane/afrisenti` (arq) |
| `labr` | LABR | Multi-dialectal (reviews) | HuggingFace `mohamedadaly/labr` |
| `hard` | HARD | Hijazi (reviews) | HuggingFace `Elnagara/hard` |

Datasets loaded from HuggingFace are downloaded automatically on first run.
See `data/README.md` for data placement details and citation requirements.

**Dataset note:** Users must follow the license and citation requirements of each
dataset. Users of MACcorpus should cite Garouani and Kharroubi (2022) — see
`data/README.md` for the full citation.

---

## Main Metric

**Macro-F1** across the three sentiment classes (Positive, Negative, Neutral).

---

## Statistical Testing

Paired bootstrap resampling on per-seed ensemble predictions, with Holm–Bonferroni
correction applied across all pairwise comparisons within each dataset. McNemar's
test is reported as a secondary diagnostic in the appendix.

---

## Command-line Environment

All shell commands in this README are written for **Git Bash** (or any Unix-like shell
such as WSL or macOS Terminal). Windows users should run `.sh` scripts inside
**Git Bash**, not PowerShell or cmd.exe.

When specifying paths in Git Bash on Windows, use the `/c/Users/...` syntax instead
of the native `C:\Users\...` form. For example:

```bash
# Windows path in Git Bash
cd /c/Users/yourname/PycharmProjects/BroadPlmVsExperts-public
```

A PowerShell equivalent (`run_experiment.ps1`) is provided for the main training
script — see the **Running Experiments** section below.

---

## Environment Setup

Python **3.10** is recommended (tested on 3.10.20).

**Option A — Conda (recommended):**

```bash
conda env create -f environment.yml
conda activate arabic_sentiment
```

**Option B — pip:**

```bash
pip install -r requirements.txt
```

---

## Data Preparation

No manual download is needed for HuggingFace datasets. For MACcorpus, the file is
already present at `data/MACcorpus.csv`.

The reproducibility split files in `data/splits/` record the exact train/validation/
test indices used in the paper. **Do not modify these files.** If they are absent,
the loaders regenerate splits from the random state, which will produce different
splits and break comparability with reported results.

---

## Running Experiments

**Debug a single model/dataset pair (no training):**

```bash
bash run_experiment.sh --model marbert --dataset astd --debug
```

**Run one training seed:**

```bash
bash run_experiment.sh --model marbert --dataset astd --seed 42 --split-seed 42
```

Checkpoint and evaluation artifacts are saved to
`checkpoints/<model>_<dataset>_split_42/seed_<seed>/`.

**Run all experiments (all models × datasets × seeds):**

```bash
bash run_all_experiments.sh
```

Runs the full hypothesis-driven experiment plan across all five seeds sequentially.
To preview commands without executing:

```bash
DRY_RUN=1 bash run_all_experiments.sh
```

**Run the ASTD weighted-loss ablation only:**

```bash
bash run_astd_weighted.sh
```

**Windows users:** `run_experiment.ps1` provides equivalent PowerShell commands.

---

## Aggregating Results

After all training seeds are complete, run:

```bash
bash run_aggregate.sh
```

This produces per-seed CSVs, the pivoted results table, and bootstrap + McNemar
significance tests under `results/csv/` and `results/tex/`. Pass `--verbose` to
print per-run metrics during aggregation.

---

## Regenerating Figures and Tables

**Figures** (saved to `outputs/paper_figures/`):

```bash
python scripts/plot_figure1_main_results.py
python scripts/plot_figure2_significance.py
python scripts/plot_figure3_astd_weighted_ablation.py
```

Pre-generated figures are also included in `figures/`.

**Dataset statistics table:**

```bash
python scripts/report_dataset_counts.py
```

**Per-seed Macro-F1 appendix table:**

```bash
python scripts/report_per_seed_macro_f1.py
```

**Secondary diagnostics (appendix):**

```bash
python scripts/report_secondary_diagnostics.py
```

**Pre-experiment dialect and token length analysis:**

```bash
bash run_analysis.sh
```

---

## Reproducibility

- Seeds **42–46** were used for the five training runs of each model/dataset pair.
- Split seed **42** was used for all custom train/validation/test splits.
- Deterministic seeding (`PYTHONHASHSEED`, `torch.manual_seed`, cuDNN flags) is
  set automatically by the launcher scripts.

---

## Hardware

Experiments were run on a CUDA-enabled GPU. CPU execution is possible for small
tests but is not recommended for full reproduction.

---

## Results

Pre-computed result CSVs and LaTeX tables are provided in `results/` to allow
inspection without re-running training.

---

## Citation

If you use this code or results in your work, please cite:

```bibtex
@article{PLACEHOLDER,
  title   = {Broad-Coverage versus Dialect-Specialized Arabic PLMs for Sentiment
             Analysis: A Controlled Multi-Dataset Evaluation with Paired Statistical
             Testing},
  author  = {PLACEHOLDER},
  year    = {2025},
  journal = {PLACEHOLDER}
}
```

Users of MACcorpus must also cite:

```bibtex
@inproceedings{garouani2022mac,
  title     = {MAC: An Open and Free Moroccan Arabic Corpus for Sentiment Analysis},
  author    = {Garouani, Moncef and Kharroubi, Jamal},
  booktitle = {Innovations in Smart Cities Applications Volume 5},
  series    = {Lecture Notes in Networks and Systems},
  volume    = {393},
  pages     = {849--858},
  year      = {2022},
  publisher = {Springer, Cham},
  doi       = {10.1007/978-3-030-94191-8_68}
}
```
