#!/usr/bin/env bash
# run_experiment.sh — reproducible launcher for marbertExperiment
#
# PYTHONHASHSEED must be set here, before the interpreter starts.
# Setting it inside Python (os.environ) has no effect.
#
# Usage:
#   bash run_experiment.sh [--model MODEL] [--dataset DATASET] [--seed SEED]
#                          [--split-seed SPLIT_SEED] [--class-weighted-loss]
#                          [--resume] [--overwrite] [--debug]
#
# Examples:
#   bash run_experiment.sh --model marbert --dataset astd --seed 42 --split-seed 42
#   bash run_experiment.sh --model arabert --dataset hard --seed 123 --class-weighted-loss
#
# To run multiple seeds in sequence:
#   for SEED in 42 123 456; do
#       bash run_experiment.sh --model marbert --dataset astd --seed $SEED --split-seed 42
#   done

set -euo pipefail

# ── Reproducibility ────────────────────────────────────────────────────────────
# PYTHONHASHSEED must be fixed before the interpreter starts.
# We use a fixed value (0) to disable hash randomisation entirely.
# Override by setting PYTHONHASHSEED in the environment before calling this script:
#   PYTHONHASHSEED=42 bash run_experiment.sh ...
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"

# Required by PyTorch for deterministic CUDA ops on Ampere+ GPUs.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

# ── Python executable ──────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

echo "[launcher] PYTHONHASHSEED=${PYTHONHASHSEED}"
echo "[launcher] CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}"
echo "[launcher] python: $(${PYTHON} --version)"
echo "[launcher] args: $*"
echo ""

exec "${PYTHON}" scripts/run_sa.py --resume "$@"
