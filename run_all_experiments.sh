#!/usr/bin/env bash
# run_all_experiments.sh — hypothesis-driven experiment launcher
#
# Design:
#   - Runs are defined by an explicit dataset → models mapping (RUN_PLAN).
#   - NOT a Cartesian product of all models × datasets.
#   - One run = one dataset × one model × one train seed.
#   - Split seed is fixed at 42 for all runs.
#   - Class-weighted loss is OFF (enable only for ablations via a separate script).
#   - Execution is sequential.
#
# Usage:
#   bash run_all_experiments.sh
#
# To do a dry run (print commands without executing):
#   DRY_RUN=1 bash run_all_experiments.sh
#
# To run all experiments in debug mode (small data, fast epoch):
#   DEBUG=1 bash run_all_experiments.sh

set -euo pipefail

# ── Global log file ────────────────────────────────────────────────────────────
mkdir -p logs
LOG_FILE="logs/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
START_TIME=$(date +%s)

# ── Reproducibility ────────────────────────────────────────────────────────────
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

# Assumes the correct conda/venv environment is already activated.
# Override by setting PYTHON before calling this script:
#   PYTHON=/path/to/env/bin/python bash run_all_experiments.sh
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

SPLIT_SEED=42
SEEDS=(42 43 44 45 46)
DRY_RUN="${DRY_RUN:-0}"
DEBUG="${DEBUG:-0}"
DEBUG_FLAG=""
[[ "${DEBUG}" == "1" ]] && DEBUG_FLAG="--debug"

# ── Preflight checks ───────────────────────────────────────────────────────────
[[ -f scripts/run_sa.py ]] || { echo "ERROR: scripts/run_sa.py not found. Run this script from the project root."; exit 1; }
[[ ${#SEEDS[@]} -gt 0 ]] || { echo "ERROR: No seeds defined."; exit 1; }

# ── Run plan ───────────────────────────────────────────────────────────────────
# Format: "DATASET:model1,model2,..."
# Edit this list to finalize your experimental design.
# MARBERT is the reference model and appears on every dataset.
# Other models are dataset-specific based on methodological relevance.
RUN_PLAN=(
    "astd:marbert,arabert,egybert"
    "arsas:marbert,arabert,egybert"
    "afrisenti_ary:marbert,arabert,darijabert"
    "afrisenti_arq:marbert,arabert,dziribert"
    "labr:marbert,arabert"
    "maccorpus:marbert,arabert,darijabert"
    "hard:marbert,arabert"
)

# ── Weighted-loss ablation plan ────────────────────────────────────────────────
# Class-weighted cross-entropy, run on ASTD only (known class imbalance).
# Output dirs include _weighted suffix to avoid colliding with baseline runs.
WEIGHTED_RUN_PLAN=(
    "astd:marbert,arabert,egybert"
)

# ── Helpers ────────────────────────────────────────────────────────────────────
log() { local e=$(( $(date +%s) - START_TIME )); printf "[%s +%02d:%02d:%02d] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" $(( e/3600 )) $(( e%3600/60 )) $(( e%60 )) "$*"; }

run_cmd() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DRY RUN] $*"
    else
        "$@"
    fi
}

# ── Startup info ───────────────────────────────────────────────────────────────
log "Log file: ${LOG_FILE}"
log "PYTHONHASHSEED=${PYTHONHASHSEED}"
log "CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}"
log "Python: $(${PYTHON} --version 2>&1)"
log "Split seed: ${SPLIT_SEED}"
log "Train seeds: ${SEEDS[*]}"
[[ "${DRY_RUN}" == "1" ]] && log "DRY RUN mode — no training will execute"
echo ""

log "Run plan:"
for entry in "${RUN_PLAN[@]}"; do
    log "  ${entry}"
done
echo ""

# ── Count total runs ───────────────────────────────────────────────────────────
total=0
for entry in "${RUN_PLAN[@]}"; do
    models_str="${entry##*:}"
    IFS=',' read -ra models <<< "${models_str}"
    total=$(( total + ${#models[@]} * ${#SEEDS[@]} ))
done
for entry in "${WEIGHTED_RUN_PLAN[@]}"; do
    models_str="${entry##*:}"
    IFS=',' read -ra models <<< "${models_str}"
    total=$(( total + ${#models[@]} * ${#SEEDS[@]} ))
done
log "Total runs planned: ${total} (including weighted-loss ablations)"
echo ""

# ── Main loop ──────────────────────────────────────────────────────────────────
completed=0
failed=0
failed_runs=()

for entry in "${RUN_PLAN[@]}"; do
    dataset="${entry%%:*}"
    models_str="${entry##*:}"
    IFS=',' read -ra models <<< "${models_str}"

    for model in "${models[@]}"; do
        for seed in "${SEEDS[@]}"; do
            completed=$(( completed + 1 ))
            out_dir="checkpoints/${model}_${dataset}_split_${SPLIT_SEED}/seed_${seed}"

            log "[${completed}/${total}] dataset=${dataset} model=${model} seed=${seed}"
            log "  -> ${out_dir}"
            log "  CMD: ${PYTHON} scripts/run_sa.py --model ${model} --dataset ${dataset} --seed ${seed} --split-seed ${SPLIT_SEED} --out ${out_dir} --resume"

            if run_cmd "${PYTHON}" scripts/run_sa.py \
                --model      "${model}"       \
                --dataset    "${dataset}"     \
                --seed       "${seed}"        \
                --split-seed "${SPLIT_SEED}"  \
                --out        "${out_dir}"     \
                --resume ${DEBUG_FLAG}; then
                log "  OK"
            else
                log "  FAILED"
                failed=$(( failed + 1 ))
                failed_runs+=("${dataset}/${model}/seed_${seed}")
            fi
            echo ""
        done
    done
done

# ── Weighted-loss ablation loop ───────────────────────────────────────────────
log "Starting weighted-loss ablation runs..."
echo ""

for entry in "${WEIGHTED_RUN_PLAN[@]}"; do
    dataset="${entry%%:*}"
    models_str="${entry##*:}"
    IFS=',' read -ra models <<< "${models_str}"

    for model in "${models[@]}"; do
        for seed in "${SEEDS[@]}"; do
            completed=$(( completed + 1 ))
            out_dir="checkpoints/${model}_${dataset}_weighted_split_${SPLIT_SEED}/seed_${seed}"

            log "[${completed}/${total}] dataset=${dataset} model=${model} seed=${seed} (weighted)"
            log "  -> ${out_dir}"
            log "  CMD: ${PYTHON} scripts/run_sa.py --model ${model} --dataset ${dataset} --seed ${seed} --split-seed ${SPLIT_SEED} --out ${out_dir} --class-weighted-loss --resume"

            if run_cmd "${PYTHON}" scripts/run_sa.py \
                --model                "${model}"       \
                --dataset              "${dataset}"     \
                --seed                 "${seed}"        \
                --split-seed           "${SPLIT_SEED}"  \
                --out                  "${out_dir}"     \
                --class-weighted-loss                   \
                --resume ${DEBUG_FLAG}; then
                log "  OK"
            else
                log "  FAILED"
                failed=$(( failed + 1 ))
                failed_runs+=("${dataset}/${model}_weighted/seed_${seed}")
            fi
            echo ""
        done
    done
done

# ── Summary ────────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════"
log "Done. ${completed} runs attempted, ${failed} failed."
if [[ ${failed} -gt 0 ]]; then
    log "Failed runs:"
    for r in "${failed_runs[@]}"; do
        echo "    - ${r}"
    done
    exit 1
fi
