#!/usr/bin/env bash
# run_astd_weighted.sh — ASTD weighted-loss ablation (all models, all seeds)
#
# Usage:
#   bash run_astd_weighted.sh
#   DRY_RUN=1 bash run_astd_weighted.sh

set -euo pipefail

mkdir -p logs
LOG_FILE="logs/astd_weighted_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
START_TIME=$(date +%s)

export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

DATASET="astd"
MODELS=(marbert arabert egybert)
SEEDS=(42 43 44 45 46)
SPLIT_SEED=42
DRY_RUN="${DRY_RUN:-0}"

log() { local e=$(( $(date +%s) - START_TIME )); printf "[%s +%02d:%02d:%02d] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" $(( e/3600 )) $(( e%3600/60 )) $(( e%60 )) "$*"; }

run_cmd() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DRY RUN] $*"
    else
        "$@"
    fi
}

total=$(( ${#MODELS[@]} * ${#SEEDS[@]} ))
log "Log file: ${LOG_FILE}"
log "PYTHONHASHSEED=${PYTHONHASHSEED}"
log "Python: $(${PYTHON} --version 2>&1)"
log "Dataset: ${DATASET} (weighted loss)"
log "Models: ${MODELS[*]}"
log "Seeds: ${SEEDS[*]}"
log "Total runs: ${total}"
[[ "${DRY_RUN}" == "1" ]] && log "DRY RUN mode"
echo ""

completed=0
failed=0
failed_runs=()

for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        completed=$(( completed + 1 ))
        out_dir="checkpoints/${model}_${DATASET}_weighted_split_${SPLIT_SEED}/seed_${seed}"

        log "[${completed}/${total}] model=${model} seed=${seed}"
        log "  -> ${out_dir}"

        if run_cmd "${PYTHON}" scripts/run_sa.py \
            --model              "${model}"      \
            --dataset            "${DATASET}"    \
            --seed               "${seed}"       \
            --split-seed         "${SPLIT_SEED}" \
            --out                "${out_dir}"    \
            --class-weighted-loss                \
            --resume; then
            log "  OK"
        else
            log "  FAILED"
            failed=$(( failed + 1 ))
            failed_runs+=("${model}/seed_${seed}")
        fi
        echo ""
    done
done

echo "========================================"
log "Done. ${completed} runs attempted, ${failed} failed."
if [[ ${failed} -gt 0 ]]; then
    log "Failed runs:"
    for r in "${failed_runs[@]}"; do
        echo "    - ${r}"
    done
    exit 1
fi
