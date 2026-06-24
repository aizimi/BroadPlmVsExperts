#!/usr/bin/env bash
# run_aggregate.sh — aggregate results across all seeds and produce summary CSV
#
# Reads checkpoints/ and writes all output files under results/ by default.
#
# Usage:
#   bash run_aggregate.sh [--root DIR] [--out FILE] [--verbose]
#
# Examples:
#   bash run_aggregate.sh
#   bash run_aggregate.sh --root checkpoints --out results/results_summary.csv --verbose

set -euo pipefail

# ── Global log file ────────────────────────────────────────────────────────────
mkdir -p logs
LOG_FILE="logs/aggregate_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
START_TIME=$(date +%s)

# ── Environment ────────────────────────────────────────────────────────────────
# Assumes the correct conda/venv environment is already activated.
# Override: PYTHON=/path/to/python bash run_aggregate.sh
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

# Preflight check — must be run from the project root.
[[ -f scripts/aggregate_results.py ]] || { echo "ERROR: scripts/aggregate_results.py not found. Run from project root."; exit 1; }

log() { local e=$(( $(date +%s) - START_TIME )); printf "[%s +%02d:%02d:%02d] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" $(( e/3600 )) $(( e%3600/60 )) $(( e%60 )) "$*"; }

log "Log file: ${LOG_FILE}"
log "Python: $(${PYTHON} --version 2>&1)"
log "CMD: ${PYTHON} scripts/aggregate_results.py $*"
echo ""

"${PYTHON}" scripts/aggregate_results.py "$@"

echo ""
log "Aggregation complete. Log saved to: ${LOG_FILE}"
