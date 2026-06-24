#!/usr/bin/env bash
# run_analysis.sh — runs dialect classification and token length analysis
#
# These are pre-experiment analysis scripts, independent of training.
# Outputs:
#   outputs/classification/dialect_distribution_fine.csv
#   outputs/classification/dialect_distribution_coarse.csv
#   outputs/length_analysis/   (written by analyze_lengths.py)
#
# Usage:
#   bash run_analysis.sh

set -euo pipefail

# ── Global log file ────────────────────────────────────────────────────────────
mkdir -p logs
LOG_FILE="logs/analysis_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
START_TIME=$(date +%s)

# ── Environment ────────────────────────────────────────────────────────────────
# Assumes the correct conda/venv environment is already activated.
# Override: PYTHON=/path/to/python bash run_analysis.sh
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

# Preflight check — must be run from the project root.
[[ -f scripts/classification.py ]]  || { echo "ERROR: scripts/classification.py not found. Run from project root."; exit 1; }
[[ -f scripts/analyze_lengths.py ]] || { echo "ERROR: scripts/analyze_lengths.py not found. Run from project root."; exit 1; }

log() { local e=$(( $(date +%s) - START_TIME )); printf "[%s +%02d:%02d:%02d] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" $(( e/3600 )) $(( e%3600/60 )) $(( e%60 )) "$*"; }

log "Log file: ${LOG_FILE}"
log "Python: $(${PYTHON} --version 2>&1)"
echo ""

# ── Dialect classification ─────────────────────────────────────────────────────
log "Running dialect classification (classification.py)..."
log "CMD: ${PYTHON} scripts/classification.py"
"${PYTHON}" scripts/classification.py
log "Dialect classification done."
echo ""

# ── Token length analysis ──────────────────────────────────────────────────────
log "Running token length analysis (analyze_lengths.py)..."
log "CMD: ${PYTHON} scripts/analyze_lengths.py"
"${PYTHON}" scripts/analyze_lengths.py
log "Token length analysis done."
echo ""

log "All analysis complete. Log saved to: ${LOG_FILE}"
