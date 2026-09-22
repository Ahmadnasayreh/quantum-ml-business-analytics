#!/usr/bin/env bash
# =====================================================================
# reproduce.sh -- one-command reproduction of every reported result.
#
#   ./reproduce.sh smoke   structural check, a few minutes (DEFAULT)
#   ./reproduce.sh full    the numbers reported in the manuscript
#
# Requires Python 3.10 and bash (Linux, macOS, or Git Bash / WSL on Windows).
# =====================================================================

set -euo pipefail

MODE="${1:-smoke}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
    echo "usage: $0 [smoke|full]" >&2
    exit 1
fi

echo "=============================================================="
echo " qmlgwo reproduction -- mode: ${MODE}"
echo " started: $(date -Is)"
echo "=============================================================="

# --- 1. environment ---------------------------------------------------
if [[ ! -d .venv ]]; then
    echo "[1/4] creating virtual environment"
    (command -v python3 >/dev/null && python3 -m venv .venv) || python -m venv .venv
fi
# shellcheck disable=SC1091
if [[ -f .venv/Scripts/activate ]]; then source .venv/Scripts/activate; else source .venv/bin/activate; fi
echo "[1/4] installing pinned dependencies"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
python -m ipykernel install --user --name qmlgwo --display-name "Python (qmlgwo)" >/dev/null

# --- 2. data check ----------------------------------------------------
echo "[2/4] checking datasets"
MISSING=0
FILES=(
  "data/Bank Customer Churn Prediction/Bank Customer Churn Prediction.csv"
  "data/bank-marketing-uci/bank.csv"
  "data/HR Analytics Employee Promotion Data/train.csv"
  "data/Loan Prediction Problem Dataset/train_u6lujuX_CVtuZ9i.csv"
)
for f in "${FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "      MISSING: $f"
        MISSING=1
    else
        echo "      ok: $f ($(wc -l < "$f") lines)"
    fi
done
if [[ $MISSING -eq 1 ]]; then
    echo ""
    echo "Place the four public benchmark CSVs as described in data/README.md."
    exit 1
fi

# --- 3. select preset -------------------------------------------------
echo "[3/4] setting CONFIG = ${MODE^^}"
python - "$MODE" <<'PY'
import json, sys
from pathlib import Path

mode = sys.argv[1].upper()
for nb_path in Path(".").glob("Notebook_*.ipynb"):
    nb = json.loads(nb_path.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        cell["source"] = [
            line.replace("CONFIG = SMOKE", f"CONFIG = {mode}")
                .replace("CONFIG = FULL",  f"CONFIG = {mode}")
            for line in cell["source"]
        ]
    nb_path.write_text(json.dumps(nb, indent=1))
    print(f"      {nb_path.name} -> CONFIG = {mode}")
PY

# --- 4. execute -------------------------------------------------------
echo "[4/4] executing notebooks (this is the long part)"
mkdir -p executed logs
for nb in Notebook_A_bank_churn_marketing.ipynb Notebook_B_hr_loan.ipynb; do
    echo "      running ${nb}"
    jupyter nbconvert \
        --to notebook --execute "$nb" \
        --ExecutePreprocessor.kernel_name=qmlgwo \
        --ExecutePreprocessor.timeout=-1 \
        --output-dir executed \
        --output "${nb%.ipynb}_executed.ipynb" \
        2> "logs/${nb%.ipynb}.log" \
        || { echo "      FAILED -- see logs/${nb%.ipynb}.log"; exit 1; }
    echo "      done: executed/${nb%.ipynb}_executed.ipynb"
done

echo "=============================================================="
echo " finished: $(date -Is)"
echo " results : $ROOT/results/"
echo " figures : $ROOT/figures/"
echo " notebooks with outputs: $ROOT/executed/"
echo "=============================================================="
