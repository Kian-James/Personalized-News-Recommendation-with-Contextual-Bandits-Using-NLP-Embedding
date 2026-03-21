#!/usr/bin/env bash
# run.sh — One-command reproduction of all results
# Usage:
#   bash run.sh          # full run
#   bash run.sh --fast   # quick smoke-test (~5 min)

set -euo pipefail

FAST_FLAG=""
if [[ "${1:-}" == "--fast" ]]; then
    FAST_FLAG="--fast"
    echo ">>> [FAST MODE] Using small data subset"
fi

echo ""
echo "============================================================"
echo "  News Recommender System — Full Pipeline"
echo "============================================================"
echo ""

# 0. Environment check
echo "[0/4] Checking environment..."
python -c "import torch, sklearn, sentence_transformers, pandas; \
           print(f'  torch={torch.__version__}  sklearn={sklearn.__version__}')"

# 1. Data pipeline — must run first to generate splits
echo ""
echo "[1/4] Running data pipeline..."
python src/data_pipeline.py

# 2. EDA
echo ""
echo "[2/4] Running EDA..."
python notebooks/01_eda.py

# 3. Train all components (NLP + CNN + RL) — saves to next run folder
echo ""
echo "[3/4] Training..."
python src/train.py $FAST_FLAG

# 4. Evaluation — auto-reads from latest run folder
echo ""
echo "[4/4] Running evaluation suite..."
python src/eval.py

echo ""
echo "============================================================"
echo "  All done!"
echo "  Results saved to experiments/results/run*/"
echo "  Run 'python src/eval.py --run run1' to re-evaluate a past run."
echo "============================================================"