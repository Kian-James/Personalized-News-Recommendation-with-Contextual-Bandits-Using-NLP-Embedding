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

# 3. Train all components (NLP + CNN + RL)
echo ""
echo "[3/4] Training..."
python src/train.py $FAST_FLAG

# 4. Evaluation
echo ""
echo "[4/4] Running evaluation suite..."
python src/eval.py

echo ""
echo "============================================================"
echo "  All done! Results saved to experiments/results/"
echo "============================================================"
echo ""
echo "Key output files:"
echo "  experiments/results/cnn_metrics.json      — TextCNN accuracy & F1"
echo "  experiments/results/ranking_metrics.json  — nDCG & Hit@K"
echo "  experiments/results/ablation_results.csv  — Ablation study"
echo "  experiments/results/rl_summary.json       — RL agent comparison"
echo "  experiments/results/*.png                 — All plots & curves"