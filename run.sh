#!/bin/bash

# -------------------------------
# One-command run for Personalized News Recommendation
# -------------------------------

echo "----------------------------"
echo "Starting Training..."
echo "----------------------------"
python3 src/train.py

echo "----------------------------"
echo "Starting Evaluation..."
echo "----------------------------"
python3 src/eval.py

echo "----------------------------"
echo "Done. All logs, tables, and plots are saved in results/ and logs/"
echo "Training table: results/training_table.csv"
echo "Evaluation table: results/evaluation_table.csv"
echo "Plots: results/reward_curve.png, results/ndcg_curve.png, results/hit_curve.png"
echo "Metrics summary: results/metrics.json"
echo "----------------------------"