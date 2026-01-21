#!/bin/bash
# CorrSteer ICML Experiments - Sequential Run
# Run on DGX 9: nohup bash run_all_experiments.sh > experiments.log 2>&1 &

set -e
cd ~/corrsteer
source .venv/bin/activate

echo "=== Starting CorrSteer Experiments $(date) ==="

# 1. Selection Bias Control
echo "=== 1. Selection Bias Control ==="
echo "[1.1] Shuffle labels..."
python train.py train --task=mmlu --layer=global --shuffle_labels --eval

echo "[1.2] Random features..."
python train.py train --task=mmlu --layer=global --random_features --eval

# 2. Feature Stability (multiple seeds)
echo "=== 2. Feature Stability ==="
for seed in 1 2 3 4 5; do
  echo "[2.$seed] Seed $seed..."
  python train.py train --task=mmlu --layer=global --seed=$seed --eval
done

# 3. Robustness - Scale
echo "=== 3. Robustness - Scale ==="
for scale in 0.5 1.0 2.0 3.0; do
  echo "[3] Scale $scale..."
  python train.py train --task=mmlu --layer=global --scale=$scale --eval
done

# 4. Robustness - Topk
echo "=== 4. Robustness - Topk ==="
for k in 1 2 3 5 10; do
  echo "[4] Topk $k..."
  python train.py train --task=mmlu --layer=foreach --topk=$k --eval
done

# 5. Safety Evaluation
echo "=== 5. Safety Evaluation ==="
echo "[5.1] XSTest..."
python eval.py safety_eval --task=xstest --model=gemma2b --export_samples

echo "[5.2] HarmBench..."
python eval.py safety_eval --task=harmbench --model=gemma2b --export_samples

# 6. Format Evaluation
echo "=== 6. Format Evaluation ==="
echo "[6.1] MMLU baseline..."
python eval.py format_eval --task=mmlu --model=gemma2b

echo "[6.2] MMLU with select_token..."
python eval.py format_eval --task=mmlu --model=gemma2b --select_token

echo "=== All Experiments Complete $(date) ==="
