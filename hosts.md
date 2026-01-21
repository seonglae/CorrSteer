# CorrSteer Distributed Training Hosts

## Host Registry

| Name | Hostname | IP | GPU | SSH | Path |
|------|----------|----|----|-----|------|
| DGX 7 | spark-06aa | 100.70.143.80 | DGX GB10 | `seonglae@100.70.143.80` | `~/corrsteer` |
| DGX 9 | spark-9ea3 | 100.99.92.18 | DGX GB10 | `seonglae@100.99.92.18` | `~/corrsteer` |
| Windows | home-seonglae | 100.113.68.13 | RTX 3080 (WSL) | `Seonglae@100.113.68.13` | `/mnt/c/Home/Projects/AI/corrsteer` |
| Mac | SEONGLAE-HOLISTIC | 100.117.200.18 | M2 Pro (MPS) | `seonglaecho@100.117.200.18` | `~/Projects/corrsteer-code` |

## DGX 7 (100.70.143.80, DGX GB10)

```bash
# Connect
ssh seonglae@100.70.143.80

# Run experiments
cd ~/corrsteer
source .venv/bin/activate

# Selection bias control
python train.py train --task=mmlu --layer=global --shuffle_labels --eval
python train.py train --task=mmlu --layer=global --random_features --eval

# Robustness (scale)
for scale in 0.5 1.0 2.0 3.0; do
  python train.py train --task=mmlu --layer=global --scale=$scale --eval
done

# Safety evaluation
python eval.py safety_eval --task=xstest --model=gemma2b --export_samples

# Check status
tail -f train.log
ps aux | grep train.py
nvidia-smi
```

## DGX 9 (100.99.92.18, DGX GB10)

```bash
# Connect
ssh seonglae@100.99.92.18

# Run experiments
cd ~/corrsteer
source .venv/bin/activate

# Feature stability (multiple seeds)
for seed in 1 2 3 4 5; do
  python train.py train --task=mmlu --layer=global --seed=$seed --eval
done

# Topk experiments
for k in 1 2 3 5 10; do
  python train.py train --task=mmlu --layer=foreach --topk=$k --eval
done

# Format evaluation
python eval.py format_eval --task=mmlu --model=gemma2b
python eval.py format_eval --task=mmlu --model=gemma2b --select_token

# Check status
tail -f train.log
ps aux | grep train.py
nvidia-smi
```

## Windows (100.113.68.13, RTX 3080)

**Important**: All Linux commands must go through WSL (`-d Ubuntu`).

```bash
# Connect
ssh Seonglae@100.113.68.13

# Run training (via WSL -d Ubuntu)
wsl -d Ubuntu -e bash -c "cd /mnt/c/Home/Projects/AI/corrsteer && source .venv/bin/activate && python train.py train --task=mmlu --layer=global --eval"

# Check GPU (PowerShell, not WSL)
nvidia-smi

# Check process (via WSL)
wsl -d Ubuntu -e bash -c "ps aux | grep python"
```

## Mac (100.117.200.18, M2 Pro)

```bash
# Connect
ssh seonglaecho@100.117.200.18

# Update code
cd ~/Projects/corrsteer-code && git pull

# Run experiments (MPS)
source .venv/bin/activate
python train.py train --task=mmlu --layer=global --eval

# Note: MPS has no nvidia-smi equivalent
```

## Experiment Matrix

| Experiment | Host | Command |
|-----------|------|---------|
| Shuffle labels | DGX 7 | `python train.py train --task=mmlu --layer=global --shuffle_labels --eval` |
| Random features | DGX 7 | `python train.py train --task=mmlu --layer=global --random_features --eval` |
| Scale sweep | DGX 7 | `for scale in 0.5 1.0 2.0 3.0; do python train.py train --scale=$scale --eval; done` |
| Seed stability | DGX 9 | `for seed in 1 2 3 4 5; do python train.py train --seed=$seed --eval; done` |
| Topk sweep | DGX 9 | `for k in 1 2 3 5 10; do python train.py train --topk=$k --eval; done` |
| Safety eval | DGX 7 | `python eval.py safety_eval --task=xstest --export_samples` |
| Format eval | DGX 9 | `python eval.py format_eval --task=mmlu` |

## Sync Rules (CRITICAL)

**Code: Local -> Remote hosts (push)**
**Output: Remote hosts -> Local output/ (pull, keep newer files)**

NEVER create separate folders like output_dgx or output_server9. Always sync directly to output/.

```bash
# Push code to DGX 7
rsync -avz --exclude='.venv' --exclude='output' --exclude='*.pyc' ./ seonglae@100.70.143.80:~/corrsteer/

# Push code to DGX 9
rsync -avz --exclude='.venv' --exclude='output' --exclude='*.pyc' ./ seonglae@100.99.92.18:~/corrsteer/

# Pull output from DGX 7 to local (update only, keep newer)
rsync -avzu seonglae@100.70.143.80:~/corrsteer/output/ ./output/

# Pull output from DGX 9 to local (update only, keep newer)
rsync -avzu seonglae@100.99.92.18:~/corrsteer/output/ ./output/
```

## Quick Reference

| Action | Command |
|--------|---------|
| Update code | `git pull` |
| Check logs | `tail -f train.log` |
| Kill process | `pkill -f train.py` |
| GPU usage | `nvidia-smi` (Linux/Windows) |
| **Sync output** | `rsync -avzu seonglae@HOST:~/corrsteer/output/ ./output/` |

## New Host Setup

```bash
# 1. Check required tools
which rsync git uv || echo "Missing tools - install with apt/brew"

# 2. Set GitHub token (for private repos)
export GITHUB_TOKEN="ghp_xxx"
git config --global credential.helper store
echo "https://$GITHUB_TOKEN@github.com" > ~/.git-credentials

# 3. Clone repo
git clone https://github.com/seonglae/corrsteer.git ~/corrsteer
cd ~/corrsteer

# 4. Setup Python environment
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,cuda]"  # or without cuda for CPU/MPS

# 5. Verify setup
python train.py train --task=mmlu --layer=global --limit=10 --eval
```
