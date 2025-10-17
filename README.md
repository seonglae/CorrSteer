# CorrSteer: Correlation-Guided Feature Selection for Language Model Steering

This repository contains the implementation of **CorrSteer**, a correlation-based method for steering language models using Sparse Autoencoders (SAEs). CorrSteer identifies task-relevant SAE features through Pearson correlation with task outcomes and applies targeted steering interventions during inference.


## Key Features

This project implements reinforcement learning techniques for steering language models using sparse autoencoders.

## Setup

1. Install Astral UV:
```bash
pip install uv
```

2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate
# On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
uv pip install -e .
```

## Usage

### Training
```bash
# MMLU with raw activations
python train.py train --model=gemma2b --task=mmlu --layer=global --raw

# MMLU with mean pooling
python train.py train --model=gemma2b --task=mmlu --layer=global --pool=mean

# BBQ disambiguation with mask
python train.py train --model=gemma2b --task=bbq --layer=global --mask=all --filter_value=disambig

# HarmBench with raw activations
python train.py train --model=gemma2b --task=harmbench --layer=global --raw

# SimpleQA with mean pooling
python train.py train --model=gemma2b --task=simpleqa --layer=global --pool=mean
```

### Evaluation
```bash
# Baseline evaluation
python eval.py baseline --task=mmlu
```

## Project Structure
- `corrsteer/`: Main package
  - `config.py`: Configuration files
  - `dataset.py`: Dataset loading and processing
  - `model.py`: Model and SAE integration
  - `steer.py`: Steering hooks
  - `utils.py`: Utility functions
- `train.py`: Training script
- `eval.py`: Evaluation script
- `sft.py`: Supervised fine-tuning script
