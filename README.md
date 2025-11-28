# CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Demo-yellow)](https://huggingface.co/spaces/seonglae/CorrSteer)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Slides](https://img.shields.io/badge/PPT-Slidev-skyblue.svg)](https://corrsteer.vercel.app/)

Implementation of CorrSteer, a generation-time steering method using correlated Sparse Autoencoder (SAE) features.

## Key Features

- **Correlation-based feature selection** from generation-time activations
- **Streaming computation** with O(1) memory complexity
- **Multi-layer strategies** (CorrSteer-S/A/P)
- **Side Effect Ratio (SER)** for measuring unintended changes

## Setup

Install [Astral UV](https://github.com/astral-sh/uv):
```bash
pip install uv
```

Create virtual environment and install:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

## Usage

### Training

```bash
# MMLU with SAE features
python train.py train --model=gemma2b --task=mmlu --layer=global --eval

# MMLU with raw activations
python train.py train --model=gemma2b --task=mmlu --layer=global --raw --eval

# MMLU with mean pooling
python train.py train --model=gemma2b --task=mmlu --layer=global --pool=mean --eval

# BBQ disambiguation
python train.py train --model=gemma2b --task=bbq --layer=global --mask=all --filter_value=disambig --eval

# HarmBench with raw activations
python train.py train --model=gemma2b --task=harmbench --layer=global --raw --eval

# SimpleQA with mean pooling
python train.py train --model=gemma2b --task=simpleqa --layer=global --pool=mean --eval

# GSM8K with mean pooling for both correlation and steering
python train.py train --model=gemma2b --task=gsm8k --layer=foreach --pool=mean --steer_pool=mean --eval
```

### Evaluation

```bash
# Baseline evaluation
python eval.py baseline --task=mmlu
```

### Multi-Layer Strategies

```bash
# CorrSteer-S: Single best feature globally
python train.py train --task=mmlu --layer=global --eval

# CorrSteer-A: Top feature from each layer
python train.py train --task=mmlu --layer=foreach --eval

# CorrSteer-P: Validation-based pruning
python train.py train --task=mmlu --layer=foreach --validate --eval
```

## Project Structure

```
corrsteer/
├── config.py       # Dataset and model configurations
├── dataset.py      # Data loading and processing
├── model.py        # Model and SAE integration
├── steer.py        # Steering hooks for inference
└── utils.py        # Utility functions

train.py            # Training with streaming correlation
eval.py             # Evaluation with SER computation
sft.py              # Supervised fine-tuning
```


## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- [SAE Lens](https://github.com/jbloomAus/SAELens) for SAE implementation
- [Gemma Scope](https://huggingface.co/google/gemma-scope-2b-pt-res) for pretrained SAEs
- [HuggingFace](https://huggingface.co/) for model hosting
