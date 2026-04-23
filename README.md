# PEFT Performance Trade-offs Under Limited Data

> Investigating the Performance and Resource Trade-offs of Different PEFT Methods Under Limited Sampling

**Course:** DSAI 5207 Modern Deep Learning | **Group:** 6

**Members:** Hu Zhitao, Geng Xuekang, Ma Yuhong, Zhang Xingdong, Cheng Yuk Pui

---

## Overview

This project empirically investigates how Parameter-Efficient Fine-Tuning (PEFT) methods perform under extreme data scarcity. We compare four adaptation strategies — **Zero-shot**, **Full Fine-tuning**, **LoRA**, and **BitFit** — on RoBERTa-base for 4-class emotion classification across training set sizes of 200, 500, 1000, and 2000 examples, evaluated on four different hardware platforms.

### Key Findings

| Metric | Full Fine-tuning | LoRA | BitFit |
|---|---|---|---|
| Trainable params | 125M (100%) | ~0.3% | ~0.08% (~97k) |
| Accuracy at N=200 | 39.3% (≈ zero-shot) | **59.96%** | **52.71%** |
| Accuracy at N=2000 | 81.17% | **81.44%** | 79.71% |
| VRAM at N=2000 (RTX 3060) | 5.3 GB | 3.2 GB (↓40%) | 2.8 GB (↓47%) |
| Training time (RTX 3060) | ~77s | ~77s | **57s (↓26%)** |
| M4 VRAM (LoRA) | 2.34 GB | **0.53 GB (↓77%)** | 0.53 GB (↓77%) |

- **LoRA** is the most robust method across all sample sizes.
- **BitFit** provides the best efficiency trade-off — fastest, lowest memory — with acceptable accuracy loss.
- **Two-stage training** (classifier warm-up → joint fine-tuning) is essential; without it, PEFT methods fail to beat zero-shot at N=200.

---

## Project Structure

```
.
├── scripts/
│   ├── download_data.py      # Download TweetEval-emotion and extract training subsets
│   ├── train_lora.py         # LoRA fine-tuning
│   ├── train_bitfit.py       # BitFit fine-tuning
│   ├── train_full.py         # Full fine-tuning
│   ├── train_zeroshot.py     # Zero-shot inference (no training)
│   ├── run_all.py            # Run all experiments in batch
│   └── summarize_results.py  # Aggregate and compare results
├── src/
│   ├── config.py             # Shared hyperparameters and model/dataset config
│   ├── data_loader.py        # Dataset loading, tokenization, DataLoader building
│   └── utils.py              # Device detection, VRAM monitoring, logging, metrics
├── data/
│   ├── raw/                  # Full TweetEval-emotion dataset (train/val/test splits)
│   └── subsets/              # Sampled training subsets: {200, 500, 1000, 2000}
├── results/                  # JSON result files per method and sample size
├── logs/                     # Training logs
├── ppt.pdf                   # Presentation slides
├── report.pdf                # Written report
├── speech.txt                # Presentation script
└── README.md
```

---

## Setup

### Dependencies

```bash
pip install torch transformers peft datasets scikit-learn pandas tqdm numpy
```

### Data Preparation

```bash
# Download TweetEval-emotion and create stratified training subsets
python scripts/download_data.py

# Verify existing subsets
python scripts/download_data.py --verify-only
```

This creates `data/raw/` (full dataset) and `data/subsets/{200,500,1000,2000}/` with fixed seed=42 for reproducibility.

---

## Usage

All scripts support `--help` for available arguments. Common usage:

```bash
# Run a single experiment
python scripts/train_lora.py --sample-size 500 --batch-size 32
python scripts/train_bitfit.py --sample-size 200
python scripts/train_full.py --sample-size 2000
python scripts/train_zeroshot.py

# Run all methods × all sample sizes
python scripts/run_all.py

# Run only specific methods or sizes
python scripts/run_all.py --methods lora bitfit --sample-sizes 200 1000

# Summarize results
python scripts/summarize_results.py --report
python scripts/summarize_results.py --output results/comparison.csv
```

### Key Hyperparameters

| Parameter | Value |
|---|---|
| Base model | `roberta-base` (125M params) |
| Dataset | TweetEval `emotion` (anger/joy/optimism/sadness) |
| Batch size | 32 (RTX); 16 on low-VRAM GPUs |
| Optimizer | AdamW, weight decay 0.01 |
| Random seed | 42 |

**LoRA:** rank=8, alpha=32, dropout=0.1, applied to Q/V matrices.

**Two-stage training:**

| Stage | LoRA | BitFit |
|---|---|---|
| Warm-up epochs (head only, LR=1e-3) | 3 | 2 |
| Joint epochs (head LR / adapter LR) | 5 (5e-4 / 5e-5) | 5 (5e-4 / 5e-4) |

---

## Hardware Tested

| Platform | Memory | Notes |
|---|---|---|
| NVIDIA RTX 2060 | 6 GB | Desktop GPU |
| NVIDIA RTX 3060 Laptop | 6 GB | Laptop GPU |
| NVIDIA RTX 4070 | 12 GB | High-end desktop GPU |
| Apple MacBook Air M4 | 16 GB unified | MPS backend |

The `run_all.py` script auto-detects available hardware and adjusts batch sizes accordingly.

---

## Results

### Test Accuracy (RTX 3060 Laptop)

| Method | N=200 | N=500 | N=1000 | N=2000 |
|---|---|---|---|---|
| Zero-shot | 39.30% | 39.30% | 39.30% | 39.30% |
| Full Fine-tuning | 39.27% | 61.17% | 73.87% | 81.17% |
| **LoRA** | **59.96%** | **70.60%** | **76.63%** | **81.44%** |
| BitFit | 52.71% | 68.57% | 75.09% | 79.71% |

### Cross-Device Consistency (N=2000)

| Device | LoRA | BitFit |
|---|---|---|
| RTX 2060 | 81.4% ±0.4% | 79.7% ±0.5% |
| RTX 3060 Laptop | 81.4% ±0.4% | 79.7% ±0.5% |
| RTX 4070 | 81.4% ±0.4% | 79.7% ±0.5% |
| MacBook Air M4 | 81.4% ±0.4% | 79.7% ±0.5% |

Performance is stable across GPU architectures and precision formats.

---

## Citation

If you use this code or report in your work, please cite:

```
@misc{peft2026project,
  title = {Investigating the Performance and Resource Trade-offs of Different PEFT Methods Under Limited Sampling},
  author = {Hu Zhitao and Geng Xuekang and Ma Yuhong and Zhang Xingdong and Cheng Yuk Pui},
  school = {The Hong Kong Polytechnic University, DSAI 5207 Modern Deep Learning},
  year = {2026}
}
```
