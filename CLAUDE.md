# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the Hierarchical Reasoning Model (HRM) - a 27M parameter recurrent neural architecture for complex reasoning tasks. It achieves state-of-the-art results on ARC-AGI, Sudoku, and maze-solving with only 1000 training samples.

## Setup / Installation

To set up the project and install dependencies:

```bash
# First, clone the adam-atan2-pytorch optimizer into the project root.
# It is a separate repository and not included with the initial clone of this project.
git clone https://github.com/lucidrains/adam-atan2-pytorch.git adam-atan2-pytorch

# Install all dependencies, including adam-atan2-pytorch from the local directory
pip install -r requirements.txt
```

- The `adam-atan2-pytorch` optimizer is installed via `-e ./adam-atan2-pytorch` in `requirements.txt`.

- If the optimizer is available on PyPI, you may also install it directly (though local cloning is recommended for specific versions):

```bash
pip install adam-atan2-pytorch
```

### Setup with uv and Python 3.11

To set up the project using uv for faster dependency management with Python 3.11:

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the adam-atan2-pytorch optimizer
git clone https://github.com/lucidrains/adam-atan2-pytorch.git adam-atan2-pytorch

# Create virtual environment with Python 3.11
uv venv --python 3.11

# Activate the environment
source .venv/bin/activate

# Install dependencies from pyproject.toml
uv sync
```

This will install all required dependencies, including the local editable adam-atan2-pytorch.

## Common Commands

### Dataset Building
```bash
# ARC-1 dataset (960 examples)
python dataset/build_arc_dataset.py

# ARC-2 dataset (1120 examples)  
python dataset/build_arc_dataset.py --dataset-dirs dataset/raw-data/ARC-AGI-2/data --output-dir data/arc-2-aug-1000

# Sudoku dataset (1000 examples)
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000

# Maze dataset (1000 examples)
python dataset/build_maze_dataset.py
```

### Training
```bash
# Single GPU quick demo (Sudoku)
OMP_NUM_THREADS=8 python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 global_batch_size=384 lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0

# Multi-GPU training (8 GPUs)
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py

# Override config parameters
python pretrain.py data_path=data/arc-2-aug-1000 lr=1e-4 epochs=20000
```

### Evaluation
```bash
# Evaluate a checkpoint
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 evaluate.py checkpoint=<CHECKPOINT_PATH>
```

## Architecture

The codebase follows a hierarchical reasoning design with two processing levels:

### Core Model Structure
- **Entry Point**: `models/hrm/hrm_act_v1.py:HierarchicalReasoningModel_ACTV1`
- **H-level**: Abstract planning module (slow processing)
- **L-level**: Detailed computation module (rapid processing)  
- **ACT Mechanism**: Adaptive computation time with Q-learning for halting decisions

### Key Components
1. **Training Pipeline** (`pretrain.py`):
   - Distributed training with PyTorch DDP
   - W&B integration for experiment tracking
   - Dual optimizer setup (AdamATan2 for model, SGD for puzzle embeddings)

2. **Model Forward Pass**:
   - Input → Embeddings → Recurrent H/L cycles → ACT decisions → Output
   - Gradient-free computation except last step (for efficiency)
   - Carry state pattern maintains hidden states across cycles

3. **Loss Computation** (`models/hrm/act_loss_head.py`):
   - Language modeling loss (stablemax or softmax)
   - Q-learning losses for halt/continue decisions
   - Tracks accuracy, exact accuracy, and computation steps

4. **Dataset Handling** (`puzzle_dataset.py`):
   - Memory-mapped numpy arrays for efficiency
   - Supports ARC, Sudoku, and maze tasks
   - Handles puzzle-specific embeddings

### Configuration System
- Uses Hydra with configs in `/config/`
- Main config: `cfg_pretrain.yaml`
- Architecture config: `arch/hrm_v1.yaml`
- Override via CLI: `python pretrain.py key=value`

### Important Implementation Details
- Custom CUDA extensions in `models/sparse_emb/` for efficiency
- FlashAttention integration (v2 for Ampere, v3 for Hopper GPUs)
- Checkpoint format compatible with torch.compile
- Extensive use of torch.inference_mode() for evaluation
- Careful distributed training synchronization

## Development Notes

- No test suite exists - verify changes by running training/evaluation
- No linting configured - follow existing code style
- Dataset visualization available via `puzzle_visualizer.html`
- ARC evaluation notebook: `arc_eval.ipynb`
- Requires CUDA 12.6+ and appropriate GPU support
- Git submodules must be initialized: `git submodule update --init --recursive`