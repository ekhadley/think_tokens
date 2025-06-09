# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase exploring "thinking tokens" - training language models to use intermediate reasoning steps before producing final outputs. The primary task is modular addition, serving as a controlled environment to study different training approaches for thinking token models.

## Environment Setup

```bash
# Use python directly - assumes pipenv environment is active
python script_name.py

# Example experiments
python add_normal.py        # Baseline without thinking
python add_think_fixed_blind_super.py  # Current best approach
```

## Architecture: Two-Vocabulary System

### Core Components
- **Standard GPT-2** (`normal.py`): Baseline transformer
- **GPT2Thinking** (`supervised_rollout_think.py`): Extended model with dual vocabulary
  - Normal tokens (0 to `d_normal_vocab-1`): Regular text/numbers
  - Thinking tokens (`d_normal_vocab` to `d_vocab_total-2`): Internal reasoning
  - Special `end_thought` token: Transition from thinking to answering

### Configuration Classes
- `ModelConfig`: Standard transformer settings
- `ThinkingModelConfig`: Adds `d_thought_vocab` for thinking token count
- `TrainingConfig`: RL hyperparameters including exploration epsilon

## Experiment Progression (by complexity)

### Baseline
- `add_normal.py`: Standard supervised learning on addition

### Thinking Token Variants
- `add_think.py`: Basic variable-length thinking sequences
- `add_think_fixed.py`: Fixed-length thinking sequences (~20x faster)
- `add_think_fixed_blind.py`: Model only sees thinking tokens during prediction
- `add_think_fixed_blind_super.py`: Uses manually crafted correct thinking sequences
- `add_think_fixed_blind_super_search.py`: Adds exhaustive search over thinking space

### Current Best Practice
The "blind + supervised + clean rewards" approach works for max=100 problems. The naming convention isolates specific training difficulties:
- **fixed**: Predetermined thinking sequence length
- **blind**: Force reliance on thinking tokens by hiding original question
- **super**: Provide correct thinking sequences for supervision
- **search**: Exhaustive exploration of thinking possibilities

## Key Training Patterns

### Reinforcement Learning Setup
- Group-based training: Multiple rollouts per question for variance estimation
- Epsilon-greedy exploration: Balance random vs model-generated tokens  
- Reward structure: Mean-centered across groups, with entropy bonus
- Credit assignment: Think token rewards weighted by prediction accuracy

### Dataset Generation
- `makeAdditionDataset()`: Creates modular addition problems
- Configurable difficulty via `input_max` parameter
- Dataset stores metadata globally rather than per-sample for efficiency

### Evaluation Functions
- `benchmark_addition_think_*()`: Specialized benchmarks for different thinking approaches
- Color-coded output: Blue (normal), Cyan (thinking), Magenta (special tokens)

## Utilities (`utils.py`)

### Model Management
- `loadModel()`: Load checkpoints from `saves/` directory
- `LoadNormalModelAsThinking()`: Convert standard to thinking models
- Plotting functions with custom `imshow()` for tensor visualization

### Key Dependencies
- PyTorch + Transformers for model architecture
- eindex for efficient tensor operations
- wandb for experiment tracking
- Custom tokenization for arithmetic tasks

## Research Insights

### Training Challenges
1. **Bootstrap Problem**: Must learn to produce and use thinking tokens simultaneously
2. **Sparse Rewards**: Final answer accuracy provides limited learning signal
3. **Exploration**: Large thinking token action space requires careful exploration

### Current Status
- Works for small problems (max=100) with supervised approach
- Scaling to larger problems (max=1000) reveals fundamental training difficulties
- Next research direction: Improve reward structure beyond "clean" binary rewards