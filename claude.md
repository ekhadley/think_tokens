# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project exploring "thinking tokens" - a mechanism for transformer models to produce internal reasoning tokens before generating final outputs. The project implements and compares different training approaches for models that can use thinking tokens to improve their reasoning capabilities on tasks like arithmetic and language modeling.

## Environment Setup

```bash
# Dependencies managed via pipenv
pipenv install
pipenv shell

# All scripts can be run directly with python
python add_normal.py
python add_think.py
python add_think2.py
```

## Key Architecture Components

### Model Types
- **Normal GPT2** (`normal.py`): Standard transformer for baseline comparisons
- **GPT2Thinking**: Extended transformer that can generate thinking tokens in addition to normal vocabulary tokens

### Core Training Approaches
1. **Normal Training** (`add_normal.py`): Standard supervised learning on addition tasks
2. **Thinking Token Training** (`add_think.py`, `add_think2.py`): RL approach where models learn to use thinking tokens to improve prediction accuracy
3. **Rollout Training** (`rollout_think.py`, `supervised_rollout_think.py`): Advanced RL training with reference models and discounted rewards

### Configuration Classes
- `ModelConfig`: Standard transformer configuration 
- `ThinkingModelConfig`: Extended config with separate vocabularies for normal tokens (`d_normal_vocab`) and thinking tokens (`d_thought_vocab`)
- `TrainingConfig`: Training hyperparameters including RL-specific settings like `gamma` for reward discounting

## Running Experiments

### Addition Task Experiments
```bash
# Normal model baseline
python add_normal.py

# Thinking token models with RL training
python add_think.py
python add_think2.py
```

### Language Model Training
```bash
# Normal transformer baseline
python normal.py

# Thinking token transformer with rollout training
python rollout_think.py

# Supervised rollout approach
python supervised_rollout_think.py
```

## Key Utilities

### Dataset Creation
- `SimpleTokenizer`: Custom tokenizer for arithmetic tasks 
- `makeAdditionDataset()`: Generates addition problem datasets with configurable difficulty
- Dataset attributes store metadata like `input_max`, `question_len` instead of per-row fields for efficiency

### Model Loading/Saving
- `loadModel()`: Load saved model checkpoints from `saves/` directory
- `LoadNormalModelAsThinking()`: Convert normal models to thinking token models by expanding vocabularies

### Evaluation
- `benchmark_addition()`: Evaluate normal model accuracy on addition tasks
- `benchmark_addition_think()`: Specialized benchmark for thinking token models with rollout generation

## Thinking Token Implementation Details

### Token Mechanics
- Thinking tokens have IDs >= `d_normal_vocab` 
- Special `end_thought` token (highest ID) signals end of reasoning phase
- Models alternate between thinking phase (internal reasoning) and prediction phase (final output)

### Training Rewards
- **Prediction Reward**: Logprob of correct answer token - primary signal
- **Think Reward**: Weighted by normalized prediction performance across rollouts
- **Entropy Reward**: Encourages exploration in thinking token space to prevent collapse

### Color-coded Visualization
- Blue: Normal vocabulary tokens
- Cyan: Thinking tokens (`<think{id}>`)
- Magenta: Special tokens like `<end_thought>`

## Current Research Challenges

Based on implementation, key open problems include:
- Preventing models from ignoring thinking tokens during training
- Proper credit assignment for which thinking tokens were actually useful
- Balancing exploration vs exploitation in the large thinking token action space
- Using reference models vs. mean performance for reward baselines
- Optimizing parallel rollout generation for efficiency