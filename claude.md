# Think Tokens Project Documentation

## Overview
This project explores the concept of "thinking tokens" - special tokens added to language models that allow them to perform intermediate reasoning steps without being constrained by natural language. The project implements various training approaches for models that can use these thinking tokens to improve their performance on tasks like addition.

## Core Concept
The key idea is to expand a language model's vocabulary with additional tokens that:
- Have no textual representation or meaning
- Can be used by the model for intermediate computation
- Are trained via reinforcement learning rather than supervised learning
- Allow the model to externalize intermediate results before making predictions

## Project Structure

### Main Training Scripts

#### `add_normal.py`
- Implements a standard GPT2 model for addition tasks
- Uses a simple tokenizer for digits 0-9, '+', and '='
- Trains the model with supervised learning on addition problems
- Includes benchmarking functionality to evaluate model accuracy

#### `add_think.py`
- Extends the addition task to use thinking tokens
- Implements reinforcement learning training where:
  - Model generates rollouts with thinking tokens
  - Rewards are based on correct answer prediction
  - Uses REINFORCE-style gradient updates with entropy regularization
- Includes specialized benchmarking for thinking models

#### `normal.py`
- Basic GPT2 implementation for standard language modeling
- Trains on the Fineweb-edu dataset
- Serves as a baseline and can be loaded into thinking models

#### `rollout_think.py`
- Implements thinking tokens for general language modeling
- Uses a reference model (GPT2-XL or Llama) to evaluate output quality
- Trains with RL where rewards come from reference model's assessment
- Includes repetition penalty and entropy regularization

#### `supervised_rollout_think.py`
- Alternative training approach using supervised rollouts
- For each position in a sequence, generates thinking tokens then predicts next real token
- More principled but less parallelizable than other approaches
- Includes special end_thought token to terminate thinking sequences

#### `srt2.py`
- Simplified version of supervised_rollout_think.py
- Focuses on next token prediction after thinking sequences
- Used for debugging and testing core concepts

### Utility Files

#### `utils.py`
- Configuration classes: `ModelConfig`, `ThinkingModelConfig`, `TrainingConfig`
- Helper functions for sampling, tokenization, and model loading
- Visualization utilities using Plotly
- Color constants for terminal output
- Function to convert normal models to thinking models

### Data and Documentation

#### `datasets/`
- Contains pickle files with preprocessed datasets
- `additions_10K_100K.pkl`: Addition problems for training

#### `notes/`
- `notes.md`: Detailed explanation of thinking tokens concept
- `notes.pdf`: PDF version of notes
- `image.png`: Visualization of model output with thinking tokens

#### `README.md`
- Project TODO list and experimental ideas
- Discusses fundamental challenges like credit assignment for thinking tokens
- Proposes solutions using reference models or multiple rollouts (GRPO-style)

## Key Concepts

### Thinking Tokens
- Extra vocabulary tokens added beyond normal text tokens
- Represented as `<think{id}>` in visualizations
- Model learns their meaning through RL based on downstream performance
- Can be used for arbitrary intermediate computation

### Training Approaches
1. **Rollout-based RL**: Generate full sequences, evaluate with reference model
2. **Supervised Rollout**: Generate thinking tokens for each position, maximize next token prediction
3. **Addition-specific**: Simplified task to test thinking token effectiveness

### Challenges Addressed
- **Credit Assignment**: How to determine which thinking tokens were useful
- **Exploration**: Large action space with new tokens
- **Baseline Selection**: Using reference models vs. mean performance

## Technical Details

### Model Architecture
- Based on GPT2 with expanded vocabulary
- Separate embedding dimensions for normal and thinking tokens
- Special tokens like `<end_thought>` to control thinking sequences

### Training Features
- Entropy regularization to encourage exploration
- Discount factors for temporal credit assignment
- Various sampling strategies (temperature, top-k, top-p)
- Gradient accumulation and batch processing

### Visualization
- Color-coded output: blue (normal tokens), cyan (thinking tokens), magenta (special tokens)
- Wandb integration for experiment tracking
- Plotly-based visualization utilities

## Future Directions
The README suggests several experimental directions:
- Continuous thinking tokens (Coconut-style)
- Pretrained model expansion with random embeddings
- Synthetic dataset experiments
- Alternative RL algorithms beyond REINFORCE