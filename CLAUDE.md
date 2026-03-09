# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Self-Attention PPO (Proximal Policy Optimization) implementation in PyTorch, inspired by [Attentive Multi-Task Deep Reinforcement Learning](https://arxiv.org/pdf/1904.03367.pdf). Trains an agent to play Atari games (default: ALE/Pong-v5) using PPO with a self-attention mechanism inserted into the CNN feature extractor.

## Commands

```bash
# Setup (requires uv: https://docs.astral.sh/uv/)
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt

# Train
python main.py

# Tensorboard
tensorboard --logdir runs

# Lint & format (uses ruff)
ruff format .
ruff check .
```

## Architecture

**Entry point:** `main.py` — configures hyperparameters and launches PPO training with TensorBoard logging to `runs/`.

**`src/model.py`** — Neural network with shared CNN backbone for actor-critic:
- `ActorCriticNet`: Conv layers → self-attention after first conv → linear layers → separate actor (policy logits) and critic (value) heads. Forward pass returns (logits, values, sampled actions).
- `MultiHeadAttention` / `ScaledDotProductAttention`: Spatial self-attention using 1x1 convolutions for Q/K/V projections with residual connection. Applied to feature maps between conv layers.

**`src/ppo.py`** — PPO algorithm (`PPO` class):
- `_rollout()`: Collects trajectories from parallel environments, computes GAE advantages and returns.
- `_update()`: Runs multiple PPO epochs with minibatch updates (clipped surrogate objective + value loss + entropy bonus). Uses deep copy of model for old policy ratios.
- `_test_env()`: Evaluates current policy on a single environment.
- Saves best model checkpoint to `model.pt`.
- Uses `gymnasium.vector.AsyncVectorEnv` for parallel environment execution.

**`src/env_utils.py`** — Environment factory using gymnasium built-ins: `AtariPreprocessing` (frame skip, grayscale, resize to 84x84, scale to [0,1]) → `FrameStackObservation` (4 frames).

## Key Details

- Observation shape after wrappers: `(4, 84, 84)` — 4 stacked grayscale frames
- No test suite exists in this project
- Uses gymnasium API — `step()` returns 5 values (obs, reward, terminated, truncated, info)
- Device auto-detected: CUDA → MPS → CPU
