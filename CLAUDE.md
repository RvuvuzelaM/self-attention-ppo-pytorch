# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Self-Attention PPO (Proximal Policy Optimization) implementation in PyTorch, inspired by [Attentive Multi-Task Deep Reinforcement Learning](https://arxiv.org/pdf/1904.03367.pdf). Trains an agent to play Atari games (default: PongNoFrameskip-v4) using PPO with a self-attention mechanism inserted into the CNN feature extractor.

## Commands

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train
python main.py

# Tensorboard
tensorboard --logdir runs

# Lint (uses black formatter)
black .
```

## Architecture

**Entry point:** `main.py` — configures hyperparameters and launches PPO training with TensorBoard logging to `runs/`.

**`src/model.py`** — Neural network with shared CNN backbone for actor-critic:
- `ActorCriticNet`: Conv layers → self-attention after first conv → linear layers → separate actor (policy logits) and critic (value) heads. Forward pass returns (logits, values, sampled actions).
- `MultiHeadAttention` / `ScaledDotProductAttention`: Spatial self-attention using 1x1 convolutions for Q/K/V projections with residual connection. Applied to feature maps between conv layers.

**`src/ppo.py`** — PPO algorithm (`PPO` class):
- `_rollout()`: Collects trajectories from parallel environments, computes GAE advantages and returns.
- `_update()`: Runs multiple PPO epochs with minibatch updates (clipped surrogate objective + value loss + entropy bonus). Uses deep copy of model for old policy ratios.
- `_test_env()`: Evaluates current policy on a single environment. Hardcoded to CUDA.
- Saves best model checkpoint to `model.pt`.

**`src/multiprocess.py`** — `SubprocVecEnv`: Runs multiple gym environments in parallel using `multiprocessing.Pipe`/`Process` with cloudpickle serialization.

**`src/wrappers.py`** — Atari preprocessing wrapper chain applied in order: `MaxAndSkipEnv` → `FireResetEnv` → `ProcessFrame84` (resize to 84x84 grayscale) → `ScaledFloatFrame` (normalize to [0,1]) → `ImageToPyTorch` (channels-first) → `BufferWrapper` (frame stacking, 4 frames).

## Key Details

- Observation shape after wrappers: `(4, 84, 84)` — 4 stacked grayscale frames
- No test suite exists in this project
- `_test_env()` in `src/ppo.py:204` hardcodes `"cuda"` device instead of using `self.device`
- Uses older gym API (v0.17.2) — `step()` returns 4 values, not 5
