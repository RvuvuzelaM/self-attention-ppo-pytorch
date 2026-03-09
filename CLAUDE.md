# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PPO (Proximal Policy Optimization) implementation in PyTorch, based on [CleanRL's ppo_atari.py](https://github.com/vwxyzjn/cleanrl). Trains an agent to play Atari games (default: ALE/Pong-v5) using PPO with a standard CNN feature extractor.

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
- `Agent`: Standard CNN (3 conv layers → flatten → linear 512) with orthogonal init. Separate actor and critic heads. API: `get_value(x)` and `get_action_and_value(x, action=None)`. Input is uint8, divided by 255 inside the model.

**`src/ppo.py`** — PPO algorithm (`PPO` class):
- `train()`: Main training loop following CleanRL structure — rollout collection with pre-allocated buffers, GAE computation, flattened minibatch PPO updates with per-minibatch advantage normalization and value clipping using stored rollout values.
- `_run_episode()`: Evaluates current policy on a single environment.
- Saves best model checkpoint to `model.pt`.
- Uses `gymnasium.vector.AsyncVectorEnv` for parallel environment execution.

**`src/env_utils.py`** — Environment factory: `RecordEpisodeStatistics` → `EpisodicLifeEnv` → `AtariPreprocessing` (frame skip, grayscale, resize to 84x84, uint8) → `ClipRewardWrapper` → `FrameStackObservation` (4 frames).

## Key Details

- Observation shape after wrappers: `(4, 84, 84)` — 4 stacked grayscale frames (uint8, model normalizes to float internally)
- No test suite exists in this project
- Uses gymnasium API — `step()` returns 5 values (obs, reward, terminated, truncated, info)
- Device auto-detected: CUDA → MPS → CPU
