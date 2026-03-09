# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PPO (Proximal Policy Optimization) implementation in PyTorch, based on [CleanRL's ppo_atari.py](https://github.com/vwxyzjn/cleanrl). Trains an agent to play Atari games (default: ALE/Pong-v5) using PPO with selectable attention variants for A/B comparison.

## Commands

```bash
# Setup (requires uv: https://docs.astral.sh/uv/)
uv venv --python 3.12
uv pip install -r requirements.txt

# Train (select attention variant)
uv run python main.py --attention none     # plain CNN baseline
uv run python main.py --attention single   # single-head spatial self-attention
uv run python main.py --attention multi    # multi-head spatial self-attention

# Watch trained agent
uv run python play.py --attention none
uv run python play.py --attention multi --model model.pt

# Tensorboard (compare all runs)
uv run tensorboard --logdir runs

# Lint & format (uses ruff)
uv run ruff format .
uv run ruff check .
```

## Architecture

**Entry point:** `main.py` — `--attention` flag selects model variant. Configures hyperparameters and launches PPO training with TensorBoard logging to `runs/ppo_pong_{attention}/{time}`.

**`src/models/`** — Self-contained model variants, each with its own `Agent` class:
- `__init__.py`: `make_agent(envs, attention="none")` factory that imports the right module.
- `plain.py`: Standard CNN (3 conv layers → flatten → linear 512) with orthogonal init. Same as original CleanRL architecture.
- `single_head.py`: conv1 → ReLU → `SpatialSelfAttention` (1×1 conv Q/K/V, scaled dot-product, residual) → conv2 → ReLU → conv3 → ReLU → flatten → fc.
- `multi_head.py`: conv1 → ReLU → `MultiHeadSpatialAttention` (4 heads, 1×1 conv Q/K/V, per-head attention, output projection, residual + LayerNorm) → conv2 → ReLU → conv3 → ReLU → flatten → fc. Based on Zambaldi et al.

All agents share the same API: `get_value(x)` and `get_action_and_value(x, action=None)`. Input is uint8, divided by 255 inside the model.

**`src/ppo.py`** — PPO algorithm (`PPO` class):
- Accepts `attention` param, uses `make_agent()` to build the model.
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
- Attention is inserted after the first conv layer (32 channels, 20×20 spatial)
