## Self-Attention PPO Pytorch

PPO implementation with selectable spatial self-attention variants for Atari, inspired by [Zambaldi et al.](https://arxiv.org/pdf/1904.03367.pdf). Compare plain CNN, single-head, and multi-head attention side-by-side.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (`ale-py` doesn't ship wheels for 3.14 yet).

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

## Train

```bash
uv run python main.py --attention none     # plain CNN baseline
uv run python main.py --attention single   # single-head spatial self-attention
uv run python main.py --attention multi    # multi-head spatial self-attention (Zambaldi et al.)
```

Each variant logs to its own TensorBoard directory under `runs/`.

## Watch a trained agent play

After training, a `model.pt` checkpoint is saved automatically. To watch the agent play:

```bash
uv run python play.py                              # plain CNN, 3 games
uv run python play.py --attention multi --games 5  # multi-head model, 5 games
uv run python play.py --model best.pt              # use a different checkpoint
uv run python play.py --action-repeat 8            # hold each action for 8 frames (calmer play)
```

By default `--action-repeat 4` is used — the agent holds each action for 4 frames before choosing a new one, reducing jittery micro-movements. Set to `1` for original per-frame behaviour.

## Compare runs with Tensorboard

```bash
uv run tensorboard --logdir runs
```

## Model variants

| Flag | Architecture | Description |
|------|-------------|-------------|
| `--attention none` | Plain CNN | 3 conv layers → flatten → linear 512 (CleanRL baseline) |
| `--attention single` | Single-head attention | conv1 → self-attention (1×1 Q/K/V + residual) → conv2 → conv3 |
| `--attention multi` | Multi-head attention | conv1 → 4-head attention (Q/K/V + output proj + residual + LayerNorm) → conv2 → conv3 |

Attention (single/multi) is applied after the first conv layer on the 20×20 spatial feature map (32 channels). 

## Lint

```bash
uv run ruff format .
uv run ruff check .
```
