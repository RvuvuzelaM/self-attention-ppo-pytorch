## Self-Attention PPO Pytorch

I was inspired by [this paper](https://arxiv.org/pdf/1904.03367.pdf) which described few methods to approach for `Attention` for `Reinforcement Learning`.  
I decided that it will be best to implement simplest one.

This implementation don't have to be correct even though it works better than version without `Attention`.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (`ale-py` doesn't ship wheels for 3.14 yet).

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
```

## Train

```bash
uv run main.py
```

## Watch a trained agent play

After training, a `model.pt` checkpoint is saved automatically. To watch the agent play:

```bash
uv run play.py              # play 3 games of Pong
uv run play.py --games 5    # play 5 games
uv run play.py --model best.pt  # use a different checkpoint
```

## Tensorboard

```bash
uv run tensorboard --logdir runs
```

## Lint

```bash
uv run ruff format .
uv run ruff check .
```
