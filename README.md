## Self-Attention PPO Pytorch

I was inspired by [this paper](https://arxiv.org/pdf/1904.03367.pdf) which described few methods to approach for `Attention` for `Reinforcement Learning`.  
I decided that it will be best to implement simplest one.

This implementation don't have to be correct even though it works better than version without `Attention`.

## Setup 

```bash
# Create and activate virtual environment.
python3 -m venv venv
# Install packages
pip install -r requirements.txt
# Run program
python main.py
```

## Tensorboard

To run `tensorboard` use this command:
```bash
tensorboard --logdir runs
```

## Lint

To lint code use this command:
```bash
black .
```

## Troubleshooting

* (Ubuntu) During installation of `tensorboard` I encountered an error that required from me to install `python3-dev` package.
