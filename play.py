"""Watch a trained agent play Pong (or another Atari game).

Usage:
    python play.py                  # play 3 games of Pong
    python play.py --games 5        # play 5 games
    python play.py --model best.pt  # load a different checkpoint
    python play.py --attention multi # use multi-head attention model
"""

import argparse

import torch

from src.env_utils import make_env_with_wrappers
from src.models import make_agent


def _detect_device():
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _SingleEnvShim:
    """Minimal shim so Agent(envs) works with a single env."""

    def __init__(self, env):
        self.single_action_space = env.action_space


def play(env_name, model_path, num_games, attention):
    device = _detect_device()
    env = make_env_with_wrappers(env_name, render_mode="human")

    model = make_agent(_SingleEnvShim(env), attention=attention)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    for game in range(1, num_games + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_t = torch.Tensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, _ = model.get_action_and_value(state_t)
            state, reward, terminated, truncated, _ = env.step(action.cpu().item())
            done = terminated or truncated
            total_reward += reward

        print(f"Game {game}: reward = {total_reward}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch a trained PPO agent play")
    parser.add_argument("--env", default="ALE/Pong-v5", help="Gymnasium env ID")
    parser.add_argument(
        "--model", default="model.pt", help="Path to saved model weights"
    )
    parser.add_argument("--games", type=int, default=3, help="Number of games to play")
    parser.add_argument(
        "--attention",
        choices=["none", "single", "multi"],
        default="none",
        help="Attention variant used when training the model",
    )
    args = parser.parse_args()

    play(args.env, args.model, args.games, args.attention)
