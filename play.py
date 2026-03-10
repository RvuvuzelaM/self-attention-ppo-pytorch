"""Watch a trained agent play Pong (or another Atari game).

Usage:
    python play.py
    python play.py --games 5
    python play.py --model best.pt
    python play.py --attention multi
"""

import argparse
import time

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


def play(env_name, model_path, num_games, attention, action_repeat):
    device = _detect_device()

    env = make_env_with_wrappers(
        env_name,
        render_mode="human",
        action_repeat=action_repeat,
    )

    model = make_agent(_SingleEnvShim(env), attention=attention)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    try:
        for game in range(1, num_games + 1):
            state, _ = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                state_t = torch.as_tensor(
                    state, dtype=torch.float32, device=device
                ).unsqueeze(0)

                with torch.no_grad():
                    action, _, _, _ = model.get_action_and_value(state_t)

                state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                total_reward += reward

            print(f"Game {game}: reward = {total_reward}")

            if game < num_games:
                time.sleep(1.0)
    finally:
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
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=4,
        help="Hold each action for N frames before choosing a new one (default: 4)",
    )

    # Kept only so old commands like `--fps 30` or `--scale 4` do not crash.
    # These are ignored in human-render mode.
    parser.add_argument("--fps", type=int, default=30, help=argparse.SUPPRESS)
    parser.add_argument("--scale", type=int, default=4, help=argparse.SUPPRESS)

    args = parser.parse_args()

    play(
        args.env,
        args.model,
        args.games,
        args.attention,
        args.action_repeat,
    )
