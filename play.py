"""Watch a trained agent play Pong (or another Atari game).

Usage:
    python play.py                  # play 3 games of Pong
    python play.py --games 5        # play 5 games
    python play.py --model best.pt  # load a different checkpoint
"""

import argparse

import torch

from src.env_utils import make_env_with_wrappers
from src.model import ActorCriticNet


def _detect_device():
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def play(env_name, model_path, num_games):
    device = _detect_device()
    env = make_env_with_wrappers(env_name, render_mode="human")

    obs_shape = env.observation_space.shape
    action_space = env.action_space.n
    model = ActorCriticNet(obs_shape, action_space)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    for game in range(1, num_games + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                _, _, action = model(state_t)
            state, reward, terminated, truncated, _ = env.step(action.to("cpu").item())
            done = terminated or truncated
            total_reward += reward

        print(f"Game {game}: reward = {total_reward}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch a trained PPO agent play")
    parser.add_argument("--env", default="ALE/Pong-v5", help="Gymnasium env ID")
    parser.add_argument("--model", default="model.pt", help="Path to saved model weights")
    parser.add_argument("--games", type=int, default=3, help="Number of games to play")
    args = parser.parse_args()

    play(args.env, args.model, args.games)
