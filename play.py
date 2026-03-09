"""Watch a trained agent play Pong (or another Atari game).

Usage:
    python play.py                  # play 3 games of Pong
    python play.py --games 5        # play 5 games
    python play.py --model best.pt  # load a different checkpoint
    python play.py --attention multi # use multi-head attention model
"""

import argparse

import pygame
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


def play(env_name, model_path, num_games, attention, scale, fps):
    device = _detect_device()
    env = make_env_with_wrappers(env_name, render_mode="rgb_array")

    model = make_agent(_SingleEnvShim(env), attention=attention)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    pygame.init()
    screen = None
    clock = pygame.time.Clock()

    for game in range(1, num_games + 1):
        state, _ = env.reset()
        frame = env.render()
        if screen is None:
            h, w = frame.shape[:2]
            screen = pygame.display.set_mode((w * scale, h * scale))
            pygame.display.set_caption(f"PPO Agent — {env_name}")

        done = False
        total_reward = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    return

            state_t = torch.Tensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, _ = model.get_action_and_value(state_t)
            state, reward, terminated, truncated, _ = env.step(action.cpu().item())
            done = terminated or truncated
            total_reward += reward

            frame = env.render()
            surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
            surf = pygame.transform.scale(surf, (w * scale, h * scale))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(fps)

        print(f"Game {game}: reward = {total_reward}")

    env.close()
    pygame.quit()


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
        "--scale", type=int, default=4, help="Window scale factor (default: 4)"
    )
    parser.add_argument(
        "--fps", type=int, default=24, help="Playback speed in FPS (default: 24)"
    )
    args = parser.parse_args()

    play(args.env, args.model, args.games, args.attention, args.scale, args.fps)
