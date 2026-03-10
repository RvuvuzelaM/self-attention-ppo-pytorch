import argparse
import time

from torch.utils.tensorboard import SummaryWriter

from src.ppo import PPO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Atari training")
    parser.add_argument(
        "--attention",
        choices=["none", "single", "multi"],
        default="none",
        help="Attention variant: none (plain CNN), single, or multi-head",
    )
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=2,
        help="Repeat each action for N env steps (default: 2)",
    )
    args = parser.parse_args()

    ENV = "ALE/Pong-v5"
    total_timesteps = 10_000_000
    n_envs = 8
    n_steps = 128
    num_minibatches = 4
    update_epochs = 4
    lr = 2.5e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.1
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    anneal_lr = True

    writer = SummaryWriter(f"runs/ppo_pong_{args.attention}/{time.time()}")
    try:
        ppo = PPO(
            ENV,
            n_envs,
            n_steps,
            num_minibatches,
            total_timesteps,
            writer,
            clip_coef=clip_coef,
            gamma=gamma,
            gae_lambda=gae_lambda,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            lr=lr,
            update_epochs=update_epochs,
            anneal_lr=anneal_lr,
            attention=args.attention,
            action_repeat=args.action_repeat,
        )
        ppo.train()
    except KeyboardInterrupt:
        pass
    finally:
        writer.close()
