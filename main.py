import time

from torch.utils.tensorboard import SummaryWriter

from src.ppo import PPO

if __name__ == "__main__":
    ENV = "ALE/Pong-v5"
    max_epochs = 200
    gamma = 0.99

    n_envs = 8
    n_steps = 128

    batch_size = 256
    v_loss_coef = 0.5
    entropy_coef = 0.01

    epsilon = 0.2
    lr = 2.5e-4
    writer = SummaryWriter("runs/no_attention/" + str(time.time()))
    try:
        ppo = PPO(
            ENV,
            max_epochs,
            n_envs,
            n_steps,
            batch_size,
            writer,
            lr=lr,
            v_loss_coef=v_loss_coef,
            entropy_coef=entropy_coef,
            epsilon=epsilon,
        )
        ppo.train()
    except KeyboardInterrupt:
        pass
    finally:
        writer.close()
