import time

from torch.utils.tensorboard import SummaryWriter

from src.ppo import PPO

if __name__ == "__main__":
    ENV = "PongNoFrameskip-v4"
    max_epochs = 200
    gamma = 0.99

    n_envs = 64
    n_steps = 512

    batch_size = 64
    v_loss_coef = 0.5

    max_grad_norm = 0.1

    epsilon = 0.2
    lr = 3e-4
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
            max_grad_norm=max_grad_norm,
            epsilon=epsilon,
        )
        ppo.train()
    except KeyboardInterrupt:
        pass
    finally:
        writer.close()
