import numpy as np
import torch
import torch.nn as nn
from torch import optim
from gymnasium.vector import AsyncVectorEnv

from .model import Agent
from .env_utils import make_env_function, make_env_with_wrappers


def _detect_device():
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PPO:
    def __init__(
        self,
        env_name,
        n_envs,
        n_steps,
        num_minibatches,
        total_timesteps,
        writer,
        clip_coef=0.1,
        gamma=0.99,
        gae_lambda=0.95,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        lr=2.5e-4,
        update_epochs=4,
        anneal_lr=True,
    ):
        self.envs = AsyncVectorEnv([make_env_function(env_name) for _ in range(n_envs)])
        self.eval_env = make_env_with_wrappers(env_name)

        self.device = _detect_device()
        self.model = Agent(self.envs).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)

        self.writer = writer
        self.env_name = env_name

        self.n_envs = n_envs
        self.n_steps = n_steps
        self.batch_size = n_envs * n_steps
        self.num_minibatches = num_minibatches
        self.minibatch_size = self.batch_size // num_minibatches

        self.total_timesteps = total_timesteps
        self.num_iterations = total_timesteps // self.batch_size

        self.gamma = gamma
        self.clip_coef = clip_coef
        self.gae_lambda = gae_lambda
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.anneal_lr = anneal_lr
        self.lr = lr

    def train(self):
        obs = torch.zeros(
            (self.n_steps, self.n_envs) + self.envs.single_observation_space.shape
        ).to(self.device)
        actions = torch.zeros((self.n_steps, self.n_envs)).to(self.device)
        logprobs = torch.zeros((self.n_steps, self.n_envs)).to(self.device)
        rewards = torch.zeros((self.n_steps, self.n_envs)).to(self.device)
        dones = torch.zeros((self.n_steps, self.n_envs)).to(self.device)
        values = torch.zeros((self.n_steps, self.n_envs)).to(self.device)

        next_obs, _ = self.envs.reset()
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.n_envs).to(self.device)

        global_step = 0
        best_score = -22

        for iteration in range(1, self.num_iterations + 1):
            # LR annealing
            if self.anneal_lr:
                frac = 1.0 - (iteration - 1) / self.num_iterations
                self.optimizer.param_groups[0]["lr"] = frac * self.lr

            # --- Rollout ---
            for step in range(self.n_steps):
                global_step += self.n_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = self.model.get_action_and_value(
                        next_obs
                    )
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                next_obs_np, reward, terminated, truncated, infos = self.envs.step(
                    action.cpu().numpy()
                )
                done = np.logical_or(terminated, truncated)
                rewards[step] = torch.tensor(reward, dtype=torch.float32).to(self.device).view(-1)
                next_obs = torch.Tensor(next_obs_np).to(self.device)
                next_done = torch.Tensor(done).to(self.device)

                # Log episode returns from RecordEpisodeStatistics
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info is not None and "episode" in info:
                            ep_return = info["episode"]["r"]
                            ep_length = info["episode"]["l"]
                            print(
                                f"global_step={global_step}, "
                                f"episodic_return={ep_return:.1f}"
                            )
                            self.writer.add_scalar(
                                "charts/episodic_return", ep_return, global_step
                            )
                            self.writer.add_scalar(
                                "charts/episodic_length", ep_length, global_step
                            )

            # --- GAE ---
            with torch.no_grad():
                next_value = self.model.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.n_steps)):
                    if t == self.n_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + self.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            # --- Flatten batches ---
            b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # --- PPO update ---
            clipfracs = []
            for epoch in range(self.update_epochs):
                b_inds = np.random.permutation(self.batch_size)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.model.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss with clipping
                    newvalue = newvalue.view(-1)
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

            # --- Logging ---
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            self.writer.add_scalar(
                "charts/learning_rate",
                self.optimizer.param_groups[0]["lr"],
                global_step,
            )
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            self.writer.add_scalar(
                "charts/explained_variance", explained_var, global_step
            )

            # Periodic evaluation
            if iteration % 100 == 0:
                score = np.mean([self._run_episode() for _ in range(10)])
                print(f"iteration={iteration}, eval_score={score:.1f}")
                self.writer.add_scalar("charts/eval_return", score, global_step)

                if score > best_score:
                    best_score = score
                    torch.save(self.model.state_dict(), "model.pt")
                    print(f"  saved best model (score={score:.1f})")

                if best_score >= 20:
                    print("Finished training!")
                    break

        self.envs.close()
        self.eval_env.close()

    def _run_episode(self):
        state, _ = self.eval_env.reset()
        done = False
        total_reward = 0

        while not done:
            state_t = torch.Tensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, _, _, _ = self.model.get_action_and_value(state_t)
            next_state, reward, terminated, truncated, _ = self.eval_env.step(
                action.cpu().item()
            )
            done = terminated or truncated
            state = next_state
            total_reward += reward

        return total_reward
