import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Categorical
from gymnasium.vector import AsyncVectorEnv

from .model import ActorCriticNet
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
        max_epochs,
        n_envs,
        n_steps,
        batch_size,
        writer,
        epsilon=0.2,
        gamma=0.99,
        lambda_=0.95,
        v_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        lr=2.5e-4,
        ppo_epochs=4,
    ):
        self.envs = AsyncVectorEnv([make_env_function(env_name) for _ in range(n_envs)])
        self.env = make_env_with_wrappers(env_name)

        self.obs_space = self.envs.single_observation_space.shape
        action_space = self.envs.single_action_space.n
        self.model = ActorCriticNet(self.obs_space, action_space)
        self.device = _detect_device()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)

        self.writer = writer
        self.env_name = env_name

        self.max_epochs = max_epochs
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.batch_num = self.n_steps * self.n_envs // self.batch_size

        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.v_loss_coef = v_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs

        total_updates = max_epochs * ppo_epochs * self.batch_num
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda step: 1.0 - step / total_updates
        )

    def train(self):
        states, actions, log_probs, advantages, returns = self._rollout()
        best_score = -22

        for epoch in range(self.max_epochs):
            self._update(states, actions, log_probs, advantages, returns, epoch)
            del states
            states, actions, log_probs, advantages, returns = self._rollout()

            if (epoch + 1) % 10 == 0:
                print("epoch:", epoch + 1, end=", ")
                score = np.mean([self._run_episode(self.env) for _ in range(10)])

                if score > best_score:
                    best_score = score
                    torch.save(self.model.state_dict(), "model.pt")
                    print("saved best model with", end=" ")

                self.writer.add_scalar("Score/50episodes", score, epoch + 1)
                print("score:", score)

                if best_score >= 20:
                    print("Finished training!")
                    break
            else:
                print("epoch:", epoch + 1)

    def _rollout(self):
        states, actions, rewards, masks, values, log_probs = (
            self._collect_trajectories()
        )

        with torch.no_grad():
            _, last_val, _ = self.model(states[-1])
            last_val = last_val.to("cpu").numpy()

        advantages = self._compute_gae(rewards, values, masks, last_val)
        returns = self._compute_returns(advantages, values)

        states = states[:-1].view(-1, *self.obs_space)
        actions = torch.from_numpy(actions).long().view(-1, 1)
        log_probs = torch.from_numpy(log_probs).view(-1, 1)
        returns = torch.from_numpy(returns).view(-1, 1)
        advantages = torch.from_numpy(advantages).view(-1, 1)

        return states, actions, log_probs, advantages, returns

    def _collect_trajectories(self):
        states = torch.zeros([self.n_steps + 1, self.n_envs, *self.obs_space]).to(
            self.device
        )
        masks = np.ones([self.n_steps + 1, self.n_envs, 1], dtype=np.float32)
        rewards = np.zeros([self.n_steps, self.n_envs, 1], dtype=np.float32)
        actions = np.zeros([self.n_steps, self.n_envs, 1], dtype=np.int32)
        values = np.zeros([self.n_steps, self.n_envs, 1], dtype=np.float32)
        log_probs = np.zeros([self.n_steps, self.n_envs, 1], dtype=np.float32)

        obs, _ = self.envs.reset()
        states[0] = torch.from_numpy(obs)
        masks[0] = 0.0

        with torch.no_grad():
            for t in range(self.n_steps):
                logits, vals, acts = self.model(states[t])

                dist = Categorical(logits=logits)
                lp = dist.log_prob(acts.squeeze(-1)).unsqueeze(-1)

                actions[t] = acts.to("cpu").numpy()
                values[t] = vals.to("cpu").numpy()
                log_probs[t] = lp.to("cpu").numpy()
                states_np, rewards[t, :, 0], terminated, truncated, _ = self.envs.step(
                    actions[t].squeeze()
                )
                dones = np.logical_or(terminated, truncated)
                masks[t][dones] = 0
                states[t + 1] = torch.from_numpy(states_np)

        return states, actions, rewards, masks, values, log_probs

    def _compute_gae(self, rewards, values, masks, last_val):
        n_steps = rewards.shape[0]
        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(n_steps)):
            next_val = last_val if t == n_steps - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * masks[t + 1] * next_val - values[t]
            gae = delta + self.gamma * self.lambda_ * masks[t + 1] * gae
            advantages[t] = gae
        return advantages

    def _compute_returns(self, advantages, values):
        return advantages + values

    def _update(self, states, actions, old_log_probs, advantages, returns, epoch):
        policy_losses = []
        entropies = []
        value_losses = []
        losses = []

        old_values = None

        for ppo_epoch in range(self.ppo_epochs):
            rand_list = (
                torch.randperm(self.batch_num * self.batch_size)
                .view(-1, self.batch_size)
                .tolist()
            )

            for ind in rand_list:
                batch = states[ind]
                actor_logits, vals, _ = self.model(batch)

                batch_old_log_probs = old_log_probs[ind].to(self.device)
                batch_actions = actions[ind].to(self.device)
                batch_advantages = advantages[ind].to(self.device)
                batch_returns = returns[ind].to(self.device)

                if old_values is None:
                    batch_old_values = batch_returns
                else:
                    batch_old_values = old_values[ind].to(self.device)

                policy_loss, value_loss, entropy, total_loss = self._compute_losses(
                    actor_logits,
                    vals,
                    batch_old_log_probs,
                    batch_old_values,
                    batch_actions,
                    batch_advantages,
                    batch_returns,
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                losses.append(total_loss.item())
                entropies.append(entropy.item())

            if ppo_epoch == 0:
                with torch.no_grad():
                    _, old_vals, _ = self.model(states)
                old_values = old_vals.detach()

        self._log_metrics(
            {
                "PolicyLoss": np.mean(policy_losses),
                "ValueLoss": np.mean(value_losses),
                "Loss": np.mean(losses),
                "Entropy": np.mean(entropies),
            },
            epoch,
        )

    def _compute_losses(
        self,
        actor_logits,
        values,
        old_log_probs,
        old_values,
        actions,
        advantages,
        returns,
    ):
        dist = Categorical(logits=actor_logits)
        log_probs = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
        entropy = dist.entropy().mean()

        # Advantage normalization
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss (clipped surrogate)
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * adv
        surr2 = ratio.clamp(min=1 - self.epsilon, max=1 + self.epsilon) * adv
        policy_loss = torch.min(surr1, surr2).mean()

        # Value loss with clipping
        value_pred_clipped = old_values + (values - old_values).clamp(
            -self.epsilon, self.epsilon
        )
        vf_loss_unclipped = (values - returns).pow(2)
        vf_loss_clipped = (value_pred_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(vf_loss_unclipped, vf_loss_clipped).mean()

        total_loss = (
            -policy_loss + self.v_loss_coef * value_loss - self.entropy_coef * entropy
        )

        return policy_loss, value_loss, entropy, total_loss

    def _log_metrics(self, metrics, epoch):
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, epoch + 1)

    def _run_episode(self, env):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, _, action = self.model(state)
            next_state, reward, terminated, truncated, _ = env.step(
                action.to("cpu").item()
            )
            done = terminated or truncated
            state = next_state
            total_reward += reward

        return total_reward

    def _test_env(self):
        return self._run_episode(self.env)

    def eval(self, num_of_games):
        eval_env = make_env_with_wrappers(self.env_name, render_mode="human")
        self.model.load_state_dict(torch.load("model.pt", weights_only=True))
        self.model.eval()

        for _ in range(num_of_games):
            self._run_episode(eval_env)

        eval_env.close()
