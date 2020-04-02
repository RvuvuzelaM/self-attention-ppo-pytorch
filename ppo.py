import copy

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from multiprocess_env import SubprocVecEnv
from networks import ActorCriticCNN
from torch import optim
from torch.distributions import Categorical
from wrappers import make_env_function, make_env_with_wrappers


class PPO:
    def __init__(self, env_name, max_epochs, n_envs, n_steps, batch_size, writer,
                epsilon=0.2, gamma=0.99, lambda_=0.95, v_loss_coef=0.5, 
                entropy_coef=0.001, max_grad_norm=0.5, lr=0.004, ppo_epochs=4):
        self.envs = SubprocVecEnv([make_env_function(env_name) for _ in range(n_envs)])
        self.env = make_env_with_wrappers(env_name)

        self.obs_space = self.envs.observation_space.shape
        action_space = self.envs.action_space.n
        self.model = ActorCriticCNN(self.obs_space, action_space)
        self.device = 'cuda:0' if T.cuda.is_available() else 'cpu:0'
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.writer = writer

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


    def train(self):
        states, actions, rewards, advantages, returns, masks = self._rollout()

        best_score = -22

        for epoch in range(self.max_epochs):
            self._update(states, actions, rewards, advantages, returns, masks, epoch)
            del states
            states, actions, rewards, advantages, returns, masks = self._rollout()

            if (epoch + 1) % 10 == 0:
                print('epoch:', epoch+1, end=', ')
                score = np.mean([self._test_env() for _ in range(25)])

                if score > best_score:
                    best_score = score
                    T.save(self.model.state_dict(), 'model.pt')
                    print('saved best model with', end=' ')

                self.writer.add_scalar('Score/50episodes', score, epoch+1)
                print('score:', score)

                if best_score >= 20:
                    print('Finished training!')
                    break
            else:
                print('epoch:', epoch+1)


    def _rollout(self):
        states = T.zeros([self.n_steps+1, self.n_envs, *self.obs_space]).to(self.device)
        masks = np.ones([self.n_steps+1, self.n_envs, 1], dtype=np.float32)
        rewards = np.zeros([self.n_steps, self.n_envs, 1], dtype=np.float32)
        actions = np.zeros([self.n_steps, self.n_envs, 1], dtype=np.int32)
        values = np.zeros([self.n_steps+1, self.n_envs, 1], dtype=np.float32)

        states[0] = T.from_numpy(self.envs.reset())
        masks[0] = 0.0

        with T.no_grad():
            for t in range(self.n_steps):
                _, vals, acts = self.model(states[t])

                actions[t] = acts.to('cpu').numpy()
                values[t] = vals.to('cpu').numpy()
                states_np, rewards[t, :, 0], dones, _ = self.envs.step(actions[t])
                masks[t][dones] = 0
                states[t+1] = T.from_numpy(states_np)

            values[-1] = self.model.value(states[-1]).to('cpu').numpy()
            advantages = np.zeros([self.n_steps, self.n_envs, 1], dtype=np.float32)
            
            returns = np.zeros([self.n_steps, self.n_envs, 1], dtype=np.float32)
            returns[-1] = rewards[-1] + self.gamma * masks[-1] * values[-1]
            for t in reversed(range(self.n_steps-1)):
                returns[t] = rewards[t] + self.gamma * masks[t+1] * returns[t+1]

            gae = 0
            for t in reversed(range(self.n_steps)):
                delta = - values[t] + rewards[t]  + self.gamma * values[t+1]
                gae = delta + self.gamma * self.lambda_ * masks[t] * gae
                advantages[t] = gae

        states = states[:-1].view(-1, *self.obs_space)
        actions = T.from_numpy(actions).long().view(-1, 1)
        returns = T.from_numpy(returns).view(-1, 1)
        advantages = T.from_numpy(advantages).view(-1, 1)

        del values

        return states, actions, rewards, advantages, returns, masks

    def _update(self, states, actions, rewards, advantages, returns, masks, epoch):
        old_model = copy.deepcopy(self.model)

        policy_losses = np.array([])
        entropies = np.array([])
        value_losses = np.array([])
        losses = np.array([])

        for _ in range(self.ppo_epochs):
            rand_list = T.randperm(self.batch_num*self.batch_size).view(-1, self.batch_size).tolist()

            for ind in rand_list:
                batch = states[ind]
                actor_logits, vals, _ = self.model(batch)
                log_probs = F.log_softmax(actor_logits, dim=1)
                with T.no_grad():
                    old_actor_logits, _, _ = old_model(batch)
                    old_log_probs = F.log_softmax(old_actor_logits, dim=1)

                adv = advantages[ind].to(self.device)
                A = returns[ind].to(self.device) - vals

                action = actions[ind].to(self.device)

                old_log_probs = old_log_probs.gather(1, action)
                log_probs = log_probs.gather(1, action)

                r = (log_probs - old_log_probs).exp()

                clip = r.clamp(min=1-self.epsilon, max=1+self.epsilon)
                L, _ = T.stack([r * adv.detach(), clip * adv.detach()]).min(0)
                v_l = A.pow(2).mean()
                L = L.mean()

                entropy = Categorical(F.softmax(actor_logits, dim=1)).entropy().mean()

                loss = -L + self.v_loss_coef * v_l -self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_losses = np.append(policy_losses, L.cpu().detach().numpy())
                value_losses = np.append(value_losses, v_l.cpu().detach().numpy())
                losses = np.append(losses, loss.cpu().detach().numpy())
                entropies = np.append(entropies, entropy.cpu().detach().numpy())
        
        policy_loss = policy_losses.mean()
        value_loss = value_losses.mean()
        loss = losses.mean()
        entropy = entropies.mean()

        self.writer.add_scalar('PolicyLoss', policy_loss, epoch+1)
        self.writer.add_scalar('ValueLoss', value_loss, epoch+1)
        self.writer.add_scalar('Loss', loss, epoch+1)
        self.writer.add_scalar('Entropy', entropy, epoch+1)

        del states, actions, rewards, advantages, returns, masks


    def _test_env(self):

        state = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            state = T.FloatTensor(state).unsqueeze(0).to('cuda')
            action = self.model.act(state).to('cpu')
            next_state, reward, done, _ = self.env.step(action)

            state = next_state
            total_reward += reward

        return total_reward

    def eval(self, num_of_games):
        for _ in range(num_of_games):
            self.model.load_state_dict(T.load('model.pt'))
            self.model.eval()

            state = self.env.reset()
            done = False

            while not done:
                state = T.FloatTensor(state).unsqueeze(0).to('cuda')
                action = self.model.act(state).to('cpu')
                self.env.render(mode='human')
                state, _, done, _ = self.env.step(action)
