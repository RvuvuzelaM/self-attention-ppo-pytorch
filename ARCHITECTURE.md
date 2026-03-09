# Architecture

Detailed technical reference for the Self-Attention PPO codebase. This document covers every component, traces tensor shapes through the entire pipeline, and explains the algorithms with their mathematical foundations.

## Table of Contents

- [Project Overview](#project-overview)
- [File Structure](#file-structure)
- [Environment Pipeline](#environment-pipeline)
- [Neural Network](#neural-network)
- [PPO Algorithm](#ppo-algorithm)
- [End-to-End Data Flow](#end-to-end-data-flow)
- [Hyperparameters](#hyperparameters)

---

## Project Overview

This project implements Proximal Policy Optimization (PPO) with a spatial self-attention mechanism inserted into the CNN feature extractor. Inspired by [Attentive Multi-Task Deep Reinforcement Learning](https://arxiv.org/pdf/1904.03367.pdf), the attention layer lets the agent learn which spatial regions of the game frame are important for decision-making.

The default environment is **ALE/Pong-v5** (Atari Pong via the Arcade Learning Environment). The agent learns to play from raw pixel observations using an actor-critic architecture trained with PPO.

**Key idea:** A standard PPO agent uses a CNN to extract features from stacked game frames. This project adds a spatial self-attention layer between the first and second convolution layers, allowing the network to weigh spatial positions by their relevance before further processing. Every spatial position in the feature map attends to every other position, enabling the network to capture long-range spatial dependencies.

---

## File Structure

```
self-attention-ppo-pytorch/
├── main.py              # Entry point — sets hyperparameters, creates PPO, launches training
├── src/
│   ├── env_utils.py     # Environment factory — wrapper chain for Atari preprocessing
│   ├── model.py         # ActorCriticNet, SpatialSelfAttention
│   └── ppo.py           # PPO class — rollout collection, GAE, update loop, evaluation
├── requirements.txt     # Dependencies (torch, gymnasium, ale-py, tensorboard, ruff, etc.)
├── CLAUDE.md            # Instructions for Claude Code
└── runs/                # TensorBoard log directory (created at runtime)
```

**Runtime artifacts:**
- `model.pt` — Best model checkpoint (saved when evaluation score improves)
- `runs/no_attention/<timestamp>/` — TensorBoard event files

---

## Environment Pipeline

**File:** `src/env_utils.py`

The environment pipeline transforms raw Atari frames into a compact tensor suitable for the CNN. Each wrapper in the chain performs a specific preprocessing step.

### Wrapper Chain

```
gym.make("ALE/Pong-v5", frameskip=1)     Raw ALE environment, frameskip disabled
        │
        ▼
EpisodicLifeEnv                           Treat life loss as episode boundary
        │
        ▼
AtariPreprocessing                        Noop reset, frame skip, resize, grayscale, scale
        │
        ▼
ClipRewardWrapper                         Clip rewards to {-1, 0, +1}
        │
        ▼
FrameStackObservation(stack_size=4)       Stack 4 consecutive frames
        │
        ▼
Observation shape: (4, 84, 84)            4 channels, 84x84 pixels, float32 in [0, 1]
```

### Wrapper Details

#### 1. `gym.make("ALE/Pong-v5", frameskip=1)`

Creates the base Atari environment. `frameskip=1` is set at the ALE level because frame skipping is handled by `AtariPreprocessing` instead (to enable max-pooling between skipped frames).

#### 2. `EpisodicLifeEnv` (custom wrapper)

Treats each **life loss** as an episode termination signal. This is a common Atari training trick:

- **Why:** In games with multiple lives (Pong has none, but many Atari games do), the agent gets a clearer learning signal by treating each life as a separate episode during training.
- **Mechanism:** After a life loss, `terminated` is set to `True` so GAE treats it as an episode boundary. However, the environment only truly resets (calls `env.reset()`) when all lives are gone. On a "fake" done, it instead sends a no-op action (action 0) to advance the environment.

#### 3. `AtariPreprocessing` (gymnasium built-in)

| Parameter          | Value   | Effect                                                    |
|--------------------|---------|-----------------------------------------------------------|
| `noop_max`         | 30      | On reset, execute 0-30 random no-op actions for stochasticity |
| `frame_skip`       | 4       | Repeat each action for 4 frames, max-pool last 2 for output  |
| `screen_size`      | 84      | Resize to 84x84 pixels                                    |
| `grayscale_obs`    | True    | Convert RGB to grayscale                                   |
| `grayscale_newaxis`| False   | Output shape is (84, 84), not (84, 84, 1)                 |
| `scale_obs`        | True    | Scale pixel values from [0, 255] to [0.0, 1.0]            |

**Observation transform:** `(210, 160, 3) RGB uint8` → `(84, 84) grayscale float32`

#### 4. `ClipRewardWrapper` (custom wrapper)

Applies `np.sign(reward)` to clip all rewards to `{-1, 0, +1}`. This normalizes the reward scale across different Atari games and stabilizes training.

#### 5. `FrameStackObservation(stack_size=4)`

Stacks the 4 most recent frames along a new first axis. This gives the agent temporal information (e.g., ball velocity in Pong).

**Final observation shape:** `(4, 84, 84)` — 4 grayscale frames, each 84x84, float32 in [0, 1].

### Factory Functions

- `make_env_with_wrappers(env_name, render_mode=None)` — Creates a single wrapped environment.
- `make_env_function(env_name)` — Returns a thunk (zero-argument callable) that creates a wrapped environment. Used by `AsyncVectorEnv` to spawn environments in separate processes.

---

## Neural Network

**File:** `src/model.py`

### ActorCriticNet

A shared-backbone actor-critic network. The CNN processes observations into a feature vector, which is then fed to separate actor (policy) and critic (value) heads.

#### Architecture Diagram

```
Input: (B, 4, 84, 84)
        │
        ▼
Conv2d(4→32, 8x8, stride=4) + ReLU      → (B, 32, 20, 20)
        │
        ▼
SpatialSelfAttention(32)                  → (B, 32, 20, 20)  [with residual]
        │
        ▼
Conv2d(32→64, 4x4, stride=2) + ReLU      → (B, 64, 9, 9)
        │
        ▼
Conv2d(64→64, 3x3, stride=1) + ReLU      → (B, 64, 7, 7)
        │
        ▼
Flatten                                   → (B, 3136)
        │
        ▼
Linear(3136→512) + ReLU                   → (B, 512)
        │
        ├──► Linear(512→6)                → (B, 6)   actor logits
        │
        └──► Linear(512→1)                → (B, 1)   critic value
```

*(6 = number of actions in Pong; varies by game)*

#### Tensor Shape Walkthrough

| Layer                    | Output Shape       | Calculation                        |
|--------------------------|--------------------|------------------------------------|
| Input                    | (B, 4, 84, 84)    | 4 stacked grayscale frames         |
| Conv1 (8x8, stride 4)   | (B, 32, 20, 20)   | floor((84 - 8) / 4 + 1) = 20      |
| Self-Attention           | (B, 32, 20, 20)   | Same spatial dims, residual added  |
| Conv2 (4x4, stride 2)   | (B, 64, 9, 9)     | floor((20 - 4) / 2 + 1) = 9       |
| Conv3 (3x3, stride 1)   | (B, 64, 7, 7)     | floor((9 - 3) / 1 + 1) = 7        |
| Flatten                  | (B, 3136)          | 64 * 7 * 7 = 3136                 |
| Linear1                  | (B, 512)           |                                    |
| Actor head               | (B, A)             | A = action space size              |
| Critic head              | (B, 1)             |                                    |

#### Weight Initialization

Orthogonal initialization is used throughout, following best practices from [Implementation Matters in Deep RL](https://arxiv.org/abs/2005.12729):

| Layer(s)          | Gain        | Bias | Rationale                                    |
|-------------------|-------------|------|----------------------------------------------|
| Conv1, Conv2, Conv3, Linear1 | √2  | 0.0  | Standard for ReLU layers (preserves variance) |
| Actor head        | 0.01        | 0.0  | Near-uniform initial policy (exploration)     |
| Critic head       | 1.0         | 0.0  | Standard scale for value predictions          |

#### Forward Pass

```python
def forward(self, x):
    h = self.shared_layer(x)     # CNN + attention + flatten + linear
    actor_logits = self.actor(h)  # Raw unnormalized action scores
    values = self.critic(h)       # State value estimate V(s)
    prob = F.softmax(actor_logits, dim=-1)
    acts = prob.multinomial(1)    # Sample action from policy
    return actor_logits, values, acts
```

Returns a tuple of `(logits, values, actions)` where:
- `logits`: (B, A) — raw scores, used to construct `Categorical` distribution in PPO
- `values`: (B, 1) — estimated V(s) for each state
- `acts`: (B, 1) — sampled action indices

### SpatialSelfAttention

Full spatial self-attention using 1x1 convolutions for Q/K/V projections, with a residual connection. All H×W spatial positions attend to every other position, enabling the network to capture long-range spatial dependencies across the entire feature map.

#### Mechanism

```
Input: (B, 32, 20, 20)  [feature maps from Conv1]
        │
        ├── w_q: Conv2d(32, 32, 1) ──► Q: (B, 32, 20, 20)
        ├── w_k: Conv2d(32, 32, 1) ──► K: (B, 32, 20, 20)
        └── w_v: Conv2d(32, 32, 1) ──► V: (B, 32, 20, 20)
                                            │
                                     flatten + permute
                                            │
                                            ▼
                                    Q, K, V: (B, 400, 32)   [H*W=400 spatial positions]
                                            │
                                            ▼
                              Q @ K^T / √32 → (B, 400, 400)  [full spatial attention]
                                            │
                                      softmax(dim=-1)
                                            │
                                     attn @ V → (B, 400, 32)
                                            │
                                    reshape to (B, 32, 20, 20)
                                            │
                                     + residual (input)
                                            │
                                            ▼
                                     (B, 32, 20, 20)
```

**1x1 convolutions** act as learned linear projections applied independently to each spatial position, transforming the 32-channel feature vector at each (h, w) location into query, key, and value representations.

#### Attention Pattern

The spatial dimensions are flattened to a single sequence of H×W = 400 positions:

- `Q @ K^T` computes `(B, 400, 32) @ (B, 32, 400)` → `(B, 400, 400)`
- Every spatial position attends to every other position (full pairwise attention)
- Division by `√d_k` (where `d_k = 32`) prevents softmax saturation
- Softmax normalizes attention weights along the last dimension
- `attn @ V` produces `(B, 400, 32)` — weighted combination of values

This creates **full spatial self-attention**: each position in the 20×20 feature map can attend to any other position regardless of distance, capturing both horizontal and vertical spatial relationships.

#### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(Q K^T / √d_k) V
```

| Step              | Shape                | Notes                                    |
|-------------------|----------------------|------------------------------------------|
| Q, K, V input     | (B, 400, 32)        | Flattened from (B, C, H, W)             |
| Q @ K^T           | (B, 400, 400)       | Full pairwise attention scores           |
| / √32             | (B, 400, 400)       | Scaled to prevent gradient issues        |
| softmax(dim=-1)   | (B, 400, 400)       | Normalized attention weights             |
| @ V               | (B, 400, 32)        | Weighted value vectors                   |
| reshape           | (B, 32, 20, 20)     | Back to spatial feature map              |

---

## PPO Algorithm

**File:** `src/ppo.py`

### Overview

The `PPO` class implements the full training pipeline:

1. **Collect trajectories** from parallel environments
2. **Compute advantages** using Generalized Advantage Estimation (GAE)
3. **Compute returns** from advantages + values
4. **Update policy** using clipped surrogate objective over multiple epochs

### Device Detection

```
CUDA → MPS → CPU
```

Checked at initialization via `_detect_device()`. The model and trajectory states are placed on the detected device.

### Training Loop

```python
def train(self):
    # Initial rollout
    states, actions, log_probs, advantages, returns = self._rollout()
    best_score = -22  # Pong minimum score

    for epoch in range(max_epochs):  # 200 epochs
        # Update policy using collected data
        self._update(states, actions, log_probs, advantages, returns, epoch)

        # Collect new trajectories with updated policy
        states, actions, log_probs, advantages, returns = self._rollout()

        # Every 10 epochs: evaluate over 10 episodes
        if (epoch + 1) % 10 == 0:
            score = mean of 10 evaluation episodes
            if score > best_score:
                save model to model.pt
            if best_score >= 20:
                break  # Solved!
```

Note: the very first rollout happens **before** the loop, so epoch 0 updates using data from the initial (random) policy. Each subsequent epoch first updates, then collects fresh data for the next iteration.

### Trajectory Collection (`_collect_trajectories`)

Runs the policy for `n_steps` steps across `n_envs` parallel environments.

#### Buffer Shapes

| Buffer      | Shape                            | Dtype   | Storage | Notes                    |
|-------------|----------------------------------|---------|---------|--------------------------|
| `states`    | (129, 8, 4, 84, 84)             | float32 | device  | n_steps+1 for bootstrap  |
| `masks`     | (129, 8, 1)                     | float32 | CPU     | 0 at episode boundaries  |
| `rewards`   | (128, 8, 1)                     | float32 | CPU     |                          |
| `actions`   | (128, 8, 1)                     | int32   | CPU     |                          |
| `values`    | (128, 8, 1)                     | float32 | CPU     | V(s) predictions         |
| `log_probs` | (128, 8, 1)                     | float32 | CPU     | log π(a|s)               |

*(With default n_steps=128, n_envs=8)*

#### Collection Step

For each timestep `t` in `range(128)`:

1. **Forward pass:** Feed `states[t]` through model → get `logits`, `values`, `actions`
2. **Log probs:** Create `Categorical(logits=logits)`, compute `log_prob(actions)`
3. **Store:** Save actions, values, log_probs to buffers
4. **Environment step:** Send actions to all 8 environments simultaneously via `AsyncVectorEnv`
5. **Process results:** Store rewards, compute done flags (`terminated | truncated`), set masks, store next states

The environments are reset at the start of each rollout with `self.envs.reset()`. When an episode terminates at step `t`, `masks[t+1]` is set to 0, preventing GAE from bootstrapping the value of the auto-reset observation at `states[t+1]`.

### Generalized Advantage Estimation (GAE)

GAE balances bias-variance in advantage estimation via parameter λ.

#### Mathematical Foundation

The temporal difference (TD) error at step t:

```
δ_t = r_t + γ * mask_{t+1} * V(s_{t+1}) - V(s_t)
```

The GAE advantage is an exponentially-weighted sum of TD errors:

```
A_t = δ_t + (γλ) * mask_{t+1} * δ_{t+1} + (γλ)² * mask_{t+1} * mask_{t+2} * δ_{t+2} + ...
```

Computed efficiently via backward recursion:

```
A_{T-1} = δ_{T-1}
A_t     = δ_t + γ * λ * mask_{t+1} * A_{t+1}
```

#### Implementation

```python
def _compute_gae(self, rewards, values, masks, last_val):
    advantages = np.zeros_like(rewards)  # (128, 8, 1)
    gae = 0
    for t in reversed(range(n_steps)):   # t = 127, 126, ..., 0
        next_val = last_val if t == n_steps - 1 else values[t + 1]
        delta = rewards[t] + gamma * masks[t + 1] * next_val - values[t]
        gae = delta + gamma * lambda_ * masks[t + 1] * gae
        advantages[t] = gae
    return advantages
```

- `last_val`: V(s_128) from an extra forward pass — used to bootstrap the final step
- `masks[t + 1]`: Zeroes out future terms at episode boundaries (prevents value leaking across episodes)
- When `mask = 0`: `delta = reward - V(s_t)` (no bootstrap), `gae = delta` (advantage resets)

#### GAE λ Spectrum

| λ value | Behavior                     |
|---------|------------------------------|
| λ = 0   | Pure TD(0): A_t = δ_t (high bias, low variance) |
| λ = 1   | Monte Carlo-like (low bias, high variance) |
| λ = 0.95| Default — good balance       |

### Returns Computation

```python
def _compute_returns(self, advantages, values):
    return advantages + values  # R_t = A_t + V(s_t)
```

This is equivalent to the discounted return target. Used as the regression target for the value function.

### Flattening for Update

After GAE, the time and environment dimensions are merged into a single batch dimension:

| Tensor        | Before                | After            |
|---------------|-----------------------|------------------|
| `states`      | (128, 8, 4, 84, 84)  | (1024, 4, 84, 84) |
| `actions`     | (128, 8, 1)           | (1024, 1)        |
| `log_probs`   | (128, 8, 1)           | (1024, 1)        |
| `returns`     | (128, 8, 1)           | (1024, 1)        |
| `advantages`  | (128, 8, 1)           | (1024, 1)        |

Total samples per rollout: `128 * 8 = 1024`

### PPO Update (`_update`)

For each epoch, the 1024 samples are shuffled into minibatches and used to update the policy.

#### Minibatch Shuffling

```python
rand_list = torch.randperm(batch_num * batch_size).view(-1, batch_size).tolist()
# batch_num = 1024 // 256 = 4
# rand_list: 4 lists of 256 random indices (covering indices 0-1023)
```

Each PPO epoch creates a fresh random permutation split into 4 minibatches of 256.

#### Per-Minibatch Update

For each minibatch of 256 samples:

1. **Forward pass** on batch states → `actor_logits` (256, 6), `values` (256, 1)
2. **Compute losses** (see below)
3. **Backprop:** `optimizer.zero_grad()` → `total_loss.backward()` → gradient clipping → `optimizer.step()` → `scheduler.step()`

#### Value Clipping for Old Values

On the first PPO epoch (`ppo_epoch == 0`), `old_values` is initialized to `batch_returns` (no clipping possible). After completing the first PPO epoch, the model's current value predictions for all states are computed in chunks (to avoid memory pressure from the attention layer's O(N²) cost) and saved as `old_values`. Subsequent PPO epochs (1, 2, 3) use these stored values for value clipping.

### Loss Computation (`_compute_losses`)

#### 1. Advantage Normalization

```python
adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

Normalizes advantages within each minibatch to zero mean and unit variance. This stabilizes training by ensuring consistent gradient magnitudes regardless of the reward scale.

#### 2. Policy Loss (Clipped Surrogate Objective)

The core PPO mechanism prevents destructively large policy updates:

```python
ratio = exp(log_prob - old_log_prob)         # π_new(a|s) / π_old(a|s)
surr1 = ratio * adv                          # Unclipped objective
surr2 = clamp(ratio, 1-ε, 1+ε) * adv        # Clipped objective
policy_loss = min(surr1, surr2).mean()       # Pessimistic bound
```

- `ratio > 1`: New policy assigns higher probability to this action
- `ratio < 1`: New policy assigns lower probability
- Clipping to `[1 - 0.2, 1 + 0.2]` = `[0.8, 1.2]` limits how much the policy can change
- `min` takes the pessimistic (lower) bound — this prevents the policy from exploiting large advantages with large ratio changes

#### 3. Value Loss (Clipped)

```python
value_pred_clipped = old_values + clamp(values - old_values, -ε, +ε)
vf_loss_unclipped = (values - returns)²
vf_loss_clipped   = (value_pred_clipped - returns)²
value_loss = 0.5 * max(vf_loss_unclipped, vf_loss_clipped).mean()
```

Similar clipping philosophy as policy loss — prevents the value function from changing too rapidly.

#### 4. Entropy Bonus

```python
entropy = Categorical(logits=actor_logits).entropy().mean()
```

Higher entropy = more exploration. Added to the objective to prevent premature convergence to a deterministic policy.

#### 5. Total Loss

```python
total_loss = -policy_loss + v_loss_coef * value_loss - entropy_coef * entropy
```

| Term             | Sign     | Effect                               |
|------------------|----------|--------------------------------------|
| `-policy_loss`   | minimize | Maximize expected advantage          |
| `+value_loss`    | minimize | Improve value predictions            |
| `-entropy`       | minimize | Maximize entropy (encourage exploration) |

### Learning Rate Annealing

```python
total_updates = max_epochs * ppo_epochs * batch_num  # 200 * 4 * 4 = 3200
scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0 - step / total_updates)
```

Linear decay from `2.5e-4` to `0` over all optimizer steps. The scheduler steps once per minibatch update, not per epoch. This means the learning rate smoothly decreases throughout the entire training run.

### Evaluation (`_run_episode`, `_test_env`, `eval`)

#### During Training (`_run_episode` + `_test_env`)

Every 10 epochs, the current policy is evaluated for 10 episodes on a **single, separate environment** (`self.env`). Actions are sampled from the policy (not greedy). The mean score is logged to TensorBoard.

#### Post-Training (`eval`)

Loads the best checkpoint (`model.pt`), creates an environment with `render_mode="human"` for visual playback, and runs the specified number of episodes.

---

## End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          ENVIRONMENT LAYER                              │
│                                                                         │
│  AsyncVectorEnv (8 parallel processes)                                  │
│  ┌──────────────────────────────────────────────┐                       │
│  │ ALE/Pong-v5                                  │                       │
│  │   → EpisodicLifeEnv (life→done)              │  × 8 envs             │
│  │   → AtariPreprocessing (84x84, grayscale)    │                       │
│  │   → ClipRewardWrapper ({-1,0,+1})            │                       │
│  │   → FrameStackObservation (4 frames)         │                       │
│  └──────────────────────────────────────────────┘                       │
│         │ obs: (8, 4, 84, 84)     ▲ actions: (8,)                       │
└─────────┼─────────────────────────┼─────────────────────────────────────┘
          ▼                         │
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRAJECTORY COLLECTION                            │
│                                                                         │
│  for t in range(128):                                                   │
│    model(states[t]) → logits, values, actions                           │
│    envs.step(actions) → next_states, rewards, dones                     │
│                                                                         │
│  Buffers filled: states(129,8,4,84,84) actions(128,8,1)                 │
│                  rewards(128,8,1) values(128,8,1) log_probs(128,8,1)    │
└─────────┬───────────────────────────────────────────────────────────────┘
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          GAE + RETURNS                                   │
│                                                                         │
│  Bootstrap: model(states[128]) → last_val                               │
│  GAE backward pass: advantages(128,8,1)                                 │
│  Returns = advantages + values                                          │
│                                                                         │
│  Flatten: (128*8=1024) samples                                          │
│  states(1024,4,84,84) actions(1024,1) log_probs(1024,1)                 │
│  advantages(1024,1) returns(1024,1)                                     │
└─────────┬───────────────────────────────────────────────────────────────┘
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          PPO UPDATE                                      │
│                                                                         │
│  for ppo_epoch in range(4):                                             │
│    shuffle 1024 samples into 4 minibatches of 256                       │
│                                                                         │
│    for each minibatch:                                                  │
│      ┌──────────────────────────────────────────────────────────┐       │
│      │  Forward: states(256,4,84,84) → logits(256,6), vals(256,1)│      │
│      │                                                          │       │
│      │  Policy loss: clipped surrogate objective                │       │
│      │  Value loss:  clipped value regression                   │       │
│      │  Entropy:     categorical entropy bonus                  │       │
│      │                                                          │       │
│      │  total = -policy + 0.5*value - 0.01*entropy              │       │
│      │                                                          │       │
│      │  Backprop → clip grads (0.5) → optimizer step → LR step │       │
│      └──────────────────────────────────────────────────────────┘       │
│                                                                         │
│  Total: 4 epochs × 4 batches = 16 gradient updates per rollout          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Hyperparameters

### Training

| Parameter        | Value    | Variable         | Description                                                 |
|------------------|----------|------------------|-------------------------------------------------------------|
| Environment      | ALE/Pong-v5 | `ENV`         | Atari Pong via Arcade Learning Environment                  |
| Max epochs       | 200      | `max_epochs`     | Maximum training epochs (early stop at score >= 20)         |
| Parallel envs    | 8        | `n_envs`         | Number of environments in AsyncVectorEnv                    |
| Rollout steps    | 128      | `n_steps`        | Steps per environment per rollout                           |
| Samples/rollout  | 1024     | n_steps * n_envs | Total transitions collected per rollout                     |
| Batch size       | 256      | `batch_size`     | Minibatch size for PPO updates                              |
| Batches/epoch    | 4        | `batch_num`      | 1024 / 256 = 4 minibatches per PPO epoch                    |
| PPO epochs       | 4        | `ppo_epochs`     | Number of passes over the rollout data per training epoch   |
| Updates/epoch    | 16       | ppo_epochs * batch_num | Gradient steps per training epoch                   |
| Total updates    | 3200     | max_epochs * 16  | Total gradient steps over full training                     |

### Optimization

| Parameter        | Value    | Variable         | Description                                                 |
|------------------|----------|------------------|-------------------------------------------------------------|
| Learning rate    | 2.5e-4   | `lr`             | Initial Adam learning rate                                  |
| Adam epsilon     | 1e-5     | (hardcoded)      | Numerical stability for Adam denominator                    |
| LR schedule      | Linear   | `scheduler`      | Linearly decays to 0 over all 3200 updates                  |
| Max grad norm    | 0.5      | `max_grad_norm`  | Global gradient clipping threshold                          |

### PPO

| Parameter        | Value    | Variable         | Description                                                 |
|------------------|----------|------------------|-------------------------------------------------------------|
| Clip epsilon     | 0.2      | `epsilon`        | PPO clipping range [1-ε, 1+ε] for both policy and value    |
| Discount (γ)     | 0.99     | `gamma`          | Reward discount factor                                      |
| GAE lambda (λ)   | 0.95     | `lambda_`        | GAE bias-variance tradeoff                                  |
| Value loss coef  | 0.5      | `v_loss_coef`    | Weight of value loss in total loss                          |
| Entropy coef     | 0.01     | `entropy_coef`   | Weight of entropy bonus in total loss                       |

### Evaluation

| Parameter        | Value    | Description                                                      |
|------------------|----------|------------------------------------------------------------------|
| Eval frequency   | 10 epochs| Evaluate every 10 training epochs                                |
| Eval episodes    | 10       | Average score over 10 episodes per evaluation                    |
| Solve threshold  | 20       | Training stops when best score reaches 20 (Pong max is 21)      |
