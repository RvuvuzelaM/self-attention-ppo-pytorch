import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class SpatialSelfAttention(nn.Module):
    """Single-head spatial self-attention via 1x1 convolutions with residual."""

    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.scale = channels**-0.5

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        q = self.query(x).view(B, C, N).permute(0, 2, 1)  # (B, N, C)
        k = self.key(x).view(B, C, N)  # (B, C, N)
        v = self.value(x).view(B, C, N).permute(0, 2, 1)  # (B, N, C)

        attn = torch.bmm(q, k) * self.scale  # (B, N, N)
        attn = attn.softmax(dim=-1)
        out = torch.bmm(attn, v)  # (B, N, C)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return x + out


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            SpatialSelfAttention(32),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
