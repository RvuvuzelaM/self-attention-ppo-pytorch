import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MultiHeadSpatialAttention(nn.Module):
    """Multi-head spatial self-attention (Zambaldi et al.) with residual + LayerNorm."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5

        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # Project and reshape to (B, heads, N, head_dim)
        q = self.query(x).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        k = self.key(x).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        v = self.value(x).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)  # (B, heads, N, head_dim)

        # Concat heads and project
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.out_proj(out)

        # Residual + LayerNorm (per spatial position)
        out = x + out
        out = out.view(B, C, N).permute(0, 2, 1)  # (B, N, C)
        out = self.norm(out)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            MultiHeadSpatialAttention(32, num_heads=4),
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
