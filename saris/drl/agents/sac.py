from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TanhTransform, TransformedDistribution
import numpy as np


def init_module_weights(module: torch.nn.Module, orthogonal_init: bool = False):
    if isinstance(module, nn.Linear):
        if orthogonal_init:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1e-2)


class Fourier(nn.Module):
    """Fourier features for encoding the input signal."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        assert out_features % 2 == 0, "The number of output features must be even."
        self.fourier = nn.Linear(in_features, out_features // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fourier(x)
        x = 2 * np.pi * x
        x = torch.cat([torch.sin(x), torch.cos(x)], axis=-1)
        return x


class Actor(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        action_scale: float = 1.0,
        log_std_multiplier: float = 1.0,
        log_std_offset: float = 0.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.action_scale = action_scale
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.pos_dim = 12
        self.angle_dim = self.ob_dim - self.pos_dim
        ff_dim = 128

        # positions
        self.pos_fourier = Fourier(self.pos_dim, self.angle_dim)
        self.pos_embedding = nn.Linear(1, ff_dim, bias=False)
        pos_layers = [
            TransformerBlock(ff_dim, 4),
            TransformerBlock(ff_dim, 4),
        ]
        self.pos_network = nn.Sequential(*pos_layers)
        self.pos_down = nn.Linear(ff_dim, 1)

        # angles
        self.angle_embedding = nn.Linear(1, ff_dim, bias=False)
        angle_layers = [
            nn.Linear(ff_dim, ff_dim),
            nn.GELU(),
            TransformerBlock(ff_dim, 4),
            TransformerBlock(ff_dim, 4),
            TransformerBlock(ff_dim, 4),
        ]
        self.angle_network = nn.Sequential(*angle_layers)
        self.angle_down = nn.Linear(ff_dim, 1)

        self.connect_layer = nn.Linear(self.angle_dim * 2, ff_dim)
        self.combine_network = nn.Sequential(MLPBlock(ff_dim, ff_dim), MLPBlock(ff_dim, ff_dim))
        self.fc_mean = nn.Linear(ff_dim, self.ac_dim)
        self.fc_log_std = nn.Linear(ff_dim, self.ac_dim)

        self.pos_network.apply(lambda m: init_module_weights(m, True))
        self.angle_network.apply(lambda m: init_module_weights(m, True))
        self.connect_layer.apply(lambda m: init_module_weights(m, True))
        self.combine_network.apply(lambda m: init_module_weights(m, True))

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # positions
        pos = observations[..., self.angle_dim :]
        pos = self.pos_fourier(pos)
        pos = pos.unsqueeze(-1)
        pos = self.pos_embedding(pos)
        pos_shape = pos.shape
        pos = pos.reshape(-1, pos_shape[-2], pos_shape[-1])
        pos = self.pos_network(pos)
        pos = pos.reshape(pos_shape)
        pos = self.pos_down(pos)
        pos = pos.squeeze(-1)

        # angles
        angles = observations[..., : self.angle_dim]
        angles = angles.unsqueeze(-1)
        angles = self.angle_embedding(angles)
        angle_shape = angles.shape
        angles = angles.reshape(-1, angle_shape[-2], angle_shape[-1])
        angles = self.angle_network(angles)
        angles = angles.reshape(angle_shape)
        angles = self.angle_down(angles)
        angles = angles.squeeze(-1)

        # connect
        pos_angles = torch.cat([pos, angles], dim=-1)
        pos_angles = self.connect_layer(pos_angles)

        # combine
        combined = self.combine_network(pos_angles)

        # mean and log_std
        mean = self.fc_mean(combined)
        log_std = self.fc_log_std(combined)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        return mean, log_std

    def get_action(self, obs: torch.Tensor):

        # action
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        actions = self.action_scale * y_t

        # log_prob
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale

        actions = self.modify_action(actions)
        mean = self.modify_action(mean)

        return actions, log_prob, mean

    def modify_action(self, acts: torch.tensor):
        action_shape = acts.shape
        last_dim = acts.shape[-1]
        all_but_last_dim = acts.shape[:-1]
        acts = acts.view(*all_but_last_dim, last_dim // 3, 3)
        acts[..., 0] = torch.div(acts[..., 0], 3.0)
        acts[..., 1:] = torch.deg2rad(acts[..., 1:])
        acts = acts.view(*action_shape)
        return acts


class SoftQNetwork(nn.Module):
    def __init__(self, ob_dim: int, ac_dim: int):
        super().__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim

        self.pos_dim = 12
        self.angle_dim = self.ob_dim - self.pos_dim
        ff_dim = 128

        # positions
        self.pos_fourier = Fourier(self.pos_dim, self.angle_dim)
        self.pos_embedding = nn.Linear(1, ff_dim, bias=False)
        pos_layers = [
            TransformerBlock(ff_dim, 4),
            TransformerBlock(ff_dim, 4),
        ]
        self.pos_network = nn.Sequential(*pos_layers)
        self.pos_down = nn.Linear(ff_dim, 1)

        # angles
        self.angle_embedding = nn.Linear(1, ff_dim, bias=False)
        angle_layers = [
            nn.Linear(ff_dim, ff_dim),
            nn.GELU(),
            TransformerBlock(ff_dim, 4),
            TransformerBlock(ff_dim, 4),
            TransformerBlock(ff_dim, 4),
        ]
        self.angle_network = nn.Sequential(*angle_layers)
        self.angle_down = nn.Linear(ff_dim, 1)

        self.connect_layer = nn.Linear(self.angle_dim * 2, ff_dim)

        # action
        action_layers = [
            nn.Linear(ac_dim, ff_dim),
            nn.GELU(),
            MLPBlock(ff_dim, ff_dim),
        ]
        self.action_network = nn.Sequential(*action_layers)

        self.combine_network = nn.Sequential(MLPBlock(ff_dim, ff_dim), MLPBlock(ff_dim, ff_dim))
        self.activation = nn.GELU()
        self.combine_layer = nn.Linear(ff_dim, 1)

        self.pos_network.apply(lambda m: init_module_weights(m, True))
        self.angle_network.apply(lambda m: init_module_weights(m, True))
        self.connect_layer.apply(lambda m: init_module_weights(m, True))
        self.action_network.apply(lambda m: init_module_weights(m, True))
        self.combine_network.apply(lambda m: init_module_weights(m, True))
        self.combine_layer.apply(lambda m: init_module_weights(m, True))

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:

        # positions
        pos = observations[..., self.angle_dim :]
        pos = self.pos_fourier(pos)
        pos = pos.unsqueeze(-1)
        pos = self.pos_embedding(pos)
        pos = self.pos_network(pos)
        pos = self.pos_down(pos)
        pos = pos.squeeze(-1)

        # angles
        angles = observations[..., : self.angle_dim]
        angles = angles.unsqueeze(-1)
        angles = self.angle_embedding(angles)
        angles = self.angle_network(angles)
        angles = self.angle_down(angles)
        angles = angles.squeeze(-1)

        # connect
        pos_angles = torch.cat([pos, angles], dim=-1)
        pos_angles = self.connect_layer(pos_angles)

        # action
        action = self.action_network(actions)

        # combine
        combined = self.combine_network(pos_angles + action)
        combined = self.activation(combined)
        q_values = self.combine_layer(combined)

        return q_values


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = MLPBlock(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_output, _ = self.attention(x, x, x)
        out1 = self.layer_norm1(x + self.dropout(attention_output))
        out2 = self.mlp(out1)
        return out2


class MLPBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, multiplier: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features * multiplier),
            nn.GELU(),
            nn.Linear(out_features * multiplier, out_features),
        )
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(self.block(x))
