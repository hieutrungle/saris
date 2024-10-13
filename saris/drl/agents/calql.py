from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TanhTransform, TransformedDistribution
import numpy as np


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


def init_module_weights(module: torch.nn.Module, orthogonal_init: bool = False):
    if isinstance(module, nn.Linear):
        if orthogonal_init:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1e-2)


class ReparameterizedTanhGaussian(nn.Module):
    def __init__(
        self,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        no_tanh: bool = False,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(
        self, mean: torch.Tensor, log_std: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)

        return action_sample, log_prob


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


# class TanhGaussianPolicy(nn.Module):
#     def __init__(
#         self,
#         state_dim: int,
#         action_dim: int,
#         action_scale: float = 1.0,
#         log_std_multiplier: float = 1.0,
#         log_std_offset: float = -1.0,
#         orthogonal_init: bool = False,
#         no_tanh: bool = False,
#     ):
#         super().__init__()
#         self.observation_dim = state_dim
#         self.action_dim = action_dim
#         self.action_scale = action_scale
#         self.orthogonal_init = orthogonal_init
#         self.no_tanh = no_tanh

#         self.fourier = Fourier(state_dim, 256)
#         self.base_network = nn.Sequential(
#             nn.Linear(256, 256),
#             nn.GELU(),
#             nn.Linear(256, 256),
#             nn.LayerNorm(256),
#             nn.GELU(),
#             nn.Linear(256, 256),
#             nn.GELU(),
#             nn.Linear(256, 2 * action_dim),
#         )

#         if orthogonal_init:
#             self.base_network.apply(lambda m: init_module_weights(m, True))
#         else:
#             init_module_weights(self.base_network[-1], False)

#         self.log_std_multiplier = Scalar(log_std_multiplier)
#         self.log_std_offset = Scalar(log_std_offset)
#         self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

#     def log_prob(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
#         if actions.ndim == 3:
#             observations = extend_and_repeat(observations, 1, actions.shape[1])
#         fourier = self.fourier(observations)
#         base_network_output = self.base_network(fourier)
#         mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
#         log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
#         log_probs = self.tanh_gaussian.log_prob(mean, log_std, actions)
#         return log_probs

#     def forward(
#         self,
#         observations: torch.Tensor,
#         deterministic: bool = False,
#         repeat: bool = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         if repeat is not None:
#             observations = extend_and_repeat(observations, 1, repeat)
#         fourier = self.fourier(observations)
#         base_network_output = self.base_network(fourier)
#         mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
#         log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
#         actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
#         actions: torch.Tensor = self.action_scale * actions

#         batch_size = actions.shape[0]
#         if actions.ndim == 3:
#             n_actions = actions.shape[1]
#             actions = actions.reshape(batch_size, n_actions, -1, 3)
#             actions[..., 1] = torch.deg2rad(actions[..., 1])
#             actions[..., 2] = torch.deg2rad(actions[..., 2])
#             actions = actions.reshape(batch_size, n_actions, -1)
#         elif actions.ndim == 2:
#             actions = actions.reshape(batch_size, -1, 3)
#             actions[..., 1] = torch.deg2rad(actions[..., 1])
#             actions[..., 2] = torch.deg2rad(actions[..., 2])
#             actions = actions.reshape(batch_size, -1)
#         return actions, log_probs

#     @torch.no_grad()
#     def act(self, ob: np.ndarray):
#         with torch.no_grad():
#             actions, _ = self(ob, not self.training)
#         return actions


class TanhGaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_scale: float = 1.0,
        log_std_multiplier: float = 1.0,
        log_std_offset: float = -1.0,
        orthogonal_init: bool = False,
        no_tanh: bool = False,
    ):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.pos_dim = 12
        self.angle_dim = self.observation_dim - self.pos_dim
        ff_dim = 128

        # positions
        self.pos_fourier = Fourier(self.pos_dim, self.angle_dim)
        self.pos_embedding = nn.Linear(1, ff_dim, bias=False)
        pos_layers = [TransformerBlock(ff_dim, 4)]
        self.pos_network = nn.Sequential(*pos_layers)
        self.pos_down = nn.Linear(ff_dim, 1)

        # angles
        self.angle_embedding = nn.Linear(1, ff_dim, bias=False)
        angle_layers = [
            nn.Linear(ff_dim, ff_dim),
            nn.GELU(),
            TransformerBlock(ff_dim, 4),
            TransformerBlock(ff_dim, 4),
        ]
        self.angle_network = nn.Sequential(*angle_layers)
        self.angle_down = nn.Linear(ff_dim, 1)

        self.connect_layer = nn.Linear(self.angle_dim * 2, 256)

        combine_layers = [
            MLPBlock(256, 256),
            nn.Linear(256, 2 * action_dim),
        ]
        self.combine_network = nn.Sequential(*combine_layers)

        self.pos_network.apply(lambda m: init_module_weights(m, True))
        self.angle_network.apply(lambda m: init_module_weights(m, True))
        self.connect_layer.apply(lambda m: init_module_weights(m, True))
        self.combine_network.apply(lambda m: init_module_weights(m, True))

        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
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
        network_output = self.combine_network(pos_angles)

        mean, log_std = torch.split(network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        log_probs = self.tanh_gaussian.log_prob(mean, log_std, actions)
        return log_probs

    def forward(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
        repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)

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
        network_output = self.combine_network(pos_angles)

        mean, log_std = torch.split(network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        actions: torch.Tensor = self.action_scale * actions

        batch_size = actions.shape[0]
        if actions.ndim == 3:
            n_actions = actions.shape[1]
            actions = actions.reshape(batch_size, n_actions, -1, 3)
            actions[..., 1] = torch.deg2rad(actions[..., 1])
            actions[..., 2] = torch.deg2rad(actions[..., 2])
            actions = actions.reshape(batch_size, n_actions, -1)
        elif actions.ndim == 2:
            actions = actions.reshape(batch_size, -1, 3)
            actions[..., 1] = torch.deg2rad(actions[..., 1])
            actions[..., 2] = torch.deg2rad(actions[..., 2])
            actions = actions.reshape(batch_size, -1)
        return actions, log_probs

    @torch.no_grad()
    def act(self, obs: np.ndarray):
        with torch.no_grad():
            actions, _ = self(obs, not self.training)
        return actions


class FullyConnectedQFunction(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        orthogonal_init: bool = False,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.orthogonal_init = orthogonal_init

        self.pos_dim = 12
        self.angle_dim = self.observation_dim - self.pos_dim
        ff_dim = 128

        # positions
        self.pos_fourier = Fourier(self.pos_dim, self.angle_dim)
        self.pos_embedding = nn.Linear(1, ff_dim, bias=False)
        pos_layers = [TransformerBlock(ff_dim, 4)]
        self.pos_network = nn.Sequential(*pos_layers)
        self.pos_down = nn.Linear(ff_dim, 1)

        # angles
        self.angle_embedding = nn.Linear(1, ff_dim, bias=False)
        angle_layers = [
            nn.Linear(ff_dim, ff_dim),
            nn.GELU(),
            TransformerBlock(ff_dim, 4),
            TransformerBlock(ff_dim, 4),
        ]
        self.angle_network = nn.Sequential(*angle_layers)
        self.angle_down = nn.Linear(ff_dim, 1)

        self.connect_layer = nn.Linear(self.angle_dim * 2, 256)

        # action
        action_layers = [
            nn.Linear(action_dim, 256),
            nn.GELU(),
            MLPBlock(256, 256),
        ]
        self.action_network = nn.Sequential(*action_layers)

        self.activation = nn.GELU()
        self.combine_layer = nn.Linear(256, 1)

        self.pos_network.apply(lambda m: init_module_weights(m, True))
        self.angle_network.apply(lambda m: init_module_weights(m, True))
        self.connect_layer.apply(lambda m: init_module_weights(m, True))
        self.action_network.apply(lambda m: init_module_weights(m, True))
        self.combine_layer.apply(lambda m: init_module_weights(m, True))

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])

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
        combined = self.activation(pos_angles + action)
        q_values = self.combine_layer(combined)
        q_values = torch.squeeze(q_values, dim=-1)

        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values


# class FullyConnectedQFunction(nn.Module):
#     def __init__(
#         self,
#         observation_dim: int,
#         action_dim: int,
#         orthogonal_init: bool = False,
#         n_hidden_layers: int = 2,
#     ):
#         super().__init__()
#         self.observation_dim = observation_dim
#         self.action_dim = action_dim
#         self.orthogonal_init = orthogonal_init

#         self.pos_dim = 12
#         self.angle_dim = self.observation_dim - self.pos_dim

#         self.pos_fourier = Fourier(self.pos_dim, 256)
#         pos_layers = [
#             TransformerBlock(256, 4),
#         ]
#         self.pos_network = nn.Sequential(*pos_layers)

#         angle_layers = [
#             nn.Linear(self.angle_dim, 256),
#             nn.GELU(),
#             TransformerBlock(256, 4),
#             TransformerBlock(256, 4),
#         ]
#         self.angle_network = nn.Sequential(*angle_layers)
#         self.connect_layer = nn.Linear(256, 1)

#         action_layers = [
#             nn.Linear(action_dim, 256),
#             nn.GELU(),
#             MLPBlock(256, 256),
#         ]
#         self.action_network = nn.Sequential(*action_layers)

#         self.activation = nn.GELU()
#         self.combine_layer = nn.Linear(256, 1)

#         # angle_layers = [
#         #     nn.Linear()
#         # ]

#         # layers = [
#         #     nn.Linear(observation_dim + action_dim, 256),
#         #     nn.GELU(),
#         # ]
#         # for _ in range(n_hidden_layers - 1):
#         #     layers.append(nn.LayerNorm(256))
#         #     layers.append(nn.Linear(256, 256))
#         #     layers.append(nn.GELU())
#         # layers.append(nn.Linear(256, 1))

#         # self.network = nn.Sequential(*layers)
#         self.pos_network.apply(lambda m: init_module_weights(m, True))
#         self.angle_network.apply(lambda m: init_module_weights(m, True))
#         self.connect_layer.apply(lambda m: init_module_weights(m, True))
#         self.action_network.apply(lambda m: init_module_weights(m, True))
#         self.combine_layer.apply(lambda m: init_module_weights(m, True))
#         # if orthogonal_init:
#         #     self.pos_network.apply(lambda m: init_module_weights(m, True))

#         # else:
#         #     init_module_weights(self.network[-1], False)

#     def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
#         multiple_actions = False
#         batch_size = observations.shape[0]
#         if actions.ndim == 3 and observations.ndim == 2:
#             multiple_actions = True
#             observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
#                 -1, observations.shape[-1]
#             )
#             actions = actions.reshape(-1, actions.shape[-1])
#         input_tensor = torch.cat([observations, actions], dim=-1)
#         q_values = torch.squeeze(self.network(input_tensor), dim=-1)
#         if multiple_actions:
#             q_values = q_values.reshape(batch_size, -1)
#         return q_values


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
