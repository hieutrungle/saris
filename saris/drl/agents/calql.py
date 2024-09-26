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

        self.base_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
        )

        if orthogonal_init:
            self.base_network.apply(lambda m: init_module_weights(m, True))
        else:
            init_module_weights(self.base_network[-1], False)

        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        _, log_probs = self.tanh_gaussian(mean, log_std, False)
        return log_probs

    def forward(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
        repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        return self.action_scale * actions, log_probs

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        with torch.no_grad():
            actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()


class FullyConnectedQFunction(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        orthogonal_init: bool = False,
        n_hidden_layers: int = 2,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.orthogonal_init = orthogonal_init

        layers = [
            nn.Linear(observation_dim + action_dim, 256),
            nn.GELU(),
        ]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.LayerNorm(256))
            layers.append(nn.Linear(256, 256))
            layers.append(nn.GELU())
        layers.append(nn.Linear(256, 1))

        self.network = nn.Sequential(*layers)
        if orthogonal_init:
            self.network.apply(lambda m: init_module_weights(m, True))
        else:
            init_module_weights(self.network[-1], False)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])
        input_tensor = torch.cat([observations, actions], dim=-1)
        q_values = torch.squeeze(self.network(input_tensor), dim=-1)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant
