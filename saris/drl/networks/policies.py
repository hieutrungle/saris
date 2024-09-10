import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from saris.drl.networks.network_utils import Activation, _str_to_activation
from typing import Sequence
from saris.drl.networks.mlp import MLP, ResidualMLP
from saris.drl.networks.common_blocks import Fourier
from typing import Tuple

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class BasePolicy(nn.Module):
    def __init__(self):
        super(BasePolicy, self).__init__()

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute an action distribution for a given observation:
            {mean, log_std} if Gaussian, {mean, _} if deterministic.

        Return:
            means: (batch_size, action_dim)
            log_stds or None: (batch_size, action_dim)
        """
        raise NotImplementedError

    @staticmethod
    def sample(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute an action for a given observation.

        Return:
            actions: (batch_size, action_dim)
            log_probs: (batch_size, 1)
            means: (batch_size, action_dim)
        """
        raise NotImplementedError


class GaussianPolicy(BasePolicy):
    def __init__(
        self,
        num_observations: int,
        num_actions: int,
        hidden_sizes: Sequence[int],
        activation: Activation,
        action_space=None,
    ):
        super().__init__()
        if isinstance(activation, str):
            activation = _str_to_activation[activation]

        self.fourier = Fourier(num_observations, hidden_sizes[0] // 2)
        self.mlp = MLP(
            in_features=hidden_sizes[0],
            out_features=hidden_sizes[-1],
            features=hidden_sizes[1:-1],
            activation=activation,
        )

        self.mean_linear = nn.Linear(hidden_sizes[-1], num_actions)
        self.log_std_linear = nn.Linear(hidden_sizes[-1], num_actions)
        self.epsilon = 1e-6

        # self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        mixed = self.fourier(observations)
        mixed = self.mlp(mixed)
        means = self.mean_linear(mixed)
        log_stds = self.log_std_linear(mixed)
        log_stds = torch.clamp(log_stds, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return means, log_stds

    def sample(
        self,
        observations: torch.Tensor,
        num_samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute an action for a given observation.

        Return:
            actions: (num_samples, batch_size, action_dim)
            log_probs: (batch_size, 1)
            means: (batch_size, action_dim)
        """

        means, log_stds = self.forward(observations)
        stds = torch.exp(log_stds)
        normal = Normal(means, stds)
        # for reparameterization trick (means + std * N(0,1))
        x_t = normal.rsample(sample_shape=(num_samples,))
        y_t = torch.tanh(x_t)
        actions = y_t * self.action_scale + self.action_bias

        # Enforcing Action Bound
        log_probs = normal.log_prob(x_t)
        log_probs -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_probs = torch.sum(log_probs, dim=-1, keepdim=True)
        log_probs = torch.mean(log_probs, dim=0)

        means = torch.tanh(means) * self.action_scale + self.action_bias

        return actions, log_probs, means

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(BasePolicy):
    def __init__(
        self,
        num_inputs: int,
        num_actions: int,
        hidden_sizes: Sequence[int],
        activation: Activation,
        action_space=None,
    ):
        super().__init__()
        if isinstance(activation, str):
            activation = _str_to_activation[activation]

        self.fourier = Fourier(num_inputs, hidden_sizes[0] // 2)
        self.mlp = MLP(
            in_features=hidden_sizes[0],
            out_features=hidden_sizes[-1] // 2,
            features=hidden_sizes[1:-1],
            activation=activation,
        )

        self.mean = nn.Linear(hidden_sizes[-1], num_actions)
        self.noise = torch.Tensor(num_actions)

        # self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.0
            self.action_bias = 0.0
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mixed = self.fourier(observations)
        mixed = self.mlp(mixed)
        means = self.mean_linear(mixed)
        means = torch.tanh(means) * self.action_scale + self.action_bias
        return means, None

    def sample(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        means = self.forward(observations)
        noise = self.noise.normal_(0.0, std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        actions = means + noise
        return actions, torch.tensor(0.0), means

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
