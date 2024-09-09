import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from saris.drl.networks.network_utils import Activation, _str_to_activation
from typing import Sequence
from saris.drl.networks.mlp import MLP, ResidualMLP
from saris.drl.networks.common_blocks import Fourier

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class GaussianPolicy(nn.Module):
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

        self.mean_linear = nn.Linear(hidden_sizes[-1], num_actions)
        self.log_std_linear = nn.Linear(hidden_sizes[-1], num_actions)

        self.apply(weights_init_)

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

    def forward(self, observations: torch.Tensor, train: bool = False) -> torch.Tensor:
        mixed = self.fourier(observations)
        mixed = self.mlp(mixed)
        means = self.mean_linear(mixed)
        log_stds = self.log_std_linear(mixed)
        log_stds = torch.clamp(log_stds, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return means, log_stds

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

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

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0.0, std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.0), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
