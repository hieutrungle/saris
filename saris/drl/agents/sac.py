import torch.nn as nn
import torch
from typing import Tuple, Sequence, Union
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import gymnasium as gym


class SoftQNetwork(nn.Module):
    def __init__(self, ob_shape: Sequence[int], ac_shape: Sequence[int]):
        super().__init__()
        self.fc1 = nn.Linear(np.prod(ob_shape) + np.prod(ac_shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(
        self,
        ob_shape: Sequence[int],
        ac_shape: Sequence[int],
        action_high: Union[float, Sequence[float]],
        action_low: Union[float, Sequence[float]],
        log_std_min: float = -5,
        log_std_max: float = 2,
    ):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(np.prod(ob_shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(ac_shape))
        self.fc_logstd = nn.Linear(256, np.prod(ac_shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (action_high - action_low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (action_high + action_low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        # From SpinUp / Denis Yarats
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        return mean, log_std

    def get_actions(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class Agent(nn.Module):
    def __init__(self, ob_space: gym.spaces.Dict, ac_space: gym.spaces.Box):
        super().__init__()
        self.actor = DictActor(ob_space, ac_space)
        self.qf1 = DictSoftQNetwork(ob_space, ac_space)
        self.qf2 = DictSoftQNetwork(ob_space, ac_space)
        self.target_qf1 = DictSoftQNetwork(ob_space, ac_space)
        self.target_qf2 = DictSoftQNetwork(ob_space, ac_space)
        self.target_qf1.load_state_dict(self.qf1.state_dict())
        self.target_qf2.load_state_dict(self.qf2.state_dict())

    def get_actions(self, obs: dict[str, torch.Tensor]):
        _, _, mean = self.actor.get_actions(obs)
        return mean

    def get_trainable_actions(self, obs: dict[str, torch.Tensor]):
        action, log_prob, mean = self.actor.get_actions(obs)
        return action, log_prob, mean

    def get_q_values(
        self, obs: dict[str, torch.Tensor], a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q1 = self.qf1(obs, a)
        q2 = self.qf2(obs, a)
        return q1, q2

    def get_target_q_values(
        self, obs: dict[str, torch.Tensor], a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q1 = self.target_qf1(obs, a)
        q2 = self.target_qf2(obs, a)
        return q1, q2

    def update_target(self, tau: float):
        for target_param, param in zip(self.target_qf1.parameters(), self.qf1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_qf2.parameters(), self.qf2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class DictSoftQNetwork(nn.Module):
    def __init__(self, ob_space: gym.spaces.Dict, ac_space: gym.spaces.Box):
        super().__init__()
        assert isinstance(ob_space, gym.spaces.Dict)
        assert isinstance(ac_space, gym.spaces.Box)
        angle_shape = ob_space["angles"].shape
        gain_shape = ob_space["gain"].shape

        self.angle_fc1 = nn.Linear(np.prod(angle_shape), 256)
        self.angle_fc2 = nn.Linear(256, 256)
        self.angle_layer_norm = nn.LayerNorm(256)

        self.gain_fc1 = nn.Linear(np.prod(gain_shape), 64)
        self.gain_fc2 = nn.Linear(64, 256)
        self.gain_layer_norm = nn.LayerNorm(256)

        self.fc1 = nn.Linear(256 + np.prod(ac_space.shape), 256)
        self.layer_norm1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, obs, a):

        angle = obs["angles"]
        angle = F.gelu(self.angle_fc1(angle))
        angle = F.gelu(self.angle_fc2(angle))
        angle = self.angle_layer_norm(angle)

        gain = obs["gain"]
        gain = F.gelu(self.gain_fc1(gain))
        gain = F.gelu(self.gain_fc2(gain))
        gain = self.gain_layer_norm(gain)

        x = angle + gain
        x = torch.cat([x, a], 1)
        x = F.gelu(self.fc1(x))
        x = self.layer_norm1(x)
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        return x


class DictActor(nn.Module):
    def __init__(
        self,
        ob_space: gym.spaces.Dict,
        ac_space: gym.spaces.Box,
        log_std_min: float = -5,
        log_std_max: float = 2,
    ):
        super().__init__()
        assert isinstance(ob_space, gym.spaces.Dict)
        assert isinstance(ac_space, gym.spaces.Box)
        angle_shape = ob_space["angles"].shape
        gain_shape = ob_space["gain"].shape

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.angle_fc1 = nn.Linear(np.prod(angle_shape), 256)
        self.angle_fc2 = nn.Linear(256, 256)
        self.angle_layer_norm = nn.LayerNorm(256)

        self.gain_fc1 = nn.Linear(np.prod(gain_shape), 64)
        self.gain_fc2 = nn.Linear(64, 256)
        self.gain_layer_norm = nn.LayerNorm(256)

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(ac_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(ac_space.shape))

        # action rescaling
        action_high = ac_space.high
        action_low = ac_space.low
        self.register_buffer(
            "action_scale",
            torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32),
        )

    def forward(self, obs: dict):

        angle = obs["angles"]
        angle = F.gelu(self.angle_fc1(angle))
        angle = F.gelu(self.angle_fc2(angle))
        angle = self.angle_layer_norm(angle)

        gain = obs["gain"]
        gain = F.gelu(self.gain_fc1(gain))
        gain = F.gelu(self.gain_fc2(gain))
        gain = self.gain_layer_norm(gain)

        x = angle + gain
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        # From SpinUp / Denis Yarats
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        return mean, log_std

    def get_actions(self, obs: dict):
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        actions = y_t * self.action_scale + self.action_bias
        log_probs = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_probs -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return actions, log_probs, mean
