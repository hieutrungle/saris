from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TanhTransform, TransformedDistribution
import numpy as np
from gymnasium import spaces, vector


def init_module_weights(module: torch.nn.Module, orthogonal_init: bool = False):
    if isinstance(module, nn.Linear):
        if orthogonal_init:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1e-2)


class Embedder(nn.Module):

    def __init__(
        self,
        input_dims,
        include_input=False,
        min_freq_exp=0.0,
        max_freq_exp=4.0,
        num_freqs=6,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.min_freq_exp = min_freq_exp
        self.max_freq_exp = max_freq_exp
        self.num_freqs = num_freqs
        self.out_dim = self.input_dims * self.num_freqs * 2

    def forward(self, in_tensor):

        scaled_in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        freqs = 2 ** torch.linspace(
            self.min_freq_exp, self.max_freq_exp, self.num_freqs, device=in_tensor.device
        )

        # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_in_tensor[..., None] * freqs
        # [..., "input_dim" * "num_scales"]
        scaled_inputs = scaled_inputs.reshape(*scaled_inputs.shape[:-2], -1)

        encoded_inputs = torch.sin(
            torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1)
        )
        #     )
        return encoded_inputs


class QNetwork(nn.Module):
    def __init__(
        self,
        ob_space: spaces.Tuple,
        ac_space: spaces.Box,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.real_channel_shape = ob_space[0].shape
        self.imag_channel_shape = ob_space[1].shape
        self.angle_shape = ob_space[2].shape
        self.position_shape = ob_space[3].shape
        self.ac_shape = ac_space.shape

        self.real_channel_dim = np.prod(self.real_channel_shape)
        self.imag_channel_dim = np.prod(self.imag_channel_shape)
        self.angle_dim = np.prod(self.angle_shape)
        self.position_dim = np.prod(self.position_shape)

        self.real_start = 0
        self.imag_start = self.real_channel_dim
        self.angle_start = self.imag_start + self.imag_channel_dim
        self.pos_start = self.angle_start + self.angle_dim

        ff_dim = 128

        # positions
        self.pos_embed = Embedder(np.prod(self.position_shape), num_freqs=5)
        pos_out_dim = self.pos_embed.out_dim
        self.pos_layers = [
            nn.Linear(pos_out_dim, ff_dim, device=device),
            nn.GELU(),
            MLPBlock(ff_dim, ff_dim, device=device),
        ]
        self.pos_network = nn.Sequential(*self.pos_layers)

        # angles
        self.angle_layers = [
            nn.Linear(self.angle_dim, ff_dim, device=device),
            nn.GELU(),
            MLPBlock(ff_dim, ff_dim, device=device),
            MLPBlock(ff_dim, ff_dim, device=device),
        ]
        self.angle_network = nn.Sequential(*self.angle_layers)

        # channels
        self.real_channel_layers = [
            nn.Linear(np.prod(self.real_channel_shape), ff_dim * 2, device=device),
            nn.GELU(),
            MLPBlock(ff_dim * 2, ff_dim, device=device),
            MLPBlock(ff_dim, ff_dim, device=device),
        ]
        self.real_channel_network = nn.Sequential(*self.real_channel_layers)

        self.imag_channel_layers = [
            nn.Linear(np.prod(self.imag_channel_shape), ff_dim * 2, device=device),
            nn.GELU(),
            MLPBlock(ff_dim * 2, ff_dim, device=device),
            MLPBlock(ff_dim, ff_dim, device=device),
        ]
        self.imag_channel_network = nn.Sequential(*self.imag_channel_layers)

        self.chanel_combine_layer = [nn.Linear(ff_dim * 2, ff_dim, device=device), nn.GELU()]
        self.channel_connect_network = nn.Sequential(*self.chanel_combine_layer)

        # Connect channel + pos + angles
        self.connect_layer = [nn.Linear(ff_dim * 3, ff_dim, device=device), nn.GELU()]
        self.connect_network = nn.Sequential(*self.connect_layer)

        # action
        action_layers = [
            nn.Linear(np.prod(ac_space.shape), ff_dim, device=device),
            nn.GELU(),
            MLPBlock(ff_dim, ff_dim, device=device),
        ]
        self.action_network = nn.Sequential(*action_layers)

        # Combine all
        self.combine_network = nn.Sequential(
            nn.Linear(ff_dim * 2, ff_dim, device=device), nn.GELU()
        )
        self.combine_layer = nn.Linear(ff_dim, 1, device=device)

    def forward(self, obs, acs):
        batch_size = obs.shape[0]
        real_channel = obs[:batch_size, self.real_start : self.imag_start].reshape(
            batch_size, *self.real_channel_shape
        )
        imag_channel = obs[:batch_size, self.imag_start : self.angle_start].reshape(
            batch_size, *self.imag_channel_shape
        )
        angles = angles = obs[:batch_size, self.angle_start : self.pos_start].reshape(
            batch_size, *self.angle_shape
        )
        pos = obs[:batch_size, self.pos_start :].reshape(batch_size, *self.position_shape)

        # positions
        pos = self.pos_embed(pos)
        pos = self.pos_network(pos)

        # angles
        angles = obs[
            :batch_size, self.real_channel_dim + self.imag_channel_dim + self.position_dim :
        ].reshape(batch_size, *self.angle_shape)
        angles = self.angle_network(angles)

        # channels
        real_channel = real_channel.reshape(real_channel.shape[0], -1)
        real_channel = self.real_channel_network(real_channel)

        imag_channel = imag_channel.reshape(imag_channel.shape[0], -1)
        imag_channel = self.imag_channel_network(imag_channel)

        channel = torch.cat([real_channel, imag_channel], dim=-1)
        channel = self.channel_connect_network(channel)

        # connect
        combined = torch.cat([channel, angles, pos], dim=-1)
        combined = self.connect_network(combined)

        # action
        action = self.action_network(acs)

        # combine
        ob_ac = self.combine_network(torch.cat([combined, action], dim=-1))
        q_values = self.combine_layer(ob_ac)
        return q_values


class Actor(nn.Module):
    def __init__(
        self,
        ob_space: spaces.Tuple,
        ac_space: spaces.Box,
        envs: vector.VectorEnv,
        exploration_noise: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.real_channel_shape = ob_space[0].shape
        self.imag_channel_shape = ob_space[1].shape
        self.angle_shape = ob_space[2].shape
        self.position_shape = ob_space[3].shape
        self.ac_shape = ac_space.shape

        self.real_channel_dim = np.prod(self.real_channel_shape)
        self.imag_channel_dim = np.prod(self.imag_channel_shape)
        self.angle_dim = np.prod(self.angle_shape)
        self.position_dim = np.prod(self.position_shape)

        self.real_start = 0
        self.imag_start = self.real_channel_dim
        self.angle_start = self.imag_start + self.imag_channel_dim
        self.pos_start = self.angle_start + self.angle_dim

        ff_dim = 128

        # positions
        self.pos_embed = Embedder(np.prod(self.position_shape), num_freqs=5)
        pos_out_dim = self.pos_embed.out_dim
        self.pos_layers = [
            nn.Linear(pos_out_dim, ff_dim, device=device),
            nn.GELU(),
            MLPBlock(ff_dim, ff_dim, device=device),
        ]
        self.pos_network = nn.Sequential(*self.pos_layers)

        # angles
        self.angle_layers = [
            nn.Linear(self.angle_dim, ff_dim, device=device),
            nn.GELU(),
            MLPBlock(ff_dim, ff_dim, device=device),
            MLPBlock(ff_dim, ff_dim, device=device),
        ]
        self.angle_network = nn.Sequential(*self.angle_layers)

        # channels
        self.real_channel_layers = [
            nn.Linear(np.prod(self.real_channel_shape), ff_dim * 2, device=device),
            nn.GELU(),
            MLPBlock(ff_dim * 2, ff_dim, device=device),
            MLPBlock(ff_dim, ff_dim, device=device),
        ]
        self.real_channel_network = nn.Sequential(*self.real_channel_layers)

        self.imag_channel_layers = [
            nn.Linear(np.prod(self.imag_channel_shape), ff_dim * 2, device=device),
            nn.GELU(),
            MLPBlock(ff_dim * 2, ff_dim, device=device),
            MLPBlock(ff_dim, ff_dim, device=device),
        ]
        self.imag_channel_network = nn.Sequential(*self.imag_channel_layers)

        self.chanel_combine_layer = [nn.Linear(ff_dim * 2, ff_dim, device=device), nn.GELU()]
        self.channel_connect_network = nn.Sequential(*self.chanel_combine_layer)

        # Connect channel + pos + angles
        self.connect_layer = [nn.Linear(ff_dim * 3, ff_dim, device=device), nn.GELU()]
        self.connect_network = nn.Sequential(*self.connect_layer)

        self.fc_mean = nn.Linear(ff_dim, np.prod(self.ac_shape), device=device)

        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (envs.single_action_space.high - envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
                device=device,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (envs.single_action_space.high + envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
                device=device,
            ),
        )
        self.register_buffer("exploration_noise", torch.as_tensor(exploration_noise))

    def forward(self, obs):
        batch_size = obs.shape[0]
        real_channel = obs[:batch_size, self.real_start : self.imag_start].reshape(
            batch_size, *self.real_channel_shape
        )
        imag_channel = obs[:batch_size, self.imag_start : self.angle_start].reshape(
            batch_size, *self.imag_channel_shape
        )
        angles = angles = obs[:batch_size, self.angle_start : self.pos_start].reshape(
            batch_size, *self.angle_shape
        )
        pos = obs[:batch_size, self.pos_start :].reshape(batch_size, *self.position_shape)

        # positions
        pos = self.pos_embed(pos)
        pos = self.pos_network(pos)

        # angles
        angles = obs[
            :batch_size, self.real_channel_dim + self.imag_channel_dim + self.position_dim :
        ].reshape(batch_size, *self.angle_shape)
        angles = self.angle_network(angles)

        # channels
        real_channel = real_channel.reshape(real_channel.shape[0], -1)
        real_channel = self.real_channel_network(real_channel)

        imag_channel = imag_channel.reshape(imag_channel.shape[0], -1)
        imag_channel = self.imag_channel_network(imag_channel)

        channel = torch.cat([real_channel, imag_channel], dim=-1)
        channel = self.channel_connect_network(channel)

        # connect
        combined = torch.cat([channel, angles, pos], dim=-1)
        combined = self.connect_network(combined)

        # mean and log_std
        mean = self.fc_mean(combined).tanh()

        return mean * self.action_scale + self.action_bias

    def explore(self, obs):
        act = self(obs)
        return act + torch.randn_like(act).mul(self.action_scale * self.exploration_noise)


class MLPBlock(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, multiplier: int = 2, device=torch.device("cpu")
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features * multiplier, bias=False, device=device),
            nn.GELU(),
            nn.Linear(out_features * multiplier, out_features, bias=False, device=device),
        )
        self.layer_norm = nn.LayerNorm(out_features, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(self.block(x))
