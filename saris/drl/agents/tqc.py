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


class TQC(nn.Module):
    def __init__(
        self,
        envs: vector.VectorEnv,
        device: torch.device = torch.device("cpu"),
        n_quantiles=25,
    ):
        super().__init__()

        self.n_quantiles = n_quantiles

        ac_space = envs.single_action_space

        channel_space = envs.get_attr("channel_space")
        angle_space = envs.get_attr("angle_space")
        position_space = envs.get_attr("position_space")

        self.channel_shape = channel_space[0].shape
        self.angle_shape = angle_space[0].shape
        self.position_shape = position_space[0].shape

        ff_dim = 256

        # positions
        self.pos_embed = Embedder(np.prod(self.position_shape), num_freqs=5)
        pos_out_dim = self.pos_embed.out_dim

        self.ob_layers = [
            nn.Linear(
                np.prod(self.channel_shape) + np.prod(self.angle_shape) + pos_out_dim,
                400,
                device=device,
            ),
            nn.GELU(),
            MLPBlock(400, ff_dim, device=device),
            MLPBlock(ff_dim, ff_dim, device=device),
            # MLPBlock(ff_dim, ff_dim, device=device),
        ]
        self.connect_network = nn.Sequential(*self.ob_layers)

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
        self.combine_layer = nn.Linear(ff_dim, n_quantiles, device=device)

    def forward(self, obs, acs):
        batch_size = obs.shape[0]
        pos = obs[..., -np.prod(self.position_shape) :]

        # positions
        pos = self.pos_embed(pos)

        # angles
        channel_angle = obs[..., : -np.prod(self.position_shape)]

        ob = torch.cat([channel_angle, pos], dim=-1)
        combined = self.connect_network(ob)

        # action
        action = self.action_network(acs)

        # combine
        ob_ac = self.combine_network(torch.cat([combined, action], dim=-1))
        quantiles = self.combine_layer(ob_ac)
        return quantiles


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(
        self,
        envs: vector.VectorEnv,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        ac_space = envs.single_action_space
        self.ac_shape = ac_space.shape

        channel_space = envs.get_attr("channel_space")
        angle_space = envs.get_attr("angle_space")
        position_space = envs.get_attr("position_space")

        self.channel_shape = channel_space[0].shape
        self.angle_shape = angle_space[0].shape
        self.position_shape = position_space[0].shape

        ff_dim = 256

        # positions
        self.pos_embed = Embedder(np.prod(self.position_shape), num_freqs=5)
        pos_out_dim = self.pos_embed.out_dim

        self.ob_layers = [
            nn.Linear(
                np.prod(self.channel_shape) + np.prod(self.angle_shape) + pos_out_dim,
                400,
                device=device,
            ),
            nn.GELU(),
            MLPBlock(400, ff_dim, device=device),
            MLPBlock(ff_dim, ff_dim, device=device),
            # MLPBlock(ff_dim, ff_dim, device=device),
        ]
        self.connect_network = nn.Sequential(*self.ob_layers)

        self.fc_mean = nn.Linear(ff_dim, np.prod(self.ac_shape), device=device)
        self.fc_log_std = nn.Linear(ff_dim, np.prod(self.ac_shape), device=device)

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

    def forward(self, obs):
        batch_size = obs.shape[0]
        pos = obs[..., -np.prod(self.position_shape) :]

        # positions
        pos = self.pos_embed(pos)

        # angles
        channel_angle = obs[..., : -np.prod(self.position_shape)]

        ob = torch.cat([channel_angle, pos], dim=-1)
        combined = self.connect_network(ob)

        # mean and log_std
        mean = self.fc_mean(combined)
        log_std = self.fc_log_std(combined)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-8)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


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
        return self.layer_norm(self.block(x) + x)
