# source: https://github.com/nakamotoo/Cal-QL/tree/main
# https://arxiv.org/pdf/2303.05479.pdf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import copy
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import saris
from saris.drl.agents import calql_simplified
from saris.utils import utils, pytorch_utils, buffers, running_mean, load_data, cudagraphs
import importlib.resources
import wandb
import gymnasium as gym
from saris.drl.envs import register_envs
import tqdm
import json
import traceback
import torchinfo
import time
from collections import deque
from torchrl.data import LazyTensorStorage, ReplayBuffer

# from tensordict.nn.cudagraphs import CudaGraphModule
from tensordict.nn import TensorDictModule
from tensordict import TensorDict
import tensordict as td

register_envs()
TensorBatch = Tuple[torch.Tensor]


@dataclass
class TrainConfig:
    # Experiment
    command: str = "train"  # Command for "train" or "eval"
    env_id: str = "wireless-sigmap-v0"  # environment name
    offline_iterations: int = int(0)  # Number of offline updates
    online_iterations: int = int(5_001)  # Number of online updates
    learning_starts: int = int(900)  # Number of steps before learning starts
    checkpoint_path: Optional[str] = None  # Save path
    load_model: str = "-1"  # Model load file name for resume training, "" doesn't load
    offline_data_dir: str = "-1"  # Offline data directory
    sionna_config_file: str = ""  # Sionna config file
    verbose: bool = False  # Print debug information
    save_freq: int = int(100)  # How often (time steps) we save

    # Environment
    ep_len: int = 75  # Max length of episode
    eval_ep_len: int = 50  # Max length of evaluation episode
    num_envs: int = 4  # Number of parallel environments
    seed: int = 10  # Sets Gym, PyTorch and Numpy seeds
    eval_seed: int = 100  # Eval environment seed

    # CQL
    n_updates: int = 10  # Number of updates per step
    offline_buffer_size: int = 50_000  # Offline replay buffer size
    online_buffer_size: int = 75_000  # Online replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.85  # Discount factor
    alpha_multiplier: float = 1.0  # Multiplier for alpha in loss
    use_automatic_entropy_tuning: bool = True  # Tune entropy
    backup_entropy: bool = False  # Use backup entropy
    policy_lr: float = 1e-4  # Policy learning rate
    qf_lr: float = 3e-4  # Critics learning rate
    tau: float = 5e-3  # Target network update rate
    target_update_period: int = 1  # Frequency of target nets updates
    cql_alpha: float = 5.0  # CQL offline regularization parameter
    cql_alpha_online: float = 2.0  # CQL online regularization parameter
    cql_n_actions: int = 10  # Number of sampled actions
    cql_importance_sample: bool = True  # Use importance sampling
    cql_lagrange: bool = True  # Use Lagrange version of CQL
    cql_target_action_gap: float = 0.8  # Action gap
    cql_temp: float = 1.0  # CQL temperature
    cql_max_target_backup: bool = True  # Use max target backup
    cql_clip_diff_min: float = -200  # Q-function lower loss clipping
    cql_clip_diff_max: float = 200  # Q-function upper loss clipping
    orthogonal_init: bool = True  # Orthogonal initialization
    normalize: bool = True  # Normalize states
    q_n_hidden_layers: int = 2  # Number of hidden layers in Q networks

    # Cal-QL
    mixing_ratio: float = 0.0  # Data mixing ratio for online tuning, should be ~0.1

    # Wandb logging
    project: str = "SARIS"  # wandb project name
    group: str = "Cal-QL"  # wandb group name
    name: str = "Online-Learning"  # wandb run name

    def __post_init__(self):
        lib_dir = importlib.resources.files(saris)
        source_dir = os.path.dirname(lib_dir)
        self.source_dir = source_dir

        if self.checkpoint_path is None:
            raise ValueError("Checkpoints path is required for training")

        device = pytorch_utils.init_gpu()
        self.device = device


def make_env(
    config: TrainConfig,
    idx: int,
    eval_mode: bool,
    capture_video: Optional[bool] = None,
    run_name: Optional[str] = None,
):

    import tensorflow as tf

    def thunk():

        seed = config.seed if not eval_mode else config.eval_seed
        max_episode_steps = config.eval_ep_len if eval_mode else config.ep_len
        seed += idx
        env = gym.make(
            config.env_id,
            idx=idx,
            sionna_config_file=config.sionna_config_file,
            log_string=config.name,
            eval_mode=eval_mode,
            seed=seed,
            max_episode_steps=max_episode_steps,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        env = gym.wrappers.FlattenObservation(env)
        env.action_space.seed(config.seed)
        env.observation_space.seed(config.seed)

        return env

    return thunk


def wandb_init(config: TrainConfig) -> None:
    key_filename = os.path.join(config.source_dir, "tmp_wandb_api_key.txt")
    with open(key_filename, "r") as f:
        key_api = f.read().strip()
    wandb.login(relogin=True, key=key_api, host="https://api.wandb.ai")
    wandb.init(
        config=config,
        dir=config.checkpoint_path,
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4())[:5],
        mode="offline",
    )


def normalize_observations(observations: np.ndarray) -> np.ndarray:
    return (observations - np.mean(observations, axis=0)) / (np.std(observations, axis=0) + 1e-6)


def get_return_to_go(dataset: Dict, config: TrainConfig) -> np.ndarray:
    returns = np.full((dataset["rewards"].shape), np.nan)
    ep_ret, ep_len = 0.0, 0
    cur_rewards = []
    N = dataset["rewards"].shape[0]
    terminals = []

    for t, (r, d) in enumerate(zip(dataset["rewards"], dataset["terminations"])):
        ep_ret += float(r)
        cur_rewards.append(float(r))
        terminals.append(float(d))
        ep_len += 1
        is_last_step = (t == N - 1) or (ep_len == config.ep_len)

        if d or is_last_step:
            discounted_returns = [0] * ep_len
            prev_return = 0
            for i in reversed(range(ep_len)):
                discounted_returns[i] = cur_rewards[i] + config.discount * prev_return * (
                    1 - terminals[i]
                )
                prev_return = discounted_returns[i]
            returns[t - ep_len + 1 : t + 1] = np.array(discounted_returns)[..., np.newaxis]
            ep_ret, ep_len = 0.0, 0
            cur_rewards = []
            terminals = []
    return returns


class CalQL:
    def __init__(
        self,
        critic_1: calql_simplified.FullyConnectedQFunction,
        critic_2: calql_simplified.FullyConnectedQFunction,
        critic_optimizer: torch.optim.Optimizer,
        critic_scheduler: torch.optim.lr_scheduler._LRScheduler,
        actor: calql_simplified.TanhGaussianPolicy,
        actor_optimizer: torch.optim.Optimizer,
        actor_scheduler: torch.optim.lr_scheduler._LRScheduler,
        target_entropy: float,
        discount: float = 0.9,
        alpha_multiplier: float = 1.0,
        use_automatic_entropy_tuning: bool = True,
        backup_entropy: bool = False,
        policy_lr: bool = 3e-4,
        qf_lr: bool = 3e-4,
        tau: float = 5e-3,
        target_update_period: int = 1,
        cql_n_actions: int = 10,
        cql_importance_sample: bool = True,
        cql_lagrange: bool = True,
        cql_target_action_gap: float = 0.8,
        cql_temp: float = 1.0,
        cql_alpha: float = 5.0,
        cql_max_target_backup: bool = True,
        cql_clip_diff_min: float = -100,
        cql_clip_diff_max: float = 100,
        device: str = "cpu",
    ):
        super().__init__()

        self.discount = discount
        self.target_entropy = target_entropy
        self.alpha_multiplier = alpha_multiplier
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.backup_entropy = backup_entropy
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.tau = tau
        self.target_update_period = target_update_period
        self.cql_n_actions = cql_n_actions
        self.cql_importance_sample = cql_importance_sample
        self.cql_lagrange = cql_lagrange
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_temp = cql_temp
        self.cql_alpha = cql_alpha
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self._device = device

        self.total_it = 0

        self.critic_1 = critic_1
        self.critic_2 = critic_2

        self.target_critic_1 = deepcopy(self.critic_1).to(device)
        self.target_critic_2 = deepcopy(self.critic_2).to(device)

        self.actor = actor

        self.actor_scaler = torch.cuda.amp.GradScaler()
        self.actor_optimizer = actor_optimizer
        self.actor_scheduler = actor_scheduler

        self.critic_scaler = torch.cuda.amp.GradScaler()
        self.critic_optimizer = critic_optimizer
        self.critic_scheduler = critic_scheduler

        if self.use_automatic_entropy_tuning:
            self.log_alpha = calql_simplified.Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(),
                lr=self.policy_lr,
            )
        else:
            self.log_alpha = None

        self.log_alpha_prime = calql_simplified.Scalar(1.0)
        self.alpha_prime_optimizer = torch.optim.Adam(
            self.log_alpha_prime.parameters(),
            lr=self.qf_lr,
        )

        self._calibration_enabled = True
        self.total_it = 0

    def update_target_network(self, tau: float):
        for target_param, param in zip(
            self.target_critic_1.parameters(), self.critic_1.parameters()
        ):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)

        for target_param, param in zip(
            self.target_critic_2.parameters(), self.critic_2.parameters()
        ):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)

    def switch_calibration(self):
        self._calibration_enabled = not self._calibration_enabled

    def _alpha_and_alpha_loss(self, observations: torch.Tensor, log_pi: torch.Tensor):
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha() * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha().exp() * self.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.alpha_multiplier)
        return alpha, alpha_loss

    def _policy_loss(
        self,
        observations: torch.Tensor,
        new_actions: torch.Tensor,
        alpha: torch.Tensor,
        log_pi: torch.Tensor,
    ) -> torch.Tensor:
        q_new_actions = torch.min(
            self.critic_1(observations, new_actions),
            self.critic_2(observations, new_actions),
        )
        policy_loss = (alpha * log_pi - q_new_actions).mean()
        return policy_loss

    def _q_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        mc_returns: torch.Tensor,
        alpha: torch.Tensor,
        log_dict: Dict,
    ) -> torch.Tensor:
        q1_predicted = self.critic_1(observations, actions)
        q2_predicted = self.critic_2(observations, actions)

        with torch.no_grad():
            if self.cql_max_target_backup:
                new_next_actions, next_log_pi = self.actor(
                    next_observations, repeat=self.cql_n_actions
                )
                target_q_values = torch.min(
                    self.target_critic_1(next_observations, new_next_actions),
                    self.target_critic_2(next_observations, new_next_actions),
                )
                target_q_values, max_target_indices = torch.max(target_q_values, dim=-1)
                next_log_pi = torch.gather(
                    next_log_pi, -1, max_target_indices.unsqueeze(-1)
                ).squeeze(-1)
            else:
                new_next_actions, next_log_pi = self.actor(next_observations)
                target_q_values = torch.min(
                    self.target_critic_1(next_observations, new_next_actions),
                    self.target_critic_2(next_observations, new_next_actions),
                )

            if self.backup_entropy:
                target_q_values = target_q_values - alpha * next_log_pi

            target_q_values = target_q_values.unsqueeze(-1)
            td_target = rewards + (1.0 - dones) * self.discount * target_q_values
            td_target = td_target.squeeze(-1)
        qf1_loss = F.mse_loss(q1_predicted, td_target)
        qf2_loss = F.mse_loss(q2_predicted, td_target)

        # CQL
        batch_size = actions.shape[0]
        action_dim = actions.shape[-1]
        cql_random_actions = actions.new_empty(
            (batch_size, self.cql_n_actions, action_dim), requires_grad=False
        ).uniform_(-1, 1)

        with torch.no_grad():
            cql_current_acts, cql_current_log_pis = self.actor(
                observations, repeat=self.cql_n_actions
            )
            cql_next_acts, cql_next_log_pis = self.actor(
                next_observations, repeat=self.cql_n_actions
            )
            cql_current_acts, cql_current_log_pis = (cql_current_acts, cql_current_log_pis)
            cql_next_acts, cql_next_log_pis = (cql_next_acts, cql_next_log_pis)

        cql_q1_rand = self.critic_1(observations, cql_random_actions)
        cql_q2_rand = self.critic_2(observations, cql_random_actions)
        cql_q1_current_acts = self.critic_1(observations, cql_current_acts)
        cql_q2_current_acts = self.critic_2(observations, cql_current_acts)
        cql_q1_next_acts = self.critic_1(observations, cql_next_acts)
        cql_q2_next_acts = self.critic_2(observations, cql_next_acts)

        # Calibration
        lower_bounds = mc_returns.reshape(-1, 1).repeat(1, cql_q1_current_acts.shape[1])

        num_vals = torch.sum(lower_bounds == lower_bounds)
        bound_rate_q1_current_actions = torch.sum(cql_q1_current_acts < lower_bounds) / num_vals
        bound_rate_q2_current_actions = torch.sum(cql_q2_current_acts < lower_bounds) / num_vals
        bound_rate_q1_next_actions = torch.sum(cql_q1_next_acts < lower_bounds) / num_vals
        bound_rate_q2_next_actions = torch.sum(cql_q2_next_acts < lower_bounds) / num_vals

        """ Cal-QL: bound Q-values with MC return-to-go """
        if self._calibration_enabled:
            cql_q1_current_acts = torch.maximum(cql_q1_current_acts, lower_bounds)
            cql_q2_current_acts = torch.maximum(cql_q2_current_acts, lower_bounds)
            cql_q1_next_acts = torch.maximum(cql_q1_next_acts, lower_bounds)
            cql_q2_next_acts = torch.maximum(cql_q2_next_acts, lower_bounds)

        cql_cat_q1 = torch.cat(
            [cql_q1_rand, torch.unsqueeze(q1_predicted, 1), cql_q1_next_acts, cql_q1_current_acts],
            dim=1,
        )
        cql_cat_q2 = torch.cat(
            [cql_q2_rand, torch.unsqueeze(q2_predicted, 1), cql_q2_next_acts, cql_q2_current_acts],
            dim=1,
        )
        cql_std_q1 = torch.std(cql_cat_q1, dim=1)
        cql_std_q2 = torch.std(cql_cat_q2, dim=1)

        if self.cql_importance_sample:
            random_density = np.log(0.5**action_dim)
            cql_cat_q1 = torch.cat(
                [
                    cql_q1_rand - random_density,
                    cql_q1_next_acts - cql_next_log_pis.detach(),
                    cql_q1_current_acts - cql_current_log_pis.detach(),
                ],
                dim=1,
            )
            cql_cat_q2 = torch.cat(
                [
                    cql_q2_rand - random_density,
                    cql_q2_next_acts - cql_next_log_pis.detach(),
                    cql_q2_current_acts - cql_current_log_pis.detach(),
                ],
                dim=1,
            )

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q1_predicted, self.cql_clip_diff_min, self.cql_clip_diff_max
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q2_predicted, self.cql_clip_diff_min, self.cql_clip_diff_max
        ).mean()

        if self.cql_lagrange:
            alpha_prime = torch.clamp(torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0)
            cql_min_qf1_loss = (
                alpha_prime * self.cql_alpha * (cql_qf1_diff - self.cql_target_action_gap)
            )
            cql_min_qf2_loss = (
                alpha_prime * self.cql_alpha * (cql_qf2_diff - self.cql_target_action_gap)
            )

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        else:
            cql_min_qf1_loss = cql_qf1_diff * self.cql_alpha
            cql_min_qf2_loss = cql_qf2_diff * self.cql_alpha
            alpha_prime_loss = observations.new_tensor(0.0)
            alpha_prime = observations.new_tensor(0.0)

        qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss

        log_dict.update(
            dict(
                qf1_loss=qf1_loss.item(),
                qf2_loss=qf2_loss.item(),
                alpha=alpha.item(),
                average_qf1=q1_predicted.mean().item(),
                average_qf2=q2_predicted.mean().item(),
                average_target_q=target_q_values.mean().item(),
            )
        )

        log_dict.update(
            dict(
                cql_std_q1=cql_std_q1.mean().item(),
                cql_std_q2=cql_std_q2.mean().item(),
                cql_q1_rand=cql_q1_rand.mean().item(),
                cql_q2_rand=cql_q2_rand.mean().item(),
                cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
                cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
                cql_qf1_diff=cql_qf1_diff.mean().item(),
                cql_qf2_diff=cql_qf2_diff.mean().item(),
                cql_q1_current_acts=cql_q1_current_acts.mean().item(),
                cql_q2_current_acts=cql_q2_current_acts.mean().item(),
                cql_q1_next_acts=cql_q1_next_acts.mean().item(),
                cql_q2_next_acts=cql_q2_next_acts.mean().item(),
                alpha_prime_loss=alpha_prime_loss.item(),
                alpha_prime=alpha_prime.item(),
                bound_rate_q1_current_actions=bound_rate_q1_current_actions.item(),  # noqa
                bound_rate_q2_current_actions=bound_rate_q2_current_actions.item(),  # noqa
                bound_rate_q1_next_actions=bound_rate_q1_next_actions.item(),
                bound_rate_q2_next_actions=bound_rate_q2_next_actions.item(),
            )
        )

        return qf_loss

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        (observations, actions, rewards, next_observations, dones, mc_returns) = batch
        self.total_it += 1

        new_actions, log_pi = self.actor(observations)

        alpha, alpha_loss = self._alpha_and_alpha_loss(observations, log_pi)

        """ Policy loss """
        policy_loss = self._policy_loss(observations, new_actions, alpha, log_pi)

        log_dict = dict(
            log_pi=log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=alpha.item(),
        )

        """ Q function loss """
        qf_loss = self._q_loss(
            observations, actions, next_observations, rewards, dones, mc_returns, alpha, log_dict
        )

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.actor_scaler.scale(policy_loss).backward()
        self.actor_scaler.unscale_(self.actor_optimizer)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.75)
        self.actor_scaler.step(self.actor_optimizer)
        self.actor_scaler.update()
        self.actor_scheduler.step()

        self.critic_optimizer.zero_grad(set_to_none=True)
        self.critic_scaler.scale(qf_loss).backward()
        self.critic_scaler.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), 0.75
        )
        self.critic_scaler.step(self.critic_optimizer)
        self.critic_scaler.update()
        self.critic_scheduler.step()

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.tau)

        lr_dict = {
            "actor_lr": self.actor_optimizer.param_groups[0]["lr"],
            "critic_lr": self.critic_optimizer.param_groups[0]["lr"],
        }
        log_dict.update(lr_dict)

        # replace keys with "train/" prefix for log_dict
        log_dict = {f"train/{k}": v for k, v in log_dict.items()}

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic_1.state_dict(),
            "critic2": self.critic_2.state_dict(),
            "critic1_target": self.target_critic_1.state_dict(),
            "critic2_target": self.target_critic_2.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "sac_log_alpha": self.log_alpha,
            "sac_log_alpha_optim": self.alpha_optimizer.state_dict(),
            "cql_log_alpha": self.log_alpha_prime,
            "cql_log_alpha_optim": self.alpha_prime_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict=state_dict["actor"])
        self.critic_1.load_state_dict(state_dict=state_dict["critic1"])
        self.critic_2.load_state_dict(state_dict=state_dict["critic2"])

        self.target_critic_1.load_state_dict(state_dict=state_dict["critic1_target"])
        self.target_critic_2.load_state_dict(state_dict=state_dict["critic2_target"])

        self.critic_optimizer.load_state_dict(state_dict=state_dict["critic_optimizer"])
        self.actor_optimizer.load_state_dict(state_dict=state_dict["actor_optim"])

        self.log_alpha = state_dict["sac_log_alpha"]
        self.alpha_optimizer.load_state_dict(state_dict=state_dict["sac_log_alpha_optim"])

        self.log_alpha_prime = state_dict["cql_log_alpha"]
        self.alpha_prime_optimizer.load_state_dict(state_dict=state_dict["cql_log_alpha_optim"])
        self.total_it = state_dict["total_it"]


def train(trainer: CalQL, config: TrainConfig, envs: gym.vector.VectorEnv) -> None:

    assert (
        config.offline_iterations == 0
    ), f"Offline pretraining is not supported yet, Please set offline_iterations to 0"

    print("---------------------------------------")
    print(f"Training Cal-QL, Env: {config.env_id}")
    print(f"Training Seed: {config.seed}")
    print("---------------------------------------")

    batch_size_offline = int(config.batch_size * config.mixing_ratio)
    batch_size_online = config.batch_size - batch_size_offline
    ob_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]

    # Load offline data
    if config.offline_data_dir != "-1":
        print(f"Loading offline data from {config.offline_data_dir}")
        offline_buffer = buffers.ReplayBuffer(
            ob_dim,
            action_dim,
            config.offline_buffer_size,
        )

        offline_data = load_data.load_offline_dataset(
            config.offline_data_dir,
            offline_buffer.max_size(),
        )

        mc_returns = get_return_to_go(offline_data, config)
        offline_data["mc_returns"] = np.array(mc_returns)
        assert (
            offline_data["mc_returns"].shape == offline_data["rewards"].shape
        ), f"MC returns shape: {offline_data['mc_returns'].shape}, rewards shape: {offline_data['rewards'].shape}"

        offline_data["observations"] = normalize_observations(offline_data["observations"])
        offline_data["next_observations"] = normalize_observations(
            offline_data["next_observations"]
        )

        offline_data["dones"] = offline_data["terminations"]
        offline_batch = {
            "observations": offline_data["observations"],
            "actions": offline_data["actions"],
            "rewards": offline_data["rewards"],
            "next_observations": offline_data["next_observations"],
            "dones": offline_data["dones"],
            "mc_returns": offline_data["mc_returns"],
        }

        offline_buffer.load_dataset(**offline_batch)

    online_buffer = buffers.ReplayBuffer(
        ob_dim,
        action_dim,
        config.online_buffer_size,
    )

    if config.load_model != "-1":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))

    # evaluations = []
    obs, env_infos = envs.reset(seed=config.seed)
    # dones = False
    wandb.define_metric("train_return/step")
    wandb.define_metric("train_return/*", step_metric="train_return/step")
    train_return_log = {"train_return/step": 0}

    # Create save directory for buffer
    local_assets_dir = utils.get_dir(config.source_dir, "local_assets")
    buffer_saved_name = os.path.join("replay_buffer", config.name)
    buffer_saved_dir = utils.get_dir(local_assets_dir, buffer_saved_name)

    if config.offline_iterations > 0:
        print("Offline pretraining")
    else:
        print(f"No offline pretraining, starting online training")

    # Create running meanstd for normalization
    obs_rms = running_mean.RunningMeanStd(shape=envs.single_observation_space.shape)

    # Training loop
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    t = tqdm.tqdm(
        range(int(config.offline_iterations) + int(config.online_iterations)), dynamic_ncols=True
    )
    for step in t:
        if step == config.offline_iterations:
            print("Online tuning")
            trainer.switch_calibration()
            trainer.cql_alpha = config.cql_alpha_online
        online_log = {}
        if step >= config.offline_iterations:
            acts, _ = trainer.actor(torch.tensor(obs, device=config.device, dtype=torch.float32))
            acts = pytorch_utils.to_numpy(acts)
            try:
                next_obs, rews, terminations, truncations, env_infos = envs.step(acts)
            except Exception as e:
                print(f"Error at step {step}")
                print(f"Error: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                envs.close()
                envs = gym.vector.AsyncVectorEnv(
                    [make_env(config, i, eval_mode=False) for i in range(config.num_envs)],
                    context="spawn",
                )
                obs, env_infos = envs.reset(seed=config.seed)
                continue
            dones = terminations

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = copy.deepcopy(next_obs)
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = env_infos["final_observation"][idx]

            obs_rms.update(torch.tensor(obs, dtype=torch.float))

            rews = np.asarray(rews)[..., np.newaxis]
            dones = np.asarray(dones)[..., np.newaxis]
            truncations = np.asarray(truncations)[..., np.newaxis]
            terminations = np.asarray(terminations)[..., np.newaxis]

            online_buffer.add_batch_transition(obs, acts, rews, real_next_obs, dones)

            if "final_info" in env_infos:
                returns = [info["episode"]["r"] for info in env_infos["final_info"]]
                return_mean = np.mean(returns)
                return_std = np.std(returns)
                train_return_log["train_return/return_mean"] = return_mean
                train_return_log["train_return/return_std"] = return_std
                wandb.log(train_return_log)
                train_return_log["train_return/step"] = step

                print(f"step={step}, return_mean={return_mean}, return_std={return_std}\n")

                path_gains = [info["path_gain"] for info in env_infos["final_info"]]
                next_path_gains = [info["next_path_gain"] for info in env_infos["final_info"]]
            else:
                path_gains = env_infos["path_gain"]
                next_path_gains = env_infos["next_path_gain"]

            # save data to file
            utils.save_data(
                {f"{step}": path_gains},
                os.path.join(buffer_saved_dir, "path_gains.txt"),
            )
            utils.save_data(
                {f"{step}": next_path_gains},
                os.path.join(buffer_saved_dir, "next_path_gains.txt"),
            )
            utils.save_data(
                {f"{step}": obs},
                os.path.join(buffer_saved_dir, "observations.txt"),
            )
            utils.save_data(
                {f"{step}": acts},
                os.path.join(buffer_saved_dir, "actions.txt"),
            )
            utils.save_data(
                {f"{step}": real_next_obs},
                os.path.join(buffer_saved_dir, "next_observations.txt"),
            )
            utils.save_data(
                {f"{step}": terminations},
                os.path.join(buffer_saved_dir, "terminations.txt"),
            )
            utils.save_data(
                {f"{step}": truncations},
                os.path.join(buffer_saved_dir, "truncations.txt"),
            )
            utils.save_data(
                {f"{step}": rews},
                os.path.join(buffer_saved_dir, "rewards.txt"),
            )

            obs = next_obs

            if step < config.learning_starts:
                continue

        for j in range(config.n_updates):
            if step < config.offline_iterations:
                batch = offline_buffer.sample(config.batch_size)
                batch = [b.to(config.device) for b in batch]
            else:
                # if j < config.n_updates * 2 // 3:
                #     # mixing training with offline data + online data
                #     offline_batch = offline_buffer.sample(batch_size_offline)
                #     online_batch = online_buffer.sample(batch_size_online)
                #     batch = [
                #         torch.vstack(tuple(b)).to(config.device)
                #         for b in zip(offline_batch, online_batch)
                #     ]
                # else:
                # online training with online data
                batch = online_buffer.sample(config.batch_size)
                batch = [b.to(config.device) for b in batch]

            batch[0] = normalize_obs(batch[0], obs_rms)
            batch[3] = normalize_obs(batch[3], obs_rms)

            log_dict = trainer.train(batch)

        log_dict[
            "train/offline_iter" if step < config.offline_iterations else "train/online_iter"
        ] = (step if step < config.offline_iterations else step - config.offline_iterations)
        log_dict.update(online_log)
        log_dict["train/step"] = step
        wandb.log(log_dict)

        if step % config.save_freq == 0 and step > 0:
            print(f"Savings model at iteration {step}")
            saved_path = os.path.join(config.checkpoint_path, f"checkpoint_{step}.pt")
            torch.save(trainer.state_dict(), saved_path)
            saved_path = os.path.join(config.checkpoint_path, f"checkpoint.pt")
            torch.save(trainer.state_dict(), saved_path)

    rms_path = os.path.join(config.checkpoint_path, "rms.pt")
    torch.save({"obs_rms": obs_rms}, rms_path)


@torch.no_grad()
def eval_actor(
    envs: gym.vector.VectorEnv, actor: calql_simplified.TanhGaussianPolicy, config: TrainConfig
) -> Tuple[np.ndarray, np.ndarray]:
    actor.eval()
    episode_rewards = []
    eval_infos = []
    obs, info = envs.reset()
    episode_reward = np.zeros(envs.num_envs)
    rms = torch.load(config.checkpoint_path + "/rms.pt")
    obs_rms = rms["obs_rms"]
    t = tqdm.tqdm(range(config.eval_ep_len), dynamic_ncols=True)
    for _ in t:
        obs = torch.tensor(obs, device=config.device, dtype=torch.float32)
        obs = normalize_obs(obs, obs_rms)
        acts = actor.act(obs)
        acts = pytorch_utils.to_numpy(acts)
        obs, rews, terminations, truncations, env_info = envs.step(acts)
        episode_reward += rews
        eval_infos.append(env_info)
    episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards), eval_infos


def eval(trainer: CalQL, config: TrainConfig, envs: gym.vector.VectorEnv) -> None:

    print("---------------------------------------")
    print(f"Evaluating model using the last checkpoint at {config.checkpoint_path}")
    print(f"Evaluation Seed: {config.eval_seed}")

    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")

    saved_path = os.path.join(config.checkpoint_path, f"checkpoint.pt")
    trainer.load_state_dict(torch.load(saved_path))
    actor = trainer.actor
    eval_episodic_returns, eval_infos = eval_actor(envs, actor, config)

    # Save evaluation results
    json_path = os.path.join(config.checkpoint_path, "eval_results.yaml")

    for eval_step, eval_info in enumerate(eval_infos):
        utils.save_data(eval_info, json_path)

        log_dict = {}
        if "final_info" in eval_info:
            total_returns = [info["episode"]["r"] for info in eval_info["final_info"]]
            total_return_mean = np.mean(total_returns)
            total_return_std = np.std(total_returns)

            path_gains = [info["path_gain"] for info in eval_info["final_info"]]
            log_dict.update(
                {
                    "eval/total_return_mean": total_return_mean,
                    "eval/total_return_std": total_return_std,
                }
            )
        else:
            path_gains = eval_info["path_gain"]

        for i, gains in enumerate(path_gains):
            log_dict[f"eval/path_gain_mean_{i}"] = np.mean(gains)
            log_dict[f"eval/path_gain_std_{i}"] = np.std(gains)
        log_dict["eval/path_gain_mean"] = np.mean([np.mean(gains) for gains in path_gains])
        log_dict["eval/path_gain_std"] = np.std([np.mean(gains) for gains in path_gains])
        log_dict["eval/step"] = eval_step

        wandb.log(log_dict)

    print(f"Evaluation over 1 episode with {config.eval_ep_len} steps: ")
    for i, ep_returns in enumerate(eval_episodic_returns):
        for env_idx, ep_return in enumerate(ep_returns):
            print(f"Env-{env_idx}: {ep_return}")
    print("---------------------------------------\n")


def normalize_obs(
    observations: torch.Tensor,
    obs_rms: running_mean.RunningMeanStd,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    mean = obs_rms.mean.to(observations.device)
    var = obs_rms.var.to(observations.device)
    return (observations - mean) / torch.sqrt(var + epsilon)


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


def make_env_tmp(env_id, seed, idx, capture_video=False, run_name=None):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


@pyrallis.wrap()
def main(config: TrainConfig):
    sionna_config = utils.load_config(config.sionna_config_file)

    # set random seeds
    pytorch_utils.init_seed(config.seed)
    if config.verbose:
        utils.log_args(config)
        utils.log_config(sionna_config)

    # if config.command.lower() == "train":
    #     envs: gym.vector.AsyncVectorEnv = gym.vector.AsyncVectorEnv(
    #         [make_env(config, i, eval_mode=False) for i in range(config.num_envs)],
    #         context="spawn",
    #     )
    # elif config.command.lower() == "eval":
    #     envs: gym.vector.AsyncVectorEnv = gym.vector.AsyncVectorEnv(
    #         [make_env(config, i, eval_mode=True) for i in range(config.num_envs)],
    #         context="spawn",
    #     )
    # else:
    #     raise ValueError(f"Invalid command: {config.command}, available commands: train, eval")

    envs: gym.vector.AsyncVectorEnv = gym.vector.AsyncVectorEnv(
        [make_env_tmp("HalfCheetah-v5", config.seed, 0)]
    )

    # Init checkpoints
    print(f"Checkpoints path: {config.checkpoint_path}")
    os.makedirs(config.checkpoint_path, exist_ok=True)
    with open(os.path.join(config.checkpoint_path, "train_config.yaml"), "w") as f:
        pyrallis.dump(config, f)

    # Cal-QL trainer
    ob_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]

    actor = calql_simplified.TanhGaussianPolicy(ob_dim, action_dim, action_scale=2.0).to(
        config.device
    )
    torchinfo.summary(
        actor,
        input_size=(1, ob_dim),
        col_names=["input_size", "output_size", "num_params"],
    )

    actor_detach = calql_simplified.TanhGaussianPolicy(ob_dim, action_dim, action_scale=2.0).to(
        config.device
    )
    # Copy params to actor_detach without grad
    TensorDict.from_module(actor).data.to_module(actor_detach)
    policy = TensorDictModule(
        actor_detach.get_action, in_keys=["observations"], out_keys=["actions"]
    )

    def get_q_params():
        critic_1 = calql_simplified.FullyConnectedQFunction(ob_dim, action_dim).to(config.device)
        critic_2 = calql_simplified.FullyConnectedQFunction(ob_dim, action_dim).to(config.device)
        torchinfo.summary(
            critic_1,
            input_size=[(1, ob_dim), (1, action_dim)],
            col_names=["input_size", "output_size", "num_params"],
        )

        qnet_params = TensorDict.from_modules(critic_1, critic_2, as_module=True)
        qnet_target = qnet_params.data.clone()

        qnet = calql_simplified.FullyConnectedQFunction(ob_dim, action_dim).to("meta")
        qnet_params.to_module(qnet)

        return qnet_params, qnet_target, qnet

    qnet_params, qnet_target, qnet = get_q_params()

    q_optimizer = torch.optim.AdamW(qnet.parameters(), config.qf_lr)
    total_iterations = config.offline_iterations + config.online_iterations
    q_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        q_optimizer,
        config.n_updates * total_iterations,
        eta_min=config.qf_lr / 10,
    )

    actor_optimizer = torch.optim.AdamW(actor.parameters(), config.policy_lr)
    actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        actor_optimizer,
        config.n_updates * total_iterations,
        eta_min=config.policy_lr / 10,
    )

    target_entropy = -torch.prod(
        torch.Tensor(envs.single_action_space.shape).to(config.device)
    ).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=config.device)
    alpha = log_alpha.detach().exp()
    alpha_optimizer = torch.optim.AdamW([log_alpha], lr=config.policy_lr)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(storage=LazyTensorStorage(config.online_buffer_size, device=config.device))

    def batched_qf(params, obs, acts, next_q_values=None):
        with params.to_module(qnet):
            q_values = qnet(obs, acts)
            if next_q_values is not None:
                loss = F.mse_loss(q_values.view(-1), next_q_values)
                return loss
            return q_values

    def update_main(data):
        # optimize the model
        q_optimizer.zero_grad()
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = actor.get_action(data["next_observations"])
            qf_next_target = torch.vmap(batched_qf, in_dims=(0, None, None))(
                qnet_target, data["next_observations"], next_state_actions
            )
            min_qf_next_target = qf_next_target.min(dim=0).values
            min_qf_next_target -= alpha * next_state_log_pi
            next_q_values = data["rewards"].flatten() + (
                1 - data["dones"].flatten()
            ).float() * config.discount * min_qf_next_target.view(-1)

        qf_loss = torch.vmap(batched_qf, in_dims=(0, None, None, None))(
            qnet_params, data["observations"], data["actions"], next_q_values
        )
        qf_loss = torch.sum(qf_loss)

        qf_loss.backward()
        q_optimizer.step()
        q_scheduler.step()
        return TensorDict(qf_loss=qf_loss.detach())

    def update_policy(data):
        actor_optimizer.zero_grad()
        pi, log_pi, _ = actor.get_action(data["observations"])
        qf_pi = torch.vmap(batched_qf, in_dims=(0, None, None))(
            qnet_params, data["observations"], pi
        )
        min_qf_pi = qf_pi.min(dim=0).values
        actor_loss = torch.mean(alpha * log_pi - min_qf_pi)

        actor_loss.backward()
        actor_optimizer.step()
        actor_scheduler.step()

        alpha_optimizer.zero_grad()
        with torch.no_grad():
            _, log_pi, _ = actor.get_action(data["observations"])
        alpha_loss = -torch.mean(log_alpha.exp() * (log_pi + target_entropy))

        alpha_loss.backward()
        alpha_optimizer.step()

        return TensorDict(
            alpha=alpha.detach(), actor_loss=actor_loss.detach(), alpha_loss=alpha_loss.detach()
        )

    def extend_and_sample(transitions):
        rb.extend(transitions)
        return rb.sample(config.batch_size)

    mode = None
    update_main = torch.compile(update_main, mode=mode)
    update_policy = torch.compile(update_policy, mode=mode)
    policy = torch.compile(policy, mode=mode)

    update_main = cudagraphs.CudaGraphModule(update_main, in_keys=[], out_keys=[])
    update_pool = cudagraphs.CudaGraphModule(update_policy, in_keys=[], out_keys=[])

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=config.seed)
    obs = torch.as_tensor(obs, device=config.device, dtype=torch.float)
    pbar = tqdm.tqdm(range(config.total_timesteps))
    start_time = None
    max_ep_ret = -float("inf")
    avg_returns = deque(maxlen=20)
    desc = ""

    for global_step in pbar:
        if global_step == config.measure_burnin + config.learning_starts:
            start_time = time.time()
            measure_burnin = global_step

        # ALGO LOGIC: put action logic here
        if global_step < config.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions = policy(obs)
            actions = actions.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                r = float(info["episode"]["r"])
                max_ep_ret = max(max_ep_ret, r)
                avg_returns.append(r)
            desc = f"global_step={global_step}, episodic_return={torch.tensor(avg_returns).mean(): 4.2f} (max={max_ep_ret: 4.2f})"

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        next_obs = torch.as_tensor(next_obs, device=config.device, dtype=torch.float)
        real_next_obs = next_obs.clone()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = torch.as_tensor(
                    infos["final_observation"][idx], device=config.device, dtype=torch.float
                )
        # obs = torch.as_tensor(obs, device=device, dtype=torch.float)
        transition = TensorDict(
            observations=obs,
            next_observations=real_next_obs,
            actions=torch.as_tensor(actions, device=config.device, dtype=torch.float),
            rewards=torch.as_tensor(rewards, device=config.device, dtype=torch.float),
            terminations=terminations,
            dones=terminations,
            batch_size=obs.shape[0],
            device=config.device,
        )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        data = extend_and_sample(transition)

        # ALGO LOGIC: training.
        if global_step > config.learning_starts:
            out_main = update_main(data)
            if global_step % config.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    config.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    out_main.update(update_policy(data))

                    alpha.copy_(log_alpha.detach().exp())

            # update the target networks
            if global_step % config.target_network_frequency == 0:
                # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
                qnet_target.lerp_(qnet_params.data, config.tau)

            if global_step % 100 == 0 and start_time is not None:
                speed = (global_step - measure_burnin) / (time.time() - start_time)
                pbar.set_description(f"{speed: 4.4f} sps, " + desc)
                with torch.no_grad():
                    logs = {
                        "episode_return": torch.tensor(avg_returns).mean(),
                        "actor_loss": out_main["actor_loss"].mean(),
                        "alpha_loss": out_main.get("alpha_loss", 0),
                        "qf_loss": out_main["qf_loss"].mean(),
                    }
                wandb.log(
                    {
                        "speed": speed,
                        **logs,
                    },
                    step=global_step,
                )

    envs.close()


if __name__ == "__main__":
    main()
