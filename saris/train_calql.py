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
from saris.drl.agents import calql
from saris.utils import utils, pytorch_utils, buffers
import importlib
import wandb
import gymnasium as gym
from saris.drl.envs import register_envs
import tqdm
import json


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
    load_model: str = ""  # Model load file name for resume training, "" doesn't load
    sionna_config_file: str = ""  # Sionna config file
    verbose: bool = False  # Print debug information
    save_freq: int = int(100)  # How often (time steps) we save

    # Environment
    ep_len: int = 75  # Max length of episode
    eval_ep_len: int = 50  # Max length of evaluation episode
    num_envs: int = 6  # Number of parallel environments
    seed: int = 10  # Sets Gym, PyTorch and Numpy seeds
    eval_seed: int = 100  # Eval environment seed

    # CQL
    n_updates: int = 10  # Number of updates per step
    buffer_size: int = 10_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.85  # Discount factor
    alpha_multiplier: float = 1.0  # Multiplier for alpha in loss
    use_automatic_entropy_tuning: bool = True  # Tune entropy
    backup_entropy: bool = False  # Use backup entropy
    policy_lr: float = 1e-4  # Policy learning rate
    qf_lr: float = 3e-4  # Critics learning rate
    soft_target_update_rate: float = 5e-3  # Target network update rate
    bc_steps: int = int(0)  # Number of BC steps at start
    target_update_period: int = 1  # Frequency of target nets updates
    cql_alpha: float = 5.0  # CQL offline regularization parameter
    cql_alpha_online: float = 5.0  # CQL online regularization parameter
    cql_n_actions: int = 10  # Number of sampled actions
    cql_importance_sample: bool = True  # Use importance sampling
    cql_lagrange: bool = True  # Use Lagrange version of CQL
    cql_target_action_gap: float = -1.0  # Action gap
    cql_temp: float = 1.0  # CQL temperature
    cql_max_target_backup: bool = True  # Use max target backup
    cql_clip_diff_min: float = -200  # Q-function lower loss clipping
    cql_clip_diff_max: float = 200  # Q-function upper loss clipping
    orthogonal_init: bool = True  # Orthogonal initialization
    normalize: bool = True  # Normalize states
    normalize_reward: bool = True  # Normalize reward
    q_n_hidden_layers: int = 2  # Number of hidden layers in Q networks

    # Cal-QL
    mixing_ratio: float = 0.0  # Data mixing ratio for online tuning, should be ~0.1
    is_sparse_reward: bool = False  # Use sparse reward

    # Wandb logging
    project: str = "SARIS"  # wandb project name
    group: str = "Cal-QL"  # wandb group name
    name: str = "Online-Learning"  # wandb run name

    def __post_init__(self):
        lib_dir = importlib.resources.files(saris)
        source_dir = os.path.dirname(lib_dir)
        self.source_dir = source_dir

        # self.name = f"{self.name}__{self.env_id}__{str(uuid.uuid4())[:8]}"
        if self.checkpoint_path is None:
            raise ValueError("Checkpoints path is required for training")

        # if self.checkpoint_path is not None:
        #     self.checkpoint_path = os.path.join(self.checkpoint_path, self.name)
        # else:
        #     log_dir = os.path.join(self.source_dir, "local_assets", "logs")
        #     log_path = os.path.join(log_dir, self.name)
        #     self.checkpoint_path = log_path

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


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (state - state_mean) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    env = gym.wrappers.FlattenObservation(env)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


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
    )
    # save_name = os.path.join(config["checkpoint_path"], "run")
    # wandb.run.save(save_name, base_path=config["checkpoint_path"])


# def is_goal_reached(reward: float, info: Dict) -> bool:
#     if "goal_achieved" in info:
#         return info["goal_achieved"]
#     return reward > 0  # Assuming that reaching target is a positive reward


@torch.no_grad()
def eval_actor(
    envs: gym.vector.VectorEnv, actor: calql.TanhGaussianPolicy, config: TrainConfig
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


# def return_reward_range(dataset: Dict, max_episode_steps: int) -> Tuple[float, float]:
#     returns, lengths = [], []
#     ep_ret, ep_len = 0.0, 0
#     for r, d in zip(dataset["rewards"], dataset["terminals"]):
#         ep_ret += float(r)
#         ep_len += 1
#         if d or ep_len == max_episode_steps:
#             returns.append(ep_ret)
#             lengths.append(ep_len)
#             ep_ret, ep_len = 0.0, 0
#     lengths.append(ep_len)  # but still keep track of number of steps
#     assert sum(lengths) == len(dataset["rewards"])
#     return min(returns), max(returns)


# def get_return_to_go(dataset: Dict, env: gym.Env, config: TrainConfig) -> np.ndarray:
#     returns = []
#     ep_ret, ep_len = 0.0, 0
#     cur_rewards = []
#     terminals = []
#     N = len(dataset["rewards"])
#     for t, (r, d) in enumerate(zip(dataset["rewards"], dataset["terminals"])):
#         ep_ret += float(r)
#         cur_rewards.append(float(r))
#         terminals.append(float(d))
#         ep_len += 1
#         is_last_step = (
#             (t == N - 1)
#             or (
#                 np.linalg.norm(dataset["observations"][t + 1] - dataset["next_observations"][t])
#                 > 1e-6
#             )
#             or ep_len == env._max_episode_steps
#         )

#         if d or is_last_step:
#             discounted_returns = [0] * ep_len
#             prev_return = 0
#             if (
#                 config.is_sparse_reward
#                 and r == env.ref_min_score * config.reward_scale + config.reward_bias
#             ):
#                 discounted_returns = [r / (1 - config.discount)] * ep_len
#             else:
#                 for i in reversed(range(ep_len)):
#                     discounted_returns[i] = cur_rewards[i] + config.discount * prev_return * (
#                         1 - terminals[i]
#                     )
#                     prev_return = discounted_returns[i]
#             returns += discounted_returns
#             ep_ret, ep_len = 0.0, 0
#             cur_rewards = []
#             terminals = []
#     return returns


# def modify_reward(
#     dataset: Dict,
#     env_name: str,
#     max_episode_steps: int = 1000,
#     reward_scale: float = 1.0,
#     reward_bias: float = 0.0,
# ) -> Dict:
#     modification_data = {}
#     if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
#         min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
#         dataset["rewards"] /= max_ret - min_ret
#         dataset["rewards"] *= max_episode_steps
#         modification_data = {
#             "max_ret": max_ret,
#             "min_ret": min_ret,
#             "max_episode_steps": max_episode_steps,
#         }
#     dataset["rewards"] = dataset["rewards"] * reward_scale + reward_bias
#     return modification_data


def modify_reward_online(
    reward: float,
    env_name: str,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
    **kwargs,
) -> float:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        reward /= kwargs["max_ret"] - kwargs["min_ret"]
        reward *= kwargs["max_episode_steps"]
    reward = reward * reward_scale + reward_bias
    return reward


class CalQL:
    def __init__(
        self,
        critic_1: calql.FullyConnectedQFunction,
        critic_2: calql.FullyConnectedQFunction,
        critic_optimizer: torch.optim.Optimizer,
        critic_scheduler: torch.optim.lr_scheduler._LRScheduler,
        actor: calql.TanhGaussianPolicy,
        actor_optimizer: torch.optim.Optimizer,
        actor_scheduler: torch.optim.lr_scheduler._LRScheduler,
        target_entropy: float,
        discount: float = 0.99,
        alpha_multiplier: float = 1.0,
        use_automatic_entropy_tuning: bool = True,
        backup_entropy: bool = False,
        policy_lr: bool = 3e-4,
        qf_lr: bool = 3e-4,
        soft_target_update_rate: float = 5e-3,
        bc_steps=100000,
        target_update_period: int = 1,
        cql_n_actions: int = 10,
        cql_importance_sample: bool = True,
        cql_lagrange: bool = False,
        cql_target_action_gap: float = -1.0,
        cql_temp: float = 1.0,
        cql_alpha: float = 5.0,
        cql_max_target_backup: bool = False,
        cql_clip_diff_min: float = -np.inf,
        cql_clip_diff_max: float = np.inf,
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
        self.soft_target_update_rate = soft_target_update_rate
        self.bc_steps = bc_steps
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
            self.log_alpha = calql.Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(),
                lr=self.policy_lr,
            )
        else:
            self.log_alpha = None

        self.log_alpha_prime = calql.Scalar(1.0)
        self.alpha_prime_optimizer = torch.optim.Adam(
            self.log_alpha_prime.parameters(),
            lr=self.qf_lr,
        )

        self._calibration_enabled = True
        self.total_it = 0

    def update_target_network(self, soft_target_update_rate: float):
        soft_update(self.target_critic_1, self.critic_1, soft_target_update_rate)
        soft_update(self.target_critic_2, self.critic_2, soft_target_update_rate)

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
        actions: torch.Tensor,
        new_actions: torch.Tensor,
        alpha: torch.Tensor,
        log_pi: torch.Tensor,
    ) -> torch.Tensor:
        if self.total_it <= self.bc_steps:
            log_probs = self.actor.log_prob(observations, actions)
            policy_loss = (alpha * log_pi - log_probs).mean()
        else:
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

        if self.cql_max_target_backup:
            new_next_actions, next_log_pi = self.actor(next_observations, repeat=self.cql_n_actions)
            target_q_values, max_target_indices = torch.max(
                torch.min(
                    self.target_critic_1(next_observations, new_next_actions),
                    self.target_critic_2(next_observations, new_next_actions),
                ),
                dim=-1,
            )
            next_log_pi = torch.gather(next_log_pi, -1, max_target_indices.unsqueeze(-1)).squeeze(
                -1
            )
        else:
            new_next_actions, next_log_pi = self.actor(next_observations)
            target_q_values = torch.min(
                self.target_critic_1(next_observations, new_next_actions),
                self.target_critic_2(next_observations, new_next_actions),
            )

        if self.backup_entropy:
            target_q_values = target_q_values - alpha * next_log_pi

        target_q_values = target_q_values.unsqueeze(-1)
        td_target = rewards + (1.0 - dones) * self.discount * target_q_values.detach()
        td_target = td_target.squeeze(-1)
        qf1_loss = F.mse_loss(q1_predicted, td_target.detach())
        qf2_loss = F.mse_loss(q2_predicted, td_target.detach())

        # CQL
        batch_size = actions.shape[0]
        action_dim = actions.shape[-1]
        cql_random_actions = actions.new_empty(
            (batch_size, self.cql_n_actions, action_dim), requires_grad=False
        ).uniform_(-1, 1)
        cql_current_actions, cql_current_log_pis = self.actor(
            observations, repeat=self.cql_n_actions
        )
        cql_next_actions, cql_next_log_pis = self.actor(
            next_observations, repeat=self.cql_n_actions
        )
        cql_current_actions, cql_current_log_pis = (
            cql_current_actions.detach(),
            cql_current_log_pis.detach(),
        )
        cql_next_actions, cql_next_log_pis = (
            cql_next_actions.detach(),
            cql_next_log_pis.detach(),
        )

        cql_q1_rand = self.critic_1(observations, cql_random_actions)
        cql_q2_rand = self.critic_2(observations, cql_random_actions)
        cql_q1_current_actions = self.critic_1(observations, cql_current_actions)
        cql_q2_current_actions = self.critic_2(observations, cql_current_actions)
        cql_q1_next_actions = self.critic_1(observations, cql_next_actions)
        cql_q2_next_actions = self.critic_2(observations, cql_next_actions)

        # Calibration
        lower_bounds = mc_returns.reshape(-1, 1).repeat(1, cql_q1_current_actions.shape[1])

        num_vals = torch.sum(lower_bounds == lower_bounds)
        bound_rate_cql_q1_current_actions = (
            torch.sum(cql_q1_current_actions < lower_bounds) / num_vals
        )
        bound_rate_cql_q2_current_actions = (
            torch.sum(cql_q2_current_actions < lower_bounds) / num_vals
        )
        bound_rate_cql_q1_next_actions = torch.sum(cql_q1_next_actions < lower_bounds) / num_vals
        bound_rate_cql_q2_next_actions = torch.sum(cql_q2_next_actions < lower_bounds) / num_vals

        """ Cal-QL: bound Q-values with MC return-to-go """
        if self._calibration_enabled:
            cql_q1_current_actions = torch.maximum(cql_q1_current_actions, lower_bounds)
            cql_q2_current_actions = torch.maximum(cql_q2_current_actions, lower_bounds)
            cql_q1_next_actions = torch.maximum(cql_q1_next_actions, lower_bounds)
            cql_q2_next_actions = torch.maximum(cql_q2_next_actions, lower_bounds)

        cql_cat_q1 = torch.cat(
            [
                cql_q1_rand,
                torch.unsqueeze(q1_predicted, 1),
                cql_q1_next_actions,
                cql_q1_current_actions,
            ],
            dim=1,
        )
        cql_cat_q2 = torch.cat(
            [
                cql_q2_rand,
                torch.unsqueeze(q2_predicted, 1),
                cql_q2_next_actions,
                cql_q2_current_actions,
            ],
            dim=1,
        )
        cql_std_q1 = torch.std(cql_cat_q1, dim=1)
        cql_std_q2 = torch.std(cql_cat_q2, dim=1)

        if self.cql_importance_sample:
            random_density = np.log(0.5**action_dim)
            cql_cat_q1 = torch.cat(
                [
                    cql_q1_rand - random_density,
                    cql_q1_next_actions - cql_next_log_pis.detach(),
                    cql_q1_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )
            cql_cat_q2 = torch.cat(
                [
                    cql_q2_rand - random_density,
                    cql_q2_next_actions - cql_next_log_pis.detach(),
                    cql_q2_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q1_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q2_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
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
                cql_q1_current_actions=cql_q1_current_actions.mean().item(),
                cql_q2_current_actions=cql_q2_current_actions.mean().item(),
                cql_q1_next_actions=cql_q1_next_actions.mean().item(),
                cql_q2_next_actions=cql_q2_next_actions.mean().item(),
                alpha_prime_loss=alpha_prime_loss.item(),
                alpha_prime=alpha_prime.item(),
                bound_rate_cql_q1_current_actions=bound_rate_cql_q1_current_actions.item(),  # noqa
                bound_rate_cql_q2_current_actions=bound_rate_cql_q2_current_actions.item(),  # noqa
                bound_rate_cql_q1_next_actions=bound_rate_cql_q1_next_actions.item(),
                bound_rate_cql_q2_next_actions=bound_rate_cql_q2_next_actions.item(),
            )
        )

        return qf_loss

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        (observations, actions, rewards, next_observations, dones, mc_returns) = batch
        self.total_it += 1

        new_actions, log_pi = self.actor(observations)

        alpha, alpha_loss = self._alpha_and_alpha_loss(observations, log_pi)

        """ Policy loss """
        policy_loss = self._policy_loss(observations, actions, new_actions, alpha, log_pi)

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
            self.update_target_network(self.soft_target_update_rate)

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

    # is_env_with_goal = config.env.startswith(ENVS_WITH_GOAL)
    batch_size_offline = int(config.batch_size * config.mixing_ratio)
    batch_size_online = config.batch_size - batch_size_offline - config.num_envs
    ob_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]

    # dataset = d4rl.qlearning_dataset(env)

    # Online learning running mean

    # reward_mod_dict = {}
    # if config.normalize_reward:
    #     reward_mod_dict = modify_reward(
    #         dataset,
    #         config.env,
    #         reward_scale=config.reward_scale,
    #         reward_bias=config.reward_bias,
    #     )
    # mc_returns = get_return_to_go(dataset, env, config)
    # dataset["mc_returns"] = np.array(mc_returns)
    # assert len(dataset["mc_returns"]) == len(dataset["rewards"])

    # if config.normalize:
    #     state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    # else:
    #     state_mean, state_std = 0, 1

    # dataset["observations"] = normalize_states(dataset["observations"], state_mean, state_std)
    # dataset["next_observations"] = normalize_states(
    #     dataset["next_observations"], state_mean, state_std
    # )
    # offline_buffer = ReplayBuffer(
    #     ob_dim,
    #     action_dim,
    #     config.buffer_size,
    #     config.device,
    # )
    online_buffer = buffers.ReplayBuffer(
        ob_dim,
        action_dim,
        config.buffer_size,
    )
    # offline_buffer.load_d4rl_dataset(dataset)

    if config.load_model != "":
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
    obs_rms = RunningMeanStd(shape=envs.single_observation_space.shape)
    rewards_rms = RunningMeanStd(shape=(1,))

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
            next_obs, rews, terminations, truncations, env_infos = envs.step(acts)
            dones = terminations
            # if not goal_achieved:
            #     goal_achieved = is_goal_reached(reward, env_infos)

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = copy.deepcopy(next_obs)
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = env_infos["final_observation"][idx]

            obs_rms.update(torch.tensor(obs, dtype=torch.float))
            rewards_rms.update(torch.tensor(rews, dtype=torch.float))

            # online_buffer.add_batch_transition(obs, acts, rews, next_obs, dones)
            current_batch = (
                torch.tensor(obs, dtype=torch.float32),
                torch.tensor(acts, dtype=torch.float32),
                torch.tensor(rews, dtype=torch.float32).unsqueeze(-1),
                torch.tensor(real_next_obs, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32).unsqueeze(-1),
                torch.zeros_like(torch.tensor(rews, dtype=torch.float32)).unsqueeze(-1),
            )
            online_buffer.add_batch_transition(
                current_batch[0],
                current_batch[1],
                current_batch[2],
                current_batch[3],
                current_batch[4],
            )

            if "final_info" in env_infos:
                returns = [info["episode"]["r"] for info in env_infos["final_info"]]
                return_mean = np.mean(returns)
                return_std = np.std(returns)
                train_return_log["train_return/return_mean"] = return_mean
                train_return_log["train_return/return_std"] = return_std
                wandb.log(train_return_log)
                train_return_log["train_return/step"] += 1

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
                os.path.join(buffer_saved_dir, "obs.txt"),
            )
            utils.save_data(
                {f"{step}": acts},
                os.path.join(buffer_saved_dir, "actions.txt"),
            )
            utils.save_data(
                {f"{step}": real_next_obs},
                os.path.join(buffer_saved_dir, "next_obs.txt"),
            )
            utils.save_data(
                {f"{step}": dones},
                os.path.join(buffer_saved_dir, "dones.txt"),
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
                pass
                # batch = offline_buffer.sample(config.batch_size)
                # batch = [b.to(config.device) for b in batch]
            else:
                # offline_batch = offline_buffer.sample(batch_size_offline)
                online_batch = online_buffer.sample(batch_size_online)
                batch = [
                    torch.vstack(tuple(b)).to(config.device)
                    for b in zip(current_batch, online_batch)
                ]
                batch[0] = normalize_obs(batch[0], obs_rms)
                batch[2] = normalize_reward(batch[2], rewards_rms)
                batch[3] = normalize_obs(batch[3], obs_rms)

                # batch = [
                #     torch.vstack(tuple(b)).to(config.device) for b in zip(offline_batch, online_batch)
                # ]

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
    torch.save({"obs_rms": obs_rms, "rewards_rms": rewards_rms}, rms_path)

    wandb.finish()


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


# From gymnasium/wrappers/normalize.py
class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = torch.zeros(shape, dtype=torch.float)
        self.var = torch.ones(shape, dtype=torch.float)
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_count = x.shape[0]
        batch_mean = torch.mean(x, axis=0)
        if batch_count == 1:
            batch_var = torch.zeros_like(batch_mean)
        else:
            batch_var = torch.var(x, axis=0)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    def __repr__(self):
        return f"RunningMeanStd(mean={self.mean}, var={self.var}, count={self.count})"


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def normalize_obs(
    observations: torch.Tensor,
    obs_rms: RunningMeanStd,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    mean = obs_rms.mean.to(observations.device)
    var = obs_rms.var.to(observations.device)
    return (observations - mean) / torch.sqrt(var + epsilon)


def normalize_reward(
    rewards: torch.Tensor,
    rewards_rms: RunningMeanStd,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    return rewards / ((rewards_rms.var).to(rewards.device) + epsilon)


@pyrallis.wrap()
def main(config: TrainConfig):
    sionna_config = utils.load_config(config.sionna_config_file)

    # set random seeds
    pytorch_utils.init_seed(config.seed)
    if config.verbose:
        utils.log_args(config)
        utils.log_config(sionna_config)

    if config.command.lower() == "train":
        envs = gym.vector.AsyncVectorEnv(
            [make_env(config, i, eval_mode=False) for i in range(config.num_envs)],
            context="spawn",
        )
    elif config.command.lower() == "eval":
        envs = gym.vector.AsyncVectorEnv(
            [make_env(config, i, eval_mode=True) for i in range(config.num_envs)],
            context="spawn",
        )
    else:
        raise ValueError(f"Invalid command: {config.command}, available commands: train, eval")

    # Init checkpoints
    print(f"Checkpoints path: {config.checkpoint_path}")
    os.makedirs(config.checkpoint_path, exist_ok=True)
    with open(os.path.join(config.checkpoint_path, "train_config.yaml"), "w") as f:
        pyrallis.dump(config, f)

    # Cal-QL trainer
    ob_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]

    critic_1 = calql.FullyConnectedQFunction(
        ob_dim,
        action_dim,
        config.orthogonal_init,
        config.q_n_hidden_layers,
    ).to(config.device)
    critic_2 = calql.FullyConnectedQFunction(
        ob_dim,
        action_dim,
        config.orthogonal_init,
        config.q_n_hidden_layers,
    ).to(config.device)
    critic_optimizer = torch.optim.AdamW(
        list(critic_1.parameters()) + list(critic_2.parameters()), config.qf_lr
    )
    critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        critic_optimizer,
        config.ep_len * config.n_updates,
        eta_min=config.qf_lr / 10,
    )

    actor = calql.TanhGaussianPolicy(
        ob_dim,
        action_dim,
        orthogonal_init=config.orthogonal_init,
    ).to(config.device)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), config.policy_lr)
    actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        actor_optimizer,
        config.ep_len * config.n_updates,
        eta_min=config.policy_lr / 10,
    )

    kwargs = {
        "critic_1": critic_1,
        "critic_2": critic_2,
        "critic_optimizer": critic_optimizer,
        "critic_scheduler": critic_scheduler,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "actor_scheduler": actor_scheduler,
        "discount": config.discount,
        "soft_target_update_rate": config.soft_target_update_rate,
        "device": config.device,
        # CQL
        "target_entropy": -np.prod(envs.single_action_space.shape).item(),
        "alpha_multiplier": config.alpha_multiplier,
        "use_automatic_entropy_tuning": config.use_automatic_entropy_tuning,
        "backup_entropy": config.backup_entropy,
        "policy_lr": config.policy_lr,
        "qf_lr": config.qf_lr,
        "bc_steps": config.bc_steps,
        "target_update_period": config.target_update_period,
        "cql_n_actions": config.cql_n_actions,
        "cql_importance_sample": config.cql_importance_sample,
        "cql_lagrange": config.cql_lagrange,
        "cql_target_action_gap": config.cql_target_action_gap,
        "cql_temp": config.cql_temp,
        "cql_alpha": config.cql_alpha,
        "cql_max_target_backup": config.cql_max_target_backup,
        "cql_clip_diff_min": config.cql_clip_diff_min,
        "cql_clip_diff_max": config.cql_clip_diff_max,
    }

    # Initialize actor
    trainer = CalQL(**kwargs)
    wandb_init(config)

    if config.command.lower() == "train":
        train(trainer, config, envs)
    elif config.command.lower() == "eval" and config.checkpoint_path != "":
        eval(trainer, config, envs)
    else:
        raise ValueError(f"Invalid command: {config.command}, available commands: train, eval")

    envs.close()


if __name__ == "__main__":
    main()
