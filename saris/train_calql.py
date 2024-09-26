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

import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import saris
from saris.drl.agents import calql
from saris.utils import utils, pytorch_utils
import importlib
import wandb
import gymnasium as gym
from saris.drl.envs import register_envs


register_envs()
TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    # Experiment
    env_id: str = "wireless-sigmap-v0"  # environment name
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    offline_iterations: int = int(0)  # Number of offline updates
    # offline_iterations: int = int(1e6)  # Number of offline updates
    online_iterations: int = int(1e6)  # Number of online updates
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    sionna_config_file: str = ""  # Sionna config file
    verbose: bool = False  # Print debug information

    # Environment
    ep_len: int = 1000  # Max length of episode
    num_envs: int = 1  # Number of parallel environments
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_seed: int = 0  # Eval environment seed

    # CQL
    buffer_size: int = 4_000  # Replay buffer size
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
    reward_scale: float = 0.8  # Reward scale for normalization
    reward_bias: float = 0.0  # Reward bias for normalization

    # Cal-QL
    mixing_ratio: float = 0.0  # Data mixing ratio for online tuning, should be 0.1
    is_sparse_reward: bool = False  # Use sparse reward

    # Wandb logging
    project: str = "SARIS"  # wandb project name
    group: str = "Cal-QL"  # wandb group name
    name: str = "Data-Collection"  # wandb run name

    def __post_init__(self):
        self.name = (
            f"{self.project}__{self.group}__{self.name}__{self.env_id}__{str(uuid.uuid4())[:8]}"
        )
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

        lib_dir = importlib.resources.files(saris)
        source_dir = os.path.dirname(lib_dir)
        self.source_dir = source_dir

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
        seed += idx
        env = gym.make(
            config.env_id,
            idx=idx,
            sionna_config_file=config.sionna_config_file,
            log_string=config.name,
            eval_mode=eval_mode,
            seed=seed,
            max_episode_steps=config.ep_len,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=config.ep_len)
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


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._mc_returns = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError("Replay buffer is smaller than the dataset you are trying to load!")
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._mc_returns[:n_transitions] = self._to_tensor(data["mc_returns"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        mc_returns = self._mc_returns[indices]
        return [states, actions, rewards, next_states, dones, mc_returns]

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._dones[self._pointer] = self._to_tensor(done)
        self._mc_returns[self._pointer] = 0.0

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)


def set_env_seed(env: Optional[gym.Env], seed: int):
    env.seed(seed)
    env.action_space.seed(seed)


def set_seed(seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False):
    if env is not None:
        set_env_seed(env, seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


def is_goal_reached(reward: float, info: Dict) -> bool:
    if "goal_achieved" in info:
        return info["goal_achieved"]
    return reward > 0  # Assuming that reaching target is a positive reward


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    successes = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        goal_achieved = False
        while not done:
            action = actor.act(state, device)
            state, reward, done, env_infos = env.step(action)
            episode_reward += reward
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
                # Valid only for environments with goal
        successes.append(float(goal_achieved))
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards), np.mean(successes)


def return_reward_range(dataset: Dict, max_episode_steps: int) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def get_return_to_go(dataset: Dict, env: gym.Env, config: TrainConfig) -> np.ndarray:
    returns = []
    ep_ret, ep_len = 0.0, 0
    cur_rewards = []
    terminals = []
    N = len(dataset["rewards"])
    for t, (r, d) in enumerate(zip(dataset["rewards"], dataset["terminals"])):
        ep_ret += float(r)
        cur_rewards.append(float(r))
        terminals.append(float(d))
        ep_len += 1
        is_last_step = (
            (t == N - 1)
            or (
                np.linalg.norm(dataset["observations"][t + 1] - dataset["next_observations"][t])
                > 1e-6
            )
            or ep_len == env._max_episode_steps
        )

        if d or is_last_step:
            discounted_returns = [0] * ep_len
            prev_return = 0
            if (
                config.is_sparse_reward
                and r == env.ref_min_score * config.reward_scale + config.reward_bias
            ):
                discounted_returns = [r / (1 - config.discount)] * ep_len
            else:
                for i in reversed(range(ep_len)):
                    discounted_returns[i] = cur_rewards[i] + config.discount * prev_return * (
                        1 - terminals[i]
                    )
                    prev_return = discounted_returns[i]
            returns += discounted_returns
            ep_ret, ep_len = 0.0, 0
            cur_rewards = []
            terminals = []
    return returns


def modify_reward(
    dataset: Dict,
    env_name: str,
    max_episode_steps: int = 1000,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
) -> Dict:
    modification_data = {}
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
        modification_data = {
            "max_ret": max_ret,
            "min_ret": min_ret,
            "max_episode_steps": max_episode_steps,
        }
    dataset["rewards"] = dataset["rewards"] * reward_scale + reward_bias
    return modification_data


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
        critic_1,
        critic_1_optimizer,
        critic_2,
        critic_2_optimizer,
        actor,
        actor_optimizer,
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

        self.actor_optimizer = actor_optimizer
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2_optimizer = critic_2_optimizer

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
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            mc_returns,
        ) = batch
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
            observations,
            actions,
            next_observations,
            rewards,
            dones,
            mc_returns,
            alpha,
            log_dict,
        )

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic_1.state_dict(),
            "critic2": self.critic_2.state_dict(),
            "critic1_target": self.target_critic_1.state_dict(),
            "critic2_target": self.target_critic_2.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
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

        self.critic_1_optimizer.load_state_dict(state_dict=state_dict["critic_1_optimizer"])
        self.critic_2_optimizer.load_state_dict(state_dict=state_dict["critic_2_optimizer"])
        self.actor_optimizer.load_state_dict(state_dict=state_dict["actor_optim"])

        self.log_alpha = state_dict["sac_log_alpha"]
        self.alpha_optimizer.load_state_dict(state_dict=state_dict["sac_log_alpha_optim"])

        self.log_alpha_prime = state_dict["cql_log_alpha"]
        self.alpha_prime_optimizer.load_state_dict(state_dict=state_dict["cql_log_alpha_optim"])
        self.total_it = state_dict["total_it"]


def train(config: TrainConfig, env: gym.Env, eval_env: gym.Env) -> None:

    # is_env_with_goal = config.env.startswith(ENVS_WITH_GOAL)
    batch_size_offline = int(config.batch_size * config.mixing_ratio)
    batch_size_online = config.batch_size - batch_size_offline

    max_steps = env.spec.max_episode_steps

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

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
    # env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    # eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)
    # offline_buffer = ReplayBuffer(
    #     state_dim,
    #     action_dim,
    #     config.buffer_size,
    #     config.device,
    # )
    online_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
    )
    # offline_buffer.load_d4rl_dataset(dataset)

    log_dir = os.path.join(config.source_dir, "local_assets", "logs")
    log_path = os.path.join(log_dir, config.name)
    print(f"Checkpoints path: {log_path}")
    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, "config.yaml"), "w") as f:
        pyrallis.dump(config, f)

    critic_1 = calql.FullyConnectedQFunction(
        state_dim,
        action_dim,
        config.orthogonal_init,
        config.q_n_hidden_layers,
    ).to(config.device)
    critic_2 = calql.FullyConnectedQFunction(
        state_dim,
        action_dim,
        config.orthogonal_init,
        config.q_n_hidden_layers,
    ).to(config.device)
    critic_1_optimizer = torch.optim.AdamW(list(critic_1.parameters()), config.qf_lr)
    critic_2_optimizer = torch.optim.AdamW(list(critic_2.parameters()), config.qf_lr)

    actor = calql.TanhGaussianPolicy(
        state_dim,
        action_dim,
        orthogonal_init=config.orthogonal_init,
    ).to(config.device)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), config.policy_lr)

    kwargs = {
        "critic_1": critic_1,
        "critic_2": critic_2,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2_optimizer": critic_2_optimizer,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config.discount,
        "soft_target_update_rate": config.soft_target_update_rate,
        "device": config.device,
        # CQL
        "target_entropy": -np.prod(env.action_space.shape).item(),
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

    print("---------------------------------------")
    print(
        f"Training Cal-QL, Env: {config.env_id}, Training Seed: {config.seed}, Env Seed: {config.env_seed}"
    )
    print("---------------------------------------")

    # Initialize actor
    trainer = CalQL(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    evaluations = []
    state, info = env.reset()
    done = False
    episode_return = 0
    episode_step = 0
    # goal_achieved = False

    eval_successes = []
    train_successes = []

    if config.offline_iterations > 0:
        print("Offline pretraining")
    else:
        print(f"No offline pretraining, starting online training")
    exit()
    for t in range(int(config.offline_iterations) + int(config.online_iterations)):
        if t == config.offline_iterations:
            print("Online tuning")
            trainer.switch_calibration()
            trainer.cql_alpha = config.cql_alpha_online
        online_log = {}
        if t >= config.offline_iterations:
            episode_step += 1
            action, _ = actor(
                torch.tensor(
                    state.reshape(1, -1),
                    device=config.device,
                    dtype=torch.float32,
                )
            )
            action = action.cpu().data.numpy().flatten()
            next_state, reward, done, env_infos = env.step(action)

            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
            episode_return += reward
            real_done = False  # Episode can timeout which is different from done
            if done and episode_step < max_steps:
                real_done = True

            if config.normalize_reward:
                reward = modify_reward_online(
                    reward,
                    config.env,
                    reward_scale=config.reward_scale,
                    reward_bias=config.reward_bias,
                    **reward_mod_dict,
                )
            online_buffer.add_transition(state, action, reward, next_state, real_done)
            state = next_state

            if done:
                state, done = env.reset(), False
                # Valid only for envs with goal, e.g. AntMaze, Adroit
                if is_env_with_goal:
                    train_successes.append(goal_achieved)
                    online_log["train/regret"] = np.mean(1 - np.array(train_successes))
                    online_log["train/is_success"] = float(goal_achieved)
                online_log["train/episode_return"] = episode_return
                normalized_return = eval_env.get_normalized_score(episode_return)
                online_log["train/d4rl_normalized_episode_return"] = normalized_return * 100.0
                online_log["train/episode_length"] = episode_step
                episode_return = 0
                episode_step = 0
                goal_achieved = False

        if t < config.offline_iterations:
            batch = offline_buffer.sample(config.batch_size)
            batch = [b.to(config.device) for b in batch]
        else:
            offline_batch = offline_buffer.sample(batch_size_offline)
            online_batch = online_buffer.sample(batch_size_online)
            batch = [
                torch.vstack(tuple(b)).to(config.device) for b in zip(offline_batch, online_batch)
            ]

        log_dict = trainer.train(batch)
        log_dict["offline_iter" if t < config.offline_iterations else "online_iter"] = (
            t if t < config.offline_iterations else t - config.offline_iterations
        )
        log_dict.update(online_log)
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores, success_rate = eval_actor(
                eval_env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            eval_log = {}
            normalized = eval_env.get_normalized_score(np.mean(eval_scores))
            # Valid only for envs with goal, e.g. AntMaze, Adroit
            if t >= config.offline_iterations and is_env_with_goal:
                eval_successes.append(success_rate)
                eval_log["eval/regret"] = np.mean(1 - np.array(train_successes))
                eval_log["eval/success_rate"] = success_rate
            normalized_eval_score = normalized * 100.0
            eval_log["eval/d4rl_normalized_score"] = normalized_eval_score
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            print("---------------------------------------")
            if config.checkpoints_path:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            wandb.log(eval_log, step=trainer.total_it)


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
        batch_mean = torch.mean(x, axis=0)
        batch_var = torch.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


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


def create_running_meanstd_for_dict(
    ob_space: gym.spaces.Dict,
) -> Dict[str, RunningMeanStd]:
    dict_shape = {k: v.shape for k, v in ob_space.items()}
    return {k: RunningMeanStd(shape=v) for k, v in dict_shape.items()}


def update_rms_dict(rms_dict: Dict[str, RunningMeanStd], data: Dict[str, torch.Tensor]):
    for k, v in data.items():
        rms_dict[k].update(torch.tensor(v, dtype=torch.float))
    return rms_dict


def normalize_obs(
    observations: Dict[str, torch.Tensor],
    obs_rms: Dict[str, RunningMeanStd],
) -> Dict[str, torch.Tensor]:
    for k, v in observations.items():
        device = v.device
        obs_rms[k].update(v.detach().cpu())
        mean = (obs_rms[k].mean).to(device)
        std = torch.sqrt(obs_rms[k].var).to(device)
        mean = mean.repeat(v.shape[0], 1)
        std = std.repeat(v.shape[0], 1)
        observations[k] = (v - mean) / (std + 1e-8)
    return observations


def denormalize_obs(
    observations: Dict[str, torch.Tensor],
    obs_rms: Dict[str, RunningMeanStd],
) -> Dict[str, torch.Tensor]:
    for k, v in observations.items():
        device = v.device
        mean = (obs_rms[k].mean).to(device)
        std = torch.sqrt(obs_rms[k].var).to(device)
        mean = mean.repeat(v.shape[0], 1)
        std = std.repeat(v.shape[0], 1)
        observations[k] = (v * (std + 1e-8)) + mean
    return observations


@pyrallis.wrap()
def main(config: TrainConfig):
    sionna_config = utils.load_config(config.sionna_config_file)

    # set random seeds
    pytorch_utils.init_seed(config.seed)
    if config.verbose:
        utils.log_args(config)
        utils.log_config(sionna_config)

    env = make_env(config, idx=0, eval_mode=False)()
    eval_env = make_env(config, idx=0, eval_mode=True)()

    train(config, env, eval_env)


if __name__ == "__main__":
    main()
