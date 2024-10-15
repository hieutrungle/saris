import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
from typing import Tuple, Callable, Dict, Optional, Union
import math
import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import wandb
import torchinfo
import importlib.resources
import copy
import pyrallis
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule, TensorDictModule
from torchrl.data import ReplayBuffer, LazyMemmapStorage

import saris
from saris.utils import utils, pytorch_utils, running_mean
from saris.drl.agents import calql

from saris.drl.envs import register_envs

register_envs()
torch.set_float32_matmul_precision("high")


@dataclass
class TrainConfig:
    # General arguments
    command: str = "train"  # the command to run
    load_model: str = "-1"  # Model load file name for resume training, "-1" doesn't load
    offline_data_dir: str = "-1"  # Offline data directory
    checkpoint_dir: str = "-1"  # the path to save the model
    replay_buffer_dir: str = "-1"  # the path to save the replay buffer
    verbose: bool = False  # whether to log to console
    seed: int = 1  # seed of the experiment
    eval_seed: int = 100  # seed of the evaluation
    save_interval: int = 5  # the interval to save the model

    # Environment specific arguments
    env_id: str = "wireless-sigmap-v0"  # the environment id of the task
    sionna_config_file: str = "-1"  # Sionna config file
    num_envs: int = 2  # the number of parallel environments
    ep_len: int = 75  # the maximum length of an episode
    eval_ep_len: int = 75  # the maximum length of an episode

    # Algorithm specific arguments
    total_timesteps: int = 1000000  # total timesteps of the experiments
    n_updates: int = 10  # the number of updates per step
    buffer_size: int = int(1e6)  # the replay memory buffer size
    gamma: float = 0.99  # the discount factor gamma
    tau: float = 0.005  # target smoothing coefficient (default: 0.005)
    batch_size: int = 256  # the batch size of sample from the reply memory
    learning_starts: int = 5e3  # the timestep to start learning
    policy_lr: float = 3e-4  # the learning rate of the policy network optimizer
    q_lr: float = 1e-3  # the learning rate of the q network optimizer
    policy_frequency: int = 2  # the frequency of training policy (delayed)
    target_network_frequency: int = 2  # the frequency of updates for the target nerworks
    alpha: float = 0.2  # Entropy regularization coefficient

    # Wandb logging
    project: str = "SARIS"  # wandb project name
    group: str = "SAC"  # wandb group name
    name: str = "Online-Learning"  # wandb run name

    def __post_init__(self):
        lib_dir = importlib.resources.files(saris)
        source_dir = os.path.dirname(lib_dir)
        self.source_dir = source_dir

        if self.checkpoint_dir == "-1":
            raise ValueError("Checkpoints dir is required for training")
        if self.replay_buffer_dir == "-1":
            raise ValueError("Replay buffer dir is required for training")
        if self.sionna_config_file == "-1":
            raise ValueError("Sionna config file is required for training")

        device = pytorch_utils.init_gpu()
        self.device = device


def wandb_init(config: TrainConfig) -> None:
    key_filename = os.path.join(config.source_dir, "tmp_wandb_api_key.txt")
    with open(key_filename, "r") as f:
        key_api = f.read().strip()
    wandb.login(relogin=True, key=key_api, host="https://api.wandb.ai")
    wandb.init(
        config=config,
        dir=config.checkpoint_dir,
        project=config.project,
        group=config.group,
        name=config.name,
        mode="offline",
    )


def make_env(config: TrainConfig, idx: int, eval_mode: bool) -> Callable:

    def thunk() -> gym.Env:

        seed = config.seed if not eval_mode else config.eval_seed
        max_episode_steps = config.ep_len if not eval_mode else config.eval_ep_len
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


def normalize_obs(
    observations: torch.Tensor,
    obs_rms: running_mean.RunningMeanStd,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    mean = obs_rms.mean.to(observations.device)
    var = obs_rms.var.to(observations.device)
    return (observations - mean) / torch.sqrt(var + epsilon)


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, n_act, n_obs, device=None):
        super().__init__()
        self.fc1 = nn.Linear(n_act + n_obs, 256, device=device)
        self.fc2 = nn.Linear(256, 256, device=device)
        self.fc3 = nn.Linear(256, 1, device=device)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, n_obs, n_act, device=None):
        super().__init__()
        self.fc1 = nn.Linear(n_obs, 256, device=device)
        self.fc2 = nn.Linear(256, 256, device=device)
        self.fc_mean = nn.Linear(256, n_act, device=device)
        self.fc_logstd = nn.Linear(256, n_act, device=device)
        # action rescaling
        self.register_buffer("action_scale", torch.tensor(2.0, dtype=torch.float32, device=device))
        self.register_buffer("action_bias", torch.tensor(0.0, dtype=torch.float32, device=device))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
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
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


@pyrallis.wrap()
def main(config: TrainConfig):

    torch.compiler.reset()
    sionna_config = utils.load_config(config.sionna_config_file)
    # set random seeds
    pytorch_utils.init_seed(config.seed)
    if config.verbose:
        utils.log_args(config)
        utils.log_config(sionna_config)

    # env setup
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

    n_act = math.prod(envs.single_action_space.shape)
    n_obs = math.prod(envs.single_observation_space.shape)
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    # Init checkpoints
    print(f"Checkpoints dir: {config.checkpoint_dir}")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    with open(os.path.join(config.checkpoint_dir, "train_config.yaml"), "w") as f:
        pyrallis.dump(config, f)

    actor = Actor(envs, n_act=n_act, n_obs=n_obs).to(config.device)
    actor_detach = Actor(envs, n_act=n_act, n_obs=n_obs).to(config.device)
    # Copy params to actor_detach without grad
    from_module(actor).data.to_module(actor_detach)
    policy = TensorDictModule(actor_detach.get_action, in_keys=["observation"], out_keys=["action"])

    def get_q_params():
        qf1 = SoftQNetwork(envs, n_act=n_act, n_obs=n_obs).to(config.device)
        qf2 = SoftQNetwork(envs, n_act=n_act, n_obs=n_obs).to(config.device)
        qnet_params = from_modules(qf1, qf2, as_module=True)
        qnet_target = qnet_params.data.clone()

        # discard params of net
        qnet = SoftQNetwork(envs, n_act=n_act, n_obs=n_obs).to("meta")
        qnet_params.to_module(qnet)

        return qnet_params, qnet_target, qnet

    qnet_params, qnet_target, qnet = get_q_params()

    q_optimizer = optim.Adam(qnet.parameters(), lr=config.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=config.policy_lr)

    # Automatic entropy tuning
    target_entropy = -torch.prod(
        torch.Tensor(envs.single_action_space.shape).to(config.device)
    ).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=config.device)
    alpha = log_alpha.detach().exp()
    a_optimizer = optim.Adam([log_alpha], lr=config.q_lr)

    # replay buffer setup
    rb_dir = config.replay_buffer_dir
    rb = ReplayBuffer(
        storage=LazyMemmapStorage(config.buffer_size, scratch_dir=rb_dir),
        batch_size=config.batch_size,
    )

    # functions to compile
    def batched_qf(params, obs, action, next_q_value=None):
        with params.to_module(qnet):
            vals = qnet(obs, action)
            if next_q_value is not None:
                loss_val = F.mse_loss(vals.view(-1), next_q_value)
                return loss_val
            return vals

    def update_critics(data):
        # optimize the model
        q_optimizer.zero_grad()
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = actor.get_action(data["next_observations"])
            qf_next_target = torch.vmap(batched_qf, (0, None, None))(
                qnet_target, data["next_observations"], next_state_actions
            )
            min_qf_next_target, _ = qf_next_target.min(dim=0)
            min_qf_next_target -= alpha * next_state_log_pi
            next_q_value = data["rewards"].flatten() + (
                (1.0 - data["terminations"].float()).flatten()
            ) * config.gamma * min_qf_next_target.view(-1)

        qf_a_values = torch.vmap(batched_qf, (0, None, None, None))(
            qnet_params, data["observations"], data["actions"], next_q_value
        )
        qf_loss = qf_a_values.sum(0)

        qf_loss.backward()
        q_optimizer.step()
        return TensorDict(qf_loss=qf_loss.detach())

    def update_actor(data):
        actor_optimizer.zero_grad()
        pi, log_pi, _ = actor.get_action(data["observations"])
        qf_pi = torch.vmap(batched_qf, (0, None, None))(qnet_params.data, data["observations"], pi)
        min_qf_pi, _ = qf_pi.min(dim=0)
        actor_loss = torch.mean((alpha * log_pi) - min_qf_pi)

        actor_loss.backward()
        actor_optimizer.step()

        a_optimizer.zero_grad()
        with torch.no_grad():
            _, log_pi, _ = actor.get_action(data["observations"])
        alpha_loss = -torch.mean(log_alpha.exp() * (log_pi + target_entropy))

        alpha_loss.backward()
        a_optimizer.step()
        return TensorDict(
            alpha=alpha.detach(), actor_loss=actor_loss.detach(), alpha_loss=alpha_loss.detach()
        )

    mode = None  # "reduce-overhead" if not config.cudagraphs else None
    update_critics = torch.compile(update_critics, mode=mode)
    update_actor = torch.compile(update_actor, mode=mode)
    policy = torch.compile(policy, mode=mode)

    # Create running meanstd for normalization
    obs_rms = running_mean.RunningMeanStd(shape=envs.single_observation_space.shape)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=config.seed)
    # obs = torch.as_tensor(obs, device=device, dtype=torch.float)
    pbar = tqdm.tqdm(range(config.total_timesteps))
    max_ep_ret = -float("inf")
    avg_returns = deque(maxlen=config.num_envs)
    desc = ""
    wandb_init(config)
    log_infos = {}

    for global_step in pbar:

        # ALGO LOGIC: put action logic here
        if global_step < config.learning_starts * 2 / 3:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions = policy(torch.as_tensor(obs, device=config.device, dtype=torch.float))
            actions = actions.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                r = float(info["episode"]["r"][0])
                max_ep_ret = max(max_ep_ret, r)
                avg_returns.append(r)

            desc = f"global_step={global_step}, episodic_return={torch.tensor(avg_returns).mean(): 4.2f} (max={max_ep_ret: 4.2f})"
            wandb.log({"episodic_return": torch.tensor(avg_returns).mean()}, step=global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = copy.deepcopy(next_obs)
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        transition = TensorDict(
            observations=obs,
            next_observations=real_next_obs,
            actions=actions,
            rewards=rewards,
            terminations=terminations,
            dones=terminations,
            batch_size=obs.shape[0],
        )
        rb.extend(transition)
        obs_rms.update(torch.tensor(obs, dtype=torch.float))

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > config.learning_starts:

            for j in range(config.n_updates):
                data = rb.sample()
                data = TensorDict(
                    {
                        k: torch.as_tensor(v, device=config.device, dtype=torch.float)
                        for k, v in data.items()
                    }
                )
                data["observations"] = normalize_obs(data["observations"], obs_rms)
                data["next_observations"] = normalize_obs(data["next_observations"], obs_rms)

                log_infos.update(update_critics(data))
                if j % config.policy_frequency == 1:  # TD 3 Delayed update support
                    for _ in range(
                        config.policy_frequency
                    ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        log_infos.update(update_actor(data))
                        alpha.copy_(log_alpha.detach().exp())

                # update the target networks
                if j % config.target_network_frequency == 0:
                    # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
                    qnet_target.lerp_(qnet_params.data, config.tau)

            if global_step % 2 == 0 and global_step > config.learning_starts + 4:
                with torch.no_grad():
                    logs = {
                        "actor_loss": log_infos["actor_loss"].mean(),
                        "alpha_loss": log_infos.get("alpha_loss", 0).mean(),
                        "qf_loss": log_infos["qf_loss"].mean(),
                    }
                wandb.log({**logs}, step=global_step)
                pbar.set_description(
                    desc
                    + f" | actor_loss={logs['actor_loss']: 4.2f} | qf_loss={logs['qf_loss']: 4.2f}"
                )

            if global_step % config.save_interval == 0:
                torch.save(
                    {
                        "actor": actor.state_dict(),
                        "qnet_params": qnet_params.state_dict(),
                        "qnet_target": qnet_target.state_dict(),
                        "log_alpha": log_alpha,
                    },
                    os.path.join(config.checkpoint_dir, f"model_{global_step}.pth"),
                )

    wandb.finish()
    envs.close()
    envs.close_extras()


if __name__ == "__main__":
    main()
