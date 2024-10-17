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
from tensordict.nn import TensorDictModule
from torchrl.data import ReplayBuffer, LazyMemmapStorage

import saris
from saris.utils import utils, pytorch_utils, running_mean
from saris.drl.agents import sac

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
    load_replay_buffer: str = "-1"  # the path to load the replay buffer
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
        # mode="offline",
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

    ac_dim = math.prod(envs.single_action_space.shape)
    ob_dim = math.prod(envs.single_observation_space.shape)
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    # Init checkpoints
    print(f"Checkpoints dir: {config.checkpoint_dir}")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    with open(os.path.join(config.checkpoint_dir, "train_config.yaml"), "w") as f:
        pyrallis.dump(config, f)

    # Load models
    if config.load_model != "-1":
        print(f"Loading model from {config.load_model}")
        checkpoint = torch.load(config.load_model, weights_only=True)

    # Actor setup
    actor = sac.Actor(ob_dim, ac_dim, action_scale=2.0).to(config.device)
    if config.load_model != "-1":
        actor.load_state_dict(checkpoint["actor"])
    actor_detach = sac.Actor(ob_dim, ac_dim, action_scale=2.0).to(config.device)
    torchinfo.summary(
        actor_detach, (1, ob_dim), col_names=["input_size", "output_size", "num_params"]
    )
    # Copy params to actor_detach without grad
    from_module(actor).data.to_module(actor_detach)
    policy = TensorDictModule(actor_detach.get_action, in_keys=["observation"], out_keys=["action"])

    # Q function setup
    def get_q_params():
        qf1 = sac.SoftQNetwork(ob_dim, ac_dim).to(config.device)
        qf2 = sac.SoftQNetwork(ob_dim, ac_dim).to(config.device)
        torchinfo.summary(
            qf1, [(1, ob_dim), (1, ac_dim)], col_names=["input_size", "output_size", "num_params"]
        )
        qnet_params = from_modules(qf1, qf2, as_module=True)
        qnet_target_params = qnet_params.data.clone()

        # discard params of net
        qnet = sac.SoftQNetwork(ob_dim, ac_dim).to("meta")
        qnet_params.to_module(qnet)

        return qnet_params, qnet_target_params, qnet

    qnet_params, qnet_target_params, qnet = get_q_params()

    # Automatic entropy tuning
    target_entropy = -torch.prod(
        torch.Tensor(envs.single_action_space.shape).to(config.device)
    ).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=config.device)

    if config.load_model != "-1":
        qnet_params.load_state_dict(checkpoint["qnet_params"])
        qnet_target_params.load_state_dict(checkpoint["qnet_target_params"])
        log_alpha = checkpoint["log_alpha"].clone().detach().requires_grad_(True)
    alpha = log_alpha.detach().exp()
    a_optimizer = optim.AdamW([log_alpha], lr=config.q_lr)

    q_optimizer = optim.AdamW(qnet.parameters(), lr=config.q_lr)
    q_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        q_optimizer,
        config.n_updates * config.total_timesteps,
        eta_min=config.q_lr / 12,
    )

    actor_optimizer = optim.AdamW(list(actor.parameters()), lr=config.policy_lr)
    actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        actor_optimizer,
        config.n_updates * config.total_timesteps,
        eta_min=config.policy_lr / 10,
    )

    # replay buffer setup
    rb_dir = config.replay_buffer_dir
    rb = ReplayBuffer(
        storage=LazyMemmapStorage(config.buffer_size, scratch_dir=rb_dir),
        batch_size=config.batch_size,
    )
    if config.load_replay_buffer != "-1":
        print(f"Loading replay buffer from {config.load_replay_buffer}")
        rb.loads(config.load_replay_buffer)
        print(f"Replay buffer loaded with {len(rb)} samples")

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
                qnet_target_params, data["next_observations"], next_state_actions
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
        q_scheduler.step()
        return TensorDict(qf_loss=qf_loss.detach())

    def update_actor(data):
        actor_optimizer.zero_grad()
        pi, log_pi, _ = actor.get_action(data["observations"])
        qf_pi = torch.vmap(batched_qf, (0, None, None))(qnet_params.data, data["observations"], pi)
        min_qf_pi, _ = qf_pi.min(dim=0)
        actor_loss = torch.mean((alpha * log_pi) - min_qf_pi)

        actor_loss.backward()
        actor_optimizer.step()
        actor_scheduler.step()
        a_optimizer.zero_grad()
        with torch.no_grad():
            _, log_pi, _ = actor.get_action(data["observations"])
        alpha_loss = -torch.mean(log_alpha.exp() * (log_pi + target_entropy))

        alpha_loss.backward()
        a_optimizer.step()
        return TensorDict(
            alpha=alpha.detach(),
            actor_loss=actor_loss.detach(),
            alpha_loss=alpha_loss.detach(),
        )

    mode = "default"  # "reduce-overhead" if not config.cudagraphs else None
    update_critics = torch.compile(update_critics, mode=mode)
    update_actor = torch.compile(update_actor, mode=mode)
    policy = torch.compile(policy, mode=mode)

    # Create running meanstd for normalization
    obs_rms = running_mean.RunningMeanStd(shape=envs.single_observation_space.shape)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=config.seed)
    pbar = tqdm.tqdm(range(config.total_timesteps), dynamic_ncols=True)
    max_ep_ret = -float("inf")
    avg_returns = deque(maxlen=config.num_envs)
    desc = ""
    wandb_init(config)

    for global_step in pbar:

        # ALGO LOGIC: put action logic here
        if global_step < config.learning_starts * 2 / 3:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions = policy(torch.as_tensor(obs, device=config.device, dtype=torch.float))
            actions = actions.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        rewards = np.asarray(rewards, dtype=np.float32)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                r = float(info["episode"]["r"][0])
                max_ep_ret = max(max_ep_ret, r)
                avg_returns.append(r)

            avg_ret = torch.tensor(avg_returns).mean()
            std_ret = torch.tensor(avg_returns).std()
            log_dict = {"episodic_return": avg_ret, "episodic_return_std": std_ret}

            desc = f"global_step={global_step}, episodic_return={avg_ret: 4.2f} (max={max_ep_ret: 4.2f})"
            wandb.log(log_dict, step=global_step)

            path_gains = [info["path_gain"] for info in infos["final_info"]]
            next_path_gains = [info["next_path_gain"] for info in infos["final_info"]]
        else:
            path_gains = infos["path_gain"]
            next_path_gains = infos["next_path_gain"]
        path_gains = np.stack(path_gains)
        next_path_gains = np.stack(next_path_gains)
        path_gains = torch.as_tensor(path_gains, dtype=torch.float)
        next_path_gains = torch.as_tensor(next_path_gains, dtype=torch.float)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = copy.deepcopy(next_obs)
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        if global_step == config.total_timesteps - 1:
            truncations = [True] * len(truncations)
        transition = TensorDict(
            observations=obs,
            next_observations=real_next_obs,
            actions=actions,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            path_gains=path_gains,
            next_path_gains=next_path_gains,
            batch_size=obs.shape[0],
        )
        rb.extend(transition)
        obs_rms.update(torch.tensor(obs, dtype=torch.float))

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > config.learning_starts:

            log_infos = {}
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

                with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
                    log_infos.update(update_critics(data))
                if j % config.policy_frequency == 1:  # TD 3 Delayed update support
                    for _ in range(config.policy_frequency):
                        # compensate for the delay by doing 'actor_update_interval' instead of 1
                        with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
                            log_infos.update(update_actor(data))
                        alpha.copy_(log_alpha.detach().exp())

                # update the target networks
                if j % config.target_network_frequency == 0:
                    # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
                    qnet_target_params.lerp_(qnet_params.data, config.tau)

            if global_step > config.learning_starts + 3:
                with torch.no_grad():
                    q_lr = q_optimizer.param_groups[0]["lr"]
                    a_lr = actor_optimizer.param_groups[0]["lr"]
                    logs = {
                        "actor_loss": log_infos["actor_loss"].mean(),
                        "alpha_loss": log_infos.get("alpha_loss", 0).mean(),
                        "qf_loss": log_infos["qf_loss"].mean(),
                        "alpha": alpha.item(),
                        "q_lr": q_lr,
                        "a_lr": a_lr,
                    }
                wandb.log({**logs}, step=global_step)
                pbar.set_description(
                    desc
                    + f" | actor_loss={logs['actor_loss']: 4.3f} | qf_loss={logs['qf_loss']: 4.3f}"
                )

            if global_step % config.save_interval == 0:
                torch.save(
                    {
                        "actor": actor.state_dict(),
                        "qnet_params": qnet_params.state_dict(),
                        "qnet_target_params": qnet_target_params.state_dict(),
                        "log_alpha": log_alpha,
                    },
                    os.path.join(config.checkpoint_dir, f"model_{global_step}.pth"),
                )

    rb.dump(config.replay_buffer_dir)
    wandb.finish()
    envs.close()
    envs.close_extras()


if __name__ == "__main__":
    main()
