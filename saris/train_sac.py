import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
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
from tensordict.nn import CudaGraphModule
from torchrl.data import ReplayBuffer, LazyMemmapStorage
import traceback
import saris
from saris.utils import utils, pytorch_utils, running_mean
from saris.drl.agents import sac
import matplotlib.pyplot as plt
from saris.drl.envs import register_envs

register_envs()
torch.set_float32_matmul_precision("high")


@dataclass
class TrainConfig:

    # General arguments
    command: str = "train"  # the command to run
    load_model: str = "-1"  # Model load file name for resume training, "-1" doesn't load
    load_eval_model: str = "-1"  # Model load file name for evaluation, "-1" doesn't load
    checkpoint_dir: str = "-1"  # the path to save the model
    replay_buffer_dir: str = "-1"  # the path to save the replay buffer
    load_replay_buffer: str = "-1"  # the path to load the replay buffer
    verbose: bool = False  # whether to log to console
    seed: int = 1  # seed of the experiment
    eval_seed: int = 111  # seed of the evaluation
    save_interval: int = 100  # the interval to save the model

    # Environment specific arguments
    env_id: str = "wireless-sigmap-v0"  # the environment id of the task
    sionna_config_file: str = "-1"  # Sionna config file
    num_envs: int = 8  # the number of parallel environments
    ep_len: int = 75  # the maximum length of an episode
    eval_ep_len: int = 45  # the maximum length of an episode

    # Algorithm specific arguments
    total_timesteps: int = 10_001  # total timesteps of the experiments
    n_updates: int = 20  # the number of updates per step
    buffer_size: int = int(80_000)  # the replay memory buffer size
    gamma: float = 0.99  # the discount factor gamma
    tau: float = 0.005  # target smoothing coefficient (default: 0.005)
    batch_size: int = 256  # the batch size of sample from the reply memory
    learning_starts: int = 601  # the timestep to start learning
    policy_lr: float = 3e-4  # the learning rate of the policy network optimizer
    q_lr: float = 1e-3  # the learning rate of the q network optimizer
    warmup_steps: int = 500  # the number of warmup steps
    policy_frequency: int = 2  # the frequency of training policy (delayed)
    target_network_frequency: int = 2  # the frequency of updates for the target nerworks
    alpha: float = 0.2  # Entropy regularization coefficient

    # Wandb logging
    wandb_mode: str = "online"  # wandb mode
    project: str = "SARIS"  # wandb project name
    group: str = "SAC"  # wandb group name
    name: str = "Reward_split"  # wandb run name

    def __post_init__(self):
        lib_dir = importlib.resources.files(saris)
        source_dir = os.path.dirname(lib_dir)
        self.source_dir = source_dir

        if self.checkpoint_dir == "-1":
            raise ValueError("Checkpoints dir is required for training")
        if self.sionna_config_file == "-1":
            raise ValueError("Sionna config file is required for training")
        if self.command.lower() == "train" and self.replay_buffer_dir == "-1":
            raise ValueError("Replay buffer dir is required for training")
        if self.command.lower() == "eval" and self.load_eval_model == "-1":
            raise ValueError("Load eval model is required for evaluation")

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
        mode=config.wandb_mode,
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
        # env = gym.wrappers.FlattenObservation(env)
        env.action_space.seed(config.seed)
        env.observation_space.seed(config.seed)

        return env

    return thunk


# def normalize_obs(
#     observations: torch.Tensor,
#     obs_rms: running_mean.RunningMeanStd,
#     epsilon: float = 1e-8,
# ) -> torch.Tensor:
#     mean = obs_rms.mean.to(observations.device)
#     var = obs_rms.var.to(observations.device)
# return (observations - mean) / torch.sqrt(var + epsilon)


def normalize_obs(
    flat_obs: torch.Tensor,
    real_channel_rms: running_mean.RunningMeanStd,
    imag_channel_rms: running_mean.RunningMeanStd,
    epsilon: float = 1e-10,
):

    real_channel_len = real_channel_rms.mean.shape[0]
    real_channels = flat_obs[..., :real_channel_len]
    # whittening
    # real_mean = real_channel_rms.mean.to(flat_obs.device)
    real_var = real_channel_rms.var.to(flat_obs.device)
    # real_channels = (real_channels - real_mean) / torch.sqrt(real_var + epsilon)
    real_channels = (real_channels) / torch.sqrt(real_var + epsilon)
    # scaling
    # min_ = real_channel_rms.min.to(flat_obs.device)
    # max_ = real_channel_rms.max.to(flat_obs.device)
    # real_channels = (real_channels - min_) / (max_ - min_ + epsilon)

    imag_channel_len = imag_channel_rms.mean.shape[0]
    imag_channels = flat_obs[..., real_channel_len : real_channel_len + imag_channel_len]
    # whittening
    # imag_mean = imag_channel_rms.mean.to(flat_obs.device)
    imag_var = imag_channel_rms.var.to(flat_obs.device)
    # imag_channels = (imag_channels - imag_mean) / torch.sqrt(imag_var + epsilon)
    imag_channels = (imag_channels) / torch.sqrt(imag_var + epsilon)
    # scaling
    # min_ = imag_channel_rms.min.to(flat_obs.device)
    # max_ = imag_channel_rms.max.to(flat_obs.device)
    # imag_channels = (imag_channels - min_) / (max_ - min_ + epsilon)

    # angles
    angle_len = 72
    angles = flat_obs[
        ..., real_channel_len + imag_channel_len : real_channel_len + imag_channel_len + angle_len
    ]
    init_angles = [math.radians(135.0)] + [math.radians(90.0)] * 7
    init_angles = np.concatenate([init_angles] * 9)
    # offset
    angles = torch.sub(angles, torch.tensor(init_angles, device=angles.device, dtype=angles.dtype))
    # # normalize
    # angles = torch.div(torch.rad2deg(angles), 45.0)

    pos = flat_obs[..., real_channel_len + imag_channel_len + angle_len :]
    flat_obs = torch.cat([real_channels, imag_channels, angles, pos], dim=-1)
    return flat_obs.float()


def update_channel_rmss(
    flat_obs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    real_channel_rms: running_mean.RunningMeanStd,
    imag_channel_rms: running_mean.RunningMeanStd,
):
    real_channel_len = np.prod(real_channel_rms.mean.shape)
    real_channel_rms.update(flat_obs[..., :real_channel_len])
    imag_channel_len = np.prod(imag_channel_rms.mean.shape)
    imag_channel_rms.update(flat_obs[..., real_channel_len : real_channel_len + imag_channel_len])


def create_scheduler(optimizer, warmup_steps, num_train_steps, lr):
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1 / 10, total_iters=warmup_steps
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, num_train_steps - warmup_steps, eta_min=lr / 10
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_scheduler, cosine_scheduler], [warmup_steps]
    )
    return scheduler


def preprocess_actions(actions, action_scale):
    act_shape = actions.shape
    last_dim = actions.shape[-1]
    all_but_last_dim = actions.shape[:-1]
    actions = actions.reshape(*all_but_last_dim, last_dim // 3, 3)
    actions[..., 0] = torch.mul(actions[..., 0], 3.0)
    actions[..., 1:] = torch.rad2deg(actions[..., 1:])
    actions = actions / action_scale
    actions = actions.reshape(act_shape)
    return actions


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

    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"
    print(f"Observation space: {envs.single_observation_space}")
    print(f"Action space: {envs.single_action_space}")
    ob_space = envs.single_observation_space
    ac_space = envs.single_action_space

    # Create running meanstd for normalization
    real_channel_len = math.prod(envs.single_observation_space[0].shape)
    imag_channel_len = math.prod(envs.single_observation_space[1].shape)
    real_channel_rms = running_mean.RunningMeanStd(shape=(real_channel_len,))
    imag_channel_rms = running_mean.RunningMeanStd(shape=(imag_channel_len,))
    obs_rmss = (real_channel_rms, imag_channel_rms)

    # Init checkpoints
    print(f"Checkpoints dir: {config.checkpoint_dir}")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    with open(os.path.join(config.checkpoint_dir, "train_config.yaml"), "w") as f:
        pyrallis.dump(config, f)

    # Load models
    checkpoint = None
    if config.command.lower() == "eval":
        print(f"Loading model from {config.load_eval_model}")
        checkpoint = torch.load(config.load_eval_model, weights_only=False)
    else:
        if config.load_model != "-1":
            print(f"Loading model from {config.load_model}")
            checkpoint = torch.load(config.load_model, weights_only=False)

    # Actor setup
    actor = sac.Actor(ob_space, ac_space, envs=envs, device=config.device)
    if checkpoint != None:
        print(f"Loading actor from checkpoint!")
        actor.load_state_dict(checkpoint["actor"])
    actor_detach = sac.Actor(ob_space, ac_space, envs=envs, device=config.device)
    total_ob_dim = sum([math.prod(ob.shape) for ob in ob_space])
    tmp_obs = torch.randn((1, total_ob_dim), device=config.device)
    torchinfo.summary(
        actor_detach,
        input_data=tmp_obs,
        col_names=["input_size", "output_size", "num_params"],
    )
    # Copy params to actor_detach without grad
    from_module(actor).data.to_module(actor_detach)
    policy = TensorDictModule(actor_detach.get_action, in_keys=["observation"], out_keys=["action"])

    # Q function setup
    def get_q_params():
        qf1 = sac.SoftQNetwork(ob_space, ac_space, config.device)
        qf2 = sac.SoftQNetwork(ob_space, ac_space, config.device)
        tmp_ac = torch.randn(1, *ac_space.shape, device=config.device)
        torchinfo.summary(
            qf1,
            input_data=[tmp_obs, tmp_ac],
            col_names=["input_size", "output_size", "num_params"],
        )
        qnet_params = from_modules(qf1, qf2, as_module=True)
        qnet_target_params = qnet_params.data.clone()

        # discard params of net
        qnet = sac.SoftQNetwork(ob_space, ac_space, device="meta")
        qnet_params.to_module(qnet)

        return qnet_params, qnet_target_params, qnet

    qnet_params, qnet_target_params, qnet = get_q_params()

    del tmp_obs

    # Automatic entropy tuning
    target_entropy = -torch.prod(
        torch.Tensor(envs.single_action_space.shape).to(config.device)
    ).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=config.device)

    if checkpoint != None:
        print(f"Loading qnet and rmss from checkpoint!")
        obs_rmss = checkpoint["obs_rmss"]
        qnet_params.load_state_dict(checkpoint["qnet_params"])
        qnet_target_params.load_state_dict(checkpoint["qnet_target_params"])
        qnet_params.to_module(qnet)
        log_alpha = checkpoint["log_alpha"].clone().detach().requires_grad_(True)

    alpha = log_alpha.detach().exp()
    a_optimizer = optim.AdamW([log_alpha], lr=config.q_lr)

    q_optimizer = optim.AdamW(
        qnet_params.values(include_nested=True, leaves_only=True), lr=config.q_lr, capturable=True
    )
    q_scheduler = create_scheduler(
        q_optimizer,
        config.n_updates * config.warmup_steps,
        config.n_updates * config.total_timesteps,
        config.q_lr,
    )

    actor_optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=config.policy_lr,
        #   capturable=True
    )
    warmup_steps = int(config.n_updates * config.warmup_steps)
    total_train_steps = int(config.n_updates * config.total_timesteps)
    actor_scheduler = create_scheduler(
        actor_optimizer, warmup_steps, total_train_steps, config.policy_lr
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
        stored_obs = np.asarray(rb.storage.get("observations"))
        update_channel_rmss(torch.tensor(stored_obs), obs_rmss[0], obs_rmss[1])
        print(f"updated obs_rms: {obs_rmss}")

    envs.single_observation_space.dtype = np.float32

    if config.command.lower() == "train":
        try:
            train_agent(
                config,
                envs,
                obs_rmss,
                actor,
                policy,
                qnet_params,
                qnet_target_params,
                qnet,
                target_entropy,
                log_alpha,
                alpha,
                a_optimizer,
                q_optimizer,
                q_scheduler,
                actor_optimizer,
                actor_scheduler,
                rb,
            )
        except Exception as e:
            traceback.print_exc()
            raise e
        finally:
            rb.dump(config.replay_buffer_dir)
            wandb.finish()
            envs.close()
            envs.close_extras()
    elif config.command.lower() == "eval":
        eval(
            config,
            envs,
            obs_rmss,
            actor,
        )
        envs.close()
        envs.close_extras()
    else:
        raise ValueError(f"Invalid command: {config.command}, available commands: train, eval")


def train_agent(
    config: TrainConfig,
    envs: gym.vector.AsyncVectorEnv,
    obs_rmss: running_mean.RunningMeanStd,
    actor: sac.Actor,
    policy: Callable,
    qnet_params: TensorDict,
    qnet_target_params: TensorDict,
    qnet: sac.SoftQNetwork,
    target_entropy: float,
    log_alpha: torch.Tensor,
    alpha: torch.Tensor,
    a_optimizer: torch.optim.Optimizer,
    q_optimizer: torch.optim.Optimizer,
    q_scheduler: torch.optim.lr_scheduler._LRScheduler,
    actor_optimizer: torch.optim.Optimizer,
    actor_scheduler: torch.optim.lr_scheduler._LRScheduler,
    rb: ReplayBuffer,
):
    wandb_init(config)

    action_low, action_high = envs.single_action_space.low, envs.single_action_space.high
    action_low = torch.tensor(action_low, device=config.device, dtype=torch.float)
    action_high = torch.tensor(action_high, device=config.device, dtype=torch.float)

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
            min_qf_next_target = qf_next_target.min(0).values
            min_qf_next_target -= alpha * next_state_log_pi
            next_q_value = data["rewards"].flatten() + (
                (1.0 - data["terminations"].float()).flatten()
            ) * config.gamma * min_qf_next_target.view(-1)

        qf_a_values = torch.vmap(batched_qf, (0, None, None, None))(
            qnet_params, data["observations"], data["actions"], next_q_value
        )
        qf_loss = qf_a_values.sum(0)

        qf_loss.backward()
        torch.nn.utils.clip_grad_norm_(qnet_params, 1.0)
        q_optimizer.step()
        return TensorDict(qf_loss=qf_loss.detach())

    def update_actor(data):
        actor_optimizer.zero_grad()
        pi, log_pi, _ = actor.get_action(data["observations"])
        qf_pi = torch.vmap(batched_qf, (0, None, None))(qnet_params.data, data["observations"], pi)
        min_qf_pi = torch.min(qf_pi[0], qf_pi[1])
        # min_qf_pi = qf_pi.min(0).values
        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        actor_optimizer.step()

        a_optimizer.zero_grad()
        with torch.no_grad():
            _, log_pi, _ = actor.get_action(data["observations"])
        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
        alpha_loss.backward()
        a_optimizer.step()

        return TensorDict(
            alpha=alpha.detach(),
            actor_loss=actor_loss.detach(),
            alpha_loss=alpha_loss.detach(),
            actor_entropy=-(log_pi).mean().detach(),
        )

    mode = "default"  # "reduce-overhead" if not config.cudagraphs else None
    update_critics = torch.compile(update_critics, mode=mode)
    update_actor = torch.compile(update_actor, mode=mode)
    policy = torch.compile(policy, mode=mode)

    update_critics = CudaGraphModule(update_critics, in_keys=[], out_keys=[], warmup=5)
    update_actor = CudaGraphModule(update_actor, in_keys=[], out_keys=[], warmup=5)
    policy = CudaGraphModule(policy)

    # eval env setup
    eval_envs = gym.vector.AsyncVectorEnv(
        [make_env(config, i, eval_mode=False) for i in range(3)],
        context="spawn",
    )

    # TRY NOT TO MODIFY: start the game
    stored_flat_obs = []
    obs, _ = envs.reset(seed=config.seed)
    flat_obs = np.concatenate([ob.reshape(ob.shape[0], -1) for ob in obs], axis=-1)
    stored_flat_obs.append(flat_obs)
    pbar = tqdm.tqdm(range(config.total_timesteps), dynamic_ncols=True)
    max_ep_ret = -float("inf")
    avg_returns = deque(maxlen=envs.num_envs)
    desc = ""

    for global_step in pbar:
        # ALGO LOGIC: put action logic here
        if global_step < config.learning_starts * 9 / 10:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            torch_flat_obs = torch.tensor(flat_obs, dtype=torch.float, device=config.device)
            normalized_flat_obs = normalize_obs(torch_flat_obs, obs_rmss[0], obs_rmss[1])
            actions = policy(normalized_flat_obs)
            actions = actions.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        rewards = np.asarray(rewards, dtype=np.float32)
        # print(f"actions: {actions}")
        # print(f"rewards: {rewards}")

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            # log episodic returns
            for info in infos["final_info"]:
                r = float(info["episode"]["r"][0])
                max_ep_ret = max(max_ep_ret, r)
                avg_returns.append(r)

            avg_ret = torch.tensor(avg_returns).mean()
            std_ret = torch.tensor(avg_returns).std()
            log_dict = {"episodic_return": avg_ret, "episodic_return_std": std_ret}

            desc = f"global_step={global_step}, episodic_return={avg_ret: 4.2f} (max={max_ep_ret: 4.2f})"
            wandb.log(log_dict, step=global_step)

            # update channel rms normalization
            if global_step < config.learning_starts:
                stored_flat_obs = np.concatenate(stored_flat_obs, axis=0)
                update_channel_rmss(torch.tensor(stored_flat_obs), obs_rmss[0], obs_rmss[1])
                torch.save(
                    {"obs_rmss": obs_rmss}, os.path.join(config.checkpoint_dir, "obs_rmss.pth")
                )
                stored_flat_obs = []

            # get path gains
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
        # next_obs: Tuple(batched_real, batched_imag, batched_pos)
        real_next_obs = list(copy.deepcopy(next_obs))
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs = infos["final_observation"]
                # List[Tuple(Real, Imag, Pos), Tuple(Real, Imag, Pos)]
                # need to convert to Tuple(batched_real, batched_imag, batched_pos)
                real_next_obs = list(zip(*real_next_obs))
                real_next_obs = [np.stack(ob, axis=0) for ob in real_next_obs]
                break
        flat_real_next_obs = np.concatenate(
            [ob.reshape(ob.shape[0], -1) for ob in real_next_obs], axis=-1
        )

        if global_step == config.total_timesteps - 1:
            truncations = [True] * len(truncations)
        transition = TensorDict(
            observations=flat_obs,
            next_observations=flat_real_next_obs,
            actions=actions,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            path_gains=path_gains,
            next_path_gains=next_path_gains,
            batch_size=flat_obs.shape[0],
        )
        rb.extend(transition)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        flat_next_obs = np.concatenate([ob.reshape(ob.shape[0], -1) for ob in next_obs], axis=-1)
        flat_obs = flat_next_obs
        stored_flat_obs.append(flat_obs)

        # ALGO LOGIC: training.
        if global_step > config.learning_starts:
            log_infos = {}
            for j in range(config.n_updates):
                data = rb.sample()
                data = {
                    k: torch.as_tensor(v, device=config.device, dtype=torch.float)
                    for k, v in data.items()
                }
                data["observations"] = normalize_obs(data["observations"], obs_rmss[0], obs_rmss[1])
                data["next_observations"] = normalize_obs(
                    data["next_observations"], obs_rmss[0], obs_rmss[1]
                )
                data = TensorDict(data)

                log_infos.update(update_critics(data))
                # Update the learning rate
                q_scheduler.step()
                for param_group in q_optimizer.param_groups:
                    param_group["lr"] = q_scheduler.get_last_lr()[0]

                if j % config.policy_frequency == 1:  # TD 3 Delayed update support
                    for _ in range(config.policy_frequency):
                        # compensate for the delay by doing 'actor_update_interval' instead of 1
                        log_infos.update(update_actor(data))
                        alpha.copy_(log_alpha.detach().exp())
                        actor_scheduler.step()
                        for param_group in actor_optimizer.param_groups:
                            param_group["lr"] = actor_scheduler.get_last_lr()[0]

                # update the target networks
                if j % config.target_network_frequency == 1:
                    # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
                    qnet_target_params.lerp_(qnet_params.data, config.tau)

            if global_step > config.learning_starts + 5:
                with torch.no_grad():
                    q_lr = q_optimizer.param_groups[0]["lr"]
                    a_lr = actor_optimizer.param_groups[0]["lr"]
                    logs = {
                        "reward_mean": rewards.mean(),
                        "reward_std": rewards.std(),
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

            if global_step % config.save_interval == 0 or global_step == config.total_timesteps - 1:
                saved_dict = {
                    "actor": actor.state_dict(),
                    "qnet_params": qnet_params.state_dict(),
                    "qnet_target_params": qnet_target_params.state_dict(),
                    "log_alpha": log_alpha,
                    "obs_rmss": obs_rmss,
                }
                torch.save(
                    saved_dict,
                    os.path.join(config.checkpoint_dir, f"model_{global_step}.pth"),
                )
                torch.save(
                    saved_dict,
                    os.path.join(config.checkpoint_dir, f"model.pth"),
                )

            if (
                global_step % int(1.5 * config.save_interval) == 0
                or global_step == config.total_timesteps - 1
            ):
                # evaluate the model
                eval_episodic_rets = eval(config, eval_envs, obs_rmss, actor, is_plot=False)
                avg_ret = torch.tensor(eval_episodic_rets).mean()
                std_ret = torch.tensor(eval_episodic_rets).std()
                log_dict = {"eval/episodic_return": avg_ret, "eval/episodic_return_std": std_ret}
                wandb.log(log_dict, step=global_step)

    eval_envs.close()
    eval_envs.close_extras()


def eval(
    config: TrainConfig,
    envs: gym.vector.AsyncVectorEnv,
    obs_rmss: Tuple[running_mean.RunningMeanStd, running_mean.RunningMeanStd],
    actor: sac.Actor,
    is_plot: bool = True,
):

    # print(obs_rmss)
    mode = "default"
    policy = TensorDictModule(
        actor.get_action, in_keys=["observation"], out_keys=["action", "log_prob", "mean"]
    )
    policy = torch.compile(policy, mode=mode)
    policy = CudaGraphModule(policy)

    # action_low, action_high = envs.single_action_space.low, envs.single_action_space.high
    # action_low = torch.tensor(action_low, device=config.device, dtype=torch.float)
    # action_high = torch.tensor(action_high, device=config.device, dtype=torch.float)

    obs, _ = envs.reset(seed=config.seed)
    flat_obs = np.concatenate([ob.reshape(ob.shape[0], -1) for ob in obs], axis=-1)
    all_rewards = np.empty((config.eval_ep_len, envs.num_envs))
    all_path_gains = np.empty((config.eval_ep_len, envs.num_envs, 3))
    episodic_returns = np.zeros((envs.num_envs,))

    for global_step in range(config.eval_ep_len):
        # print(f"\nSTEP: {global_step}")
        torch_flat_obs = torch.tensor(flat_obs, dtype=torch.float, device=config.device)
        normalized_flat_obs = normalize_obs(torch_flat_obs, obs_rmss[0], obs_rmss[1])
        with torch.no_grad():
            # actions = actor(obs=normalized_flat_obs)
            _, _, actions = policy(observation=normalized_flat_obs)
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        rewards = np.asarray(rewards, dtype=np.float32)
        episodic_returns += rewards

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            # get path gains
            path_gains = [info["path_gain"] for info in infos["final_info"]]
            next_path_gains = [info["next_path_gain"] for info in infos["final_info"]]
        else:
            path_gains = infos["path_gain"]
            next_path_gains = infos["next_path_gain"]
        path_gains = np.stack(path_gains)
        next_path_gains = np.stack(next_path_gains)
        path_gains = torch.as_tensor(path_gains, dtype=torch.float)
        next_path_gains = torch.as_tensor(next_path_gains, dtype=torch.float)

        all_rewards[global_step, :] = rewards
        all_path_gains[global_step, ...] = path_gains

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        # next_obs: Tuple(batched_real, batched_imag, batched_pos)
        real_next_obs = list(copy.deepcopy(next_obs))
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs = infos["final_observation"]
                # List[Tuple(Real, Imag, Pos), Tuple(Real, Imag, Pos)]
                # need to convert to Tuple(batched_real, batched_imag, batched_pos)
                real_next_obs = list(zip(*real_next_obs))
                real_next_obs = [np.stack(ob, axis=0) for ob in real_next_obs]
                break
        flat_real_next_obs = np.concatenate(
            [ob.reshape(ob.shape[0], -1) for ob in real_next_obs], axis=-1
        )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        flat_next_obs = np.concatenate([ob.reshape(ob.shape[0], -1) for ob in next_obs], axis=-1)
        flat_obs = flat_next_obs

    if is_plot:
        record_path_gain_statistics(config, envs, all_rewards, all_path_gains)

    return episodic_returns


def record_path_gain_statistics(config, envs, all_rewards, all_path_gains):

    # plot path gains
    linear_path_gains = 10 ** (all_path_gains / 10)
    sum_path_gains = np.sum(linear_path_gains, axis=-1)
    db_sum_path_gains = 10 * np.log10(sum_path_gains)
    mean_path_gains = np.mean(db_sum_path_gains, axis=1)
    std_path_gains = np.std(db_sum_path_gains, axis=1)
    fig, ax = plt.subplots()
    ax.plot(mean_path_gains)
    ax.fill_between(
        range(config.eval_ep_len),
        mean_path_gains - std_path_gains,
        mean_path_gains + std_path_gains,
        alpha=0.2,
    )
    ax.set_xlabel("Steps")
    ax.set_ylabel("Path Gain")
    ax.set_title("Path Gain")
    ax.grid()
    plt.savefig(os.path.join(config.checkpoint_dir, "path_gain.png"))

    # Plot each of dm_sum_path_gains
    fig, ax = plt.subplots()
    for i in range(envs.num_envs):
        ax.plot(db_sum_path_gains[:, i])
    ax.set_xlabel("Steps")
    ax.set_ylabel("Path Gain")
    ax.set_title("Path Gain")
    labels = ["env" + str(i + config.eval_seed) for i in range(envs.num_envs)]
    ax.legend(labels)
    ax.grid()
    plt.savefig(os.path.join(config.checkpoint_dir, "all_path_gain.png"))

    # plot rewards
    mean_rewards = np.mean(all_rewards, axis=1)
    std_rewards = np.std(all_rewards, axis=1)
    fig, ax = plt.subplots()
    ax.plot(mean_rewards)
    ax.fill_between(
        range(config.eval_ep_len), mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2
    )
    ax.set_xlabel("Steps")
    ax.set_ylabel("Reward")
    ax.set_title("Rewards")
    ax.grid()
    plt.savefig(os.path.join(config.checkpoint_dir, "rewards.png"))

    # Save all_rewards and all_path_gains
    np.save(os.path.join(config.checkpoint_dir, "all_rewards.npy"), all_rewards)
    np.save(os.path.join(config.checkpoint_dir, "all_path_gains.npy"), all_path_gains)


if __name__ == "__main__":
    main()
