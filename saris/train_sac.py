import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from typing import Optional, Sequence, Dict, Any, Tuple
import argparse
from saris.utils import pytorch_utils, utils
import numpy as np
import gymnasium as gym
from saris.drl.envs import register_envs
from saris.drl.agents import sac
import importlib.resources
import saris
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common import buffers
from torch.utils.tensorboard import SummaryWriter
import time
import copy
import tqdm

register_envs()


@dataclass
class Args:
    """
    The arguments for the experiment.
    """

    # General arguments
    """the command to run the experiment"""
    command: str = "train"
    """the name of this experiment"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """seed of the experiment"""
    seed: int = 1
    """if toggled, cuda will be enabled by default"""
    cuda: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    track: bool = False
    """the wandb's project name"""
    wandb_project_name: str = "WirelessReflectiveDRL"
    """the entity (team) of wandb's project"""
    wandb_entity: str = None
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    capture_video: bool = False
    """Log interval"""
    log_interval: int = 2
    """Save interval"""
    save_interval: int = 10
    """Verbose level"""
    verbose: bool = False

    # Resume training
    """Resume training from a checkpoint"""
    resume: bool = False
    """Load step"""
    load_step: int = 0
    """Retrain the model"""
    retrain: bool = False

    # Environment specific arguments
    """the id of the environment"""
    env_id: str = "wireless-sigmap-v0"
    """Replay buffer capacity"""
    replay_buffer_capacity: int = 1000
    """the length of the episode"""
    ep_len: int = 100
    """Config file for the wireless simulation"""
    sionna_config_file: str = "sionna_config.yaml"
    """the number of parallel game environments"""
    num_envs: int = 6

    # Algorithm specific arguments
    # TODO: decrease gamma to 0.9 since the episode length is 100, which is short
    # TODO: decrease buffer size to 1000 since old data are not useful
    """total timesteps of the experiments, this will be divided by the number of parallel environments"""
    total_timesteps: int = 32009
    """the replay memory buffer size"""
    buffer_size: int = int(4000)
    """the discount factor gamma"""
    gamma: float = 0.99
    """target smoothing coefficient (default: 0.005)"""
    tau: float = 0.005
    """the batch size of sample from the reply memory"""
    batch_size: int = 256
    """timestep to start learning"""
    learning_starts: int = 1000
    """the learning rate of the policy network optimizer"""
    policy_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    q_lr: float = 1e-3
    """The number of agent updates per environment step"""
    num_updates_per_step: int = 6
    """the frequency of training policy (delayed)"""
    policy_frequency: int = 2
    """the frequency of updates for the target nerworks"""
    target_network_frequency: int = 2  # Denis Yarats' implementation delays this by 2.
    """Entropy regularization coefficient."""
    alpha: float = 0.2
    """automatic tuning of the entropy coefficient"""
    autotune: bool = True


def make_env(
    env_id: str,
    args: argparse.Namespace,
    idx: int,
    capture_video: Optional[bool] = None,
    run_name: Optional[str] = None,
):

    gamma = args.gamma
    import tensorflow as tf

    def thunk():

        env = gym.make(
            env_id,
            idx=idx,
            sionna_config_file=args.sionna_config_file,
            log_string=args.log_string,
            seed=args.seed,
            max_episode_steps=args.ep_len,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=args.ep_len)
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)

        return env

    return thunk


def get_log_string(args: argparse.Namespace):

    log_string = f"{args.exp_name}__{args.env_id}__seed{args.seed}"
    log_string += f"__discount{args.gamma}"

    for replaced_str in [" ", "]", "}"]:
        log_string = log_string.replace(replaced_str, "")
    for replaced_str in ["[", ",", ".", "{"]:
        log_string = log_string.replace(replaced_str, "_")
    return log_string


def init_optimizer(
    params: Sequence[nn.Parameter], lr: float, total_steps: int
) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """
    Initializes the optimizer and learning rate scheduler.

    Args:
        params: The parameters of the model to optimize.
        optimizer_hparams: A dictionary of all hyperparameters of the optimizer.
        num_epochs: Number of epochs the model will be trained for.
        num_steps_per_epoch: Number of training steps per epoch.
    """
    optimizer = optim.AdamW(params, lr=lr, eps=1e-6)
    warmup_steps = total_steps // 5
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1 / 20, total_iters=warmup_steps
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, total_steps - warmup_steps, eta_min=lr / 10
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_scheduler, cosine_scheduler], [warmup_steps]
    )

    return optimizer, scheduler


def train(args: argparse.Namespace, envs: gym.vector.VectorEnv):

    log_dir = os.path.join(args.source_dir, "local_assets", "logs")
    log_name = os.path.join("SARIS_SAC_" + args.log_string)
    log_path = os.path.join(log_dir, log_name)
    writer = SummaryWriter(log_path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    agent = sac.Agent(envs.single_observation_space, envs.single_action_space)
    agent = agent.to(args.device)

    num_agent_update = (args.total_timesteps - args.learning_starts) * args.num_updates_per_step
    q_optimizer, q_scheduler = init_optimizer(
        list(agent.qf1.parameters()) + list(agent.qf2.parameters()), args.q_lr, num_agent_update
    )
    q_scaler = torch.cuda.amp.GradScaler()

    actor_optimizer, actor_scheduler = init_optimizer(
        agent.actor.parameters(), args.policy_lr, num_agent_update
    )
    actor_scaler = torch.cuda.amp.GradScaler()

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(envs.single_action_space.shape).to(args.device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.AdamW([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # Load checkpoint if needed
    if args.resume and args.load_step > 0:
        checkpoint_path = os.path.join(log_path, f"checkpoint_{args.load_step}.pt")
        optimizer_dict = {
            "q_optimizer": q_optimizer,
            "actor_optimizer": actor_optimizer,
        }
        if args.autotune:
            optimizer_dict["a_optimizer"] = a_optimizer

        scheduler_dict = {
            "q_scheduler": q_scheduler,
            "actor_scheduler": actor_scheduler,
        }

        agent, optimizer_dict, scheduler_dict, global_step = load_checkpoint(
            agent, optimizer_dict, scheduler_dict, checkpoint_path
        )
        q_optimizer = optimizer_dict["q_optimizer"]
        actor_optimizer = optimizer_dict["actor_optimizer"]
        if args.autotune:
            a_optimizer = optimizer_dict["a_optimizer"]
        if not args.retrain:
            print(
                f"Loaded checkpoint from {checkpoint_path} at step {global_step} and continue training"
            )
            q_scheduler = scheduler_dict["q_scheduler"]
            actor_scheduler = scheduler_dict["actor_scheduler"]
            global_step += 1
        else:
            print(f"Loaded checkpoint from {checkpoint_path} at step {global_step} and retrain")
    else:
        global_step = 0

    envs.single_observation_space.dtype = np.float32
    rb = buffers.DictReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        args.device,
        n_envs=envs.num_envs,
        handle_timeout_termination=False,
    )

    # Create save directory for buffer
    local_assets_dir = utils.get_dir(args.source_dir, "local_assets")
    buffer_saved_name = os.path.join("replay_buffer", args.log_string)
    buffer_saved_dir = utils.get_dir(local_assets_dir, buffer_saved_name)

    start_time = time.time()

    # Create running meanstd for normalization
    obs_rms = create_running_meanstd_for_dict(envs.single_observation_space)
    rewards_rms = RunningMeanStd(shape=(1,))

    obs, _ = envs.reset(seed=args.seed)
    # TODO: use some new data + old data from the replay buffer. Mix both offline and online data.
    t = tqdm.tqdm(
        range(global_step, args.total_timesteps),
        total=args.total_timesteps,
        initial=global_step,
        dynamic_ncols=True,
    )
    for global_step in t:
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            torch_obs = pytorch_utils.from_numpy(obs, args.device)
            actions, _, _ = agent.actor.get_actions(torch_obs)
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            returns = [info["episode"]["r"] for info in infos["final_info"]]
            return_mean = np.mean([info["episode"]["r"] for info in infos["final_info"]])
            return_std = np.std([info["episode"]["r"] for info in infos["final_info"]])
            writer.add_scalar("charts/return_max", np.max(returns), global_step)
            writer.add_scalar("charts/return_min", np.min(returns), global_step)
            writer.add_scalar("charts/return_mean", return_mean, global_step)
            writer.add_scalar("charts/return_std", return_std, global_step)
            print(f"global_step={global_step}, return_mean={return_mean}, return_std={return_std}")

            path_gains = [info["path_gain_dB"] for info in infos["final_info"]]
            next_path_gains = [info["next_path_gain_dB"] for info in infos["final_info"]]
        else:
            path_gains = infos["path_gain_dB"]
            next_path_gains = infos["next_path_gain_dB"]
        # save data to file
        utils.save_data(
            {f"{global_step}": path_gains},
            os.path.join(buffer_saved_dir, "path_gains.txt"),
        )
        utils.save_data(
            {f"{global_step}": next_path_gains},
            os.path.join(buffer_saved_dir, "next_path_gains.txt"),
        )

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = copy.deepcopy(next_obs)
        for idx, trunc in enumerate(truncations):
            if trunc:
                # Dict Obs
                for key in real_next_obs.keys():
                    real_next_obs[key][idx] = infos["final_observation"][idx][key]
                # real_next_obs[idx] = infos["final_observation"][idx]

        dones = np.zeros_like(terminations)
        rb.add(obs, real_next_obs, actions, rewards, dones, [])
        utils.save_data(
            {f"{global_step}": obs},
            os.path.join(buffer_saved_dir, "obs.txt"),
        )
        utils.save_data(
            {f"{global_step}": actions},
            os.path.join(buffer_saved_dir, "actions.txt"),
        )
        utils.save_data(
            {f"{global_step}": real_next_obs},
            os.path.join(buffer_saved_dir, "next_obs.txt"),
        )
        utils.save_data(
            {f"{global_step}": dones},
            os.path.join(buffer_saved_dir, "dones.txt"),
        )
        utils.save_data(
            {f"{global_step}": rewards},
            os.path.join(buffer_saved_dir, "rewards.txt"),
        )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs_rms = update_rms_dict(obs_rms, obs)
        rewards_rms.update(torch.tensor(rewards, dtype=torch.float))
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            for _ in range(args.num_updates_per_step):
                data = rb.sample(args.batch_size)
                normalized_obs = normalize_obs(data.observations, obs_rms)
                normalized_next_obs = normalize_obs(data.next_observations, obs_rms)
                normalized_rews = data.rewards / ((rewards_rms.var).to(args.device) + 1e-8)
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = agent.actor.get_actions(
                        normalized_next_obs
                    )
                    next_q1s = agent.target_qf1(normalized_next_obs, next_state_actions)
                    next_q2s = agent.target_qf2(normalized_next_obs, next_state_actions)
                    min_next_qs = torch.min(next_q1s, next_q2s)
                    next_qs = min_next_qs - alpha * next_state_log_pi
                    next_qs = next_qs.view(-1)
                    flat_rews = normalized_rews.flatten()
                    flat_dones = data.dones.flatten()
                    target_q_values = flat_rews + (1 - flat_dones) * args.gamma * next_qs

                qf1_values = agent.qf1(normalized_obs, data.actions).view(-1)
                qf2_values = agent.qf2(normalized_obs, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_values, target_q_values)
                qf2_loss = F.mse_loss(qf2_values, target_q_values)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                q_optimizer.zero_grad(set_to_none=True)
                q_scaler.scale(qf_loss).backward()
                q_scaler.unscale_(q_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(agent.qf1.parameters()) + list(agent.qf2.parameters()), max_norm=0.75
                )
                q_scaler.step(q_optimizer)
                q_scaler.update()
                q_scheduler.step()

                # TD3 delayed policy updates
                if global_step % args.policy_frequency == 0:
                    # compensate for the delay by doing 'actor_update_interval' instead of 1
                    for _ in range(args.policy_frequency):
                        pi, log_pi, _ = agent.actor.get_actions(normalized_obs)
                        qf1_pi = agent.qf1(normalized_obs, pi)
                        qf2_pi = agent.qf2(normalized_obs, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                        actor_optimizer.zero_grad(set_to_none=True)
                        actor_scaler.scale(actor_loss).backward()
                        actor_scaler.unscale_(actor_optimizer)
                        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=0.75)
                        actor_scaler.step(actor_optimizer)
                        actor_scaler.update()
                        actor_scheduler.step()

                        if args.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = agent.actor.get_actions(normalized_obs)
                            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                            a_optimizer.zero_grad()
                            alpha_loss.backward()
                            a_optimizer.step()
                            alpha = log_alpha.exp().item()

                # update the target networks
                if global_step % args.target_network_frequency == 0:
                    agent.update_target(args.tau)

            if global_step % args.log_interval == 0:
                writer.add_scalar(
                    "charts/q_learning_rate", q_optimizer.param_groups[0]["lr"], global_step
                )
                writer.add_scalar(
                    "charts/actor_learning_rate", actor_optimizer.param_groups[0]["lr"], global_step
                )
                writer.add_scalar("losses/qf1_values", qf1_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar(
                    "charts/SPS", int(global_step / (time.time() - start_time)), global_step
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

            if global_step % args.save_interval == 0:
                optimizer_dict = {
                    "q_optimizer": q_optimizer,
                    "actor_optimizer": actor_optimizer,
                }
                if args.autotune:
                    optimizer_dict["a_optimizer"] = a_optimizer

                scheduler_dict = {
                    "q_scheduler": q_scheduler,
                    "actor_scheduler": actor_scheduler,
                }

                checkpoint_path = os.path.join(log_path, f"checkpoint_{global_step}.pt")
                save_checkpoint(agent, optimizer_dict, scheduler_dict, checkpoint_path, global_step)


def eval(args: argparse.Namespace, envs: gym.vector.VectorEnv):
    import matplotlib.pyplot as plt

    log_dir = os.path.join(args.source_dir, "local_assets", "logs")
    log_name = os.path.join("SARIS_SAC_" + args.log_string)
    log_path = os.path.join(log_dir, log_name)
    writer = SummaryWriter(log_path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    agent = sac.Agent(envs.single_observation_space, envs.single_action_space)
    agent = agent.to(args.device)

    # Load checkpoint if needed
    if args.resume and args.load_step > 0:
        checkpoint_path = os.path.join(log_path, f"checkpoint_{args.load_step}.pt")
        optimizer_dict = {}
        scheduler_dict = {}

        agent, optimizer_dict, scheduler_dict, global_step = load_checkpoint(
            agent, optimizer_dict, scheduler_dict, checkpoint_path
        )
        print(f"Loaded checkpoint from {checkpoint_path} at step {global_step} and Evaluate")
    else:
        raise ValueError("Evaluation requires a checkpoint to load")

    trajs = eval_trajectories(agent, envs, args)

    returns = []
    for rews, done in zip(trajs["rews"], trajs["dones"]):
        # calculate the return for each index
        return_ = 0
        traj_return = np.zeros_like(rews)
        for i in range(len(rews)):
            return_ = rews[i] + return_ * (1 - done[i])
            traj_return[i] = return_
        returns.append(traj_return)
    returns = np.array(returns)
    print(f"return: {returns}")
    mean_return = np.mean(returns, axis=0)
    std_return = np.std(returns, axis=0)
    for i, (mean_return_, std_return_) in enumerate(zip(mean_return, std_return)):
        writer.add_scalar(f"eval/mean_return", mean_return_, global_step)
        writer.add_scalar(f"eval/std_return", std_return_, global_step)

    # Plot the returns
    plt.figure()
    plt.plot(mean_return)
    plt.fill_between(
        range(len(mean_return)),
        mean_return - std_return,
        mean_return + std_return,
        alpha=0.3,
    )
    plt.xlabel("Step")
    plt.ylabel("Return")
    plt.title("Return")
    plt.savefig(os.path.join(log_path, "return.png"))
    plt.close()

    mean_path_gains = np.mean(trajs["path_gains"], axis=0)
    std_path_gains = np.std(trajs["path_gains"], axis=0)
    for i, (mean_path_gain, std_path_gain) in enumerate(zip(mean_path_gains, std_path_gains)):
        writer.add_scalar(f"eval/mean_path_gain", mean_path_gain, global_step)
        writer.add_scalar(f"eval/std_path_gain", std_path_gain, global_step)
    print(f"path_gain: {trajs['path_gains']}")

    mean_rewards = np.mean(trajs["rews"], axis=0)
    std_rewards = np.std(trajs["rews"], axis=0)
    for i, (mean_reward, std_reward) in enumerate(zip(mean_rewards, std_rewards)):
        writer.add_scalar(f"eval/mean_reward", mean_reward, global_step)
        writer.add_scalar(f"eval/std_reward", std_reward, global_step)
    print(f"reward: {trajs['rews']}")

    # Plot the path gains
    plt.figure()
    plt.plot(mean_path_gains)
    plt.fill_between(
        range(len(mean_path_gains)),
        mean_path_gains - std_path_gains,
        mean_path_gains + std_path_gains,
        alpha=0.3,
    )
    plt.xlabel("Step")
    plt.ylabel("Path Gain (dB)")
    plt.title("Path Gain")
    plt.savefig(os.path.join(log_path, "path_gain.png"))
    plt.close()


def eval_trajectories(agent: sac.Agent, envs: gym.vector.VectorEnv, args: argparse.Namespace):
    num_envs = envs.num_envs
    ac_shape = envs.single_action_space.shape

    obs = []
    acts = np.full((num_envs, args.ep_len, *ac_shape), np.nan)
    rews = np.full((num_envs, args.ep_len), np.nan)
    next_obs = []
    dones = np.full((num_envs, args.ep_len), np.nan)
    path_gains = np.full((num_envs, args.ep_len), np.nan)

    (observations, infos) = envs.reset()

    for step in tqdm.tqdm(range(args.ep_len)):
        observations = pytorch_utils.from_numpy(observations, args.device)
        _, _, actions = agent.actor.get_actions(observations)
        actions = pytorch_utils.to_numpy(actions)
        next_observations, rewards, terminations, truncations, infos = envs.step(actions)

        rewards = np.asarray(rewards, dtype=np.float32)
        obs.append(observations)
        acts[:, step] = actions
        rews[:, step] = rewards
        next_obs.append(next_observations)
        dones[:, step] = np.logical_or(terminations, truncations)

        if "final_info" in infos:
            path_gain = [info["path_gain_dB"] for info in infos["final_info"]]
            next_path_gain = [info["next_path_gain_dB"] for info in infos["final_info"]]
        else:
            path_gain = infos["path_gain_dB"]
            next_path_gain = infos["next_path_gain_dB"]
        path_gains[:, step] = path_gain

        observations = next_observations

    return {
        "obs": obs,
        "acts": acts,
        "rews": rews,
        "next_obs": next_obs,
        "dones": dones,
        "path_gains": path_gains,
    }


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


def save_checkpoint(
    agent: sac.Agent,
    optimizer_dict: Dict[str, optim.Optimizer],
    scheduler_dict: Dict[str, optim.lr_scheduler._LRScheduler],
    checkpoint_path: str,
    global_step: int,
):
    optimizer_state_dicts = {name: opt.state_dict() for name, opt in optimizer_dict.items()}
    checkpoint = {
        "agent_state_dict": agent.state_dict(),
        "global_step": global_step,
    }
    checkpoint.update(optimizer_state_dicts)
    checkpoint.update(scheduler_dict)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    agent: sac.Agent,
    optimizer_dict: Dict[str, optim.Optimizer],
    scheduler_dict: Dict[str, optim.lr_scheduler._LRScheduler],
    checkpoint_path: str,
) -> Tuple[sac.Agent, Dict[str, optim.Optimizer], int]:
    checkpoint = torch.load(checkpoint_path)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    for name, opt in optimizer_dict.items():
        opt.load_state_dict(checkpoint[name])
    for name, scheduler in scheduler_dict.items():
        scheduler_dict[name] = checkpoint[name]
    return agent, optimizer_dict, scheduler_dict, checkpoint["global_step"]


def main():
    args = parse_agrs()
    sionna_config = utils.load_config(args.sionna_config_file)

    # set random seeds
    pytorch_utils.init_seed(args.seed)
    if args.verbose:
        utils.log_args(args)
        utils.log_config(sionna_config)

    # Env
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id, args, i) for i in range(args.num_envs)],
        context="spawn",
    )
    ob_space = envs.single_observation_space
    ac_space = envs.single_action_space
    print(f"Observation space: {ob_space}")
    print(f"Action space: {ac_space}")

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=args.log_string,
            monitor_gym=True,
            save_code=True,
        )

    if args.command == "train":
        train(args, envs)
    elif args.command == "eval":
        eval(args, envs)

    envs.close()


def parse_agrs():

    args = tyro.cli(Args)
    lib_dir = importlib.resources.files(saris)
    source_dir = os.path.dirname(lib_dir)
    args.source_dir = source_dir

    args.total_timesteps = args.total_timesteps // args.num_envs

    device = pytorch_utils.init_gpu()
    args.device = device

    args.log_string = get_log_string(args)

    return args


if __name__ == "__main__":
    main()
