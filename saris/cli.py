import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from typing import Optional
import argparse
from saris.utils import pytorch_utils, utils
import numpy as np
import gymnasium as gym
from saris.drl.envs import register_envs
from saris.drl.trainers import trainer_module
from saris.drl.agents import ppo_agent
import importlib.resources
import saris
import torch
import tyro
from dataclasses import dataclass

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
    save_interval: int = 2
    """Verbose level"""
    verbose: bool = False
    """Resume training from a checkpoint"""
    resume: bool = False

    # Environment specific arguments
    """Replay buffer capacity"""
    replay_buffer_capacity: int = 1000
    """the length of the episode"""
    ep_len: int = 100
    """Config file for the wireless simulation"""
    sionna_config_file: str = "sionna_config.yaml"

    # Algorithm specific arguments
    """the id of the environment"""
    env_id: str = "wireless-sigmap-v0"
    """total timesteps of the experiments"""
    total_timesteps: int = 32000
    """the learning rate of the optimizer"""
    learning_rate: float = 1e-4
    """the number of parallel game environments"""
    num_envs: int = 8
    """the number of steps to run in each environment per policy rollout"""
    num_steps: int = 16
    """the discount factor gamma"""
    gamma: float = 0.75
    """the lambda for the general advantage estimation"""
    gae_lambda: float = 0.9
    """the number of mini-batches"""
    num_minibatches: int = 2
    """the K epochs to update the policy"""
    update_epochs: int = 10
    """Toggles advantages normalization"""
    norm_adv: bool = True
    """the surrogate clipping coefficient"""
    clip_coef: float = 0.2
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    clip_vloss: bool = True
    """coefficient of the entropy"""
    ent_coef: float = 0.05
    """coefficient of the value function"""
    vf_coef: float = 0.5
    """the maximum norm for the gradient clipping"""
    max_grad_norm: float = 0.5
    """the target KL divergence threshold"""
    target_kl: float = None
    """the alpha parameter for RPO"""
    rpo_alpha: float = 0.5

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


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
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        # if capture_video:
        #     if run_name is None:
        #         run_name = f"{env_id}_seed_{seed}_idx_{idx}"
        #     else:
        #         if idx == 0:
        #             env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
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


def get_ppo_trainer_config(envs: gym.Env, args: argparse.Namespace):
    ob_space = envs.single_observation_space
    ac_space = envs.single_action_space
    print(f"Observation space: {ob_space}")
    print(f"Action space: {ac_space}")

    ob_shape = envs.single_observation_space.shape
    ac_shape = envs.single_action_space.shape
    print(f"Observation shape: {ob_shape}")
    print(f"Action shape: {ac_shape}")

    # Trainer
    trainer_config = {
        "observation_shape": ob_shape,
        "action_shape": ac_shape,
        "agent_class": ppo_agent.Agent,
        "agent_hparams": {
            "ob_shape": ob_shape,
            "ac_shape": ac_shape,
            "rpo_alpha": args.rpo_alpha,
        },
        "agent_optimizer_hparams": {
            "optimizer": "adamw",
            "lr": args.learning_rate,
        },
        "seed": args.seed,
        "logger_params": {
            "log_dir": os.path.join(args.source_dir, "local_assets", "logs"),
            "log_name": os.path.join("SARIS_PPO_" + args.log_string),
        },
        "args": args,
        "enable_progress_bar": True,
        "device": args.device,
        "train_dtype": torch.float16,
    }
    return trainer_config


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
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.env_id, args, i) for i in range(args.num_envs)]
    # )

    # Trainer
    trainer_config = get_ppo_trainer_config(envs, args)
    trainer = trainer_module.TrainerModule(**trainer_config)
    trainer.print_class_variables()

    if args.command == "train":
        trainer.train_agent(envs, args)
    elif args.command == "eval":
        trainer.eval_agent(envs, args)
    else:
        raise ValueError(f"Invalid command: {args.command}")

    envs.close()


def parse_agrs():

    args = tyro.cli(Args)
    lib_dir = importlib.resources.files(saris)
    source_dir = os.path.dirname(lib_dir)
    args.source_dir = source_dir

    device = pytorch_utils.init_gpu()
    args.device = device

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    args.log_string = get_log_string(args)

    return args


if __name__ == "__main__":
    main()
