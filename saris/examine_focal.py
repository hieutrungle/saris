import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from typing import Optional, Sequence, Dict, Any, Tuple
import argparse
from saris.utils import pytorch_utils, utils
import numpy as np
import importlib.resources
import saris
from dataclasses import dataclass
import numpy as np
import tyro
import time
import copy
import tqdm
import tensorflow as tf
from saris.sigmap import signal_cmap
from sionna.channel import (
    cir_to_ofdm_channel,
    subcarrier_frequencies,
    time_lag_discrete_time_channel,
    cir_to_time_channel,
    time_to_ofdm_channel,
)
import glob
import gymnasium as gym
from saris.drl.envs import register_envs

register_envs()


@dataclass
class Args:
    """
    The arguments for the experiment.
    """

    # General arguments
    """the name of this experiment"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """seed of the experiment"""
    seed: int = 1
    """if toggled, cuda will be enabled by default"""
    cuda: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    track: bool = False
    """the wandb's project name"""
    wandb_project_name: str = "WirelessReflectorDRL"
    """the entity (team) of wandb's project"""
    wandb_entity: str = None
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    capture_video: bool = False
    """Log interval"""
    log_interval: int = 5
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
    env_id: str = "focal-v0"
    """Replay buffer capacity"""
    replay_buffer_capacity: int = 1000
    """the length of the episode"""
    ep_len: int = 100
    """Config file for the wireless simulation"""
    sionna_config_file: str = "sionna_config.yaml"
    """the number of parallel game environments"""
    num_envs: int = 6


def make_env(
    env_id: str,
    args: argparse.Namespace,
    idx: int,
    capture_video: Optional[bool] = None,
    run_name: Optional[str] = None,
):

    import tensorflow as tf

    def thunk():

        env = gym.make(
            env_id,
            idx=idx,
            sionna_config_file=args.sionna_config_file,
            log_string=args.log_string,
            seed=args.seed,
            # max_episode_steps=args.ep_len,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=args.ep_len)
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)

        return env

    return thunk


def get_log_string(args: argparse.Namespace):

    log_string = f"{args.exp_name}__{args.env_id}__seed{args.seed}"

    for replaced_str in [" ", "]", "}"]:
        log_string = log_string.replace(replaced_str, "")
    for replaced_str in ["[", ",", ".", "{"]:
        log_string = log_string.replace(replaced_str, "_")
    return log_string


def main():
    args = parse_agrs()
    sionna_config = utils.load_config(args.sionna_config_file)

    # set random seeds
    pytorch_utils.init_seed(args.seed)
    if args.verbose:
        utils.log_args(args)
        utils.log_config(sionna_config)

    # Env
    env = make_env(args.env_id, args, 0)()

    done = False
    env.unwrapped.eval()
    obs, _ = env.reset()
    step = 0
    while not done:
        step += 1
        action = env.action_space.sample()
        obs, reward, termination, truncation, info = env.step(action)
        print(f"Step {step}:")
        print(f"\tPath gain: {info['path_gain_dB']}")
        action = env.action_space.sample()
        done = termination or truncation

    print(f"Completed visualization for {args.env_id}!")


def parse_agrs():

    args = tyro.cli(Args)
    lib_dir = importlib.resources.files(saris)
    source_dir = os.path.dirname(lib_dir)
    args.source_dir = source_dir

    device = pytorch_utils.init_gpu()
    args.device = device

    args.log_string = get_log_string(args)

    return args


if __name__ == "__main__":
    main()
