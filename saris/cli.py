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
from saris.drl.networks import actor, critic
from saris.drl.trainers import sac_trainer
import importlib.resources
import saris


def make_env(
    env_id: str,
    sionna_config_file: str,
    drl_config_file: str,
    log_string: str,
    idx: int,
    seed: int,
    capture_video: Optional[bool] = None,
    run_name: Optional[str] = None,
):
    drl_config = utils.load_yaml_file(drl_config_file)

    def thunk():
        env = gym.make(
            env_id,
            sionna_config_file=sionna_config_file,
            log_string=log_string,
            seed=seed,
        )
        env = gym.wrappers.TimeLimit(env, max_episode_steps=drl_config["ep_len"])
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if run_name is None:
                run_name = f"{env_id}_seed_{seed}_idx_{idx}"
            else:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def get_log_string(drl_config: dict):
    # Log string
    # current_time = time.strftime("%Y%m%d-%H%M%S")
    log_string = "{}-{}-s{}-aclr{}-crlr{}-allr{}-b{}-d{}".format(
        drl_config["exp_name"],
        drl_config["env_id"],
        drl_config["hidden_sizes"],
        drl_config["actor_learning_rate"],
        drl_config["critic_learning_rate"],
        drl_config["alpha_learning_rate"],
        drl_config["batch_size"],
        drl_config["discount"],
    )
    log_string += f"-tem{drl_config['temperature']}"
    log_string += f"-stu{drl_config['tau']}"  # soft_target_update_rate
    for replaced_str in [" ", "]", "}"]:
        log_string = log_string.replace(replaced_str, "")
    for replaced_str in ["[", ",", ".", "{"]:
        log_string = log_string.replace(replaced_str, "_")
    # log_string = f"{log_string}_{current_time}"
    return log_string


def get_trainer_config(env: gym.Env, drl_config: dict, args: argparse.Namespace):
    ob_space = env.observation_space
    ac_space = env.action_space
    print(f"Observation space: {ob_space}")
    print(f"Action space: {ac_space}")
    num_obs = env.observation_space.shape[0]
    num_acts = env.action_space.shape[0]

    # Trainer
    trainer_config = {
        "observation_shape": ob_space.shape,
        "action_shape": ac_space.shape,
        "actor_class": actor.Actor,
        "actor_hparams": {
            "num_observations": num_obs,
            "num_actions": num_acts,
            "features": drl_config["hidden_sizes"],
            "activation": "tanh",
            "dtype": "bfloat16",
        },
        "critic_class": critic.Crtic,
        "critic_hparams": {
            "num_observations": num_obs,
            "num_actions": num_acts,
            "features": drl_config["hidden_sizes"],
            "activation": "relu",
            "dtype": "bfloat16",
        },
        "actor_optimizer_hparams": {
            "optimizer": "adamw",
            "lr": drl_config["actor_learning_rate"],
        },
        "critic_optimizer_hparams": {
            "optimizer": "adamw",
            "lr": drl_config["critic_learning_rate"],
        },
        "num_actor_samples": drl_config["num_actor_samples"],
        "num_critic_updates": drl_config["num_critic_updates"],
        "num_critics": drl_config["num_critics"],
        "discount": drl_config["discount"],
        "tau": drl_config["tau"],
        "grad_accum_steps": 1,
        "seed": args.seed,
        "logger_params": {
            "log_dir": os.path.join(args.source_dir, "local_assets", "logs"),
            "log_name": os.path.join("SARIS_SAC_" + drl_config["log_string"]),
        },
        "enable_progress_bar": True,
        "device": args.device,
        "debug": False,
    }
    return trainer_config


def main():
    args = parse_agrs()
    drl_config = utils.load_config(args.drl_config_file)
    drl_config["log_string"] = get_log_string(drl_config)
    sionna_config = utils.load_config(args.sionna_config_file)

    # set random seeds
    pytorch_utils.init_seed(args.seed)
    if args.verbose:
        utils.log_args(args)
        utils.log_config(drl_config)
        utils.log_config(sionna_config)

    # Env
    register_envs()
    env = make_env(
        env_id=drl_config["env_id"],
        sionna_config_file=args.sionna_config_file,
        drl_config_file=args.drl_config_file,
        log_string=drl_config["log_string"],
        idx=0,
        seed=args.seed,
    )()
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert (
        not discrete
    ), "Our wireless DRL implementation only supports continuous action spaces."

    # Trainer
    trainer_config = get_trainer_config(env, drl_config, args)
    trainer = sac_trainer.SoftActorCriticTrainer(**trainer_config)
    trainer.print_class_variables()

    if args.command == "train":
        trainer.train_agent(env, drl_config, args)
    # elif args.command == "eval":
    #     trainer.eval_agent(env, drl_config, args)
    else:
        raise ValueError(f"Invalid command: {args.command}")


def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drl_config_file", "-dcfg", type=str, required=True)
    parser.add_argument("--sionna_config_file", "-scfg", type=str, required=True)
    parser.add_argument("--command", "-cmd", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--resume", "-r", action="store_true")

    args = parser.parse_args()
    lib_dir = importlib.resources.files(saris)
    source_dir = os.path.dirname(lib_dir)
    args.source_dir = source_dir

    device = pytorch_utils.init_gpu()
    args.device = device
    return args


if __name__ == "__main__":
    main()
