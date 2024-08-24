import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".30"

# Jax acceleration flags
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true "
    "--xla_gpu_triton_gemm_any=True "
    # "--xla_gpu_enable_async_collectives=true "
    # "--xla_gpu_enable_latency_hiding_scheduler=true "
    # "--xla_gpu_enable_highest_priority_async_stream=true "
)

import argparse
from saris.utils import utils
import numpy as np
import gymnasium as gym
from saris.drl.envs import register_envs
from saris.drl.networks import actor, critic
from saris.drl.trainers import sac_trainer
from saris.drl.infrastructure.replay_buffer import ReplayBuffer
import jax
import importlib.resources
import saris
import time
import tqdm


def make_env(
    env_id: str,
    sionna_config_file: str,
    drl_config_file: str,
    log_string: str,
    idx: int,
    seed: int,
    capture_video: bool = None,
    run_name: str = None,
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
    log_string += f"-stu{drl_config['ema_decay']}"  # soft_target_update_rate
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

    # Trainer
    trainer_config = {
        "observation_shape": ob_space.shape,
        "action_shape": ac_space.shape,
        "actor_class": actor.Actor,
        "actor_hparams": {
            "num_actions": ac_space.shape[0],
            "features": drl_config["hidden_sizes"],
            "activation": "tanh",
            "dtype": "bfloat16",
        },
        "critic_class": critic.Crtic,
        "critic_hparams": {
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
        "num_critics": drl_config["num_critics"],
        "discount": drl_config["discount"],
        "ema_decay": drl_config["ema_decay"],
        "grad_accum_steps": 1,
        "seed": args.seed,
        "logger_params": {
            "log_dir": os.path.join(args.source_dir, "local_assets", "logs"),
            "log_name": os.path.join("SARIS_SAC_" + drl_config["log_string"]),
        },
        "enable_progress_bar": True,
        "debug": False,
    }
    return trainer_config


def train_model(
    trainer: sac_trainer.SoftActorCriticTrainer,
    env: gym.Env,
    drl_config: dict,
    args: argparse.Namespace,
):
    print(f"*" * 80)
    print(
        f"Training {trainer.actor_class.__name__} and {trainer.critic_class.__name__}"
    )

    local_assets_dir = utils.get_dir(args.source_dir, "local_assets")
    buffer_saved_name = os.path.join("replay_buffer", drl_config["log_string"])
    buffer_saved_dir = utils.get_dir(local_assets_dir, buffer_saved_name)
    replay_buffer = ReplayBuffer(
        drl_config["replay_buffer_capacity"], buffer_saved_dir, seed=args.seed
    )

    ep_len = env.spec.max_episode_steps
    best_episodic_return = -np.inf
    best_episodic_return_std = np.inf
    best_episodic_return_epoch = 0
    best_episodic_return_std_epoch = 0

    # Load model if exists
    if args.resume and trainer.logger.log_dir is not None:
        print(f"Loading model from {trainer.logger.log_dir}")
        trainer.load_models()
        start_step = trainer.checkpoint_manager.latest_step()
    else:
        print(f"Training from scratch")
        start_step = 0
    start_step = int(start_step)

    # Training loop
    (observation, info) = env.reset()

    for step in tqdm.tqdm(
        range(start_step, drl_config["total_steps"]),
        total=drl_config["total_steps"],
        dynamic_ncols=True,
        initial=start_step,
    ):
        # accumulate data in replay buffer
        if step < drl_config["random_steps"] * 1 / 2:
            observation, info = env.reset()
            action = env.action_space.sample()
        elif step < drl_config["random_steps"]:
            action = env.action_space.sample()
        else:
            action = trainer.get_action(observation)

        try:
            next_observation, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            print(f"Error in step {step}: {e}")
            time.sleep(2)
            continue

        done = terminated or truncated
        reward = np.asarray(reward, dtype=np.float32)
        done = np.asarray(done, dtype=np.float16)

        print(f"step={step}, info={info}")
        print(f"observation={observation}")
        print(f"action={action}")
        print(f"next_observation={next_observation}")
        print(f"reward={reward}")
        print(f"done={done}")
        exit()

        replay_buffer.insert(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done,
        )

        if done:
            trainer.logger.log_metrics({"train_return": info["episode"]["r"]}, step)
            trainer.logger.log_metrics({"train_ep_len": info["episode"]["l"]}, step)
            observation, info = env.reset()
        else:
            observation = next_observation

        # train agent
        if step > drl_config["training_starts"]:
            for _ in range(drl_config["num_train_steps_per_env_step"]):
                batch = replay_buffer.sample(drl_config["batch_size"])
                obs, actions, rewards, next_obs, dones = (
                    batch["observations"],
                    batch["actions"],
                    batch["rewards"],
                    batch["next_observations"],
                    batch["dones"],
                )
                update_info = trainer.update(
                    obs, actions, rewards, next_obs, dones, step
                )

            # logging
            if step % args.log_interval == 0:
                trainer.logger.log_metrics(update_info, step)
                trainer.logger.flush()

            if reward > best_return:
                best_return = reward
                trainer.save_models(step)
    trainer.wait_for_checkpoint()
    return

    # for i in range(1):
    #     ob, info = env.reset()
    #     done = False
    #     j = 0
    #     while not done:
    #         ac = ac_space.sample()
    #         ob, rew, terminated, truncated, info = env.step(ac)
    #         print(
    #             f"step={j}, reward={rew}, truncated={truncated}, terminated={terminated}, info={info}"
    #         )
    #         if "episode" in info.keys():
    #             print(f"global_step={j}, episodic_return={info['episode']['r']}")
    #         done = terminated or truncated
    #         if done or j >= 300:
    #             break
    pass


def eval_model(trainer, env, drl_config, args):
    print(f"*" * 80)
    print(
        f"Evaluating {trainer.actor_class.__name__} and {trainer.critic_class.__name__}"
    )
    pass


def main():
    args = parse_agrs()
    drl_config = utils.load_yaml_file(args.drl_config_file)
    drl_config["log_string"] = get_log_string(drl_config)
    sionna_config = utils.load_yaml_file(args.sionna_config_file)

    # set random seeds
    np.random.seed(args.seed)
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
        train_model(trainer, env, drl_config, args)
    elif args.command == "eval":
        eval_model(trainer, env, drl_config, args)
    else:
        raise ValueError(f"Invalid command: {args.command}")


def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drl_config_file", "-dcfg", type=str, required=True)
    parser.add_argument("--sionna_config_file", "-scfg", type=str, required=True)
    parser.add_argument("--command", "-cmd", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--resume", "-r", action="store_true")

    args = parser.parse_args()
    lib_dir = importlib.resources.files(saris)
    source_dir = os.path.dirname(lib_dir)
    args.source_dir = source_dir
    return args


if __name__ == "__main__":
    main()
