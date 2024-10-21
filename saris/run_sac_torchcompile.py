import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import uuid
from dataclasses import dataclass
import importlib.resources
import saris
import subprocess
import pyrallis


@dataclass
class TrainConfig:

    # General arguments
    command: str = "train"  # the command to run
    load_model: str = "-1"  # Model load file name for resume training, "-1" doesn't load
    checkpoint_dir: str = "-1"  # the path to save the model
    replay_buffer_dir: str = "-1"  # the path to save the replay buffer
    load_replay_buffer: str = "-1"  # the path to load the replay buffer
    verbose: bool = False  # whether to log to console
    seed: int = 1  # seed of the experiment
    eval_seed: int = 100  # seed of the evaluation
    save_interval: int = 100  # the interval to save the model

    # Environment specific arguments
    env_id: str = "wireless-sigmap-v0"  # the environment id of the task
    sionna_config_file: str = "-1"  # Sionna config file
    num_envs: int = 8  # the number of parallel environments
    ep_len: int = 75  # the maximum length of an episode
    eval_ep_len: int = 75  # the maximum length of an episode

    # Algorithm specific arguments
    total_timesteps: int = 10_001  # total timesteps of the experiments
    n_updates: int = 20  # the number of updates per step
    buffer_size: int = int(80_000)  # the replay memory buffer size
    gamma: float = 0.85  # the discount factor gamma
    tau: float = 0.005  # target smoothing coefficient (default: 0.005)
    batch_size: int = 256  # the batch size of sample from the reply memory
    learning_starts: int = 701  # the timestep to start learning
    policy_lr: float = 3e-4  # the learning rate of the policy network optimizer
    q_lr: float = 1e-3  # the learning rate of the q network optimizer
    policy_frequency: int = 2  # the frequency of training policy (delayed)
    target_network_frequency: int = 2  # the frequency of updates for the target nerworks
    alpha: float = 0.2  # Entropy regularization coefficient
    action_scale: float = 9.0  # the scale of the action

    # Wandb logging
    project: str = "SARIS"  # wandb project name
    group: str = "SAC"  # wandb group name
    name: str = "Online-Learning"  # wandb run name

    def __post_init__(self):
        lib_dir = importlib.resources.files(saris)
        source_dir = os.path.dirname(lib_dir)
        self.source_dir = source_dir

        self.name = f"{self.group}__{self.name}__{self.env_id}__{str(uuid.uuid4())[:8]}"
        if self.checkpoint_dir == "-1":
            checkpoint_dir = os.path.join(self.source_dir, "local_assets", "logs")
            self.checkpoint_dir = os.path.join(checkpoint_dir, self.name)
        if self.replay_buffer_dir == "-1":
            replay_buffer_dir = os.path.join(self.source_dir, "local_assets", "replay_buffers")
            self.replay_buffer_dir = os.path.join(replay_buffer_dir, self.name)


@pyrallis.wrap()
def main(config: TrainConfig):

    base_cmd = ["python", "./saris/sac_torchcompile.py"]
    base_cmd += [
        # general arguments
        "--load_model",
        str(config.load_model),
        "--checkpoint_dir",
        str(config.checkpoint_dir),
        "--replay_buffer_dir",
        str(config.replay_buffer_dir),
        "--load_replay_buffer",
        str(config.load_replay_buffer),
        "--verbose",
        str(config.verbose),
        "--seed",
        str(config.seed),
        "--eval_seed",
        str(config.eval_seed),
        "--save_interval",
        str(config.save_interval),
        # environment specific arguments
        "--env_id",
        str(config.env_id),
        "--sionna_config_file",
        str(config.sionna_config_file),
        "--num_envs",
        str(config.num_envs),
        "--ep_len",
        str(config.ep_len),
        "--eval_ep_len",
        str(config.eval_ep_len),
        # algorithm specific arguments
        "--total_timesteps",
        str(config.total_timesteps),
        "--n_updates",
        str(config.n_updates),
        "--buffer_size",
        str(config.buffer_size),
        "--gamma",
        str(config.gamma),
        "--tau",
        str(config.tau),
        "--batch_size",
        str(config.batch_size),
        "--learning_starts",
        str(config.learning_starts),
        "--policy_lr",
        str(config.policy_lr),
        "--q_lr",
        str(config.q_lr),
        "--policy_frequency",
        str(config.policy_frequency),
        "--target_network_frequency",
        str(config.target_network_frequency),
        "--alpha",
        str(config.alpha),
        "--action_scale",
        str(config.action_scale),
        # wandb logging
        "--project",
        str(config.project),
        "--group",
        str(config.group),
        "--name",
        str(config.name),
    ]

    train_cmd = base_cmd + ["--command", "train"]
    subprocess.run(train_cmd, check=True)

    # train_cmd = base_cmd + ["--command", "eval"]
    # subprocess.run(train_cmd, check=True)


if __name__ == "__main__":
    main()
