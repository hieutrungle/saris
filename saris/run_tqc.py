import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import uuid
from dataclasses import dataclass
import importlib.resources
import saris
import subprocess
import pyrallis
import copy
import signal


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
    seed: int = 54  # seed of the experiment
    eval_seed: int = 7  # seed of the evaluation
    save_interval: int = 150  # the interval to save the model

    # Environment specific arguments
    env_id: str = "wireless-sigmap-v0"  # the environment id of the task
    sionna_config_file: str = "-1"  # Sionna config file
    num_envs: int = 3  # the number of parallel environments
    ep_len: int = 100  # the maximum length of an episode
    eval_ep_len: int = 100  # the maximum length of an episode

    # Algorithm specific arguments
    total_timesteps: int = 2_001  # total timesteps of the experiments
    n_updates: int = 5  # the number of updates per step
    buffer_size: int = int(40_000)  # the replay memory buffer size
    gamma: float = 0.985  # the discount factor gamma
    tau: float = 0.015  # target smoothing coefficient (default: 0.005)
    batch_size: int = 256  # the batch size of sample from the reply memory
    learning_starts: int = 201  # the timestep to start learning
    policy_lr: float = 3e-4  # the learning rate of the policy network optimizer
    q_lr: float = 1e-3  # the learning rate of the q network optimizer
    warmup_steps: int = 300  # the number of warmup steps
    policy_frequency: int = 2  # the frequency of training policy (delayed)
    target_network_frequency: int = 2  # the frequency of updates for the target nerworks
    alpha: float = 0.2  # Entropy regularization coefficient
    n_quantiles: int = 25  # the number of quantiles
    top_quantiles_to_drop_per_net: int = 2  # the number of top quantiles to drop per network

    # Wandb logging
    wandb_mode: str = "online"  # wandb mode
    project: str = "SARIS"  # wandb project name
    group: str = "TQC_Shaped_Reward"  # wandb group name
    name: str = "Unchanged_NN"  # wandb run name

    def __post_init__(self):
        lib_dir = importlib.resources.files(saris)
        source_dir = os.path.dirname(lib_dir)
        self.source_dir = source_dir

        self.name = f"{self.name}__{self.env_id}__{str(uuid.uuid4())[:8]}"
        if self.checkpoint_dir == "-1":
            checkpoint_dir = os.path.join(self.source_dir, "local_assets", "logs")
            self.checkpoint_dir = os.path.join(checkpoint_dir, self.name)
        if self.replay_buffer_dir == "-1":
            replay_buffer_dir = os.path.join(self.source_dir, "local_assets", "replay_buffers")
            self.replay_buffer_dir = os.path.join(replay_buffer_dir, self.name)


@pyrallis.wrap()
def main(config: TrainConfig):

    base_cmd = get_base_cmd(config)

    if config.load_eval_model == "-1":

        def handle_interrupt(signum, frame):
            print("Gracefully exiting subprocess...")
            process.send_signal(signal.SIGINT)  # Send SIGINT to the subprocess
            process.wait(timeout=10.0)  # Wait for the subprocess to finish
            exit(0)

        signal.signal(signal.SIGINT, handle_interrupt)

        try:
            print()
            print("*" * 50)
            print(f"TRAINING TQC: Training the Agent on {config.env_id}")
            print("*" * 50)
            print()
            train_config = copy.deepcopy(config)
            replay_buffer_dir = train_config.replay_buffer_dir
            name = train_config.name
            checkpoint_dir = train_config.checkpoint_dir
            if train_config.learning_starts < 401:
                train_config.learning_starts = 401
            train_cmd = get_base_cmd(train_config) + ["--command", "train"]
            process = subprocess.Popen(train_cmd)
            process.wait()  # Wait for the subprocess to finish

            train_config.learning_starts = config.learning_starts
            for i in range(1, 14):
                train_config.name = name + f"_{i}"
                train_config.seed += 10
                train_config.load_model = os.path.join(train_config.checkpoint_dir, "model.pth")
                train_config.load_replay_buffer = train_config.replay_buffer_dir
                train_config.checkpoint_dir = checkpoint_dir + f"_{i}"
                train_config.replay_buffer_dir = replay_buffer_dir + f"_{i}"
                train_cmd = get_base_cmd(train_config) + ["--command", "train"]
                process = subprocess.Popen(train_cmd)
                process.wait()

            # train_config.seed += 5
            # train_config.load_model = os.path.join(train_config.checkpoint_dir, "model.pth")
            # train_config.load_replay_buffer = train_config.replay_buffer_dir
            # train_config.checkpoint_dir = checkpoint_dir + "_1"
            # train_config.replay_buffer_dir = replay_buffer_dir + "_1"
            # train_cmd = get_base_cmd(train_config) + ["--command", "train"]
            # process = subprocess.Popen(train_cmd)
            # process.wait()  # Wait for the subprocess to finish

            # config.seed += 5
            # config.load_model = os.path.join(config.checkpoint_dir, "model.pth")
            # config.load_replay_buffer = config.replay_buffer_dir
            # config.replay_buffer_dir = replay_buffer_dir + "2"
            # train_cmd = get_base_cmd(config) + ["--command", "train"]
            # process = subprocess.Popen(train_cmd)
            # process.wait()  # Wait for the subprocess to finish

        except KeyboardInterrupt:
            handle_interrupt(signal.SIGINT, None)

        print()
        print("*" * 50)
        print(
            f"EVALUATION TQC: Use the latest model from {train_config.checkpoint_dir} for evaluation"
        )
        print("*" * 50)
        print()
        config.load_eval_model = os.path.join(train_config.checkpoint_dir, "model.pth")
    else:
        print()
        print("*" * 50)
        print(f"EVALUATION: Use the model at {config.load_eval_model} for evaluation")
        print("*" * 50)
        print()

    eval_config = copy.deepcopy(config)
    eval_config.replay_buffer_dir = os.path.join(
        config.source_dir, "local_assets", "replay_buffers", "tmp"
    )
    eval_cmd = get_base_cmd(eval_config)
    eval_cmd = eval_cmd + ["--command", "eval"]
    eval_cmd = eval_cmd + ["--load_eval_model", str(config.load_eval_model)]

    subprocess.run(eval_cmd, check=True)


def get_base_cmd(config: TrainConfig):
    base_cmd = ["python", "./saris/train_tqc.py"]
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
        "--warmup_steps",
        str(config.warmup_steps),
        "--policy_frequency",
        str(config.policy_frequency),
        "--target_network_frequency",
        str(config.target_network_frequency),
        "--alpha",
        str(config.alpha),
        "--n_quantiles",
        str(config.n_quantiles),
        "--top_quantiles_to_drop_per_net",
        str(config.top_quantiles_to_drop_per_net),
        # wandb logging
        "--wandb_mode",
        str(config.wandb_mode),
        "--project",
        str(config.project),
        "--group",
        str(config.group),
        "--name",
        str(config.name),
    ]

    return base_cmd


if __name__ == "__main__":
    main()
