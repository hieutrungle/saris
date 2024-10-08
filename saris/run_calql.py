import os
import uuid
from dataclasses import dataclass
from typing import Optional
import pyrallis
import saris
import importlib.resources
import subprocess


@dataclass
class TrainConfig:
    # Experiment
    command: str = "train"  # Command for "train" or "eval"
    env_id: str = "wireless-sigmap-v0"  # environment name
    offline_iterations: int = int(0)  # Number of offline updates
    online_iterations: int = int(10_001)  # Number of online updates
    learning_starts: int = int(300)  # Number of steps before learning starts
    checkpoint_path: Optional[str] = None  # Save path
    load_model: str = "-1"  # Model load file name for resume training, "" doesn't load
    offline_data_dir: str = "-1"  # Offline data directory
    sionna_config_file: str = ""  # Sionna config file
    verbose: bool = False  # Print debug information
    save_freq: int = int(100)  # How often (time steps) we save

    # Environment
    ep_len: int = 75  # Max length of episode
    eval_ep_len: int = 50  # Max length of evaluation episode
    num_envs: int = 1  # Number of parallel environments
    seed: int = 10  # Sets Gym, PyTorch and Numpy seeds
    eval_seed: int = 100  # Eval environment seed

    # CQL
    n_updates: int = 20  # Number of updates per step
    offline_buffer_size: int = 300_000  # Offline replay buffer size
    online_buffer_size: int = 75_000  # Online replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.85  # Discount factor
    alpha_multiplier: float = 1.0  # Multiplier for alpha in loss
    use_automatic_entropy_tuning: bool = True  # Tune entropy
    backup_entropy: bool = False  # Use backup entropy
    policy_lr: float = 1e-4  # Policy learning rate
    qf_lr: float = 3e-4  # Critics learning rate
    tau: float = 5e-3  # Target network update rate
    target_update_period: int = 1  # Frequency of target nets updates
    cql_alpha: float = 5.0  # CQL offline regularization parameter
    cql_alpha_online: float = 2.0  # CQL online regularization parameter
    cql_n_actions: int = 10  # Number of sampled actions
    cql_importance_sample: bool = True  # Use importance sampling
    cql_lagrange: bool = True  # Use Lagrange version of CQL
    cql_target_action_gap: float = 0.8  # Action gap
    cql_temp: float = 1.0  # CQL temperature
    cql_max_target_backup: bool = True  # Use max target backup
    cql_clip_diff_min: float = -200  # Q-function lower loss clipping
    cql_clip_diff_max: float = 200  # Q-function upper loss clipping
    orthogonal_init: bool = True  # Orthogonal initialization
    normalize: bool = True  # Normalize states
    normalize_reward: bool = True  # Normalize reward
    q_n_hidden_layers: int = 2  # Number of hidden layers in Q networks

    # Cal-QL
    mixing_ratio: float = 0.0  # Data mixing ratio for online tuning, should be ~0.1
    is_sparse_reward: bool = False  # Use sparse reward

    # Wandb logging
    project: str = "SARIS"  # wandb project name
    group: str = "Cal-QL"  # wandb group name
    name: str = "Online-Learning"  # wandb run name

    def __post_init__(self):
        lib_dir = importlib.resources.files(saris)
        source_dir = os.path.dirname(lib_dir)
        self.source_dir = source_dir

        self.name = f"{self.group}__{self.name}__{self.env_id}__{str(uuid.uuid4())[:8]}"
        if self.checkpoint_path is None:
            log_dir = os.path.join(self.source_dir, "local_assets", "logs")
            log_path = os.path.join(log_dir, self.name)
            self.checkpoint_path = log_path


@pyrallis.wrap()
def main(config: TrainConfig):

    base_cmd = [
        "poetry",
        "run",
        "train_calql",
        "--env_id",
        str(config.env_id),
        "--offline_iterations",
        str(config.offline_iterations),
        "--online_iterations",
        str(config.online_iterations),
        "--learning_starts",
        str(config.learning_starts),
        "--checkpoint_path",
        str(config.checkpoint_path),
        "--load_model",
        str(config.load_model),
        "--offline_data_dir",
        str(config.offline_data_dir),
        "--sionna_config_file",
        str(config.sionna_config_file),
        "--verbose",
        str(config.verbose),
        "--save_freq",
        str(config.save_freq),
        "--ep_len",
        str(config.ep_len),
        "--eval_ep_len",
        str(config.eval_ep_len),
        "--num_envs",
        str(config.num_envs),
        "--seed",
        str(config.seed),
        "--eval_seed",
        str(config.eval_seed),
        "--n_updates",
        str(config.n_updates),
        "--offline_buffer_size",
        str(config.offline_buffer_size),
        "--online_buffer_size",
        str(config.online_buffer_size),
        "--batch_size",
        str(config.batch_size),
        "--discount",
        str(config.discount),
        "--alpha_multiplier",
        str(config.alpha_multiplier),
        "--use_automatic_entropy_tuning",
        str(config.use_automatic_entropy_tuning),
        "--backup_entropy",
        str(config.backup_entropy),
        "--policy_lr",
        str(config.policy_lr),
        "--qf_lr",
        str(config.qf_lr),
        "--tau",
        str(config.tau),
        "--target_update_period",
        str(config.target_update_period),
        "--cql_alpha",
        str(config.cql_alpha),
        "--cql_alpha_online",
        str(config.cql_alpha_online),
        "--cql_n_actions",
        str(config.cql_n_actions),
        "--cql_importance_sample",
        str(config.cql_importance_sample),
        "--cql_lagrange",
        str(config.cql_lagrange),
        "--cql_target_action_gap",
        str(config.cql_target_action_gap),
        "--cql_temp",
        str(config.cql_temp),
        "--cql_max_target_backup",
        str(config.cql_max_target_backup),
        "--cql_clip_diff_min",
        str(config.cql_clip_diff_min),
        "--cql_clip_diff_max",
        str(config.cql_clip_diff_max),
        "--orthogonal_init",
        str(config.orthogonal_init),
        "--normalize",
        str(config.normalize),
        "--normalize_reward",
        str(config.normalize_reward),
        "--q_n_hidden_layers",
        str(config.q_n_hidden_layers),
        "--mixing_ratio",
        str(config.mixing_ratio),
        "--is_sparse_reward",
        str(config.is_sparse_reward),
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
