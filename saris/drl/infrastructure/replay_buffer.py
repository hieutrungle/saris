import numpy as np
from typing import Tuple, Union
import os
import json


class ReplayBuffer:
    def __init__(
        self,
        max_size: int = 100000,
        saved_dir="",
        name="wireless_replay_buffer",
        prefix_idx=0,
        seed=0,
    ):
        """
        A replay buffer for wireless environments.

        It is an empty DataBaches object with the ability to insert data.
        """
        super().__init__()
        self.max_size = max_size
        self.saved_dir = saved_dir
        self.name = name
        self.prefix_idx = prefix_idx
        self.seed = seed
        self.size_counter = 0
        self.saved_path = os.path.join(self.saved_dir, f"{self.name}.txt")
        self.rng = np.random.default_rng(seed=self.seed)

        self.observations: np.ndarray = None
        self.actions: np.ndarray = None
        self.rewards: np.ndarray = None
        self.next_observations: np.ndarray = None
        self.dones: np.ndarray = None

        if os.path.exists(self.saved_path):
            # TODO: make this more efficient
            tmp_container = self._load_n_to_last_line(self.saved_path, n=self.max_size)
            for line in tmp_container:
                batch = json.loads(line)
                batch = self.to_numpy(batch)
                self.insert(
                    observation=batch["observation"],
                    action=batch["action"],
                    reward=batch["reward"],
                    next_observation=batch["next_observation"],
                    done=batch["done"],
                    is_saved=False,
                )
            print(f"Loaded {len(tmp_container)} samples from {self.saved_path}")
            print(f"Current replay buffer size: {self.size_counter}")

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions from the replay buffer.

        Use like:
            observation, action, reward, next_observation, done = replay_buffer.sample(batch_size)
        """
        indices = self.rng.choice(self.size_counter, batch_size, replace=False)
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
        }

    def __len__(self):
        return self.size_counter

    def _load_n_to_last_line(self, filename: str, n: int = 1) -> list:
        """Returns n entries before last line of a file (n=1 gives last line)"""
        container = []
        num_newlines = 0
        with open(filename, "rb") as f:
            try:
                f.seek(-2, os.SEEK_END)
                while num_newlines < n:
                    f.seek(-2, os.SEEK_CUR)
                    if f.read(1) == b"\n":
                        num_newlines += 1
                        pos = f.tell()
                        container.append(f.readline().decode())
                        f.seek(pos)
            except OSError:
                f.seek(0)
                container.append(f.readline().decode())

        return container

    def to_numpy(self, data: Union[dict, list, float]) -> Union[np.ndarray, dict]:
        if isinstance(data, dict):
            return {key: self.to_numpy(value) for key, value in data.items()}
        elif (
            isinstance(data, list) or isinstance(data, tuple) or isinstance(data, float)
        ):
            return np.array(data)
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        return

    def insert(
        self,
        /,
        observation: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_observation: np.ndarray,
        done: np.ndarray,
        is_saved: bool = True,
    ):
        """
        Insert a single transition into the replay buffer.

        Use like:
            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
        """
        if self.observations is None:
            self.observations = np.empty((self.max_size, *observation.shape))
            self.actions = np.empty((self.max_size, *action.shape))
            self.rewards = np.empty((self.max_size, *reward.shape))
            self.next_observations = np.empty((self.max_size, *next_observation.shape))
            self.dones = np.empty((self.max_size, *done.shape))

        idx = self.size_counter % self.max_size
        self.observations[idx] = observation
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_observations[idx] = next_observation
        self.dones[idx] = done

        self.size_counter += 1

        if is_saved:
            self.save_data_to_file(
                self.saved_path, observation, action, reward, next_observation, done
            )

    def insert_batch(
        self,
        /,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        is_saved: bool = True,
    ):
        """
        Insert a batch of transitions into the replay buffer.

        Use like:
            replay_buffer.insert_batch(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
        """
        if self.observations is None:
            self.observations = np.empty((self.max_size, *observations[0].shape))
            self.actions = np.empty((self.max_size, *actions[0].shape))
            self.rewards = np.empty((self.max_size, *rewards[0].shape))
            self.next_observations = np.empty(
                (self.max_size, *next_observations[0].shape)
            )
            self.dones = np.empty((self.max_size, *dones[0].shape))

        idxes = np.arange(self.size_counter, self.size_counter + observations.shape[0])
        idxes = idxes % self.max_size
        self.observations[idxes] = observations
        self.actions[idxes] = actions
        self.rewards[idxes] = rewards
        self.next_observations[idxes] = next_observations
        self.dones[idxes] = dones

        self.size_counter += observations.shape[0]

        if is_saved:
            for i in range(observations.shape[0]):
                self.save_data_to_file(
                    self.saved_path,
                    observations[i],
                    actions[i],
                    rewards[i],
                    next_observations[i],
                    dones[i],
                )

    def save_data_to_file(
        self,
        saved_path: str,
        observation: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_observation: np.ndarray,
        done: np.ndarray,
    ) -> None:
        batch = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
            "done": done,
        }
        with open(saved_path, "a") as f:
            json.dump(batch, f, cls=NpEncoder)
            f.write("\n")


class NpEncoder(json.JSONEncoder):
    # json format for saving numpy array
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
