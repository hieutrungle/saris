from typing import Tuple, Dict
import numpy as np
import torch

TensorBatch = Tuple[torch.Tensor]


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._observations = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_observations = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._mc_returns = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32).to(self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_dataset(
        self,
        observations: np.ndarray[float],
        actions: np.ndarray[float],
        rewards: np.ndarray[float],
        next_observations: np.ndarray[float],
        dones: np.ndarray[bool],
        mc_returns: np.ndarray[float],
    ):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = observations.shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError("Replay buffer is smaller than the dataset you are trying to load!")
        self.add_batch_transition(
            observations, actions, rewards, next_observations, dones, mc_returns
        )

        print(f"Dataset size: {len(self)}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        observations = self._observations[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_observations = self._next_observations[indices]
        dones = self._dones[indices]
        mc_returns = self._mc_returns[indices]
        return (observations, actions, rewards, next_observations, dones, mc_returns)

    def add_transition(
        self,
        observation: np.ndarray[float],
        action: np.ndarray[float],
        reward: np.ndarray[float],
        next_observation: np.ndarray[float],
        done: np.ndarray[bool],
        mc_return: np.ndarray[float] = 0.0,
    ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._observations[self._pointer] = self._to_tensor(observation)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._next_observations[self._pointer] = self._to_tensor(next_observation)
        self._dones[self._pointer] = self._to_tensor(done)
        self._mc_returns[self._pointer] = self._to_tensor(mc_return)

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)

    def add_batch_transition(
        self,
        observations: np.ndarray[float],
        actions: np.ndarray[float],
        rewards: np.ndarray[float],
        next_observations: np.ndarray[float],
        dones: np.ndarray[bool],
        mc_returns: np.ndarray[float] = 0.0,
    ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        batch_size = observations.shape[0]
        assigned_indices = np.arange(self._pointer, self._pointer + batch_size) % self._buffer_size
        self._observations[assigned_indices] = self._to_tensor(observations)
        self._actions[assigned_indices] = self._to_tensor(actions)
        self._rewards[assigned_indices] = self._to_tensor(rewards)
        self._next_observations[assigned_indices] = self._to_tensor(next_observations)
        self._dones[assigned_indices] = self._to_tensor(dones)
        self._mc_returns[assigned_indices] = self._to_tensor(mc_returns)

        self._pointer = (self._pointer + batch_size) % self._buffer_size
        self._size = min(self._size + batch_size, self._buffer_size)

    def __len__(self):
        return self._size

    def max_size(self):
        return self._buffer_size
