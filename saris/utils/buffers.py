from typing import Tuple
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

        self._states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._mc_returns = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32).to(self._device)

    # # Loads data in d4rl format, i.e. from Dict[str, np.array].
    # def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
    #     if self._size != 0:
    #         raise ValueError("Trying to load data into non-empty replay buffer")
    #     n_transitions = data["observations"].shape[0]
    #     if n_transitions > self._buffer_size:
    #         raise ValueError("Replay buffer is smaller than the dataset you are trying to load!")
    #     self._states[:n_transitions] = self._to_tensor(data["observations"])
    #     self._actions[:n_transitions] = self._to_tensor(data["actions"])
    #     self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
    #     self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
    #     self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
    #     self._mc_returns[:n_transitions] = self._to_tensor(data["mc_returns"][..., None])
    #     self._size += n_transitions
    #     self._pointer = min(self._size, n_transitions)

    #     print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        mc_returns = self._mc_returns[indices]
        return (states, actions, rewards, next_states, dones, mc_returns)

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._dones[self._pointer] = self._to_tensor(done)
        self._mc_returns[self._pointer] = 0.0

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)

    def add_batch_transition(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        batch_size = states.shape[0]
        assigned_indices = np.arange(self._pointer, self._pointer + batch_size) % self._buffer_size
        self._states[assigned_indices] = self._to_tensor(states)
        self._actions[assigned_indices] = self._to_tensor(actions)
        self._rewards[assigned_indices] = self._to_tensor(rewards)
        self._next_states[assigned_indices] = self._to_tensor(next_states)
        self._dones[assigned_indices] = self._to_tensor(dones)
        self._mc_returns[assigned_indices] = 0.0

        self._pointer = (self._pointer + batch_size) % self._buffer_size
        self._size = min(self._size + batch_size, self._buffer_size)
