from typing import Tuple
import os
import re
import subprocess
import time
import numpy as np
from gymnasium import Env, spaces
from saris.utils import utils


class WirelessEnvV0(Env):

    def __init__(
        self,
        sionna_config_file: str,
        log_string: str = "WirelessEnvV0",
        seed: int = 0,
        **kwargs
    ):
        super(WirelessEnvV0, self).__init__()
        self.sionna_config_file = sionna_config_file
        self.log_string = log_string
        self.seed = seed
        self.np_rng = np.random.default_rng(self.seed)
        self.current_time = "_" + time.strftime("%d-%m-%Y_%H-%M-%S")

        self._rx_position = np.zeros((3,))
        self._tx_position = np.zeros((3,))
        self._focal_points = np.zeros((6,))  # 2 focal points for each device

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        self.state = np.concatenate(
            [self._rx_position, self._tx_position, self._focal_points]
        )
        self.info = {}

    def _get_observation(self) -> dict:
        observation = np.concatenate(
            [self._rx_position, self._tx_position, self._focal_points]
        )
        return observation

    def _get_observation_space(self) -> spaces.Box:
        observation_shape = self._get_observation().shape
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=observation_shape, dtype=np.float32
        )

    def _get_action_space(self) -> spaces.Box:
        action_shape = self._focal_points.shape
        return spaces.Box(low=-1.0, high=1.0, shape=action_shape, dtype=np.float32)

    def reset(self, seed: int = None, options: dict = None) -> np.ndarray:
        if seed is not None:
            self.seed = seed
            self.np_rng = np.random.default_rng(self.seed)
        sionna_config = utils.load_yaml_file(self.sionna_config_file)
        self._rx_position = list(sionna_config["rx_position"])
        self._rx_position = np.asarray(self._rx_position, dtype=np.float32)
        self._tx_position = list(sionna_config["tx_position"])
        self._tx_position = np.asarray(self._tx_position, dtype=np.float32)
        # TODO: get random uniform number from a sphere with radius that is half the distance between RIS and RX; RIS and tx
        delta_focal_points = self.np_rng.uniform(-2, 2, size=self._focal_points.shape)
        self._focal_points = (
            np.concatenate((self._rx_position, self._tx_position)) + delta_focal_points
        )
        self._focal_points = np.asarray(self._focal_points, dtype=np.float32)
        self.state = self._get_observation()

        return self.state, self.info

    def step(
        self, action: np.ndarray, **kwargs
    ) -> Tuple[dict, float, bool, bool, dict]:

        self._focal_points = self._focal_points + action
        next_observation = self._get_observation()

        truncated = False
        reward = self._get_reward()
        terminated = False

        return next_observation, reward, terminated, truncated, self.info

    def _get_reward(self) -> float:
        return 0.0
