import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation

from typing import Tuple, Optional
from collections import OrderedDict
import subprocess
import time
import numpy as np
from gymnasium import Env, spaces
import pickle
import glob
import math
import copy
from saris.utils import utils
from saris.blender_script import shared_utils
from saris import sigmap
from sionna.channel import (
    cir_to_ofdm_channel,
    subcarrier_frequencies,
    time_lag_discrete_time_channel,
    cir_to_time_channel,
    time_to_ofdm_channel,
)
import tensorflow as tf


class SharedAPV0(Env):

    def __init__(
        self,
        idx: int,
        sionna_config_file: str,
        log_string: str = "SharedAPV0",
        seed: int = 0,
        **kwargs,
    ):
        super(SharedAPV0, self).__init__()

        self.idx = idx
        self.log_string = log_string
        self.seed = seed + idx
        self.np_rng = np.random.default_rng(self.seed)

        tf.random.set_seed(self.seed)
        print(f"using GPU: {tf.config.experimental.list_physical_devices('GPU')}")

        self.sionna_config = utils.load_config(sionna_config_file)

        # positions
        self.rx_pos = np.array(self.sionna_config["rx_positions"], dtype=np.float32)
        self.rt_pos = np.array(self.sionna_config["rt_positions"], dtype=np.float32)
        self.tx_pos = np.array(self.sionna_config["tx_positions"], dtype=np.float32)

        # orient the tx
        tx_orientations = []
        for i in range(len(self.rt_pos)):
            r, theta, phi = compute_rot_angle(self.tx_pos[i], self.rt_pos[i])
            tx_orientations.append([phi, theta - math.pi / 2, 0.0])
        self.sionna_config["tx_orientations"] = tx_orientations

        # Set up logging
        self.current_time = "_" + time.strftime("%d-%m-%Y_%H-%M-%S")

        # Set up action and observation space
        # Load the reflector configuration, angle is in radians
        reflector_configs = shared_utils.get_config_shared_ap()
        self.theta_configs = reflector_configs[0]
        self.phi_configs = reflector_configs[1]
        self.num_groups = reflector_configs[2]
        self.num_elements_per_group = reflector_configs[3]

        # angles = [theta, phi] for each tile
        # theta: zenith angle, phi: azimuth angle
        self.init_thetas = []
        self.init_phis = []
        # self.init_angles = []
        self.angle_spaces = []
        theta_highs = []
        phi_highs = []
        theta_lows = []
        phi_lows = []

        for i in range(len(self.rt_pos)):
            init_theta = self.theta_configs[0][i]
            init_phi = self.phi_configs[0][i]
            # init_per_group = [init_phi] + [init_theta] * self.num_elements_per_group
            # self.init_angles.append(np.concatenate([init_per_group] * self.num_groups))

            theta_high = self.theta_configs[2][i]
            phi_high = self.phi_configs[2][i]
            per_group_high = [phi_high] + [theta_high] * self.num_elements_per_group
            angle_high = np.concatenate([per_group_high] * self.num_groups)
            theta_low = self.theta_configs[1][i]
            phi_low = self.phi_configs[1][i]
            per_group_low = [phi_low] + [theta_low] * self.num_elements_per_group
            angle_low = np.concatenate([per_group_low] * self.num_groups)
            self.angle_spaces.append(spaces.Box(low=angle_low, high=angle_high, dtype=np.float32))

            # storage
            self.init_thetas.append(init_theta)
            self.init_phis.append(init_phi)
            theta_highs.append(theta_high)
            phi_highs.append(phi_high)
            theta_lows.append(theta_low)
            phi_lows.append(phi_low)

        # position space
        self.position_space = spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(len(np.concatenate([self.rx_pos, self.rt_pos]).flatten()),),
            dtype=np.float32,
        )

        # Channels space: CSI from both reflectors
        self.bandwidth = 100e6  # 100MHz
        self.maximum_delay_spread = 10e-9  # 10ns
        # self.maximum_delay_spread = 1e-6  # 1us
        # (self.l_min, self.l_max) = time_lag_discrete_time_channel(
        #     self.bandwidth, self.maximum_delay_spread
        # )
        self.l_min = 0
        self.l_max = 0
        num_tx_ants = self.sionna_config["tx_num_rows"] * self.sionna_config["tx_num_cols"]
        self.channel_spaces = []
        shape = ((len(self.rx_pos) * num_tx_ants * 2 * int(self.l_max - self.l_min + 1)),)
        for _ in range(len(self.rt_pos)):
            self.channel_spaces.append(
                spaces.Box(low=-1.0, high=1.0, shape=shape, dtype=np.float32)
            )

        # Observation space
        self._initialize_observation_space()

        # action space
        self._initialize_action_space()

        # focal vecs space <-> action space
        # Action is a changes in focals [delta_r, delta_theta, _delta_phi] for each group
        # focals = [r, theta, phi] for each group
        self._initialize_focal_spaces(theta_highs, phi_highs, theta_lows, phi_lows)

        # Reward set up
        self.taken_steps = 0.0
        self.prev_gains = [0.0 for _ in range(len(self.rx_pos))]
        self.cur_gains = [0.0 for _ in range(len(self.rx_pos))]

        # range for new rx positions
        self.rx_pos_range = [[-4.39634, 4.40138], [-19.6031, -10.7053]]  # x  # y
        self.obstacle_pos = [[2.37, -17.656], [-2.6, -11.945], [1.86, -10.7654], [-1.71, -15.6846]]
        self.restricted_areas = [
            [[-3.3031, -11.2042], [-1.89527, -11.2409], [-1.80826, -12.6548], [-3.15467, -12.603]],
            [[1.10165, 0.034565], [-14.487, -1.39775], [-12.2953, -2.18156]],
        ]

        self.default_sionna_config = copy.deepcopy(self.sionna_config)

        self.eval_mode = False

    def _initialize_focal_spaces(self, theta_highs, phi_highs, theta_lows, phi_lows):
        r_high = 40.0
        r_low = 5.0
        self.focal_spaces = []
        for i in range(len(self.rt_pos)):
            theta_high = theta_highs[i]
            theta_low = theta_lows[i]
            phi_high = phi_highs[i]
            phi_low = phi_lows[i]
            self.focal_spaces.append(
                spaces.Box(
                    low=np.asarray([r_low, theta_low, phi_low] * self.num_groups),
                    high=np.asarray([r_high, theta_high, phi_high] * self.num_groups),
                    dtype=np.float32,
                )
            )

    def _initialize_action_space(self):
        action_space_shape = tuple((3 * self.num_groups,))
        action_spaces = []
        for i in range(len(self.rt_pos)):
            action_spaces.append(
                spaces.Box(low=-1.0, high=1.0, shape=action_space_shape, dtype=np.float32)
            )
        low = np.asarray([space.low for space in action_spaces])
        high = np.asarray([space.high for space in action_spaces])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _initialize_observation_space(self):
        channel_space = np.concatenate(
            [space.low for space in self.channel_spaces]
            + [space.high for space in self.channel_spaces]
        )
        angle_space = np.concatenate(
            [space.low for space in self.angle_spaces] + [space.high for space in self.angle_spaces]
        )
        position_space = np.concatenate([self.position_space.low, self.position_space.high])
        low = np.concatenate(
            [
                channel_space[: len(channel_space) // 2],
                angle_space[: len(angle_space) // 2],
                position_space[: len(position_space) // 2],
            ]
        )
        high = np.concatenate(
            [
                channel_space[len(channel_space) // 2 :],
                angle_space[len(angle_space) // 2 :],
                position_space[len(position_space) // 2 :],
            ]
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _cal_area(self, pt1, pt2, pt3):

        return abs(
            (pt1[0] * (pt2[1] - pt3[1]) + pt2[0] * (pt3[1] - pt1[1]) + pt3[0] * (pt1[1] - pt2[1]))
            / 2.0
        )

    def _cal_distance(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def _is_inside(self, border, target):
        # borrow and modified from https://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-2d-point-is-within-a-polygon
        degree = 0
        for i in range(len(border)):
            a = border[i]
            b = border[(i + 1) % len(border)]

            # calculate distance of vector
            e1 = self._cal_distance(a[0], a[1], b[0], b[1])
            e2 = self._cal_distance(target[0], target[1], a[0], a[1])
            e3 = self._cal_distance(target[0], target[1], b[0], b[1])

            # calculate direction of vector
            ta_x = a[0] - target[0]
            ta_y = a[1] - target[1]
            tb_x = b[0] - target[0]
            tb_y = b[1] - target[1]

            cross = tb_y * ta_x - tb_x * ta_y
            clockwise = cross < 0

            # calculate sum of angles
            if clockwise:
                degree += math.degrees(math.acos((e2 * e2 + e3 * e3 - e1 * e1) / (2.0 * e2 * e3)))
            else:
                degree -= math.degrees(math.acos((e2 * e2 + e3 * e3 - e1 * e1) / (2.0 * e2 * e3)))

        if abs(abs(round(degree)) - 360.0) <= 2.0:
            return True
        return False

    def _is_eligible(self, pt, obstacle_pos, rx_pos):
        """
        Check if the position is eligible for new rx_pos.
        This function checks if the new rx_pos is not too close to obstacles and not too close to existing rx_pos.
        """
        # append new rx_pos that are not too close to obstacles
        is_obs_overlaped = False
        for pos in obstacle_pos:
            if np.linalg.norm(np.array(pos[:2]) - np.array(pt)) < 0.7:
                is_obs_overlaped = True
                # print(f"Obstacle at {pos[:2]} is too close to {pt}")
                break

        # make sure that distance between rx_pos is at least 1
        if not is_obs_overlaped:
            if not any(np.linalg.norm(np.array(pos[:2]) - np.array(pt)) < 1.5 for pos in rx_pos):
                # print("Eligible position found at", pt)
                return True
        # print(f"Position {pt} is not eligible")
        return False

    def _prepare_rx_positions(self):
        rx_pos = []
        while len(rx_pos) < len(self.sionna_config["rx_positions"]):
            x = self.np_rng.uniform(low=self.rx_pos_range[0][0], high=self.rx_pos_range[0][1])
            y = self.np_rng.uniform(low=self.rx_pos_range[1][0], high=self.rx_pos_range[1][1])
            pt = [x, y]
            if self._is_eligible(pt, self.obstacle_pos, rx_pos):
                rx_pos.append([x, y, 1.5])

        return rx_pos

    def reset(self, seed: int = None, options: dict = None) -> Tuple[dict, dict]:
        super().reset(seed=seed)

        self.sionna_config = copy.deepcopy(self.default_sionna_config)

        # reset rx_pos
        rx_pos = self._prepare_rx_positions()
        self.sionna_config["rx_positions"] = rx_pos
        self.rx_pos = np.array(rx_pos, dtype=np.float32)
        # print(f"rx_pos: {rx_pos}")
        # self.positions = np.asarray(rx_pos, dtype=np.float32).flatten()

        # We have 2 reflector -> 2 sets of Spherical_focal_vecs
        # theta_inits = [np.deg2rad(90.0), np.deg2rad(90.0)]
        # phi_inits = [np.deg2rad(22.7), np.deg2rad(-25.0)]

        # focal_spaces: contains the focal space for each reflector (implicitly represents the angles, observation)
        # focals: contain the focal action for each reflector (actual values to get the angles of each reflector)
        # angles: contain the angles of each reflector (from 'focals')
        # channels: contain the channels of each reflector (from Blender + Sionna using 'angles')
        # positions: UE positions + Reflector positions

        # noise to focals
        start_init = False
        if options is not None:
            start_init = options.get("start_init", False)
            print(f"\nRESET with start_init: {start_init}")

        self.focals = [None for _ in range(len(self.rt_pos))]
        for i in range(len(self.rt_pos)):
            low = self.focal_spaces[i].low
            high = self.focal_spaces[i].high
            if start_init:
                self.focals[i] = self.np_rng.uniform(low=low, high=high)
            else:
                self.focals[i] = self.np_rng.normal(
                    loc=(low + high) / 2.0, scale=abs(high - low) / 9.0
                )

            self.focals[i] = np.clip(self.focals[i], low, high)
        self.focals = np.asarray(self.focals, dtype=np.float32)

        # angles is a sets of angles from two reflectors
        # angles[0] is from reflector 1, angles[1] is from reflector 2
        self.angles = self._blender_step(self.focals)

        for i in range(len(self.angle_spaces)):
            self.angles[i] = np.clip(
                self.angles[i], self.angle_spaces[i].low, self.angle_spaces[i].high
            )
            self.angles[i] = np.asarray(self.angles[i], dtype=np.float32)

        eval_mode = self.eval_mode
        if options is not None:
            if "eval_mode" in options:
                eval_mode = options["eval_mode"]
        # eval_mode = False if options is None else options.get("eval_mode", False)
        self.eval_mode = eval_mode
        channels, self.prev_gains = self._run_sionna_dB(eval_mode)
        self.cur_gains = self.prev_gains

        # conversion of all components of the observation
        # Components: channels, angles, positions (each has 1 sets of values for each reflector)

        # Observation
        # Reflector 1: channel[0], angles[0], rx_pos + self.rt_pos[0]
        # Reflector 2: channel[1], angles[1], rx_pos + self.rt_pos[1]

        # return global observation
        # flatten_channels: [channel[0], channel[1]] -> concat((channel[0], channel[1]), axis=-1)
        # flatten_angles: [angles[0], angles[1]] -> concat((angles[0], angles[1]), axis=-1)
        # flatten_positions: [rx_pos + self.rt_pos[0], rx_pos + self.rt_pos[1]] -> concat((rx_pos, self.rt_pos), axis=-1)
        # observation: [flatten_channels, flatten_angles, flatten_positions]

        # Local observation is placed in the info dictionary
        # info: {[channel[0], angles[0], rx_pos + self.rt_pos[0]], [channel[1], angles[1], rx_pos + self.rt_pos[1]]}

        channels = np.asarray(channels)
        real_channels = np.asarray(channels.real, dtype=np.float32)
        imag_channels = np.asarray(channels.imag, dtype=np.float32)
        global_channels = np.concatenate([real_channels, imag_channels], axis=0).flatten()
        global_angles = self.angles.flatten()
        global_positions = np.concatenate([self.rx_pos, self.rt_pos], axis=0).flatten()
        # print(f"global_channels: {global_channels}")
        # print(f"global_positions: {global_positions}")
        # print(f"global_positions shape: {global_positions.shape}")
        # print(f"global_channels shape: {global_channels.shape}")
        # print(f"global_angles shape: {global_angles.shape}")
        observation = np.concatenate(
            [global_channels, global_angles, global_positions], axis=-1
        ).flatten()

        # Reflector local observation
        info = {}
        for i in range(len(self.rt_pos)):
            local_channels = np.concatenate(
                [real_channels[i : i + 1], imag_channels[i : i + 1]], axis=0
            ).flatten()
            local_positions = np.concatenate(
                [self.rx_pos, self.rt_pos[i : i + 1]], axis=0
            ).flatten()
            # print(f"local_channels: {local_channels}")
            # print(f"local_channels shape: {local_channels.shape}")
            # print(f"local_positions shape: {local_positions.shape}")
            local_ob = np.concatenate(
                [local_channels, self.angles[i].flatten(), local_positions], axis=-1
            ).flatten()
            info[i] = local_ob

        # real_channels = np.asarray(channels.real, dtype=np.float32)
        # imag_channels = np.asarray(channels.imag, dtype=np.float32)
        # channels = np.concatenate([real_channels, imag_channels], axis=-1)
        # observation = OrderedDict(channels=channels, angles=self.angles, positions=self.positions)

        self.taken_steps = 0.0

        return observation, info

        # # noise to focals
        # start_init = False
        # if options is not None:
        #     start_init = options.get("start_init", False)
        #     print(f"\nRESET with start_init: {start_init}")

        # if start_init:
        #     noise = self.np_rng.uniform(low=self.focal_noise_low, high=self.focal_noise_high)
        #     self.focals = np.asarray([15.0, np.deg2rad(90), np.deg2rad(135)] * self.num_groups)
        #     self.focals += noise
        # else:
        #     low = self.focal_vec_space.low
        #     high = self.focal_vec_space.high
        #     self.focals = self.np_rng.normal(loc=(low + high) / 2.0, scale=abs(high - low) / 9.0)

        # # self.focals += noise
        # self.focals = np.clip(self.focals, self.focal_vec_space.low, self.focal_vec_space.high)
        # self.angles = self._blender_step(self.focals)
        # self.angles = np.clip(
        #     self.angles, self.angle_space.low, self.angle_space.high, dtype=np.float32
        # )

        # channels, self.prev_gains = self._run_sionna_dB(eval_mode=self.eval_mode)
        # self.cur_gains = self.prev_gains

        # real_channels = np.asarray(channels.real, dtype=np.float32)
        # imag_channels = np.asarray(channels.imag, dtype=np.float32)
        # channels = np.concatenate([real_channels, imag_channels], axis=-1)
        # observation = OrderedDict(channels=channels, angles=self.angles, positions=self.positions)

        # self.taken_steps = 0.0

        # return observation, {}

    def step(self, action: np.ndarray, **kwargs) -> Tuple[dict, float, bool, bool, dict]:

        self.taken_steps += 1.0
        self.prev_gains = self.cur_gains

        # actions: [num_reflectors, num_groups * 3]: [r, theta, phi] for each group
        acs = []
        for a in action:
            tmp = np.reshape(a, (self.num_groups, 3))
            tmp[:, 0] = tmp[:, 0]
            tmp[:, 1] = np.deg2rad(tmp[:, 1])
            tmp[:, 2] = np.deg2rad(tmp[:, 2])
            a = np.reshape(tmp, a.shape)
            acs.append(a)
        action = np.asarray(acs, dtype=np.float32)

        # tmp = np.reshape(action, (self.num_groups, 3))
        # tmp[:, 0] = tmp[:, 0]
        # tmp[:, 1] = np.deg2rad(tmp[:, 1])
        # tmp[:, 2] = np.deg2rad(tmp[:, 2])
        # action = np.reshape(tmp, action.shape)

        self.focals = self.focals + action
        low = np.asarray([space.low for space in self.focal_spaces])
        high = np.asarray([space.high for space in self.focal_spaces])
        out_of_bounds = np.sum((self.focals < low) + (self.focals > high), dtype=np.float32)
        self.focals = np.clip(self.focals, low, high)

        self.angles = self._blender_step(self.focals)
        self.angles = np.asarray(self.angles, dtype=np.float32)
        # if angles values are out of bounds, print warning
        low = np.asarray([space.low for space in self.angle_spaces])
        high = np.asarray([space.high for space in self.angle_spaces])
        if np.any(self.angles < low) or np.any(self.angles > high):
            print("Warning: angles out of bounds")

        truncated = False
        if self.taken_steps > 100:
            truncated = True
        terminated = False
        channels, self.cur_gains = self._run_sionna_dB(eval_mode=self.eval_mode)

        reward = self._cal_reward(self.prev_gains, self.cur_gains, out_of_bounds)

        step_info = {
            "prev_path_gains": self.prev_gains,
            "path_gains": self.cur_gains,
        }

        channels = np.asarray(channels)
        real_channels = np.asarray(channels.real, dtype=np.float32)
        imag_channels = np.asarray(channels.imag, dtype=np.float32)
        global_channels = np.concatenate([real_channels, imag_channels], axis=0).flatten()
        global_angles = self.angles.flatten()
        global_positions = np.concatenate([self.rx_pos, self.rt_pos], axis=0).flatten()
        next_observation = np.concatenate(
            [global_channels, global_angles, global_positions], axis=-1
        ).flatten()

        for i in range(len(self.rt_pos)):
            local_channels = np.concatenate(
                [real_channels[i : i + 1], imag_channels[i : i + 1]], axis=0
            ).flatten()
            local_positions = np.concatenate(
                [self.rx_pos, self.rt_pos[i : i + 1]], axis=0
            ).flatten()
            # print(f"local_channels: {local_channels}")
            # print(f"local_channels shape: {local_channels.shape}")
            # print(f"local_positions shape: {local_positions.shape}")
            local_ob = np.concatenate(
                [local_channels, self.angles[i].flatten(), local_positions], axis=-1
            ).flatten()
            step_info[f"local_ob_{i}"] = local_ob

        # real_channels = np.asarray(channels.real, dtype=np.float32)
        # imag_channels = np.asarray(channels.imag, dtype=np.float32)
        # channels = np.concatenate([real_channels, imag_channels], axis=-1)
        # next_observation = OrderedDict(
        #     channels=channels, angles=self.angles, positions=self.positions
        # )

        return next_observation, reward, terminated, truncated, step_info

    def _cal_reward(
        self, prev_gains: np.ndarray, cur_gains: np.ndarray, out_of_bounds: float
    ) -> float:

        adjusted_gain = np.mean(cur_gains)
        adjusted_gain = np.where(
            adjusted_gain < -95.0,
            (adjusted_gain + 95.0) / 10.0,
            (adjusted_gain + 95.0) / 5.0 + 1.5,
        )
        gain_diff = np.mean(cur_gains - prev_gains)

        reward = float(adjusted_gain + 0.03 * gain_diff - 0.3 * out_of_bounds) / 2.0

        return reward

    def _blender_step(self, focals: np.ndarray[float]) -> np.ndarray[float]:
        """
        Step the environment using Blender.

        If action is not given, the environment stays the same with the given angles.
        """
        # Blender export
        blender_app = utils.get_os_dir("BLENDER_APP")
        blender_dir = utils.get_os_dir("BLENDER_DIR")
        source_dir = utils.get_os_dir("SOURCE_DIR")
        assets_dir = utils.get_os_dir("ASSETS_DIR")
        tmp_dir = utils.get_os_dir("TMP_DIR")
        scene_name = f"{self.sionna_config['scene_name']}_{self.idx}"
        blender_output_dir = os.path.join(assets_dir, "blender", scene_name)

        data_path = os.path.join(
            tmp_dir, f"data-{self.log_string}-{self.current_time}-{self.idx}.pkl"
        )

        with open(data_path, "wb") as f:
            pickle.dump(focals, f)

        blender_script = os.path.join(source_dir, "saris", "blender_script", "bl_shared_ap.py")
        blender_cmd = [
            blender_app,
            "-b",
            os.path.join(blender_dir, "models", f"{scene_name}.blend"),
            "--python",
            blender_script,
            "--",
            "-i",
            data_path,
            "-o",
            blender_output_dir,
        ]
        bl_output_txt = os.path.join(tmp_dir, "bl_outputs.txt")
        subprocess.run(blender_cmd, check=True, stdout=open(bl_output_txt, "w"))
        # subprocess.run(blender_cmd, check=True)

        with open(data_path, "rb") as f:
            angles = pickle.load(f)
        angles = np.asarray(angles, dtype=np.float32)
        return angles

    def _run_sionna_dB(self, eval_mode: bool = False) -> np.ndarray[np.complex64, float]:

        # self._prepare_geometry()
        # path gain shape: [num_rx]
        channels, path_gains = self._run_sionna(eval_mode=eval_mode)
        path_gain_dBs = utils.linear2dB(path_gains)
        return channels, path_gain_dBs

    def _run_sionna(self, eval_mode: bool = False) -> Tuple[tf.Tensor, np.ndarray]:

        # Set up geometry paths for Sionna script
        assets_dir = utils.get_os_dir("ASSETS_DIR")
        scene_name = f"{self.sionna_config['scene_name']}_{self.idx}"
        blender_output_dir = os.path.join(assets_dir, "blender", scene_name)
        compute_scene_dir = os.path.join(blender_output_dir, "ceiling_idx")
        compute_scene_path = glob.glob(os.path.join(compute_scene_dir, "*.xml"))[0]
        viz_scene_dir = os.path.join(blender_output_dir, "idx")
        viz_scene_path = glob.glob(os.path.join(viz_scene_dir, "*.xml"))[0]

        sig_cmap = sigmap.engine.SignalCoverageMap(
            self.sionna_config, compute_scene_path, viz_scene_path
        )

        paths = sig_cmap.compute_paths()
        paths.normalize_delays = False
        cir = paths.cir()
        # a: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, num_time_steps], tf.complex
        a, tau = cir
        # [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, self.l_max - self.l_min + 1], tf.complex
        channels: tf.Tensor = cir_to_time_channel(self.bandwidth, a, tau, self.l_min, self.l_max)

        coverage_map = sig_cmap.compute_cmap()
        # path_gains = sig_cmap.get_path_gain(coverage_map)
        path_gains = []
        for pos in coverage_map.rx_pos:
            path_gain = coverage_map.path_gain[:, pos[1], pos[0]]
            path_gains.append(path_gain[0])
        path_gains = np.asarray(path_gains)

        if eval_mode:
            # Path for outputing iamges if we want to visualize the coverage map
            img_dir = os.path.join(
                assets_dir, "images", self.log_string + self.current_time + f"_{self.idx}"
            )
            render_filename = utils.create_filename(img_dir, f"{scene_name}_00000.png")
            sig_cmap.render_to_file(coverage_map, filename=render_filename)

        channels = tf.squeeze(channels, axis=(0, 2, 5, 6))
        channels = tf.transpose(channels, perm=[1, 0, 2])
        channels = np.asarray(channels, dtype=np.complex64)
        sig_cmap.free_memory()

        return channels, path_gains


def compute_rot_angle(pt1: list, pt2: list) -> Tuple[float, float, float]:
    """Compute the rotation angles for vector pt1 to pt2."""
    x = pt2[0] - pt1[0]
    y = pt2[1] - pt1[1]
    z = pt2[2] - pt1[2]

    return cartesian2spherical(x, y, z)


def cartesian2spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    # theta: zenith angle (0, pi), phi: azimuthal angle (0, 2pi)
    r = math.sqrt(x**2 + y**2 + z**2)
    theta = math.acos(z / r)
    phi = math.atan2(y, x)
    return r, theta, phi


def spherical2cartesian(r: float, theta: float, phi: float) -> Tuple[float, float, float]:
    # theta: zenith angle (0, pi), phi: azimuthal angle (0, 2pi)
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)
    return x, y, z
