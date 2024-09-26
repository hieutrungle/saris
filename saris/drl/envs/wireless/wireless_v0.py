import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation

# import orderdict
from collections import OrderedDict
from typing import Tuple
import re
import subprocess
import time
import numpy as np
from gymnasium import Env, spaces
from saris.utils import utils
import pickle
import glob
import json
from saris.blender_script import shared_utils

from saris.sigmap import signal_cmap
from sionna.channel import (
    cir_to_ofdm_channel,
    subcarrier_frequencies,
    time_lag_discrete_time_channel,
    cir_to_time_channel,
    time_to_ofdm_channel,
)

import importlib
import tensorflow as tf


class WirelessEnvV0(Env):

    def __init__(
        self,
        idx: int,
        sionna_config_file: str,
        log_string: str = "WirelessEnvV0",
        eval_mode: bool = False,
        seed: int = 0,
        **kwargs,
    ):
        super(WirelessEnvV0, self).__init__()

        policy = tf.keras.mixed_precision.Policy("mixed_bfloat16")
        tf.keras.mixed_precision.set_global_policy(policy)

        self.idx = idx
        self.log_string = log_string
        self.seed = seed + idx
        self.np_rng = np.random.default_rng(self.seed)

        tf.config.experimental.set_memory_growth(
            tf.config.experimental.list_physical_devices("GPU")[0], True
        )
        tf.random.set_seed(self.seed)

        self.sionna_config = utils.load_config(sionna_config_file)
        self.current_time = "_" + time.strftime("%d-%m-%Y_%H-%M-%S")

        # Set up action and observation space
        reflector_config = shared_utils.set_up_reflector()
        self.lead_follow_dict, self.init_angles, self.angle_deltas = reflector_config
        self.num_lead_tiles = len(self.lead_follow_dict.keys())
        # angles = [theta, phi] for each tile
        # theta: azimuth angle, phi: elevation angle
        init_theta, init_phi = self.init_angles
        min_delta, max_delta = self.angle_deltas

        theta_high = [init_theta + max_delta] * self.num_lead_tiles
        phi_high = [init_phi + max_delta] * self.num_lead_tiles
        angle_high = np.concatenate([theta_high, phi_high])

        theta_low = [init_theta + min_delta] * self.num_lead_tiles
        phi_low = [init_phi + min_delta] * self.num_lead_tiles
        angle_low = np.concatenate([theta_low, phi_low])

        self.angle_space = spaces.Box(low=angle_low, high=angle_high, dtype=np.float32)

        # Power average of the equivalent channel
        self.gain_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        self.angles = None

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

        self.taken_steps = 0.0
        self.cur_gain = 0.0
        self.next_gain = 0.0

        self.info = {}
        self.eval_mode = eval_mode

        rx_pos = self.sionna_config["rx_position"]
        rx_pos_xs = np.arange(-11.0, -14.0, -0.5)
        rx_pos_ys = np.arange(-2.5, -4.6, -0.5)
        rx_pos_zs = np.array([rx_pos[2]])

        print(f"env_idx: {self.idx}")
        self.rx_poss = []
        for rx_pos_x in rx_pos_xs:
            rx_pos_ys = rx_pos_ys[::-1]
            for rx_pos_y in rx_pos_ys:
                for rx_pos_z in rx_pos_zs:
                    self.rx_poss.append([rx_pos_x, rx_pos_y, rx_pos_z])
                    print(f"rx_pos: {rx_pos_x}, {rx_pos_y}, {rx_pos_z}")
        print(f"Number of rx positions: {len(self.rx_poss)}")
        self.rx_idx = 0

        # for i in range(100):
        #     print(f"i: {i} - rx_idx: {self.rx_idx} - rx_pos: {self.rx_poss[self.rx_idx]}")
        #     self.rx_idx = self.rx_idx + 1
        #     if self.rx_idx % len(self.rx_poss) == 0:
        #         self.rx_idx = 0
        #         self.rx_poss = self.rx_poss[::-1]
        #         print()

    def _get_observation_space(self) -> spaces.Box:
        observation_space = spaces.Dict(
            {
                "angles": self.angle_space,
                "gain": self.gain_space,
            }
        )
        return observation_space

    def _get_action_space(self) -> spaces.Box:
        action_space = spaces.Box(low=-1, high=1, shape=self.angle_space.shape)
        return action_space

    def reset(self, seed: int = None, options: dict = None) -> Tuple[dict, dict]:
        super().reset(seed=seed, options=options)

        self.sionna_config["rx_position"] = self.rx_poss[self.rx_idx]
        self.rx_idx = self.rx_idx + 1
        if self.rx_idx % len(self.rx_poss) == 0:
            self.rx_idx = 0
            self.rx_poss = self.rx_poss[::-1]

        self.angles = self.np_rng.uniform(low=self.angle_space.low, high=self.angle_space.high)
        self.angles = np.clip(self.angles, self.angle_space.low, self.angle_space.high)

        self.cur_gain = self._cal_path_gain_dB(eval_mode=self.eval_mode)
        self.next_gain = self.cur_gain

        observation = OrderedDict(
            {
                "angles": np.array(self.angles, dtype=np.float32),
                "gain": np.array([self.cur_gain], dtype=np.float32),
            }
        )

        self.taken_steps = 0.0

        return observation, {}

    def step(self, action: np.ndarray, **kwargs) -> Tuple[dict, float, bool, bool, dict]:

        self.taken_steps += 1.0
        self.cur_gain = self.next_gain

        self.angles = np.clip(self.angles + action, self.angle_space.low, self.angle_space.high)

        truncated = False
        terminated = False

        self.next_gain = 0.0
        self.next_gain = self._cal_path_gain_dB(eval_mode=self.eval_mode)
        next_observation = {
            "angles": np.asarray(self.angles, dtype=np.float32),
            "gain": np.asarray([self.next_gain], dtype=np.float32),
        }

        reward = self._cal_reward(self.cur_gain, self.next_gain, self.taken_steps)

        step_info = {
            "path_gain_dB": self.cur_gain,
            "next_path_gain_dB": self.next_gain,
            "reward": reward,
        }

        return next_observation, reward, terminated, truncated, step_info

    def _cal_reward(self, cur_gain: float, next_gain: float, time_taken: float) -> float:
        threshold = -90.0  # dB
        scaled_gain = 0.8 * (cur_gain - threshold)
        gain_diff = 0.1 * (next_gain - cur_gain)
        cost_time = -0.1 * time_taken
        reward = scaled_gain + gain_diff + cost_time
        return reward

    def _cal_path_gain_dB(self, eval_mode: bool = False) -> float:

        self._prepare_geometry()
        path_gain = self._cal_path_gain_sionna(eval_mode=eval_mode)
        path_gain_dB = utils.linear2dB(path_gain)

        return path_gain_dB

    def _prepare_geometry(self) -> None:
        """
        Prepare geometry for Sionna script using Blender.
        This function saves current states of thetas and phis to a pickle file and runs the Blender script.
        """
        # Blender export
        blender_app = utils.get_os_dir("BLENDER_APP")
        blender_dir = utils.get_os_dir("BLENDER_DIR")
        source_dir = utils.get_os_dir("SOURCE_DIR")
        assets_dir = utils.get_os_dir("ASSETS_DIR")
        tmp_dir = utils.get_os_dir("TMP_DIR")
        scene_name = f"{self.sionna_config['scene_name']}_{self.idx}"
        blender_output_dir = os.path.join(assets_dir, "blender", scene_name)
        # // ! TODO: Need to convert from degrees to radians
        angles = np.deg2rad(self.angles)
        angles = (angles[: len(angles) // 2], angles[len(angles) // 2 :])
        angle_path = os.path.join(
            tmp_dir, f"angles-{self.log_string}-{self.current_time}-{self.idx}.pkl"
        )
        with open(angle_path, "wb") as f:
            pickle.dump(angles, f)

        blender_script = os.path.join(source_dir, "saris", "blender_script", "bl_drl.py")

        blender_cmd = [
            blender_app,
            "-b",
            os.path.join(blender_dir, "models", f"{scene_name}.blend"),
            "--python",
            blender_script,
            "--",
            "-i",
            angle_path,
            "-o",
            blender_output_dir,
        ]
        try:
            bl_output_txt = os.path.join(tmp_dir, "bl_outputs.txt")
            subprocess.run(blender_cmd, check=True, stdout=open(bl_output_txt, "w"))
        except subprocess.CalledProcessError as e:
            raise Exception(f"Error running Blender command: {e}")

    def _cal_path_gain_sionna(self, eval_mode: bool = False) -> float:

        # Set up geometry paths for Sionna script
        assets_dir = utils.get_os_dir("ASSETS_DIR")
        scene_name = f"{self.sionna_config['scene_name']}_{self.idx}"
        blender_output_dir = os.path.join(assets_dir, "blender", scene_name)
        compute_scene_dir = os.path.join(blender_output_dir, "ceiling_idx")
        compute_scene_path = glob.glob(os.path.join(compute_scene_dir, "*.xml"))[0]
        viz_scene_dir = os.path.join(blender_output_dir, "idx")
        viz_scene_path = glob.glob(os.path.join(viz_scene_dir, "*.xml"))[0]

        sig_cmap = signal_cmap.SignalCoverageMap(
            self.sionna_config, compute_scene_path, viz_scene_path
        )

        bandwidth = 20e6
        if not eval_mode:
            paths = sig_cmap.compute_paths()
            cir = paths.cir()
            a, tau = cir
            (l_min, l_max) = time_lag_discrete_time_channel(bandwidth)
            h_time = cir_to_time_channel(bandwidth, a, tau, l_min, l_max)
            h_time_avg_power = tf.reduce_mean(tf.reduce_sum(tf.abs(h_time) ** 2, axis=-1)).numpy()
            path_gain = h_time_avg_power
        else:
            # Path for outputing iamges if we want to visualize the coverage map
            img_dir = os.path.join(
                assets_dir, "images", self.log_string + self.current_time + f"_{self.idx}"
            )
            render_filename = utils.create_filename(img_dir, f"{scene_name}_00000.png")
            coverage_map = sig_cmap.compute_cmap()
            path_gain = sig_cmap.get_path_gain(coverage_map)
            sig_cmap.render_to_file(coverage_map, filename=render_filename)

        return path_gain
