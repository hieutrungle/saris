import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation

from collections import OrderedDict
from typing import Tuple
import subprocess
import time
import numpy as np
from gymnasium import Env, spaces
from saris.utils import utils
import pickle
import glob
from saris.blender_script import shared_utils
import math
from saris import sigmap
from sionna.channel import (
    cir_to_ofdm_channel,
    subcarrier_frequencies,
    time_lag_discrete_time_channel,
    cir_to_time_channel,
    time_to_ofdm_channel,
)
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

        self.idx = idx
        self.log_string = log_string
        self.seed = seed + idx
        self.np_rng = np.random.default_rng(self.seed)

        policy = tf.keras.mixed_precision.Policy("mixed_bfloat16")
        tf.keras.mixed_precision.set_global_policy(policy)
        tf.config.experimental.set_memory_growth(
            tf.config.experimental.list_physical_devices("GPU")[0], True
        )
        tf.random.set_seed(self.seed)

        self.sionna_config = utils.load_config(sionna_config_file)

        # Modify config for tx orientation
        def sign(num):
            return -1 if num < 0 else 1

        def compute_rot_angle(tile_center: list, pt: list):
            """Compute the rotation angles for the tile.
            return: (r, theta, phi)
                `r`: distance from the tile center to a point
                `theta`: rotation in y-axis
                `phi`: rotation in z-axis
            """
            x = tile_center[0] - pt[0]
            y = tile_center[1] - pt[1]
            z = tile_center[2] - pt[2]

            r = math.sqrt(x**2 + y**2 + z**2)
            theta = math.acos(z / r)  # rotation in y-axis
            phi = sign(y) * math.acos(x / math.sqrt(x**2 + y**2))  # rotation in z-axis

            return (r, theta, phi)

        ris_pos = self.sionna_config["ris_positions"][0]
        tx_pos = self.sionna_config["tx_positions"][0]
        r, theta, phi = compute_rot_angle(ris_pos, tx_pos)
        self.sionna_config["tx_orientations"] = [[phi, math.pi / 2 - theta, 0.0]]

        # Set up logging
        self.current_time = "_" + time.strftime("%d-%m-%Y_%H-%M-%S")

        # Set up action and observation space
        reflector_config = shared_utils.get_reflector_config()
        self.theta_config, self.phi_config, self.num_groups = reflector_config

        # angles = [theta, phi] for each tile
        # theta: azimuth angle, phi: elevation angle

        theta_high = [self.theta_config[2]] * self.num_groups
        phi_high = [self.phi_config[2]] * self.num_groups
        angle_high = np.concatenate([theta_high, phi_high])

        theta_low = [self.theta_config[1]] * self.num_groups
        phi_low = [self.phi_config[1]] * self.num_groups
        angle_low = np.concatenate([theta_low, phi_low])

        self.angle_space = spaces.Box(low=angle_low, high=angle_high, dtype=np.float32)

        # Power average of the equivalent channel
        num_rxs = len(self.sionna_config["rx_positions"])
        self.gain_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_rxs,), dtype=np.float32)

        self.angles = None

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

        self.taken_steps = 0.0
        self.cur_gain = 0.0
        self.next_gain = 0.0

        self.info = {}
        self.eval_mode = eval_mode

    def _get_observation_space(self) -> spaces.Box:
        observation_space = spaces.Dict(
            {
                "angles": self.angle_space,
                "gains": self.gain_space,
            }
        )
        return observation_space

    def _get_action_space(self) -> spaces.Box:
        action_space = spaces.Box(low=-1, high=1, shape=self.angle_space.shape)
        return action_space

    def reset(self, seed: int = None, options: dict = None) -> Tuple[dict, dict]:
        super().reset(seed=seed, options=options)

        self.angles = self.np_rng.uniform(low=self.angle_space.low, high=self.angle_space.high)
        self.angles = np.clip(self.angles, self.angle_space.low, self.angle_space.high)

        self.cur_gain = self._cal_path_gain_dB(eval_mode=self.eval_mode)
        self.next_gain = self.cur_gain

        observation = OrderedDict(
            {
                "angles": np.array(self.angles, dtype=np.float32),
                "gains": np.array(self.cur_gain, dtype=np.float32),
            }
        )
        print(f"observation: {observation}")

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
            "gains": np.asarray(self.next_gain, dtype=np.float32),
        }
        print(f"next_observation: {next_observation}")

        reward = self._cal_reward(self.cur_gain, self.next_gain, self.taken_steps)

        step_info = {
            "path_gain": self.cur_gain,
            "next_path_gain": self.next_gain,
            "reward": reward,
        }

        return next_observation, reward, terminated, truncated, step_info

    def _cal_reward(
        self, cur_gains: np.ndarray, next_gains: np.ndarray, time_taken: float
    ) -> float:

        # multiplication in linear is addition in dB
        threshold = -90.0  # dB
        avg_fairness = np.mean(cur_gains - threshold)

        # total gain
        cur_gain_linear = utils.dB2linear(cur_gains - threshold)
        avg_gain = utils.linear2dB(np.mean(cur_gain_linear))  # dB

        gain_diff = 0.25 * np.mean(next_gains - cur_gains)
        cost_time = -0.1 * time_taken

        reward = 0.3 * avg_fairness + 0.7 * avg_gain + gain_diff + cost_time

        return float(reward)

    def _cal_path_gain_dB(self, eval_mode: bool = False) -> np.ndarray[float]:

        self._prepare_geometry()
        # path gain shape: [num_rx]
        path_gains = self._cal_path_gain_sionna(eval_mode=eval_mode)
        path_gain_dBs = utils.linear2dB(path_gains)

        return path_gain_dBs

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
            "--index",
            str(self.taken_steps),
        ]
        bl_output_txt = os.path.join(tmp_dir, "bl_outputs.txt")
        # subprocess.run(blender_cmd, check=True)
        subprocess.run(blender_cmd, check=True, stdout=open(bl_output_txt, "w"))
        # try:
        #     subprocess.run(blender_cmd, check=True, stdout=open(bl_output_txt, "w"))
        # except subprocess.CalledProcessError as e:
        #     raise Exception(f"Error running Blender command: {e}")

    def _cal_path_gain_sionna(self, eval_mode: bool = False) -> float:

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

        bandwidth = 20e6
        if not eval_mode:
            paths = sig_cmap.compute_paths()
            cir = paths.cir()
            # a: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, num_time_steps], tf.complex
            a, tau = cir
            (l_min, l_max) = time_lag_discrete_time_channel(bandwidth)
            # [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1], tf.complex
            h_time = cir_to_time_channel(bandwidth, a, tau, l_min, l_max)
            h_time = h_time[0]
            # [num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps]
            h_time_sum_power = tf.reduce_sum(tf.abs(h_time) ** 2, axis=-1)
            # [num_rx, num_rx_ant]
            h_time_avg_power = tf.reduce_mean(h_time_sum_power, axis=(1, 2, 3, 4))
            # h_time_avg_power shape: [num_rx]
            path_gain = h_time_avg_power.numpy()
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
