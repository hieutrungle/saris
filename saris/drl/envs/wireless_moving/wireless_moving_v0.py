import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation

from typing import Tuple, Optional
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


class WirelessMovingV0(Env):

    def __init__(
        self,
        idx: int,
        sionna_config_file: str,
        log_string: str = "WirelessMovingV0",
        eval_mode: bool = False,
        seed: int = 0,
        **kwargs,
    ):
        super(WirelessMovingV0, self).__init__()

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

        ris_pos = self.sionna_config["ris_positions"][0]
        tx_pos = self.sionna_config["tx_positions"][0]
        r, theta, phi = compute_rot_angle(tx_pos, ris_pos)
        self.sionna_config["tx_orientations"] = [[theta, math.pi / 2 - phi, 0.0]]

        # Set up logging
        self.current_time = "_" + time.strftime("%d-%m-%Y_%H-%M-%S")

        # Set up action and observation space
        reflector_config = shared_utils.get_reflector_config()

        self.theta_config = reflector_config[0]
        self.phi_config = reflector_config[1]
        self.num_groups = reflector_config[2]
        self.num_elements_per_group = reflector_config[3]

        # angles = [theta, phi] for each tile
        # theta: azimuth angle, phi: elevation angle
        init_theta = self.theta_config[0]
        init_phi = self.phi_config[0]
        init_per_group = [init_theta] + [init_phi] * self.num_elements_per_group
        self.init_angles = np.concatenate([init_per_group] * self.num_groups)

        # angles space
        theta_high = self.theta_config[2]
        phi_high = self.phi_config[2]
        per_group_high = [theta_high] + [phi_high] * self.num_elements_per_group
        angle_high = np.concatenate([per_group_high] * self.num_groups)
        theta_low = self.theta_config[1]
        phi_low = self.phi_config[1]
        per_group_low = [theta_low] + [phi_low] * self.num_elements_per_group
        angle_low = np.concatenate([per_group_low] * self.num_groups)
        self.angle_space = spaces.Box(low=angle_low, high=angle_high, dtype=np.float32)

        # position space
        rx_positions = np.array(self.sionna_config["rx_positions"]).flatten()
        ris_positions = np.array(self.sionna_config["ris_positions"]).flatten()
        self.positions = np.concatenate([rx_positions, ris_positions], dtype=np.float32)
        self.position_space = spaces.Box(
            low=-100.0, high=100.0, shape=(len(self.positions),), dtype=np.float32
        )

        # focal vecs space for action space
        self.init_focal_vecs = np.asarray([10.0, init_theta, init_phi] * self.num_groups)
        r_high = 35.0
        focal_vec_high = np.asarray([r_high, theta_high, phi_high] * self.num_groups)
        r_low = 5.0
        focal_vec_low = np.asarray([r_low, theta_low, phi_low] * self.num_groups)
        self.focal_vec_space = spaces.Box(low=focal_vec_low, high=focal_vec_high, dtype=np.float32)

        # Action is a focal_vec [delta_r, delta_theta, _delta_phi] for each group
        # spherical_focal_vecs = [r, theta, phi] for each group
        self.spherical_focal_vecs = None
        self.angles = None

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

        self.taken_steps = 0.0
        self.cur_gain = 0.0
        self.next_gain = 0.0
        self.ep_step = 0

        self.info = {}
        self.eval_mode = eval_mode
        self.default_positions = copy.deepcopy(self.positions)
        self.default_sionna_config = copy.deepcopy(self.sionna_config)

    def _get_observation_space(self) -> spaces.Box:
        observation_space = spaces.Dict(
            {
                "angles": self.angle_space,
                "positions": self.position_space,
            }
        )
        return observation_space

    def _get_action_space(self) -> spaces.Box:
        # each group has 3 elements: 1 phi, 1 theta, and 1 r
        action_space_shape = tuple((3 * self.num_groups,))
        action_space = spaces.Box(low=-1.0, high=1.0, shape=action_space_shape)
        return action_space

    def reset(self, seed: int = None, options: dict = None) -> Tuple[dict, dict]:
        super().reset(seed=seed, options=options)

        self.ep_step = 0
        self.positions = copy.deepcopy(self.default_positions)
        self.sionna_config = copy.deepcopy(self.default_sionna_config)

        # noise to spherical_focal_vecs
        noise = self.np_rng.normal(loc=0.0, scale=0.05, size=self.init_focal_vecs.shape)
        self.spherical_focal_vecs = self.init_focal_vecs
        self.spherical_focal_vecs += noise
        self.spherical_focal_vecs = np.clip(
            self.spherical_focal_vecs, self.focal_vec_space.low, self.focal_vec_space.high
        )
        self.angles = self._blender_step(self.spherical_focal_vecs)
        self.angles = np.clip(self.angles, self.angle_space.low, self.angle_space.high)

        self.cur_gain = self._cal_path_gain_dB(eval_mode=self.eval_mode)
        self.next_gain = self.cur_gain

        observation = {
            "angles": np.array(self.angles, dtype=np.float32),
            "positions": self.positions,
        }

        self.taken_steps = 0.0

        return observation, {}

    def step(self, action: np.ndarray, **kwargs) -> Tuple[dict, float, bool, bool, dict]:

        self.taken_steps += 1.0
        self.cur_gain = self.next_gain

        # rx position change
        if self.ep_step % 25 == 0 and self.ep_step != 0:
            self.positions[3] = self.positions[3] - 2.0
            rx_positions = self.sionna_config["rx_positions"]
            rx_positions[1][0] = self.positions[3]
            self.sionna_config["rx_positions"] = rx_positions
        self.ep_step += 1

        # action: [num_groups * 3]: num_groups * [phi, theta, r]
        self.spherical_focal_vecs = self.spherical_focal_vecs + action
        self.spherical_focal_vecs = np.clip(
            self.spherical_focal_vecs, self.focal_vec_space.low, self.focal_vec_space.high
        )

        self.angles = self._blender_step(self.spherical_focal_vecs)
        self.angles = np.clip(self.angles, self.angle_space.low, self.angle_space.high)

        truncated = False
        terminated = False
        self.next_gain = 0.0
        self.next_gain = self._cal_path_gain_dB(eval_mode=self.eval_mode)
        next_observation = {
            "angles": np.asarray(self.angles, dtype=np.float32),
            "positions": self.positions,
        }

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

        total_gain = np.sum(utils.dB2linear(cur_gains))
        total_gain = utils.linear2dB(total_gain)  # dB

        gain_diff = np.sum(next_gains - cur_gains)
        cost_time = time_taken

        lower_ = -100.0
        upper_ = -80.0

        reward = total_gain + 0.4 * gain_diff - 0.05 * cost_time
        reward = (reward - lower_) / (upper_ - lower_)

        return float(reward)

    def _blender_step(self, spherical_focal_vecs: np.ndarray[float]) -> np.ndarray[float]:
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
            pickle.dump(spherical_focal_vecs, f)

        blender_script = os.path.join(source_dir, "saris", "blender_script", "bl_drl.py")
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

        with open(data_path, "rb") as f:
            angles = pickle.load(f)
        angles = np.asarray(angles, dtype=np.float32)
        return angles

    def _cal_path_gain_dB(self, eval_mode: bool = False) -> np.ndarray[float]:

        # self._prepare_geometry()
        # path gain shape: [num_rx]
        path_gains = self._cal_path_gain_sionna(eval_mode=eval_mode)
        path_gain_dBs = utils.linear2dB(path_gains)

        return path_gain_dBs

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


def compute_rot_angle(pt1: list, pt2: list) -> Tuple[float, float, float]:
    """Compute the rotation angles for vector pt1 to pt2."""
    x = pt2[0] - pt1[0]
    y = pt2[1] - pt1[1]
    z = pt2[2] - pt1[2]

    return cartesian2spherical(x, y, z)


def cartesian2spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    r = math.sqrt(x**2 + y**2 + z**2)
    theta = math.atan2(y, x)
    phi = math.acos(z / r)
    return r, theta, phi


def spherical2cartesian(r: float, theta: float, phi: float) -> Tuple[float, float, float]:
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    return x, y, z
