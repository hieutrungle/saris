import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation

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
import tensorflow as tf
from saris.sigmap import signal_cmap
from sionna.channel import (
    cir_to_ofdm_channel,
    subcarrier_frequencies,
    OFDMChannel,
    ApplyOFDMChannel,
    CIRDataset,
    time_lag_discrete_time_channel,
    cir_to_time_channel,
    time_to_ofdm_channel,
)


class WirelessEnvV0(Env):

    def __init__(
        self,
        sionna_config_file: str,
        log_string: str = "WirelessEnvV0",
        seed: int = 0,
        **kwargs,
    ):
        super(WirelessEnvV0, self).__init__()
        self.sionna_config_file = sionna_config_file
        self.log_string = log_string
        self.seed = seed
        self.np_rng = np.random.default_rng(self.seed)
        tf.random.set_seed(self.seed)
        self.current_time = "_" + time.strftime("%d-%m-%Y_%H-%M-%S")

        # Set up action and observation space
        reflector_config = shared_utils.set_up_reflector()
        self.lead_follow_dict, self.init_angles, self.angle_deltas = reflector_config
        self.num_lead_tiles = len(self.lead_follow_dict.keys())
        # angles = [theta, phi] for each tile
        # theta: azimuth angle, phi: elevation angle
        init_theta, init_phi = self.init_angles
        min_delta, max_delta = self.angle_deltas
        self.lead_theta_space = spaces.Box(
            low=init_theta + min_delta,
            high=init_theta + max_delta,
            shape=(self.num_lead_tiles,),
            dtype=np.float32,
        )
        self.lead_phi_space = spaces.Box(
            low=init_phi + min_delta,
            high=init_phi + max_delta,
            shape=(self.num_lead_tiles,),
            dtype=np.float32,
        )

        # Power average of the equivalent channel
        self.channel_power_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )

        self.thetas = np.zeros(self.num_lead_tiles)
        self.phis = np.zeros(self.num_lead_tiles)
        self.channel_power = np.zeros(1)

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

        self.observation_dim = self._get_observation_dim()
        self.action_dim = self._get_action_dim()

        self.taken_steps = 0.0
        self.cur_gain = 0.0
        self.next_gain = 0.0

        self.info = {}
        self.use_cmap = False
        self.location_known = False
        self.eval = False

    def _get_observation_space(self) -> spaces.Box:
        observation_space = spaces.Dict(
            {
                "thetas": self.lead_theta_space,
                "phis": self.lead_phi_space,
                "channel_power": self.channel_power_space,
            }
        )
        return observation_space

    def _get_observation_dim(self) -> int:
        observation_dim = 0
        for key, value in self.observation_space.items():
            observation_dim += np.prod(value.shape)
        return observation_dim

    def _get_action_space(self) -> spaces.Box:
        action_space = spaces.Dict(
            {
                "delta_thetas": spaces.Box(
                    low=-1.0, high=1.0, shape=(self.num_lead_tiles,), dtype=np.float32
                ),
                "delta_phis": spaces.Box(
                    low=-1.0, high=1.0, shape=(self.num_lead_tiles,), dtype=np.float32
                ),
            }
        )
        return action_space

    def _get_action_dim(self) -> int:
        action_dim = 0
        for key, value in self.action_space.items():
            action_dim += np.prod(value.shape)
        return action_dim

    def _get_observation(self) -> dict:
        observation = np.concatenate([self.thetas, self.phis, self.channel_power])
        observation = {
            "thetas": self.thetas,
            "phis": self.phis,
            "channel_power": self.channel_power,
        }
        return observation

    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)
        if seed is not None:
            self.seed = seed
            self.np_rng = np.random.default_rng(self.seed)

        self.thetas = self.np_rng.uniform(
            low=self.lead_theta_space.low,
            high=self.lead_theta_space.high,
            size=(self.num_lead_tiles,),
        )
        self.thetas = np.asarray(self.thetas, dtype=np.float32)
        self.phis = self.np_rng.uniform(
            low=self.lead_phi_space.low,
            high=self.lead_phi_space.high,
            size=(self.num_lead_tiles,),
        )
        self.phis = np.asarray(self.phis, dtype=np.float32)
        self.channel_power = self._cal_path_gain(use_cmap=self.use_cmap)
        self.channel_power = np.array([self.channel_power])

        self.cur_gain = self.channel_power[0]
        self.next_gain = self.cur_gain

        self.info = {}
        self.taken_steps = 0.0

        return self._get_observation(), self.info

    def step(
        self, action: np.ndarray, **kwargs
    ) -> Tuple[dict, float, bool, bool, dict]:

        self.taken_steps += 1.0
        self.cur_gain = self.next_gain

        truncated = False
        terminated = False
        # TODO: constraint fot angles
        self.thetas = np.clip(
            self.thetas + action["delta_thetas"],
            self.lead_theta_space.low,
            self.lead_theta_space.high,
        )
        self.phis = np.clip(
            self.phis + action["delta_phis"],
            self.lead_phi_space.low,
            self.lead_phi_space.high,
        )
        self.channel_power = np.array([self.cur_gain])

        next_observation = self._get_observation()

        self.next_gain = self._cal_path_gain(use_cmap=self.use_cmap)

        reward = self._cal_reward(self.cur_gain, self.next_gain, self.taken_steps)

        step_info = {
            "path_gain_dB": self.cur_gain,
            "next_path_gain_dB": self.next_gain,
            "reward": reward,
        }
        self.info.update(step_info)

        return next_observation, reward, terminated, truncated, self.info

    def _cal_reward(
        self, cur_gain: float, next_gain: float, time_taken: float
    ) -> float:
        # threshold_gain = -90
        # scaled_cur_gain = 1.25 * (cur_gain - threshold_gain)
        gain_diff = 2.0 * (next_gain - cur_gain)
        cost_time = -0.02 * time_taken
        reward = gain_diff + cost_time
        # reward = scaled_cur_gain + gain_diff + cost_time
        return reward

    def _cal_path_gain(self, use_cmap: bool = False) -> float:

        start_time = time.time()
        self._prepare_geometry()
        path_gain_dB = self._cal_path_gain_sionna(use_cmap=use_cmap)

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
        blender_output_dir = os.path.join(assets_dir, "blender")
        tmp_dir = utils.get_os_dir("TMP_DIR")

        # ! TODO: Need to convert from degrees to radians
        theta_phi = (self.thetas, self.phis)
        angle_path = os.path.join(
            tmp_dir, f"angles-{self.log_string}-{self.current_time}.pkl"
        )
        with open(angle_path, "wb") as f:
            pickle.dump(theta_phi, f)

        blender_script = os.path.join(
            source_dir, "saris", "blender_script", "bl_drl.py"
        )
        scene_name = utils.load_yaml_file(self.sionna_config_file)["scene_name"]

        blender_cmd = [
            blender_app,
            "-b",
            os.path.join(blender_dir, "models", f"{scene_name}.blend"),
            "--python",
            blender_script,
            "--",
            "-cfg",
            self.sionna_config_file,
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

    def _cal_path_gain_sionna(self, use_cmap: bool = False) -> float:

        assets_dir = utils.get_os_dir("ASSETS_DIR")

        # Set up geometry paths for Sionna script
        scene_name = utils.load_yaml_file(self.sionna_config_file)["scene_name"]
        blender_output_dir = os.path.join(assets_dir, "blender", scene_name)
        compute_scene_dir = os.path.join(blender_output_dir, "ceiling_idx")
        compute_scene_path = glob.glob(os.path.join(compute_scene_dir, "*.xml"))[0]
        viz_scene_dir = os.path.join(blender_output_dir, "idx")
        viz_scene_path = glob.glob(os.path.join(viz_scene_dir, "*.xml"))[0]

        # Path for outputing iamges if we want to visualize the coverage map
        source_dir = utils.get_os_dir("SOURCE_DIR")
        img_dir = os.path.join(
            assets_dir, "images", self.log_string + self.current_time
        )
        mitsuba_filename = utils.load_yaml_file(self.sionna_config_file)[
            "mitsuba_filename"
        ]
        render_filename = utils.create_filename(
            img_dir, f"{mitsuba_filename}_00000.png"
        )

        # Set up path for saving results
        # tmp_dir = utils.get_os_dir("TMP_DIR")
        # results_name = f"path-gain-{self.log_string}-{self.current_time}.pkl"
        # results_file = os.path.join(tmp_dir, results_name)

        config = utils.load_config(self.sionna_config_file)
        sig_cmap = signal_cmap.SignalCoverageMap(
            config, compute_scene_path, viz_scene_path
        )

        bandwidth = 20e6
        if not use_cmap:
            paths = sig_cmap.compute_paths()
            cir = paths.cir()
            a, tau = cir
            (l_min, l_max) = time_lag_discrete_time_channel(bandwidth)
            h_time = cir_to_time_channel(bandwidth, a, tau, l_min, l_max)
            h_time_avg_power = tf.reduce_mean(
                tf.reduce_sum(tf.abs(h_time) ** 2, axis=-1)
            ).numpy()
            path_gain = h_time_avg_power
        else:
            coverage_map = sig_cmap.compute_cmap()
            path_gain = sig_cmap.get_path_gain(coverage_map)
            sig_cmap.render_to_file(coverage_map, filename=render_filename)

        print(f"Path gain: {path_gain}")
        return path_gain

        # # Run Sionna script
        # siona_script = os.path.join(
        #     source_dir, "saris", "sub_tasks", "calc_pathgain.py"
        # )

        # sionna_cmd = [
        #     "python",
        #     siona_script,
        #     "-cfg",
        #     self.sionna_config_file,
        #     "--compute_scene_path",
        #     compute_scene_path,
        #     "--viz_scene_path",
        #     viz_scene_path,
        #     "--saved_path",
        #     render_filename,
        #     "--results_path",
        #     results_file,
        #     "--seed",
        #     str(self.seed),
        # ]
        # if use_cmap:
        #     sionna_cmd.append("--use_cmap")
        # sionna_output_txt = os.path.join(tmp_dir, "sionna_outputs.txt")
        # try:
        #     subprocess.run(sionna_cmd, check=True, stdout=open(sionna_output_txt, "a"))
        # except subprocess.CalledProcessError as e:
        #     raise Exception(f"Error running Sionna command: {e}")
        # finally:
        #     pass

        # with open(results_file, "rb") as f:
        #     results_dict = pickle.load(f)
        #     path_gain = float(results_dict["path_gain"])
        # path_gain_dB = utils.linear2dB(path_gain)
        # return path_gain_dB
