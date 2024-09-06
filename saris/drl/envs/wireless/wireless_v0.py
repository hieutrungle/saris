from typing import Tuple
import os
import re
import subprocess
import time
import numpy as np
from gymnasium import Env, spaces
from saris.utils import utils
import pickle
import glob
import json


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
        self.current_time = "_" + time.strftime("%d-%m-%Y_%H-%M-%S")

        self._rx_position = np.zeros((3,))
        self._tx_position = np.zeros((3,))
        self._focal_points = np.zeros((1, 6))  # 2 focal points for each device

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        self.taken_steps = 0.0
        self.cur_gain = 0.0
        self.next_gain = 0.0

        self.info = {}
        self.use_cmap = False
        self.location_known = False
        self.eval = False

    def _get_observation(self) -> dict:
        observation = np.concatenate(
            [self._rx_position, self._tx_position, self._focal_points.reshape(-1)]
        )
        return observation

    def _get_observation_space(self) -> spaces.Box:
        observation_shape = self._get_observation().shape
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=observation_shape, dtype=np.float32
        )

    def _get_action_space(self) -> spaces.Box:
        action_shape = self._focal_points.reshape(-1).shape
        return spaces.Box(low=-1.0, high=1.0, shape=action_shape, dtype=np.float32)

    def reset(self, seed: int = None, options: dict = None) -> np.ndarray:
        super().reset(seed=seed, options=options)
        if seed is not None:
            self.seed = seed
            self.np_rng = np.random.default_rng(self.seed)
        sionna_config = utils.load_yaml_file(self.sionna_config_file)
        self._rx_position = list(sionna_config["rx_position"])
        self._rx_position = np.asarray(self._rx_position, dtype=np.float32)
        self._tx_position = list(sionna_config["tx_position"])
        self._tx_position = np.asarray(self._tx_position, dtype=np.float32)
        pos = np.concatenate((self._rx_position, self._tx_position))
        focal_point_shape = self._focal_points.shape
        pos = np.tile(pos, (focal_point_shape[0], 1))

        # TODO: get random uniform number from a sphere with radius that is half the distance between RIS and RX; RIS and tx
        focal_shape = self._focal_points.shape
        if self.location_known:
            delta_focal_points = self.np_rng.uniform(-1, 1, focal_shape)
            self._focal_points = pos + delta_focal_points
        else:
            ris_locations = utils.load_yaml_file(self.sionna_config_file)[
                "ris_locations"
            ]
            ris_locations = np.array(ris_locations)
            ris_locations = np.tile(ris_locations, (focal_shape[0], 2))
            if not self.eval:
                delta_focal_points = self.np_rng.uniform(-3, 3, focal_shape)
            else:
                delta_focal_points = np.zeros(focal_shape)

            self._focal_points = ris_locations + delta_focal_points

        self._focal_points = np.asarray(self._focal_points, dtype=np.float32)

        self.cur_gain = self._cal_path_gain(self._focal_points, use_cmap=self.use_cmap)
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
        if np.any(np.abs(self._focal_points) > 100):
            terminated = True

        action = np.reshape(action, self._focal_points.shape)
        self._focal_points = self._focal_points + action
        next_observation = self._get_observation()
        self.next_gain = self._cal_path_gain(self._focal_points, use_cmap=self.use_cmap)

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
        threshold_gain = -90
        scaled_cur_gain = 1.25 * (cur_gain - threshold_gain)
        gain_diff = 2.0 * (next_gain - cur_gain)
        cost_time = -0.02 * time_taken
        reward = scaled_cur_gain + gain_diff + cost_time
        return reward

    def _cal_path_gain(self, focal_points: np.ndarray, use_cmap: bool = False) -> float:

        self._prepare_geometry(focal_points)
        path_gain_dB = self._cal_path_gain_sionna(use_cmap=use_cmap)
        return path_gain_dB

    def _prepare_geometry(self, focal_points: np.ndarray) -> None:

        # Blender export
        blender_app = utils.get_os_dir("BLENDER_APP")
        blender_dir = utils.get_os_dir("BLENDER_DIR")
        source_dir = utils.get_os_dir("SOURCE_DIR")
        assets_dir = utils.get_os_dir("ASSETS_DIR")
        blender_output_dir = os.path.join(assets_dir, "blender")
        tmp_dir = utils.get_os_dir("TMP_DIR")

        focal_name = f"focal_pts-{self.log_string}-{self.current_time}.pkl"
        focal_path = os.path.join(tmp_dir, focal_name)
        with open(focal_path, "wb") as f:
            pickle.dump(focal_points, f)

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
            focal_path,
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
        tmp_dir = utils.get_os_dir("TMP_DIR")
        results_name = f"path-gain-{self.log_string}-{self.current_time}.pkl"
        results_file = os.path.join(tmp_dir, results_name)

        # Run Sionna script
        siona_script = os.path.join(
            source_dir, "saris", "sub_tasks", "calc_pathgain.py"
        )

        sionna_cmd = [
            "python",
            siona_script,
            "-cfg",
            self.sionna_config_file,
            "--compute_scene_path",
            compute_scene_path,
            "--viz_scene_path",
            viz_scene_path,
            "--saved_path",
            render_filename,
            "--results_path",
            results_file,
            "--seed",
            str(self.seed),
        ]
        if use_cmap:
            sionna_cmd.append("--use_cmap")
        sionna_output_txt = os.path.join(tmp_dir, "sionna_outputs.txt")
        try:
            subprocess.run(sionna_cmd, check=True, stdout=open(sionna_output_txt, "a"))
        except subprocess.CalledProcessError as e:
            raise Exception(f"Error running Sionna command: {e}")
        finally:
            pass

        with open(results_file, "rb") as f:
            results_dict = pickle.load(f)
            path_gain = float(results_dict["path_gain"])
        path_gain_dB = utils.linear2dB(path_gain)
        return path_gain_dB
