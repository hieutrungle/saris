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

        self.info = {}
        self.use_cmap = False

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
        delta_focal_points = self.np_rng.uniform(-2, 2, size=self._focal_points.shape)
        self._focal_points = pos + delta_focal_points
        self._focal_points = np.asarray(self._focal_points, dtype=np.float32)

        return self._get_observation(), self.info

    def step(
        self, action: np.ndarray, **kwargs
    ) -> Tuple[dict, float, bool, bool, dict]:

        action = np.reshape(action, self._focal_points.shape)
        self._focal_points = self._focal_points + action
        next_observation = self._get_observation()

        truncated = False
        reward = self._calculate_reward(use_cmap=self.use_cmap)
        terminated = False

        return next_observation, reward, terminated, truncated, self.info

    def _calculate_reward(self, use_cmap: bool) -> float:

        # Generate geometry file
        self._run_blender()

        # Run Sionna to get reward
        reward = self._run_sionna(use_cmap=use_cmap)

        return reward

    def _run_blender(self):

        # Blender export
        blender_app = utils.get_os_dir("BLENDER_APP")
        blender_dir = utils.get_os_dir("BLENDER_DIR")
        source_dir = utils.get_os_dir("SOURCE_DIR")
        assets_dir = utils.get_os_dir("ASSETS_DIR")
        blender_output_dir = os.path.join(assets_dir, "blender")
        tmp_dir = utils.get_os_dir("TMP_DIR")

        focal_name = "focal_pts-" + self.log_string + ".pkl"
        focal_path = os.path.join(tmp_dir, focal_name)
        with open(focal_path, "wb") as f:
            pickle.dump(self._focal_points, f)

        blender_script = os.path.join(
            source_dir, "saris", "blender_script", "bl_drl.py"
        )
        scene_name = utils.load_yaml_file(self.sionna_config_file)["scene_name"]

        blender_command = [
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
            subprocess.run(blender_command, check=True)
            # subprocess.run(blender_command, check=True, stdout=open(bl_output_txt, "w"))
        except subprocess.CalledProcessError as e:
            raise Exception(f"Error running Blender command: {e}")

    def _run_sionna(self, use_cmap: bool = False) -> float:
        path_gain = self._cal_path_gain(use_cmap=use_cmap)
        return path_gain

    def _cal_path_gain(self, use_cmap: bool = False) -> float:

        assets_dir = utils.get_os_dir("ASSETS_DIR")

        # Sionna simulation
        scene_name = utils.load_yaml_file(self.sionna_config_file)["scene_name"]
        for file in glob.glob(os.path.join(assets_dir, "blender", scene_name)):
            print(file)

        exit(0)
        compute_scene_path = (
            subprocess.check_output(
                f'find {os.path.join(assets_dir, "blender", scene_name)} -type d -name "ceiling_idx*"',
                shell=True,
            )
            .decode()
            .strip()
        )
        compute_scene_path = (
            subprocess.check_output(
                f'find "{compute_scene_path}" -type f -name "*.xml"', shell=True
            )
            .decode()
            .strip()
        )
        viz_scene_path = (
            subprocess.check_output(
                f'find {os.path.join(assets_dir, "blender", scene_name)} -type d -name "idx*"',
                shell=True,
            )
            .decode()
            .strip()
        )
        viz_scene_path = (
            subprocess.check_output(
                f'find "{viz_scene_path}" -type f -name "*.xml"', shell=True
            )
            .decode()
            .strip()
        )

        sigmap_dir = utils.get_os_dir("SIGMAP_DIR")

        img_dir = os.path.join(
            assets_dir, "images", self.log_string + self.current_time
        )
        mitsuba_filename = utils.load_yaml_file(self.sionna_config_file)[
            "mitsuba_filename"
        ]
        render_filename = utils.create_filename(
            img_dir, f"{mitsuba_filename}_00000.png"
        )

        siona_script = os.path.join(
            sigmap_dir, "sigmap", "sub_tasks", "calc_pathgain.py"
        )
        sionna_command = [
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
            "--seed",
            str(self.seed),
        ]
        if use_cmap:
            sionna_command.append("--use_cmap")
        tmp_dir = utils.get_tmp_dir()
        sionna_output_txt = os.path.join(tmp_dir, "sionna_outputs.txt")
        try:
            subprocess.run(
                sionna_command, check=True, stdout=open(sionna_output_txt, "a")
            )
        except subprocess.CalledProcessError as e:
            raise Exception(f"Error running Sionna command: {e}")
        finally:
            pass

        results_name = "path_gain-" + self.log_string + self.current_time + ".txt"
        results_file = os.path.join(tmp_dir, results_name)
        with open(results_file, "r") as f:
            results_dict = json.load(f)
            path_gain = results_dict["path_gain"]

        path_gain = float(path_gain)
        path_gain_dB = utils.linear2dB(path_gain)
        plt.clf()
        plt.close("all")
        return path_gain_dB
