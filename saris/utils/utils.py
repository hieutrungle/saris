import os
import shutil
import re
import argparse
import glob
from typing import Dict, List, Union, Tuple
import yaml
import json
import numpy as np


def mkdir_not_exists(folder_dir: str) -> None:
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)


def mkdir_with_replacement(folder_dir: str) -> None:
    if os.path.exists(folder_dir):
        shutil.rmtree(folder_dir)
    os.makedirs(folder_dir)


def create_filename(dir: str, filename: str) -> str:
    """Create a filename in the given directory. If the filename already exists, append a number to the filename."""
    mkdir_not_exists(dir)
    filename = os.path.join(dir, filename)
    tmp_filename = filename
    i = 0
    while os.path.exists(tmp_filename):
        i += 1
        ext = filename.rpartition(".")[2]
        name = filename.rpartition(".")[0]
        name = re.sub(r"\d+$", "", name)
        while name[-1] == "_":
            name = name[:-1]
        tmp_filename = name + f"_{i:05d}." + ext
    filename = tmp_filename
    return filename


# Sorting
def tryint(s: str) -> int:
    try:
        return int(s)
    except:
        return s


# Sorting
def alphanum_key(s: str) -> List[Union[str, int]]:
    """Turn a string into a list of string and number chunks.
    "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split("([0-9]+)", s)]


# Sorting
def sort_nicely(l: List[str]):
    """Sort the given list in the way that humans expect."""
    l.sort(key=alphanum_key)


# Logging
def log_args(args: argparse.Namespace) -> None:
    """Logs arguments to the console."""
    print(f"{'*'*23} ARGS BEGIN {'*'*23}")
    message = ""
    for k, v in args.__dict__.items():
        if isinstance(v, str):
            message += f"{k} = '{v}'\n"
        else:
            message += f"{k} = {v}\n"
    print(f"{message}")
    print(f"{'*'*24} ARGS END {'*'*24}\n")


def log_config(config: Dict[str, Union[str, float, bool]]) -> None:
    """Logs configuration to the console."""
    print(f"{'*'*23} CONFIG BEGIN {'*'*23}")
    message = ""
    for k, v in config.items():
        if isinstance(v, str):
            message += f"{k} = '{v}'\n"
        else:
            message += f"{k} = {v}\n"
    print(f"{message}")
    print(f"{'*'*24} CONFIG END {'*'*24}\n")


# Input and output folders
def get_input_folders(
    args: argparse.Namespace, config: Dict[str, Union[str, float, bool]]
) -> Tuple[str]:
    # Scene directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    asset_dir = os.path.join(parent_dir, "assets")
    mkdir_not_exists(asset_dir)
    blender_scene_dir = os.path.join(asset_dir, "blender")
    mkdir_not_exists(blender_scene_dir)

    cm_scene_folders = glob.glob(
        os.path.join(
            blender_scene_dir,
            f"{config.scene_name}",
            f"ceiling_idx_{args.index}*",
        )
    )
    cm_scene_folders = sorted(
        cm_scene_folders, key=lambda x: float(re.findall("(\d+)", x)[0])
    )
    sort_nicely(cm_scene_folders)

    viz_scene_folders = glob.glob(
        os.path.join(
            blender_scene_dir,
            f"{config.scene_name}",
            f"idx_{args.index}*",
        )
    )
    viz_scene_folders = sorted(
        viz_scene_folders, key=lambda x: float(re.findall("(\d+)", x)[0])
    )
    sort_nicely(viz_scene_folders)

    return (cm_scene_folders, viz_scene_folders)


def get_source_dir() -> str:
    current_dir = os.path.dirname(os.path.realpath(__file__))
    source_dir = os.path.dirname(os.path.dirname(current_dir))
    return source_dir


def get_asset_dir() -> str:
    source_dir = get_source_dir()
    assets_dir = os.path.join(source_dir, "assets")
    return assets_dir


def get_image_dir(config: Dict[str, Union[str, float, bool]]) -> str:
    assets_dir = get_asset_dir()
    img_dir = os.path.join(assets_dir, "images", f"{config.scene_name}")
    mkdir_not_exists(img_dir)

    return img_dir


def get_video_dir(config: Dict[str, Union[str, float, bool]]) -> str:
    assets_dir = get_asset_dir()
    video_dir = os.path.join(assets_dir, "videos")
    mkdir_not_exists(video_dir)
    return video_dir


def get_results_dir(config: Dict[str, Union[str, float, bool]]) -> str:
    assets_dir = get_asset_dir()
    results_dir = os.path.join(assets_dir, "results")
    mkdir_not_exists(results_dir)
    return results_dir


def dict_to_csv(d: Dict[str, Union[str, float, bool]]) -> str:
    """Converts a dictionary to a csv string."""
    csv = ""
    for k, v in d.items():
        csv += f"{k},{v}\n"
    return csv


def load_yaml_file(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def write_yaml_file(file_path: str, data: dict) -> None:
    tmp_file = file_path.split(".")[0] + "_tmp.yaml"
    with open(tmp_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    os.rename(tmp_file, file_path)


class NpEncoder(json.JSONEncoder):
    # json format for saving numpy array
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_tmp_dir(source_dir: str) -> str:
    tmp_dir = os.path.join(source_dir, "tmp")
    mkdir_not_exists(tmp_dir)
    return tmp_dir


def get_assets_dir(source_dir: str) -> str:
    assets_dir = os.path.join(source_dir, "assets")
    mkdir_not_exists(assets_dir)
    return assets_dir


def get_os_dir(name):
    os_dir = os.getenv(name)
    if os_dir is None:
        raise Exception(f"{name} environment variable is not set.")
    mkdir_not_exists(os_dir)
    return os_dir


# Read and write files
def read_first_line(file_path: str) -> str:
    with open(file_path, "rb") as f:
        first_line = f.readline().decode()
    return first_line


def read_last_line(file_path: str) -> str:
    with open(file_path, "rb") as f:
        try:  # catch OSError in case of a one line file
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b"\n":
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        last_line = f.readline().decode()
    return last_line


def read_n_to_last_line(filename, n=1) -> str:
    """Returns the nth before last line of a file (n=1 gives last line)"""
    num_newlines = 0
    with open(filename, "rb") as f:
        try:
            f.seek(-2, os.SEEK_END)
            while num_newlines < n:
                f.seek(-2, os.SEEK_CUR)
                if f.read(1) == b"\n":
                    num_newlines += 1
        except OSError:
            f.seek(0)
        n_to_last_line = f.readline().decode()

    return n_to_last_line


# Conversion
def linear2dB(x: float) -> float:
    return float(10 * np.log10(x))


def dB2linear(x: float) -> float:
    return float(10 ** (x / 10))


def cartesian2spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def spherical2cartesian(
    r: float, theta: float, phi: float
) -> Tuple[float, float, float]:
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def add_batch_dimension(data: Union[np.ndarray, dict]):
    if isinstance(data, dict):
        return {k: add_batch_dimension(v) for k, v in data.items()}
    else:
        return data[None, ...]
