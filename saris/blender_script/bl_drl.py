import bpy
from mathutils import Vector
import math
import os
import os, sys, inspect
import pickle

# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(
    os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0])
)
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

import bl_utils, bl_parser


def export_drl_hallway_device_state(args, config):
    # // TODO: get device states from replay buffer
    devices = []
    devices_names = []
    for k, v in bpy.data.collections.items():
        if "Reflector" in k:
            devices_names.append(k)
            devices.append(v.objects)

    # Same tmp_file as in wireless.py -> self._cal_reward()
    tmp_dir = os.getenv("TMP_DIR")
    tmp_file = os.path.join(tmp_dir, "device_states.pkl")

    with open(tmp_file, "rb") as f:
        device_states = pickle.load(f)

    tile_tuples = zip(*devices)
    for j, tile_tuple in enumerate(tile_tuples):
        for i, tile in enumerate(tile_tuple):

            tile.rotation_euler = [0, device_states[i][j][0], device_states[i][j][1]]
            tile.scale = [0.1, 0.1, 0.01]

    # Save files without ceiling
    folder_dir = os.path.join(
        args.output_dir,
        f"{config.scene_name}",
        f"idx",
    )
    bl_utils.mkdir_with_replacement(folder_dir)
    bl_utils.save_mitsuba_xml(
        folder_dir, config.mitsuba_filename, [*devices_names, "Wall", "Floor"]
    )

    # Save files with ceiling
    folder_dir = os.path.join(
        args.output_dir,
        f"{config.scene_name}",
        f"ceiling_idx",
    )
    bl_utils.mkdir_with_replacement(folder_dir)
    bl_utils.save_mitsuba_xml(
        folder_dir,
        config.mitsuba_filename,
        [*devices_names, "Wall", "Floor", "Ceiling"],
    )


def export_drl_hallway_focal_pts(args, config):

    # Each device has multiple tiles
    devices = []
    devices_names = []
    for k, v in bpy.data.collections.items():
        if "Reflector" in k:
            devices_names.append(k)
            devices.append(v.objects)

    with open(args.input_path, "rb") as f:
        tuple_focal_pts = pickle.load(f)  # focal_pts: [num_devices, 6]

    for device, focal_pts in zip(devices, tuple_focal_pts):
        focal_pts = focal_pts.reshape(-1, 3)
        pt1 = focal_pts[0]
        pt2 = focal_pts[1]
        for tile in device:
            r, theta, phi = bl_utils.compute_rot_angle_3pts(
                pt1,
                bl_utils.get_center_bbox(tile),
                pt2,
            )
            tile.rotation_euler = [0, theta, phi]
            tile.scale = [0.1, 0.1, 0.01]

    # Save files without ceiling
    folder_dir = os.path.join(
        args.output_dir,
        f"{config.scene_name}",
        f"idx",
    )
    bl_utils.mkdir_with_replacement(folder_dir)
    bl_utils.save_mitsuba_xml(
        folder_dir, config.mitsuba_filename, [*devices_names, "Wall", "Floor"]
    )

    # Save files with ceiling
    folder_dir = os.path.join(
        args.output_dir,
        f"{config.scene_name}",
        f"ceiling_idx",
    )
    bl_utils.mkdir_with_replacement(folder_dir)
    bl_utils.save_mitsuba_xml(
        folder_dir,
        config.mitsuba_filename,
        [*devices_names, "Wall", "Floor", "Ceiling"],
    )


def main():
    args = create_argparser().parse_args()
    config = bl_utils.make_conf(args.config_file)
    export_drl_hallway_focal_pts(args, config)
    # export_drl_hallway(args, config)


def create_argparser() -> bl_parser.ArgumentParserForBlender:
    """Parses command line arguments."""
    parser = bl_parser.ArgumentParserForBlender()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    parser.add_argument("--input_path", "-i", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("--index", type=str)
    return parser


if __name__ == "__main__":
    main()
