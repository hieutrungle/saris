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

import bl_utils, bl_parser, shared_utils


def export_drl_hallway_angle(args, config):

    lead_follow_dict, _, _ = shared_utils.set_up_reflector()

    # Each device has multiple tiles
    devices = []
    devices_names = []
    for k, v in bpy.data.collections.items():
        if "Reflector" in k:
            sorted_objects = sorted(v.objects, key=lambda x: x.name)
            devices_names.append(k)
            devices.append(sorted_objects)

    with open(args.input_path, "rb") as f:
        # angles: Tuple[List[float], List[float]]
        angles = pickle.load(f)
        (thetas, phis) = angles

    for i, (lead_idx, follow_idxs) in enumerate(lead_follow_dict.items()):
        theta = thetas[i]
        phi = phis[i]

        devices[0][lead_idx].rotation_euler = [0, theta, phi]
        devices[0][lead_idx].scale = [0.1, 0.1, 0.01]

        for follow_idx in follow_idxs:
            devices[0][follow_idx].rotation_euler = [0, theta, phi]
            devices[0][follow_idx].scale = [0.1, 0.1, 0.01]

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
    export_drl_hallway_angle(args, config)


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
