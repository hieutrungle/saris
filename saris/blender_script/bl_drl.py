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


def export_drl_hallway_hex(args):

    # unit: degrees
    theta_config, phi_config = shared_utils.get_reflector_config()
    theta_range = (theta_config[1], theta_config[2])
    phi_range = (phi_config[1], phi_config[2])

    devices_names = []
    object_dict = {}
    for k, v in bpy.data.collections.items():
        if "Reflector" in k:
            devices_names.append(k)
            sorted_objects = sorted(v.objects, key=lambda x: x.name)
            for object in sorted_objects:
                concat_name = object.name.strip().split(".")
                group_name = concat_name[0]

                tmp = object_dict.get(group_name, [])
                tmp.append(object)
                object_dict[group_name] = tmp

    with open(args.input_path, "rb") as f:
        # angles: Tuple[List[float], List[float]]
        angles = pickle.load(f)
        (thetas, phis) = angles

    for i, (group_name, objects) in enumerate(object_dict.items()):
        theta = thetas[i]
        phi = phis[i]
        theta = shared_utils.constraint_angle(theta, theta_range)
        phi = shared_utils.constraint_angle(phi, phi_range)

        for obj in objects:
            obj.rotation_euler = [0, theta, phi]

    # Save files without ceiling
    folder_dir = os.path.join(args.output_dir, f"idx")
    bl_utils.mkdir_with_replacement(folder_dir)
    bl_utils.save_mitsuba_xml(folder_dir, "hallway", [*devices_names, "Wall", "Floor"])

    # Save files with ceiling
    folder_dir = os.path.join(args.output_dir, f"ceiling_idx")
    bl_utils.mkdir_with_replacement(folder_dir)
    bl_utils.save_mitsuba_xml(
        folder_dir,
        "hallway",
        [*devices_names, "Wall", "Floor", "Ceiling"],
    )


def export_drl_hallway_angle(args):

    lead_follow_dict, init_angles, angle_deltas = shared_utils.set_up_reflector()
    theta = init_angles[0]
    theta_low = math.radians(theta + angle_deltas[0])
    theta_high = math.radians(theta + angle_deltas[1])
    delta_theta = (theta_low, theta_high)

    phi = init_angles[1]
    phi_low = math.radians(phi + angle_deltas[0])
    phi_high = math.radians(phi + angle_deltas[1])
    delta_phi = (phi_low, phi_high)

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
        theta = shared_utils.constraint_angle(theta, delta_theta)
        phi = shared_utils.constraint_angle(phi, delta_phi)

        devices[0][lead_idx].rotation_euler = [0, theta, phi]
        devices[0][lead_idx].scale = [0.1, 0.1, 0.01]

        for follow_idx in follow_idxs:
            devices[0][follow_idx].rotation_euler = [0, theta, phi]
            devices[0][follow_idx].scale = [0.1, 0.1, 0.01]

    # Save files without ceiling
    folder_dir = os.path.join(args.output_dir, f"idx")
    bl_utils.mkdir_with_replacement(folder_dir)
    bl_utils.save_mitsuba_xml(folder_dir, "hallway", [*devices_names, "Wall", "Floor"])

    # Save files with ceiling
    folder_dir = os.path.join(args.output_dir, f"ceiling_idx")
    bl_utils.mkdir_with_replacement(folder_dir)
    bl_utils.save_mitsuba_xml(
        folder_dir,
        "hallway",
        [*devices_names, "Wall", "Floor", "Ceiling"],
    )


def main():
    args = create_argparser().parse_args()
    # export_drl_hallway_angle(args)
    export_drl_hallway_hex(args)


def create_argparser() -> bl_parser.ArgumentParserForBlender:
    """Parses command line arguments."""
    parser = bl_parser.ArgumentParserForBlender()
    parser.add_argument("--input_path", "-i", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("--index", type=str)
    return parser


if __name__ == "__main__":
    main()
