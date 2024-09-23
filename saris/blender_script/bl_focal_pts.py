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


def export_drl_hallway_focal_pts(args):

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
        focal_pts = pickle.load(f)  # focal_pts: [num_devices, 6]

    print(f"focal_pts: {focal_pts}")
    print(f"delta_theta: {delta_theta}")
    print(f"delta_phi: {delta_phi}")
    for device in devices:
        focal_pts = focal_pts.reshape(-1, 3)
        pt1 = focal_pts[0]
        pt2 = focal_pts[1]
        for tile in device:
            r, theta, phi = bl_utils.compute_rot_angle_3pts(
                pt1,
                bl_utils.get_center_bbox(tile),
                pt2,
            )
            print(f"r: {r}, theta: {theta}, phi: {phi}")
            theta = shared_utils.constraint_angle(theta, delta_theta)
            phi = shared_utils.constraint_angle(phi, delta_phi)
            print(f"After constraint: r: {r}, theta: {theta}, phi: {phi}")
            tile.rotation_euler = [0, theta, phi]
            tile.scale = [0.1, 0.1, 0.01]

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
    export_drl_hallway_focal_pts(args)


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
