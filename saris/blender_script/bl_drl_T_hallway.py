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


# TODO: change for T hallway
def export_drl_hallway_hex(args):

    # unit: degrees
    theta_config, phi_config, num_groups, num_elements_per_group = (
        shared_utils.get_reflector_config()
    )
    theta_range = (theta_config[1], theta_config[2])
    phi_range = (phi_config[1], phi_config[2])

    devices_names = []
    object_dict = {f"Group{i:02d}": [] for i in range(1, num_groups + 1)}

    for k, v in bpy.data.collections.items():
        if "Reflector" in k:
            devices_names.append(k)
            sorted_objects = sorted(v.objects, key=lambda x: x.name)
            for obj in sorted_objects:
                concat_name = obj.name.strip().split(".")
                group_name = concat_name[0]
                object_dict[group_name].append(obj)

    with open(args.input_path, "rb") as f:
        # data: Tuple[np.ndarray[float], np.ndarray[float]]
        spherical_focal_vecs = pickle.load(f)
    spherical_focal_vecs = spherical_focal_vecs.reshape(num_groups, 3)

    # Angle container
    angles = [0.0 for _ in range(num_groups * (num_elements_per_group + 1))]

    for i, (group_name, objects) in enumerate(object_dict.items()):
        mid_tile = objects[num_elements_per_group // 2]
        r_mid, theta_mid, phi_mid = spherical_focal_vecs[i]
        focal_vec = bl_utils.spherical2cartesian(r_mid, theta_mid, phi_mid)
        focal_pt = bl_utils.get_center_bbox(mid_tile) + Vector(focal_vec)
        theta_mid = shared_utils.constraint_angle(theta_mid, theta_range)
        angles[i * (num_elements_per_group + 1)] = theta_mid

        for j, obj in enumerate(objects):
            center = bl_utils.get_center_bbox(obj)
            r, theta, phi = bl_utils.compute_rot_angle(center, focal_pt)
            theta = shared_utils.constraint_angle(theta, theta_range)
            phi = shared_utils.constraint_angle(phi, phi_range)
            obj.rotation_euler = [0, phi, theta]
            angles[i * (num_elements_per_group + 1) + j + 1] = phi

    result_path = args.input_path
    with open(result_path, "wb") as f:
        pickle.dump(angles, f)

    # Save files without ceiling
    folder_dir = os.path.join(args.output_dir, f"idx")
    bl_utils.mkdir_with_replacement(folder_dir)
    bl_utils.save_mitsuba_xml(folder_dir, "hallway", [*devices_names, "Wall", "Floor", "Obstacles"])

    # Save files with ceiling
    folder_dir = os.path.join(args.output_dir, f"ceiling_idx")
    bl_utils.mkdir_with_replacement(folder_dir)
    bl_utils.save_mitsuba_xml(
        folder_dir,
        "hallway",
        [*devices_names, "Wall", "Floor", "Ceiling", "Obstacles"],
    )


def tmp_export_drl_hallway_hex(args):

    # unit: degrees
    theta_config, phi_config, num_groups, num_elements_per_group = (
        shared_utils.get_reflector_config()
    )
    theta_range = (math.radians(theta_config[1]), math.radians(theta_config[2]))
    phi_range = (math.radians((phi_config[1])), math.radians(phi_config[2]))

    devices_names = []
    object_dict = {f"Group{i:02d}": [] for i in range(1, num_groups + 1)}

    for k, v in bpy.data.collections.items():
        if "Reflector" in k:
            devices_names.append(k)
            sorted_objects = sorted(v.objects, key=lambda x: x.name)
            for obj in sorted_objects:
                concat_name = obj.name.strip().split(".")
                group_name = concat_name[0]
                object_dict[group_name].append(obj)

    with open(args.input_path, "rb") as f:
        # angles: Tuple[List[float], List[float]]
        angles = pickle.load(f)
        (thetas, phis) = angles

    for i, (group_name, objects) in enumerate(object_dict.items()):
        theta = thetas[i]
        phi = phis[i]
        theta = shared_utils.constraint_angle(theta, theta_range)
        phi = shared_utils.constraint_angle(phi, phi_range)

        for j, obj in enumerate(objects):
            object_dict[group_name][j].rotation_euler = [0, theta, phi]

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
