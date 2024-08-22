import bpy
from mathutils import Vector
import math
import os
import os, sys, inspect

# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(
    os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0])
)
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

import bl_utils, bl_parser


def export_beamfocusing_hallway(args, config):

    devices = []
    devices_names = []
    for k, v in bpy.data.collections.items():
        if "Reflector" in k:
            devices_names.append(k)
            devices.append(v.objects)

    for tile_tuple in zip(*devices):
        global_bbox_centers = []
        for i, tile in enumerate(tile_tuple):
            if i == 0:
                global_bbox_centers.append(config.tx_position)
            global_bbox_centers.append(bl_utils.get_center_bbox(tile))
            if i == len(tile_tuple) - 1:
                global_bbox_centers.append(config.rx_position)

        for i in range(len(tile_tuple)):
            r, theta, phi = bl_utils.compute_rot_angle_3pts(
                global_bbox_centers[i],
                global_bbox_centers[i + 1],
                global_bbox_centers[i + 2],
            )
            tile_tuple[i].rotation_euler = [0, theta, phi]
            tile_tuple[i].scale = [0.1, 0.1, 0.01]

    # Saving to mitsuba format for Sionna
    print(
        f"\nsaving with index: {args.index},  rx_pos: {config.rx_position} and tx_pos: {config.tx_position}"
    )

    # Save files without ceiling
    folder_dir = os.path.join(
        args.output_dir,
        f"{config.scene_name}",
        f"idx_{args.index}_rx_pos_{config.rx_position}",
    )
    bl_utils.mkdir_with_replacement(folder_dir)
    bl_utils.save_mitsuba_xml(
        folder_dir, config.mitsuba_filename, [*devices_names, "Wall", "Floor"]
    )

    # Save files with ceiling
    folder_dir = os.path.join(
        args.output_dir,
        f"{config.scene_name}",
        f"ceiling_idx_{args.index}_rx_pos_{config.rx_position}",
    )
    bl_utils.mkdir_with_replacement(folder_dir)
    bl_utils.save_mitsuba_xml(
        folder_dir,
        config.mitsuba_filename,
        [*devices_names, "Wall", "Floor", "Ceiling"],
    )


def export_no_reflector_scene(args, config):
    # Saving to mitsuba format for Sionna
    print(
        f"\nsaving with index: {args.index},  rx_pos: {config.rx_position} and tx_pos: {config.tx_position}"
    )

    # Save files without ceiling
    folder_dir = os.path.join(
        args.output_dir,
        f"{config.scene_name}",
        f"idx_{args.index}_rx_pos_{config.rx_position}",
    )
    bl_utils.mkdir_with_replacement(folder_dir)
    bl_utils.save_mitsuba_xml(folder_dir, config.mitsuba_filename, ["Wall", "Floor"])

    # Save files with ceiling
    folder_dir = os.path.join(
        args.output_dir,
        f"{config.scene_name}",
        f"ceiling_idx_{args.index}_rx_pos_{config.rx_position}",
    )
    bl_utils.mkdir_with_replacement(folder_dir)
    bl_utils.save_mitsuba_xml(
        folder_dir,
        config.mitsuba_filename,
        ["Wall", "Floor", "Ceiling"],
    )


def export_simple_reflector_scene(args, config):
    devices = []
    devices_names = []
    for k, v in bpy.data.collections.items():
        if "Reflector" in k:
            devices_names.append(k)
            devices.append(v.objects)

    rs, thetas, phis = [], [], []
    for idx, tile_tuple in enumerate(zip(*devices)):
        if idx == len(devices[0]) // 2:
            global_bbox_centers = []
            for i, tile in enumerate(tile_tuple):
                if i == 0:
                    global_bbox_centers.append(config.tx_position)
                global_bbox_centers.append(bl_utils.get_center_bbox(tile))
                if i == len(tile_tuple) - 1:
                    global_bbox_centers.append(config.rx_position)
            for i in range(len(tile_tuple)):
                r, theta, phi = bl_utils.compute_rot_angle_3pts(
                    global_bbox_centers[i],
                    global_bbox_centers[i + 1],
                    global_bbox_centers[i + 2],
                )
                rs.append(r)
                thetas.append(theta)
                phis.append(phi)
            break

    for idx, tile_tuple in enumerate(zip(*devices)):
        for i in range(len(tile_tuple)):
            tile_tuple[i].rotation_euler = [0, math.radians(90), math.radians(-45)]
            tile_tuple[i].scale = [0.1, 0.1, 0.01]

    # Saving to mitsuba format for Sionna
    print(
        f"\nsaving with index: {args.index},  rx_pos: {config.rx_position} and tx_pos: {config.tx_position}"
    )

    # Save files without ceiling
    folder_dir = os.path.join(
        args.output_dir,
        f"{config.scene_name}",
        f"idx_{args.index}_rx_pos_{config.rx_position}",
    )
    bl_utils.mkdir_with_replacement(folder_dir)
    bl_utils.save_mitsuba_xml(
        folder_dir, config.mitsuba_filename, [*devices_names, "Wall", "Floor"]
    )

    # Save files with ceiling
    folder_dir = os.path.join(
        args.output_dir,
        f"{config.scene_name}",
        f"ceiling_idx_{args.index}_rx_pos_{config.rx_position}",
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

    if "beamfocusing" in config.scene_name:
        export_beamfocusing_hallway(args, config)
    elif "simple_reflector" in config.scene_name:
        export_simple_reflector_scene(args, config)
    elif "no_reflector" in config.scene_name:
        export_no_reflector_scene(args, config)
    else:
        raise ValueError(f"Unknown scene name: {config.scene_name}")


def create_argparser() -> bl_parser.ArgumentParserForBlender:
    """Parses command line arguments."""
    parser = bl_parser.ArgumentParserForBlender()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("--index", type=str)
    return parser


if __name__ == "__main__":
    main()
