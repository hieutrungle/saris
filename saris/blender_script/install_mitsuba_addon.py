import addon_utils
import os
import bpy
import subprocess
import glob
import os, sys, inspect
import pickle

# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(
    os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0])
)
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

import bl_utils, bl_parser


def install_addons(args):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    sigmap_dir = os.path.dirname(os.path.dirname(current_dir))
    research_dir = os.path.dirname(sigmap_dir)
    scratch_dir = os.path.dirname(research_dir)
    blender_dir = os.path.join(scratch_dir, "blender")
    addon_dir = os.path.join(blender_dir, "addons")

    # Define path to your downloaded script
    path_to_script_dir = addon_dir

    # Define a list of the files in this folder, i.e. directory. The method listdir() will return this list from our folder of downloaded scripts.
    file_list = sorted(os.listdir(path_to_script_dir))

    # Further specificy that of this list of files, you only want the ones with the .zip extension.
    script_list = [item for item in file_list if item.endswith(".zip")]

    # Specify the file path of the individual scripts (their names joined with the location of your downloaded scripts folder) then use wm.addon_install() to install them.
    for file in file_list:
        path_to_file = os.path.join(path_to_script_dir, file)
        bpy.ops.preferences.addon_install(
            overwrite=True,
            target="DEFAULT",
            filepath=path_to_file,
            filter_folder=True,
            filter_python=False,
            filter_glob="*.py;*.zip",
        )

    # install mitsuba using blender python using subprocess
    blender_version = "3.3"
    # blender_folder = glob.glob(f"{blender_dir}/blender-{blender_version}*")[0]
    blender_app_folder = args.blender_app.split("/")[:-1]
    blender_app_folder = "/".join(blender_app_folder)
    python_exec = os.path.join(blender_app_folder, blender_version, "python", "bin")
    python_exec = glob.glob(os.path.join(python_exec, "python3*"))[0]
    # print(f"python_exec: {python_exec}")
    subprocess.run([python_exec, "-m", "ensurepip"])
    subprocess.run([python_exec, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.run([python_exec, "-m", "pip", "install", "mitsuba==3.4.1"])
    subprocess.run([python_exec, "-m", "pip", "install", "--upgrade", "PyYAML"])

    # Specify which add-ons you want enabled. For example, Crowd Render, Pie Menu Editor, etc. Use the script's python module.
    enableTheseAddons = ["mitsuba-blender"]

    # Use addon_enable() to enable them.
    for string in enableTheseAddons:
        name = enableTheseAddons
        addon_utils.enable(string)


def main():
    args = create_argparser().parse_args()
    install_addons(args)


def create_argparser() -> bl_parser.ArgumentParserForBlender:
    """Parses command line arguments."""
    parser = bl_parser.ArgumentParserForBlender()
    parser.add_argument("--blender_app", type=str, required=True)
    return parser


if __name__ == "__main__":
    main()
