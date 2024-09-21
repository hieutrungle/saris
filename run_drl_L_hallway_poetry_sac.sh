#!/usr/bin/env bash

set -o pipefail # Pipe fails when any command in the pipe fails
set -u  # Treat unset variables as an error

handle_error() {
    echo "An error occurred on line $1"
    exit 1
}

trap 'handle_error $LINENO' ERR

# # Source: https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
# # Get the directory of the script (does not solve symlink problem)
# SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# echo "Script directory: $SCRIPT_DIR"

# Get the source path of the script, even if it's called from a symlink
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPT_DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
echo "Source directory: $SCRIPT_DIR"
SOURCE_DIR=$SCRIPT_DIR
ASSETS_DIR=${SOURCE_DIR}/local_assets

# * Change this to your blender directory
RESEARCH_DIR=$(dirname $SOURCE_DIR)
HOME_DIR=$(dirname $RESEARCH_DIR)
BLENDER_DIR=${HOME_DIR}/blender 

echo Blender directory: $BLENDER_DIR
echo Coverage map directory: $SOURCE_DIR
echo -e Assets directory: $ASSETS_DIR '\n'


# Find the blender executable
# for file in ${BLENDER_DIR}/*
# do
#     if [[ "$file" == *"blender-3.3"* ]];then
#         BLENDER_APP=$file/blender
#     fi
# done
BLENDER_APP=${BLENDER_DIR}/blender-3.3.14-linux-x64/blender

# Open a random blender file to install and enable the mitsuba plugin
# mkdir -p ${BLENDER_DIR}/addons
# if [ ! -f ${BLENDER_DIR}/addons/mitsuba*.zip ]; then
#     wget -P ${BLENDER_DIR}/addons https://github.com/mitsuba-renderer/mitsuba-blender/releases/download/v0.3.0/mitsuba-blender.zip 
#     # unzip mitsuba-blender.zip -d ${BLENDER_DIR}/addons
# fi
# ${BLENDER_APP} -b ${BLENDER_DIR}/models/hallway_L_1.blend --python ${SOURCE_DIR}/saris/blender_script/install_mitsuba_addon.py -- --blender_app ${BLENDER_APP}

# get scene_name from CONFIG_FILE
# SCENE_NAME=$(python -c "import yaml; print(yaml.safe_load(open('${CONFIG_FILE}', 'r'))['scene_name'])")

# Create a temporary directory
TMP_DIR=${SOURCE_DIR}/tmp
mkdir -p ${TMP_DIR}

# Configuration files
SIONNA_CONFIG_FILE=${SOURCE_DIR}/configs/sionna_L_hallway_1.yaml

export BLENDER_APP
export BLENDER_DIR
export SOURCE_DIR
export ASSETS_DIR
export SIONNA_CONFIG_FILE
export TMP_DIR
export OPTIX_CACHE_PATH=${TMP_DIR}/optix_cache_1
mkdir -p ${OPTIX_CACHE_PATH}

##############################
# DRL run
##############################
poetry run train_sac --sionna_config_file ${SIONNA_CONFIG_FILE} --seed 10 --verbose

# export OPTIX_CACHE_PATH=${TMP_DIR}/optix_cache_1
# poetry run main --command eval -dcfg ${DRL_CONFIG_FILE} -scfg ${SIONNA_CONFIG_FILE} -v --seed 100