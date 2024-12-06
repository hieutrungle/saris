# Self Adjustable Reconfigurable Intelligent Surfaces (SARIS)

[![Documentation Status](https://readthedocs.org/projects/saris/badge/?version=latest)](https://saris.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/saris.svg)](https://badge.fury.io/py/saris)
[![Build Status](https://travis-ci.com/hieutrungle/saris.svg?branch=main)](https://travis-ci.com/hieutrungle/saris)
[![codecov](https://codecov.io/gh/hieutrungle/saris/branch/main/graph/badge.svg?token=QZQZQZQZQZ)](https://codecov.io/gh/hieutrungle/saris)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<!-- .. image:: https://img.shields.io/pypi/v/saris.svg
        :target: https://pypi.python.org/pypi/saris

.. image:: https://img.shields.io/travis/hieutrungle/saris.svg
        :target: https://travis-ci.com/hieutrungle/saris

.. image:: https://readthedocs.org/projects/saris/badge/?version=latest
        :target: https://saris.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status -->

Self Adjustable Reconfigurable Intelligent Surfaces (SARIS) / Self Adjustable Metallic Surfaces (SAMS)

## Installation

### Docker

```markdown
## TODO
[ ] Docker Build
[ ] Docker Run
```

<!-- ```bash
docker build -t pytorch-saris . -f Dockerfile
docker run --rm --runtime=nvidia --gpus all -it saris
``` -->

### Manual

#### OS

Ensure you have `Ubuntu 22.04` installed.

The home directory is structured as follows:

```bash
./home
├── .bashrc
├── .config
├── research
```

#### Virtual Environment

Ensure you have `Python 3.10-3.11` installed.

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Blender

get the blender from Google Drive

```bash
cd home
pip install gdown
gdown --folder https://drive.google.com/drive/u/1/folders/1sHqz5PRKtLQI0aEcByzKMyNwIOSG557l
```

There are two zip files, one for the blender/saved models and one for the blender config. Unzip them and put them in the root directory `home` of the system.

```bash
cd blender_gdown
unzip blender.zip
mv blender home
unzip blender_config.zip
mv blender home/.cache
cd ..
rm -rf blender_gdown
```

Now, the home directory should look like this:

```bash
./home
├── .bashrc
├── .cache
│   └── blender
├── blender
│   ├── addons
│   ├── blender-3.3.14-linux-x64
│   └── models
├── research
```

#### Dependencies

Ensure you have NVIDIA drivers installed.

```bash
cd home/research
git clone -b torch-dev-angles https://github.com/hieutrungle/saris
cd saris
pip install -e .
pip install torch==2.5.1
pip install -r requirements.txt
```

## Usage

To run DRL SAC for narrow L-shaped hallway:

```bash
cd home/research/saris
bash run_wireless_sac_static.sh
```

To run DRL SAC for narrow L-shaped hallway with moving users:

```bash
cd home/research/saris
bash run_wireless_sac_moving.sh
```

## Post-Processing

After running the DRL SAC, the directory should look like this:

```bash
./home/research/saris
├── configs
├── default_scenes
├── docs
├── notebooks
├── local_assets
│   ├── blender
│   ├── images
│   │   ├── Parallel_env_0
│   │   │   ├── env_name_idx_00000.png
│   │   │   ├── env_name_idx_00001.png
│   │   │   ├── env_name_idx_00002.png
│   │   ├── Parallel_env_1
│   │   │   ├── env_name_idx_00000.png
│   │   │   ├── env_name_idx_00001.png
│   └── logs
│       ├── Run_name_idx_0
│       │   ├── all_path_gain.png
│       │   ├── all_path_gain.npy
│       │   ├── all_rewards.npy
│       │   ├── path_gain.png
│       │   ├── rewards.png
│       │   └── train_config.yaml
│       ├── Run_name_idx_1
├── saris
...
```

### Video Generation

Remember to create a directory for the videos: `mkdir {OUTPUT_VIDEO_DIR}`

```bash
ffmpeg -framerate 5 -i {PATH_TO_IMAGES}_%05d.png -r 30 -pix_fmt yuv420p {OUTPUT_VIDEO_PATH}.mp4
```

Example:

```bash
ffmpeg -framerate 5 -i ./tmp_long_short_mean_adjusted_local_assets/images/SAC_Mean_Adjusted__orin__wireless-sigmap-v0__fecc18e6_03-12-2024_17-09-34_0/hallway_L_0_%05d.png -r 30 -pix_fmt yuv420p ./tmp_long_short_mean_adjusted_local_assets/videos/SAC_Mean_Adjusted__orin__wireless-sigmap-v0__fecc18e6_03-12-2024_17-09-34_0.mp4
```

## Completed Tasks

```markdown
[x] DRL for narrow L-shaped hallway
[x] Stable training using Soft Actor Critic (SAC) with only RX CSI
```

## Uncompleted Tasks

<!-- ```markdown
[ ] DRL for wide L-shaped hallway
[ ] DRL for wide U-shaped hallway
[ ] DRL for wide room
[ ] DRL for wide room with multiple users
[ ] DRL for wide room with multiple users and UE position change -->

```markdown
[ ] Increase the length and size of the neural neuwork after mixing observation and action for critic network
[ ] DRL for wide room with multiple users
[ ] DRL for wide room with multiple users and UE position change
```

## Citing

There is a series of papers that are being written to describe the SARIS project. Please cite the following paper if you use this package in your research:

First proof of concept paper:

```bibtex
@INPROCEEDINGS{10757704,
  author={Le, Hieu and Bedir, Oguz and Ibrahim, Mostafa and Tao, Jian and Ekin, Sabit},
  booktitle={2024 IEEE 100th Vehicular Technology Conference (VTC2024-Fall)}, 
  title={Guiding Wireless Signals with Arrays of Metallic Linear Fresnel Reflectors: A Low-cost, Frequency-versatile, and Practical Approach}, 
  year={2024},
  volume={},
  number={},
  pages={1-7},
  keywords={Wireless communication;Vehicular and wireless technologies;Solid modeling;Three-dimensional displays;Systematics;Reconfigurable intelligent surfaces;Ray tracing;Reflection;Resource management;Gain;reconfigurable intelligent surfaces (RIS);specular reflections;path gain;received signal strength (RSS);ray tracing;coverage map},
  doi={10.1109/VTC2024-Fall63153.2024.10757704}}
```

Second DRL paper: being written

## Acknowledgement

Texas Wireless Lab - PI - Dr. Sabit Ekin

Texas A&M University

This package was created with Cookiecutter and the `briggySmalls/cookiecutter-pypackage`_ project template.
