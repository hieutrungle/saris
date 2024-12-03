# SARIS

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

Self Adjustable Reconfigurable Intelligent Surfaces (SARIS)

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
pip install gdown
gdown --folder https://drive.google.com/drive/u/1/folders/1sHqz5PRKtLQI0aEcByzKMyNwIOSG557l
```

There are two zip files, one for the blender/saved models and one for the blender config. Unzip them and put them in the root directory `home` of the system.

```bash
unzip blender.zip
mv blender home
unzip blender_config.zip
mv blender home/.cache
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

## Acknowledgement

Texas Wireless Lab - PI - Dr. Sabit Ekin

Texas A&M University

This package was created with Cookiecutter and the `briggySmalls/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`briggySmalls/cookiecutter-pypackage`: https://github.com/briggySmalls/cookiecutter-pypackage
