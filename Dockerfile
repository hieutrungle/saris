# FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base-all

# RUN echo 'APT::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker
# RUN echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker

# RUN --mount=type=cache,target=/var/lib/apt/lists \
#     --mount=type=cache,target=/var/cache,sharing=locked \
#     apt-get update \
#     && apt-get upgrade --assume-yes \
#     && apt-get install --assume-yes --no-install-recommends python3-pip build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget python-is-python3 software-properties-common git

# RUN groupadd -r saris --gid=1280 && \
#     useradd -r -g saris --uid=1280 --create-home --shell /bin/bash saris
# # switch user from 'root' to saris and also to the home directory that it owns 
# USER root
# # FROM base-all AS builder

# # RUN pip install poetry==1.8.3

# # ENV POETRY_NO_INTERACTION=1 \
# #     POETRY_VIRTUALENVS_IN_PROJECT=1 \
# #     POETRY_VIRTUALENVS_CREATE=1 \
# #     POETRY_CACHE_DIR=/tmp/poetry_cache

# # WORKDIR /app

# # COPY pyproject.toml poetry.lock ./
# # RUN touch README.md

# # RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --no-root

# FROM base-all AS runtime

# RUN mkdir /home/saris/research
# WORKDIR /home/saris/research
# RUN echo "cd ~/Desktop/Java\ Files" >> ~/.bashrc
# RUN pip install gdown

# # clone from github
# RUN git clone -b torch-dev-angles https://github.com/hieutrungle/saris
# RUN cd ./saris && pip install -e .
# RUN python -m pip install --upgrade pip
# RUN pip3 install --upgrade --pre torch==2.6.0.dev20241118+cu124 --index-url https://download.pytorch.org/whl/nightly/cu124
# RUN pip3 install -r ./saris/requirements.txt

# # TODO: add Blender installation
# RUN --mount=type=cache,target=/var/lib/apt/lists \
#     --mount=type=cache,target=/var/cache,sharing=locked \
#     apt-get install unzip
# RUN gdown --folder https://drive.google.com/drive/u/2/folders/1sHqz5PRKtLQI0aEcByzKMyNwIOSG557l
# RUN unzip blender_gdown/blender.zip
# RUN mv blender ../

# RUN unzip blender_gdown/blender_config.zip

# # wandb login key
# COPY wandb_api_key.txt ./
# RUN mv wandb_api_key.txt ./saris/tmp_wandb_api_key.txt

# # USER saris
# # ENTRYPOINT ["bash", "run_drl_L_hallway_calql.sh"]

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS bpy-builder

ARG DEBIAN_FRONTEND=noninteractive
ARG BLENDER_VERSION="v3.3.14"
ARG BLENDER_LIBRARY_VERSION="3.3"

# Update and upgrade packages
RUN apt-get update && apt-get upgrade -y

# Basic requirements (see https://wiki.blender.org/wiki/Building_Blender/Linux/Ubuntu#Quick_Setup)
RUN apt-get install -y \
    build-essential \
    git \
    subversion \
    cmake \
    libx11-dev \
    libxxf86vm-dev \
    libxcursor-dev \
    libxi-dev \
    libxrandr-dev \
    libxinerama-dev \
    libglew-dev 

# # Wayland requirements (see https://wiki.blender.org/wiki/Building_Blender/Linux/Ubuntu#Quick_Setup)
# RUN apt-get install -y \
#     libwayland-dev \
#     wayland-protocols \
#     libegl-dev \
#     libxkbcommon-dev \
#     libdbus-1-dev \
#     linux-libc-dev

# Python requirements
RUN apt-get install -y \
    python3-dev \
    python3-distutils \
    python3-pip \
    python3-apt

# Clone a shallow copy of the blender sources
RUN mkdir -p /opt/blender-git/
WORKDIR /opt/blender-git/
RUN git clone https://github.com/blender/blender.git -c advice.detachedHead=false --depth 1 --branch ${BLENDER_VERSION}
WORKDIR /opt/blender-git/blender
RUN git checkout -b my-branch

## Checkout submodules
RUN git submodule foreach git checkout ${BLENDER_VERSION}

# Download a copy of the blender libraries
RUN mkdir -p /opt/blender-git/lib
WORKDIR /opt/blender-git/lib
RUN svn export https://svn.blender.org/svnroot/bf-blender/tags/blender-${BLENDER_LIBRARY_VERSION}-release/lib/linux_centos7_x86_64/

# Set build flags
## Python
ARG WITH_PYTHON_INSTALL=OFF
ARG WITH_AUDASPACE=OFF
ARG WITH_PYTHON_MODULE=ON

## GPU
ARG WITH_CYCLES_DEVICE_CUDA=ON
ARG WITH_CYCLES_CUDA_BINARIES=ON

# Compile blender
WORKDIR /opt/blender-git/blender
RUN make update
RUN make bpy

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update 
RUN apt-get install -y build-essential 
RUN apt-get install -y zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget software-properties-common unzip
RUN apt-get install -y python3-dev python3-opencv python3-pip
RUN apt-get install --assume-yes --no-install-recommends git

# RUN mkdir /home
WORKDIR /home/research
RUN git clone -b torch-dev-angles https://github.com/hieutrungle/saris

WORKDIR /home/research/saris

# Upgrade pip
RUN python3 -m pip install --upgrade pip
RUN pip install gdown

# Install any python packages you need
RUN pip3 install -e .
RUN pip3 install torch==2.5.1
RUN python3 -m pip install -r requirements.txt

# Set the working directory
WORKDIR /home

# Blender
RUN apt-get install -y apt-transport-https \
    ca-certificates \
    git \
    subversion \
    cmake \
    python3 \
    libx11-dev \
    libxxf86vm-dev \
    libxcursor-dev \
    libxi-dev \
    libxrandr-dev \
    libxinerama-dev \
    libglew-dev
RUN apt-get install -y nano vim
RUN pip install -U bpy
COPY --from=bpy-builder /opt/blender-git/lib/linux_centos7_x86_64/python/lib/python3.9/site-packages \
    /opt/bpy/lib/python3.9/site-packages
COPY --from=bpy-builder /opt/blender-git/lib/linux_centos7_x86_64/python/lib/python3.9/site-packages \
    /opt/bpy/lib/python3.9/site-packages
RUN apt-get install -y \
    libgomp1 \
    libxrender1

RUN gdown --folder https://drive.google.com/drive/u/2/folders/1sHqz5PRKtLQI0aEcByzKMyNwIOSG557l
RUN unzip blender_gdown/blender.zip
RUN unzip blender_gdown/blender_config.zip -d ./blender_gdown/
RUN mv blender_gdown/blender ~/.config


RUN echo "export DISPLAY=:0" >> ~/.bashrc

# Set the entrypoint
# ENTRYPOINT [ "python3" ]