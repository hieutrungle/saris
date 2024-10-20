FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base-all

RUN echo 'APT::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker
RUN echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache,sharing=locked \
    apt-get update \
    && apt-get upgrade --assume-yes \
    && apt-get install --assume-yes --no-install-recommends python3-pip build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget python-is-python3 software-properties-common git

# FROM base-all AS builder

# RUN pip install poetry==1.8.3

# ENV POETRY_NO_INTERACTION=1 \
#     POETRY_VIRTUALENVS_IN_PROJECT=1 \
#     POETRY_VIRTUALENVS_CREATE=1 \
#     POETRY_CACHE_DIR=/tmp/poetry_cache

# WORKDIR /app

# COPY pyproject.toml poetry.lock ./
# RUN touch README.md

# RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --no-root

FROM base-all AS runtime

WORKDIR /research
RUN pip install gdown

# clone from github
RUN git clone -b torch-dev-angles https://github.com/hieutrungle/saris
RUN cd ./saris && pip install -e .
RUN python -m pip install --upgrade pip
RUN pip3 install --upgrade --pre torch==2.6.0.dev20241020+cu124 --index-url https://download.pytorch.org/whl/nightly/cu124
RUN pip3 install -r ./saris/requirements.txt
RUN pip3 install sionna==0.19 tensorflow[and-cuda]
RUN pip3 install wandb torchrl-nightly==2024.10.20 tensordict-nightly==2024.10.20
RUN pip3 install -U tensorflow[and-cuda]==2.17.0

# TODO: add Blender installation

# ENTRYPOINT ["bash", "run_drl_L_hallway_calql.sh"]