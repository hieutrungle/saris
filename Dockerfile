FROM ubuntu:22.04 AS base-all

RUN echo 'APT::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker
RUN echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache,sharing=locked \
    apt-get update \
    && apt-get upgrade --assume-yes \
    && apt-get install --assume-yes --no-install-recommends python3-pip build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget python-is-python3 software-properties-common

# RUN --mount=type=cache,target=/var/lib/apt/lists \
#     --mount=type=cache,target=/var/cache,sharing=locked \ 
#     apt-get update \
#     && apt-get install --assume-yes --no-install-recommends python3.12 python3.12-dev python3.12-distutils


# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 

# RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

FROM base-all AS builder

RUN pip install poetry==1.8.3

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN touch README.md

RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --no-root

FROM base-all AS runtime

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

WORKDIR /myapp
COPY ../../blender ./blender

RUN mkdir -p research/saris

WORKDIR /myapp/research/saris
COPY saris ./saris
COPY pyproject_docker.toml ./pyproject.toml
RUN pip install -e .
COPY configs ./configs
COPY tmp_wandb_api_key.txt ./
COPY run_drl_L_hallway_calql.sh ./

ENTRYPOINT ["bash", "run_drl_L_hallway_calql.sh"]