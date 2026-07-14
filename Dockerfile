# FTP-1 training/inference image (openpi-based)
ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu22.04

ARG PYTHON_VERSION=3.11

ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=egl \
    PATH=/openpi/.venv/bin:/usr/local/bin:$PATH \
    DEVICE=cuda \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_LINK_MODE=copy

# ---- System packages ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    libegl1 \
    libgeos-dev \
    libgl1 \
    libglib2.0-0 \
    ninja-build \
    pkg-config \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && uv python install ${PYTHON_VERSION} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /openpi

ENV HOME=/root \
    HF_HOME=/root/.cache/huggingface \
    TORCH_HOME=/root/.cache/torch

# ---- Create venv ----
RUN uv venv --python ${PYTHON_VERSION}

# ---- Install project dependencies ----
COPY pyproject.toml uv.lock .python-version README.md ./
COPY src/ src/
COPY packages/ packages/
COPY scripts/ scripts/
COPY scripts_exp_zarr/ scripts_exp_zarr/
COPY assets/ assets/
COPY data_processing/ data_processing/
RUN uv sync --no-cache || uv sync --no-cache

# ---- Directories ----
RUN mkdir -p /openpi/checkpoints /openpi/data /openpi/output

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

CMD ["/bin/bash"]
