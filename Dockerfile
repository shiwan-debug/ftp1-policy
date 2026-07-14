# FTP-1 training/inference image
FROM nvidia/cuda:12.8.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=egl \
    DEVICE=cuda

# ---- System ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    build-essential cmake curl git \
    libegl1 libgeos-dev libgl1 libglib2.0-0 \
    ffmpeg ninja-build pkg-config \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install --upgrade pip

WORKDIR /openpi

# ---- Copy project ----
COPY pyproject.toml ./
COPY src/ src/
COPY packages/ packages/
COPY scripts/ scripts/
COPY scripts_exp_zarr/ scripts_exp_zarr/
COPY assets/ assets/
COPY data_processing/ data_processing/

# ---- Install dependencies ----
RUN pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu124
RUN pip install "jax[cuda12]==0.5.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install lerobot@git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
RUN pip install ./packages/openpi-client
RUN pip install -e .[dev]

# ---- Dirs ----
RUN mkdir -p /openpi/checkpoints /openpi/data /openpi/output

CMD ["/bin/bash"]
