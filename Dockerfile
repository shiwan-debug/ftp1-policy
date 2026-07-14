FROM nvidia/cuda:12.8.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=egl \
    DEVICE=cuda

# System + Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-distutils \
    build-essential cmake curl git \
    libegl1 libgeos-dev libgl1 libglib2.0-0 \
    ffmpeg ninja-build pkg-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /openpi

COPY pyproject.toml ./
COPY src/ src/
COPY packages/ packages/
COPY scripts/ scripts/
COPY scripts_exp_zarr/ scripts_exp_zarr/
COPY assets/ assets/
COPY data_processing/ data_processing/

RUN python3.11 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
RUN python3.11 -m pip install "jax[cuda12]==0.5.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN python3.11 -m pip install lerobot@git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
RUN python3.11 -m pip install dlimp@git+https://github.com/kvablack/dlimp@ad72ce3a9b414db2185bc0b38461d4101a65477a
RUN python3.11 -m pip install ./packages/openpi-client
RUN python3.11 -m pip install -e .[dev]

RUN mkdir -p /openpi/checkpoints /openpi/data /openpi/output

CMD ["python3.11"]
