#!/bin/bash
# bash parse_data_scripts/parse_data_univtac.sh
# UniVTAC HDF5 -> FTP1 Zarr (joint-only export, gripper_idx=28)

# ================================ Configuration ================================
BASE_DIR="${BASE_DIR:-UniVTAC/data}"
SAVE_DIR="${SAVE_DIR:-data_processing/output/UniVTAC}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
TASK_LIST="${TASK_LIST:-}"
DOWNSAMPLE="${DOWNSAMPLE:-}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"

# ================================ Build Command Arguments ================================
PARSE_ARGS=(
    --base_dir="${BASE_DIR}"
    --save_dir="${SAVE_DIR}"
    --image_size="${IMAGE_SIZE}"
)

if [ -n "${EPISODES_PER_TASK}" ]; then
    PARSE_ARGS+=(--episodes_per_task="${EPISODES_PER_TASK}")
fi

if [ -n "${TASK_LIST}" ]; then
    PARSE_ARGS+=(--task_list="${TASK_LIST}")
fi

if [ -n "${DOWNSAMPLE}" ]; then
    PARSE_ARGS+=(--downsample="${DOWNSAMPLE}")
fi

# ================================ Run script ================================
cd "$(dirname "$0")/.."
python -m parse_data_module.parse_data_univtac "${PARSE_ARGS[@]}"
