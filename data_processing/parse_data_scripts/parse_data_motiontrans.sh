#!/bin/bash
# bash parse_data_scripts/parse_data_motiontrans.sh

# Configuration
INPUT_DIR="${INPUT_DIR:-data_processing/SAMPLE_DATASETS/MotionTrans/raw_data}"

OUTPUT="${OUTPUT:-data_processing/output/MotionTrans}"
HAND_TO_EEF_FILE="${HAND_TO_EEF_FILE:-parse_data_module/franka_eef_to_wrist_robot_base.npy}"
MODE="${MODE:-o}"  # 'o' (only image), 'p' (with pointcloud), 's' (with stereo), 'a' (all)
N_ENCODING_THREADS="${N_ENCODING_THREADS:--1}"  # -1 keeps the parser default behavior

# Optional parameters
RESOLUTION_RESIZE="${RESOLUTION_RESIZE:-1280x720}"
RESOLUTION_CROP="${RESOLUTION_CROP:-640x480}"
RESOLUTION_IMAGE_FINAL="${RESOLUTION_IMAGE_FINAL:-224x224}"
NUM_USE_SOURCE="${NUM_USE_SOURCE:-}"
VERBOSE="${VERBOSE:-true}"

# ================================ Build Command Arguments ================================
PARSE_ARGS=(
    --input_dir=${INPUT_DIR}
    --output=${OUTPUT}
    --hand_to_eef_file=${HAND_TO_EEF_FILE}
    --mode=${MODE}
)

if [ -n "${RESOLUTION_RESIZE:-}" ]; then
    PARSE_ARGS+=(--resolution_resize=${RESOLUTION_RESIZE})
fi

if [ -n "${RESOLUTION_CROP:-}" ]; then
    PARSE_ARGS+=(--resolution_crop=${RESOLUTION_CROP})
fi

if [ -n "${RESOLUTION_IMAGE_FINAL:-}" ]; then
    PARSE_ARGS+=(--resolution_image_final=${RESOLUTION_IMAGE_FINAL})
fi

if [ -n "${NUM_USE_SOURCE:-}" ]; then
    PARSE_ARGS+=(--num_use_source=${NUM_USE_SOURCE})
fi

if [ -n "${N_ENCODING_THREADS:-}" ] && [ "${N_ENCODING_THREADS}" != "-1" ]; then
    PARSE_ARGS+=(--n_encoding_threads=${N_ENCODING_THREADS})
fi

if [ "${VERBOSE}" = true ]; then
    PARSE_ARGS+=(--verbose)
fi

# ================================ Run script ================================
cd "$(dirname "$0")/.."
python -m parse_data_module.parse_data_motiontrans "${PARSE_ARGS[@]}"
