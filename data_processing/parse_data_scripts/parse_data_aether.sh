#!/bin/bash
# bash parse_data_scripts/parse_data_aether.sh

# Configuration
BASE_DIR="${BASE_DIR:-data_processing/SAMPLE_DATASETS/AetherData}"
SAVE_DIR="${SAVE_DIR:-data_processing/output/AetherAnnotatedDS}"
MODE="${MODE:-o}"  # 'o', 'p', 's', 'd', 'a'

# Image processing parameters
MAX_FRAMES="${MAX_FRAMES:-400}"
BIN_EXTEND_FRAMES="${BIN_EXTEND_FRAMES:-48}"
SPEED_DOWNSAMPLE_RATIO="${SPEED_DOWNSAMPLE_RATIO:-6.0}"
# out_resolutions_resize=(1280, 720)"
# out_resolutions_crop=(640, 480)
# out_resolutions_image_final=(224, 224)

# ================================ Build Command Arguments ================================
PARSE_ARGS=(
    --base_dir=${BASE_DIR}
    --save_dir=${SAVE_DIR}
    --mode=${MODE}
    --speed_downsample_ratio=${SPEED_DOWNSAMPLE_RATIO}
)

if [ -n "${MAX_FRAMES:-}" ]; then
    PARSE_ARGS+=(--max_frames=${MAX_FRAMES})
fi

if [ -n "${BIN_EXTEND_FRAMES:-}" ]; then
    PARSE_ARGS+=(--bin_extend_frames=${BIN_EXTEND_FRAMES})
fi

if [ -n "${DATA_ID:-}" ]; then
    PARSE_ARGS+=(--data_id=${DATA_ID})
fi

# ================================ Run script ================================
cd "$(dirname "$0")/.."
python -m parse_data_module.parse_data_aether "${PARSE_ARGS[@]}"
