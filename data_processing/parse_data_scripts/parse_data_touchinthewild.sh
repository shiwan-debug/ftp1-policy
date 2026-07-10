#!/bin/bash
# bash data_processing/parse_data_scripts/parse_data_touchinthewild.sh

set -e

# ================================ Configuration ================================

# Root directory containing TouchInTheWild subdirectories such as
# `four_tasks/` and `indoor_data/`.
TITW_ROOT="${TITW_ROOT:-data_processing/SAMPLE_DATASETS/TouchInTheWild}"

# Output root. The parser mirrors the raw directory structure under this path.
TITW_OUTPUT="${TITW_OUTPUT:-data_processing/output/TouchInTheWild}"

# Optional: process only one archive, specified either by filename or by
# relative path, for example:
#   fluid_transfer.zarr.zip
#   four_tasks/fluid_transfer/fluid_transfer.zarr.zip
TITW_DATASET_NAME="${TITW_DATASET_NAME:-}"

TITW_STRIDE="${TITW_STRIDE:-1}"
TITW_START_EPISODE="${TITW_START_EPISODE:-0}"
TITW_MAX_EPISODES="${TITW_MAX_EPISODES:-}"
TITW_MAX_STEPS="${TITW_MAX_STEPS:-}"
TITW_INSTRUCTION="${TITW_INSTRUCTION:-}"
TITW_INSTRUCTION_PAD="${TITW_INSTRUCTION_PAD:-0}"
TITW_GRIPPER_IDX="${TITW_GRIPPER_IDX:-28}"
TITW_OVERWRITE="${TITW_OVERWRITE:-true}"
TITW_USE_GPT="${TITW_USE_GPT:-true}"
TITW_GPT_N_IMAGES="${TITW_GPT_N_IMAGES:-5}"
TITW_GPT_N_NEW="${TITW_GPT_N_NEW:-5}"
TITW_EXPANDED_JSON="${TITW_EXPANDED_JSON:-assets/TouchInTheWild/task_description_expanded.json}"

# ================================ Build Command Arguments ================================

PARSE_ARGS=(
    --root "${TITW_ROOT}"
    --output "${TITW_OUTPUT}"
    --stride ${TITW_STRIDE}
    --start_episode ${TITW_START_EPISODE}
    --instruction_pad ${TITW_INSTRUCTION_PAD}
    --gripper_idx ${TITW_GRIPPER_IDX}
)

if [ -n "${TITW_DATASET_NAME:-}" ]; then
    PARSE_ARGS+=(--dataset_name "${TITW_DATASET_NAME}")
fi

if [ -n "${TITW_MAX_EPISODES:-}" ]; then
    PARSE_ARGS+=(--max_episodes ${TITW_MAX_EPISODES})
fi

if [ -n "${TITW_MAX_STEPS:-}" ]; then
    PARSE_ARGS+=(--max_steps ${TITW_MAX_STEPS})
fi

if [ -n "${TITW_INSTRUCTION:-}" ]; then
    PARSE_ARGS+=(--instruction "${TITW_INSTRUCTION}")
fi

if [ "${TITW_OVERWRITE}" = true ]; then
    PARSE_ARGS+=(--overwrite)
fi

if [ "${TITW_USE_GPT}" = true ]; then
    PARSE_ARGS+=(--use_gpt_instruction)
    if [ -n "${TITW_EXPANDED_JSON:-}" ]; then
        PARSE_ARGS+=(--expanded_task_list_json "${TITW_EXPANDED_JSON}")
    fi
    PARSE_ARGS+=(
        --gpt_n_images ${TITW_GPT_N_IMAGES}
        --gpt_n_new_instructions ${TITW_GPT_N_NEW}
    )
fi

# ================================ Run script ================================

cd "$(dirname "$0")/.."
python -m parse_data_module.parse_data_touchinthewild "${PARSE_ARGS[@]}"
