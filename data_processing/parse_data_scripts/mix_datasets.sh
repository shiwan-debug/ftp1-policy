#!/bin/bash
set -e

# ==============================================================================
# Script to run the dataset mixing tool
# ==============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Path to the python script
PYTHON_SCRIPT="${SCRIPT_DIR}/../parse_data_module/mix_datasets.py"

# Define default paths (override via environment variables as needed).
INPUT_DATA_ROOT="${INPUT_DATA_ROOT:-data_processing/output}"
OUTPUT_CONFIG_PATH="${OUTPUT_CONFIG_PATH:-data_processing/output/mixed_dataset_config.json}"

# Target Total Episodes (Optional)
# If not set, the script now tries to find the minimum upscaling to keep all zarr files.
# If you want a specific size, uncomment below.
TOTAL_EPISODES="${TOTAL_EPISODES:-500000}"

echo "========================================================"
echo "Dataset Mixer Launcher (Config Generator)"
echo "========================================================"
echo "Python Script: ${PYTHON_SCRIPT}"
echo "Input Root:    ${INPUT_DATA_ROOT}"
echo "Output Config: ${OUTPUT_CONFIG_PATH}"
echo "========================================================"

# Check if input directory exists
if [ ! -d "$INPUT_DATA_ROOT" ]; then
    echo "Error: Input directory '$INPUT_DATA_ROOT' does not exist."
    exit 1
fi

# Ensure output directory exists
OUTPUT_DIR=$(dirname "${OUTPUT_CONFIG_PATH}")
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Construct the command
CMD="python ${PYTHON_SCRIPT} --input_dir ${INPUT_DATA_ROOT} --output_path ${OUTPUT_CONFIG_PATH}"

if [ -n "$TOTAL_EPISODES" ]; then
    CMD="${CMD} --total_episodes ${TOTAL_EPISODES}"
fi

# Optional: Add --dry_run to see stats without creating files
# CMD="${CMD} --dry_run"

# Run the command
echo "Executing: $CMD"
$CMD

echo "========================================================"
echo "Done! Check ${OUTPUT_CONFIG_PATH} for results."
echo "========================================================"
