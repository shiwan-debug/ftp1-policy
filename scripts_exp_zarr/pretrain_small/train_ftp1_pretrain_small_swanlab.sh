#!/usr/bin/env bash
export USE_SWANLAB="true"
export SWANLAB_API_KEY="YOUR_SWANLAB_API_KEY"
# Pre-create SwanLab dirs to avoid multi-process race (FileExistsError) during `import swanlab`.
export SWANLAB_SAVE_DIR="${HOME}/.swanlab"
export SWANLAB_LOG_DIR="${HOME}/swanlog/${exp_name}"


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ================================ Core parameters ================================
repo_id="pretrain_small_example"
exp_name="pretrain_small_train"
dataset_config_path="scripts_exp_zarr/pretrain_small/data_config_pretrain_small.json"

checkpoint_base_dir="/path/to/ftp1_cache/checkpoints"
assets_base_dir="/path/to/ftp1_cache/assets"
export OPENPI_DATA_HOME="/path/to/ftp1_cache/openpi"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export HYDRA_FULL_ERROR=1
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-qXmDwhp7JteGMMjAksfuh}"
export SWANLAB_SAVE_DIR="${SWANLAB_SAVE_DIR:-$HOME/.swanlab}"
export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-${REPO_ROOT}/swanlog/${exp_name}}"
mkdir -p "${SWANLAB_SAVE_DIR}" "${SWANLAB_LOG_DIR}"

local_batch_size=16
num_train_steps=150000
log_interval=25
val_interval=1000
save_interval=25000
keep_period=25000
num_workers=16
val_num_workers=2

action_down_sample_steps=1
norm_type="zscore"
state_input_mode="adarms"
model_tactile_expert_variant="gemma_small"

lr_warmup_steps=300
lr_peak_lr="5e-5"
lr_decay_lr="3e-6"

optimizer_b1=0.9
optimizer_b2=0.95
optimizer_eps="1e-8"
optimizer_weight_decay="1e-10"
optimizer_clip_gradient_norm=1.5

pytorch_weight_path=""
use_wandb="true"

num_gpus=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c . || echo "1")
if [ "${num_gpus}" -lt 1 ]; then
  num_gpus=1
fi
batch_size=$((local_batch_size * num_gpus))

TRAIN_ARGS=(
  ftp1
  --exp_name=${exp_name}
  --repo_id=${repo_id}
  --data.repo-id=${repo_id}
  --checkpoint_base_dir=${checkpoint_base_dir}
  --assets_base_dir=${assets_base_dir}
  --dataset_config_path=${dataset_config_path}
  --batch_size=${batch_size}
  --action_down_sample_steps=${action_down_sample_steps}
  --num_train_steps=${num_train_steps}
  --log_interval=${log_interval}
  --val_interval=${val_interval}
  --save_interval=${save_interval}
  --keep_period=${keep_period}
  --num_workers=${num_workers}
  --val_num_workers=${val_num_workers}
  --use_val_dataset
  --create_train_val_split
  --norm_type=${norm_type}
  --model.state_input_mode=${state_input_mode}
  --model.tactile_expert_variant=${model_tactile_expert_variant}
  --lr_schedule.warmup_steps=${lr_warmup_steps}
  --lr_schedule.peak_lr=${lr_peak_lr}
  --lr_schedule.decay_steps=${num_train_steps}
  --lr_schedule.decay_lr=${lr_decay_lr}
  --optimizer.b1=${optimizer_b1}
  --optimizer.b2=${optimizer_b2}
  --optimizer.eps=${optimizer_eps}
  --optimizer.weight_decay=${optimizer_weight_decay}
  --optimizer.clip_gradient_norm=${optimizer_clip_gradient_norm}
)

if [ "${use_wandb}" = "true" ]; then
  TRAIN_ARGS+=(--wandb_enabled)
else
  TRAIN_ARGS+=(--no-wandb_enabled)
fi

if [ -n "${pytorch_weight_path}" ]; then
  TRAIN_ARGS+=(--pytorch_weight_path=${pytorch_weight_path})
fi

if [ "${num_gpus}" -gt 1 ]; then
  uv run python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${num_gpus}" \
    scripts/zarr_train_ftp1_pytorch.py "${TRAIN_ARGS[@]}"
else
  uv run python scripts/zarr_train_ftp1_pytorch.py "${TRAIN_ARGS[@]}"
fi
