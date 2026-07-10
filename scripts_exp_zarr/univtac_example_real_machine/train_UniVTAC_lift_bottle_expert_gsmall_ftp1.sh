#!/usr/bin/env bash
# bash scripts_exp_zarr/univtac_example_real_machine/train_UniVTAC_lift_bottle_expert_gsmall_ftp1.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

set -euo pipefail

cd "${REPO_ROOT}"

task_name="lift_bottle"
task_dir="/cephfs/shared/yuanchengbo/rtac1_cache/data/UniVTAC/${task_name}"

repo_id="UniVTAC_${task_name}_zscore_mix_ftp1"
base_exp_name="UniVTAC_${task_name}_expert_gsmall_ftp1"
share_commit="${SHARE_COMMIT-}"
if [ -n "${share_commit}" ]; then
    exp_name="${share_commit}_${base_exp_name}"
else
    exp_name="${base_exp_name}"
fi
export USE_SWANLAB="true"
export SWANLAB_API_KEY="YOUR_SWANLAB_API_KEY"
# Pre-create SwanLab dirs to avoid multi-process race (FileExistsError) during `import swanlab`.
export SWANLAB_SAVE_DIR="${HOME}/.swanlab"
export SWANLAB_LOG_DIR="${HOME}/swanlog/${exp_name}"

dataset_config_path="${SCRIPT_DIR}/dataset_univtac_${task_name}.json"


check_norm_params_snapshot="false"

action_down_sample_steps=1
norm_type="zscore"
norm_image_tactile_mode="channel_wise"
state_input_mode="adarms"
proprioception_pose_rep="relative"
action_pose_rep="relative"
proprioception_joint_rep="abs"
action_joint_rep="mix"

export CUDA_VISIBLE_DEVICES=0
use_torch_compile="true"
gradient_checkpointing_enable="false"

use_wandb="true"
local_batch_size=16
num_train_steps=20000
log_interval=100
val_interval=2000
save_interval=20000
keep_period=20000
num_workers=12
val_num_workers=2
# pytorch_weight_path="/cephfs/shared/yuanchengbo/rtac1_cache/ckpt_pi05_base/ckpt_pytorch/pi05"
pytorch_weight_path="/cephfs/shared/yuanchengbo/rtac1_cache/pretrain_ckpt/ftp1_pretrain_v0426_50kstep"

model_tactile_expert_variant="gemma_small"
model_use_tactile_input="true"
model_state_input_mode="${state_input_mode}"
model_frozen_shared_chunk="false"

lr_warmup_steps=500
lr_peak_lr="${LR_PEAK_LR-1e-4}"
lr_decay_lr="${LR_DECAY_LR-1e-5}"

optimizer_b1=0.9
optimizer_b2=0.95
optimizer_eps=1e-8
optimizer_weight_decay=1e-10
optimizer_clip_gradient_norm=1.0

checkpoint_base_dir="/cephfs/shared/yuanchengbo/rtac1_cache/checkpoints"
assets_base_dir="/cephfs/shared/yuanchengbo/rtac1_cache/assets"
openpi_data_home="/cephfs/shared/yuanchengbo/rtac1_cache/openpi"
tmp_dir="/cephfs/shared/yuanchengbo/rtac1_cache/assets/tmp"

export OPENPI_DATA_HOME="${openpi_data_home}"
export HYDRA_FULL_ERROR=1
export TMPDIR="${tmp_dir}"
export TORCH_COMPILE_DIR="${TMPDIR}/torch_compile"
export TORCHDYNAMO_VERBOSE=0
unset TORCH_LOGS 2>/dev/null || true
mkdir -p "${TMPDIR}" "${TORCH_COMPILE_DIR}"

if [ ! -d "${task_dir}" ]; then
    echo "Task directory not found: ${task_dir}"
    exit 1
fi

if ! find "${task_dir}" -mindepth 1 -maxdepth 1 -type d -name "*.zarr" | grep -q .; then
    echo "No *.zarr directory found under: ${task_dir}"
    exit 1
fi

TASK_NAME="${task_name}" TASK_DIR="${task_dir}" DATASET_CONFIG_PATH="${dataset_config_path}" python - <<'PYCFG'
import json
import os

task_name = os.environ["TASK_NAME"]
task_dir = os.environ["TASK_DIR"]
dataset_config_path = os.environ["DATASET_CONFIG_PATH"]

payload = {
    "datasets": [
        {
            "name": f"UniVTAC_{task_name}",
            "path": task_dir,
            "use_trajectory_ratio": 1.0,
            "enabled": True,
        }
    ],
    "default_use_trajectory_ratio": 1.0,
    "description": "Auto-generated per-task dataset config for UniVTAC FTP1 training.",
}

with open(dataset_config_path, "w") as f:
    json.dump(payload, f, indent=2)
PYCFG

num_gpus=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
if [ "${num_gpus}" -lt 1 ]; then
    num_gpus=1
fi
world_size="${num_gpus}"
batch_size=$((local_batch_size * world_size))

echo "task_name=${task_name}"
echo "repo_id=${repo_id}"
echo "exp_name=${exp_name}"
echo "dataset_config_path=${dataset_config_path}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, world_size=${world_size}, batch_size=${batch_size}"

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
    --norm_type=${norm_type}
    --norm_image_tactile_mode=${norm_image_tactile_mode}
)

TRAIN_ARGS+=(--use_val_dataset)
TRAIN_ARGS+=(--create_train_val_split)

if [ "${gradient_checkpointing_enable}" = "true" ]; then
    TRAIN_ARGS+=(--gradient_checkpointing_enable)
else
    TRAIN_ARGS+=(--no-gradient_checkpointing_enable)
fi

if [ "${check_norm_params_snapshot}" = "true" ]; then
    TRAIN_ARGS+=(--check_norm_params_snapshot)
else
    TRAIN_ARGS+=(--no-check_norm_params_snapshot)
fi

if [ "${use_torch_compile}" = "true" ]; then
    TRAIN_ARGS+=(--use_torch_compile)
else
    TRAIN_ARGS+=(--no-use_torch_compile)
fi

if [ "${use_wandb}" = "true" ]; then
    TRAIN_ARGS+=(--wandb_enabled)
else
    TRAIN_ARGS+=(--no-wandb_enabled)
fi

if [ -n "${pytorch_weight_path}" ]; then
    TRAIN_ARGS+=(--pytorch_weight_path=${pytorch_weight_path})
fi

if [ -n "${model_tactile_expert_variant}" ]; then
    TRAIN_ARGS+=(--model.tactile_expert_variant=${model_tactile_expert_variant})
fi

if [ "${model_use_tactile_input}" = "true" ]; then
    TRAIN_ARGS+=(--model.use_tactile_input)
else
    TRAIN_ARGS+=(--model.no_use_tactile_input)
fi

if [ -n "${model_state_input_mode}" ]; then
    TRAIN_ARGS+=(--model.state_input_mode=${model_state_input_mode})
fi

if [ "${model_frozen_shared_chunk}" = "true" ]; then
    TRAIN_ARGS+=(--model.tactile_tokenizer_config.frozen_shared_chunk)
else
    TRAIN_ARGS+=(--model.tactile_tokenizer_config.no_frozen_shared_chunk)
fi

TRAIN_ARGS+=(--proprioception_pose_rep=${proprioception_pose_rep})
TRAIN_ARGS+=(--action_pose_rep=${action_pose_rep})
TRAIN_ARGS+=(--proprioception_joint_rep=${proprioception_joint_rep})
TRAIN_ARGS+=(--action_joint_rep=${action_joint_rep})

TRAIN_ARGS+=(--lr_schedule.warmup_steps=${lr_warmup_steps})
TRAIN_ARGS+=(--lr_schedule.peak_lr=${lr_peak_lr})
TRAIN_ARGS+=(--lr_schedule.decay_lr=${lr_decay_lr})

TRAIN_ARGS+=(--optimizer.b1=${optimizer_b1})
TRAIN_ARGS+=(--optimizer.b2=${optimizer_b2})
TRAIN_ARGS+=(--optimizer.eps=${optimizer_eps})
TRAIN_ARGS+=(--optimizer.weight_decay=${optimizer_weight_decay})
TRAIN_ARGS+=(--optimizer.clip_gradient_norm=${optimizer_clip_gradient_norm})

export JAX_PLATFORM=cpu

if [ "${num_gpus}" -gt 1 ]; then
  uv run python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${num_gpus}" \
    scripts/zarr_train_ftp1_pytorch.py "${TRAIN_ARGS[@]}"
else
  uv run python scripts/zarr_train_ftp1_pytorch.py "${TRAIN_ARGS[@]}"
fi
