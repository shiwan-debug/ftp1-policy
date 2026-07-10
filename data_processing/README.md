<h1 align="center">FTP-1 Data Processing Tutorial</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2606.13102">
    <img src="https://img.shields.io/badge/Arxiv-2606.13102-b31b1b" alt="Arxiv">
  </a>
  <a href="https://ftp1-policy.github.io/">
    <img src="https://img.shields.io/badge/%F0%9F%8C%90%20Project%20Page-Website-4f8f8b" alt="Project Page">
  </a>
  <a href="https://huggingface.co/MJJJJ1064/ftp1_v0426_50kstep">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-ftp1__v0426__50kstep-f2a93b" alt="Model">
  </a>
  <a href="https://www.modelscope.cn/datasets/Eureka1064/FTP-1-Dataset">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-FTP--1--Dataset-ff6f61" alt="Dataset">
  </a>
  <a href="./README.md">
    <img src="https://img.shields.io/badge/English-README-5865a6" alt="English">
  </a>
  <a href="./README_zh.md">
    <img src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-README-b06a4e" alt="中文">
  </a>
</p>

<p align="center">
  <img src="../assets/dataset.png" alt="FTP-1 dataset overview" width="100%">
</p>

## Table of Contents

- [1. Installation](#dp-installation)
- [2. Overview](#dp-overview)
  - [2.1 Zarr Directory Layout](#dp-zarr-layout)
  - [2.2 Zarr Keys](#dp-zarr-keys)
  - [2.3 Pose Coordinate Definition](#dp-pose-definition)
- [3. Observation](#dp-observation)
  - [3.1 Camera](#dp-camera)
  - [3.2 Tactile](#dp-tactile)
- [4. Action Space](#dp-action-space)
  - [4.1 Usually required](#dp-required)
  - [4.2 Embodiment state keys](#dp-embodiment-state)
  - [4.3 hand_joints_idx keys](#dp-hand-joints-idx)
- [5. Examples](#dp-examples)

<a id="dp-installation"></a>
## 1. Installation
```bash
conda create -n ftp_data python=3.10
conda activate ftp_data
pip install -r requirements.txt
```

For datasets that rely on ZED2 camera input, also install the [ZED SDK Python bindings](https://www.stereolabs.com/docs/app-development/python/install).

<a id="dp-overview"></a>
## 2. Overview

This document describes the public zarr contract produced by `data_processing/`. All exported arrays are time-major: the first dimension is always `T`.

The expected pipeline is:

```text
original dataset -> parser in data_processing/ -> standardized .zarr -> FTP1 training / loading
```

In other words, this document describes the standardized intermediate zarr format that parser scripts should generate from raw source data before model training.

[2026.06.29] You can also try [TLabel](https://github.com/liesliy/tlabel), a GUI-based data processing tool to export your tactile data with FTP-1 training format. More details could be found in  https://github.com/michaelyuancb/ftp1-policy/issues/2

<a id="dp-zarr-layout"></a>
### 2.1 Zarr Directory Layout

In practice, parsers usually write one processed dataset / domain directory, and that directory contains one or more `.zarr` files for different tasks, splits, or episode batches. A typical layout is:

```text
<dataset_or_domain>/
  task_a.zarr
  task_b.zarr
  ...
```

or, for batched export:

```text
<dataset_or_domain>/
  episode_batch_0000.zarr
  episode_batch_0001.zarr
  ...
```

The training loader then reads all `.zarr` files under that directory as one dataset domain.

<a id="dp-zarr-keys"></a>
### 2.2 Zarr Keys

Typical keys include:

```text
timestamps: (T,)
camera_main_rgb / camera_ego_rgb: (T, H, W, 3)
right_wrist_camera_rgb / left_wrist_camera_rgb: (T, H, W, 3)
camera_ego_pose: (T, 6)
left_wrist_pose / right_wrist_pose: (T, 6)
left_arm_joints / right_arm_joints: (T, J)
left_hand_joints / right_hand_joints: (T, K)
left_hand_joints_idx / right_hand_joints_idx: (T, K)
<side>_tactile_data_<group>: (T, N, *tac_shape)
<side>_tactile_area_<group>: (T, N)
<side>_tactile_sensor_<group>: (T,)
<side>_tactile_type_<group>: (T,)
sub_task_instruction: (T,)
```

`T` is episode length; `H, W` are image height/width; `J` is the number of arm-joint channels; `K` is the number of hand or gripper channels; `N` is the number of tactile channels in one group; `tac_shape` is the per-channel tactile shape after the `N` axis.

<a id="dp-pose-definition"></a>
### 2.3 Pose Coordinate Definition

For dexterous-hand data, along the egocentric camera viewing direction, the camera, left wrist, and right wrist coordinate frames all follow the same convention: `z` forward, `x` right, `y` down.

For gripper and UMI-style data, we approximately align the embodiment to a closed human-hand grasp convention; refer to [definition_pose_gripper_umi.png](./definition_pose_gripper_umi.png).

<a id="dp-observation"></a>
## 3. Observation

<a id="dp-camera"></a>
### 3.1 Camera

The current loader in [`dataset_zarr.py`](../src/openpi/dataset_zarr.py) supports these RGB keys:

`camera_main_rgb`, `camera_ego_rgb`, `right_wrist_camera_rgb`, `left_wrist_camera_rgb`.

At least one RGB stream should exist. The current lookup priority is:

```text
camera_main_rgb -> camera_ego_rgb -> right_wrist_camera_rgb -> left_wrist_camera_rgb
```

`camera_ego_pose` is the standard camera-pose field currently consumed by the loader. Pose fields use `(x, y, z, rx, ry, rz)` with `rotvec` rotation.

<a id="dp-tactile"></a>
### 3.2 Tactile

Each tactile group is exported with four aligned members:

```text
<side>_tactile_data_<group>: (T, N, *tac_shape)
<side>_tactile_area_<group>: (T, N)
<side>_tactile_sensor_<group>: (T,)
<side>_tactile_type_<group>: (T,)
```

`data` stores the tactile tensor; `area` stores function-area ids; `sensor` stores the sensor name; `type` is one of `state`, `binary`, `image`.

The `area` definition follows the tactile / force functional-area convention illustrated in [definition_tactile_torque_function_area.png](./definition_tactile_torque_function_area.png).

Multiple functional areas may be grouped into one tensor when they share the same geometry and semantics, for example `*_tactile_data_fingers`.

Example (gripper vision-tactile, UniVTAC): if one gripper exports two tactile pads as RGB images, then a typical group can be
`right_tactile_data_gripper: (T, 2, 224, 224, 3)`, `right_tactile_area_gripper: (T, 2)` with values like `[0, 1]`,
`right_tactile_sensor_gripper: (T,)` with values like `GelSightMini`, and `right_tactile_type_gripper: (T,)` with value `image`.

<a id="dp-action-space"></a>
## 4. Action Space

FTP1 does not require a standalone `actions` key in the raw zarr. Training targets are derived from future state trajectories.

Internally, FTP1 packs embodiment-specific state/action signals into a unified action space (UAS) layout; see [`dataset_zarr.py`](../src/openpi/dataset_zarr.py) for the concrete packing logic. At a high level, one step is concatenated as:

```text
[left arm block (48)] + [right arm block (48)] + [ego/head block (3+6)] + [supplementary block (15)]
```

Each arm block is conceptually:

```text
[wrist pose (3+6)] + [arm joints (7)] + [hand / gripper joints (32)]
```

This is why raw zarr data only needs to provide the relevant state streams such as `*_wrist_pose`, `*_arm_joints`, `*_hand_joints`, and `*_hand_joints_idx`; the training loader builds the final FTP1 action/state tensor from those fields instead of reading a pre-concatenated `actions` array. More details please refer to our paper, especially the UAS description.

<a id="dp-required"></a>
### 4.1 Usually required

For a dataset to be usable, the following are normally expected:

- `timestamps`: canonical sequence axis
- `sub_task_instruction`: per-step language instruction
- at least one RGB key
- embodiment state keys needed to derive future actions

<a id="dp-embodiment-state"></a>
### 4.2 Embodiment state keys

- `left_wrist_pose / right_wrist_pose`: wrist pose trajectory in world frame
- `left_arm_joints / right_arm_joints`: arm-joint trajectory; consumed directly when present
- `left_hand_joints / right_hand_joints`: hand or gripper trajectory
- `left_hand_joints_idx / right_hand_joints_idx`: canonical slot ids for each exported hand-joint channel

<a id="dp-hand-joints-idx"></a>
### 4.3 hand_joints_idx keys

`*_hand_joints` may be a 21D human hand, a dexterous-hand actuator vector, or a 1D gripper scalar, so cross-embodiment semantics are carried by `*_hand_joints_idx`, not by channel position alone.

For hand_joints_idx, the primary reference is [definition_faas_human_hand_joint.png](./definition_faas_human_hand_joint.png). For broader FAAS alignment across embodiments, also refer to [definition_faas_general.png](./definition_faas_general.png) and the paper. If a dataset exports MANO-derived 21D joints, [definition_mano_hand_index.jpg](./definition_mano_hand_index.jpg) provides the MANO keypoint order.

Examples:

- Gripper: if `left_hand_joints` has shape `(T, 1)`, then `left_hand_joints_idx` usually has shape `(T, 1)` filled with `28`, e.g. `[[28], [28], ...]`.
- Human hand: if `left_hand_joints` has shape `(T, 21)` in MANO-derived order, then `left_hand_joints_idx` usually has shape `(T, 21)` with repeated FAAS mapping `[1, 26, 2, 3, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 27]`.

The MANO channel order itself follows [definition_mano_hand_index.jpg](./definition_mano_hand_index.jpg).

<a id="dp-examples"></a>
## 5. Examples

The current examples cover four common source types:

- Human data: `parse_data_aether.sh`
- Dexterous-hand robot data: `parse_data_motiontrans.sh`
- UMI-style data: `parse_data_touchinthewild.sh`
- Gripper data: `parse_data_univtac.sh`

Example for human data:

```bash
bash parse_data_scripts/parse_data_aether.sh
```

Other common entry points:

```bash
bash parse_data_scripts/parse_data_motiontrans.sh
bash parse_data_scripts/parse_data_touchinthewild.sh
bash parse_data_scripts/parse_data_univtac.sh
```

After parsing, you can visualize one generated zarr with:

```bash
python visualize_zarr_data.py --data_path <one_output_zarr> -p
```
