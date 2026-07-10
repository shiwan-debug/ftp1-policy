# FTP-1：面向接触丰富操作、跨触觉传感器的通用基础触觉策略

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
  <a href="#feishu-group">
    <img src="https://img.shields.io/badge/Feishu-Group-2f6bff" alt="Feishu Group">
  </a>
</p>

<p align="center">
  <img src="./assets/teaser.png" alt="FTP-1 Overview" width="100%">
</p>

## 目录

- [简介](#intro-zh)
- [模型与数据集下载](#model-dataset-download-zh)
- [安装](#installation-zh)
- [FTP-1 模型后训练](#post-training-zh)
  - [数据准备](#data-preparation-zh)
  - [后训练](#training-zh)
  - [FTP-1 推理](#inference-zh)
  - [UniVTAC 上的评测](#evaluation-univtac-zh)
- [FTP-1 模型预训练](#pre-training-zh)
- [致谢](#acknowledgment-zh)

<a id="intro-zh"></a>
## 简介

FTP-1 是首个通用基础触觉策略，通过预训练学习可迁移的触觉操作能力，并支持跨多种触觉传感器与机器人 embodiment 的泛化。

这里的 **“通用”** 主要体现在：

 - **传感器通用**：FTP-1 支持多种触觉模态，包括基于图像、阵列式以及状态式触觉传感器。
 - **数据规模化**：FTP-1 在约 3000 小时的大规模异构触觉操作数据上进行预训练，覆盖人类示教、灵巧手以及夹爪机器人。
 - **embodiment 可迁移**：FTP-1 可以在不同传感器和机器人 embodiment 上进行微调，并能够迁移到未见过的传感器与平台，在性能上获得可观提升。

本仓库包含 FTP-1 的代码库，以及在下游触觉操作任务上微调 FTP-1 的教程，同时也提供 FTP-1 预训练所使用的代码。

<a id="model-dataset-download-zh"></a>
## 模型与数据集下载

- **模型检查点**

| 模型名称 | Hugging Face 仓库 | ModelScope 仓库 | 说明 |
| :--- | :--- | :--- | :--- |
| ftp1_pretrain_v0426_50kstep | [🤗 ftp1_v0426_50kstep](https://huggingface.co/michaelyuanqwq/ftp1_v0426_50kstep) | [🤖 ftp1_v0426_50kstep](https://www.modelscope.cn/models/michaelyuancb/ftp1_v0426_50kstep) | 我们的 v0426 通用触觉策略预训练模型，训练步数为 50k |
| ftp1_univtac_finetune | [🤗 ftp1_univtac_finetune](https://huggingface.co/michaelyuanqwq/ftp1_univtac_finetune) | [🤖 ftp1_univtac_finetune](https://www.modelscope.cn/models/michaelyuancb/ftp1_univtac_finetune) | 在 UniVTAC 基准上微调的 FTP-1-v0426，包含 6 个任务检查点 |

- **预训练数据集**

| 模型名称 | Hugging Face 仓库 | ModelScope 仓库 | 说明 |
| :--- | :--- | :--- | :--- |
| FTP-1-Dataset | [🤗 FTP-1-Dataset](https://huggingface.co/datasets/MJJJJ1064/FTP-1-Dataset) | [🤖 FTP-1-Dataset](https://www.modelscope.cn/datasets/Eureka1064/FTP-1-Dataset) | 用于预训练 FTP-1 模型的数据集 |

<a id="installation-zh"></a>
## 安装

本仓库构建在 [openpi](https://github.com/Physical-Intelligence/openpi) 之上，因此整体安装流程与其非常接近。

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

更多环境配置细节可参考 [README_openpi.md](./README_openpi.md)。

<a id="post-training-zh"></a>
## FTP-1 模型后训练

我们以 [UniVTAC](https://univtac.github.io/) 为例，说明如何在传感器特定、embodiment 特定的任务上完成 FTP-1 的完整微调流程。

<a id="data-preparation-zh"></a>
### 数据准备

FTP-1 的训练流水线基于 zarr 数据格式。在训练前，需要先将原始数据集转换为符合 FTP-1 兼容 key 约定的 zarr 文件。更多细节请参考 [data_processing/README.md](./data_processing/README.md)。

<a id="training-zh"></a>
### 后训练

示例微调脚本位于 `scripts_exp_zarr/univtac`。首先需要根据上一步生成的 zarr 数据准备数据集配置文件。示例文件位于：

```bash
scripts_exp_zarr/univtac/dataset_univtac.json
```

然后为 UniVTAC 域计算归一化统计量：

```bash
bash scripts_exp_zarr/univtac/compute_norm_stats_univtac_example.sh
```

生成的归一化文件会保存在 `assets/` 目录下。对于在预训练阶段已经见过的 embodiment，建议直接复用预训练得到的归一化资产，而不是从头重新统计。你可以从[模型与数据集下载](#model-dataset-download-zh)一节列出的预训练 checkpoint 中拷贝相关文件。对于在预训练阶段已经见过的传感器，我们也建议使用预训练 checkpoint 来初始化触觉编码器。更具体的内容可参考已下载预训练 checkpoint 目录中的 `hpt_tokenizer/` 和 `normalization/` 文件夹。归一化准备完成后，即可启动微调训练：

```bash
bash scripts_exp_zarr/univtac/train_univtac_example.sh
```

在实际启动训练前，建议重点检查 `train_univtac_example.sh` 和 `train_univtac_example_swanlab.sh` 中的以下参数：

- `repo_id` 和 `exp_name`：决定归一化资产和 checkpoint 的命名与组织方式。
- `dataset_config_path`：指定训练所使用的数据集 JSON，其中包含各个 domain 的 zarr 路径。
- `checkpoint_base_dir`、`assets_base_dir` 和 `OPENPI_DATA_HOME`：决定 checkpoint、归一化文件和运行时缓存的保存位置。
- `pytorch_weight_path`：指定用于初始化的预训练 FTP-1 或 $\pi_{0.5}$ checkpoint。
- `batch_size`、`num_train_steps`、`lr_warmup_steps`、`lr_peak_lr` 和 `lr_decay_lr`：定义主要的优化规模和学习率调度。
- `state_input_mode` 和 `model_tactile_expert_variant`：控制策略使用的输入形式以及 tactile expert 主干结构。
- `proprioception_pose_rep`、`action_pose_rep`、`proprioception_joint_rep` 和 `action_joint_rep`：定义机器人状态与动作的表示格式，并且应与数据准备和归一化阶段保持一致。
- `CUDA_VISIBLE_DEVICES`、`pytorch_training_precision` 和 `use_torch_compile`：控制设备分配和性能相关的运行时设置。

如果希望使用 [SwanLab](https://swanlab.cn/) 记录训练过程，可以运行：

```bash
bash scripts_exp_zarr/univtac/train_univtac_example_swanlab.sh
```

该启动脚本与普通训练脚本使用相同的训练参数，但会额外设置 SwanLab 相关环境变量，包括 `SWANLAB_API_KEY`、`SWANLAB_SAVE_DIR` 和 `SWANLAB_LOG_DIR`。

<a id="inference-zh"></a>
### FTP-1 推理

如果希望在独立环境中实例化一个训练好的 FTP-1 checkpoint 并执行推理，推荐使用 `src/openpi/policies/ftp1_inference_wrapper.py` 中的 `FTP1InferenceWrapper`。这个封装会自动处理 checkpoint 加载、输入归一化以及动作反归一化。

```python
from openpi.policies import FTP1InferenceWrapper
import numpy as np

wrapper = FTP1InferenceWrapper(
    checkpoint_dir="/path/to/checkpoint/7999",
    domain_name="your_domain_name",
    device="cuda",
    num_inference_steps=10,
)

images = {
    "camera_ego_rgb_0": np.zeros((224, 224, 3), dtype=np.uint8),
}
state = np.zeros((1, wrapper.get_state_dim()), dtype=np.float32)
prompt = "insert the tube"
tactiles = {
    "right_tactile_gripper": np.zeros((1, 2, 224, 224, 3), dtype=np.float32),
}
tactile_function_areas = {
    "right_tactile_gripper": [0, 1],
}
tactile_sensors = {
    "right_tactile_gripper": "GelSightMini",
}

action = wrapper.infer(
    images=images,
    state=state,
    prompt=prompt,
    tactiles=tactiles,
    tactile_function_areas=tactile_function_areas,
    tactile_sensors=tactile_sensors,
)
```

在当前 `FTP1InferenceWrapper` 这条推理路径下，通常**不需要**你显式管理 `batch_size`：这个封装本质上是单样本推理接口，内部默认使用 batch size `1`。但你仍然需要遵守输入中的时间维 `T` 约定：

- `images` 通常以单帧 `(H, W, 3)` 的形式传入，wrapper 会自动补 batch 维。
- `state` 的形状应为 `(T, state_dim)`。对于当前大多数 FTP-1 checkpoint，标准做法是使用 `T=1`，因为推理阶段通常只消费当前状态（默认 `disable_history=True`）。
- `tactiles` 的形状应为 `(T, num_areas, ...)`，而在大多数标准在线推理场景下，这里通常同样取 `T=1`。

返回的 `action` 是已经反归一化后的动作 chunk，形状为 `(action_horizon, action_dim)`，其中 `action_horizon` 由 checkpoint 配置决定（FTP-1 默认是 32）。在大多数实际部署中，通常执行其中的第一步或前几步动作，然后基于最新观测再次调用推理。

对于标准的 FTP-1 触觉 checkpoint，通常应当像上面的示例一样，在 `wrapper.infer(...)` 中同时提供 `tactiles`、`tactile_function_areas` 和 `tactile_sensors`。只有当该 checkpoint 本身是在不使用触觉输入的条件下训练得到时，这些参数才可以省略。更完整的输入输出格式说明以及更详细的示例，请直接参考 `src/openpi/policies/ftp1_inference_wrapper.py`。

如果只是想做 checkpoint 加载或调试，也可以参考 `scripts/zarr_infer_ftp1.py`，该脚本会加载模型并进入 `pdb` 供手动检查。

<a id="evaluation-univtac-zh"></a>
### UniVTAC 上的评测

我们在 [UniVTAC/Installation_FTP1.md](./UniVTAC/Installation_FTP1.md) 中提供了 FTP-1 在 UniVTAC 基准环境中的安装说明。完成环境配置后，运行：

```bash
cd UniVTAC
bash scripts/shell/eval_ftp1_batch.sh
```

这会启动评测流程。评测结果保存在 `UniVTAC/eval_results` 中。

你也可以直接下载我们已经微调好的 checkpoint，而无需自己重新完成微调训练。

<a id="pre-training-zh"></a>
## 5. FTP-1 模型预训练

如果你想查看一个轻量级的预训练示例，可以参考 `scripts_exp_zarr/pretrain_small`，其中提供了归一化统计量计算和 FTP-1 预训练启动的最小示例流程。

当多个 domain 一起参与训练时，FTP-1 会自动启用其面向大规模异构数据的预训练基础设施。该基础设施会将来自不同 domain 的数据自动分配到不同 GPU 上，使得每个 GPU 内的 batch 样本保持相同的数据格式，从而支持高效的大规模异构训练。domain-specific 模块的梯度会独立更新，而 shared 模块的梯度会先合并，再进行联合更新。

<a id="acknowledgment-zh"></a>
## 6. 致谢

本仓库基于 [OpenPi](https://github.com/Physical-Intelligence/openpi)、[MotionTrans](https://motiontrans.github.io/)、[MotionTrans-Pi](https://github.com/michaelyuancb/motiontrans-pi0)、[UniVTAC](https://github.com/univtac/UniVTAC) 和 [T3-Encoder](https://github.com/alanzjl/t3) 的代码实现。我们衷心感谢这些开源项目对本工作的支持与贡献。同时，也感谢我们的 AI 协作伙伴 [Codex](https://chatgpt.com/codex/) 和 [ClaudeCode](https://code.claude.com/)。

<a id="feishu-group"></a>
## 7. Feishu Group / 飞书交流群

<p align="center">
  <a href="./assets/FTP1_feishugroup.jpg">
    <img src="./assets/FTP1_feishugroup.jpg" alt="Feishu Group / 飞书交流群" width="320">
  </a>
</p>
