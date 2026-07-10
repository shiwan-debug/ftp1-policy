from __future__ import annotations

from collections.abc import Mapping
import json
import pathlib

import numpy as np
import torch

ACTION_GROUP_STATS_FILENAME_TEMPLATE = "action_group_frequency_stats_{split}.json"


def get_action_group_stats_path(assets_dir: str | pathlib.Path, split: str = "train") -> pathlib.Path:
    return pathlib.Path(assets_dir) / ACTION_GROUP_STATS_FILENAME_TEMPLATE.format(split=split)


def get_ftp1_action_group_slices(
    single_arm_action_rep_dim: int,
    arm_joints_dim: int,
    reserved_action_dim: int,
) -> dict[str, tuple[int, int]]:
    action_dim = 2 * single_arm_action_rep_dim + 9 + reserved_action_dim
    return {
        "right-wrist-pos": (0, 3),
        "right-wrist-rot": (3, 9),
        "right-arm-joints": (9, 9 + arm_joints_dim),
        "right-hand-joints": (9 + arm_joints_dim, single_arm_action_rep_dim),
        "left-wrist-pos": (single_arm_action_rep_dim, single_arm_action_rep_dim + 3),
        "left-wrist-rot": (single_arm_action_rep_dim + 3, single_arm_action_rep_dim + 9),
        "left-arm-joints": (single_arm_action_rep_dim + 9, single_arm_action_rep_dim + 9 + arm_joints_dim),
        "left-hand-joints": (single_arm_action_rep_dim + 9 + arm_joints_dim, 2 * single_arm_action_rep_dim),
        "head-track-pos": (2 * single_arm_action_rep_dim, 2 * single_arm_action_rep_dim + 3),
        "head-track-rot": (2 * single_arm_action_rep_dim + 3, 2 * single_arm_action_rep_dim + 9),
        "supplementary-joints": (2 * single_arm_action_rep_dim + 9, action_dim),
    }


def _to_numpy_mask(action_mask: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(action_mask, torch.Tensor):
        action_mask = action_mask.detach().cpu().numpy()
    action_mask_np = np.asarray(action_mask)
    if action_mask_np.ndim == 2:
        action_mask_np = action_mask_np[None, ...]
    if action_mask_np.ndim != 3:
        raise ValueError(f"Expected action_mask with shape (B,T,D) or (T,D), got {action_mask_np.shape}")
    return action_mask_np


def count_action_group_presence(
    action_mask: np.ndarray | torch.Tensor,
    group_slices: Mapping[str, tuple[int, int]],
) -> tuple[int, dict[str, int]]:
    action_mask_np = _to_numpy_mask(action_mask)
    batch_size = int(action_mask_np.shape[0])
    present_counts: dict[str, int] = {}
    for group_name, (start_idx, end_idx) in group_slices.items():
        group_mask = action_mask_np[..., start_idx:end_idx]
        group_present = np.sum(group_mask, axis=(1, 2)) > 0
        present_counts[group_name] = int(np.sum(group_present))
    return batch_size, present_counts


def build_action_group_frequency_weights(
    group_frequency: Mapping[str, float],
    group_names: list[str],
    *,
    gamma: float,
    eps: float,
    clip_min: float,
    clip_max: float,
    normalize_mean: bool = True,
) -> dict[str, float]:
    positive_group_names = [group_name for group_name in group_names if float(group_frequency.get(group_name, 0.0)) > 0.0]
    if positive_group_names:
        positive_group_frequencies = np.asarray(
            [float(group_frequency[group_name]) for group_name in positive_group_names],
            dtype=np.float64,
        )
        # Use the geometric mean as the reference so dominant groups can be mildly down-weighted
        # while low-frequency groups are up-weighted on the same multiplicative scale.
        reference_frequency = float(np.exp(np.mean(np.log(positive_group_frequencies))))
    else:
        reference_frequency = 1.0

    weights: dict[str, float] = {}
    for group_name in group_names:
        freq = float(group_frequency.get(group_name, 0.0))
        if freq <= 0.0:
            # A never-seen group will never be valid in training loss aggregation, so keep it neutral
            # and exclude it from reference/normalization statistics.
            weights[group_name] = 1.0
            continue
        weight = (reference_frequency / (freq + eps)) ** gamma
        weight = float(np.clip(weight, clip_min, clip_max))
        weights[group_name] = weight

    if normalize_mean and positive_group_names:
        mean_weight = float(sum(weights[group_name] for group_name in positive_group_names) / len(positive_group_names))
        if mean_weight > 0:
            for group_name in positive_group_names:
                weights[group_name] = weights[group_name] / mean_weight
    return weights


def load_action_group_frequency_weights(
    stats_path: str | pathlib.Path,
    *,
    gamma: float,
    eps: float,
    clip_min: float,
    clip_max: float,
    normalize_mean: bool = True,
) -> tuple[dict[str, float], dict]:
    stats_path = pathlib.Path(stats_path)
    with open(stats_path) as f:
        stats = json.load(f)

    group_frequency = stats.get("group_frequency", {})
    group_names = list(stats.get("group_names", list(group_frequency.keys())))
    if not group_names:
        raise ValueError(f"No group_names found in action group stats file: {stats_path}")

    weights = build_action_group_frequency_weights(
        group_frequency,
        group_names,
        gamma=gamma,
        eps=eps,
        clip_min=clip_min,
        clip_max=clip_max,
        normalize_mean=normalize_mean,
    )
    return weights, stats


def compute_group_balanced_masked_loss(
    per_element_loss: torch.Tensor,
    action_masks: torch.Tensor,
    group_slices: Mapping[str, tuple[int, int]],
    *,
    group_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if per_element_loss.shape != action_masks.shape:
        raise ValueError(
            f"per_element_loss and action_masks must have the same shape, got {per_element_loss.shape} vs {action_masks.shape}"
        )
    if per_element_loss.ndim != 3:
        raise ValueError(f"Expected per_element_loss with shape (B,T,D), got {per_element_loss.shape}")

    batch_size = per_element_loss.shape[0]
    device = per_element_loss.device
    dtype = per_element_loss.dtype

    if group_weights is None:
        group_weights = torch.ones(len(group_slices), dtype=dtype, device=device)
    else:
        group_weights = group_weights.to(device=device, dtype=dtype)
        if group_weights.numel() != len(group_slices):
            raise ValueError(
                f"group_weights length mismatch: expected {len(group_slices)}, got {group_weights.numel()}"
            )

    group_loss_terms = []
    group_valid_terms = []
    for start_idx, end_idx in group_slices.values():
        group_loss = per_element_loss[..., start_idx:end_idx]
        group_mask = action_masks[..., start_idx:end_idx].to(device=device, dtype=dtype)
        group_loss_sum = (group_loss * group_mask).sum(dim=(1, 2))
        group_valid_count = group_mask.sum(dim=(1, 2))
        group_valid = group_valid_count > 0

        safe_group_valid_count = torch.clamp_min(group_valid_count, 1.0)
        normalized_group_loss = torch.where(
            group_valid,
            group_loss_sum / safe_group_valid_count,
            torch.zeros(batch_size, dtype=dtype, device=device),
        )

        group_loss_terms.append(normalized_group_loss)
        group_valid_terms.append(group_valid.to(dtype))

    group_loss_tensor = torch.stack(group_loss_terms, dim=1)
    group_valid_tensor = torch.stack(group_valid_terms, dim=1)
    weighted_valid = group_valid_tensor * group_weights[None, :]
    denom = weighted_valid.sum(dim=1)
    valid_samples = (denom > 0).to(dtype=dtype)
    weighted_loss_sum = (group_loss_tensor * weighted_valid).sum(dim=1)
    sample_loss = weighted_loss_sum / denom.clamp_min(1.0)
    return (sample_loss * valid_samples).sum() / valid_samples.sum().clamp_min(1.0)
