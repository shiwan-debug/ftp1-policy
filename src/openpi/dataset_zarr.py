import copy
from datetime import datetime
import hashlib
import json
import logging
import math
import os
import pathlib
import random
import re
import shutil
import time
from typing import Dict, List, Protocol, SupportsIndex, TypeVar

import cv2
from filelock import FileLock
import jax.numpy as jnp
import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st
import torch
import torchvision
from tqdm import tqdm
import zarr

from openpi.imagecodecs_numcodecs import register_codecs
from openpi.json_utils import _deserialize_params_value
from openpi.json_utils import _serialize_params_value
from openpi.json_utils import dump_json_with_inline_lists
from openpi.models_pytorch.ftp1_model_config import FTP1_SINGLE_ARM_JOINT_DIM
from openpi.norm_stats_utils import load_norm_stats_jsonable_with_override
from openpi.normalization import apply_norm_stats
from openpi.normalization import calculate_norm_stats
from openpi.pose_repr_utils import convert_pose_mat_rep
from openpi.pose_utils import mat_to_pose9d
from openpi.pose_utils import pose_to_mat
from openpi.replay_buffer import ReplayBuffer
from openpi.ftp1_action_groups import count_action_group_presence
from openpi.ftp1_action_groups import get_action_group_stats_path
from openpi.ftp1_action_groups import get_ftp1_action_group_slices
from openpi.tactile_contact_detection import build_contact_thresholds_jsonable
from openpi.tactile_contact_detection import compute_tactile_delta_score
from openpi.tactile_contact_detection import compute_tactile_delta_scores
from openpi.tactile_contact_detection import CONTACT_DETECTOR_VERSION
from openpi.tactile_contact_detection import CONTACT_THRESHOLD_FILENAME
from openpi.tactile_contact_detection import DEFAULT_CONTACT_THRESHOLD_K
from openpi.tactile_contact_detection import is_contact_score_above_threshold
from openpi.tactile_contact_detection import is_tactile_data_key
from openpi.tactile_contact_detection import load_contact_thresholds_jsonable_with_override
from openpi.tactile_contact_detection import require_stream_contact_threshold
import openpi.shared.normalize as _shared_normalize
from openpi.tactile_identity import build_tactile_sensor_type_shape_key_from_data
from openpi.tactile_identity import canonicalize_tactile_encoder_type
from openpi.tactile_identity import canonicalize_tactile_sensor_name

register_codecs()

FTP1_GRIPPER_HAND_SLOT_INDEX = 28
SPLIT_STRATEGY_VERSION = 4


def _canonicalize_action_joint_rep(rep: str | None) -> str | None:
    if rep is None:
        return None
    normalized = str(rep).strip().lower()
    if normalized in {"abs", "absolute"}:
        return "absolute"
    if normalized == "relative":
        return "relative"
    if normalized == "mix":
        return "mix"
    return None


def _action_joint_rep_uses_relative_arm(rep: str | None) -> bool:
    normalized = _canonicalize_action_joint_rep(rep)
    return normalized in {"relative", "mix"}


def _action_joint_rep_uses_relative_hand(rep: str | None) -> bool:
    normalized = _canonicalize_action_joint_rep(rep)
    return normalized in {"relative", "mix"}


def _action_joint_rep_absolute_hand_slots(rep: str | None) -> tuple[int, ...]:
    normalized = _canonicalize_action_joint_rep(rep)
    if normalized == "mix":
        # In mix mode, all hand slots are relative except the gripper slot.
        return (FTP1_GRIPPER_HAND_SLOT_INDEX,)
    return ()


def _action_joint_rep_uses_relative_supplementary(rep: str | None) -> bool:
    normalized = _canonicalize_action_joint_rep(rep)
    return normalized == "relative"


def _should_compute_contact_for_non_tactile_dropout(non_tactile_dropout_ratio: float) -> bool:
    return float(non_tactile_dropout_ratio) > 0.0


def collate_fn_handle_strings(batch):
    """Custom collate function that handles string fields (like tactile_sensor)
    by converting numpy string arrays to Python strings and collating dicts properly.
    Also handles memory-mapped arrays from zarr by copying them.

    This function avoids using default_collate's shared memory mechanism in worker processes
    by manually stacking tensors, which prevents "Trying to resize storage that is not resizable" errors.
    """

    def convert_string_value(value):
        """Convert numpy string array or string scalar to Python string."""
        if isinstance(value, np.ndarray):
            # Check if it's a string array
            if value.dtype.kind in ["U", "S"]:  # Unicode or byte string
                # If it's a scalar (0-d array), convert to Python string
                if value.ndim == 0:
                    return str(value.item())
                # If it's an array, convert to list
                return value.tolist()
            # For non-string arrays, return as is (will be handled by default collate)
            return value
        elif isinstance(value, (str, bytes)):
            # Already a Python string/bytes
            return str(value) if isinstance(value, bytes) else value
        else:
            # For other types, return as is
            return value

    def collate_tensors(tensors):
        """Manually collate tensors without using shared memory."""
        if not tensors:
            return None
        # Convert numpy arrays to tensors with copy to ensure they own their data
        tensor_list = []
        for t in tensors:
            if isinstance(t, np.ndarray):
                # Copy array to ensure tensor owns its data (prevents resize errors)
                t = torch.from_numpy(np.array(t, copy=True))
            elif isinstance(t, torch.Tensor):
                # If already a tensor, clone if it might share storage
                if t.storage().is_shared():
                    t = t.clone()
            tensor_list.append(t)
        # Stack tensors manually (avoids shared memory path in default_collate)
        return torch.stack(tensor_list, dim=0)

    def collate_recursive(values):
        """Recursively collate nested structures."""
        if not values:
            return None

        first = values[0]

        # Handle numpy arrays and tensors
        if isinstance(first, (np.ndarray, torch.Tensor)):
            return collate_tensors(values)

        # Handle dictionaries
        if isinstance(first, dict):
            result = {}
            all_keys = set()
            for v in values:
                all_keys.update(v.keys())
            for k in all_keys:
                nested_values = [v.get(k) for v in values]
                result[k] = collate_recursive(nested_values)
            return result

        # Handle lists/tuples
        if isinstance(first, (list, tuple)):
            # Collate each position
            result = []
            for i in range(len(first)):
                nested_values = [v[i] for v in values]
                result.append(collate_recursive(nested_values))
            return type(first)(result)

        # For scalars, strings, etc., try default collate as fallback
        try:
            return torch.utils.data.dataloader.default_collate(values)
        except (RuntimeError, TypeError):
            # If default collate fails, just keep as list
            return values

    # Process each sample in the batch
    processed_batch = []
    for sample in batch:
        processed_sample = {}
        for key, value in sample.items():
            if key in ["tactile_sensor", "tactile_type"]:
                # tactile_sensor and tactile_type are dicts, convert each value (which may be numpy string)
                if isinstance(value, dict):
                    processed_sample[key] = {k: convert_string_value(v) for k, v in value.items()}
                else:
                    processed_sample[key] = convert_string_value(value)
            else:
                processed_sample[key] = value
        processed_batch.append(processed_sample)

    # Manually collate dictionaries to avoid shared memory issues
    result = {}
    if not processed_batch:
        return result

    # Get all keys from first sample
    sample_keys = processed_batch[0].keys()

    for key in sample_keys:
        values = [sample[key] for sample in processed_batch]

        if key in ["tactile_sensor", "tactile_type"]:
            # Handle string dicts manually
            if isinstance(values[0], dict):
                dicts = values
                all_keys = set()
                for d in dicts:
                    all_keys.update(d.keys())
                result[key] = {k: [d.get(k) for d in dicts] for k in all_keys}
            else:
                result[key] = values
        else:
            # Use recursive collation for all other types (tensors, nested dicts, etc.)
            result[key] = collate_recursive(values)

    return result


_RAW_JOINT_GLITCH_THRESH = 100.0  # rad — sensor glitches are typically orders of magnitude above ±π


def _replace_joint_glitches(joints: np.ndarray) -> np.ndarray:
    """Replace frames whose joint values exceed _RAW_JOINT_GLITCH_THRESH with the nearest clean frame.

    Handles isolated single-frame sensor glitches (e.g. values of 1e30) that appear in some
    datasets (e.g. RH20TCfg1OptoForce). Called before interpolation / norm-stats accumulation
    so the glitch frames never pollute training or normalization statistics.
    """
    bad_mask = np.any(~np.isfinite(joints) | (np.abs(joints) > _RAW_JOINT_GLITCH_THRESH), axis=-1)
    if not np.any(bad_mask):
        return joints
    joints = joints.copy()
    good_indices = np.where(~bad_mask)[0]
    if len(good_indices) == 0:
        logging.warning("_replace_joint_glitches: all frames are bad – returning zeros")
        return np.zeros_like(joints)
    for bad_idx in np.where(bad_mask)[0]:
        nearest = good_indices[np.argmin(np.abs(good_indices - bad_idx))]
        joints[bad_idx] = joints[nearest]
        logging.warning(
            "_replace_joint_glitches: frame %d has extreme joint value (max |v|=%.3e), replaced with frame %d",
            bad_idx,
            float(np.max(np.abs(joints[nearest]))),
            int(nearest),
        )
    return joints


_CORRUPT_NORM_STD_THRESH = 1000.0  # any std > 1000 is a data-corruption artifact
_CORRUPT_NORM_MEAN_THRESH = 1e6  # any |mean| > 1e6 is a data-corruption artifact


def _sanitize_loaded_norm_stats(norm_stats: dict | None, domain_name: str = "") -> None:
    """In-place: reset catastrophically wrong norm-stat dims to identity (mean=0, std=1).

    Isolated sensor glitches in raw data (e.g. a single frame with joint value 5e34)
    can cause the computed mean/std for that dimension to be astronomically large.
    This makes the stored JSON unusable until norm stats are recomputed.
    As a runtime safety net we detect those dims and reset them to a no-op normalization,
    which keeps training stable and prevents validation RMSE from blowing up.

    The fix is approximate – identity normalization is not ideal.  Recompute norm stats
    with the glitch-filtered dataset loader (_replace_joint_glitches) for correct values.
    """
    if norm_stats is None:
        return
    params = norm_stats.get("params", {})
    norm_dim = norm_stats.get("norm_dim", {})
    for key, p in params.items():
        if not isinstance(p, dict) or "std" not in p or "mean" not in p:
            continue
        std_arr = np.asarray(p["std"], dtype=np.float64)
        mean_arr = np.asarray(p["mean"], dtype=np.float64)
        nd = np.asarray(norm_dim.get(key, []))
        bad_mask = (
            ~np.isfinite(std_arr)
            | (np.abs(std_arr) > _CORRUPT_NORM_STD_THRESH)
            | ~np.isfinite(mean_arr)
            | (np.abs(mean_arr) > _CORRUPT_NORM_MEAN_THRESH)
        )
        if not np.any(bad_mask):
            continue
        bad_dims = nd[bad_mask].tolist() if len(nd) == len(std_arr) else np.where(bad_mask)[0].tolist()
        logging.warning(
            "Domain %s | Corrupt norm stats in '%s' at %d dim(s) %s "
            "(std or mean out of [−1e6, 1e6] / [0, 1000]). "
            "Resetting to identity (mean=0, std=1). "
            "Re-run compute_norm_stats with the latest code to fix permanently.",
            domain_name,
            key,
            int(np.sum(bad_mask)),
            bad_dims,
        )
        std_arr[bad_mask] = 1.0
        mean_arr[bad_mask] = 0.0
        p["std"] = std_arr.tolist()
        p["mean"] = mean_arr.tolist()


def norm_stats_to_jsonable(norm_stats: dict) -> dict:
    jsonable = norm_stats.copy()
    jsonable["params"] = _serialize_params_value(norm_stats["params"])
    return jsonable


def norm_stats_from_jsonable(json_stats: dict) -> dict:
    parsed = json_stats.copy()
    parsed["params"] = _deserialize_params_value(json_stats["params"])
    return parsed


def _build_old_norm_stats_path(
    assets_root: str | pathlib.Path,
    old_repo_id: str,
    relative_path: str | pathlib.Path,
) -> pathlib.Path:
    return pathlib.Path(assets_root) / old_repo_id / relative_path


def _log_norm_override_result(
    *,
    current_path: pathlib.Path,
    override_path: pathlib.Path,
    replaced_keys: list[str],
) -> None:
    if replaced_keys:
        preview = ", ".join(replaced_keys[:6])
        if len(replaced_keys) > 6:
            preview += ", ..."
        logging.info(
            "Loaded norm stats from %s and overrode %d keys using %s (%s)",
            current_path,
            len(replaced_keys),
            override_path,
            preview,
        )
    elif override_path.exists():
        logging.info("Loaded norm stats from %s with no overlapping override keys from %s", current_path, override_path)
    else:
        logging.info("Loaded norm stats from %s; override source missing at %s", current_path, override_path)


def _reshape_tactile_batch_for_grouped_norm(tactile_batch: torch.Tensor) -> torch.Tensor:
    """Flatten the tactile area axis into the batch axis for shared sensor+type+shape norm stats."""
    if tactile_batch.dim() < 4:
        raise ValueError(f"Expected tactile batch shape (B, T, N, *D), got {tuple(tactile_batch.shape)}")
    batch_size, time_steps, num_areas = tactile_batch.shape[:3]
    trailing_shape = tuple(tactile_batch.shape[3:])
    return tactile_batch.reshape(batch_size * num_areas, time_steps, *trailing_shape)


def _resolve_val_episode_count(total_pool_episodes: int, val_ratio: float) -> int:
    """Resolve requested val episode count from val_ratio.

    Compatibility:
    - val_ratio < 1: ratio mode (legacy behavior), floor(total * ratio).
    - val_ratio >= 1: fixed-count mode, int(val_ratio) episodes per norm domain.
    """
    total = int(total_pool_episodes)
    if total <= 0:
        return 0

    ratio = float(val_ratio)
    if ratio >= 1.0:
        requested = int(ratio)
    else:
        requested = int(total * ratio)
    return max(0, min(requested, total))


def _build_group_episode_splits(
    entry_episode_counts: list[int], val_ratio: float, seed: int
) -> list[tuple[list[int], list[int]]]:
    """Split pooled episodes into per-entry train/val episode indices.

    Episodes from all entries in the same norm_stats_domain_name are pooled first, then sampled
    globally using ``val_ratio``. The selected val episodes are mapped back to each entry so the
    val allocation is distributed across copy/partial datasets.
    """
    if any(count < 0 for count in entry_episode_counts):
        raise ValueError(f"Episode counts must be non-negative, got {entry_episode_counts}")

    total_episodes = int(sum(entry_episode_counts))
    if total_episodes == 0:
        return [([], []) for _ in entry_episode_counts]

    val_episodes = _resolve_val_episode_count(total_episodes, float(val_ratio))

    if val_episodes > 0:
        rng = np.random.default_rng(int(seed))
        val_global = np.sort(rng.choice(total_episodes, size=val_episodes, replace=False).astype(np.int64))
    else:
        val_global = np.empty((0,), dtype=np.int64)

    split_pairs: list[tuple[list[int], list[int]]] = []
    start = 0
    for count in entry_episode_counts:
        end = start + int(count)
        all_local = np.arange(count, dtype=np.int64)
        local_mask = (val_global >= start) & (val_global < end)
        val_local = (val_global[local_mask] - start).astype(np.int64)
        if val_local.size == 0:
            train_local = all_local
        elif val_local.size >= count:
            train_local = np.empty((0,), dtype=np.int64)
        else:
            train_local = np.setdiff1d(all_local, val_local, assume_unique=True)
        split_pairs.append((train_local.tolist(), val_local.tolist()))
        start = end

    return split_pairs


def _canonical_dataset_source_id(dataset_path: str) -> str:
    return os.path.realpath(os.path.abspath(os.path.expanduser(str(dataset_path))))


def _build_group_episode_splits_by_source(
    entry_source_ids: list[str],
    entry_episode_counts: list[int],
    source_pool_episode_counts: dict[str, int],
    val_ratio: float,
    seed: int,
) -> list[tuple[list[int], list[int]]]:
    """Build per-entry train/val splits while deduplicating shared source episodes.

    Splitting is performed over unique source episodes (source_id + raw episode index), then
    projected back to each entry's selected prefix. This guarantees no source episode appears
    in train for one entry and val for another entry under the same norm domain.
    """
    if len(entry_source_ids) != len(entry_episode_counts):
        raise ValueError(
            f"Mismatched lengths: len(entry_source_ids)={len(entry_source_ids)} "
            f"!= len(entry_episode_counts)={len(entry_episode_counts)}"
        )

    if any(count < 0 for count in entry_episode_counts):
        raise ValueError(f"Entry episode counts must be non-negative, got {entry_episode_counts}")

    total_pool_episodes = int(sum(source_pool_episode_counts.values()))
    if total_pool_episodes == 0:
        return [([], []) for _ in entry_episode_counts]

    for source_id, entry_count in zip(entry_source_ids, entry_episode_counts, strict=True):
        if source_id not in source_pool_episode_counts:
            raise ValueError(f"source_id {source_id!r} missing in source_pool_episode_counts")
        if int(entry_count) > int(source_pool_episode_counts[source_id]):
            raise ValueError(
                f"Entry count {entry_count} exceeds source pool count "
                f"{source_pool_episode_counts[source_id]} for source_id={source_id!r}"
            )

    val_episodes = _resolve_val_episode_count(total_pool_episodes, float(val_ratio))

    if val_episodes > 0:
        rng = np.random.default_rng(int(seed))
        val_global = np.sort(rng.choice(total_pool_episodes, size=val_episodes, replace=False).astype(np.int64))
    else:
        val_global = np.empty((0,), dtype=np.int64)

    source_val_indices: dict[str, np.ndarray] = {}
    offset = 0
    for source_id, source_count in source_pool_episode_counts.items():
        source_count = int(source_count)
        source_start = offset
        source_end = source_start + source_count
        local_mask = (val_global >= source_start) & (val_global < source_end)
        source_val_indices[source_id] = (val_global[local_mask] - source_start).astype(np.int64)
        offset = source_end

    split_pairs: list[tuple[list[int], list[int]]] = []
    for source_id, entry_count in zip(entry_source_ids, entry_episode_counts, strict=True):
        entry_count = int(entry_count)
        all_local = np.arange(entry_count, dtype=np.int64)
        source_val = source_val_indices[source_id]
        if source_val.size == 0:
            val_local = np.empty((0,), dtype=np.int64)
            train_local = all_local
        else:
            val_local = source_val[source_val < entry_count]
            if val_local.size == 0:
                train_local = all_local
            elif val_local.size >= entry_count:
                train_local = np.empty((0,), dtype=np.int64)
            else:
                train_local = np.setdiff1d(all_local, val_local, assume_unique=True)
        split_pairs.append((train_local.tolist(), val_local.tolist()))

    return split_pairs


def _build_group_episode_splits_by_source_positions(
    entry_source_ids: list[str],
    entry_selected_source_positions: list[np.ndarray],
    source_pool_positions: dict[str, np.ndarray],
    val_ratio: float,
    seed: int,
    entry_use_trajectory_ratios: list[float] | None = None,
) -> list[tuple[list[int], list[int]]]:
    """Build per-entry train/val splits from explicit source-episode identities.

    Each source episode is identified by ``(source_id, source_episode_position)``. We first
    sample validation identities from the union pool, then project them back to each
    entry's local selected-episode ordering.
    """
    if len(entry_source_ids) != len(entry_selected_source_positions):
        raise ValueError(
            f"Mismatched lengths: len(entry_source_ids)={len(entry_source_ids)} "
            f"!= len(entry_selected_source_positions)={len(entry_selected_source_positions)}"
        )
    if entry_use_trajectory_ratios is not None and len(entry_use_trajectory_ratios) != len(entry_source_ids):
        raise ValueError(
            f"Mismatched lengths: len(entry_use_trajectory_ratios)={len(entry_use_trajectory_ratios)} "
            f"!= len(entry_source_ids)={len(entry_source_ids)}"
        )

    normalized_pool_positions: dict[str, np.ndarray] = {}
    for source_id, pool_positions in source_pool_positions.items():
        pool_array = np.asarray(pool_positions, dtype=np.int64)
        if pool_array.ndim != 1:
            raise ValueError(f"Pool positions for source_id={source_id!r} must be 1-D")
        if pool_array.size > 0 and np.any(pool_array < 0):
            raise ValueError(f"Pool positions for source_id={source_id!r} must be non-negative")
        normalized_pool_positions[source_id] = np.unique(pool_array)

    pool_position_sets = {
        source_id: set(pool_positions.tolist()) for source_id, pool_positions in normalized_pool_positions.items()
    }

    normalized_entry_positions: list[np.ndarray] = []
    for source_id, selected_positions in zip(entry_source_ids, entry_selected_source_positions, strict=True):
        if source_id not in normalized_pool_positions:
            raise ValueError(f"source_id {source_id!r} missing in source_pool_positions")
        entry_array = np.asarray(selected_positions, dtype=np.int64)
        if entry_array.ndim != 1:
            raise ValueError("Each entry selected source positions must be 1-D")
        if entry_array.size > 0 and np.any(entry_array < 0):
            raise ValueError("Entry selected source positions must be non-negative")
        if any(int(pos) not in pool_position_sets[source_id] for pos in entry_array.tolist()):
            raise ValueError(f"Entry selected positions are not a subset of source pool for source_id={source_id!r}")
        normalized_entry_positions.append(entry_array)

    pool_identities: list[tuple[str, int]] = []
    for source_id, pool_positions in normalized_pool_positions.items():
        for source_position in pool_positions.tolist():
            pool_identities.append((source_id, int(source_position)))

    total_pool_episodes = len(pool_identities)
    if total_pool_episodes == 0:
        return [([], []) for _ in entry_source_ids]

    val_ratio_float = float(val_ratio)
    val_episodes = _resolve_val_episode_count(total_pool_episodes, val_ratio_float)
    fixed_count_mode = val_ratio_float >= 1.0

    source_val_positions: dict[str, set[int]] = {source_id: set() for source_id in normalized_pool_positions}
    if val_episodes > 0:
        rng = np.random.default_rng(int(seed))
        sampled_pool_indices = rng.choice(total_pool_episodes, size=val_episodes, replace=False).astype(np.int64)
        for sampled_idx in sampled_pool_indices.tolist():
            source_id, source_position = pool_identities[int(sampled_idx)]
            source_val_positions[source_id].add(source_position)

    # In fixed-count mode, each selected val source-episode should be evaluated only once
    # across copy/partial entries within the same norm domain. We therefore assign a single
    # owner entry for each val identity using highest use_trajectory_ratio first, then
    # dataset-config order for tie-break.
    owner_entry_by_identity: dict[tuple[str, int], int] = {}
    if fixed_count_mode:
        if entry_use_trajectory_ratios is None:
            entry_use_trajectory_ratios = [0.0] * len(entry_source_ids)

        per_entry_local_idx: list[dict[int, int]] = []
        for entry_array in normalized_entry_positions:
            local_map: dict[int, int] = {}
            for local_idx, source_position in enumerate(entry_array.tolist()):
                local_map[int(source_position)] = int(local_idx)
            per_entry_local_idx.append(local_map)

        for source_id, val_positions in source_val_positions.items():
            for source_position in val_positions:
                candidates = [
                    entry_idx
                    for entry_idx, entry_source_id in enumerate(entry_source_ids)
                    if entry_source_id == source_id and int(source_position) in per_entry_local_idx[entry_idx]
                ]
                if not candidates:
                    raise ValueError(
                        f"Val identity ({source_id!r}, {int(source_position)}) has no candidate entry to own it."
                    )
                owner_entry = max(candidates, key=lambda idx: (float(entry_use_trajectory_ratios[idx]), -idx))
                owner_entry_by_identity[(source_id, int(source_position))] = int(owner_entry)

    split_pairs: list[tuple[list[int], list[int]]] = []
    for entry_idx, (source_id, entry_array) in enumerate(
        zip(entry_source_ids, normalized_entry_positions, strict=True)
    ):
        train_local: list[int] = []
        val_local: list[int] = []
        source_val = source_val_positions[source_id]
        for local_idx, source_position in enumerate(entry_array.tolist()):
            source_position_int = int(source_position)
            if source_position_int not in source_val:
                train_local.append(local_idx)
                continue

            if fixed_count_mode:
                owner_entry = owner_entry_by_identity[(source_id, source_position_int)]
                if owner_entry == entry_idx:
                    val_local.append(local_idx)
                # else: drop from this copy/partial entry to avoid duplicated validation.
            else:
                # Legacy behavior: all entries containing a selected source episode include it in val.
                val_local.append(local_idx)
        split_pairs.append((train_local, val_local))

    return split_pairs


def _dataset_entry_init_cache_key(entry: dict) -> tuple[str, float, str]:
    return (
        str(entry["path"]),
        float(entry["use_trajectory_ratio"]),
        str(entry["norm_stats_domain_name"]),
    )


def _dedupe_dataset_entries_for_init(dataset_entries: list[dict]) -> tuple[list[dict], dict[int, int]]:
    """Deduplicate entries that would produce identical ZarrDataset initialization.

    Returns:
        unique_entries: first-seen representative entries.
        entry_to_unique_idx: mapping from original entry index to representative index.
    """
    unique_entries: list[dict] = []
    unique_idx_by_key: dict[tuple[str, float, str], int] = {}
    entry_to_unique_idx: dict[int, int] = {}

    for idx, entry in enumerate(dataset_entries):
        cache_key = _dataset_entry_init_cache_key(entry)
        if cache_key not in unique_idx_by_key:
            unique_idx_by_key[cache_key] = len(unique_entries)
            unique_entries.append(entry)
        entry_to_unique_idx[idx] = unique_idx_by_key[cache_key]

    return unique_entries, entry_to_unique_idx


def _format_preview_list(values: list[str], max_items: int = 8) -> str:
    if len(values) <= max_items:
        return str(values)
    head_count = max_items // 2
    tail_count = max_items - head_count
    head = values[:head_count]
    tail = values[-tail_count:]
    return f"{head + ['...'] + tail} (total={len(values)})"


def get_replay_buffer(dataset_path, cache_dir):
    if dataset_path is None:
        return None
    if cache_dir is None:
        replay_buffer = ReplayBuffer.create_from_path(zarr_path=dataset_path, mode="r")
    else:
        # determine path name
        mod_time = os.path.getmtime(dataset_path)
        stamp = datetime.fromtimestamp(mod_time).isoformat()
        stem_name = os.path.basename(dataset_path).split(".")[0]
        cache_name = "_".join([stem_name, stamp])
        cache_dir = pathlib.Path(os.path.expanduser(cache_dir))
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir.joinpath(cache_name + ".zarr.mdb")
        lock_path = cache_dir.joinpath(cache_name + ".lock")

        # load cached file
        print("Acquiring lock on cache.")
        with FileLock(lock_path):
            # cache does not exist
            if not cache_path.exists():
                try:
                    with zarr.LMDBStore(
                        str(cache_path), writemap=True, metasync=False, sync=False, map_async=True, lock=False
                    ) as lmdb_store:
                        print(f"Copying data to {str(cache_path)}")
                        ReplayBuffer.copy_from_path(zarr_path=dataset_path, store=lmdb_store, compressors="disk")
                    print("Cache written to disk!")
                except Exception as e:
                    shutil.rmtree(cache_path)
                    raise e

        # open read-only lmdb store
        store = zarr.LMDBStore(str(cache_path), readonly=True, lock=False)
        replay_buffer = ReplayBuffer.create_from_group(group=zarr.group(store))
    return replay_buffer


def get_replay_buffer_list(dataset_path, cache_dir):
    dataset_path_list_tmp = os.listdir(dataset_path)
    dataset_path_list = []
    for data_p in dataset_path_list_tmp:
        if data_p.endswith(".zarr"):
            dataset_path_list.append(data_p)
    replay_buffer_list = []
    for data_p in dataset_path_list:
        # print(data_p)
        replay_buffer_list.append(get_replay_buffer(os.path.join(dataset_path, data_p), cache_dir))
    return replay_buffer_list, dataset_path_list


def get_instruction_from_filename_list(filename_list):
    instruction_list = []
    for filename in filename_list:
        # instruction位于两个+中间，例如XXXX+instruction+XXXX
        if "+" in filename and filename.find("+") > 0:
            instruction = filename[filename.find("+") + 1 : filename.rfind("+")]
            instruction = instruction.strip()
            instruction = instruction.replace("_", " ")
            if not instruction.endswith("."):
                instruction += "."
            if "+" in instruction:
                raise ValueError(f"Filename {filename} contains multiple instructions between + signs.")
            if len(instruction) > 0:
                instruction_list.append(instruction)
            else:
                raise ValueError(f"Filename {filename} contains empty instruction between + signs.")
        else:
            raise ValueError(f"Filename {filename} does not contain instruction in [ ] format.")
    return instruction_list


T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class ZarrDataset(Dataset):
    def __init__(
        self,
        data_config,
        dataset_path,
        domain_name,
        action_horizon: int,
        split="train",
        use_trajectory_ratio: float = 1.0,
        norm_stats_domain_name: str = None,
    ):
        replay_buffer_list, zarr_file_list = get_replay_buffer_list(dataset_path=dataset_path, cache_dir=None)
        self.zarr_file_list = zarr_file_list
        # Use norm_stats_domain_name to share a single set of norm stats across copies of the same source dataset.
        _norm_dir = norm_stats_domain_name if norm_stats_domain_name else domain_name
        self.norm_stats_domain_name = _norm_dir
        self.assets_dir = os.path.join(data_config.assets_dirs, data_config.repo_id, _norm_dir)
        self.domain_name = domain_name
        self._split_group_entries_cache = None
        self.dataset_path = dataset_path
        self.use_trajectory_ratio = use_trajectory_ratio
        os.makedirs(self.assets_dir, exist_ok=True)
        self.ignore_rgb = False

        self.data_config = data_config
        self.single_arm_action_rep_dim = data_config.single_arm_action_rep_dim
        self.reserved_action_dim = data_config.reserved_action_dim
        self.single_hand_num_tactile_areas = data_config.single_hand_num_tactile_areas

        self.disable_history = bool(getattr(data_config, "disable_history", True))
        self.image_down_sample_steps = list(getattr(data_config, "image_down_sample_steps", []))
        self.state_down_sample_steps = list(getattr(data_config, "state_down_sample_steps", []))
        self.tactile_down_sample_steps = list(getattr(data_config, "tactile_down_sample_steps", []))
        if self.disable_history:
            nonempty_history = {
                "image_down_sample_steps": self.image_down_sample_steps,
                "state_down_sample_steps": self.state_down_sample_steps,
                "tactile_down_sample_steps": self.tactile_down_sample_steps,
            }
            configured_history = {name: value for name, value in nonempty_history.items() if len(value) > 0}
            if configured_history:
                raise ValueError(
                    "disable_history=True requires history=1 for all modalities, so "
                    "`image_down_sample_steps`, `state_down_sample_steps`, and "
                    "`tactile_down_sample_steps` must all be empty. "
                    f"Got: {configured_history}"
                )
        self.image_hisory_length = len(self.image_down_sample_steps) + 1
        self.state_hisory_length = len(self.state_down_sample_steps) + 1
        self.tactile_hisory_length = len(self.tactile_down_sample_steps) + 1
        self.action_horizon = action_horizon
        self.action_down_sample_steps = data_config.action_down_sample_steps

        self.proprioception_pose_rep = data_config.proprioception_pose_rep
        self.action_pose_rep = data_config.action_pose_rep
        self.proprioception_joint_rep = data_config.proprioception_joint_rep
        self.action_joint_rep = data_config.action_joint_rep
        self.use_domain_condition = data_config.use_domain_condition
        self.non_tactile_dropout_ratio = float(getattr(data_config, "non_tactile_dropout_ratio", 0.0))
        self.contact_detection_threshold_k = float(
            getattr(data_config, "contact_detection_threshold_k", DEFAULT_CONTACT_THRESHOLD_K)
        )
        # Image-type tactile normalization uses incremental channel statistics.
        # These two controls are sampling budgets (not memory allocation):
        # - pool_max_size: max channel vectors processed per tactile key (0 = unlimited)
        # - batch_samples: max channel vectors sampled per key per batch (0 = all)
        self.norm_image_channel_pool_max_size = int(
            max(0, getattr(data_config, "norm_image_channel_pool_max_size", 200000))
        )
        self.norm_image_channel_batch_samples = int(
            max(0, getattr(data_config, "norm_image_channel_batch_samples", 4096))
        )
        self.norm_image_tactile_mode = getattr(data_config, "norm_image_tactile_mode", "channel_wise")
        if self.norm_image_tactile_mode not in {"channel_wise", "image_norm"}:
            raise ValueError(
                f"Invalid norm_image_tactile_mode: {self.norm_image_tactile_mode}. "
                "Please choose from ['channel_wise', 'image_norm']"
            )

        self._get_all_episodes(replay_buffer_list, use_trajectory_ratio=use_trajectory_ratio)

        self.datas = [replay_buffer.data for replay_buffer in replay_buffer_list]
        self._cached_data_keys = []
        self._cached_data_key_sets = []
        for data in self.datas:
            keys = tuple(data.keys())
            self._cached_data_keys.append(keys)
            self._cached_data_key_sets.append(set(keys))

        if data_config.create_train_val_split:
            assert data_config.use_val_dataset
        if data_config.create_train_val_split and split == "train":
            self.create_train_val_split()
        self.get_split_indices(split)
        del replay_buffer_list

        self.skip_normalization = False
        self.norm_stats = None
        self._contact_thresholds = None
        self._contact_thresholds_loaded = False
        self.indices_org = copy.deepcopy(self.indices)

    def get_split_indices(self, split):
        if self.data_config.use_val_dataset:
            self.indices = self.get_indices(split)
        else:
            self.indices = list(range(self.episode_idx_range[0], self.episode_idx_range[-1]))

    def _get_all_episodes(self, replay_buffer_list, use_trajectory_ratio: float = 1.0) -> List:
        """获取所有episode的数据

        Args:
            replay_buffer_list: List of replay buffers (zarr files)
            use_trajectory_ratio: Fixed ratio of trajectories to use from each zarr file (takes first N trajectories)
        """
        if use_trajectory_ratio <= 0.0 or use_trajectory_ratio > 1.0:
            raise ValueError(f"use_trajectory_ratio must be in (0, 1], got {use_trajectory_ratio}")

        episodes = []
        data_idxs = []
        ep_to_zarr = []
        episode_idx_range = [0]
        starts = []
        ends = []

        n_episodes = 0
        episode_to_zarr = {}  # episode_idx -> zarr_idx mapping for per-zarr split
        for idx_ep, replay_buffer in enumerate(tqdm(replay_buffer_list)):
            num_episodes_total = int(replay_buffer.episode_ends.shape[0])
            # Apply use_trajectory_ratio: take first N trajectories from this zarr file
            num_episodes_to_use = int(num_episodes_total * use_trajectory_ratio)
            data = replay_buffer.data
            has_left_used = "lefthand_used" in data
            has_right_used = "righthand_used" in data

            for idx in range(num_episodes_to_use):
                eps_start = 0 if idx == 0 else replay_buffer.episode_ends[idx - 1]
                eps_end = replay_buffer.episode_ends[idx]
                frame_indices = np.arange(eps_start, eps_end)

                # If both hands are unused at a frame, drop that frame (when flags exist)
                if has_left_used and has_right_used:
                    left_used = data["lefthand_used"][eps_start:eps_end]
                    right_used = data["righthand_used"][eps_start:eps_end]
                    keep_mask = np.logical_or(left_used, right_used)
                    frame_indices = frame_indices[keep_mask]

                data_idxs.extend(frame_indices.tolist())
                episodes.extend([n_episodes] * frame_indices.shape[0])
                # Record which zarr this episode belongs to
                episode_to_zarr[n_episodes] = idx_ep
                n_episodes += 1
                starts.extend([eps_start] * frame_indices.shape[0])
                ends.extend([eps_end] * frame_indices.shape[0])
                ep_to_zarr.extend([idx_ep] * frame_indices.shape[0])
                episode_idx_range.append(episode_idx_range[-1] + frame_indices.shape[0])

        self.start_frames = starts  # the detailed start idx inside zarr for data[i]
        self.end_frames = ends  # the detailed end idx inside zarr for data[i]
        self.zarr_idxs = ep_to_zarr  # the idxs of zarr for data[i]
        self.episodes_idxs = episodes  # the (global) episode idx for data[i]
        self.data_idxs = data_idxs  # the detailed idx inside zarr for data[i]
        self.n_episodes = len(episodes)
        self.episode_idx_range = episode_idx_range  # cumulative sum of episode lengths, used for indexing
        self.episode_to_zarr = episode_to_zarr  # episode_idx -> zarr_idx mapping for per-zarr split

    def _legacy_split_keys_for_ratio(self, use_trajectory_ratio: float) -> tuple[str, str]:
        if use_trajectory_ratio >= 1.0:
            return "train_episode_idx", "val_episode_idx"
        tag = f"_r{use_trajectory_ratio:.4f}"
        return f"train_episode_idx{tag}", f"val_episode_idx{tag}"

    def _get_norm_domain_group_entries(self) -> list[dict]:
        if self._split_group_entries_cache is not None:
            return self._split_group_entries_cache

        dataset_config_path = getattr(self.data_config, "dataset_config_path", "")
        if not dataset_config_path:
            self._split_group_entries_cache = []
            return self._split_group_entries_cache

        config_path = pathlib.Path(dataset_config_path)
        if not config_path.exists():
            self._split_group_entries_cache = []
            return self._split_group_entries_cache

        with open(config_path, "r") as f:
            dataset_config = json.load(f)

        default_ratio = float(dataset_config.get("default_use_trajectory_ratio", 1.0))
        target_norm_domain = self.norm_stats_domain_name
        group_entries = []
        for ds in dataset_config.get("datasets", []):
            if not ds.get("enabled", True):
                continue
            dataset_path = ds.get("path")
            if not dataset_path:
                continue
            dataset_name = ds.get("name", pathlib.Path(dataset_path).name)
            norm_domain = ds.get("norm_stats_domain_name", dataset_name)
            if norm_domain != target_norm_domain:
                continue
            group_entries.append(
                {
                    "name": dataset_name,
                    "path": dataset_path,
                    "use_trajectory_ratio": float(ds.get("use_trajectory_ratio", default_ratio)),
                }
            )

        self._split_group_entries_cache = group_entries
        return self._split_group_entries_cache

    def _resolve_current_split_entry(self, group_entries: list[dict]) -> dict | None:
        if not group_entries:
            return None

        by_name = [entry for entry in group_entries if entry["name"] == self.domain_name]
        if len(by_name) == 1:
            return by_name[0]

        by_path_and_ratio = [
            entry
            for entry in group_entries
            if entry["path"] == self.dataset_path
            and math.isclose(
                float(entry["use_trajectory_ratio"]),
                float(self.use_trajectory_ratio),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ]
        if len(by_path_and_ratio) == 1:
            return by_path_and_ratio[0]

        return None

    def _build_split_copy_id(self, entry: dict) -> str:
        safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", str(entry["name"])).strip("_") or "dataset"
        digest = hashlib.sha1(
            f"{entry['path']}|{float(entry['use_trajectory_ratio']):.8f}".encode("utf-8")
        ).hexdigest()[:10]
        return f"{safe_name}_{digest}"

    def _split_keys_for_entry(self, entry: dict, total_entries: int) -> tuple[str, str]:
        if total_entries <= 1:
            return self._legacy_split_keys_for_ratio(float(entry["use_trajectory_ratio"]))
        copy_id = self._build_split_copy_id(entry)
        return f"train_episode_idx_{copy_id}", f"val_episode_idx_{copy_id}"

    def _count_episodes_per_zarr_file(self, dataset_path: str) -> list[int]:
        replay_buffer_list, _ = get_replay_buffer_list(dataset_path=dataset_path, cache_dir=None)
        return [int(replay_buffer.episode_ends.shape[0]) for replay_buffer in replay_buffer_list]

    @staticmethod
    def _build_selected_source_episode_positions(
        episode_counts_per_zarr_file: list[int], use_trajectory_ratio: float
    ) -> np.ndarray:
        selected_positions: list[int] = []
        offset = 0
        for count in episode_counts_per_zarr_file:
            count_int = int(count)
            if count_int < 0:
                raise ValueError(f"Episode count must be non-negative, got {count_int}")
            selected_count = int(count_int * float(use_trajectory_ratio))
            if selected_count > 0:
                selected_positions.extend(range(offset, offset + selected_count))
            offset += count_int
        return np.array(selected_positions, dtype=np.int64)

    def create_train_val_split(self):
        """Create train/val split for one norm_stats_domain_name group.

        Splitting is done on unique source episodes (source path + raw episode index), then
        projected to each copy/partial entry by its exact selected source-episode positions.
        This guarantees train/val isolation across copy datasets that share the same source data
        while matching per-zarr-file ratio flooring used during dataset loading.
        """
        split_file_path = self._get_split_file_path()

        group_entries = self._get_norm_domain_group_entries()
        current_entry = self._resolve_current_split_entry(group_entries)
        if not group_entries or current_entry is None:
            # Fallback for ad-hoc construction without a usable dataset_config_path.
            group_entries = [
                {
                    "name": self.domain_name,
                    "path": self.dataset_path,
                    "use_trajectory_ratio": float(self.use_trajectory_ratio),
                }
            ]

        required_keys = []
        for entry in group_entries:
            train_key, val_key = self._split_keys_for_entry(entry, len(group_entries))
            required_keys.extend([train_key, val_key])

        existing = {}
        if os.path.exists(split_file_path):
            with open(split_file_path, "r") as f:
                existing = json.load(f)

        if existing.get("_split_strategy_version") == SPLIT_STRATEGY_VERSION and all(
            key in existing for key in required_keys
        ):
            print(
                f"train_val_split.json already exists at {split_file_path}, skipping creation to preserve existing split."
            )
            return

        episode_counts_per_zarr_file_cache: dict[str, list[int]] = {}
        source_pool_positions: dict[str, np.ndarray] = {}
        entry_source_ids: list[str] = []
        entry_selected_source_positions: list[np.ndarray] = []

        for entry in group_entries:
            source_id = _canonical_dataset_source_id(entry["path"])
            if source_id not in episode_counts_per_zarr_file_cache:
                episode_counts_per_zarr_file_cache[source_id] = self._count_episodes_per_zarr_file(entry["path"])

            selected_source_positions = self._build_selected_source_episode_positions(
                episode_counts_per_zarr_file_cache[source_id],
                float(entry["use_trajectory_ratio"]),
            )

            entry_source_ids.append(source_id)
            entry_selected_source_positions.append(selected_source_positions)

            if source_id not in source_pool_positions:
                source_pool_positions[source_id] = selected_source_positions
            else:
                source_pool_positions[source_id] = np.union1d(
                    source_pool_positions[source_id], selected_source_positions
                )

        split_pairs = _build_group_episode_splits_by_source_positions(
            entry_source_ids=entry_source_ids,
            entry_selected_source_positions=entry_selected_source_positions,
            source_pool_positions=source_pool_positions,
            val_ratio=float(self.data_config.val_ratio),
            seed=int(self.data_config.seed),
            entry_use_trajectory_ratios=[float(entry["use_trajectory_ratio"]) for entry in group_entries],
        )

        os.makedirs(self.assets_dir, exist_ok=True)
        for entry, (train_episode_idx, val_episode_idx) in zip(group_entries, split_pairs, strict=True):
            train_key, val_key = self._split_keys_for_entry(entry, len(group_entries))
            existing[train_key] = train_episode_idx
            existing[val_key] = val_episode_idx

        existing["_split_strategy_version"] = SPLIT_STRATEGY_VERSION
        existing["_split_strategy"] = "norm_stats_domain_name_unique_source_episode"

        with open(split_file_path, "w") as f:
            json.dump(existing, f)
        print(f"Created new train_val_split file at {split_file_path}")

    def _split_keys(self):
        group_entries = self._get_norm_domain_group_entries()
        current_entry = self._resolve_current_split_entry(group_entries)
        if current_entry is None:
            return self._legacy_split_keys_for_ratio(float(self.use_trajectory_ratio))
        return self._split_keys_for_entry(current_entry, len(group_entries))

    def get_indices(self, split):
        split_file_path = self._get_split_file_path()
        legacy_split_file_path = os.path.join(self.assets_dir, "train_val_split.json")
        if not os.path.exists(split_file_path):
            if split_file_path != legacy_split_file_path and os.path.exists(legacy_split_file_path):
                print(
                    f"Bound split file not found at {split_file_path}. "
                    f"Falling back to legacy split file: {legacy_split_file_path}"
                )
                split_file_path = legacy_split_file_path
            else:
                raise ValueError(
                    f"Split file not found: {split_file_path}. Please run with --create_train_val_split to generate it."
                )

        train_key, val_key = self._split_keys()
        split_key = train_key if split == "train" else val_key

        with open(split_file_path, "r") as f:
            data = json.load(f)

        if split_key not in data:
            legacy_train_key, legacy_val_key = self._legacy_split_keys_for_ratio(float(self.use_trajectory_ratio))
            legacy_split_key = legacy_train_key if split == "train" else legacy_val_key
            if legacy_split_key in data:
                split_key = legacy_split_key
            elif f"{split}_episode_idx" in data:
                # Compatibility with very old split files.
                split_key = f"{split}_episode_idx"
            else:
                raise ValueError(
                    f"Split key '{split_key}' not found in {split_file_path}. Available keys: {sorted(data.keys())}"
                )

        split_idx = data[split_key]
        indices = []
        for episode_idx in split_idx:
            start_idx = self.episode_idx_range[episode_idx]
            end_idx = self.episode_idx_range[episode_idx + 1]
            indices.extend(list(range(start_idx, end_idx)))
        return indices

    def _get_split_file_path(self) -> str:
        """Resolve split path based on compatibility/binding toggle."""
        if not getattr(self.data_config, "bind_train_val_split_to_dataset_config", False):
            # All copies sharing the same norm_stats_domain_name (same assets_dir) write into
            # a single train_val_split.json, with per-copy keys handled by _split_keys().
            return os.path.join(self.assets_dir, "train_val_split.json")

        dataset_config_path = getattr(self.data_config, "dataset_config_path", "")
        if not dataset_config_path:
            return os.path.join(self.assets_dir, "train_val_split.json")

        config_path = pathlib.Path(dataset_config_path).expanduser()
        if not config_path.is_absolute():
            config_path = config_path.resolve()
        config_path_str = str(config_path)

        config_sha1 = "missing"
        if config_path.exists():
            with open(config_path, "rb") as f:
                config_sha1 = hashlib.sha1(f.read()).hexdigest()

        split_binding = {
            "dataset_config_path": config_path_str,
            "dataset_config_sha1": config_sha1,
            "norm_stats_domain_name": self.norm_stats_domain_name,
            "val_ratio": float(self.data_config.val_ratio),
            "seed": int(self.data_config.seed),
        }
        binding_key = json.dumps(split_binding, ensure_ascii=True, sort_keys=True)
        split_id = hashlib.sha1(binding_key.encode("utf-8")).hexdigest()[:12]
        return os.path.join(self.assets_dir, f"train_val_split_{split_id}.json")

    def get_val_dataset(self):
        val_set = copy.copy(self)
        val_set.indices = self.get_indices("val")
        val_set.indices_org = copy.deepcopy(val_set.indices)
        return val_set

    def __len__(self):
        return len(self.indices)

    def _get_contact_thresholds_path(self) -> str:
        return os.path.join(self.assets_dir, CONTACT_THRESHOLD_FILENAME)

    def _get_contact_thresholds_override_path(self) -> pathlib.Path | None:
        old_repo_id = getattr(self.data_config, "reuse_norm_repo_id", "")
        if not old_repo_id:
            return None
        return _build_old_norm_stats_path(
            self.data_config.assets_dirs,
            old_repo_id,
            pathlib.Path(self.domain_name) / CONTACT_THRESHOLD_FILENAME,
        )

    def _load_contact_thresholds(self) -> dict | None:
        if self._contact_thresholds_loaded:
            return self._contact_thresholds

        threshold_path = pathlib.Path(self._get_contact_thresholds_path())
        if not threshold_path.exists():
            raise ValueError(
                f"Domain {self.domain_name} | Contact thresholds not found at {threshold_path}. "
                f"Please run scripts/zarr_compute_norm_stats.py to generate {CONTACT_THRESHOLD_FILENAME} "
                "for the same repo_id used at train/eval time."
            )

        override_path = self._get_contact_thresholds_override_path()
        contact_thresholds_json, replaced_keys = load_contact_thresholds_jsonable_with_override(
            threshold_path,
            override_path,
        )
        if override_path is not None:
            if replaced_keys:
                logging.info(
                    "Loaded contact thresholds from %s and overrode %d streams using %s",
                    threshold_path,
                    len(replaced_keys),
                    override_path,
                )
            elif override_path.exists():
                logging.info(
                    "Loaded contact thresholds from %s with no overlapping override streams from %s",
                    threshold_path,
                    override_path,
                )
            else:
                logging.info(
                    "Loaded contact thresholds from %s; override source missing at %s",
                    threshold_path,
                    override_path,
                )

        self._contact_thresholds = contact_thresholds_json
        self._contact_thresholds_loaded = True
        return self._contact_thresholds

    def _get_stream_contact_threshold(self, stream_key: str) -> float | None:
        contact_thresholds = self._load_contact_thresholds()
        return require_stream_contact_threshold(
            contact_thresholds,
            stream_key,
            domain_name=self.domain_name,
            threshold_path=self._get_contact_thresholds_path(),
        )

    def _frame_has_contact_for_hand(
        self,
        data,
        data_keys,
        frame_idx: int,
        episode_start_idx: int,
        hand_side: str | None = None,
    ) -> bool:
        if frame_idx <= episode_start_idx:
            return False

        for key in data_keys:
            if not is_tactile_data_key(key):
                continue
            if hand_side is not None and not key.startswith(f"{hand_side}_tactile_data_"):
                continue

            threshold = self._get_stream_contact_threshold(key)
            score = compute_tactile_delta_score(data[key][frame_idx - 1], data[key][frame_idx])
            if is_contact_score_above_threshold(score, threshold):
                return True

        return False

    def _get_single_arm_data(
        self,
        data,
        data_keys,
        data_key_set,
        hand_side,
        state_target_idx,
        tactile_target_idx,
        action_idx_slice,
        start_idx,
        end_idx,
        cam_proj,
        *,
        compute_contact: bool,
    ):
        fast_no_history = (
            self.disable_history
            and self.state_hisory_length == 1
            and self.tactile_hisory_length == 1
        )
        action_mask = np.zeros((action_idx_slice.shape[0], self.single_arm_action_rep_dim))

        flag_no_data = f"{hand_side}_hand_joints" not in data_key_set
        if flag_no_data:
            blank_states = np.zeros((state_target_idx.shape[0], self.single_arm_action_rep_dim))
            blank_actions = np.zeros((action_idx_slice.shape[0], self.single_arm_action_rep_dim))
            blank_tactile_inputs = dict()
            blank_tactile_function_area_inputs = dict()
            blank_tactile_sensor_inputs = dict()
            blank_tactile_type_inputs = dict()
            return (
                blank_states,
                blank_actions,
                blank_tactile_inputs,
                blank_tactile_function_area_inputs,
                blank_tactile_sensor_inputs,
                blank_tactile_type_inputs,
                False,
                action_mask,
            )

        # Calculate interpolation range for state separately
        state_interpolation_start = max(int(state_target_idx[0]) - 5, start_idx)
        state_interpolation_end = min(int(state_target_idx[-1]) + 2 + 5, end_idx)

        # ================================ Get Wrist Pose #================================
        # Optimize: Read full pose array once, then slice columns (reduces zarr access from 2 to 1)
        if f"{hand_side}_wrist_pose" in data_key_set:
            if fast_no_history:
                obs_pose = data[f"{hand_side}_wrist_pose"][state_target_idx]
            else:
                wrist_pose_full = data[f"{hand_side}_wrist_pose"][state_interpolation_start:state_interpolation_end]
                rot_preprocess = st.Rotation.from_rotvec
                rot_postprocess = st.Rotation.as_rotvec
                slerp = st.Slerp(
                    times=np.arange(state_interpolation_start, state_interpolation_end),
                    rotations=rot_preprocess(wrist_pose_full[:, 3:]),
                )
                output_rot = rot_postprocess(slerp(state_target_idx))
                interp = si.interp1d(
                    x=np.arange(state_interpolation_start, state_interpolation_end),
                    y=wrist_pose_full[:, :3],
                    axis=0,
                    assume_sorted=True,
                )
                output_pos = interp(state_target_idx)
                obs_pose = np.concatenate([output_pos, output_rot], axis=-1)
            # pose_to_mat returns a new array, no need for .copy()
            cam_obs_pose_mat = cam_proj @ pose_to_mat(obs_pose)
            relative_pose_base = cam_obs_pose_mat[-1]
            cam_obs_pose_mat = convert_pose_mat_rep(
                cam_obs_pose_mat,
                base_pose_mat=relative_pose_base,
                pose_rep=self.proprioception_pose_rep,
                backward=False,
            )

            cam_obs_pose = mat_to_pose9d(cam_obs_pose_mat)  # (Ts, 9)
        else:
            cam_obs_pose = np.zeros((state_target_idx.shape[0], 9))

        if f"{hand_side}_arm_joints" in data_key_set:
            if fast_no_history:
                arm_obs_joints = data[f"{hand_side}_arm_joints"][state_target_idx]
                arm_obs_joints = _replace_joint_glitches(arm_obs_joints)
            else:
                arm_joints_full = data[f"{hand_side}_arm_joints"][state_interpolation_start:state_interpolation_end]
                arm_joints_full = _replace_joint_glitches(arm_joints_full)
                interp = si.interp1d(
                    x=np.arange(state_interpolation_start, state_interpolation_end),
                    y=arm_joints_full,
                    axis=0,
                    assume_sorted=True,
                )
                arm_obs_joints = interp(state_target_idx)
            joints_dim = arm_obs_joints.shape[-1]
            if joints_dim > FTP1_SINGLE_ARM_JOINT_DIM:
                raise ValueError(
                    f"Arm joints have more than {FTP1_SINGLE_ARM_JOINT_DIM} dimensions: {arm_obs_joints.shape[-1]}, only support dim <= {FTP1_SINGLE_ARM_JOINT_DIM}."
                )
            arm_obs_joints = np.concatenate(
                [arm_obs_joints, np.zeros((arm_obs_joints.shape[0], FTP1_SINGLE_ARM_JOINT_DIM - joints_dim))], axis=-1
            )
            relative_arm_joints_base = arm_obs_joints[-1]
            if self.proprioception_joint_rep == "relative":
                arm_obs_joints = arm_obs_joints - relative_arm_joints_base
        else:
            arm_obs_joints = np.zeros((state_target_idx.shape[0], FTP1_SINGLE_ARM_JOINT_DIM))

        # ================================ Get Hand & Gripper State ================================
        if fast_no_history:
            hand_obs_joints = data[f"{hand_side}_hand_joints"][state_target_idx]  # (Ts, N_joints)
        else:
            interp = si.interp1d(
                x=np.arange(state_interpolation_start, state_interpolation_end),
                y=data[f"{hand_side}_hand_joints"][state_interpolation_start:state_interpolation_end],
                axis=0,
                assume_sorted=True,
            )
            hand_obs_joints = interp(state_target_idx)  # (Ts, N_joints)
        relative_hand_base = hand_obs_joints[-1]
        if self.proprioception_joint_rep == "relative":
            hand_obs_joints = hand_obs_joints - relative_hand_base

        # ================================ Get Concatenated State ================================
        T = hand_obs_joints.shape[0]
        hand_idx = (data[f"{hand_side}_hand_joints_idx"][state_target_idx[-1]]).astype(np.int32)
        states = np.zeros((T, self.single_arm_action_rep_dim))
        states[:, :9] = cam_obs_pose
        states[:, 9 : 9 + FTP1_SINGLE_ARM_JOINT_DIM] = arm_obs_joints
        states[:, 9 + FTP1_SINGLE_ARM_JOINT_DIM + hand_idx] = hand_obs_joints

        # ================================ Get Wrist Action ================================
        if f"{hand_side}_wrist_pose" in data_key_set:
            action_wrist_key = f"{hand_side}_wrist_pose"
            action_pose = data[action_wrist_key][action_idx_slice]  # (Ta, 6)
            # pose_to_mat returns a new array, no need for .copy()
            cam_action_pose_mat = cam_proj @ pose_to_mat(action_pose)

            cam_action_pose_mat = convert_pose_mat_rep(
                cam_action_pose_mat, base_pose_mat=relative_pose_base, pose_rep=self.action_pose_rep, backward=False
            )
            cam_action_pose = mat_to_pose9d(cam_action_pose_mat)
            action_mask[:, :9] = 1
        else:
            cam_action_pose = np.zeros((action_idx_slice.shape[0], 9))
            action_mask[:, :9] = 0

        if f"{hand_side}_arm_joints" in data_key_set:
            action_arm_key = f"{hand_side}_arm_joints"
            action_arm_joints = data[action_arm_key][action_idx_slice]  # (Ta, N_joints)
            action_arm_joints = _replace_joint_glitches(action_arm_joints)
            joints_dim = action_arm_joints.shape[-1]
            if joints_dim > FTP1_SINGLE_ARM_JOINT_DIM:
                raise ValueError(
                    f"Arm joints have more than {FTP1_SINGLE_ARM_JOINT_DIM} dimensions: {action_arm_joints.shape[-1]}, only support dim <= {FTP1_SINGLE_ARM_JOINT_DIM}."
                )
            action_arm_joints = np.concatenate(
                [action_arm_joints, np.zeros((action_arm_joints.shape[0], FTP1_SINGLE_ARM_JOINT_DIM - joints_dim))],
                axis=-1,
            )
            if _action_joint_rep_uses_relative_arm(self.action_joint_rep):
                action_arm_joints = action_arm_joints - relative_arm_joints_base
            action_mask[:, 9 : 9 + joints_dim] = 1
        else:
            action_arm_joints = np.zeros((action_idx_slice.shape[0], FTP1_SINGLE_ARM_JOINT_DIM))
            action_mask[:, 9 : 9 + FTP1_SINGLE_ARM_JOINT_DIM] = 0

        # ================================ Get Hand & Gripper Action ================================
        action_hand_key = f"{hand_side}_hand_joints"
        hand_action_joints_raw = data[action_hand_key][action_idx_slice]  # (Ta, N_joints)
        hand_action_joints = hand_action_joints_raw.copy()
        if _action_joint_rep_uses_relative_hand(self.action_joint_rep):
            hand_action_joints = hand_action_joints - relative_hand_base
        absolute_hand_slots = _action_joint_rep_absolute_hand_slots(self.action_joint_rep)
        if absolute_hand_slots:
            absolute_hand_slots_arr = np.asarray(absolute_hand_slots, dtype=hand_idx.dtype)
            absolute_hand_mask = np.isin(hand_idx, absolute_hand_slots_arr)
            if np.any(absolute_hand_mask):
                hand_action_joints[:, absolute_hand_mask] = hand_action_joints_raw[:, absolute_hand_mask]

        # ================================ Get Concatenated Action ================================
        T = hand_action_joints.shape[0]
        actions = np.zeros((T, self.single_arm_action_rep_dim))
        actions[:, :9] = cam_action_pose
        actions[:, 9 : 9 + FTP1_SINGLE_ARM_JOINT_DIM] = action_arm_joints
        actions[:, 9 + FTP1_SINGLE_ARM_JOINT_DIM + hand_idx] = hand_action_joints
        action_mask[:, 9 + FTP1_SINGLE_ARM_JOINT_DIM + hand_idx] = 1

        # ================================ Get Tactile Single ================================
        # Calculate interpolation range for tactile separately
        tactile_interpolation_start = max(int(tactile_target_idx[0]) - 5, start_idx)
        tactile_interpolation_end = min(int(tactile_target_idx[-1]) + 2 + 5, end_idx)

        tactile_inputs = dict()
        tactile_function_area_inputs = dict()
        tactile_sensor_inputs = dict()
        tactile_type_inputs = dict()
        for key in data_keys:
            if key.startswith(f"{hand_side}_tactile_data_"):
                tactile_detail_key = key[len(f"{hand_side}_tactile_data_") :]
                if fast_no_history:
                    tactile_values = data[f"{hand_side}_tactile_data_{tactile_detail_key}"][tactile_target_idx]
                else:
                    org_values = data[f"{hand_side}_tactile_data_{tactile_detail_key}"][
                        tactile_interpolation_start:tactile_interpolation_end
                    ]
                    interp = si.interp1d(
                        x=np.arange(tactile_interpolation_start, tactile_interpolation_end),
                        y=org_values,
                        axis=0,
                        assume_sorted=True,
                    )
                    tactile_values = interp(tactile_target_idx)  # Use tactile_target_idx instead of state_target_idx
                tactile_type = data[f"{hand_side}_tactile_type_{tactile_detail_key}"][tactile_target_idx[-1]]
                if tactile_type == "image":
                    if tactile_values.ndim == 4:  # (T, N, H, W)
                        # interp(...) returns numpy.ndarray, so use numpy ops (not torch.Tensor.unsqueeze)
                        tactile_values = np.repeat(tactile_values[..., None], 3, axis=-1)  # (T, N, H, W, 3)
                    if tactile_values.shape[-2] != 224 or tactile_values.shape[-1] != 224:
                        resize_image = []
                        T = tactile_values.shape[0]
                        N = tactile_values.shape[1]
                        for i in range(tactile_values.shape[0]):
                            for j in range(tactile_values.shape[1]):
                                resize_image.append(cv2.resize(tactile_values[i, j], (224, 224)))
                        tactile_values = np.array(resize_image).reshape(T, N, 224, 224, 3)
                tactile_inputs[f"{hand_side}_tactile_{tactile_detail_key}"] = tactile_values
                tactile_function_area_inputs[f"{hand_side}_tactile_{tactile_detail_key}"] = data[
                    f"{hand_side}_tactile_area_{tactile_detail_key}"
                ][tactile_target_idx[-1]].astype(np.int32)

                tactile_sensor_inputs[f"{hand_side}_tactile_{tactile_detail_key}"] = data[
                    f"{hand_side}_tactile_sensor_{tactile_detail_key}"
                ][tactile_target_idx[-1]]
                tactile_type_inputs[f"{hand_side}_tactile_{tactile_detail_key}"] = tactile_type
                if hand_side == "left":
                    tactile_function_area_inputs[f"{hand_side}_tactile_{tactile_detail_key}"] += (
                        self.single_hand_num_tactile_areas
                    )

        is_contact = False
        if compute_contact:
            is_contact = self._frame_has_contact_for_hand(
                data,
                data_keys,
                int(tactile_target_idx[-1]),
                start_idx,
                hand_side=hand_side,
            )

        if f"{hand_side}_hand_usage" in data_key_set:
            action_usage_mask = data[f"{hand_side}_hand_usage"][action_idx_slice][:, None]  # (Ta, 1)
            action_mask = action_mask * action_usage_mask

        return (
            states,
            actions,
            tactile_inputs,
            tactile_function_area_inputs,
            tactile_sensor_inputs,
            tactile_type_inputs,
            is_contact,
            action_mask,
        )

    def set_ignore_rgb(self, value: bool):
        self.ignore_rgb = value

    def set_skip_normalization(self, value: bool):
        self.skip_normalization = value

    def __getitem__(self, idx: SupportsIndex, show_time: bool = False) -> Dict:
        if show_time:
            total_time = time.time()
            print("=============================================")
            st_time = time.time()
        idx = self.indices[idx]
        start_idx = self.start_frames[idx]  # inside_buffer start_idx(of the episode containing this frame)
        end_idx = self.end_frames[idx]  # inside_buffer end_idx(of the episode containing this frame)
        zarr_idx = self.zarr_idxs[idx]
        data = self.datas[zarr_idx]
        data_keys = self._cached_data_keys[zarr_idx]
        data_key_set = self._cached_data_key_sets[zarr_idx]
        episode_idx = self.episodes_idxs[idx]  # inside_buffer episode_idx
        idx = self.data_idxs[idx]  # inside_buffer data_idx

        if show_time:
            print(f"Time taken to get item: {time.time() - st_time} seconds")
            st_time = time.time()

        # ============================== add image data ==============================
        image_target_idx = np.array(
            [idx]
            + [idx - self.image_down_sample_steps[history_idx] for history_idx in range(self.image_hisory_length - 1)]
        )
        image_target_idx = np.clip(image_target_idx[::-1], start_idx, end_idx - 1)

        image_inputs = dict()
        image_masks_inputs = dict()

        # Find the first available image key to determine dimensions
        # Try in order: camera_ego_rgb, right_wrist_camera_rgb, left_wrist_camera_rgb
        possible_image_keys = ["camera_main_rgb", "camera_ego_rgb", "right_wrist_camera_rgb", "left_wrist_camera_rgb"]
        base_image_key = None
        base_image = None
        height, width, channels = None, None, None

        for image_key in possible_image_keys:
            if image_key in data_key_set:
                base_image_key = image_key
                base_image = data[image_key][idx]
                # Extract dimensions from base_image
                # Detect format: [H, W, C] or [C, H, W]
                if len(base_image.shape) == 3:
                    if base_image.shape[0] == 3:
                        # [C, H, W] format
                        channels, height, width = base_image.shape
                    elif base_image.shape[2] == 3:
                        # [H, W, C] format
                        height, width, channels = base_image.shape
                    else:
                        raise ValueError(
                            f"Cannot determine image format from shape: {base_image.shape}, expected [H, W, C] or [C, H, W]"
                        )
                else:
                    raise ValueError(
                        f"Unexpected base_image shape: {base_image.shape}, expected [H, W, C] or [C, H, W]"
                    )
                assert channels == 3, f"Expected 3 channels for images, got {channels}"
                break

        # If no image key is available, we still need to set default dimensions
        if height is None or width is None or channels is None:
            raise ValueError(
                f"Cannot determine image dimensions for key: {base_image_key}, at lease one of key in {possible_image_keys} should exist."
            )

        if not self.ignore_rgb:
            for i in range(self.image_hisory_length):
                for image_key in possible_image_keys:
                    if image_key not in data_key_set:
                        continue

                    # IO optimization: reuse base_image if it's the same key and index
                    if base_image_key == image_key and int(image_target_idx[i]) == idx:
                        img = base_image if base_image is not None else data[image_key][int(image_target_idx[i])]
                    else:
                        img = data[image_key][int(image_target_idx[i])]

                    # Normalize to [-1, 1] (creates new array, so no need to copy base_image)
                    img = img.astype(np.float32) / 255.0 * 2.0 - 1.0

                    # Ensure image is in [H, W, C] format (channels last)
                    if len(img.shape) == 3 and img.shape[0] == 3:
                        # Convert from [C, H, W] to [H, W, C]
                        img = np.transpose(img, (1, 2, 0))  # [C, H, W] -> [H, W, C]
                    elif len(img.shape) == 3 and img.shape[2] == 3:
                        # Already in [H, W, C] format, no conversion needed
                        pass
                    else:
                        raise ValueError(f"Unexpected image shape: {img.shape}, expected [H, W, C]")
                    image_inputs[f"{image_key}_{i}"] = img
                    image_masks_inputs[f"{image_key}_{i}"] = np.True_

        if show_time:
            print(f"Time taken to get image data: {time.time() - st_time} seconds")
            st_time = time.time()

        if "camera_ego_pose" in data_key_set:
            try:
                cam_proj = np.linalg.inv(pose_to_mat(data["camera_ego_pose"][idx]))
                if not np.all(np.isfinite(cam_proj)):
                    cam_proj = np.eye(4)
            except np.linalg.LinAlgError:
                cam_proj = np.eye(4)
        else:
            cam_proj = np.eye(4)

        if show_time:
            print(f"Time taken to get cam proj data: {time.time() - st_time} seconds")
            st_time = time.time()

        # ============================== get arm proprioception & action ==============================
        state_target_idx = np.array(
            [idx]
            + [idx - self.state_down_sample_steps[history_idx] for history_idx in range(self.state_hisory_length - 1)]
        )
        state_target_idx = np.clip(state_target_idx[::-1], start_idx, end_idx - 1)  # history->now

        tactile_target_idx = np.array(
            [idx]
            + [
                idx - self.tactile_down_sample_steps[history_idx]
                for history_idx in range(self.tactile_hisory_length - 1)
            ]
        )
        tactile_target_idx = np.clip(tactile_target_idx[::-1], start_idx, end_idx - 1)  # history->now

        slice_end = min(end_idx, idx + (self.action_horizon - 1) * self.action_down_sample_steps + 1)
        action_idx_slice = np.arange(
            idx, slice_end, self.action_down_sample_steps
        )  # (idx: slice_end: self.action_down_sample_steps)
        compute_contact = (not self.skip_normalization) and _should_compute_contact_for_non_tactile_dropout(
            self.non_tactile_dropout_ratio
        )

        if show_time:
            print(f"Time taken to get index data: {time.time() - st_time} seconds")
            st_time = time.time()

        (
            states_left,
            actions_left,
            tactile_inputs_left,
            tactile_function_area_inputs_left,
            tactile_sensor_inputs_left,
            tactile_type_inputs_left,
            is_contact_left,
            action_mask_left,
        ) = self._get_single_arm_data(
            data,
            data_keys,
            data_key_set,
            "left",
            state_target_idx,
            tactile_target_idx,
            action_idx_slice,
            start_idx,
            end_idx,
            cam_proj,
            compute_contact=compute_contact,
        )
        if show_time:
            print(f"Time taken to get left arm data: {time.time() - st_time} seconds")
            st_time = time.time()

        (
            states_right,
            actions_right,
            tactile_inputs_right,
            tactile_function_area_inputs_right,
            tactile_sensor_inputs_right,
            tactile_type_inputs_right,
            is_contact_right,
            action_mask_right,
        ) = self._get_single_arm_data(
            data,
            data_keys,
            data_key_set,
            "right",
            state_target_idx,
            tactile_target_idx,
            action_idx_slice,
            start_idx,
            end_idx,
            cam_proj,
            compute_contact=compute_contact,
        )

        if show_time:
            print(f"Time taken to get right arm data: {time.time() - st_time} seconds")
            st_time = time.time()

        # ============================== get head proprioception & action ==============================
        flag_camera_static = False

        if "camera_ego_pose" in data_key_set:
            static_check_start = max(int(state_target_idx[0]) - 5, start_idx)
            static_check_end = min(int(action_idx_slice[-1]) + 5, end_idx)
            # 判断data['camera_ego_pose'][static_check_start: static_check_end]是否没有变化
            camear_pose_read = data["camera_ego_pose"][static_check_start:static_check_end]
            if np.all(np.isclose(camear_pose_read, camear_pose_read[0], atol=1e-6)):
                # If camera is static, we ignore the camera pose prediction and make its mask to be all zeros.
                flag_camera_static = True
            else:
                # Calculate interpolation range for head state (uses state_target_idx)
                head_interpolation_start = max(int(state_target_idx[0]) - 5, start_idx)
                head_interpolation_end = min(int(state_target_idx[-1]) + 5, end_idx)

                # Optimize: Read full pose array once, then slice columns (reduces zarr access from 2 to 1)
                ego_pose_full = camear_pose_read[
                    head_interpolation_start - static_check_start : head_interpolation_end - static_check_start
                ]
                if self.disable_history and self.state_hisory_length == 1:
                    ego_pose = data["camera_ego_pose"][state_target_idx]
                else:
                    rot_preprocess = st.Rotation.from_rotvec
                    rot_postprocess = st.Rotation.as_rotvec
                    slerp = st.Slerp(
                        times=np.arange(head_interpolation_start, head_interpolation_end),
                        rotations=rot_preprocess(ego_pose_full[:, 3:]),
                    )
                    output_rot = rot_postprocess(slerp(state_target_idx))
                    interp = si.interp1d(
                        x=np.arange(head_interpolation_start, head_interpolation_end),
                        y=ego_pose_full[:, :3],
                        axis=0,
                        assume_sorted=True,
                    )
                    output_pos = interp(state_target_idx)
                    ego_pose = np.concatenate([output_pos, output_rot], axis=-1)
                # pose_to_mat returns a new array, no need for .copy()
                cam_ego_pose_mat = pose_to_mat(ego_pose)
                relative_ego_pose_base = cam_ego_pose_mat[-1]
                cam_ego_pose_mat = convert_pose_mat_rep(
                    cam_ego_pose_mat,
                    base_pose_mat=relative_ego_pose_base,
                    pose_rep=self.proprioception_pose_rep,
                    backward=False,
                )
                cam_ego_pose = mat_to_pose9d(cam_ego_pose_mat)  # (Ts, 9)
                action_pose = camear_pose_read[
                    action_idx_slice - static_check_start
                ]  # data['camera_ego_pose'][action_idx_slice]
                # pose_to_mat returns a new array, no need for .copy()
                cam_action_pose_mat = pose_to_mat(action_pose)
                cam_action_pose_mat = convert_pose_mat_rep(
                    cam_action_pose_mat,
                    base_pose_mat=relative_ego_pose_base,
                    pose_rep=self.action_pose_rep,
                    backward=False,
                )
                cam_action_pose = mat_to_pose9d(cam_action_pose_mat)
                action_mask_head = np.ones((action_idx_slice.shape[0], 9))
        else:
            flag_camera_static = True

        if flag_camera_static:
            # all fill 0 means no camera prediction.
            # if camera exists (but nearly static), it should be xyz(000) + rpy(000) with matrix representation: [0,0,0,1,0,0,0,1,0]
            cam_ego_pose = np.stack(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * state_target_idx.shape[0], axis=0
            )  # shape: (Ts, 9)
            cam_action_pose = np.stack(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * action_idx_slice.shape[0], axis=0
            )  # shape: (Ta, 9)
            action_mask_head = np.zeros((action_idx_slice.shape[0], 9))

        if show_time:
            print(f"Time taken to get head data: {time.time() - st_time} seconds")
            st_time = time.time()

        # ============================== get supplementary joints proprioception & action (on reserved di) ==============================

        supp_states = np.zeros((state_target_idx.shape[0], self.reserved_action_dim))
        supp_actions = np.zeros((action_idx_slice.shape[0], self.reserved_action_dim))
        supp_action_mask = np.zeros((action_idx_slice.shape[0], self.reserved_action_dim))

        if "supplementary_joints" in data_key_set:
            static_check_start = max(int(state_target_idx[0]) - 5, start_idx)
            static_check_end = min(int(action_idx_slice[-1]) + 5, end_idx)
            # 判断data['camera_ego_pose'][static_check_start: static_check_end]是否没有变化
            supp_joints_read = data["supplementary_joints"][static_check_start:static_check_end]
            head_interpolation_start = max(int(state_target_idx[0]) - 5, start_idx)
            head_interpolation_end = min(int(state_target_idx[-1]) + 5, end_idx)

            # Optimize: Read full pose array once, then slice columns (reduces zarr access from 2 to 1)
            supp_joints_full = supp_joints_read[
                head_interpolation_start - static_check_start : head_interpolation_end - static_check_start
            ]
            if self.disable_history and self.state_hisory_length == 1:
                obs_supp_joints = data["supplementary_joints"][state_target_idx]
            else:
                supp_joints_interp = si.interp1d(
                    x=np.arange(head_interpolation_start, head_interpolation_end),
                    y=supp_joints_full,
                    axis=0,
                    assume_sorted=True,
                )
                obs_supp_joints = supp_joints_interp(state_target_idx)
            obs_supp_joints_dim = obs_supp_joints.shape[-1]
            obs_supp_joints_base = obs_supp_joints[-1]
            if self.proprioception_joint_rep == "relative":
                obs_supp_joints = obs_supp_joints - obs_supp_joints_base
            action_supp_joints = supp_joints_read[
                action_idx_slice - static_check_start
            ]  # data['supplementary_joints'][action_idx_slice]
            if _action_joint_rep_uses_relative_supplementary(self.action_joint_rep):
                action_supp_joints = action_supp_joints - obs_supp_joints_base
            supp_states[:, :obs_supp_joints_dim] = obs_supp_joints
            supp_actions[:, :obs_supp_joints_dim] = action_supp_joints
            supp_action_mask[:, :obs_supp_joints_dim] = 1

        # ============================== get concatenated state & action ==============================

        states = np.concatenate([states_right, states_left, cam_ego_pose, supp_states], axis=-1)
        actions = np.concatenate([actions_right, actions_left, cam_action_pose, supp_actions], axis=-1)
        action_mask = np.concatenate([action_mask_right, action_mask_left, action_mask_head, supp_action_mask], axis=-1)
        tactile_inputs = {**tactile_inputs_right, **tactile_inputs_left}
        tactile_function_area_inputs = {**tactile_function_area_inputs_right, **tactile_function_area_inputs_left}
        tactile_sensor_inputs = {**tactile_sensor_inputs_right, **tactile_sensor_inputs_left}
        tactile_type_inputs = {**tactile_type_inputs_right, **tactile_type_inputs_left}
        is_contact = is_contact_right or is_contact_left
        action_real_horizon = actions.shape[0]
        state_masks = np.ones(states.shape[:-1], dtype=np.bool_)

        padding = np.repeat(actions[-1:], self.action_horizon - action_real_horizon, axis=0)
        actions = np.concatenate([actions, padding], axis=0)
        padding_mask = np.repeat(action_mask[-1:], self.action_horizon - action_real_horizon, axis=0)
        action_mask = np.concatenate([action_mask, padding_mask], axis=0)
        if is_contact and random.random() <= self.non_tactile_dropout_ratio:
            state_masks[...] = np.False_
            for key in image_inputs.keys():
                image_masks_inputs[key] = np.False_

        if show_time:
            print(f"Time taken to get pad & concate data package: {time.time() - st_time} seconds")
            st_time = time.time()

        # ============================== get final data package ==============================
        if "sub_task_instruction" in data_key_set:
            instruction_list = data["sub_task_instruction"][action_idx_slice]
            instruction_list = [str(x) for x in instruction_list]
            prompt = instruction_list[0]
            # If a sampled action chunk crosses an instruction boundary, we keep the chunk prompt
            # as the first-step instruction, and mask out the loss for all steps after the first change.
            instruction_arr = np.asarray(instruction_list, dtype=str)
            instruction_change_idx = np.where(instruction_arr != prompt)[0]
            if instruction_change_idx.size > 0:
                first_change_t = int(instruction_change_idx[0])
                action_mask[first_change_t:, ...] = 0
        else:
            prompt = "finish tasks."
        prompt = prompt.strip()
        if not prompt.endswith("."):
            prompt += "."
        # Add domain condition to prompt if enabled
        if self.use_domain_condition:
            domain_prefix = f"[{self.domain_name}] "
            prompt = domain_prefix + prompt

        sample = {
            "image": image_inputs,  # dict[ (H, W, C) ], with values in [-1, 1], channels last format
            "image_mask": image_masks_inputs,  # dict[ np.True_ / np.False_ ]
            "tactile": tactile_inputs,  # dict[ (T, *tac_shape)]
            "tactile_function_area": tactile_function_area_inputs,  # dict[ int ]
            "tactile_sensor": tactile_sensor_inputs,  # dict[ str ]
            "tactile_type": tactile_type_inputs,  # dict[ str ]
            "state": states,  # (T, self.single_arm_action_rep_dim * 2 + 9), with 2 * (9dof-wrist first + hand joints dim) + 9dof-ego-head dim
            "state_mask": state_masks,  # (T,), True if state token/timestep is valid
            "actions": actions,  # (T, self.single_arm_action_rep_dim * 2 + 9), with 2 * (9dof-wrist first + hand joints dim) + 9dof-ego-head dim
            "action_mask": action_mask,  # (T, self.single_arm_action_rep_dim * 2 + 9), with 2 * (9dof-wrist first + hand joints dim) + 9dof-ego-head dim
            "prompt": prompt,  # str
            "domain_name": self.domain_name,  # str
        }

        # sample_norm = apply_norm_stats(sample, self.norm_stats, unnormalize=False)
        # sample_unnormalized = apply_norm_stats(sample, self.norm_stats, unnormalize=True)
        # import pdb; pdb.set_trace()

        if show_time:
            print(f"Time taken to get sample: {time.time() - st_time} seconds")
            st_time = time.time()

        if not self.skip_normalization:
            if self.norm_stats is None:
                raise ValueError(
                    f"Domain {self.domain_name} | Norm stats not found. Please run load_norm_stats() first."
                )
            sample = apply_norm_stats(sample, self.norm_stats, unnormalize=False)

        if show_time:
            print(f"Time taken to get norm data: {time.time() - st_time} seconds")
            st_time = time.time()
            print(f"Total time taken to get sample: {time.time() - total_time} seconds")

        return sample

    def set_sample_ratio(self, sample_ratio: float):
        if sample_ratio < 1.0:
            interval_size = int(1.0 / sample_ratio)
            self.indices = self.indices_org[::interval_size]
        else:
            self.indices = copy.deepcopy(self.indices_org)

    def generate_independent_norm_stats(
        self,
        batch_size,
        independent_norm_mode="all",
        norm_time_dim=False,
        norm_type="quantile",
        sample_ratio: float = 1.0,
        num_workers=8,
        cache_max_size: int = None,
        image_channel_pool_max_size: int | None = None,
        image_channel_batch_samples: int | None = None,
        image_tactile_norm_mode: str | None = None,
        verbose: bool = True,
    ):
        """
        # independent_norm_mode:
        # 1. none: only do independent normalization for tactile.
        # 2. joint: do independent normalization for tactile & hand joints.
        # 3. all: do independent normalization for tactile & hand joints & wrist/head pose.
        norm_time_dim: whether to do independent normalization on the time dimension.
        norm_type: normalization type, one of 'quantile' or 'zscore'.
        sample_ratio: the ratio of the dataset to be sampled.
        num_workers: the number of workers to be used.
        cache_max_size: maximum number of batches to cache (None = no limit).
        image_channel_pool_max_size: max channel vectors processed per image tactile key
            when updating incremental stats (0 = unlimited).
        image_channel_batch_samples: max channel vectors sampled per image tactile key
            from each batch before updating incremental stats (0 = use all vectors).
        image_tactile_norm_mode: normalization mode for image-type tactile:
            'channel_wise' (default) or 'image_norm'.
        verbose: whether to print one-time logs when image channel cache reaches cap.
        """
        if image_channel_pool_max_size is None:
            image_channel_pool_max_size = self.norm_image_channel_pool_max_size
        if image_channel_batch_samples is None:
            image_channel_batch_samples = self.norm_image_channel_batch_samples
        if image_tactile_norm_mode is None:
            image_tactile_norm_mode = self.norm_image_tactile_mode
        if image_tactile_norm_mode not in {"channel_wise", "image_norm"}:
            raise ValueError(
                "Invalid image_tactile_norm_mode: "
                f"{image_tactile_norm_mode}. Please choose from ['channel_wise', 'image_norm']"
            )

        self.set_sample_ratio(sample_ratio)
        self.set_skip_normalization(True)
        self.set_ignore_rgb(True)

        # This ensures we get random batches, and we can break when cache is full
        shuffle = True
        data_loader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn_handle_strings,
            shuffle=shuffle,
        )
        data_cache_tactile = dict()
        data_cache_tactile_type = dict()
        # Incremental image tactile channel stats (constant memory per tactile key).
        data_cache_image_channel_stats = dict()
        image_channel_seen_counts = dict()
        image_channel_remaining_budget = dict()
        image_channel_cap_logged = set()
        data_cache_action_mask = None
        if independent_norm_mode != "none":
            data_cache_action = list()
            data_cache_state = list()
            data_cache_action_mask = list()

        n_batch = 0

        planned_total_batches = len(data_loader)
        if cache_max_size is not None:
            max_cache_batches = max(0, math.floor((cache_max_size - 1) / batch_size))
            planned_total_batches = min(planned_total_batches, max_cache_batches)

        def sample_image_channel_vectors(tactile_data, max_samples, batch_budget=None):
            """Sample channel vectors from image tactile tensor.

            tactile_data expected shape: (B, T, N, H, W, C).
            Returns sampled vectors of shape (S, C), where:
            - S = all vectors if max_samples <= 0 and batch_budget is None
            - S <= max_samples (if max_samples > 0)
            - S <= batch_budget (if provided)
            """
            if isinstance(tactile_data, torch.Tensor):
                tactile_np = tactile_data.detach().cpu().numpy()
            else:
                tactile_np = np.asarray(tactile_data)
            if tactile_np.ndim < 2:
                return None

            # Flatten all non-channel dimensions to sample channel vectors.
            channel_dim = tactile_np.shape[-1]
            if channel_dim <= 0:
                return None
            flat = tactile_np.reshape(-1, channel_dim)
            if flat.shape[0] == 0:
                return None

            sample_limit = flat.shape[0]
            if max_samples > 0:
                sample_limit = min(sample_limit, max_samples)
            if batch_budget is not None:
                sample_limit = min(sample_limit, max(0, int(batch_budget)))

            if sample_limit <= 0:
                return None
            if sample_limit >= flat.shape[0]:
                return flat.astype(np.float32, copy=False)

            idx = np.random.choice(flat.shape[0], size=sample_limit, replace=False)
            return flat[idx].astype(np.float32, copy=False)

        for batch in tqdm(
            data_loader, desc=f"iterating {self.domain_name} dataset to get independent keys normalization"
        ):
            # Check if we've reached cache_max_size (in terms of number of batches)
            if cache_max_size is not None and (n_batch + 1) * batch_size >= cache_max_size:
                break

            for key in batch["tactile"].keys():
                tactile_data = batch["tactile"][key]  # Shape: (B, T, N, *tac_shape)
                tactile_function_areas = batch["tactile_function_area"][key]  # Shape: (B, N)
                # In per-domain norm-stats generation, tactile metadata is static for each tactile key.
                tactile_sensor_name = canonicalize_tactile_sensor_name(batch["tactile_sensor"][key][0])
                tactile_type_value = canonicalize_tactile_encoder_type(batch["tactile_type"][key][0])
                tactile_norm_key = build_tactile_sensor_type_shape_key_from_data(
                    tactile_sensor_name,
                    tactile_type_value,
                    tactile_data,
                    tactile_function_areas=tactile_function_areas,
                )
                if tactile_norm_key not in data_cache_tactile:
                    data_cache_tactile[tactile_norm_key] = []
                    data_cache_tactile_type[tactile_norm_key] = None  # Will be set once we see the first batch

                if data_cache_tactile_type[tactile_norm_key] is None:
                    data_cache_tactile_type[tactile_norm_key] = tactile_type_value
                elif data_cache_tactile_type[tactile_norm_key] != tactile_type_value:
                    raise ValueError(
                        f"Mixed tactile types found for grouped norm key {tactile_norm_key}: "
                        f"{data_cache_tactile_type[tactile_norm_key]} vs {tactile_type_value}"
                    )

                # Replace image-type tactile data with None (to save memory), and
                # incrementally update channel-level statistics.
                if tactile_type_value == "image":
                    if image_tactile_norm_mode == "channel_wise":
                        if tactile_norm_key not in data_cache_image_channel_stats:
                            data_cache_image_channel_stats[tactile_norm_key] = _shared_normalize.RunningStats()
                            image_channel_seen_counts[tactile_norm_key] = 0
                            if image_channel_pool_max_size > 0:
                                image_channel_remaining_budget[tactile_norm_key] = image_channel_pool_max_size

                        batch_budget = None
                        if image_channel_pool_max_size > 0:
                            remaining_budget = image_channel_remaining_budget[tactile_norm_key]
                            if remaining_budget <= 0:
                                tactile_data = None
                                data_cache_tactile[tactile_norm_key].append(tactile_data)
                                continue
                            remaining_batches = max(1, planned_total_batches - n_batch)
                            # Allocate budget across remaining batches to reduce early-batch bias.
                            batch_budget = int(math.ceil(remaining_budget / remaining_batches))
                        sampled_vectors = sample_image_channel_vectors(
                            tactile_data,
                            max_samples=image_channel_batch_samples,
                            batch_budget=batch_budget,
                        )
                        if sampled_vectors is not None and len(sampled_vectors) > 0:
                            # Compute image tactile stats in preprocessed domain first.
                            sampled_vectors = sampled_vectors / 255.0 * 2.0 - 1.0
                            data_cache_image_channel_stats[tactile_norm_key].update(sampled_vectors)
                            image_channel_seen_counts[tactile_norm_key] += sampled_vectors.shape[0]
                            if image_channel_pool_max_size > 0:
                                image_channel_remaining_budget[tactile_norm_key] = max(
                                    0,
                                    image_channel_remaining_budget[tactile_norm_key] - sampled_vectors.shape[0],
                                )
                                if (
                                    verbose
                                    and image_channel_remaining_budget[tactile_norm_key] == 0
                                    and tactile_norm_key not in image_channel_cap_logged
                                ):
                                    print(
                                        f"[NormStats][{self.domain_name}] image tactile group '{tactile_norm_key}' reached "
                                        f"norm_image_channel_pool_max_size={image_channel_pool_max_size} "
                                        f"(seen={image_channel_seen_counts[tactile_norm_key]})."
                                    )
                                    image_channel_cap_logged.add(tactile_norm_key)
                    tactile_data = None
                else:
                    tactile_data = _reshape_tactile_batch_for_grouped_norm(tactile_data)

                # Directly append to cache (no need to check since we break at loop start)
                data_cache_tactile[tactile_norm_key].append(tactile_data)

            if independent_norm_mode != "none":
                # Directly append to cache (no need to check since we break at loop start)
                data_cache_action.append(batch["actions"])
                data_cache_state.append(batch["state"])
                data_cache_action_mask.append(batch["action_mask"])

            n_batch += 1

        # data_cache_tactile_type is already a dict[str, str] mapping key -> type string
        # No need to process further
        if independent_norm_mode == "none":
            data_cache = {
                "norm_time_dim": norm_time_dim,
                "norm_type": norm_type,
                "data_cache": {
                    "tactile": data_cache_tactile,
                },
                "data_norm_dim": {
                    "tactile": None,
                },
            }
        elif independent_norm_mode == "joint":
            action_norm_dim = list(range(9, self.single_arm_action_rep_dim))
            action_norm_dim = action_norm_dim + list(
                range(self.single_arm_action_rep_dim + 9, 2 * self.single_arm_action_rep_dim)
            )
            action_norm_dim = action_norm_dim + list(
                range(
                    2 * self.single_arm_action_rep_dim + 9,
                    2 * self.single_arm_action_rep_dim + 9 + self.reserved_action_dim,
                )
            )
            data_cache = {
                "norm_time_dim": norm_time_dim,
                "norm_type": norm_type,
                "data_cache": {
                    "tactile": data_cache_tactile,
                    "actions": data_cache_action,
                    "state": data_cache_state,
                },
                "data_norm_dim": {
                    "tactile": None,
                    "actions": action_norm_dim,
                    "state": action_norm_dim,
                },
            }
        elif independent_norm_mode == "all":
            data_cache = {
                "norm_time_dim": norm_time_dim,
                "norm_type": norm_type,
                "data_cache": {
                    "tactile": data_cache_tactile,
                    "actions": data_cache_action,
                    "state": data_cache_state,
                },
                "data_norm_dim": {
                    "tactile": None,
                    "actions": None,
                    "state": None,
                },
            }
        else:
            raise ValueError(
                f"Invalid independent_norm_mode: {independent_norm_mode}, please choose from ['none', 'hand', 'all']"
            )

        finalized_image_channel_params = {}
        for tactile_key, running_stats in data_cache_image_channel_stats.items():
            try:
                image_stats = running_stats.get_statistics()
            except ValueError:
                # Not enough samples to estimate statistics robustly.
                finalized_image_channel_params[tactile_key] = None
                continue

            if norm_type == "quantile":
                finalized_image_channel_params[tactile_key] = {
                    "q01": np.asarray(image_stats.q01),
                    "q99": np.asarray(image_stats.q99),
                    "input_preprocess": "div255_mul2_minus1",
                }
            elif norm_type == "zscore":
                finalized_image_channel_params[tactile_key] = {
                    "mean": np.asarray(image_stats.mean),
                    "std": np.asarray(image_stats.std),
                    "input_preprocess": "div255_mul2_minus1",
                }
            else:
                finalized_image_channel_params[tactile_key] = None

        self.norm_stats = calculate_norm_stats(
            data_cache,
            data_cache_tactile_type,
            data_cache_action_mask,
            data_cache_image_channel_params=finalized_image_channel_params,
            image_tactile_norm_mode=image_tactile_norm_mode,
        )
        norm_stats_to_store = norm_stats_to_jsonable(self.norm_stats)
        dump_json_with_inline_lists(
            norm_stats_to_store,
            os.path.join(
                self.assets_dir,
                f"independent_norm_stats_{independent_norm_mode}_t{int(norm_time_dim)}_{norm_type}.json",
            ),
            indent=4,
        )

        self.set_sample_ratio(1.0)
        self.set_skip_normalization(False)
        self.set_ignore_rgb(False)

    def load_independent_norm_stats(self, independent_norm_mode, norm_time_dim, norm_type):
        norm_stats_path = (
            pathlib.Path(self.assets_dir)
            / f"independent_norm_stats_{independent_norm_mode}_t{int(norm_time_dim)}_{norm_type}.json"
        )
        print(f"Loading independent norm stats from {norm_stats_path}")
        if not norm_stats_path.exists():
            raise ValueError(
                f"Domain {self.domain_name} | Independent norm stats not found. Please run generate_independent_norm_stats() first."
            )
        old_repo_id = getattr(self.data_config, "reuse_norm_repo_id", "")
        override_path = None
        if old_repo_id:
            override_path = _build_old_norm_stats_path(
                self.data_config.assets_dirs,
                old_repo_id,
                pathlib.Path(self.domain_name) / norm_stats_path.name,
            )
        norm_stats_json, replaced_keys = load_norm_stats_jsonable_with_override(norm_stats_path, override_path)
        if override_path is not None:
            _log_norm_override_result(
                current_path=norm_stats_path,
                override_path=override_path,
                replaced_keys=replaced_keys,
            )
        self.norm_stats = norm_stats_from_jsonable(norm_stats_json)
        _sanitize_loaded_norm_stats(self.norm_stats, domain_name=self.domain_name)

    def generate_contact_detection_thresholds(self) -> dict:
        stream_scores: dict[str, list[np.ndarray]] = {}
        stream_total_score_counts: dict[str, int] = {}
        stream_keys_seen: set[str] = set()
        seen_episodes: set[int] = set()

        for dataset_internal_idx in tqdm(
            self.indices,
            desc=f"iterating {self.domain_name} dataset to get tactile contact thresholds",
        ):
            episode_idx = self.episodes_idxs[dataset_internal_idx]
            if episode_idx in seen_episodes:
                continue
            seen_episodes.add(episode_idx)

            zarr_idx = self.zarr_idxs[dataset_internal_idx]
            start_idx = self.start_frames[dataset_internal_idx]
            end_idx = self.end_frames[dataset_internal_idx]
            data = self.datas[zarr_idx]

            for key in self._cached_data_keys[zarr_idx]:
                if not is_tactile_data_key(key):
                    continue

                stream_keys_seen.add(key)
                tactile_values = np.asarray(data[key][start_idx:end_idx])
                scores = compute_tactile_delta_scores(tactile_values)
                finite_scores = scores[np.isfinite(scores)]
                stream_total_score_counts[key] = stream_total_score_counts.get(key, 0) + int(finite_scores.size)
                positive_scores = finite_scores[finite_scores > 0]
                if positive_scores.size == 0:
                    continue
                stream_scores.setdefault(key, []).append(positive_scores.astype(np.float32, copy=False))

        finalized_stream_scores = {
            key: np.concatenate(value_list) if value_list else np.zeros((0,), dtype=np.float32)
            for key, value_list in stream_scores.items()
        }
        for key in stream_keys_seen:
            finalized_stream_scores.setdefault(key, np.zeros((0,), dtype=np.float32))

        contact_thresholds = build_contact_thresholds_jsonable(
            finalized_stream_scores,
            threshold_k=self.contact_detection_threshold_k,
        )
        stream_stats = contact_thresholds["streams"]
        for key, stats in stream_stats.items():
            all_score_count = int(stream_total_score_counts.get(key, 0))
            positive_score_count = int(stats.get("positive_score_count") or 0)
            nonpositive_score_count = max(0, all_score_count - positive_score_count)
            stats["all_score_count"] = all_score_count
            stats["nonpositive_score_count"] = nonpositive_score_count
            stats["positive_score_fraction"] = (
                float(positive_score_count / all_score_count) if all_score_count > 0 else None
            )
        save_path = pathlib.Path(self._get_contact_thresholds_path())
        dump_json_with_inline_lists(contact_thresholds, save_path, indent=2)
        num_valid_thresholds = sum(1 for stream in stream_stats.values() if stream.get("threshold") is not None)
        print(
            f"Saved contact thresholds for {self.domain_name} to {save_path}: "
            f"{num_valid_thresholds}/{len(stream_stats)} streams have valid thresholds"
        )
        self._contact_thresholds = contact_thresholds
        self._contact_thresholds_loaded = True
        return {
            "path": str(save_path),
            "detector_version": contact_thresholds["detector_version"],
            "score_type": contact_thresholds["score_type"],
            "threshold_k": contact_thresholds["threshold_k"],
            "streams": stream_stats,
        }


class MultiZarrDataset(Dataset):
    def __init__(self, data_config, action_horizon: int, split="train"):
        self.data_config = data_config
        self.action_horizon = action_horizon
        self.split = split
        self.domain_dataset_list = []
        self.domain_list = []
        self.skip_normalization = False
        self.single_arm_action_rep_dim = data_config.single_arm_action_rep_dim
        self.assets_dir = os.path.join(data_config.assets_dirs, data_config.repo_id)
        os.makedirs(self.assets_dir, exist_ok=True)

        self.seed = data_config.seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        # Load dataset configuration from JSON file
        if not hasattr(data_config, "dataset_config_path") or not data_config.dataset_config_path:
            raise ValueError("dataset_config_path is required. Please provide a path to dataset_config.json")

        config_path = pathlib.Path(data_config.dataset_config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Dataset config file not found: {config_path}")

        with open(config_path, "r") as f:
            self.dataset_config = json.load(f)

        # Extract enabled datasets and their trajectory ratios
        self.dataset_trajectory_ratios = {}  # Map from dataset name to use_trajectory_ratio
        dataset_entries = []
        default_ratio = float(self.dataset_config.get("default_use_trajectory_ratio", 1.0))
        for config_idx, ds in enumerate(self.dataset_config.get("datasets", [])):
            if not ds.get("enabled", True):
                continue

            dataset_path = ds["path"]
            dataset_name = ds.get("name", dataset_path.split("/")[-1])
            norm_stats_domain_name = ds.get("norm_stats_domain_name", dataset_name)
            use_trajectory_ratio = float(ds.get("use_trajectory_ratio", default_ratio))
            dataset_entries.append(
                {
                    "name": dataset_name,
                    "path": dataset_path,
                    "norm_stats_domain_name": norm_stats_domain_name,
                    "use_trajectory_ratio": use_trajectory_ratio,
                    "config_index": config_idx,
                }
            )
            self.dataset_trajectory_ratios[dataset_name] = use_trajectory_ratio

        if not dataset_entries:
            raise ValueError("No enabled datasets found in configuration")

        self.dataset_path_list = [entry["path"] for entry in dataset_entries]
        self.domain_list = [entry["name"] for entry in dataset_entries]
        self.norm_stats_domain_name_list = [entry["norm_stats_domain_name"] for entry in dataset_entries]

        unique_entries, entry_to_unique_idx = _dedupe_dataset_entries_for_init(dataset_entries)
        unique_dataset_cache: dict[int, ZarrDataset] = {}

        print(
            "[MultiZarrDataset] enabled entries: "
            f"{len(dataset_entries)}, unique init groups: {len(unique_entries)}, "
            f"reused clones: {len(dataset_entries) - len(unique_entries)}"
        )
        print(f"[MultiZarrDataset] domain_list preview: {_format_preview_list(self.domain_list)}")
        ratio_preview = [f"{entry['name']}:{float(entry['use_trajectory_ratio']):.4f}" for entry in dataset_entries]
        print(f"[MultiZarrDataset] dataset_trajectory_ratios preview: {_format_preview_list(ratio_preview)}")

        for entry_idx, entry in enumerate(dataset_entries):
            representative_idx = entry_to_unique_idx[entry_idx]
            if representative_idx not in unique_dataset_cache:
                domain_dataset = ZarrDataset(
                    data_config,
                    entry["path"],
                    entry["name"],
                    action_horizon,
                    split,
                    use_trajectory_ratio=float(entry["use_trajectory_ratio"]),
                    norm_stats_domain_name=entry["norm_stats_domain_name"],
                )
                unique_dataset_cache[representative_idx] = domain_dataset
            else:
                base_dataset = unique_dataset_cache[representative_idx]
                domain_dataset = copy.copy(base_dataset)
                domain_dataset.domain_name = entry["name"]
                domain_dataset.dataset_path = entry["path"]
                domain_dataset.use_trajectory_ratio = float(entry["use_trajectory_ratio"])
                domain_dataset.norm_stats_domain_name = entry["norm_stats_domain_name"]
                domain_dataset.assets_dir = os.path.join(
                    data_config.assets_dirs,
                    data_config.repo_id,
                    domain_dataset.norm_stats_domain_name,
                )
                domain_dataset._split_group_entries_cache = None
                domain_dataset.get_split_indices(split)
                domain_dataset.indices_org = copy.deepcopy(domain_dataset.indices)

            self.domain_dataset_list.append(domain_dataset)

        self.norm_stats = None
        self._generate_domain_dataset_length()

    def _generate_domain_dataset_length(self):
        self.domain_dataset_length = [0]
        for domain_dataset in self.domain_dataset_list:
            self.domain_dataset_length.append(len(domain_dataset))
        self.domain_dataset_length = np.array(self.domain_dataset_length)
        self.domain_dataset_length_cumsum = np.cumsum(self.domain_dataset_length)

    def __len__(self):
        return self.domain_dataset_length_cumsum[-1]

    def get_val_dataset(self):
        val_set = copy.copy(self)
        val_set.domain_dataset_list = [
            domain_dataset.get_val_dataset() for domain_dataset in val_set.domain_dataset_list
        ]
        val_set._generate_domain_dataset_length()
        return val_set

    def set_sample_ratio(self, sample_ratio: float):
        """Set sample ratio for datasets (random frame sampling after initialization).

        This is different from use_trajectory_ratio which is applied during initialization
        to select a fixed set of trajectories. This method randomly samples frames from
        the already-loaded trajectories.

        Args:
            sample_ratio: If float, applies to all datasets. If dict, maps dataset name to ratio.
        """
        if isinstance(sample_ratio, dict):
            # Apply dataset-specific ratios
            for idx, domain_dataset in enumerate(self.domain_dataset_list):
                dataset_name = self.domain_list[idx]  # domain_list contains dataset_name
                if dataset_name in sample_ratio:
                    domain_dataset.set_sample_ratio(sample_ratio[dataset_name])
        else:
            # Apply same ratio to all datasets
            for domain_dataset in self.domain_dataset_list:
                domain_dataset.set_sample_ratio(sample_ratio)
        self._generate_domain_dataset_length()

    def set_skip_normalization(self, value: bool):
        for domain_dataset in self.domain_dataset_list:
            domain_dataset.set_skip_normalization(value)
        self.skip_normalization = value

    def set_ignore_rgb(self, value: bool):
        for domain_dataset in self.domain_dataset_list:
            domain_dataset.set_ignore_rgb(value)

    def generate_norm_stats(
        self,
        batch_size,
        independent_norm_mode="joint",
        norm_time_dim=False,
        norm_type="zscore",
        sample_ratio: float = 1.0,
        num_workers=16,
        independent_norm_cache_max_size: int = None,
        image_channel_pool_max_size: int | None = None,
        image_channel_batch_samples: int | None = None,
        image_tactile_norm_mode: str | None = None,
        verbose: bool = True,
    ):
        """
        independent_norm_mode:
        # 1. none: only do independent normalization for tactile.
        # 2. joint: do independent normalization for tactile & hand joints.
        # 3. all: do independent normalization for tactile & hand joints & wrist pose.
        norm_time_dim: whether to do independent normalization on the time dimension.
        norm_type: normalization type, one of 'quantile' or 'zscore'.
        sample_ratio: the ratio of the dataset to be sampled.
        num_workers: the number of workers to be used.
        independent_norm_cache_max_size: maximum number of batches to cache (None = no limit). When exceeded, uses reservoir sampling to maintain a fixed-size cache.
        image_channel_pool_max_size: max channel vectors processed per image tactile key
            when updating incremental stats (0 = unlimited).
        image_channel_batch_samples: max channel vectors sampled per image tactile key
            from each batch before updating incremental stats (0 = use all vectors).
        image_tactile_norm_mode: normalization mode for image-type tactile:
            'channel_wise' (default) or 'image_norm'.
        verbose: whether to print one-time logs when image channel cache reaches cap.

        Note: For merged dataset, we iterate through each domain separately to avoid
        mixing data from different domains (which may have different shapes).

        For datasets sharing the same norm_stats_domain_name (i.e. multiple copies of
        the same source), only the copy with the highest use_trajectory_ratio is used to
        compute norm stats, since it best represents the full data distribution.
        """

        # Build a map: norm_stats_domain_name -> dataset with highest trajectory ratio
        best_per_domain: dict[str, ZarrDataset] = {}
        for domain_dataset in self.domain_dataset_list:
            # Resolve the effective norm domain (mirrors ZarrDataset.__init__ logic)
            norm_dir = os.path.join(
                domain_dataset.data_config.assets_dirs,
                domain_dataset.data_config.repo_id,
            )
            # assets_dir is already <assets_dirs>/<repo_id>/<norm_domain>
            norm_domain = os.path.relpath(domain_dataset.assets_dir, norm_dir)
            ratio = domain_dataset.use_trajectory_ratio
            if norm_domain not in best_per_domain or ratio > best_per_domain[norm_domain].use_trajectory_ratio:
                best_per_domain[norm_domain] = domain_dataset

        for norm_domain, domain_dataset in best_per_domain.items():
            print(
                f"[generate_norm_stats] Computing norm stats for domain '{norm_domain}' "
                f"using dataset '{domain_dataset.domain_name}' "
                f"(use_trajectory_ratio={domain_dataset.use_trajectory_ratio:.4f})"
            )
            domain_dataset.generate_independent_norm_stats(
                batch_size,
                independent_norm_mode,
                norm_time_dim,
                norm_type,
                sample_ratio,
                num_workers,
                independent_norm_cache_max_size,
                image_channel_pool_max_size,
                image_channel_batch_samples,
                image_tactile_norm_mode,
                verbose,
            )

        self.set_sample_ratio(sample_ratio)
        self.set_skip_normalization(True)
        self.set_ignore_rgb(True)

        data_cache_action_mask = None
        if independent_norm_mode != "all":
            data_cache_action = list()
            data_cache_state = list()
            data_cache_action_mask = list()

            # Iterate through each domain separately to avoid mixing different shapes
            for domain_name, domain_dataset in zip(self.domain_list, self.domain_dataset_list):
                domain_dataset.set_sample_ratio(sample_ratio)
                domain_dataset.set_skip_normalization(True)
                domain_dataset.set_ignore_rgb(True)

                domain_data_loader = torch.utils.data.DataLoader(
                    dataset=domain_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    collate_fn=collate_fn_handle_strings,
                )

                for batch in tqdm(
                    domain_data_loader, desc=f"iterating {domain_name} dataset to get share keys normalization"
                ):
                    data_cache_action.append(batch["actions"])
                    data_cache_state.append(batch["state"])
                    data_cache_action_mask.append(batch["action_mask"])
        if independent_norm_mode == "all":
            data_cache = {
                "norm_time_dim": norm_time_dim,
                "norm_type": norm_type,
                "data_cache": dict(),
                "data_norm_dim": dict(),
            }
        elif independent_norm_mode == "joint":
            action_norm_dim = list(range(9))
            action_norm_dim = action_norm_dim + list(
                range(self.single_arm_action_rep_dim, self.single_arm_action_rep_dim + 9)
            )
            action_norm_dim = action_norm_dim + list(
                range(2 * self.single_arm_action_rep_dim, 2 * self.single_arm_action_rep_dim + 9)
            )
            data_cache = {
                "norm_time_dim": norm_time_dim,
                "norm_type": norm_type,
                "data_cache": {
                    "actions": data_cache_action,
                    "state": data_cache_state,
                },
                "data_norm_dim": {
                    "actions": action_norm_dim,
                    "state": action_norm_dim,
                },
            }
        elif independent_norm_mode == "none":
            data_cache = {
                "norm_time_dim": norm_time_dim,
                "norm_type": norm_type,
                "data_cache": {
                    "actions": data_cache_action,
                    "state": data_cache_state,
                },
                "data_norm_dim": {
                    "actions": None,
                    "state": None,
                },
            }
        else:
            raise ValueError(
                f"Invalid independent_norm_mode: {independent_norm_mode}, please choose from ['none', 'hand', 'all']"
            )

        self.norm_stats = calculate_norm_stats(data_cache, data_cache_action_mask=data_cache_action_mask)
        norm_stats_to_store = norm_stats_to_jsonable(self.norm_stats)
        dump_json_with_inline_lists(
            norm_stats_to_store,
            os.path.join(
                self.assets_dir, f"share_norm_stats_{independent_norm_mode}_t{int(norm_time_dim)}_{norm_type}.json"
            ),
            indent=4,
        )

        self.set_sample_ratio(1.0)
        self.set_skip_normalization(False)
        self.set_ignore_rgb(False)

    def load_norm_stats(self, independent_norm_mode, norm_time_dim, norm_type):
        norm_stats_path = (
            pathlib.Path(self.assets_dir)
            / f"share_norm_stats_{independent_norm_mode}_t{int(norm_time_dim)}_{norm_type}.json"
        )
        print(f"Loading shared norm stats from {norm_stats_path}")
        old_repo_id = getattr(self.data_config, "reuse_norm_repo_id", "")
        override_path = None
        if old_repo_id:
            override_path = _build_old_norm_stats_path(
                self.data_config.assets_dirs,
                old_repo_id,
                norm_stats_path.name,
            )
        norm_stats_json, replaced_keys = load_norm_stats_jsonable_with_override(norm_stats_path, override_path)
        if override_path is not None:
            _log_norm_override_result(
                current_path=norm_stats_path,
                override_path=override_path,
                replaced_keys=replaced_keys,
            )
        self.norm_stats = norm_stats_from_jsonable(norm_stats_json)

        for domain_dataset in self.domain_dataset_list:
            domain_dataset.load_independent_norm_stats(independent_norm_mode, norm_time_dim, norm_type)

    def generate_contact_detection_thresholds(self) -> dict:
        domain_stats = {}
        for domain_name, domain_dataset in zip(self.domain_list, self.domain_dataset_list):
            domain_stats[domain_name] = domain_dataset.generate_contact_detection_thresholds()
        return {
            "detector_version": CONTACT_DETECTOR_VERSION,
            "domains": domain_stats,
        }

    def generate_action_group_frequency_stats(
        self,
        batch_size: int,
        sample_ratio: float = 1.0,
        num_workers: int = 16,
        split: str | None = None,
    ) -> dict:
        split_name = split or self.split
        group_slices = get_ftp1_action_group_slices(
            self.single_arm_action_rep_dim,
            FTP1_SINGLE_ARM_JOINT_DIM,
            self.data_config.reserved_action_dim,
        )
        group_names = list(group_slices.keys())
        total_sample_count = 0
        total_present_count = {group_name: 0 for group_name in group_names}
        domain_statistics = {}

        self.set_sample_ratio(sample_ratio)
        self.set_skip_normalization(True)
        self.set_ignore_rgb(True)

        try:
            for domain_name, domain_dataset in zip(self.domain_list, self.domain_dataset_list):
                domain_dataset.set_sample_ratio(sample_ratio)
                domain_dataset.set_skip_normalization(True)
                domain_dataset.set_ignore_rgb(True)

                domain_present_count = {group_name: 0 for group_name in group_names}
                domain_sample_count = len(domain_dataset)
                if domain_sample_count > 0:
                    sample = domain_dataset[0]
                    _, sample_present_count = count_action_group_presence(
                        sample["action_mask"],
                        group_slices,
                    )
                    for group_name in group_names:
                        if sample_present_count[group_name] > 0:
                            domain_present_count[group_name] = domain_sample_count
                            total_present_count[group_name] += domain_sample_count
                total_sample_count += domain_sample_count

                if domain_sample_count > 0:
                    domain_frequency = {
                        group_name: domain_present_count[group_name] / domain_sample_count for group_name in group_names
                    }
                else:
                    domain_frequency = {group_name: 0.0 for group_name in group_names}

                domain_statistics[domain_name] = {
                    "sample_count": domain_sample_count,
                    "group_present_count": domain_present_count,
                    "group_frequency": domain_frequency,
                }
        finally:
            self.set_sample_ratio(1.0)
            self.set_skip_normalization(False)
            self.set_ignore_rgb(False)

        group_frequency = {}
        for group_name in group_names:
            if total_sample_count > 0:
                group_frequency[group_name] = total_present_count[group_name] / total_sample_count
            else:
                group_frequency[group_name] = 0.0

        dataset_config_path = pathlib.Path(self.data_config.dataset_config_path).expanduser()
        if dataset_config_path.exists():
            with open(dataset_config_path, "rb") as f:
                dataset_config_sha1 = hashlib.sha1(f.read()).hexdigest()
        else:
            dataset_config_sha1 = "missing"

        stats = {
            "split": split_name,
            "sample_ratio": sample_ratio,
            "sample_count": total_sample_count,
            "counting_method": "domain_based_group_presence",
            "repo_id": self.data_config.repo_id,
            "dataset_config_path": str(dataset_config_path),
            "dataset_config_sha1": dataset_config_sha1,
            "single_arm_action_rep_dim": self.single_arm_action_rep_dim,
            "arm_joints_dim": FTP1_SINGLE_ARM_JOINT_DIM,
            "reserved_action_dim": self.data_config.reserved_action_dim,
            "group_names": group_names,
            "group_present_count": total_present_count,
            "group_frequency": group_frequency,
            "domain_statistics": domain_statistics,
        }

        stats_path = get_action_group_stats_path(self.assets_dir, split=split_name)
        dump_json_with_inline_lists(stats, stats_path, indent=2)
        print(f"Action group frequency stats saved to: {stats_path}")
        return stats

    @property
    def default_action_group_frequency_stats_path(self):
        return get_action_group_stats_path(self.assets_dir, split=self.split)

    def generate_tactile_input_config(self, save_path: str = None):
        tactile_input_config = dict()
        for domain_dataset in self.domain_dataset_list:
            domain_name = domain_dataset.domain_name
            tactile_input_config[domain_name] = dict()
            data_sample = domain_dataset[0]
            for key in data_sample["tactile"].keys():
                data = data_sample["tactile"][key]
                data_function_area = data_sample["tactile_function_area"][key]
                data_sensor = data_sample["tactile_sensor"][key]
                data_type = data_sample["tactile_type"][key]
                tactile_input_config[domain_name][key] = {
                    "shape": data.shape,
                    "function_areas": data_function_area.astype(np.int32).tolist(),
                    "sensor": data_sensor,
                    "type": data_type,
                }

        if save_path is None:
            save_path = os.path.join(self.assets_dir, f"tactile_input_config.json")
        dump_json_with_inline_lists(tactile_input_config, save_path, indent=4)
        return save_path

    @property
    def default_tactile_input_config_path(self):
        return os.path.join(self.assets_dir, f"tactile_input_config.json")

    def __getitem__(self, idx: SupportsIndex) -> Dict:
        domain_dataset_idx = np.searchsorted(self.domain_dataset_length_cumsum, idx, side="right") - 1
        domain_dataset_idx_inside = idx - self.domain_dataset_length_cumsum[domain_dataset_idx]
        data = self.domain_dataset_list[domain_dataset_idx][domain_dataset_idx_inside]
        # print(self.__len__(), idx, self.domain_dataset_length_cumsum, domain_dataset_idx, domain_dataset_idx_inside)
        if not self.skip_normalization:
            if self.norm_stats is None:
                raise ValueError(f"Multi-domain dataset | Norm stats not found. Please run load_norm_stats() first.")
            data = apply_norm_stats(data, self.norm_stats, unnormalize=False)
        data["data_idx"] = idx
        return data


# ========================= Usage Process =========================
# 1. create train_dataset, the train-val-split will be create automatically.
# 2. val_dataset = train_dataset.get_val_dataset()
# 3. run train_dataset.generate_norm_stats(), the norm_stats will be generated and save locally.
# 4. run train_dataset.load_norm_stats() & val_dataset.load_norm_stats() before training.
