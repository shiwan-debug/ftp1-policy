from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


def _canonicalize_tactile_metadata_value(value: Any, field_name: str) -> str:
    """Convert nested / array-backed tactile metadata into a stable string."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _canonicalize_tactile_metadata_value(value.item(), field_name)
        if value.size == 0:
            raise ValueError(f"Empty {field_name} array")
        return _canonicalize_tactile_metadata_value(value.reshape(-1)[0], field_name)
    if isinstance(value, list | tuple):
        if len(value) == 0:
            raise ValueError(f"Empty {field_name} sequence")
        return _canonicalize_tactile_metadata_value(value[0], field_name)
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def canonicalize_tactile_sensor_name(sensor: Any) -> str:
    return _canonicalize_tactile_metadata_value(sensor, "tactile sensor")


def canonicalize_tactile_encoder_type(encoder_type: Any) -> str:
    return _canonicalize_tactile_metadata_value(encoder_type, "tactile type")


def get_tactile_encoder_shape_from_config_shape(config_shape: Sequence[int]) -> tuple[int, ...]:
    """Return the per-area tactile shape from config shape [T, N, *D]."""
    shape = tuple(int(x) for x in config_shape)
    if len(shape) < 3:
        raise ValueError(f"Expected config shape [T, N, *D], got {shape}")
    return shape[2:]


def get_tactile_encoder_shape_from_data_shape(
    data_shape: Sequence[int],                           # (T, N, *tac_shape) or (B, T, N, *tac_shape)
    tactile_function_areas: Any | None = None,           # (N,) or (B, N)
) -> tuple[int, ...]:
    """Return the per-area tactile shape from runtime data shape.

    Sample-level tactile arrays are typically shaped (T, N, *D), while batched arrays
    are (B, T, N, *D). If tactile_function_areas are provided, we use its rank to infer
    how many leading non-feature dimensions to drop robustly in both cases. Batched
    callers should always pass tactile_function_areas so both B and N are excluded.
    """
    shape = tuple(int(x) for x in data_shape)
    if tactile_function_areas is None:
        # Used by single-sample tactile tensors and config-style [T, N, *D] shapes.
        if len(shape) < 3:
            raise ValueError(f"Expected tactile data shape with at least 3 dims, got {shape}")
        return shape[2:]

    if isinstance(tactile_function_areas, torch.Tensor):
        func_area_ndim = tactile_function_areas.dim()   # (B, N) -> 2
    else:
        func_area_ndim = np.asarray(tactile_function_areas).ndim
    leading_dims = 1 + int(func_area_ndim)   # (B, T, N, *tac_shape) -> 3
    if len(shape) <= leading_dims:
        raise ValueError(
            f"Cannot infer tactile encoder shape from data shape {shape} with tactile_function_areas ndim={func_area_ndim}"
        )
    return shape[leading_dims:]


def build_tactile_sensor_type_shape_key(
    sensor: Any,
    encoder_type: Any,
    tactile_encoder_shape: Sequence[int],
) -> str:
    sensor_name = canonicalize_tactile_sensor_name(sensor)
    tactile_type = canonicalize_tactile_encoder_type(encoder_type)
    shape = tuple(int(x) for x in tactile_encoder_shape)
    if len(shape) == 0:
        raise ValueError(f"Expected non-empty tactile encoder shape, got {shape}")
    return f"{sensor_name}_{tactile_type}_{'_'.join(map(str, shape))}"


def build_tactile_sensor_type_shape_key_from_config(
    sensor: Any,
    encoder_type: Any,
    config_shape: Sequence[int],
) -> str:
    return build_tactile_sensor_type_shape_key(
        sensor,
        encoder_type,
        get_tactile_encoder_shape_from_config_shape(config_shape),
    )


def build_tactile_sensor_type_shape_key_from_data(
    sensor: Any,
    encoder_type: Any,
    tactile_data: Any,                           # (T, N, *tac_shape) or (B, T, N, *tac_shape)
    *,
    tactile_function_areas: Any | None = None,   # (N,) or (B, N)
) -> str:
    if isinstance(tactile_data, torch.Tensor):
        data_shape = tuple(int(x) for x in tactile_data.shape)
    else:
        data_shape = tuple(int(x) for x in np.asarray(tactile_data).shape)
    tactile_shape = get_tactile_encoder_shape_from_data_shape(
        data_shape,
        tactile_function_areas=tactile_function_areas,
    )
    return build_tactile_sensor_type_shape_key(sensor, encoder_type, tactile_shape)
