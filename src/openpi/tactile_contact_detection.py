import copy
import json
import pathlib
from typing import Any

import numpy as np

TACTILE_DATA_KEY_PREFIXES = ("left_tactile_data_", "right_tactile_data_")
CONTACT_DETECTOR_VERSION = "delta_mad_v1"
CONTACT_THRESHOLD_FILENAME = "contact_detection_thresholds.json"
ROBUST_MAD_SCALE = 1.4826
DEFAULT_CONTACT_THRESHOLD_K = 3.0
CONTACT_SCORE_QUANTILES = (
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
)


def is_tactile_data_key(key: str) -> bool:
    return key.startswith(TACTILE_DATA_KEY_PREFIXES)


def compute_tactile_delta_score(previous_frame: np.ndarray, current_frame: np.ndarray) -> float:
    previous = np.asarray(previous_frame, dtype=np.float32)
    current = np.asarray(current_frame, dtype=np.float32)
    if previous.shape != current.shape:
        raise ValueError(f"Tactile frame shape mismatch: {previous.shape} vs {current.shape}")
    if previous.size == 0:
        return 0.0

    delta = np.abs(current - previous).reshape(-1)
    finite_delta = delta[np.isfinite(delta)]
    if finite_delta.size == 0:
        return 0.0
    return float(finite_delta.mean())


def compute_tactile_delta_scores(frames: np.ndarray) -> np.ndarray:
    tactile = np.asarray(frames, dtype=np.float32)
    if tactile.ndim == 0 or tactile.shape[0] < 2:
        return np.zeros((0,), dtype=np.float32)

    flat_delta = np.abs(tactile[1:] - tactile[:-1]).reshape(tactile.shape[0] - 1, -1)
    finite_mask = np.isfinite(flat_delta)
    if np.all(finite_mask):
        return flat_delta.mean(axis=1, dtype=np.float64).astype(np.float32, copy=False)

    valid_counts = finite_mask.sum(axis=1)
    delta_sums = np.where(finite_mask, flat_delta, 0.0).sum(axis=1, dtype=np.float64)
    scores = np.zeros((flat_delta.shape[0],), dtype=np.float32)
    valid_rows = valid_counts > 0
    scores[valid_rows] = (delta_sums[valid_rows] / valid_counts[valid_rows]).astype(np.float32, copy=False)
    return scores


def summarize_score_distribution(scores: np.ndarray) -> dict[str, Any]:
    score_array = np.asarray(scores, dtype=np.float32).reshape(-1)
    finite_scores = score_array[np.isfinite(score_array)]

    summary: dict[str, Any] = {
        "count": int(finite_scores.size),
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "quantiles": {label: None for label, _ in CONTACT_SCORE_QUANTILES},
    }
    if finite_scores.size == 0:
        return summary

    finite_scores = finite_scores.astype(np.float64, copy=False)
    quantile_probs = [prob for _, prob in CONTACT_SCORE_QUANTILES]
    quantile_values = np.quantile(finite_scores, quantile_probs)
    summary.update(
        {
            "min": float(np.min(finite_scores)),
            "max": float(np.max(finite_scores)),
            "mean": float(np.mean(finite_scores)),
            "std": float(np.std(finite_scores)),
            "quantiles": {
                label: float(value) for (label, _), value in zip(CONTACT_SCORE_QUANTILES, quantile_values, strict=False)
            },
        }
    )
    return summary


def estimate_contact_threshold(
    scores: np.ndarray,
    *,
    threshold_k: float = DEFAULT_CONTACT_THRESHOLD_K,
) -> dict[str, Any]:
    score_array = np.asarray(scores, dtype=np.float32).reshape(-1)
    finite_scores = score_array[np.isfinite(score_array)]
    positive_scores = finite_scores[finite_scores > 0]
    finite_score_summary = summarize_score_distribution(finite_scores)
    positive_score_summary = summarize_score_distribution(positive_scores)

    stats: dict[str, Any] = {
        "threshold": None,
        "threshold_k": float(threshold_k),
        "robust_mad_scale": ROBUST_MAD_SCALE,
        "score_count": int(finite_scores.size),
        "positive_score_count": int(positive_scores.size),
        "finite_score_summary": finite_score_summary,
        "positive_score_summary": positive_score_summary,
        "score_median": None,
        "noise_pool_count": 0,
        "noise_median": None,
        "noise_mad": None,
        "robust_std": None,
        "positive_score_exceed_threshold_count": 0,
        "positive_score_exceed_threshold_fraction": None,
        "positive_score_threshold_quantile": None,
        "status": "no_positive_scores",
    }
    if positive_scores.size == 0:
        return stats

    score_median = float(np.median(positive_scores))
    noise_pool = positive_scores[positive_scores <= score_median]
    if noise_pool.size == 0:
        noise_pool = positive_scores
    noise_median = float(np.median(noise_pool))
    noise_mad = float(np.median(np.abs(noise_pool - noise_median)))
    robust_std = float(ROBUST_MAD_SCALE * noise_mad)
    threshold = max(0.0, noise_median + float(threshold_k) * robust_std)
    exceed_threshold_count = int(np.count_nonzero(positive_scores > threshold))

    stats.update(
        {
            "threshold": float(threshold),
            "score_median": score_median,
            "noise_pool_count": int(noise_pool.size),
            "noise_median": noise_median,
            "noise_mad": noise_mad,
            "robust_std": robust_std,
            "positive_score_exceed_threshold_count": exceed_threshold_count,
            "positive_score_exceed_threshold_fraction": float(exceed_threshold_count / positive_scores.size),
            "positive_score_threshold_quantile": float(1.0 - exceed_threshold_count / positive_scores.size),
            "status": "ok",
        }
    )
    return stats


def build_contact_thresholds_jsonable(
    stream_scores: dict[str, np.ndarray],
    *,
    threshold_k: float = DEFAULT_CONTACT_THRESHOLD_K,
) -> dict[str, Any]:
    return {
        "detector_version": CONTACT_DETECTOR_VERSION,
        "score_type": "mean_abs_delta",
        "threshold_k": float(threshold_k),
        "robust_mad_scale": ROBUST_MAD_SCALE,
        "streams": {
            stream_key: estimate_contact_threshold(scores, threshold_k=threshold_k)
            for stream_key, scores in sorted(stream_scores.items())
        },
    }


def merge_contact_thresholds_jsonable(
    current_stats: dict[str, Any],
    override_stats: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    merged = copy.deepcopy(current_stats)
    merged_streams = copy.deepcopy(current_stats.get("streams", {}))
    replaced_keys: list[str] = []

    override_streams = override_stats.get("streams", {})
    if not isinstance(override_streams, dict):
        return merged, replaced_keys

    for stream_key, current_value in current_stats.get("streams", {}).items():
        if stream_key not in override_streams:
            continue
        if current_value != override_streams[stream_key]:
            merged_streams[stream_key] = copy.deepcopy(override_streams[stream_key])
            replaced_keys.append(f"streams.{stream_key}")

    merged["streams"] = merged_streams
    return merged, replaced_keys


def load_contact_thresholds_jsonable_with_override(
    current_path: str | pathlib.Path,
    override_path: str | pathlib.Path | None = None,
) -> tuple[dict[str, Any], list[str]]:
    current_path = pathlib.Path(current_path)
    with current_path.open() as f:
        current_stats = json.load(f)

    if override_path is None:
        return current_stats, []

    override_path = pathlib.Path(override_path)
    if not override_path.exists():
        return current_stats, []

    with override_path.open() as f:
        override_stats = json.load(f)

    return merge_contact_thresholds_jsonable(current_stats, override_stats)


def get_stream_contact_threshold(contact_thresholds: dict[str, Any] | None, stream_key: str) -> float | None:
    if not isinstance(contact_thresholds, dict):
        return None
    stream_stats = contact_thresholds.get("streams", {}).get(stream_key)
    if not isinstance(stream_stats, dict):
        return None
    threshold = stream_stats.get("threshold")
    if threshold is None:
        return None
    return float(threshold)


def is_contact_score_above_threshold(score: float, threshold: float) -> bool:
    return float(score) > float(threshold)


def require_stream_contact_threshold(
    contact_thresholds: dict[str, Any] | None,
    stream_key: str,
    *,
    domain_name: str | None = None,
    threshold_path: str | pathlib.Path | None = None,
) -> float:
    threshold = get_stream_contact_threshold(contact_thresholds, stream_key)
    if threshold is not None:
        return threshold

    stream_stats = {}
    if isinstance(contact_thresholds, dict):
        stream_stats = contact_thresholds.get("streams", {})
    stream_prefix = f"Domain {domain_name} | " if domain_name else ""
    path_suffix = f" Threshold file: {threshold_path}." if threshold_path is not None else ""

    if stream_key not in stream_stats:
        available_streams = ", ".join(sorted(stream_stats.keys())[:10]) or "<none>"
        raise ValueError(
            f"{stream_prefix}Missing contact threshold for stream '{stream_key}'. "
            f"Available streams: {available_streams}.{path_suffix}"
        )

    stream_status = stream_stats[stream_key].get("status")
    raise ValueError(
        f"{stream_prefix}Contact threshold for stream '{stream_key}' is unavailable "
        f"(status={stream_status!r}).{path_suffix}"
    )
