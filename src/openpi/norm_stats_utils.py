import copy
import json
import pathlib
from typing import Any

from openpi.json_utils import dump_json_with_inline_lists

_NORM_STATS_MERGE_SECTIONS = ("params", "norm_dim", "data_dim")


def _replace_overlapping_keys(
    current_value: dict[str, Any],
    override_value: dict[str, Any],
    *,
    prefix: str,
) -> tuple[dict[str, Any], list[str]]:
    merged = copy.deepcopy(current_value)
    replaced_keys: list[str] = []

    for key, value in current_value.items():
        if key not in override_value:
            continue

        override_item = override_value[key]
        key_path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and isinstance(override_item, dict):
            merged[key], child_keys = _replace_overlapping_keys(
                value,
                override_item,
                prefix=key_path,
            )
            replaced_keys.extend(child_keys)
        else:
            merged[key] = copy.deepcopy(override_item)
            replaced_keys.append(key_path)

    return merged, replaced_keys


def merge_norm_stats_jsonable(
    current_stats: dict[str, Any],
    override_stats: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    merged = copy.deepcopy(current_stats)
    replaced_keys: list[str] = []

    for section in _NORM_STATS_MERGE_SECTIONS:
        current_section = current_stats.get(section)
        override_section = override_stats.get(section)
        if not isinstance(current_section, dict) or not isinstance(override_section, dict):
            continue

        merged_section, section_keys = _replace_overlapping_keys(
            current_section,
            override_section,
            prefix=section,
        )
        merged[section] = merged_section
        replaced_keys.extend(section_keys)

    return merged, replaced_keys


def load_norm_stats_jsonable_with_override(
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

    return merge_norm_stats_jsonable(current_stats, override_stats)


def write_norm_stats_jsonable_with_override(
    current_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
    override_path: str | pathlib.Path | None = None,
) -> list[str]:
    merged_stats, replaced_keys = load_norm_stats_jsonable_with_override(current_path, override_path)
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json_with_inline_lists(merged_stats, output_path, indent=4)
    return replaced_keys
