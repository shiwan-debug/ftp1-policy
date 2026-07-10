"""Convert TouchInTheWild tactile datasets into our zarr replay buffer format.

The TouchInTheWild zarr.zip files ship RGB wrist videos plus a single
`tactile` array and gripper state. We expose them as
`right_wrist_camera_rgb` and `right_tactile_data_gripper` so
``visualize_zarr_data.py`` and downstream code can use the same
"no ego camera" branch as ViTaMIn.

Directory structure under the root typically looks like::

    TouchInTheWild/
        four_tasks/
            fluid_transfer/fluid_transfer.zarr.zip
            pencil_insertion/pencil_insertion.zarr.zip
            test_tube_collection/test_tube_collection.zarr.zip
            whiteboard_erasing/whiteboard_erasing.zarr.zip
        indoor_data/
            hex_key_insertion/hex_key_insertion.zarr.zip
            ...
        pretrain_data/   # <-- ignored by this script

Under the output directory, we mirror this folder structure, e.g.::

    output/touchinthewild/
        four_tasks/fluid_transfer/fluid_transfer.zarr/
        indoor_data/hex_key_insertion/hex_key_insertion.zarr/

This script closely mirrors ``parse_data_vitamin.py`` and exposes the
same CLI options (stride, episode limits, GPT instruction expansion,
etc.).
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import zarr
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm

from common.replay_buffer import ReplayBuffer


def _normalize_dataset_name(name: str) -> str:
    """Normalize dataset-name-like strings for instructions.

    Remove trailing "_dataset"/"dataset" and convert underscores to
    spaces, mirroring the high_level_task_instruction logic.
    """
    base = str(name)
    base = base.replace("_dataset", "")
    base = base.replace("dataset", "")
    base = base.replace("_", " ")
    base = base.strip()
    return base or str(name)


def _open_zip_group(zip_path: Path) -> tuple[zarr.Group, zarr.storage.ZipStore]:
    """Open a .zarr.zip file as a zarr.Group and return (group, store)."""
    store = zarr.storage.ZipStore(str(zip_path), mode="r")
    group = zarr.open_group(store=store, mode="r")
    return group, store


def _iter_episode_slices(episode_ends: np.ndarray):
    """Yield slices [start:end] for each episode given episode_ends array."""
    start = 0
    for end in episode_ends.tolist():
        yield slice(start, end)
        start = end


def process_touchinthewild_dataset(
    dataset_path: Path,
    root_dir: Path,
    save_root: Path,
    stride: int = 1,
    start_episode: int = 0,
    max_episodes: Optional[int] = None,
    max_steps: Optional[int] = None,
    instruction: Optional[str] = None,
    instruction_pad: int = 0,
    gripper_idx: int = 28,
    overwrite: bool = False,
    use_gpt_instruction: bool = False,
    qwen_client=None,
    gpt_cfg: Optional[Dict] = None,
    expanded_task_map: Optional[Dict[str, List[str]]] = None,
    task_base_map: Optional[Dict[str, str]] = None,
    downsample_ratio: int = 2,
) -> None:
    """Convert a single TouchInTheWild *.zarr.zip into a replay buffer.

    Parameters mirror ``process_vitamin_dataset`` but with extra
    ``root_dir`` / ``save_root`` to support mirroring the directory
    hierarchy under the output path.
    """

    dataset_name = dataset_path.name.replace(".zarr.zip", "")

    # Mirror original folder structure under save_root, excluding root_dir
    # e.g. ROOT/four_tasks/fluid_transfer/fluid_transfer.zarr.zip ->
    #      OUTPUT/four_tasks/fluid_transfer/fluid_transfer.zarr
    try:
        rel_parent = dataset_path.parent.relative_to(root_dir)
    except ValueError:
        # If dataset_path is not under root_dir for some reason, fall back
        # to dumping directly under save_root.
        rel_parent = Path(".")

    save_dir = (save_root / rel_parent).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{dataset_name}.zarr"

    if out_path.exists() and overwrite:
        shutil.rmtree(out_path)
    if out_path.exists():
        print(f"{out_path} exists, skipping. Use --overwrite to rebuild.")
        return

    group, store = _open_zip_group(dataset_path)
    try:
        data = group["data"]
        meta = group["meta"]
        episode_ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)

        rgb = data["camera0_rgb"]
        tactile = data["camera0_tactile"]  # (T, 12, 64) float32
        eef_pos = data["robot0_eef_pos"]
        eef_rot = data["robot0_eef_rot_axis_angle"]
        # Original TouchInTheWild stores gripper width in decimeters.
        # We convert to meters in the replay buffer for consistency.
        gripper_width_dm = data["robot0_gripper_width"]

        replay_buffer = ReplayBuffer.create_from_path(out_path, mode="a")

        # Per-dataset instruction candidate list (like one task in RH20T)
        instruction_candidate_list: Optional[List[str]] = None
        fill_len = (
            int(gpt_cfg.get("fill_instruction_length", 100))
            if gpt_cfg
            else max(instruction_pad, 0)
        )
        processed = 0

        for ep_idx, ep_slice in enumerate(_iter_episode_slices(episode_ends)):
            if ep_idx < start_episode:
                continue
            if max_episodes is not None and processed >= max_episodes:
                break

            step_slice = slice(ep_slice.start, ep_slice.stop, stride)
            if max_steps is not None:
                stop = min(ep_slice.stop, ep_slice.start + max_steps * stride)
                step_slice = slice(ep_slice.start, stop, stride)

            # Synthetic timestamps: 0..T-1 after stride / truncation
            T_len = (step_slice.stop - step_slice.start - 1) // stride + 1
            if T_len <= 0:
                continue
            ts = np.arange(0, T_len, dtype=np.int64)

            pos = eef_pos[step_slice].astype(np.float32)
            # Rotate the wrist orientation around its own local z-axis
            # by +90 degrees (counterclockwise in its frame) while keeping
            # the position unchanged.
            rot_vec = eef_rot[step_slice].astype(np.float32)
            R_orig = Rotation.from_rotvec(rot_vec)
            R_z = Rotation.from_rotvec(
                np.array([0.0, 0.0, np.pi / 2.0], dtype=np.float32)
            )
            R_new = R_orig * R_z
            rot = R_new.as_rotvec().astype(np.float32)

            pose = np.concatenate([pos, rot], axis=-1)  # (T, 6)

            # Base episode fields (same style as ViTaMIn)
            # Convert gripper width from decimeters to meters.
            width_m = gripper_width_dm[step_slice].astype(np.float32) * 0.1

            episode = {
                "timestamps": ts,
                "right_wrist_pose": pose,
                "right_wrist_camera_rgb": rgb[step_slice].astype(np.uint8),
                # Right-hand joint stream uses width in meters.
                "right_hand_joints": width_m.reshape(T_len, -1),
                "right_hand_joints_idx": np.full(
                    (T_len, 1), gripper_idx, dtype=np.int64
                ),
            }

            # Also expose an explicit robot0_gripper_width_m stream
            # so that inspect_zarr_stats-like tools see the correct
            # units directly.
            episode["robot0_gripper_width_m"] = width_m.reshape(T_len, -1)

            # TouchInTheWild tactile: (T, 12, 64) float32
            # Dataset authors report this as two 12x32 sensors
            # concatenated along the last dimension. We reshape and
            # split into two channels with labels 0 and 1, mirroring
            # the ViTaMIn convention.
            tac = tactile[step_slice].astype(np.float32)
            if tac.ndim != 3 or tac.shape[1] != 12 or tac.shape[2] != 64:
                raise ValueError(
                    f"Unexpected camera0_tactile shape {tac.shape}, expected (T, 12, 64)."
                )

            # (T, 12, 64) -> two sensors of (12, 32)
            sensor0 = tac[:, :, :32]  # (T, 12, 32)
            sensor1 = tac[:, :, 32:]  # (T, 12, 32)
            right_tactile = np.stack([sensor0, sensor1], axis=1)  # (T, 2, 12, 32)

            episode["right_tactile_data_gripper"] = right_tactile
            episode["right_tactile_area_gripper"] = np.tile(
                np.array([[0, 1]], dtype=np.int64), (T_len, 1)
            )
            episode["right_tactile_sensor_gripper"] = np.array(
                ["3DViTac"] * T_len
            )
            episode["right_tactile_type_gripper"] = np.array(["state"] * T_len)

            # -------- instruction handling (use task_description_base if available) --------
            if instruction is not None:
                base_text = instruction
            else:
                # Prefer task_description_base entry keyed by dataset_name
                # (e.g. "fluid_transfer") when available.
                if task_base_map is not None and dataset_name in task_base_map:
                    base_text = str(task_base_map[dataset_name])
                else:
                    base_text = _normalize_dataset_name(dataset_name)
            if fill_len <= 0:
                fill_len = max(len(base_text), 1)
            base_text_padded = base_text.ljust(fill_len)

            if use_gpt_instruction and qwen_client is not None:
                # Lazily initialize per-dataset candidate pool using expanded_task_map or Qwen
                if instruction_candidate_list is None:
                    key = dataset_name
                    if expanded_task_map is not None and key in expanded_task_map:
                        # Already cached on disk
                        instruction_candidate_list = [
                            str(s).ljust(fill_len) for s in expanded_task_map[key]
                        ]
                    else:
                        from common.gpt_instruction_expansion import (
                            get_instruction_expansion_episodes_qwen,
                        )

                        # Build minimal episode for GPT using wrist RGB
                        episode_for_gpt = {
                            "timestamps": episode["timestamps"],
                            "right_wrist_camera_rgb": episode[
                                "right_wrist_camera_rgb"
                            ],
                            "sub_task_instruction": np.array([base_text]),
                        }

                        instructions: List[str] = []
                        try:
                            _, instructions = get_instruction_expansion_episodes_qwen(
                                episode_for_gpt,
                                image_key="right_wrist_camera_rgb",
                                n_image=int(
                                    gpt_cfg.get("n_image", 5) if gpt_cfg else 5
                                ),
                                instruction_key="sub_task_instruction",
                                client=qwen_client,
                                embodiment=(
                                    gpt_cfg.get("embodiment", "robot")
                                    if gpt_cfg
                                    else "robot"
                                ),
                                n_new_instructions=int(
                                    gpt_cfg.get("n_new_instructions", 20)
                                    if gpt_cfg
                                    else 20
                                ),
                                fill_instruction_length=fill_len,
                                model="doubao-seed-1-6-vision-250815",
                            )
                        except Exception as e:  # pragma: no cover - best-effort
                            print(
                                f"[WARN] Qwen instruction expansion failed for {dataset_name}: {e}"
                            )
                            instructions = []

                        expanded_only = (
                            [inst.ljust(fill_len) for inst in instructions]
                            if instructions
                            else []
                        )
                        # Candidate list and JSON cache: expanded instructions
                        # plus 10 copies of the original base instruction.
                        raw_list = expanded_only + [base_text_padded] * 10
                        instruction_candidate_list = raw_list
                        if expanded_task_map is not None:
                            expanded_task_map[key] = raw_list

                # Sample one candidate and tile per frame
                if instruction_candidate_list:
                    while True:
                        choice = random.choice(instruction_candidate_list)
                        if len(choice) == fill_len:
                            break
                    episode["sub_task_instruction"] = np.array(
                        [choice] * T_len
                    )
            else:
                # No GPT: simple padding & tiling
                episode["sub_task_instruction"] = np.array(
                    [base_text_padded] * T_len
                )

            for key in episode.keys():
                episode[key] = episode[key][::downsample_ratio]

            replay_buffer.add_episode(episode, compressors="disk")
            processed += 1

            tqdm.write(
                f"{dataset_name} ep {ep_idx} -> steps {episode['timestamps'].shape[0]} (saved to {out_path})"
            )
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser("Parse TouchInTheWild zarr.zip datasets")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("SAMPLE_DATASETS/TouchInTheWild"),
        help="Directory containing TouchInTheWild folders (four_tasks, indoor_data, etc.)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/touchinthewild"),
        help="Directory to save converted zarr files (folder structure is mirrored).",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "Optional single dataset filename or relative path. "
            "Examples: 'fluid_transfer.zarr.zip' or "
            "'four_tasks/fluid_transfer/fluid_transfer.zarr.zip'."
        ),
    )
    parser.add_argument("--stride", type=int, default=1, help="Temporal downsample stride.")
    parser.add_argument(
        "--start_episode",
        type=int,
        default=0,
        help="Skip episodes before this index.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Limit number of episodes for debugging.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Truncate each episode to this many steps after stride.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Override base instruction text for all datasets.",
    )
    parser.add_argument(
        "--instruction_pad",
        type=int,
        default=0,
        help="Right-pad instruction when GPT is disabled.",
    )
    parser.add_argument(
        "--gripper_idx",
        type=int,
        default=28,
        help="Joint index used for the gripper width stream.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild output zarr even if it exists.",
    )
    parser.add_argument(
        "--use_gpt_instruction",
        action="store_true",
        help="Use Qwen-based instruction expansion (RH20T / ViTaMIn style).",
    )
    parser.add_argument(
        "--gpt_n_images",
        type=int,
        default=5,
        help="Number of frames to sample as images for GPT expansion.",
    )
    parser.add_argument(
        "--gpt_n_new_instructions",
        type=int,
        default=40,
        help="How many new instructions GPT should generate per dataset.",
    )
    parser.add_argument(
        "--expanded_task_list_json",
        type=Path,
        default=None,
        help=(
            "Path to expanded_task_list.json cache (key = dataset name). "
            "Used when --use_gpt_instruction is enabled."
        ),
    )
    parser.add_argument(
        "--downsample_ratio",
        type=int,
        default=2,
    )
    args = parser.parse_args()

    root: Path = args.root
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root}")

    # Collect dataset files, skipping anything under pretrain_data
    if args.dataset_name is None:
        dataset_files = sorted(
            p
            for p in root.rglob("*.zarr.zip")
            if "pretrain_data" not in p.parts
        )
    else:
        # Allow dataset_name to be either a bare filename or a relative path
        pattern = args.dataset_name
        dataset_files = sorted(
            p
            for p in root.rglob(pattern)
            if "pretrain_data" not in p.parts
        )

    if not dataset_files:
        raise FileNotFoundError(
            f"No *.zarr.zip found under {root} (dataset_name={args.dataset_name!r})."
        )

    # Optionally load task_description_base for TouchInTheWild
    task_base_map: Optional[Dict[str, str]] = None
    try:
        assets_dir = Path(__file__).resolve().parents[2] / "assets" / "TouchInTheWild"
        base_path = assets_dir / "task_description_base.json"
        if base_path.exists():
            import json

            with open(base_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                task_base_map = {
                    str(k): (v[0] if isinstance(v, list) and v else str(v))
                    for k, v in raw.items()
                }
    except Exception:
        task_base_map = None

    # Optional: Qwen client + on-disk cache for expanded per-dataset instructions
    qwen_client = None
    expanded_task_map: Dict[str, List[str]] = {}
    expanded_task_list_path: Optional[Path] = None
    gpt_cfg: Dict = {}

    if args.use_gpt_instruction:
        from common.gpt_instruction_expansion import get_openai_client

        qwen_client = get_openai_client()
        gpt_cfg = {
            "enabled": True,
            "image_key": "right_wrist_camera_rgb",
            "n_image": args.gpt_n_images,
            "n_new_instructions": args.gpt_n_new_instructions,
            "fill_instruction_length": 100,
            "embodiment": "robot",
        }

        if args.expanded_task_list_json is not None:
            import json

            expanded_task_list_path = args.expanded_task_list_json
            if expanded_task_list_path.is_file():
                try:
                    with open(expanded_task_list_path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                        if isinstance(loaded, dict):
                            # Normalize and also drop redundant copies of the
                            # base instruction so we don't keep 5 duplicates in JSON.
                            for k, v in loaded.items():
                                if not isinstance(v, list):
                                    continue
                                clean_list: List[str] = []
                                for s in v:
                                    text = str(s).strip()
                                    if "dataset" in text or text.replace(" ", "") == str(k).replace(" ", ""):
                                        text = _normalize_dataset_name(k)
                                    clean_list.append(text)
                                # Split into base-like vs expanded strings and
                                # keep at most one copy of the base text.
                                normalized = _normalize_dataset_name(k)
                                base_like = [t for t in clean_list if t.strip() == normalized]
                                others = [t for t in clean_list if t.strip() != normalized]
                                if others:
                                    expanded_task_map[str(k)] = others
                                else:
                                    # Fallback: keep at most a single base instruction.
                                    expanded_task_map[str(k)] = base_like[:1] if base_like else []
                except Exception as e:  # pragma: no cover - best-effort
                    print(
                        f"[WARN] Failed to load expanded_task_list_json at {expanded_task_list_path}: {e}"
                    )

    def _save_expanded_task_map() -> None:
        if expanded_task_list_path is None:
            return
        import json

        expanded_task_list_path.parent.mkdir(parents=True, exist_ok=True)
        with open(expanded_task_list_path, "w", encoding="utf-8") as f:
            json.dump(expanded_task_map, f, ensure_ascii=False, indent=2)

    for path in dataset_files:
        process_touchinthewild_dataset(
            dataset_path=path,
            root_dir=root,
            save_root=args.output,
            stride=args.stride,
            start_episode=args.start_episode,
            max_episodes=args.max_episodes,
            max_steps=args.max_steps,
            instruction=args.instruction,
            instruction_pad=args.instruction_pad,
            gripper_idx=args.gripper_idx,
            overwrite=args.overwrite,
            use_gpt_instruction=args.use_gpt_instruction,
            qwen_client=qwen_client,
            gpt_cfg=gpt_cfg,
            expanded_task_map=expanded_task_map,
            task_base_map=task_base_map,
            downsample_ratio=args.downsample_ratio,
        )

    # Persist any newly expanded instructions back to JSON
    if args.use_gpt_instruction and expanded_task_list_path is not None:
        _save_expanded_task_map()


if __name__ == "__main__":
    main()


# Example usages (Windows paths):
# python -m parse_data_module.parse_data_touchinthewild \
#   --root E:\python_projects\ftp1\SAMPLE_DATASETS\TouchInTheWild \
#   --output output\\touchinthewild \
#   --overwrite
#
# python -m parse_data_module.parse_data_touchinthewild \
#   --root E:\python_projects\ftp1\SAMPLE_DATASETS\TouchInTheWild \
#   --output output\\touchinthewild \
#   --dataset_name four_tasks/fluid_transfer/fluid_transfer.zarr.zip \
#   --overwrite \
#   --use_gpt_instruction \
#   --expanded_task_list_json E:\python_projects\ftp1\ftp1\data_processing\assets\TouchInTheWild\task_description_expanded.json
