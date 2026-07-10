import logging
import os
import pathlib
import numpy as np
import safetensors
import safetensors.torch
import torch
from openpi.shared.wandb_compat import wandb
import time
import dataclasses
import shutil
import gc
import json
import tyro
import openpi.models_pytorch.ftp1_model_config as ftp1_model_config
import openpi.models_pytorch.ftp1_pytorch
from openpi.norm_stats_utils import write_norm_stats_jsonable_with_override


def move_to_device(obj, device):
    """Recursively move tensors in nested structures to device."""
    if isinstance(obj, torch.Tensor):
        # Use non_blocking=True for faster CPU->GPU transfer when pin_memory is enabled
        return obj.to(device, non_blocking=True)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    elif isinstance(obj, tuple):
        return type(obj)(move_to_device(item, device) for item in obj)
    elif isinstance(obj, list):
        # Check if list is empty
        if len(obj) == 0:
            return obj
        # Recursively process each item in the list
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, str) or obj is None:
        return obj
    elif dataclasses.is_dataclass(obj):
        # Handle dataclass objects (including Observation)
        # Get all fields and recursively move each to device
        fields = {}
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            fields[field.name] = move_to_device(value, device)
        return dataclasses.replace(obj, **fields)
    else:
        return obj


def _clean_for_json_serialization(obj):
    """
    Recursively clean an object to make it JSON serializable.
    Converts tyro.MISSING to None, and handles other non-serializable types.
    """
    if obj is tyro.MISSING:
        return None
    elif isinstance(obj, dict):
        return {k: _clean_for_json_serialization(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_clean_for_json_serialization(item) for item in obj]
    elif dataclasses.is_dataclass(obj):
        return _clean_for_json_serialization(dataclasses.asdict(obj))
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # Try to check if it's a tyro.MISSING-like object by checking the type name
        type_name = type(obj).__name__
        if "Missing" in type_name or "MISSING" in type_name:
            return None
        # For other types, try to convert to string as last resort
        try:
            # Test if it's JSON serializable
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


def _overlay_reused_norm_stats(
    current_assets_dir: pathlib.Path,
    normalization_dir: pathlib.Path,
    old_repo_id: str,
) -> None:
    old_assets_dir = current_assets_dir.parent / old_repo_id
    if not old_assets_dir.exists():
        raise FileNotFoundError(f"reuse_norm_repo_id assets not found: {old_assets_dir}")

    replaced_files = 0
    replaced_keys = 0
    for current_path in current_assets_dir.rglob("*.json"):
        if not (
            current_path.name.startswith("share_norm_stats_") or current_path.name.startswith("independent_norm_stats_")
        ):
            continue

        rel_path = current_path.relative_to(current_assets_dir)
        override_path = old_assets_dir / rel_path
        if not override_path.exists():
            continue

        dest_path = normalization_dir / rel_path
        file_replaced_keys = write_norm_stats_jsonable_with_override(current_path, dest_path, override_path)
        replaced_files += 1
        replaced_keys += len(file_replaced_keys)

    logging.info(
        "Applied reuse_norm_repo_id=%s to checkpoint normalization: %d files updated, %d keys replaced",
        old_repo_id,
        replaced_files,
        replaced_keys,
    )


def _truncate_module_name(name: str, max_depth: int = 4) -> str:
    """
    Truncate module name to maximum depth by keeping only the first max_depth parts.

    Args:
        name: Module name (e.g., "model.layer1.layer2.layer3.layer4.layer5.weight")
        max_depth: Maximum depth to keep (default: 4)

    Returns:
        Truncated name (e.g., "model.layer1.layer2.layer3...")
    """
    parts = name.split(".")
    if len(parts) <= max_depth:
        return name
    # Keep first max_depth parts and add "..." to indicate truncation
    truncated = ".".join(parts[:max_depth])
    return f"{truncated}..."


def _unwrap_compile_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Unwrap torch.compile wrapper to get the original model.

    torch.compile adds _orig_mod prefix to parameter names, which breaks checkpoint loading/saving.
    This function unwraps the compile wrapper to access the original model.

    Args:
        model: Potentially compiled model

    Returns:
        Unwrapped model (original model if not compiled, or _orig_mod if compiled)
    """
    # Check if model is wrapped by torch.compile (has _orig_mod attribute)
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    return model


def _checkpoint_has_prefix(safetensors_path: pathlib.Path, prefix: str) -> bool:
    """Return whether a safetensors checkpoint contains any key with the given prefix."""
    with safetensors.safe_open(str(safetensors_path), framework="pt") as handle:
        for key in handle.keys():
            if key.startswith(prefix):
                return True
    return False


def _detach_tactile_tokenizers_for_checkpoint_io(
    model_to_process: torch.nn.Module,
    attr_names: tuple[str, ...] = ("hpt_tactile_encoder", "ema_hpt_tactile_encoder"),
) -> list[tuple[torch.nn.Module, torch.nn.ModuleDict]]:
    """Temporarily remove tactile tokenizer ModuleDicts before safetensors save/load.

    FTP1 checkpoints persist student tokenizers separately under ``hpt_tokenizer/`` to
    avoid inflating ``model.safetensors``. FTP checkpoints historically still embedded EMA
    teacher tokenizers in ``model.safetensors``. During ``load_state_dict(strict=False)``,
    those EMA tokenizer tensors become noisy ``unexpected_keys`` if the live model has a
    different tokenizer inventory. Detaching both student and EMA tokenizer ModuleDicts
    keeps checkpoint IO behavior symmetric and avoids spurious warnings.
    """
    detached_tokenizers: list[tuple[torch.nn.Module, torch.nn.ModuleDict]] = []
    for attr_name in attr_names:
        tactile_encoder = getattr(model_to_process, attr_name, None)
        if tactile_encoder is None or not hasattr(tactile_encoder, "tokenizers"):
            continue
        tokenizers = tactile_encoder.tokenizers
        if not isinstance(tokenizers, torch.nn.ModuleDict):
            continue
        detached_tokenizers.append((tactile_encoder, tokenizers))
        tactile_encoder.tokenizers = torch.nn.ModuleDict()
    return detached_tokenizers


def _restore_tactile_tokenizers(detached_tokenizers: list[tuple[torch.nn.Module, torch.nn.ModuleDict]]) -> None:
    """Restore tokenizer ModuleDicts detached by `_detach_tactile_tokenizers_for_checkpoint_io`."""
    for tactile_encoder, tokenizers in detached_tokenizers:
        tactile_encoder.tokenizers = tokenizers


def _load_tokenizers_from_state_dict_prefix(
    tokenizers: torch.nn.ModuleDict | None,
    checkpoint_state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> tuple[int, int]:
    """Load matching tokenizer modules from a prefixed checkpoint state dict subtree."""
    if tokenizers is None:
        return 0, 0

    loaded_count = 0
    missing_count = 0
    for tokenizer_key, tokenizer in tokenizers.items():
        tokenizer_prefix = f"{prefix}{tokenizer_key}."
        tokenizer_state_dict = {
            key[len(tokenizer_prefix):]: value
            for key, value in checkpoint_state_dict.items()
            if key.startswith(tokenizer_prefix)
        }
        if not tokenizer_state_dict:
            missing_count += 1
            continue
        tokenizer.load_state_dict(tokenizer_state_dict, strict=True)
        loaded_count += 1
    return loaded_count, missing_count


def _pop_state_dict_prefix(
    checkpoint_state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    """Remove and return a prefixed subtree from a checkpoint state dict."""
    matching_keys = [key for key in checkpoint_state_dict if key.startswith(prefix)]
    extracted_state_dict = {key: checkpoint_state_dict.pop(key) for key in matching_keys}
    return extracted_state_dict


def _sync_ema_tactile_teacher_from_student(model_to_load: torch.nn.Module) -> dict[str, int]:
    """Copy loaded student tactile target-encoder weights into EMA teacher modules.

    Old non-FTP checkpoints do not serialize ema_hpt_tactile_encoder.*. When enabling FTP from those checkpoints,
    initialize the EMA target encoder from the loaded student tactile encoder for modules that affect encoded values.
    Function-area embeddings are intentionally absent from the EMA teacher: area identity is an FTP condition,
    not an FTP target.
    """
    student_hpt = getattr(model_to_load, "hpt_tactile_encoder", None)
    ema_hpt = getattr(model_to_load, "ema_hpt_tactile_encoder", None)
    if student_hpt is None or ema_hpt is None:
        return {"tokenizers_synced": 0, "tokenizers_missing": 0, "modules_synced": 0}

    student_tokenizers = getattr(student_hpt, "tokenizers", None)
    ema_tokenizers = getattr(ema_hpt, "tokenizers", None)
    tokenizers_synced = 0
    tokenizers_missing = 0
    if student_tokenizers is not None and ema_tokenizers is not None:
        for tokenizer_key, student_tokenizer in student_tokenizers.items():
            if tokenizer_key not in ema_tokenizers:
                tokenizers_missing += 1
                continue
            ema_tokenizers[tokenizer_key].load_state_dict(student_tokenizer.state_dict(), strict=True)
            tokenizers_synced += 1

    modules_synced = 0
    for module_name in ("unified_proj", "shared_image_chunk_encoder"):
        student_module = getattr(student_hpt, module_name, None)
        ema_module = getattr(ema_hpt, module_name, None)
        if student_module is None or ema_module is None:
            continue
        ema_module.load_state_dict(student_module.state_dict(), strict=True)
        modules_synced += 1

    return {
        "tokenizers_synced": tokenizers_synced,
        "tokenizers_missing": tokenizers_missing,
        "modules_synced": modules_synced,
    }


def unwrap_runtime_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap torch.compile / DDP wrappers to reach the base FTP1 model."""
    model_unwrapped = _unwrap_compile_model(model)
    if isinstance(model_unwrapped, torch.nn.parallel.DistributedDataParallel):
        model_unwrapped = model_unwrapped.module
    return model_unwrapped


def get_matching_checkpoint_parameter_names(model: torch.nn.Module, checkpoint_path: str) -> set[str]:
    """Return trainable parameter names whose checkpoint tensors match by name and shape."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    model_to_inspect = unwrap_runtime_model(model)
    named_parameters = dict(model_to_inspect.named_parameters())
    checkpoint_state_dict = safetensors.torch.load_file(checkpoint_path)

    matching_names: set[str] = set()
    for name, param in named_parameters.items():
        if name not in checkpoint_state_dict:
            continue
        if tuple(param.shape) == tuple(checkpoint_state_dict[name].shape):
            matching_names.add(name)

    return matching_names


def resolve_model_safetensors_path(checkpoint_dir: str | os.PathLike[str]) -> pathlib.Path:
    """Resolve a direct checkpoint or the latest FTP1 step directory to model.safetensors."""
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    direct_path = checkpoint_dir / "model.safetensors"
    if direct_path.exists():
        return direct_path

    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    if not checkpoint_steps:
        raise FileNotFoundError(f"No model.safetensors or checkpoint steps found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    resolved_path = checkpoint_dir / f"{latest_step}" / "model.safetensors"
    if not resolved_path.exists():
        raise FileNotFoundError(f"Resolved checkpoint step has no model.safetensors: {resolved_path}")
    return resolved_path


def collect_trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Return parameters that currently require gradients."""
    return [param for param in model.parameters() if param.requires_grad]


def load_partial_weights(model: torch.nn.Module, checkpoint_path: str, strict: bool = False):
    """
    Load weights from checkpoint with partial matching - only load weights that match by name and shape.

    This function loads weights from a checkpoint file and only applies those that have matching
    names and shapes in the current model. This is useful when loading checkpoints from different
    model architectures or configurations.

    Args:
        model: The model to load weights into (may be wrapped by torch.compile or DDP)
        checkpoint_path: Path to the checkpoint file (safetensors format)
        strict: If False, only load matching weights and skip mismatched ones. If True, raise error on mismatch.

    Returns:
        Tuple of (num_loaded, num_skipped, num_missing) where:
        - num_loaded: Number of weights successfully loaded
        - num_skipped: Number of weights in checkpoint that don't match model
        - num_missing: Number of weights in model that are not in checkpoint
    """
    # Get model to load (unwrap DDP and torch.compile if needed)
    # IMPORTANT: Model wrapping order is: DDP -> torch.compile
    # So we need to unwrap compile first, then unwrap DDP
    model_to_load = _unwrap_compile_model(model)
    # Then unwrap DDP wrapper if present
    if isinstance(model_to_load, torch.nn.parallel.DistributedDataParallel):
        model_to_load = model_to_load.module

    # Get current model state dict (from unwrapped model)
    model_state_dict = model_to_load.state_dict()

    # Load checkpoint state dict
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint_state_dict = safetensors.torch.load_file(checkpoint_path)

    # Get device from model (use first parameter's device)
    device = next(model.parameters()).device

    # Track statistics
    loaded_keys = []
    skipped_keys = []
    missing_keys = []
    shape_mismatches = []

    # Build a new state dict with only matching weights
    filtered_state_dict = {}

    # First pass: find matching weights
    for key in model_state_dict.keys():
        if key in checkpoint_state_dict:
            model_shape = model_state_dict[key].shape
            checkpoint_shape = checkpoint_state_dict[key].shape

            if model_shape == checkpoint_shape:
                # Move weight to the same device as model
                weight = checkpoint_state_dict[key].to(device)
                filtered_state_dict[key] = weight
                loaded_keys.append(key)
            else:
                shape_mismatches.append((key, model_shape, checkpoint_shape))
                if strict:
                    raise ValueError(
                        f"Shape mismatch for key '{key}': model has {model_shape}, checkpoint has {checkpoint_shape}"
                    )
        else:
            missing_keys.append(key)

    # Second pass: identify keys in checkpoint that are not in model
    for key in checkpoint_state_dict.keys():
        if key not in model_state_dict:
            skipped_keys.append(key)

    # Load the filtered state dict into the unwrapped model
    model_to_load.load_state_dict(filtered_state_dict, strict=False)

    # Log statistics
    logging.info(f"Partial weight loading statistics:")
    logging.info(f"  ✓ Loaded: {len(loaded_keys)} weights")
    logging.info(f"  ✗ Skipped (not in model): {len(skipped_keys)} weights")
    logging.info(f"  ✗ Missing (not in checkpoint): {len(missing_keys)} weights")
    logging.info(f"  ✗ Shape mismatch: {len(shape_mismatches)} weights")

    # Display all keys with truncated module names (max 4 levels deep)
    # Group by truncated names to avoid duplicates
    if loaded_keys:
        truncated_loaded = set()
        for key in loaded_keys:
            truncated_key = _truncate_module_name(key, max_depth=4)
            truncated_loaded.add(truncated_key)

        logging.info(f"  Loaded keys ({len(loaded_keys)} total, {len(truncated_loaded)} unique modules):")
        for truncated_key in sorted(truncated_loaded):
            logging.info(f"    ✓ {truncated_key}")

    if skipped_keys:
        truncated_skipped = set()
        for key in skipped_keys:
            truncated_key = _truncate_module_name(key, max_depth=4)
            truncated_skipped.add(truncated_key)

        logging.info(f"  Skipped keys ({len(skipped_keys)} total, {len(truncated_skipped)} unique modules):")
        for truncated_key in sorted(truncated_skipped):
            logging.info(f"    - {truncated_key}")

    if missing_keys:
        truncated_missing = set()
        for key in missing_keys:
            truncated_key = _truncate_module_name(key, max_depth=4)
            truncated_missing.add(truncated_key)

        logging.info(f"  Missing keys ({len(missing_keys)} total, {len(truncated_missing)} unique modules):")
        for truncated_key in sorted(truncated_missing):
            logging.info(f"    ? {truncated_key}")

    if shape_mismatches:
        truncated_mismatches = {}
        for key, model_shape, ckpt_shape in shape_mismatches:
            truncated_key = _truncate_module_name(key, max_depth=4)
            if truncated_key not in truncated_mismatches:
                truncated_mismatches[truncated_key] = []
            truncated_mismatches[truncated_key].append((key, model_shape, ckpt_shape))

        logging.warning(
            f"  Shape mismatches ({len(shape_mismatches)} total, {len(truncated_mismatches)} unique modules):"
        )
        for truncated_key in sorted(truncated_mismatches.keys()):
            mismatches = truncated_mismatches[truncated_key]
            if len(mismatches) == 1:
                key, model_shape, ckpt_shape = mismatches[0]
                logging.warning(f"    ! {truncated_key}: model {model_shape} vs checkpoint {ckpt_shape}")
            else:
                logging.warning(f"    ! {truncated_key}:")
                for key, model_shape, ckpt_shape in mismatches:
                    logging.warning(f"        {key}: model {model_shape} vs checkpoint {ckpt_shape}")

    return len(loaded_keys), len(skipped_keys), len(missing_keys)


def save_ftp1_hpt_checkpoint(
    model,
    optimizer,
    global_step,
    config,
    is_main,
    data_config,
    checkpoint_dir=None,
):
    """Save a checkpoint with model state, optimizer state, and metadata.

    For FTP1 models, hpt_tactile_encoder.tokenizers are saved separately in hpt_tokenizer/ directory.
    Other model weights are saved in model.safetensors.

    Args:
        model: The model to save
        optimizer: The optimizer to save
        global_step: Current training step
        config: Training configuration
        is_main: Whether this is the main process (for DDP)
        data_config: Data configuration
        checkpoint_dir: Optional custom checkpoint directory. If None, uses config.checkpoint_dir
    """
    if not is_main:
        return

    # Use custom checkpoint_dir if provided, otherwise use config.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = config.checkpoint_dir
    else:
        checkpoint_dir = pathlib.Path(checkpoint_dir)

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Get model to save (unwrap DDP and torch.compile if needed)
        # IMPORTANT: Model wrapping order is: DDP -> torch.compile
        # So we need to unwrap compile first, then unwrap DDP
        model_to_save = _unwrap_compile_model(model)
        # Then unwrap DDP wrapper if present
        if isinstance(model_to_save, torch.nn.parallel.DistributedDataParallel):
            model_to_save = model_to_save.module

        # Check if model has HPT tactile encoders and save student tokenizers separately.
        hpt_tactile_encoder = getattr(model_to_save, "hpt_tactile_encoder", None)
        if hpt_tactile_encoder is not None and hasattr(hpt_tactile_encoder, "tokenizers"):
            detached_tokenizers = _detach_tactile_tokenizers_for_checkpoint_io(
                model_to_save,
                attr_names=("hpt_tactile_encoder",),
            )
            try:
                # Save model state using safetensors (handle shared tensors)
                # save_model automatically handles shared tensors like embed_tokens and lm_head
                safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")
            finally:
                _restore_tactile_tokenizers(detached_tokenizers)

            # Save student tokenizers separately
            hpt_tokenizer_dir = tmp_ckpt_dir / "hpt_tokenizer"
            hpt_tokenizer_dir.mkdir(parents=True, exist_ok=True)
            hpt_tactile_encoder.save_tokenizers(hpt_tokenizer_dir)
            logging.info(f"Saved {len(hpt_tactile_encoder.tokenizers)} HPT tokenizers to {hpt_tokenizer_dir}")
        else:
            # No hpt_tactile_encoder, save model normally
            safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # Save complete model config as JSON (including tactile encoder internal config)
        # Use dataclasses.asdict to get complete config structure
        model_config_dict = dataclasses.asdict(config.model)

        # Convert non-serializable types to strings
        # Handle Variant types (they are string literals but stored as Variant objects)
        if "paligemma_variant" in model_config_dict:
            model_config_dict["paligemma_variant"] = str(model_config_dict["paligemma_variant"])
        if "action_expert_variant" in model_config_dict:
            model_config_dict["action_expert_variant"] = str(model_config_dict["action_expert_variant"])
        if "tactile_expert_variant" in model_config_dict:
            model_config_dict["tactile_expert_variant"] = str(model_config_dict["tactile_expert_variant"])

        # Handle nested dataclass (tactile_tokenizer_config)
        if "tactile_tokenizer_config" in model_config_dict:
            # tactile_tokenizer_config is already a dict from asdict, but ensure it's properly serialized
            pass  # asdict already handles nested dataclasses

        # anonymize the path to the tactile input config file in model config.
        model_config_dict["tactile_input_config_file"] = ""

        # Save model config
        with open(tmp_ckpt_dir / "model_config.json", "w") as f:
            json.dump(model_config_dict, f, indent=2)

        # Save train config as JSON (excluding non-serializable fields)
        train_config_dict = {}
        # Save all serializable fields from TrainConfig (including FTP1TrainConfig specific fields)
        for field in dataclasses.fields(config):
            field_name = field.name
            field_value = getattr(config, field_name)

            # Skip non-serializable fields
            if field_name in [
                "weight_loader",
                "freeze_filter",
                "data",
                "norm_sample_ratio",
                "norm_batch_size",
                "norm_num_workers",
            ]:
                continue  # Skip WeightLoader and Filter objects

            # Handle model config (already saved separately, but include reference)
            if field_name == "model":
                # Don't include full model config here, it's in model_config.json
                continue

            try:
                cleaned_value = _clean_for_json_serialization(field_value)
                train_config_dict[field_name] = cleaned_value
            except Exception as e:
                # If all else fails, convert to string
                logging.warning(f"Failed to serialize field {field_name}: {e}, converting to string")
                train_config_dict[field_name] = str(field_value)

        # Save train config (all values should now be JSON serializable)
        with open(tmp_ckpt_dir / "train_config.json", "w") as f:
            json.dump(train_config_dict, f, indent=2)

        # Copy tactile_input_config_file if it exists
        tactile_input_config_file = None
        if hasattr(config.model, "tactile_input_config_file") and config.model.tactile_input_config_file:
            tactile_input_config_file = config.model.tactile_input_config_file
        elif (
            hpt_tactile_encoder is not None
            and hasattr(hpt_tactile_encoder, "input_config_path")
            and hpt_tactile_encoder.input_config_path
        ):
            tactile_input_config_file = hpt_tactile_encoder.input_config_path

        if tactile_input_config_file and os.path.exists(tactile_input_config_file):
            dest_path = tmp_ckpt_dir / "tactile_input_config_file.json"
            shutil.copy2(tactile_input_config_file, dest_path)
            logging.info(f"Copied tactile_input_config_file from {tactile_input_config_file} to {dest_path}")

        # Copy assets directory (normalization stats, train_val_split, etc.) to checkpoint
        # Rename it to "normalization" in the checkpoint directory
        # Exclude tactile_input_config.json as it's already saved separately
        if data_config.repo_id is not None:
            assets_source_dir = config.assets_dirs / data_config.repo_id
            if assets_source_dir.exists() and assets_source_dir.is_dir():
                normalization_dir = tmp_ckpt_dir / "normalization"
                normalization_dir.mkdir(parents=True, exist_ok=True)

                # Copy all files from assets directory, excluding tactile_input_config.json
                files_copied = 0
                for item in assets_source_dir.iterdir():
                    if item.is_file():
                        # Skip tactile_input_config.json as it's already saved separately
                        if item.name == "tactile_input_config.json":
                            continue
                        dest_file = normalization_dir / item.name
                        shutil.copy2(item, dest_file)
                        files_copied += 1
                    elif item.is_dir():
                        # Copy subdirectories recursively
                        dest_subdir = normalization_dir / item.name
                        shutil.copytree(item, dest_subdir, dirs_exist_ok=True)
                        files_copied += 1

                if files_copied > 0:
                    logging.info(
                        f"Copied {files_copied} items from assets directory {assets_source_dir} to {normalization_dir}"
                    )
                    reuse_norm_repo_id = getattr(data_config, "reuse_norm_repo_id", "")
                    if reuse_norm_repo_id:
                        _overlay_reused_norm_stats(assets_source_dir, normalization_dir, reuse_norm_repo_id)
                else:
                    logging.warning(
                        f"Assets directory {assets_source_dir} exists but is empty or only contains tactile_input_config.json"
                    )
            else:
                logging.warning(f"Assets directory not found: {assets_source_dir}, skipping normalization assets copy")

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def serialize_train_config_for_comparison(config):
    """Serialize train config to dict for comparison (same logic as save_ftp1_hpt_checkpoint).

    Args:
        config: TrainConfig instance

    Returns:
        dict: Serialized config dictionary
    """
    train_config_dict = {}
    # Save all serializable fields from TrainConfig (including FTP1TrainConfig specific fields)
    for field in dataclasses.fields(config):
        field_name = field.name
        field_value = getattr(config, field_name)

        # Skip non-serializable fields
        if field_name in ["weight_loader", "freeze_filter"]:
            continue  # Skip WeightLoader and Filter objects

        # Handle model config (already saved separately, but include reference)
        if field_name == "model":
            # Don't include full model config here, it's in model_config.json
            continue

        # For other fields, clean and serialize
        try:
            cleaned_value = _clean_for_json_serialization(field_value)
            train_config_dict[field_name] = cleaned_value
        except Exception as e:
            # If all else fails, convert to string
            logging.warning(f"Failed to serialize field {field_name}: {e}, converting to string")
            train_config_dict[field_name] = str(field_value)

    return train_config_dict


def serialize_model_config_for_comparison(model_config):
    """Serialize model config to dict for comparison (same logic as save_ftp1_hpt_checkpoint).

    Args:
        model_config: FTP1ModelConfig instance

    Returns:
        dict: Serialized model config dictionary
    """
    model_config_dict = dataclasses.asdict(model_config)

    # Convert non-serializable types to strings (same as save logic)
    if "paligemma_variant" in model_config_dict:
        model_config_dict["paligemma_variant"] = str(model_config_dict["paligemma_variant"])
    if "action_expert_variant" in model_config_dict:
        model_config_dict["action_expert_variant"] = str(model_config_dict["action_expert_variant"])
    if "tactile_expert_variant" in model_config_dict:
        model_config_dict["tactile_expert_variant"] = str(model_config_dict["tactile_expert_variant"])

    # Handle nested dataclass (tactile_tokenizer_config)
    if "tactile_tokenizer_config" in model_config_dict:
        # tactile_tokenizer_config is already a dict from asdict
        pass

    # Anonymize tactile_input_config_file (same as save logic)
    model_config_dict["tactile_input_config_file"] = ""

    return model_config_dict


def _compare_configs(
    saved_dict: dict, current_dict: dict, config_name: str, ignore_fields: set[str], strict: bool
) -> tuple[bool, list[dict]]:
    """Compare two config dictionaries and return mismatches.

    Args:
        saved_dict: Saved config dictionary
        current_dict: Current config dictionary
        config_name: Name of the config (for error messages)
        ignore_fields: Set of keys to ignore
        strict: If True, raise error on mismatch; if False, only warn

    Returns:
        tuple: (is_match, mismatches_list)
    """
    # Only check keys that exist in saved config (keys not in saved config are ignored)
    saved_keys = set(saved_dict.keys()) - ignore_fields

    # Check if saved keys exist in current config
    missing_in_current = saved_keys - set(current_dict.keys())

    if missing_in_current:
        msg = f"{config_name}: Keys in saved config but missing in current: {missing_in_current}"
        if strict:
            raise ValueError(msg)
        else:
            logging.warning(msg)

    # Compare values for keys that exist in both configs
    mismatches = []
    for key in saved_keys:
        if key not in current_dict:
            continue  # Skip if key doesn't exist in current config (already handled above)

        saved_value = saved_dict[key]
        current_value = current_dict[key]

        # Deep comparison (handling nested dicts/lists)
        if saved_value != current_value:
            mismatches.append(
                {
                    "key": key,
                    "saved": saved_value,
                    "current": current_value,
                }
            )

    return len(mismatches) == 0, mismatches


DEFAULT_IGNORE_KEYS = [
    "resume",
    "overwrite",
    "pytorch_weight_path",
    "assets_base_dir",
    "checkpoint_base_dir",
    "wandb_enabled",
]


def check_train_config_consistency(
    current_config,
    checkpoint_dir: pathlib.Path,
    global_step: int,
    strict: bool = True,
    ignore_keys: list[str] | None = DEFAULT_IGNORE_KEYS,
    ignore_model_keys: list[str] | None = None,
):
    """Check if current train config and model config match the saved configs in checkpoint.

    The check logic:
    - Only checks keys that exist in the saved config files
    - Keys that don't exist in saved config are ignored (not checked)
    - Keys in ignore_keys lists are always ignored

    Args:
        current_config: Current TrainConfig instance
        checkpoint_dir: Checkpoint directory
        global_step: Choose checkpoint step
        strict: If True, raise error on mismatch; if False, only warn
        ignore_keys: List of keys to ignore in train_config.json.
                     Default: ['resume', 'overwrite', 'checkpoint_base_dir', 'assets_base_dir']
        ignore_model_keys: List of keys to ignore in model_config.json.
                           Default: ['tactile_input_config_file'] (path is anonymized)

    Returns:
        bool: True if all configs match, False otherwise
    """
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    ckpt_dir = checkpoint_dir / f"{global_step}"

    all_mismatches = []
    config_checks = []

    # 1. Check train_config.json
    train_config_path = ckpt_dir / "train_config.json"
    if train_config_path.exists():
        with open(train_config_path, "r") as f:
            saved_train_config_dict = json.load(f)

        current_train_config_dict = serialize_train_config_for_comparison(current_config)

        # Set default ignore fields for train config
        default_train_ignore_fields = set(DEFAULT_IGNORE_KEYS)
        if ignore_keys is None:
            train_ignore_fields = default_train_ignore_fields
        else:
            train_ignore_fields = default_train_ignore_fields | set(ignore_keys)

        is_match, mismatches = _compare_configs(
            saved_train_config_dict, current_train_config_dict, "train_config.json", train_ignore_fields, strict
        )
        if not is_match:
            all_mismatches.extend([("train_config", m) for m in mismatches])
        config_checks.append(("train_config.json", is_match))
    else:
        if strict:
            raise FileNotFoundError(f"train_config.json not found at {train_config_path}")
        else:
            logging.warning(f"train_config.json not found at {train_config_path}, skipping consistency check")

    # 2. Check model_config.json
    model_config_path = ckpt_dir / "model_config.json"
    if model_config_path.exists():
        with open(model_config_path, "r") as f:
            saved_model_config_dict = json.load(f)

        current_model_config_dict = serialize_model_config_for_comparison(current_config.model)

        # Set default ignore fields for model config
        default_model_ignore_fields = {"tactile_input_config_file"}  # Path is anonymized
        if ignore_model_keys is None:
            model_ignore_fields = default_model_ignore_fields
        else:
            model_ignore_fields = default_model_ignore_fields | set(ignore_model_keys)

        is_match, mismatches = _compare_configs(
            saved_model_config_dict, current_model_config_dict, "model_config.json", model_ignore_fields, strict
        )
        if not is_match:
            all_mismatches.extend([("model_config", m) for m in mismatches])
        config_checks.append(("model_config.json", is_match))
    else:
        if strict:
            raise FileNotFoundError(f"model_config.json not found at {model_config_path}")
        else:
            logging.warning(f"model_config.json not found at {model_config_path}, skipping consistency check")

    # Report all mismatches
    if all_mismatches:
        mismatch_msg = "Config consistency check failed:\n"
        for config_type, mismatch in all_mismatches[:20]:  # Show first 20 mismatches
            mismatch_msg += f"  - {config_type}.{mismatch['key']}:\n"
            mismatch_msg += f"    saved:   {mismatch['saved']}\n"
            mismatch_msg += f"    current: {mismatch['current']}\n"
        if len(all_mismatches) > 20:
            mismatch_msg += f"  ... and {len(all_mismatches) - 20} more mismatches\n"

        if strict:
            raise ValueError(mismatch_msg)
        else:
            logging.warning(mismatch_msg)
        return False

    # Log success for each checked config
    for config_name, is_match in config_checks:
        if is_match:
            logging.info(f"{config_name} consistency check passed")

    logging.info(f"All config consistency checks passed for checkpoint at step {global_step}")
    return True


def _load_ftp1_model_weights(
    model,
    ckpt_dir: pathlib.Path,
    device: str | torch.device = "cuda",
    *,
    load_hpt_tokenizers_strict: bool = True,
    allow_partial_load: bool = False,
):
    """Load FTP1 model weights from checkpoint directory.

    This is a helper function that loads model weights (model.safetensors and hpt_tokenizer)
    from a checkpoint directory. It handles the temporary removal of tokenizers during loading.

    Args:
        model: The model to load weights into (can be DDP-wrapped)
        ckpt_dir: The checkpoint step directory containing model.safetensors and hpt_tokenizer/
        device: Device to load checkpoints to
        load_hpt_tokenizers_strict: If True, require every current-model tokenizer file to
            exist in the checkpoint. If False, load only matching tokenizer files and skip
            missing ones with warnings.
        allow_partial_load: If True, only load keys whose names AND shapes match between
            checkpoint and current model; all mismatches are skipped.

    Returns:
        None (modifies model in-place)
    """
    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    logging.info("Loading model state...")
    safetensors_path = ckpt_dir / "model.safetensors"

    if not safetensors_path.exists():
        raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

    model_to_load = _unwrap_compile_model(model)
    if isinstance(model_to_load, torch.nn.parallel.DistributedDataParallel):
        model_to_load = model_to_load.module
    hpt_tactile_encoder = getattr(model_to_load, "hpt_tactile_encoder", None)
    ema_hpt_tactile_encoder = getattr(model_to_load, "ema_hpt_tactile_encoder", None)
    detached_tokenizers = _detach_tactile_tokenizers_for_checkpoint_io(
        model_to_load,
        attr_names=("hpt_tactile_encoder", "ema_hpt_tactile_encoder"),
    )

    checkpoint_state_dict: dict[str, torch.Tensor] = {}
    student_tokenizer_state_dict: dict[str, torch.Tensor] = {}
    ema_tokenizer_state_dict: dict[str, torch.Tensor] = {}

    try:
        if allow_partial_load:
            logging.warning(
                "FTP1 partial checkpoint loading is enabled; only matching names/shapes "
                f"will be loaded from {safetensors_path}"
            )
            load_partial_weights(model_to_load, str(safetensors_path), strict=False)
        else:
            checkpoint_state_dict = safetensors.torch.load_file(safetensors_path, device=str(device))
            student_tokenizer_state_dict = _pop_state_dict_prefix(
                checkpoint_state_dict,
                "hpt_tactile_encoder.tokenizers.",
            )
            ema_tokenizer_state_dict = _pop_state_dict_prefix(
                checkpoint_state_dict,
                "ema_hpt_tactile_encoder.tokenizers.",
            )
            # Skip expert lm_head vocab embeddings that were removed (old ckpt has full vocab, new model has [1, dim])
            _expert_lm_head_keys = [
                k for k in checkpoint_state_dict
                if k.endswith("lm_head.weight")
                and (
                    "gemma_expert." in k
                    or "gemma_tactile_expert." in k
                    or "gemma_tactile_pred_expert." in k
                )
            ]
            model_state = model_to_load.state_dict()
            for k in _expert_lm_head_keys:
                if k in model_state and checkpoint_state_dict[k].shape != model_state[k].shape:
                    logging.warning(
                        f"Skipping {k}: ckpt shape {checkpoint_state_dict[k].shape} != "
                        f"model shape {model_state[k].shape} (vocab was removed)"
                    )
                    del checkpoint_state_dict[k]
            load_result = model_to_load.load_state_dict(checkpoint_state_dict, strict=False)
            missing_keys = list(load_result.missing_keys)
            unexpected_keys = list(load_result.unexpected_keys)
            if missing_keys:
                logging.warning(f"Missing keys while loading checkpoint {ckpt_dir}: {missing_keys}")
            if unexpected_keys:
                logging.warning(f"Unexpected keys while loading checkpoint {ckpt_dir}: {unexpected_keys}")
            logging.info("Loaded model state from safetensors format with strict=False")
    finally:
        _restore_tactile_tokenizers(detached_tokenizers)

    # Load HPT tokenizers separately if they exist
    hpt_tokenizer_dir = ckpt_dir / "hpt_tokenizer"

    if hpt_tactile_encoder is not None and hpt_tokenizer_dir.exists():
        logging.info(f"Loading HPT tokenizers from {hpt_tokenizer_dir}")
        num_loaded, num_missing = hpt_tactile_encoder.load_tokenizers(
            hpt_tokenizer_dir,
            strict=load_hpt_tokenizers_strict,
        )
        if load_hpt_tokenizers_strict:
            logging.info(f"Loaded {num_loaded} HPT tokenizers, {num_missing} missing")
        else:
            logging.info(
                f"Loaded {num_loaded} HPT tokenizers, skipped {num_missing} missing tokenizer checkpoints"
            )

        # For old checkpoints, EMA tactile teacher weights are not serialized separately.
        # In that case, mirror loaded student target-encoder weights needed for FTP targets.
        if not _checkpoint_has_prefix(safetensors_path, "ema_hpt_tactile_encoder.tokenizers."):
            sync_stats = _sync_ema_tactile_teacher_from_student(model_to_load)
            if sync_stats["tokenizers_synced"] > 0 or sync_stats["modules_synced"] > 0:
                logging.info(
                    "Initialized EMA HPT teacher from loaded student encoder: "
                    "%d tokenizers synced, %d tokenizer keys missing in EMA, %d non-tokenizer modules synced.",
                    sync_stats["tokenizers_synced"],
                    sync_stats["tokenizers_missing"],
                    sync_stats["modules_synced"],
                )
    elif hpt_tactile_encoder is not None and student_tokenizer_state_dict:
        student_loaded, student_missing = _load_tokenizers_from_state_dict_prefix(
            getattr(hpt_tactile_encoder, "tokenizers", None),
            student_tokenizer_state_dict,
            prefix="hpt_tactile_encoder.tokenizers.",
        )
        logging.info(
            "Loaded inline HPT tokenizers from model.safetensors: %d loaded, %d missing",
            student_loaded,
            student_missing,
        )
    elif hpt_tactile_encoder is not None:
        logging.warning(f"HPT tokenizer directory not found at {hpt_tokenizer_dir}, skipping tokenizer loading")

    if ema_hpt_tactile_encoder is not None:
        ema_tokenizers = getattr(ema_hpt_tactile_encoder, "tokenizers", None)
        if ema_tokenizers:
            ema_loaded, ema_missing = _load_tokenizers_from_state_dict_prefix(
                ema_tokenizers,
                ema_tokenizer_state_dict,
                prefix="ema_hpt_tactile_encoder.tokenizers.",
            )
            if ema_loaded > 0 or ema_missing > 0:
                logging.info(
                    "Loaded EMA HPT teacher tokenizers from model.safetensors: %d loaded, %d missing",
                    ema_loaded,
                    ema_missing,
                )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def _format_param_count(num_params: int) -> str:
    """Format parameter count with appropriate unit (B, M, K).

    Args:
        num_params: Number of parameters

    Returns:
        Formatted string (e.g., "1.5B", "300M", "50K")
    """
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.2f}B"
    elif num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.2f}K"
    else:
        return str(num_params)


def _estimate_safetensors_size(num_params: int, dtype: torch.dtype = torch.bfloat16) -> str:
    """Estimate safetensors file size for given number of parameters.

    Args:
        num_params: Number of parameters
        dtype: Data type (default: bfloat16 = 2 bytes per param)

    Returns:
        Formatted size string (e.g., "1.5GB", "300MB", "50KB")
    """
    bytes_per_param = 2 if dtype in (torch.bfloat16, torch.float16) else 4
    size_bytes = num_params * bytes_per_param

    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1_000_000_000:.2f}GB"
    elif size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.2f}MB"
    elif size_bytes >= 1_000:
        return f"{size_bytes / 1_000:.2f}KB"
    else:
        return f"{size_bytes}B"


def _get_module_params(module: torch.nn.Module) -> tuple[int, torch.dtype]:
    """Get parameter count and dtype from a module.

    Args:
        module: PyTorch module

    Returns:
        Tuple of (num_params, dtype)
    """
    num_params = sum(p.numel() for p in module.parameters())
    # Get dtype from first parameter
    dtype = next(module.parameters()).dtype if list(module.parameters()) else torch.bfloat16
    return num_params, dtype


def print_model_statistics(model: torch.nn.Module):
    """Print detailed model statistics in a formatted table.

    Args:
        model: FTP1Pytorch model instance
    """
    stats = []

    # 1. Three experts
    paligemma_with_expert = getattr(model, "paligemma_with_expert", None)
    if paligemma_with_expert:
        # PaliGemma (VLM) - break down into components
        if hasattr(paligemma_with_expert, "paligemma"):
            paligemma = paligemma_with_expert.paligemma
            dtype = next(paligemma.parameters()).dtype if list(paligemma.parameters()) else torch.bfloat16

            # Image Encoder (Vision Tower)
            if hasattr(paligemma, "model") and hasattr(paligemma.model, "vision_tower"):
                num_params, _ = _get_module_params(paligemma.model.vision_tower)
                stats.append(("PaliGemma Image Encoder", num_params, dtype))

            # Multi-modal Projector
            if hasattr(paligemma, "model") and hasattr(paligemma.model, "multi_modal_projector"):
                num_params, _ = _get_module_params(paligemma.model.multi_modal_projector)
                stats.append(("PaliGemma Multi-modal Projector", num_params, dtype))

            # Language Model (Text Encoder)
            if hasattr(paligemma, "model") and hasattr(paligemma.model, "language_model"):
                num_params, _ = _get_module_params(paligemma.model.language_model)
                stats.append(("PaliGemma Language Model", num_params, dtype))

            # LM Head (Text Tokenizer output layer)
            if hasattr(paligemma, "lm_head"):
                num_params, _ = _get_module_params(paligemma.lm_head)
                stats.append(("PaliGemma LM Head", num_params, dtype))

        # Action Expert
        if hasattr(paligemma_with_expert, "gemma_expert"):
            num_params, dtype = _get_module_params(paligemma_with_expert.gemma_expert)
            stats.append(("Action Expert", num_params, dtype))

        # Tactile Expert
        if (
            hasattr(paligemma_with_expert, "gemma_tactile_expert")
            and paligemma_with_expert.gemma_tactile_expert is not None
        ):
            num_params, dtype = _get_module_params(paligemma_with_expert.gemma_tactile_expert)
            stats.append(("Tactile Expert", num_params, dtype))

    # 2. HPT Tactile Encoder and its tokenizers
    hpt_tactile_encoder = getattr(model, "hpt_tactile_encoder", None)
    if hpt_tactile_encoder:
        dtype = (
            next(hpt_tactile_encoder.parameters()).dtype if list(hpt_tactile_encoder.parameters()) else torch.bfloat16
        )

        # Collect all HPT Tactile Encoder components
        hpt_total_params = 0
        tokenizer_params = 0
        tokenizer_stats = []

        # Get tokenizer parameters (sum of all tokenizers)
        if hasattr(hpt_tactile_encoder, "tokenizers") and hpt_tactile_encoder.tokenizers:
            for tokenizer_key, tokenizer in hpt_tactile_encoder.tokenizers.items():
                num_params, _ = _get_module_params(tokenizer)
                tokenizer_params += num_params
                hpt_total_params += num_params
                tokenizer_stats.append((f"  └─ Tokenizer: {tokenizer_key}", num_params, dtype))

        # shared_image_chunk_encoder (if has_image_type=True)
        shared_chunk_params = 0
        if (
            hasattr(hpt_tactile_encoder, "shared_image_chunk_encoder")
            and hpt_tactile_encoder.shared_image_chunk_encoder is not None
        ):
            num_params, _ = _get_module_params(hpt_tactile_encoder.shared_image_chunk_encoder)
            if num_params > 0:
                shared_chunk_params = num_params
                hpt_total_params += num_params

        # unified_proj
        unified_proj_params = 0
        if hasattr(hpt_tactile_encoder, "unified_proj"):
            num_params, _ = _get_module_params(hpt_tactile_encoder.unified_proj)
            if num_params > 0:
                unified_proj_params = num_params
                hpt_total_params += num_params

        # tactile_blank_token
        tactile_blank_token_params = 0
        if hasattr(hpt_tactile_encoder, "tactile_blank_token"):
            num_params = hpt_tactile_encoder.tactile_blank_token.numel()
            if num_params > 0:
                tactile_blank_token_params = num_params
                hpt_total_params += num_params

        # func_area_idx_embedding (new tactile area embedding)
        func_area_idx_embedding_params = 0
        if (
            hasattr(hpt_tactile_encoder, "func_area_idx_embedding")
            and hpt_tactile_encoder.func_area_idx_embedding is not None
        ):
            num_params, _ = _get_module_params(hpt_tactile_encoder.func_area_idx_embedding)
            if num_params > 0:
                func_area_idx_embedding_params = num_params
                hpt_total_params += num_params

        # Add HPT Tactile Encoder total (all components)
        if hpt_total_params > 0:
            stats.append(("HPT Tactile Encoder (total)", hpt_total_params, dtype))

        # Add tokenizers total and individual tokenizers
        if tokenizer_params > 0:
            stats.append(("  └─ Tokenizers (total)", tokenizer_params, dtype))
            # Add individual tokenizers
            stats.extend(tokenizer_stats)

        # Add other components
        if shared_chunk_params > 0:
            stats.append(("  └─ HPT shared_image_chunk_encoder", shared_chunk_params, dtype))

        if unified_proj_params > 0:
            stats.append(("  └─ HPT unified_proj", unified_proj_params, dtype))

        if tactile_blank_token_params > 0:
            stats.append(("  └─ HPT tactile_blank_token", tactile_blank_token_params, dtype))

        if func_area_idx_embedding_params > 0:
            stats.append(("  └─ HPT func_area_idx_embedding", func_area_idx_embedding_params, dtype))

    # 3. State Encoder
    state_encoder = getattr(model, "state_encoder", None)
    if state_encoder:
        num_params, dtype = _get_module_params(state_encoder)
        stats.append(("State Encoder", num_params, dtype))

    # 4. Action projection layers
    action_in_proj = getattr(model, "action_in_proj", None)
    if action_in_proj:
        num_params, dtype = _get_module_params(action_in_proj)
        stats.append(("Action In Proj", num_params, dtype))

    action_out_proj = getattr(model, "action_out_proj", None)
    if action_out_proj:
        num_params, dtype = _get_module_params(action_out_proj)
        stats.append(("Action Out Proj", num_params, dtype))

    # 5. Time MLP layers
    time_mlp_in = getattr(model, "time_mlp_in", None)
    if time_mlp_in:
        num_params, dtype = _get_module_params(time_mlp_in)
        stats.append(("Time MLP In", num_params, dtype))

    time_mlp_out = getattr(model, "time_mlp_out", None)
    if time_mlp_out:
        num_params, dtype = _get_module_params(time_mlp_out)
        stats.append(("Time MLP Out", num_params, dtype))

    # 6. Check for other modules (excluding already processed ones)
    processed_modules = {
        "paligemma_with_expert",
        "hpt_tactile_encoder",
        "state_encoder",
        "action_in_proj",
        "action_out_proj",
        "time_mlp_in",
        "time_mlp_out",
        "config",
        "gradient_checkpointing_enabled",
        "state_input_mode",
        "action_dim",
    }
    for attr_name in dir(model):
        if attr_name in processed_modules or attr_name.startswith("_"):
            continue
        module = getattr(model, attr_name, None)
        if not isinstance(module, torch.nn.Module) or module is None:
            continue
        # Skip if it's a parameter or buffer
        if isinstance(module, (torch.nn.Parameter, torch.Tensor)):
            continue
        num_params, dtype = _get_module_params(module)
        if num_params > 0:
            stats.append((f"{attr_name.replace('_', ' ').title()}", num_params, dtype))
            processed_modules.add(attr_name)

    # Print table
    if not stats:
        logging.info("No model components found for statistics")
        return

    # Calculate column widths
    max_name_len = max(len(name) for name, _, _ in stats)
    max_count_len = max(len(f"{count:,}") for _, count, _ in stats)

    # Header
    header = f"{'Component':<{max_name_len}} | {'Params':>{max_count_len}} | {'Formatted':>12} | {'Size (est.)':>12}"
    separator = "-" * len(header)

    logging.info("\n" + "=" * len(header))
    logging.info("Model Component Statistics")
    logging.info("=" * len(header))
    logging.info(header)
    logging.info(separator)

    # Rows
    for name, num_params, dtype in stats:
        formatted = _format_param_count(num_params)
        size_est = _estimate_safetensors_size(num_params, dtype)
        row = f"{name:<{max_name_len}} | {num_params:>{max_count_len},} | {formatted:>12} | {size_est:>12}"
        logging.info(row)

    # Total
    total_params = sum(count for _, count, _ in stats)
    total_dtype = stats[0][2] if stats else torch.bfloat16
    total_formatted = _format_param_count(total_params)
    total_size = _estimate_safetensors_size(total_params, total_dtype)

    logging.info(separator)
    total_row = (
        f"{'TOTAL':<{max_name_len}} | {total_params:>{max_count_len},} | {total_formatted:>12} | {total_size:>12}"
    )
    logging.info(total_row)
    logging.info("=" * len(header) + "\n")


def load_ftp1_model(
    checkpoint_dir: pathlib.Path,
    tactile_input_config_file: str | None = None,
    device: str | torch.device = "cuda",
    model: torch.nn.Module | None = None,
    *,
    load_hpt_tokenizers_strict: bool = True,
    allow_partial_load: bool = False,
):
    """Load FTP1 model from checkpoint directory.

    This function loads the model_config.json from the checkpoint directory, handles
    tactile_input_config_file, and creates the corresponding FTP1 PyTorch model.

    Args:
        checkpoint_dir: Directory containing checkpoints. Can be:
            - A directory containing step subdirectories (e.g., checkpoint_dir/1/, checkpoint_dir/2/)
            - A direct step directory (e.g., checkpoint_dir/1/)
        tactile_input_config_file: Path to tactile input config file. If None and use_tactile_input
            is True, will try to find it in checkpoint_dir.
        device: Device to create the model on
        model: Optional existing model instance. If provided, load checkpoint
            weights/tokenizers into this model instead of creating a new model.
        load_hpt_tokenizers_strict: If True, require every tokenizer expected by the
            current model to exist in the checkpoint. If False, load only matching
            tokenizer files and skip missing ones with warnings.

    Returns:
        model: The created FTP1 PyTorch model
        model_config: The loaded FTP1ModelConfig instance
        ckpt_dir: The specific checkpoint step directory used
    """
    checkpoint_dir = pathlib.Path(checkpoint_dir)

    # Check if checkpoint_dir is a direct step directory (contains model_config.json)
    if (checkpoint_dir / "model_config.json").exists():
        ckpt_dir = checkpoint_dir
    else:
        # Find the latest checkpoint step
        checkpoint_steps = [
            int(d.name)
            for d in checkpoint_dir.iterdir()
            if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
        ]

        if not checkpoint_steps:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

        latest_step = max(checkpoint_steps)
        ckpt_dir = checkpoint_dir / f"{latest_step}"
        logging.info(f"Found {len(checkpoint_steps)} checkpoint(s), using latest step {latest_step}: {ckpt_dir}")

    # Load model_config.json
    model_config_path = ckpt_dir / "model_config.json"
    if not model_config_path.exists():
        raise FileNotFoundError(f"model_config.json not found at {model_config_path}")

    with open(model_config_path, "r") as f:
        model_config_dict = json.load(f)

    # Handle tactile_input_config_file
    if tactile_input_config_file is None:
        # Try to find it in checkpoint_dir if use_tactile_input is True
        if model_config_dict.get("use_tactile_input", False):
            tactile_config_path = ckpt_dir / "tactile_input_config_file.json"
            if tactile_config_path.exists():
                tactile_input_config_file = str(tactile_config_path)
                logging.info(f"Found tactile_input_config_file at {tactile_input_config_file}")
            else:
                logging.warning(
                    f"use_tactile_input is True but tactile_input_config_file.json not found at {tactile_config_path}"
                )

    # Replace tactile_input_config_file in config dict
    if tactile_input_config_file is not None:
        model_config_dict["tactile_input_config_file"] = tactile_input_config_file
    else:
        model_config_dict["tactile_input_config_file"] = None

    # Variant is a Literal type alias, so we can use the string value directly
    # No conversion needed - the string value is already the correct type

    # Handle nested dataclass (tactile_tokenizer_config)
    if "tactile_tokenizer_config" in model_config_dict:
        tokenizer_config_dict = dict(model_config_dict["tactile_tokenizer_config"])
        tokenizer_config_dict.pop("group_area_encoding", None)
        tokenizer_config_dict.pop("use_shared_chunk", None)
        model_config_dict["tactile_tokenizer_config"] = ftp1_model_config.FTP1TactileTokenizerConfig(
            **tokenizer_config_dict
        )
    if "future_tactile_prediction" in model_config_dict:
        future_tactile_cfg = model_config_dict["future_tactile_prediction"]
        if isinstance(future_tactile_cfg, dict):
            future_tactile_cfg = dict(future_tactile_cfg)
            if "branch_mode" in future_tactile_cfg and "attention_fusion" not in future_tactile_cfg:
                future_tactile_cfg["attention_fusion"] = (
                    future_tactile_cfg.pop("branch_mode") == "attention_fusion"
                )
            model_config_dict["future_tactile_prediction"] = ftp1_model_config.FutureTactilePredictionConfig(
                **future_tactile_cfg
            )

    # Create FTP1ModelConfig instance
    model_config = ftp1_model_config.FTP1ModelConfig(**model_config_dict)

    # Create model or use provided model
    if model is None:
        model = openpi.models_pytorch.ftp1_pytorch.FTP1Pytorch(model_config).to(device)
        logging.info(f"Created FTP1 model from checkpoint config at {ckpt_dir}")
    else:
        model = model.to(device)
        logging.info("Using provided model instance; loading checkpoint weights/tokenizers into it")

    # Load model weights from checkpoint
    logging.info("Loading model weights from checkpoint...")
    _load_ftp1_model_weights(
        model,
        ckpt_dir,
        device,
        load_hpt_tokenizers_strict=load_hpt_tokenizers_strict,
        allow_partial_load=allow_partial_load,
    )
    logging.info(f"Model weights loaded successfully from checkpoint at {ckpt_dir}")

    return model, model_config, ckpt_dir


def get_ftp1_latest_checkpoint_step(checkpoint_dir, device: str | torch.device = "cuda"):
    # Check if checkpoint_dir is a direct step directory (contains model.safetensors)
    if (checkpoint_dir / "model.safetensors").exists():
        ckpt_dir = checkpoint_dir
        # Try to get step from directory name or metadata
        try:
            latest_step = int(ckpt_dir.name)
        except ValueError:
            # If name is not a number, try to get from metadata
            metadata_path = ckpt_dir / "metadata.pt"
            if metadata_path.exists():
                metadata = torch.load(metadata_path, map_location=device, weights_only=False)
                latest_step = metadata.get("global_step", 0)
            else:
                latest_step = 0
    else:
        # Find the latest checkpoint step
        checkpoint_steps = [
            int(d.name)
            for d in checkpoint_dir.iterdir()
            if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
        ]

        if not checkpoint_steps:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

        latest_step = max(checkpoint_steps)
        ckpt_dir = checkpoint_dir / f"{latest_step}"

    return latest_step, ckpt_dir


def load_ftp1_hpt_checkpoint(model, optimizer, checkpoint_dir, device, global_step=None):
    """Load FTP1 checkpoint with separate HPT tokenizer loading.

    This function loads model weights, optimizer state, and metadata from a checkpoint.
    It reuses load_ftp1_model to handle model creation if needed, but typically
    the model is already created, so this function just loads the weights.

    Args:
        model: The model to load weights into
        optimizer: The optimizer to load state into
        checkpoint_dir: Directory containing checkpoints. Can be:
            - A directory containing step subdirectories (e.g., checkpoint_dir/1/, checkpoint_dir/2/)
            - A direct step directory (e.g., checkpoint_dir/1/)
        device: Device to load checkpoints to
        global_step: The step number to load the checkpoint from. If None, will load the latest checkpoint.

    Returns:
        global_step: The step number of the loaded checkpoint
    """
    checkpoint_dir = pathlib.Path(checkpoint_dir)

    if global_step is None:
        latest_step, ckpt_dir = get_ftp1_latest_checkpoint_step(checkpoint_dir, device)
    else:
        latest_step = global_step
        ckpt_dir = checkpoint_dir / f"{global_step}"
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"Load Checkpoint directory {ckpt_dir} not found.\n - (target) checkpoint_dir: {checkpoint_dir}\n - (target) global_step: {global_step}."
            )

    try:
        # Load model weights (reuse shared function)
        _load_ftp1_model_weights(model, ckpt_dir, device)

        # Load optimizer state
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        logging.info(f"Loaded checkpoint from step {global_step}")

        return global_step

    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}")
        raise
