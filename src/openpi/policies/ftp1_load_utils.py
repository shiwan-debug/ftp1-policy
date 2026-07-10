"""
Lightweight FTP1 model loader for inference.

This module loads FTP1 from checkpoint without importing scripts (tyro/wandb),
so it is safe to use inside Isaac Sim / Isaac Lab where typing_extensions may be shadowed.
"""

import gc
import json
import logging
import pathlib

import safetensors
import safetensors.torch
import torch

import openpi.models_pytorch.ftp1_model_config as ftp1_model_config
import openpi.models_pytorch.ftp1_pytorch


_SKIP_CHECKPOINT_KEYS = {
    "paligemma_with_expert.gemma_expert.lm_head.weight",
    "paligemma_with_expert.gemma_tactile_expert.lm_head.weight",
    "paligemma_with_expert.gemma_tactile_pred_expert.lm_head.weight",
}


def _unwrap_compile_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap torch.compile wrapper to get the original model."""
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    return model


def _filter_checkpoint_state_dict(
    checkpoint_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Drop checkpoint tensors that are intentionally not loaded into FTP1 inference models."""
    filtered_state_dict = {
        key: value for key, value in checkpoint_state_dict.items() if key not in _SKIP_CHECKPOINT_KEYS
    }
    skipped_keys = sorted(set(checkpoint_state_dict) - set(filtered_state_dict))
    if skipped_keys:
        logging.info(f"Skipping checkpoint keys during FTP1 load: {skipped_keys}")
    return filtered_state_dict


def _detach_tactile_tokenizers_for_checkpoint_io(model_to_process: torch.nn.Module) -> list[tuple[torch.nn.Module, torch.nn.ModuleDict]]:
    """Temporarily remove student/EMA tactile tokenizer ModuleDicts before state_dict load."""
    detached_tokenizers: list[tuple[torch.nn.Module, torch.nn.ModuleDict]] = []
    for attr_name in ("hpt_tactile_encoder", "ema_hpt_tactile_encoder"):
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


def _load_ftp1_model_weights(
    model: torch.nn.Module,
    ckpt_dir: pathlib.Path,
    device: str | torch.device = "cuda",
) -> None:
    """Load FTP1 model weights from checkpoint directory (safetensors + hpt_tokenizer)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    safetensors_path = ckpt_dir / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

    model_to_load = _unwrap_compile_model(model)
    if isinstance(model_to_load, torch.nn.parallel.DistributedDataParallel):
        model_to_load = model_to_load.module

    hpt_tactile_encoder = getattr(model_to_load, "hpt_tactile_encoder", None)
    ema_hpt_tactile_encoder = getattr(model_to_load, "ema_hpt_tactile_encoder", None)
    detached_tokenizers = _detach_tactile_tokenizers_for_checkpoint_io(model_to_load)
    if detached_tokenizers:
        try:
            checkpoint_state_dict = safetensors.torch.load_file(safetensors_path, device=str(device))
            checkpoint_state_dict = _filter_checkpoint_state_dict(checkpoint_state_dict)
            student_tokenizer_state_dict = _pop_state_dict_prefix(
                checkpoint_state_dict,
                "hpt_tactile_encoder.tokenizers.",
            )
            ema_tokenizer_state_dict = _pop_state_dict_prefix(
                checkpoint_state_dict,
                "ema_hpt_tactile_encoder.tokenizers.",
            )
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
    else:
        checkpoint_state_dict = safetensors.torch.load_file(safetensors_path, device=str(device))
        checkpoint_state_dict = _filter_checkpoint_state_dict(checkpoint_state_dict)
        student_tokenizer_state_dict = {}
        ema_tokenizer_state_dict = {}
        load_result = model_to_load.load_state_dict(checkpoint_state_dict, strict=False)
        missing_keys = list(load_result.missing_keys)
        unexpected_keys = list(load_result.unexpected_keys)
        if missing_keys:
            logging.warning(f"Missing keys while loading checkpoint {ckpt_dir}: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys while loading checkpoint {ckpt_dir}: {unexpected_keys}")
        logging.info("Loaded model state from safetensors format with strict=False")

    hpt_tokenizer_dir = ckpt_dir / "hpt_tokenizer"
    hpt_tactile_encoder = getattr(model_to_load, "hpt_tactile_encoder", None)
    if hpt_tactile_encoder is not None and hpt_tokenizer_dir.exists():
        logging.info(f"Loading HPT tokenizers from {hpt_tokenizer_dir}")
        num_loaded, num_missing = hpt_tactile_encoder.load_tokenizers(hpt_tokenizer_dir, strict=True)
        logging.info(f"Loaded {num_loaded} HPT tokenizers, {num_missing} missing")
    elif hpt_tactile_encoder is not None and student_tokenizer_state_dict:
        num_loaded, num_missing = _load_tokenizers_from_state_dict_prefix(
            getattr(hpt_tactile_encoder, "tokenizers", None),
            student_tokenizer_state_dict,
            prefix="hpt_tactile_encoder.tokenizers.",
        )
        logging.info(f"Loaded inline HPT tokenizers from model.safetensors: {num_loaded} loaded, {num_missing} missing")
    elif hpt_tactile_encoder is not None:
        logging.warning(
            f"HPT tokenizer directory not found at {hpt_tokenizer_dir}, skipping tokenizer loading"
        )

    if ema_hpt_tactile_encoder is not None:
        ema_loaded, ema_missing = _load_tokenizers_from_state_dict_prefix(
            getattr(ema_hpt_tactile_encoder, "tokenizers", None),
            ema_tokenizer_state_dict,
            prefix="ema_hpt_tactile_encoder.tokenizers.",
        )
        if ema_loaded > 0 or ema_missing > 0:
            logging.info(
                "Loaded EMA HPT teacher tokenizers from model.safetensors: %d loaded, %d missing",
                ema_loaded,
                ema_missing,
            )


def load_ftp1_model(
    checkpoint_dir: pathlib.Path,
    tactile_input_config_file: str | None = None,
    device: str | torch.device = "cuda",
):
    """Load FTP1 model from checkpoint directory (no tyro/wandb/scripts dependency).

    Returns:
        model: FTP1 PyTorch model
        model_config: FTP1ModelConfig instance
        ckpt_dir: pathlib.Path of the checkpoint step directory used
    """
    checkpoint_dir = pathlib.Path(checkpoint_dir)

    if (checkpoint_dir / "model_config.json").exists():
        ckpt_dir = checkpoint_dir
    else:
        checkpoint_steps = [
            int(d.name)
            for d in checkpoint_dir.iterdir()
            if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
        ]
        if not checkpoint_steps:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        latest_step = max(checkpoint_steps)
        ckpt_dir = checkpoint_dir / str(latest_step)
        logging.info(f"Using latest step {latest_step}: {ckpt_dir}")

    model_config_path = ckpt_dir / "model_config.json"
    if not model_config_path.exists():
        raise FileNotFoundError(f"model_config.json not found at {model_config_path}")

    with open(model_config_path, "r") as f:
        model_config_dict = json.load(f)

    if tactile_input_config_file is None and model_config_dict.get("use_tactile_input", False):
        tactile_config_path = ckpt_dir / "tactile_input_config_file.json"
        if tactile_config_path.exists():
            tactile_input_config_file = str(tactile_config_path)
            logging.info(f"Found tactile_input_config_file at {tactile_input_config_file}")
        else:
            logging.warning(
                f"use_tactile_input is True but tactile_input_config_file.json not found at {tactile_config_path}"
            )

    model_config_dict["tactile_input_config_file"] = tactile_input_config_file

    if "tactile_tokenizer_config" in model_config_dict:
        tokenizer_config_dict = dict(model_config_dict["tactile_tokenizer_config"])
        tokenizer_config_dict.pop("group_area_encoding", None)
        tokenizer_config_dict.pop("use_shared_chunk", None)
        tokenizer_config_dict["load_t3_pretrained_checkpoint"] = False
        model_config_dict["tactile_tokenizer_config"] = (
            ftp1_model_config.FTP1TactileTokenizerConfig(**tokenizer_config_dict)
        )
        logging.info(
            "Inference load disables tactile tokenizer T3-pretrained bootstrap; "
            "checkpoint tokenizer weights will be loaded directly."
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

    model_config = ftp1_model_config.FTP1ModelConfig(**model_config_dict)
    model = openpi.models_pytorch.ftp1_pytorch.FTP1Pytorch(model_config).to(device)

    logging.info(f"Created FTP1 model from checkpoint at {ckpt_dir}")
    logging.info("Loading model weights from checkpoint...")
    _load_ftp1_model_weights(model, ckpt_dir, device)
    logging.info(f"Model weights loaded successfully from checkpoint at {ckpt_dir}")

    return model, model_config, ckpt_dir
