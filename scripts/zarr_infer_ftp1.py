#!/usr/bin/env python3
"""
FTP1 inference script for loading a trained model from checkpoint.

This script loads a trained FTP1 model from a checkpoint directory and sets up
the model for inference. It uses the saved model_config.json to create the model
and loads the weights from the checkpoint.

Usage:
    python scripts/zarr_infer_ftp1.py \
        --checkpoint_dir /path/to/checkpoint \
        [--tactile_input_config_file /path/to/tactile_config.json]
"""

import argparse
import logging
import pathlib
import sys
import pdb


# Add project root to Python path for imports
_project_root = pathlib.Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch

from scripts.zarr_train_ftp1_utils import load_ftp1_model, print_model_statistics
from scripts.train_pytorch import init_logging

logger = logging.getLogger(__name__)


def main():
    init_logging()
    parser = argparse.ArgumentParser(
        description="Load FTP1 model from checkpoint for inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing the checkpoint (can be a step directory or parent directory)"
    )
    
    parser.add_argument(
        "--tactile_input_config_file",
        type=str,
        default=None,
        help="Path to tactile input config file. If None and use_tactile_input is True, "
             "will try to find it in checkpoint_dir"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to load model on (default: cuda if available, else cpu)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    logger.info(f"Loading FTP1 model from checkpoint: {checkpoint_dir}")
    logger.info(f"Tactile input config file: {args.tactile_input_config_file}")
    logger.info(f"Device: {args.device}")
    
    # Load model using load_ftp1_model function
    # This function will:
    # 1. Find the latest checkpoint step (or use the directory if it's a step directory)
    # 2. Load model_config.json
    # 3. Handle tactile_input_config_file
    # 4. Create FTP1Pytorch model instance
    # 5. Load weights from model.safetensors and hpt_tokenizer directory
    model, model_config, ckpt_dir = load_ftp1_model(
        checkpoint_dir=checkpoint_dir,
        tactile_input_config_file=args.tactile_input_config_file,
        device=args.device
    )
    print_model_statistics(model)

    logger.info(f"Model loaded successfully from checkpoint: {ckpt_dir}")
    logger.info(f"Model config: {model_config}")
    logger.info(f"Model device: {next(model.parameters()).device}")
    
    # Set model to eval mode for inference
    model.eval()
    logger.info("Model set to evaluation mode")
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Enter pdb for manual debugging
    logger.info("=" * 80)
    logger.info("Model loading complete. Entering pdb for manual debugging...")
    logger.info("=" * 80)
    logger.info("Available variables:")
    logger.info("  - model: The loaded FTP1Pytorch model")
    logger.info("  - model_config: The FTP1ModelConfig instance")
    logger.info("  - ckpt_dir: The checkpoint directory used")
    logger.info("  - checkpoint_dir: The original checkpoint directory")
    logger.info("=" * 80)
    
    pdb.set_trace()
    
    logger.info("Exiting inference script")


if __name__ == "__main__":
    main()

