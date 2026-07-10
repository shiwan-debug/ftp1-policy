#!/usr/bin/env python3
"""
Download JAX checkpoints from GCS to local directory and optionally convert to PyTorch.

Usage:
    # Download only
    python scripts/download_checkpoint.py --checkpoint_url gs://openpi-assets/checkpoints/pi05_base --output_dir ./checkpoints/jax
    
    # Download and convert to PyTorch
    python scripts/download_checkpoint.py --checkpoint_url gs://openpi-assets/checkpoints/pi05_base --output_dir ./checkpoints/jax --convert_to_pytorch --pytorch_output_dir ./checkpoints/pytorch --config_name pi05_aloha
"""

import argparse
import pathlib
import shutil
import logging
import subprocess
import sys

import openpi.shared.download as download

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_checkpoint(checkpoint_url: str, output_dir: str, force_download: bool = False):
    """
    Download a checkpoint from GCS to local directory.
    
    Args:
        checkpoint_url: GCS URL of the checkpoint (e.g., gs://openpi-assets/checkpoints/pi05_base)
        output_dir: Local directory to save the checkpoint
        force_download: If True, force re-download even if already exists
    """
    output_path = pathlib.Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading checkpoint from {checkpoint_url} to {output_path}")
    
    # Download to cache first (maybe_download handles GCS authentication and caching)
    cached_path = download.maybe_download(checkpoint_url, force_download=force_download)
    
    logger.info(f"Checkpoint downloaded to cache: {cached_path}")
    
    # Copy from cache to target directory
    if cached_path.is_dir():
        # If it's a directory, copy the entire directory
        target_path = output_path / cached_path.name
        if target_path.exists():
            logger.info(f"Removing existing directory: {target_path}")
            shutil.rmtree(target_path)
        
        logger.info(f"Copying from cache to {target_path}")
        shutil.copytree(cached_path, target_path)
        logger.info(f"Successfully copied checkpoint to {target_path}")
        return target_path
    else:
        # If it's a file, copy it
        target_path = output_path / cached_path.name
        if target_path.exists():
            logger.info(f"Removing existing file: {target_path}")
            target_path.unlink()
        
        logger.info(f"Copying from cache to {target_path}")
        shutil.copy2(cached_path, target_path)
        logger.info(f"Successfully copied checkpoint to {target_path}")
        return target_path


def convert_to_pytorch(
    jax_checkpoint_dir: pathlib.Path,
    pytorch_output_dir: str,
    config_name: str,
    precision: str = "bfloat16",
):
    """
    Convert JAX checkpoint to PyTorch format.
    
    Args:
        jax_checkpoint_dir: Path to the downloaded JAX checkpoint directory
        pytorch_output_dir: Directory to save the converted PyTorch checkpoint
        config_name: Config name for the model (e.g., 'pi05_aloha', 'pi0_aloha')
        precision: Precision for conversion ('float32', 'bfloat16', 'float16')
    """
    pytorch_output_path = pathlib.Path(pytorch_output_dir).resolve()
    pytorch_output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Converting JAX checkpoint to PyTorch format...")
    logger.info(f"  JAX checkpoint: {jax_checkpoint_dir}")
    logger.info(f"  PyTorch output: {pytorch_output_path}")
    logger.info(f"  Config name: {config_name}")
    logger.info(f"  Precision: {precision}")
    
    # Build the conversion command
    convert_script = pathlib.Path(__file__).parent.parent / "examples" / "convert_jax_model_to_pytorch.py"
    
    cmd = [
        sys.executable,
        str(convert_script),
        "--checkpoint_dir", str(jax_checkpoint_dir),
        "--config_name", config_name,
        "--output_path", str(pytorch_output_path),
        "--precision", precision,
    ]
    
    logger.info(f"Running conversion command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Conversion completed successfully!")
        logger.info(f"PyTorch checkpoint saved to: {pytorch_output_path}")
        if result.stdout:
            logger.debug(f"Conversion output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Conversion failed with exit code {e.returncode}")
        if e.stdout:
            logger.error(f"stdout:\n{e.stdout}")
        if e.stderr:
            logger.error(f"stderr:\n{e.stderr}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Download JAX checkpoints from GCS and optionally convert to PyTorch")
    parser.add_argument(
        "--checkpoint_url",
        type=str,
        default="gs://openpi-assets/checkpoints/pi05_base",
        help="GCS URL of the checkpoint to download",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/jax",
        help="Local directory to save the JAX checkpoint",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force re-download even if checkpoint already exists in cache",
    )
    parser.add_argument(
        "--convert_to_pytorch",
        action="store_true",
        help="Convert the downloaded JAX checkpoint to PyTorch format",
    )
    parser.add_argument(
        "--pytorch_output_dir",
        type=str,
        default="./checkpoints/pytorch",
        help="Directory to save the converted PyTorch checkpoint (only used if --convert_to_pytorch is set)",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="pi05_aloha",
        help="Config name for model conversion (e.g., 'pi05_aloha', 'pi0_aloha'). Only used if --convert_to_pytorch is set",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="Precision for PyTorch conversion. Only used if --convert_to_pytorch is set",
    )
    
    args = parser.parse_args()
    
    # Download checkpoint
    downloaded_path = download_checkpoint(args.checkpoint_url, args.output_dir, args.force_download)
    logger.info("Download completed!")
    
    # Convert to PyTorch if requested
    if args.convert_to_pytorch:
        logger.info("=" * 60)
        logger.info("Starting PyTorch conversion...")
        logger.info("=" * 60)
        convert_to_pytorch(
            downloaded_path,
            args.pytorch_output_dir,
            args.config_name,
            args.precision,
        )
        logger.info("=" * 60)
        logger.info("All operations completed successfully!")
        logger.info(f"  JAX checkpoint: {downloaded_path}")
        logger.info(f"  PyTorch checkpoint: {args.pytorch_output_dir}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()

