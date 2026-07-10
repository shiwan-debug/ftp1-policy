"""
Validation functions for FTP1 training.

This module contains functions for computing and aggregating validation metrics
in distributed training scenarios.
"""

import logging
import matplotlib
import numpy as np
import torch
import torch.distributed as dist
from openpi.shared.wandb_compat import wandb
from openpi.pose_utils import rot6d_to_mat
import scipy.spatial.transform as st
from openpi.normalization import apply_norm_stats
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_action_dim_labels(
    single_arm_action_rep_dim: int,
    arm_joints_dim: int,
    reserved_action_dim: int,
) -> list[str]:
    """Build per-dimension action labels for plotting/analysis."""
    labels: list[str] = []

    labels.extend([f"right-wrist-pos-{axis}" for axis in ("x", "y", "z")])
    labels.extend([f"right-wrist-rot6d-{i}" for i in range(6)])
    labels.extend([f"right-arm-joint-{i}" for i in range(arm_joints_dim)])
    right_hand_dim = max(0, single_arm_action_rep_dim - 9 - arm_joints_dim)
    labels.extend([f"right-hand-joint-{i}" for i in range(right_hand_dim)])

    labels.extend([f"left-wrist-pos-{axis}" for axis in ("x", "y", "z")])
    labels.extend([f"left-wrist-rot6d-{i}" for i in range(6)])
    labels.extend([f"left-arm-joint-{i}" for i in range(arm_joints_dim)])
    left_hand_dim = max(0, single_arm_action_rep_dim - 9 - arm_joints_dim)
    labels.extend([f"left-hand-joint-{i}" for i in range(left_hand_dim)])

    labels.extend([f"head-track-pos-{axis}" for axis in ("x", "y", "z")])
    labels.extend([f"head-track-rot6d-{i}" for i in range(6)])
    labels.extend([f"supplementary-joint-{i}" for i in range(reserved_action_dim)])
    return labels


def _build_first_batch_action_curve_images(
    sample_actions: torch.Tensor,
    gt_actions: torch.Tensor,
    action_masks: torch.Tensor,
    domain_names: str | None,
    val_dataset,
    single_arm_action_rep_dim: int,
    arm_joints_dim: int,
    reserved_action_dim: int,
    max_samples: int = 10,
) -> list[wandb.Image]:
    """Build per-sample action curve figures for the first validation batch."""
    sample_actions_unnorm, gt_actions_unnorm, action_masks_np = prepare_unnormalized_validation_actions(
        sample_actions=sample_actions,
        gt_actions=gt_actions,
        action_masks=action_masks,
        domain_names=domain_names,
        val_dataset=val_dataset,
    )
    dim_labels = get_action_dim_labels(
        single_arm_action_rep_dim=single_arm_action_rep_dim,
        arm_joints_dim=arm_joints_dim,
        reserved_action_dim=reserved_action_dim,
    )

    batch_size, horizon, action_dim = sample_actions_unnorm.shape
    if len(dim_labels) != action_dim:
        dim_labels = [f"action-dim-{i}" for i in range(action_dim)]

    num_samples = min(batch_size, max_samples)
    images: list[wandb.Image] = []
    time_steps = np.arange(horizon)
    for sample_idx in range(num_samples):
        # Drop invalid dimensions directly: keep dims that are valid at least once in this sample.
        valid_dims = np.where(np.any(action_masks_np[sample_idx], axis=0))[0]
        if valid_dims.size == 0:
            continue

        n_dims = int(valid_dims.size)
        n_cols = min(8, n_dims)
        n_rows = (n_dims + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.0, n_rows * 2.2), squeeze=False)
        axes_flat = axes.flatten()

        for plot_idx, dim_idx in enumerate(valid_dims):
            ax = axes_flat[plot_idx]
            valid_t_mask = action_masks_np[sample_idx, :, dim_idx]
            if np.any(valid_t_mask):
                ax.plot(time_steps[valid_t_mask], gt_actions_unnorm[sample_idx, valid_t_mask, dim_idx], label="gt_action", linewidth=1.2)
                ax.plot(time_steps[valid_t_mask], sample_actions_unnorm[sample_idx, valid_t_mask, dim_idx], label="sampled_action", linewidth=1.2)
            ax.set_title(f"[{dim_idx}] {dim_labels[dim_idx]}", fontsize=8)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=7)

        for ax in axes_flat[n_dims:]:
            ax.axis("off")

        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9)

        domain_tag = domain_names if domain_names is not None else "unknown-domain"
        fig.suptitle(
            f"Validation first batch | sample={sample_idx} | domain={domain_tag}",
            fontsize=11,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
        images.append(
            wandb.Image(
                fig,
                caption=f"sample={sample_idx}, domain={domain_tag}, valid_dims={n_dims}",
            )
        )
        plt.close(fig)

    return images


def prepare_unnormalized_validation_actions(
    sample_actions: torch.Tensor,
    gt_actions: torch.Tensor,
    action_masks: torch.Tensor,
    domain_names: str | None,
    val_dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert tensors to numpy, apply unnormalization, and return boolean masks."""
    # Use copy() to guarantee no memory aliasing with source tensors.
    sample_actions_np = sample_actions.detach().cpu().numpy().copy()
    gt_actions_np = gt_actions.detach().cpu().numpy().copy()
    action_masks_np = action_masks.detach().cpu().numpy().copy().astype(bool)

    sample_data = {"actions": sample_actions_np}
    gt_data = {"actions": gt_actions_np}

    # Step 1: shared unnormalization
    if val_dataset.norm_stats is not None:
        sample_data = apply_norm_stats(sample_data, val_dataset.norm_stats, unnormalize=True, use_keys=["actions"])
        gt_data = apply_norm_stats(gt_data, val_dataset.norm_stats, unnormalize=True, use_keys=["actions"])

    # Step 2: domain-specific independent unnormalization
    if domain_names is not None and hasattr(val_dataset, "domain_dataset_list") and hasattr(val_dataset, "domain_list"):
        if domain_names in val_dataset.domain_list:
            domain_idx = val_dataset.domain_list.index(domain_names)
            domain_dataset = val_dataset.domain_dataset_list[domain_idx]
            if domain_dataset.norm_stats is not None:
                sample_data = apply_norm_stats(sample_data, domain_dataset.norm_stats, unnormalize=True, use_keys=["actions"])
                gt_data = apply_norm_stats(gt_data, domain_dataset.norm_stats, unnormalize=True, use_keys=["actions"])

    return sample_data["actions"], gt_data["actions"], action_masks_np


def compute_validation_metrics(
    sample_actions: torch.Tensor,
    gt_actions: torch.Tensor,
    action_masks: torch.Tensor,
    domain_names: str | None,
    val_dataset,
    single_arm_action_rep_dim: int,
    arm_joints_dim: int,
    reserved_action_dim: int,
    error_quantile: float = 0.0,
    mape_threshold: float = 0.001,
) -> dict:
    """
    Compute validation metrics including RMSE and jitter metrics for different action parts and domains.
    
    Args:
        sample_actions: Predicted actions, shape (B, T, action_dim)
        gt_actions: Ground truth actions, shape (B, T, action_dim)
        domain_names: Domain names for this batch (all samples in batch are from the same domain due to domain_batch_split=True)
        val_dataset: Validation dataset (MultiZarrDataset)
        single_arm_action_rep_dim: Dimension of single arm action representation (default 32)
        error_quantile: If > 0, only compute metrics using errors between quantile and (1-quantile) percentiles.
                        For example, if error_quantile=0.1, only use errors between 10th and 90th percentiles.
        mape_threshold: Threshold for MAPE calculation. GT values with absolute value less than this threshold
                        will be excluded from MAPE computation to avoid division by very small numbers.
                        Default is 0.001.
    
    Returns:
        Dictionary containing validation metrics organized by total and domain.
        Includes:
        - rmse_* / mape_*: prediction-vs-GT metrics
        - jitter_rms_*: intra-chunk jitter metric computed from second-order temporal differences
    """
    sample_actions_unnorm, gt_actions_unnorm, action_masks_np = prepare_unnormalized_validation_actions(
        sample_actions=sample_actions,
        gt_actions=gt_actions,
        action_masks=action_masks,
        domain_names=domain_names,
        val_dataset=val_dataset,
    )
    
    # Note: We keep sample_actions_unnorm and gt_actions_unnorm for computation below
    # We'll delete intermediate arrays after they're no longer needed
    
    action_dim = 2 * single_arm_action_rep_dim + 9 + reserved_action_dim
    
    # # Without Joints Version
    # action_parts = {
    #     'right-wrist-pos': (0, 3),
    #     'right-wrist-rot': (3, 9),
    #     'right-hand-joint': (9, single_arm_action_rep_dim),
    #     'left-wrist-pos': (single_arm_action_rep_dim, single_arm_action_rep_dim + 3),
    #     'left-wrist-rot': (single_arm_action_rep_dim + 3, single_arm_action_rep_dim + 9),
    #     'left-hand-joint': (single_arm_action_rep_dim + 9, 2 * single_arm_action_rep_dim),
    #     'head-track-pos': (2 * single_arm_action_rep_dim, 2 * single_arm_action_rep_dim + 3),
    #     'head-track-rot': (2 * single_arm_action_rep_dim + 3, 2 * single_arm_action_rep_dim + 9),
    #     'total': (0, action_dim),  # Total RMSE over all action dimensions
    # }
    
    # With Joints Version
    action_parts = {
        'right-wrist-pos': (0, 3),
        'right-wrist-rot': (3, 9),
        'right-arm-joints': (9, 9 + arm_joints_dim),
        'right-hand-joint': (9 + arm_joints_dim, single_arm_action_rep_dim),
        'left-wrist-pos': (single_arm_action_rep_dim, single_arm_action_rep_dim + 3),
        'left-wrist-rot': (single_arm_action_rep_dim + 3, single_arm_action_rep_dim + 9),
        'left-arm-joints': (single_arm_action_rep_dim + 9, single_arm_action_rep_dim + 9 + arm_joints_dim),
        'left-hand-joint': (single_arm_action_rep_dim + 9 + arm_joints_dim, 2 * single_arm_action_rep_dim),
        'head-track-pos': (2 * single_arm_action_rep_dim, 2 * single_arm_action_rep_dim + 3),
        'head-track-rot': (2 * single_arm_action_rep_dim + 3, 2 * single_arm_action_rep_dim + 9),
        'supplementary-joints': (2 * single_arm_action_rep_dim + 9, action_dim),
        'total': (0, action_dim),  # Total RMSE over all action dimensions
    }
    
    # Compute RMSE for each action part (across batch and time) using numpy
    metrics_total = {}
    for part_name, (start_idx, end_idx) in action_parts.items():
        pred_part = sample_actions_unnorm[..., start_idx:end_idx]
        gt_part = gt_actions_unnorm[..., start_idx:end_idx]
        
        # Apply mask for this part
        mask_part = action_masks_np[..., start_idx:end_idx]
        # Compute MSE only for valid positions (mask == 1)
        squared_error = (pred_part - gt_part) ** 2
        absolute_error = np.abs(pred_part - gt_part)
        # Only compute mean over valid positions
        valid_count = np.sum(mask_part)
        if valid_count > 0:
            # Get all valid errors
            valid_squared_errors = squared_error[mask_part]
            valid_absolute_errors = absolute_error[mask_part]
            
            # Apply error quantile filtering if specified
            if error_quantile > 0.0 and error_quantile < 0.5:
                # Calculate quantiles of squared errors
                q_low = np.quantile(valid_squared_errors, error_quantile)
                q_high = np.quantile(valid_squared_errors, 1.0 - error_quantile)
                # Filter errors within quantile range
                quantile_mask = (valid_squared_errors >= q_low) & (valid_squared_errors <= q_high)
                filtered_squared_errors = valid_squared_errors[quantile_mask]
                
                if len(filtered_squared_errors) > 0:
                    mse = np.mean(filtered_squared_errors)
                    rmse = np.sqrt(mse)
                else:
                    rmse = -1.0
            else:
                mse = np.mean(valid_squared_errors)
                rmse = np.sqrt(mse)
            
            # Compute MAPE
            # Filter out GT values with absolute value less than threshold to avoid division by very small numbers
            valid_gt_abs = np.abs(gt_part)[mask_part]  # Get absolute GT values for valid positions
            threshold_mask = valid_gt_abs >= mape_threshold  # Filter by threshold
            
            if np.sum(threshold_mask) > 0:
                # Only compute MAPE for GT values above threshold
                filtered_absolute_errors = valid_absolute_errors[threshold_mask]
                filtered_gt_values = valid_gt_abs[threshold_mask]
                
                if error_quantile > 0.0 and error_quantile < 0.5:
                    # For MAPE, filter based on absolute error quantiles (on already threshold-filtered data)
                    q_low_abs = np.quantile(filtered_absolute_errors, error_quantile)
                    q_high_abs = np.quantile(filtered_absolute_errors, 1.0 - error_quantile)
                    quantile_mask_abs = (filtered_absolute_errors >= q_low_abs) & (filtered_absolute_errors <= q_high_abs)
                    final_absolute_errors = filtered_absolute_errors[quantile_mask_abs]
                    final_gt_values = filtered_gt_values[quantile_mask_abs]
                    if len(final_absolute_errors) > 0 and np.sum(final_gt_values) > 0:
                        mape = np.sum(final_absolute_errors) / np.sum(final_gt_values)
                    else:
                        mape = -1.0
                else:
                    if np.sum(filtered_gt_values) > 0:
                        mape = np.sum(filtered_absolute_errors) / np.sum(filtered_gt_values)
                    else:
                        mape = -1.0
            else:
                # No GT values above threshold, set MAPE to -1.0
                mape = -1.0
        else:
            # No valid data for this part, set RMSE to 0 or NaN
            rmse = -1.0
            mape = -1.0
        
        
        metrics_total[f'rmse_{part_name}'] = float(rmse)
        metrics_total[f'mape_{part_name}'] = float(mape)

        # Intra-chunk jitter metric:
        # Use second-order temporal difference on predictions to quantify local oscillation.
        # Lower jitter_rms indicates smoother action sequences.
        if pred_part.shape[1] >= 3:
            second_diff = pred_part[:, 2:, :] - 2.0 * pred_part[:, 1:-1, :] + pred_part[:, :-2, :]
            jitter_mask = mask_part[:, 2:, :] & mask_part[:, 1:-1, :] & mask_part[:, :-2, :]
            valid_second_diff = second_diff[jitter_mask]
            if valid_second_diff.size > 0:
                # Cast to float64 before squaring to avoid float32 overflow when model
                # outputs are very large (e.g. gradient explosion). Check finite to
                # gracefully handle NaN/inf without crashing the training loop.
                jitter_rms_val = np.sqrt(np.mean(valid_second_diff.astype(np.float64) ** 2))
                jitter_rms = float(jitter_rms_val) if np.isfinite(jitter_rms_val) else -1.0
            else:
                jitter_rms = -1.0
        else:
            jitter_rms = -1.0
        metrics_total[f'jitter_rms_{part_name}'] = float(jitter_rms)
        
        # Calculate human-interpretable metrics
        if 'rot' in part_name:
            # For 6D rotation parts, treat a timestep as valid only if all dims are valid.
            valid_rot_mask = np.all(mask_part, axis=-1).flatten()      # (B*T,)
            pred_part_flat = pred_part.reshape(-1, 6)
            gt_part_flat = gt_part.reshape(-1, 6)
            # Also exclude rows containing NaN/inf (e.g. from gradient explosion) to
            # prevent SVD non-convergence inside st.Rotation.from_matrix.
            finite_mask = (
                np.all(np.isfinite(pred_part_flat), axis=-1)
                & np.all(np.isfinite(gt_part_flat), axis=-1)
            )
            combined_rot_mask = valid_rot_mask & finite_mask
            if np.any(combined_rot_mask):
                pred_part_rot = st.Rotation.from_matrix(rot6d_to_mat(pred_part_flat[combined_rot_mask]))
                gt_part_rot = st.Rotation.from_matrix(rot6d_to_mat(gt_part_flat[combined_rot_mask]))
                # Rotation error in degrees: angle of relative rotation (pred vs gt)
                rel_rot = pred_part_rot * gt_part_rot.inv()
                rot_err_deg_valid = np.degrees(rel_rot.magnitude())

                # Apply error quantile filtering if specified
                if error_quantile > 0.0 and error_quantile < 0.5:
                    q_low = np.quantile(rot_err_deg_valid, error_quantile)
                    q_high = np.quantile(rot_err_deg_valid, 1.0 - error_quantile)
                    quantile_mask = (rot_err_deg_valid >= q_low) & (rot_err_deg_valid <= q_high)
                    if np.any(quantile_mask):
                        metrics_total[f'spc_RotErrDegree_{part_name}'] = float(np.mean(rot_err_deg_valid[quantile_mask]))
                    else:
                        metrics_total[f'spc_RotErrDegree_{part_name}'] = -1.0
                else:
                    metrics_total[f'spc_RotErrDegree_{part_name}'] = float(np.mean(rot_err_deg_valid))
            else:
                metrics_total[f'spc_RotErrDegree_{part_name}'] = -1.0
        if 'pos' in part_name:
            valid_pos_mask = np.all(mask_part, axis=-1).flatten()      # (B*T,)
            if np.any(valid_pos_mask):
                pred_part_pos = pred_part.reshape(-1, 3)
                gt_part_pos = gt_part.reshape(-1, 3)
                pos_errors = np.linalg.norm(pred_part_pos[valid_pos_mask] - gt_part_pos[valid_pos_mask], axis=-1)
                
                # Apply error quantile filtering if specified
                if error_quantile > 0.0 and error_quantile < 0.5:
                    q_low = np.quantile(pos_errors, error_quantile)
                    q_high = np.quantile(pos_errors, 1.0 - error_quantile)
                    quantile_mask = (pos_errors >= q_low) & (pos_errors <= q_high)
                    if np.any(quantile_mask):
                        metrics_total[f'spc_PosErrMeter_{part_name}'] = float(np.mean(pos_errors[quantile_mask]))
                    else:
                        metrics_total[f'spc_PosErrMeter_{part_name}'] = -1.0
                else:
                    metrics_total[f'spc_PosErrMeter_{part_name}'] = float(np.mean(pos_errors))
            else:
                metrics_total[f'spc_PosErrMeter_{part_name}'] = -1.0
        
    # Compute domain-specific metrics (entire batch is from the same domain)
    metrics_by_domain = {}
    if domain_names is not None:  
        metrics_by_domain[domain_names] = metrics_total.copy()
    
    # Clean up large numpy arrays that are no longer needed
    # Delete arrays after all computations are complete
    del sample_actions_unnorm, gt_actions_unnorm
    # Note: action_masks_np is used throughout the loop above, so we delete it here
    # Arrays were created in prepare_unnormalized_validation_actions and are no longer needed here.
    
    return {
        'total': metrics_total,
        'by_domain': metrics_by_domain,
    }


def aggregate_validation_metrics(val_infos: list) -> dict:
    """
    Aggregate validation metrics across all batches.
    
    Args:
        val_infos: List of validation info dictionaries from each batch
    
    Returns:
        Aggregated metrics dictionary
    """
    if not val_infos:
        return {}
    
    # Aggregate total metrics (average across batches)
    # Filter out -1.0 values which indicate invalid/missing data
    total_metrics = {}
    for key in val_infos[0]['total'].keys():
        values = [info['total'][key] for info in val_infos if key in info['total']]
        # Filter out -1.0 values (invalid data markers)
        valid_values = [v for v in values if v >= 0.0]
        if valid_values:
            total_metrics[key] = sum(valid_values) / len(valid_values)
        else:
            # If no valid values, set to -1.0 to indicate missing data
            total_metrics[key] = -1.0
    
    # Aggregate domain-level metrics
    domain_metrics = {}
    for info in val_infos:
        for domain, metrics in info.get('by_domain', {}).items():
            if domain not in domain_metrics:
                domain_metrics[domain] = {}
            for key, value in metrics.items():
                if key not in domain_metrics[domain]:
                    domain_metrics[domain][key] = []
                domain_metrics[domain][key].append(value)
    
    # Average domain metrics (filter out -1.0 values)
    for domain in domain_metrics:
        for key in domain_metrics[domain]:
            valid_values = [v for v in domain_metrics[domain][key] if v >= 0.0]
            if valid_values:
                domain_metrics[domain][key] = sum(valid_values) / len(valid_values)
            else:
                domain_metrics[domain][key] = -1.0
    
    return {
        'total': total_metrics,
        'by_domain': domain_metrics,
    }


def aggregate_validation_metrics_across_ranks(val_infos: list, use_ddp: bool, device: torch.device | None = None) -> dict:
    """
    Aggregate validation metrics across all batches and all ranks (for distributed training).
    
    Args:
        val_infos: List of validation info dictionaries from each batch on this rank
        use_ddp: Whether distributed training is enabled
    
    Returns:
        Aggregated metrics dictionary across all ranks
    """
    if not use_ddp:
        # Single GPU case, just aggregate locally
        return aggregate_validation_metrics(val_infos)
    
    # Collect all val_infos from all ranks
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # Gather all val_infos from all ranks
    # We need to serialize the val_infos to send them across ranks
    import pickle
    import numpy as np
    
    try:
        val_infos_bytes = pickle.dumps(val_infos)
        val_infos_length = len(val_infos_bytes)
    except Exception as e:
        logging.error(f"[Rank {rank}] Failed to serialize val_infos: {e}")
        raise
    
    # Create tensors for gathering
    # Use the provided device or default to current device
    # IMPORTANT: For DDP, all ranks must use the same device type (all CUDA or all CPU)
    # but each rank should use its own local device (e.g., cuda:0, cuda:1, etc.)
    if device is None:
        # Default to the current CUDA device if available, otherwise CPU
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{torch.cuda.current_device()}')
        else:
            device = torch.device('cpu')
    
    # Ensure tensor is on the correct device for this rank
    length_tensor = torch.tensor([val_infos_length], dtype=torch.long, device=device)
    
    # Gather lengths from all ranks
    gathered_lengths = [torch.zeros_like(length_tensor) for _ in range(world_size)]
    try:
        dist.all_gather(gathered_lengths, length_tensor)
    except Exception as e:
        logging.error(f"[Rank {rank}] Failed to gather lengths: {e}")
        raise
    
    max_length = max(length.item() for length in gathered_lengths)
    
    # Pad and gather the actual data
    if max_length > 0:
        val_infos_array = np.frombuffer(val_infos_bytes, dtype=np.uint8)
        val_infos_tensor = torch.zeros(max_length, dtype=torch.uint8, device=device)
        # Create a writable copy to avoid "array is not writable" warning
        val_infos_array_writable = np.array(val_infos_array, copy=True)
        val_infos_tensor[:len(val_infos_array_writable)] = torch.from_numpy(val_infos_array_writable)
        
        gathered_tensors = [torch.zeros_like(val_infos_tensor) for _ in range(world_size)]
        try:
            dist.all_gather(gathered_tensors, val_infos_tensor)
        except Exception as e:
            logging.error(f"[Rank {rank}] Failed to gather data tensors: {e}")
            raise
        
        # Deserialize all val_infos from all ranks
        all_val_infos = []
        for rank_idx, tensor in enumerate(gathered_tensors):
            length = gathered_lengths[rank_idx].item()
            if length > 0:
                try:
                    val_infos_bytes = bytes(tensor[:length].cpu().numpy().tobytes())
                    rank_val_infos = pickle.loads(val_infos_bytes)
                    all_val_infos.extend(rank_val_infos)
                except Exception as e:
                    logging.error(f"[Rank {rank}] Failed to deserialize val_infos from rank {rank_idx}: {e}")
                    raise
    else:
        all_val_infos = []
    
    # Aggregate all val_infos from all ranks
    result = aggregate_validation_metrics(all_val_infos)
    
    # Explicitly clean up tensors and temporary objects to prevent memory leaks
    # Only delete variables that were actually created
    del all_val_infos
    del gathered_lengths
    del length_tensor
    if max_length > 0:
        # These variables only exist when max_length > 0
        del gathered_tensors
        del val_infos_tensor
    # Note: We don't call empty_cache() here to avoid performance overhead
    # The caller (validation loop) will call it once at the end
    
    return result
