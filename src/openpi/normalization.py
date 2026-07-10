import numpy as np
import torch
import copy

from openpi.tactile_identity import build_tactile_sensor_type_shape_key_from_data

EPS = 1e-8
IMAGE_INPUT_PREPROCESS_DIV255_MUL2_MINUS1 = "div255_mul2_minus1"


def _apply_input_preprocess(data, params):
    if not isinstance(params, dict):
        return data
    preprocess = params.get("input_preprocess")
    if preprocess is None:
        return data
    if preprocess == IMAGE_INPUT_PREPROCESS_DIV255_MUL2_MINUS1:
        return data / 255.0 * 2.0 - 1.0
    raise ValueError(f"Unknown input_preprocess: {preprocess}")


def _invert_input_preprocess(data, params):
    if not isinstance(params, dict):
        return data
    preprocess = params.get("input_preprocess")
    if preprocess is None:
        return data
    if preprocess == IMAGE_INPUT_PREPROCESS_DIV255_MUL2_MINUS1:
        return (data + 1.0) / 2.0 * 255.0
    raise ValueError(f"Unknown input_preprocess: {preprocess}")


def _calculate_quantile(data, pre_shape=None):
    if pre_shape is None:
        pre_shape = data.shape
    q01, q99 = np.quantile(data, [0.01, 0.99], axis=0)
    # handle if some dimension has all the same value case
    value_same = np.isclose(q01, q99, atol=EPS)
    if np.any(value_same):
        original_value = q01[value_same]
        q01[value_same] = original_value - 0.5
        q99[value_same] = original_value + 0.5
    params = {
        'q01': q01.reshape(pre_shape),
        'q99': q99.reshape(pre_shape),
    }
    return params

def _calculate_z_score(data, pre_shape=None):
    if pre_shape is None:
        pre_shape = data.shape
    mean, std = np.mean(data, axis=0), np.std(data, axis=0)
    # Protect zero-variance dims AND NaN std (np.isclose(NaN, 0) == False, so NaN
    # would otherwise slip through and cause "invalid value in divide" at runtime).
    value_same = np.isclose(std, 0.0, atol=EPS) | ~np.isfinite(std)
    if np.any(value_same):
        std[value_same] = 1.0
    params = {
        'mean': mean.reshape(pre_shape),
        'std': std.reshape(pre_shape),
    }
    return params


def _normalize_quantile(data, params, dtype=np.float32):
    """Reference: LBM (https://arxiv.org/abs/2507.05331)
    
    Args:
        data: numpy array to normalize
        params: normalization parameters containing 'q01' and 'q99'
        dtype: output dtype, default is np.float32
    """
    if isinstance(params, str):
        if params == 'ignore':
            return data
        elif params == 'image_norm':
            return data / 255.0 * 2.0 - 1.0
    data = _apply_input_preprocess(data, params)
    q01, q99 = params['q01'], params['q99']
    
    # Quantile normalization:
    # (x - q01) / (q99 - q01 + 1e-6) * 2 - 1, then clip to bound
    # outliers caused by stats/data mismatches.
    data = (data - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    data = np.clip(data, -2.0, 2.0)
    # Convert to specified dtype after normalization
    return data.astype(dtype)

def _normalize_z_score(data, params, dtype=np.float32):
    """Normalize data using z-score normalization.
    
    Args:
        data: numpy array to normalize
        params: normalization parameters containing 'mean' and 'std'
        dtype: output dtype, default is np.float32
    """
    if isinstance(params, str):
        if params == 'ignore':
            return data
        elif params == 'image_norm':
            return data / 255.0 * 2.0 - 1.0
    data = _apply_input_preprocess(data, params)
    mean, std = params['mean'], params['std']
    # Runtime protection: guard against zero or NaN std that may slip through param
    # calculation (e.g. when training data contains NaN, np.isclose(NaN,0)==False).
    safe_std = np.where(np.abs(std) < EPS, 1.0, std)
    data = (data - mean) / safe_std
    # Clip to prevent catastrophically large values when std is severely under-
    # estimated (e.g. norm_sample_ratio too small + relative-pose action rep).
    # ±5σ preserves 99.9999% of a correctly normalised distribution while capping
    # gradient explosion.
    data = np.clip(data, -10.0, 10.0)
    # Convert to specified dtype after normalization
    return data.astype(dtype)

def _unnormalize_quantile(data, params):
    "Reference: LBM (https://arxiv.org/abs/2507.05331)"
    if isinstance(params, str):
        if params == 'ignore':
            return data
        elif params == 'image_norm':
            return (data + 1.0) / 2.0 * 255.0
    q01, q99 = params['q01'], params['q99']
    data = (data + 1.0) / 2.0 * (q99 - q01) + q01
    data = _invert_input_preprocess(data, params)
    # For invalid regions (q01 == q99 == 0), data remains unchanged
    return data

def _unnormalize_z_score(data, params):
    if isinstance(params, str):
        if params == 'ignore':
            return data
        elif params == 'image_norm':
            return (data + 1.0) / 2.0 * 255.0
    mean, std = params['mean'], params['std']
    data = data * std + mean
    data = _invert_input_preprocess(data, params)
    return data


def _calculate_norm_params(data, norm_type, norm_time_dim):
    """
    data: (B, T, *shape)
    norm_type: str, one of 'quantile' or 'zscore'
    """
        # Convert to numpy if needed
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    if norm_time_dim is False:  # if not normalize the time dim, flatten the data to get the norm for each feature dim
        pre_shape = data.shape[2:]
        B, T, *shape = data.shape
        data_flat = data.reshape(B*T, *shape)
    else:                       # if normalize the time dim, do not flatten the data to keep time dim as a part of 'feature dime'
        pre_shape = data.shape[1:]
        B, *shape = data.shape
        data_flat = data.reshape(B, -1)

    if norm_type == 'quantile':
        return _calculate_quantile(data_flat, pre_shape)
    elif norm_type == 'zscore':
        return _calculate_z_score(data_flat, pre_shape)
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}. Must be 'quantile' or 'zscore'")


def _calculate_quantile_with_action_mask(data, action_mask, pre_shape):
    """
    Calculate quantile normalization parameters using action mask.
    Only uses data where action_mask == 1 (valid data).
    If a dimension has no valid data, returns 'ignore' for that dimension.
    
    Args:
        data: (B, *shape) numpy array
        action_mask: (B, *shape) numpy array, 1 for valid data, 0 for invalid
        pre_shape: shape to reshape the output parameters
    
    Returns:
        params: dict with 'q01' and 'q99', or 'ignore' if no valid data
    """
    
    if pre_shape is None:
        pre_shape = data.shape[1:]
    
    # Convert to numpy if needed
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    if isinstance(action_mask, torch.Tensor):
        action_mask = action_mask.numpy()
    
    # Ensure mask is boolean (1 for valid, 0 for invalid)
    valid_mask = action_mask.astype(bool)
    
    # Reshape to (B, feature_dim) for easier processing
    B = data.shape[0]
    feature_shape = data.shape[1:]
    data_flat = data.reshape(B, -1)  # (B, feature_dim)
    mask_flat = valid_mask.reshape(B, -1)  # (B, feature_dim)
    
    # Calculate quantiles for each feature dimension
    feature_dim = data_flat.shape[1]
    q01_list = []
    q99_list = []
    
    for dim_idx in range(feature_dim):
        # Get valid data for this dimension
        valid_data = data_flat[mask_flat[:, dim_idx], dim_idx]
        
        if len(valid_data) == 0:
            # not used anyway, will be ignored in the final caluclation by action_mask
            # we set q01 = -0.5, q99 = 0.5 to make the normalize equivalent to no normalization: (0 - (-0.5)) / (0.5 - (-0.5)) * 2 - 1 = 0
            q01 = -0.5
            q99 = 0.5
        else:
            # Calculate quantiles
            q01 = np.quantile(valid_data, 0.01)
            q99 = np.quantile(valid_data, 0.99)
        if np.isclose(q01, q99, atol=EPS):
            original_value = q01
            q01 = original_value - 0.5
            q99 = original_value + 0.5
        q01_list.append(q01)
        q99_list.append(q99)
        # print(f"dim_idx: {dim_idx}, q01: {q01}, q99: {q99}, mask_ratio: {len(valid_data)}")
        # import pdb; pdb.set_trace()
    
    # Reshape back to pre_shape
    q01_array = np.array(q01_list).reshape(feature_shape)
    q99_array = np.array(q99_list).reshape(feature_shape)
    
    params = {
        'q01': q01_array.reshape(pre_shape),
        'q99': q99_array.reshape(pre_shape),
    }
    return params


def _calculate_z_score_with_action_mask(data, action_mask, pre_shape):
    """
    Calculate z-score normalization parameters using action mask.
    Only uses data where action_mask == 1 (valid data).
    If a dimension has no valid data, returns 'ignore' for that dimension.
    
    Args:
        data: (B, *shape) numpy array
        action_mask: (B, *shape) numpy array, 1 for valid data, 0 for invalid
        pre_shape: shape to reshape the output parameters
    
    Returns:
        params: dict with 'mean' and 'std', or 'ignore' if no valid data
    """
    if pre_shape is None:
        pre_shape = data.shape[1:]
    
    # Convert to numpy if needed
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    if isinstance(action_mask, torch.Tensor):
        action_mask = action_mask.numpy()
    
    # Ensure mask is boolean (1 for valid, 0 for invalid)
    valid_mask = action_mask.astype(bool)
    
    # Reshape to (B, feature_dim) for easier processing
    B = data.shape[0]
    feature_shape = data.shape[1:]
    data_flat = data.reshape(B, -1)  # (B, feature_dim)
    mask_flat = valid_mask.reshape(B, -1)  # (B, feature_dim)
    
    # Calculate mean and std for each feature dimension
    feature_dim = data_flat.shape[1]
    mean_list = []
    std_list = []
    
    for dim_idx in range(feature_dim):
        # Get valid data for this dimension
        valid_data = data_flat[mask_flat[:, dim_idx], dim_idx]
        
        if len(valid_data) == 0:
            # No valid data for this dimension, use identity parameters (mean=0, std=1)
            # This makes normalize equivalent to no normalization: (data - 0) / (1 + 1e-7) ≈ data
            mean = 0.0
            std = 1.0
        else:
            # Calculate mean and std
            mean = np.mean(valid_data)
            std = np.std(valid_data)
        # Protect zero-variance dims AND NaN std (np.isclose(NaN, 0) == False)
        if np.isclose(std, 0.0, atol=EPS) or not np.isfinite(std):
            std = 1.0
        mean_list.append(mean)
        std_list.append(std)

    # Reshape back to pre_shape
    mean_array = np.array(mean_list).reshape(feature_shape)
    std_array = np.array(std_list).reshape(feature_shape)
    
    params = {
        'mean': mean_array.reshape(pre_shape),
        'std': std_array.reshape(pre_shape),
    }
    return params


def _calculate_norm_params_with_action_mask(data, action_mask, norm_type, norm_time_dim):
    """
    data: (B, T, *shape)
    action_mask: (B, T, self.single_arm_action_rep_dim * 2 + 9)
    norm_type: str, one of 'quantile' or 'zscore'
    """
    if norm_time_dim is False:  # if not normalize the time dim, flatten the data to get the norm for each feature dim
        pre_shape = data.shape[2:]
        B, T, *shape = data.shape
        data_flat = data.reshape(B*T, *shape)
        action_mask_flat = action_mask.reshape(B*T, -1)
    else:                       # if normalize the time dim, do not flatten the data to keep time dim as a part of 'feature dime'
        pre_shape = data.shape[1:]
        B, *shape = data.shape
        data_flat = data.reshape(B, -1)
        action_mask_flat = action_mask.reshape(B, -1)

    if norm_type == 'quantile':
        return _calculate_quantile_with_action_mask(data_flat, action_mask_flat, pre_shape)
    elif norm_type == 'zscore':
        return _calculate_z_score_with_action_mask(data_flat, action_mask_flat, pre_shape)
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}. Must be 'quantile' or 'zscore'")


def _calculate_channel_level_norm_params(data, norm_type):
    """Calculate per-channel normalization params for image-like data.

    Args:
        data: Array of shape (N, C), where C is the channel dimension.
        norm_type: One of {'quantile', 'zscore'}.
    """
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"Expected channel pool with shape (N, C), got {data.shape}")

    # Image tactile channel stats are computed on preprocessed values in [-1, 1].
    data = data.astype(np.float32, copy=False) / 255.0 * 2.0 - 1.0
    pre_shape = (data.shape[-1],)
    if norm_type == 'quantile':
        params = _calculate_quantile(data, pre_shape)
    elif norm_type == 'zscore':
        params = _calculate_z_score(data, pre_shape)
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}. Must be 'quantile' or 'zscore'")
    params["input_preprocess"] = IMAGE_INPUT_PREPROCESS_DIV255_MUL2_MINUS1
    return params


def calculate_norm_stats(
    data_cache_input,
    data_cache_tactile_type=None,
    data_cache_action_mask=None,
    data_cache_image_channel_pool=None,
    data_cache_image_channel_params=None,
    image_tactile_norm_mode: str = "channel_wise",
):
    """
    A Function to calculate the normalization statistics for the data.
    Input:
        data_cache: a dictionary containing the data to be normalized.
        data_cache_tactile_type: a dictionary containing the tactile type of the data, ['state', 'binary', 'image'].
        data_cache_image_channel_pool: optional dict[str, np.ndarray] where each value is (N, C)
            sampled channel vectors for image-type tactile keys.
        data_cache_image_channel_params: optional dict[str, dict] where each value is
            precomputed per-channel params for image tactile keys (incremental mode).
        image_tactile_norm_mode: normalization mode for image-type tactile.
            - 'channel_wise': use per-channel stats when available, fallback to 'image_norm'.
            - 'image_norm': always use fixed image normalization ('image_norm').
    Output:
        norm_stats: a dictionary containing the normalization statistics.
            norm_stats['norm_time_dim']: a boolean indicating whether to normalize the time dimension.
            norm_stats['norm_type']: a str indicating normalization type ('quantile', 'zscore').
            norm_stats['params']: a dictionary containing the normalization parameters.
            norm_stats['norm_dim']: a dictionary containing the normalization dimensions.
            norm_stats['data_dim']: a dictionary containing the data dimensions.
    """
    norm_stats = dict()
    norm_time_dim = data_cache_input['norm_time_dim']
    norm_type = data_cache_input['norm_type']
    data_norm_dim = data_cache_input['data_norm_dim']
    data_cache = data_cache_input['data_cache']
    if image_tactile_norm_mode not in {"channel_wise", "image_norm"}:
        raise ValueError(
            "Invalid image_tactile_norm_mode: "
            f"{image_tactile_norm_mode}. Please choose from ['channel_wise', 'image_norm']"
        )

    norm_stats['norm_time_dim'] = norm_time_dim
    norm_stats['norm_type'] = norm_type
    norm_stats['params'] = dict()
    norm_stats['norm_dim'] = dict()
    norm_stats['data_dim'] = dict()

    for key in data_cache.keys():
        data = data_cache[key]
        norm_dim = data_norm_dim[key]
        if isinstance(data, list):          # direct normalization 
            if norm_dim is None:
                norm_dim = list(range(data[0].shape[-1]))
            if key == 'actions' and data_cache_action_mask is not None:
                data = torch.cat(data, dim=0)
                action_mask = torch.cat(data_cache_action_mask, dim=0)
                norm_stats['params'][key] = _calculate_norm_params_with_action_mask(data[..., norm_dim], action_mask[..., norm_dim], norm_type, norm_time_dim)
            else:
                data = torch.cat(data, dim=0)
                norm_stats['params'][key] = _calculate_norm_params(data[..., norm_dim], norm_type, norm_time_dim)
            norm_stats['data_dim'][key] = data.shape[-1]
            norm_stats['norm_dim'][key] = norm_dim
        elif isinstance(data, dict):
            norm_stats['data_dim'][key] = dict()
            norm_stats['norm_dim'][key] = dict()
            norm_stats['params'][key] = dict()
            for data_item_key in data.keys():
                if key == 'tactile' and data_cache_tactile_type is not None:
                    tactile_type = data_cache_tactile_type[data_item_key]
                else:
                    tactile_type = 'state'
                norm_dim_item = None if norm_dim is None else norm_dim[data_item_key]
                data_item = data[data_item_key][0]

                if data_item is None:
                    if tactile_type == 'image':
                        data_item = np.zeros((1, 224, 224, 3), dtype=np.float32)
                    else:
                        raise NotImplementedError(f"Data item is None for key: {key}, data_item_key: {data_item_key}, tactile_type: {tactile_type}")
                if norm_dim_item is None:
                    norm_dim_item = list(range(data_item.shape[-1]))
                norm_stats['norm_dim'][key][data_item_key] = norm_dim_item
                norm_stats['data_dim'][key][data_item_key] = data_item.shape[-1]
                if tactile_type == 'state' or tactile_type == 'matrix':
                    data_item = torch.cat(data[data_item_key], dim=0)
                    norm_stats['params'][key][data_item_key] = _calculate_norm_params(data_item[..., norm_dim_item], norm_type, norm_time_dim)
                elif tactile_type == 'binary':
                    norm_stats['params'][key][data_item_key] = 'ignore'
                elif tactile_type == 'image':
                    if image_tactile_norm_mode == "image_norm":
                        norm_stats['params'][key][data_item_key] = 'image_norm'
                        continue

                    image_params = None
                    if data_cache_image_channel_params is not None:
                        image_params = data_cache_image_channel_params.get(data_item_key)
                    if image_params is not None:
                        norm_stats['params'][key][data_item_key] = image_params
                        continue

                    # For image-type tactile, prefer sampled channel-level statistics to
                    # avoid caching full images in memory when computing norm stats.
                    image_pool = None
                    if data_cache_image_channel_pool is not None:
                        image_pool = data_cache_image_channel_pool.get(data_item_key)

                    if image_pool is not None and len(image_pool) > 0:
                        norm_stats['params'][key][data_item_key] = _calculate_channel_level_norm_params(
                            image_pool, norm_type
                        )
                    else:
                        # Fallback for old assets or empty pools.
                        norm_stats['params'][key][data_item_key] = 'image_norm'
                else:
                    raise ValueError(f"Invalid tactile_type: {tactile_type}, please choose from ['state', 'binary', 'image', 'matrix']")
    return norm_stats


def apply_norm_stats(data, norm_stats, unnormalize=False, use_keys=None, dtype=np.float32):
    """
    A Function to apply the normalization statistics to the data.
    Input:
        data: a dictionary containing the data to be normalized (numpy arrays).
        norm_stats: a dictionary containing the normalization statistics.
        unnormalize: a boolean indicating whether to unnormalize the data.
        use_keys: a list of keys to apply the normalization statistics to.
        dtype: output dtype for normalized data, default is np.float32.
    Output:
        data: a dictionary containing the normalized data.
    """ 
    data = copy.deepcopy(data)
    norm_time_dim = norm_stats['norm_time_dim']
    norm_type = norm_stats['norm_type']
    params = norm_stats['params']
    data_norm_dim = norm_stats['norm_dim']
    data_dim = norm_stats['data_dim']

    def _resolve_tactile_stats_key(tactile_item_key: str) -> str:
        tactile_params = params.get('tactile', {})
        if tactile_item_key in tactile_params:
            return tactile_item_key

        tactile_sensors = data.get('tactile_sensor')
        if not isinstance(tactile_sensors, dict) or tactile_item_key not in tactile_sensors:
            raise KeyError(
                f"Tactile norm stats not found for key '{tactile_item_key}', and tactile_sensor metadata is missing."
            )

        tactile_function_areas = data.get('tactile_function_area')
        func_areas = None
        if isinstance(tactile_function_areas, dict):
            func_areas = tactile_function_areas.get(tactile_item_key)

        tactile_types = data.get('tactile_type')
        if not isinstance(tactile_types, dict) or tactile_item_key not in tactile_types:
            raise KeyError(
                f"Tactile norm stats not found for key '{tactile_item_key}', and tactile_type metadata is missing."
            )

        tactile_stats_key = build_tactile_sensor_type_shape_key_from_data(
            tactile_sensors[tactile_item_key],
            tactile_types[tactile_item_key],
            data['tactile'][tactile_item_key],
            tactile_function_areas=func_areas,
        )
        if tactile_stats_key in tactile_params:
            return tactile_stats_key

        raise KeyError(
            f"Tactile norm stats not found for key '{tactile_item_key}'. "
            f"Tried sensor+type+shape key '{tactile_stats_key}'. Available tactile norm keys: {list(tactile_params.keys())[:8]}"
        )

    def _ensure_float_destination(arr, result_arr):
        """Avoid silent truncation when writing float normalization results into integer arrays."""
        if not isinstance(arr, np.ndarray):
            return arr
        if not isinstance(result_arr, np.ndarray):
            return arr
        if np.issubdtype(result_arr.dtype, np.floating) and not np.issubdtype(arr.dtype, np.floating):
            return arr.astype(result_arr.dtype, copy=False)
        return arr

    for key in data_dim.keys():
        if use_keys is not None and key not in use_keys:
            continue
        if isinstance(data[key], dict):
            if key == 'tactile':
                iter_keys = data[key].keys()
            else:
                iter_keys = params[key].keys()
            for data_item_key in iter_keys:
                if key == 'tactile':
                    stats_item_key = _resolve_tactile_stats_key(data_item_key)
                else:
                    stats_item_key = data_item_key

                data_item = data[key][data_item_key]
                norm_dim_item = data_norm_dim[key][stats_item_key]
                assert data_item.shape[-1] == data_dim[key][stats_item_key], f"Feature[{key}/{data_item_key}] dimension mismatch: current {data_item.shape[-1]} != expected {data_dim[key][stats_item_key]}"
                data_item = data_item[..., norm_dim_item]
                if unnormalize:
                    if norm_type == 'quantile':
                        data_item = _unnormalize_quantile(data_item, params[key][stats_item_key])
                    elif norm_type == 'zscore':
                        data_item = _unnormalize_z_score(data_item, params[key][stats_item_key])
                    else:
                        raise ValueError(f"Unknown normalization type: {norm_type}")
                else:
                    if norm_type == 'quantile':
                        data_item = _normalize_quantile(data_item, params[key][stats_item_key], dtype=dtype)
                    elif norm_type == 'zscore':
                        data_item = _normalize_z_score(data_item, params[key][stats_item_key], dtype=dtype)
                    else:
                        raise ValueError(f"Unknown normalization type: {norm_type}")
                data[key][data_item_key] = _ensure_float_destination(data[key][data_item_key], data_item)
                data[key][data_item_key][..., norm_dim_item] = data_item
        else:
            data_item = data[key]
            norm_dim = data_norm_dim[key]
            assert data_item.shape[-1] == data_dim[key], f"Feature[{key}] dimension mismatch: current {data_item.shape[-1]} != expected {data_dim[key]}"
            data_item = data_item[..., norm_dim]
            if unnormalize:
                if norm_type == 'quantile':
                    data_item = _unnormalize_quantile(data_item, params[key])
                elif norm_type == 'zscore':
                    data_item = _unnormalize_z_score(data_item, params[key])
                else:
                    raise ValueError(f"Unknown normalization type: {norm_type}")
            else:
                if norm_type == 'quantile':
                    data_item = _normalize_quantile(data_item, params[key], dtype=dtype)
                elif norm_type == 'zscore':
                    data_item = _normalize_z_score(data_item, params[key], dtype=dtype)
                else:
                    raise ValueError(f"Unknown normalization type: {norm_type}")
            data[key] = _ensure_float_destination(data[key], data_item)
            data[key][..., norm_dim] = data_item

    return data
