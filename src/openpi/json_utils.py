import json
import numpy as np


def _json_scalar_repr(value):
    return json.dumps(value, ensure_ascii=False)


def _is_scalar(value):
    return not isinstance(value, (list, dict))


def _format_json_value(value, indent, level):
    if isinstance(value, dict):
        if not value:
            return '{}'
        lines = ['{']
        items = list(value.items())
        for idx, (key, item) in enumerate(items):
            comma = ',' if idx < len(items) - 1 else ''
            lines.append(
                ' ' * indent * (level + 1)
                + json.dumps(key, ensure_ascii=False)
                + ': '
                + _format_json_value(item, indent, level + 1)
                + comma
            )
        lines.append(' ' * indent * level + '}')
        return '\n'.join(lines)
    if isinstance(value, list):
        if not value:
            return '[]'
        if all(_is_scalar(item) for item in value):
            inner = ', '.join(_json_scalar_repr(item) for item in value)
            return f'[{inner}]'
        lines = ['[']
        for idx, item in enumerate(value):
            comma = ',' if idx < len(value) - 1 else ''
            lines.append(' ' * indent * (level + 1) + _format_json_value(item, indent, level + 1) + comma)
        lines.append(' ' * indent * level + ']')
        return '\n'.join(lines)
    return _json_scalar_repr(value)


def dump_json_with_inline_lists(data, file_path, indent=4):
    formatted = _format_json_value(data, indent, 0)
    with open(file_path, 'w') as f:
        f.write(formatted + '\n')


def _serialize_params_value(value):
    """Serialize normalization params to JSON-compatible format.
    
    All arrays are assumed to be float64, so we directly convert to list
    without storing dtype metadata.
    """
    if isinstance(value, np.ndarray):
        # Convert to float64 and then to list (all params are float64)
        return value.astype(np.float64).tolist()
    elif isinstance(value, dict):
        return {key: _serialize_params_value(val) for key, val in value.items()}
    elif isinstance(value, list):
        return [_serialize_params_value(item) for item in value]
    else:
        return value


def _is_numeric_list(value):
    """Check if value is a list that can be converted to np.ndarray.
    
    Returns True if value is a list containing only numbers (scalars),
    or a nested list where all elements are numeric lists.
    """
    if not isinstance(value, list):
        return False
    if not value:
        return False
    # Check if all elements are numbers (1D array)
    if all(isinstance(item, (int, float)) for item in value):
        return True
    # Check if all elements are lists of numbers (multi-dimensional array)
    if all(isinstance(item, list) for item in value):
        # Recursively check if all nested lists are numeric
        return all(_is_numeric_list(item) for item in value)
    return False


def _deserialize_params_value(value):
    """Deserialize normalization params from JSON-compatible format.
    
    All arrays are assumed to be float64. Lists of numbers (including nested)
    are converted to np.ndarray, while other structures are preserved.
    """
    if isinstance(value, dict):
        # Recursively process dict values
        result = {}
        for key, val in value.items():
            result[key] = _deserialize_params_value(val)
        return result
    elif isinstance(value, list):
        # Check if list can be converted to np.ndarray (handles nested lists)
        if _is_numeric_list(value):
            return np.array(value, dtype=np.float64)
        else:
            # Recursively process list items (for non-numeric nested structures)
            return [_deserialize_params_value(item) for item in value]
    else:
        return value