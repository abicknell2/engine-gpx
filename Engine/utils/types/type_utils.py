from typing import Any, get_args, get_origin

import numpy as np


def get_type_structure(obj: Any) -> str:  # type: ignore
    """Recursively retrieves the type structure of an object, including NDArray, heterogeneous dictionaries, lists, tuples, and sets."""
    obj_type = type(obj)

    # Check for generics (like dict[str, list[str]] or NDArray[int])
    origin = get_origin(obj_type)
    args = get_args(obj_type)

    if origin:
        return f"{origin.__name__}[{', '.join(get_type_structure(arg) for arg in args)}]"

    # Handle NumPy NDArray
    if isinstance(obj, np.ndarray):
        dtype_name = obj.dtype.name  # Get dtype (e.g., int32, float64)
        shape_info = "x".join(map(str, obj.shape))  # Format shape info
        return f"NDArray[{dtype_name}, shape=({shape_info})]"

    # Handle dictionaries with mixed key-value types
    if isinstance(obj, dict):
        key_types = {get_type_structure(k) for k in obj.keys()}  # Get all key types
        value_types = {get_type_structure(v) for v in obj.values()}  # Get all value types

        key_type_str = " | ".join(sorted(key_types))
        value_type_str = " | ".join(sorted(value_types))

        return f"dict[{key_type_str}, {value_type_str}]"

    # Handle lists with mixed element types
    if isinstance(obj, list):
        if obj:  # Infer from elements
            element_types = {get_type_structure(item) for item in obj}
            return f"list[{', '.join(sorted(element_types))}]"
        else:
            return "list[?]"  # Unknown type for empty list

    # Handle tuples with mixed element types
    if isinstance(obj, tuple):
        if obj:  # Infer from elements
            tuple_types = ", ".join(get_type_structure(item) for item in obj)
            return f"tuple[{tuple_types}]"
        else:
            return "tuple[?]"  # Unknown type for empty tuple

    # Handle sets with mixed element types
    if isinstance(obj, set):
        if obj:
            set_types = {get_type_structure(item) for item in obj}
            return f"set[{', '.join(sorted(set_types))}]"
        else:
            return "set[?]"  # Unknown type for empty set

    # Base case: return the type name
    return obj_type.__name__
