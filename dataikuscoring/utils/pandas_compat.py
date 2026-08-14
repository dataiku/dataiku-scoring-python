import numpy as np


def to_numpy(obj, *args, **kwargs):
    """
    Convert a pandas object to a NumPy array across supported pandas versions.
    Duplicated from dataiku.core.compat.pandas_compat.to_numpy to avoid dependency.
    """
    try:
        return obj.to_numpy(*args, **kwargs)
    except (AttributeError, TypeError):
        na_value = kwargs.pop("na_value", None)
        if na_value is not None:
            obj = obj.fillna(na_value)

        dtype = kwargs.pop("dtype", None)
        copy = kwargs.pop("copy", False)
        if args or dtype is not None or copy:
            return np.array(obj.values, *args, dtype=dtype, copy=copy)
        return obj.values
