import logging

import numpy as np


def clean_kpi(kpi: dict):
    """Remove types such as np.ndarray, list, str, and bool. Only keep types convertible to float.

    Args:
        kpi: dict, to be cleaned
    Returns:
        cleaned_kpi: dict, only containing floats, with all other types removed.
        total: int, total number of kpis in cleaned_kpi, recursive.
    """
    total = 0
    new_kpi = {}
    for key, val in kpi.items():
        if isinstance(val, (np.ndarray, list, str, bool, tuple, type(None))):
            continue
        elif isinstance(val, dict):
            new_kpi[key], sub_total = clean_kpi(val)
            total += sub_total
        else:
            try:
                float_val = float(val)
            except TypeError as error:
                logging.warning("Filter KPI error: {%s}", error)
                continue
            new_kpi[key] = float_val
            total += 1
    return new_kpi, total
