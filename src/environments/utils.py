"""Utility functions for environments."""

import torch
import numpy as np


def aggregate_dct(dcts):
    """
    Aggregate a list of dictionaries with the same keys.

    Args:
        dcts: List of dictionaries

    Returns:
        Dictionary with stacked values
    """
    full_dct = {}
    for dct in dcts:
        for key, value in dct.items():
            if key not in full_dct:
                full_dct[key] = []
            full_dct[key].append(value)

    for key, value in full_dct.items():
        if isinstance(value[0], torch.Tensor):
            full_dct[key] = torch.stack(value)
        else:
            full_dct[key] = np.stack(value)

    return full_dct
