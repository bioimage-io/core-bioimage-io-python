import logging
from typing import List, Sequence, Tuple

import numpy as np
import xarray as xr
from imageio import imread

logger = logging.getLogger(__name__)


def binarize(tensor: xr.DataArray, threshold: float):
    return tensor > threshold


def select_outputs(*args) -> Tuple:
    """helper to select workflow outputs (to be used as a final step in a workflow)

    Returns:
        tuple: selected outputs (inputs to this op)

    """

    return args


def log(*args, log_level: int = logging.INFO, **kwargs) -> Tuple:
    """log any key word arguments (kwargs/options)

    Returns:
        tuple: positional inputs to this op

    """
    for k, v in kwargs.items():
        logger.log(
            log_level,
            f"{k}: %s",
            f"{v.shape} mean: {v.mean().item():.4f} std: {v.std().item():.4f}"
            if isinstance(v, (np.ndarray, xr.DataArray))
            else v,
        )

    return args


def load_tensors(sources: List[str], axes: Sequence[str]) -> List[xr.DataArray]:
    """load tensors"""
    assert len(sources) == len(axes)
    tensors = []
    for source, ax in zip(sources, axes):
        if source.split(".")[-1] == ".npy":
            data = np.load(str(source))
        else:
            data = imread(source)

        tensors.append(xr.DataArray(data, dims=ax))

    return tensors
