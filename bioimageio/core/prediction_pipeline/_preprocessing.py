from typing import Dict, List

import numpy as np
import xarray as xr

from bioimageio.core.resource_io.nodes import Preprocessing
from bioimageio.spec.model.raw_nodes import PreprocessingName
from ._types import Transform


def scale_linear(tensor: xr.DataArray, *, gain, offset, axes) -> xr.DataArray:
    """scale the tensor with a fixed multiplicative and additive factor"""
    scale_axes = tuple(ax for ax in tensor.dims if (ax not in axes and ax != "b"))
    if scale_axes:
        gain = xr.DataArray(np.atleast_1d(gain), dims=scale_axes)
        offset = xr.DataArray(np.atleast_1d(offset), dims=scale_axes)
    return ensure_dtype(tensor * gain + offset, dtype="float32")


def zero_mean_unit_variance(
    tensor: xr.DataArray, *, axes=None, eps=1.0e-6, mode="per_sample", mean=None, std=None
) -> xr.DataArray:
    # valid modes according to spec: "per_sample", "per_dataset", "fixed"
    # TODO implement "per_dataset"
    if mode not in ("per_sample", "fixed"):
        raise NotImplementedError(f"Unsupported mode for zero_mean_unit_variance: {mode}")

    if mode == "fixed":
        assert mean is not None and std is not None
    elif axes:
        axes = tuple(axes)
        mean, std = tensor.mean(axes), tensor.std(axes)
    else:
        mean, std = tensor.mean(), tensor.std()

    ret = (tensor - mean) / (std + eps)

    return ensure_dtype(ret, dtype="float32")


def binarize(tensor: xr.DataArray, *, threshold) -> xr.DataArray:
    return ensure_dtype(tensor > threshold, dtype="float32")


def scale_range(
    tensor: xr.DataArray, *, mode="per_sample", axes=None, min_percentile=0.0, max_percentile=100.0
) -> xr.DataArray:
    # valid modes according to spec: "per_sample", "per_dataset"
    # TODO implement per_dataset
    if mode != "per_sample":
        raise NotImplementedError(f"Unsupported mode for scale_range: {mode}")

    if axes:
        axes = tuple(axes)
        v_lower = tensor.quantile(min_percentile / 100.0, dim=axes)
        v_upper = tensor.quantile(max_percentile / 100.0, dim=axes)
    else:
        v_lower = tensor.quantile(min_percentile / 100.0)
        v_upper = tensor.quantile(max_percentile / 100.0)

    return ensure_dtype((tensor - v_lower) / v_upper, dtype="float32")


def clip(tensor: xr.DataArray, *, min: float, max: float) -> xr.DataArray:
    return ensure_dtype(tensor.clip(min=min, max=max), dtype="float32")


def ensure_dtype(tensor: xr.DataArray, *, dtype):
    """
    Convert array to a given datatype
    """
    return tensor.astype(dtype)


def sigmoid(tensor: xr.DataArray, **kwargs):
    if kwargs:
        raise NotImplementedError(f"Passed kwargs for sigmoid {kwargs}")
    return 1.0 / (1.0 + xr.ufuncs.exp(-tensor))


KNOWN_PREPROCESSING: Dict[PreprocessingName, Transform] = {
    "scale_linear": scale_linear,
    "zero_mean_unit_variance": zero_mean_unit_variance,
    "binarize": binarize,
    "clip": clip,
    "scale_range": scale_range,
    "sigmoid": sigmoid
    # "__tiktorch_ensure_dtype": ensure_dtype,
}


def chain(*functions):
    def _chained_function(tensor):
        tensor = tensor
        for fn, kwargs in functions:
            kwargs = kwargs or {}
            tensor = fn(tensor, **kwargs)

        return tensor

    return _chained_function


def make_preprocessing(preprocessing_spec: List[Preprocessing]) -> Transform:
    """
    :param preprocessing: bioimage-io spec node
    """
    preprocessing_functions = []

    step: Preprocessing
    for step in preprocessing_spec:
        fn = KNOWN_PREPROCESSING.get(step.name)
        kwargs = step.kwargs.copy() if step.kwargs else {}

        if fn is None:
            raise NotImplementedError(f"Preprocessing {step.name}")

        preprocessing_functions.append((fn, kwargs))

    return chain(*preprocessing_functions)
