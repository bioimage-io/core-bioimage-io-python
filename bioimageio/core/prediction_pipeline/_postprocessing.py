from typing import List

import xarray as xr
from bioimageio.core.resource_io.nodes import Postprocessing

from . import _preprocessing as ops
from ._types import Transform


# TODO how do we implement reference_tensor?

def scale_range(
    tensor: xr.DataArray,
    *,
    reference_tensor=None,
    mode="per_sample",
    axes=None,
    min_percentile=0.0,
    max_percentile=100.0,
) -> xr.DataArray:

    # TODO if reference tensor is passed, we need to use it to compute quantiles instead of 'tensor'
    if reference_tensor is None:
        tensor_ = tensor
    else:
        raise NotImplementedError

    # valid modes according to spec: "per_sample", "per_dataset"
    # TODO implement per_dataset
    if mode != "per_sample":
        raise NotImplementedError(f"Unsupported mode for scale_range: {mode}")

    if axes:
        axes = tuple(axes)
        v_lower = tensor_.quantile(min_percentile / 100.0, dim=axes)
        v_upper = tensor_.quantile(max_percentile / 100.0, dim=axes)
    else:
        v_lower = tensor_.quantile(min_percentile / 100.0)
        v_upper = tensor_.quantile(max_percentile / 100.0)

    return ops.ensure_dtype((tensor - v_lower) / v_upper, dtype="float32")


# TODO scale the tensor s.t. it matches the mean and variance of the reference tensor
def scale_mean_variance(tensor: xr.DataArray, *, reference_tensor, mode="per_sample"):
    raise NotImplementedError


# NOTE there is a subtle difference between pre-and-postprocessing:
# pre-processing always returns float32, because the post-processing output is consumed
# by the model. Post-processing, however, should return the dtype that is specified in the model spec
# TODO I think the easiest way to implement this is to add dtype is an option to 'make_postprocessing'
# and then apply 'ensure_dtype' to the result of the postprocessing chain
KNOWN_POSTPROCESSING = {
    "binarize": ops.binarize,
    "clip": ops.clip,
    "scale_linear": ops.scale_linear,
    "scale_range": ops.scale_range,
    "sigmoid": ops.sigmoid,
    "zero_mean_unit_variance": ops.zero_mean_unit_variance,
}


def make_postprocessing(spec: List[Postprocessing]) -> Transform:
    """
    :param preprocessing: bioimage-io spec node
    """
    functions = []

    step: Postprocessing
    for step in spec:
        fn = KNOWN_POSTPROCESSING.get(step.name)
        kwargs = step.kwargs.copy() if step.kwargs else {}

        if fn is None:
            raise NotImplementedError(f"Postprocessing {step.name}")

        functions.append((fn, kwargs))

    return ops.chain(*functions)
