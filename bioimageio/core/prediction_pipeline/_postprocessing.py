from typing import List

import xarray as xr
from bioimageio.spec.model.nodes import Postprocessing

from ._preprocessing import chain
from ._types import Transform

# REMOVE_BATCH_DIM = Postprocessing(name="__tiktorch_remove_batch_dim", kwargs=None)


def remove_batch_dim(tensor: xr.DataArray):
    batch_dim = tensor.sizes.get("b")
    assert batch_dim == 1, "batch dimension should be present and have size 1"
    return tensor.squeeze("b")


def sigmoid(tensor: xr.DataArray, **kwargs):
    if kwargs:
        raise NotImplementedError(f"Passed kwargs for sigmoid {kwargs}")
    return 1 / (1 + xr.ufuncs.exp(-tensor))


# KNOWN_POSTPROCESSING = {"__tiktorch_remove_batch_dim": remove_batch_dim, "sigmoid": sigmoid}
KNOWN_POSTPROCESSING = {"sigmoid": sigmoid}


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

    return chain(*functions)
