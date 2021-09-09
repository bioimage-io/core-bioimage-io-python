from typing import List

import xarray as xr
from bioimageio.core.resource_io.nodes import Postprocessing

from ._preprocessing import binarize, chain
from ._types import Transform


def sigmoid(tensor: xr.DataArray, **kwargs):
    if kwargs:
        raise NotImplementedError(f"Passed kwargs for sigmoid {kwargs}")
    return 1 / (1 + xr.ufuncs.exp(-tensor))


KNOWN_POSTPROCESSING = {"binarize": binarize, "sigmoid": sigmoid}


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
