import dataclasses

import numpy as np
import pytest
import xarray as xr

from bioimageio.core.prediction_pipeline._processing import KNOWN_PROCESSING
from bioimageio.core.prediction_pipeline._utils import FIXED

try:
    from typing import get_args
except ImportError:
    from typing_extensions import get_args  # type: ignore


def test_assert_dtype():
    from bioimageio.core.prediction_pipeline._processing import AssertDtype

    proc = AssertDtype("test_tensor", dtype="uint8")
    tensor = xr.DataArray(np.zeros((1,), dtype="uint8"), dims=tuple("c"))
    out = proc(tensor)
    assert out is tensor

    tensor = tensor.astype("uint16")
    with pytest.raises(AssertionError):
        out = proc(tensor)
        assert out is tensor


@pytest.mark.parametrize(
    "proc",
    list(KNOWN_PROCESSING["pre"].values()) + list(KNOWN_PROCESSING["post"].values()),
)
def test_no_req_measures_for_mode_fixed(proc):
    # check if mode=fixed is valid for this proc
    for f in dataclasses.fields(proc):
        if f.name == "mode":
            break
    else:
        raise AttributeError("Processing is missing mode attribute")
    # mode is always annotated as literals (or literals of literals)
    valid_modes = get_args(f.type)
    for inner in get_args(f.type):
        valid_modes += get_args(inner)

    if FIXED not in valid_modes:
        return

    # we might be missing required kwargs. These have marshmallow.missing value as default
    # and raise a TypeError is in __post_init__()
    proc.__post_init__ = lambda self: None  # ignore missing kwargs

    proc_instance = proc(tensor_name="tensor_name", mode=FIXED)
    req_measures = proc_instance.get_required_measures()
    assert not req_measures
