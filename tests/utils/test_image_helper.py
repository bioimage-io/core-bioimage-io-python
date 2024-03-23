import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_equal  # pyright: ignore[reportUnknownVariableType]

from bioimageio.core.common import AxisId
from bioimageio.core.utils.image_helper import (
    interprete_array,
    transpose_tensor,
)
from bioimageio.core.utils.tiling import crop_to, pad


@pytest.mark.parametrize(
    "axes",
    ["yx", "xy", "cyx", "yxc", "bczyx", "xyz", "xyzc", "bzyxc"],
)
def test_transpose_tensor_2d(axes: str):

    tensor = interprete_array(np.random.rand(256, 256), None)
    transposed = transpose_tensor(tensor, [AxisId(a) for a in axes])
    assert transposed.ndim == len(axes)


@pytest.mark.parametrize(
    "axes",
    ["zyx", "cyzx", "yzixc", "bczyx", "xyz", "xyzc", "bzyxtc"],
)
def test_transpose_tensor_3d(axes: str):
    tensor = interprete_array(np.random.rand(64, 64, 64), None)
    transposed = transpose_tensor(tensor, [AxisId(a) for a in axes])
    assert transposed.ndim == len(axes)


def test_crop_and_pad():
    tensor = xr.DataArray(np.random.rand(10, 20), dims=("x", "y"))
    sizes = {AxisId(str(k)): v for k, v in tensor.sizes.items()}
    padded = pad(tensor, {AxisId("x"): 7, AxisId("y"): (3, 3)})
    cropped = crop_to(padded, sizes)
    assert_equal(tensor, cropped)


# def test_transform_output_tensor():
#     from bioimageio.core.utils.image_helper import transform_output_tensor

#     tensor = np.random.rand(1, 3, 64, 64, 64)
#     tensor_axes = "bczyx"

#     out_ax_list = ["bczyx", "cyx", "xyc", "byxc", "zyx", "xyz"]
#     for out_axes in out_ax_list:
#         out = transform_output_tensor(tensor, tensor_axes, out_axes)
#         assert out.ndim == len(out_axes)
