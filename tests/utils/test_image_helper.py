from typing import Sequence

import numpy as np
import pytest
import xarray as xr

from bioimageio.core.common import AxisId
from bioimageio.core.utils.image_helper import interprete_array


@pytest.mark.parametrize(
    "axes", [[AxisId(a) for a in axes] for axes in ["yx", "xy", "cyx", "yxc", "bczyx", "xyz", "xyzc", "bzyxc"]]
)
def test_transpose_tensor_2d(axes: Sequence[AxisId]):
    from bioimageio.core.utils.image_helper import transpose_tensor

    tensor = interprete_array(np.random.rand(256, 256), len(axes))
    transposed = transpose_tensor(tensor, axes)
    assert transposed.ndim == len(axes)


@pytest.mark.parametrize(
    "axes", [[AxisId(a) for a in axes] for axes in ["zyx", "cyx", "yxc", "bczyx", "xyz", "xyzc", "bzyxc"]]
)
def test_transpose_tensor_3d(axes: Sequence[AxisId]):
    from bioimageio.core.utils.image_helper import transpose_tensor

    tensor = interprete_array(np.random.rand(64, 64, 64), len(axes))
    transposed = transpose_tensor(tensor, axes)
    assert transposed.ndim == len(axes)


def test_crop_and_pad():
    tensor = xr.DataArray(np.random.rand(64))


# def test_transform_output_tensor():
#     from bioimageio.core.utils.image_helper import transform_output_tensor

#     tensor = np.random.rand(1, 3, 64, 64, 64)
#     tensor_axes = "bczyx"

#     out_ax_list = ["bczyx", "cyx", "xyc", "byxc", "zyx", "xyz"]
#     for out_axes in out_ax_list:
#         out = transform_output_tensor(tensor, tensor_axes, out_axes)
#         assert out.ndim == len(out_axes)
