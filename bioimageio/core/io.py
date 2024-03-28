from pathlib import Path
from typing import Optional, Sequence, Union

import imageio

from bioimageio.core.axis import Axis, AxisLike
from bioimageio.spec.model import v0_5
from bioimageio.spec.model.v0_4 import InputTensorDescr as InputTensorDescr04
from bioimageio.spec.model.v0_4 import OutputTensorDescr as OutputTensorDescr04
from bioimageio.spec.utils import load_array

from .tensor import Tensor, TensorId


def load_tensor(
    path: Path, axes: Optional[Sequence[AxisLike]] = None, id: Optional[TensorId] = None
) -> Tensor:

    ext = path.suffix
    if ext == ".npy":
        array = load_array(path)
    else:
        is_volume = (
            True
            if axes is None
            else sum(Axis.create(a).type != "channel" for a in axes) > 2
        )
        array = imageio.volread(path) if is_volume else imageio.imread(path)

    return Tensor.from_numpy(array, axes, id=TensorId(path.stem) if id is None else id)
