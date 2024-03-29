from typing import List, Sequence, get_args

from bioimageio.core.axis import AxisLetter, AxisLike
from bioimageio.spec.model import AnyModelDescr, v0_4
from bioimageio.spec.utils import load_array

from ..tensor import Tensor, TensorId


def get_test_inputs(model: AnyModelDescr) -> List[Tensor]:
    axes = [d.axes for d in model.inputs]
    if isinstance(axes, str):
        core_axes: List[Sequence[AxisLike]] = [
            a if a in get_args(AxisLetter) else "i" for a in axes
        ]  # pyright: ignore[reportAssignmentType]
    else:
        core_axes = axes  # pyright: ignore[reportAssignmentType]

    if isinstance(model, v0_4.ModelDescr):
        arrays = [load_array(tt) for tt in model.test_inputs]
    else:
        arrays = [load_array(d.test_tensor) for d in model.inputs]

    if isinstance(model, v0_4.ModelDescr):
        tensor_ids = [TensorId(ipt.name) for ipt in model.inputs]
    else:
        tensor_ids = [ipt.id for ipt in model.inputs]

    return [
        Tensor.from_numpy(arr, dims=ax, id=t)
        for arr, ax, t in zip(arrays, core_axes, tensor_ids)
    ]


def get_test_outputs(model: AnyModelDescr) -> List[Tensor]:
    axes = [d.axes for d in model.outputs]
    if isinstance(axes, str):
        core_axes: List[Sequence[AxisLike]] = [
            a if a in get_args(AxisLetter) else "i" for a in axes
        ]  # pyright: ignore[reportAssignmentType]
    else:
        core_axes = axes  # pyright: ignore[reportAssignmentType]

    if isinstance(model, v0_4.ModelDescr):
        arrays = [load_array(tt) for tt in model.test_outputs]
    else:
        arrays = [load_array(d.test_tensor) for d in model.outputs]

    if isinstance(model, v0_4.ModelDescr):
        tensor_ids = [TensorId(ipt.name) for ipt in model.inputs]
    else:
        tensor_ids = [ipt.id for ipt in model.inputs]

    return [
        Tensor.from_numpy(arr, dims=ax, id=t)
        for arr, ax, t in zip(arrays, core_axes, tensor_ids)
    ]
