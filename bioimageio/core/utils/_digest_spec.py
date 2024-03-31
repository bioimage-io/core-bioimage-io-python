from typing import List, Sequence, Union

from bioimageio.core.axis import AxisInfo
from bioimageio.spec.model import AnyModelDescr, v0_4, v0_5
from bioimageio.spec.utils import load_array

from ..tensor import Tensor, TensorId


def get_sample_axes(
    io_descr: Sequence[
        Union[
            v0_4.InputTensorDescr,
            v0_4.OutputTensorDescr,
            v0_5.InputTensorDescr,
            v0_5.OutputTensorDescr,
        ]
    ]
):
    return [
        [
            (
                AxisInfo.create("i")
                if isinstance(a, str) and a not in ("b", "i", "t", "c", "z", "y", "x")
                else AxisInfo.create(a)
            )
            for a in d.axes
        ]
        for d in io_descr
    ]


def get_test_inputs(model: AnyModelDescr) -> List[Tensor]:
    axes = get_sample_axes(model.inputs)
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
        for arr, ax, t in zip(arrays, axes, tensor_ids)
    ]


def get_test_outputs(model: AnyModelDescr) -> List[Tensor]:
    axes = get_sample_axes(model.outputs)
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
        for arr, ax, t in zip(arrays, axes, tensor_ids)
    ]
