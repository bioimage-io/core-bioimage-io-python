from typing import Union

from bioimageio.core.axis import AxisInfo
from bioimageio.core.sample import Sample
from bioimageio.spec.model import AnyModelDescr, v0_4, v0_5
from bioimageio.spec.utils import load_array

from ..tensor import Tensor, TensorId


def get_axes_infos(
    io_descr: Union[
        v0_4.InputTensorDescr,
        v0_4.OutputTensorDescr,
        v0_5.InputTensorDescr,
        v0_5.OutputTensorDescr,
    ]
):
    return [
        (
            AxisInfo.create("i")
            if isinstance(a, str) and a not in ("b", "i", "t", "c", "z", "y", "x")
            else AxisInfo.create(a)
        )
        for a in io_descr.axes
    ]


def get_test_inputs(model: AnyModelDescr) -> Sample:
    if isinstance(model, v0_4.ModelDescr):
        tensor_ids = [TensorId(t.name) for t in model.inputs]
    else:
        tensor_ids = [t.id for t in model.inputs]

    if isinstance(model, v0_4.ModelDescr):
        arrays = [load_array(tt) for tt in model.test_inputs]
    else:
        arrays = [load_array(d.test_tensor) for d in model.inputs]

    axes = [get_axes_infos(t) for t in model.inputs]
    return Sample(
        data={
            tid: Tensor.from_numpy(arr, dims=ax)
            for tid, arr, ax in zip(tensor_ids, arrays, axes)
        }
    )


def get_test_outputs(model: AnyModelDescr) -> Sample:
    if isinstance(model, v0_4.ModelDescr):
        tensor_ids = [TensorId(t.name) for t in model.outputs]
    else:
        tensor_ids = [t.id for t in model.outputs]

    if isinstance(model, v0_4.ModelDescr):
        arrays = [load_array(tt) for tt in model.test_outputs]
    else:
        arrays = [load_array(d.test_tensor) for d in model.outputs]

    axes = [get_axes_infos(t) for t in model.outputs]

    return Sample(
        data={
            tid: Tensor.from_numpy(arr, dims=ax)
            for tid, arr, ax in zip(tensor_ids, arrays, axes)
        }
    )
