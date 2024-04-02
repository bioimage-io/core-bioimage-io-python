from typing import Iterable, Union

from bioimageio.core.tile import AbstractTile
from bioimageio.spec.model import AnyModelDescr, v0_4, v0_5
from bioimageio.spec.utils import load_array

from ..axis import AxisInfo
from ..sample import UntiledSample
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


def get_test_inputs(model: AnyModelDescr) -> UntiledSample:
    if isinstance(model, v0_4.ModelDescr):
        tensor_ids = [TensorId(t.name) for t in model.inputs]
    else:
        tensor_ids = [t.id for t in model.inputs]

    if isinstance(model, v0_4.ModelDescr):
        arrays = [load_array(tt) for tt in model.test_inputs]
    else:
        arrays = [load_array(d.test_tensor) for d in model.inputs]

    axes = [get_axes_infos(t) for t in model.inputs]
    return UntiledSample(
        data={
            tid: Tensor.from_numpy(arr, dims=ax)
            for tid, arr, ax in zip(tensor_ids, arrays, axes)
        }
    )


def get_test_outputs(model: AnyModelDescr) -> UntiledSample:
    if isinstance(model, v0_4.ModelDescr):
        tensor_ids = [TensorId(t.name) for t in model.outputs]
    else:
        tensor_ids = [t.id for t in model.outputs]

    if isinstance(model, v0_4.ModelDescr):
        arrays = [load_array(tt) for tt in model.test_outputs]
    else:
        arrays = [load_array(d.test_tensor) for d in model.outputs]

    axes = [get_axes_infos(t) for t in model.outputs]

    return UntiledSample(
        data={
            tid: Tensor.from_numpy(arr, dims=ax)
            for tid, arr, ax in zip(tensor_ids, arrays, axes)
        }
    )


def get_abstract_output_tiles(
    input_tiles: Iterable[AbstractTile], model: v0_5.ModelDescr
):
    if not isinstance(model, v0_5.ModelDescr):
        raise TypeError(f"get_abstract_output_tile() not implemented for {type(model)}")

    sample_sizes = model.get_output_tensor_sizes(input_tile.sample_sizes)
    outer_sizes = model.get_output_tensor_sizes(input_tile.outer_sizes)
    UntiledSample()
    halo = {
        t.id: {a.id: a.halo for a in t.axes if isinstance(a, v0_5.WithHalo)}
        for t in model.outputs
        if t.id in outer_sizes
    }
    inner_sizes = {
        t: {
            a: outer_sizes[t][a] - 2 * halo.get(t, {}).get(a, 0) for a in outer_sizes[t]
        }
        for t in outer_sizes
    }

    return AbstractTile(
        halo=halo,
        tile_number=input_tile.tile_number,
        tiles_in_sample=input_tile.tiles_in_sample,
        stat={},
    )
