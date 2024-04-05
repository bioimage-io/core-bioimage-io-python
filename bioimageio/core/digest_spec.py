from __future__ import annotations

import importlib.util
from functools import singledispatch
from typing import Any, Callable, Dict, Iterable, Mapping, NamedTuple, Tuple, Union

from typing_extensions import Unpack

from bioimageio.spec._internal.io_utils import HashKwargs, download
from bioimageio.spec.common import FileSource
from bioimageio.spec.model import AnyModelDescr, v0_4, v0_5
from bioimageio.spec.model.v0_4 import CallableFromDepencency, CallableFromFile
from bioimageio.spec.model.v0_5 import (
    ArchitectureFromFileDescr,
    ArchitectureFromLibraryDescr,
    ParameterizedSize,
    TensorId,
)
from bioimageio.spec.utils import load_array

from .axis import AxisId, AxisInfo, PerAxis
from .block_meta import split_multiple_shapes_into_blocks
from .common import Halo, MemberId, PerMember, TotalNumberOfBlocks
from .sample import Sample, SampleBlockMeta, sample_block_meta_generator
from .tensor import Tensor


@singledispatch
def import_callable(node: type, /) -> Callable[..., Any]:
    raise TypeError(type(node))


@import_callable.register
def _(node: CallableFromDepencency) -> Callable[..., Any]:
    module = importlib.import_module(node.module_name)
    c = getattr(module, str(node.callable_name))
    if not callable(c):
        raise ValueError(f"{node} (imported: {c}) is not callable")

    return c


@import_callable.register
def _(node: ArchitectureFromLibraryDescr) -> Callable[..., Any]:
    module = importlib.import_module(node.import_from)
    c = getattr(module, str(node.callable))
    if not callable(c):
        raise ValueError(f"{node} (imported: {c}) is not callable")

    return c


@import_callable.register
def _(node: CallableFromFile, **kwargs: Unpack[HashKwargs]):
    return _import_from_file_impl(node.source_file, str(node.callable_name), **kwargs)


@import_callable.register
def _(node: ArchitectureFromFileDescr, **kwargs: Unpack[HashKwargs]):
    return _import_from_file_impl(node.source, str(node.callable), sha256=node.sha256)


def _import_from_file_impl(
    source: FileSource, callable_name: str, **kwargs: Unpack[HashKwargs]
):
    local_file = download(source, **kwargs)
    module_name = local_file.path.stem
    importlib_spec = importlib.util.spec_from_file_location(
        module_name, local_file.path
    )
    if importlib_spec is None:
        raise ImportError(f"Failed to import {module_name} from {source}.")

    dep = importlib.util.module_from_spec(importlib_spec)
    importlib_spec.loader.exec_module(dep)  # type: ignore  # todo: possible to use "loader.load_module"?
    return getattr(dep, callable_name)


def get_axes_infos(
    io_descr: Union[
        v0_4.InputTensorDescr,
        v0_4.OutputTensorDescr,
        v0_5.InputTensorDescr,
        v0_5.OutputTensorDescr,
    ]
):
    """get a unified, simplified axis represenation from spec axes"""
    return [
        (
            AxisInfo.create("i")
            if isinstance(a, str) and a not in ("b", "i", "t", "c", "z", "y", "x")
            else AxisInfo.create(a)
        )
        for a in io_descr.axes
    ]


def get_test_inputs(model: AnyModelDescr) -> Sample:
    """returns a model's test input sample"""
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
        members={
            tid: Tensor.from_numpy(arr, dims=ax)
            for tid, arr, ax in zip(tensor_ids, arrays, axes)
        }
    )


def get_test_outputs(model: AnyModelDescr) -> Sample:
    """returns a model's test output sample"""
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
        members={
            tid: Tensor.from_numpy(arr, dims=ax)
            for tid, arr, ax in zip(tensor_ids, arrays, axes)
        }
    )


class IO_SampleBlockMeta(NamedTuple):
    input: SampleBlockMeta
    output: SampleBlockMeta

def get_input_halo(model: v0_5.ModelDescr, output_halo: PerMember[PerAxis[Halo]]):
    halo: Dict[MemberId, Dict[AxisId, Halo]] = {}
    outputs = {t.id: t for t in model.outputs}
    all_tensors = {**{t.id: t for t in model.inputs}, **outputs}

    for t, th in output_halo.items():
        axes = {a.id: a for a in outputs[t].axes}

        for a, ah in th.items():
            s = axes[a].size
            if not isinstance(s, v0_5.SizeReference):
                raise ValueError(f"Unable to map output halo for {t}.{a} to an input axis")


            axis = axes[a]
            ref_axis = {a.id: a for a in all_tensors[s.tensor_id].axes}[s.axis_id]

            total_output_halo = sum(ah)
            total_input_halo = total_output_halo * axis.scale / ref_axis.scale
            if total_input_halo != int(total_input_halo):
                raise ValueError()
            for lr in (ah.left, ah.right):
                input_halo =
    return halo

def get_block_meta(
    model: v0_5.ModelDescr,
    input_sample_shape: PerMember[PerAxis[int]],
    ns: Mapping[Tuple[TensorId, AxisId], ParameterizedSize.N],
) -> Tuple[TotalNumberOfBlocks, Iterable[IO_SampleBlockMeta]]:
    """returns an iterable yielding meta data for corresponding input and output samples"""
    if not isinstance(model, v0_5.ModelDescr):
        raise TypeError(f"get_block_meta() not implemented for {type(model)}")

    block_axis_sizes = model.get_axis_sizes(ns=ns, batch_size=1)
    input_block_shape = {
        t: {aa: s for (tt, aa), s in block_axis_sizes.inputs.items() if tt == t}
        for t in {tt for tt, _ in block_axis_sizes.inputs}
    }
    output_block_shape = {
        t: {
            aa: s
            for (tt, aa), s in block_axis_sizes.outputs.items()
            if tt == t and not isinstance(s, tuple)
        }
        for t in {tt for tt, _ in block_axis_sizes.outputs}
    }
    output_halo = {t.id: {a.id: Halo(a.halo, a.halo) for a in t.axes if isinstance(a, v0_5.WithHalo)} for t in model.outputs}
    input_halo = get_input_halo(model, output_halo)
    output_sample_shape_data_dep = model.get_output_tensor_sizes(input_sample_shape)
    output_sample_shape = {
        t: {
            a: -1 if isinstance(s, tuple) else s
            for a, s in output_sample_shape_data_dep[t].items()
        }
        for t in output_sample_shape_data_dep
    }
    n_input_blocks, input_blocks = split_multiple_shapes_into_blocks(
        input_sample_shape, input_block_shape, halo=input_halo
    )
    n_output_blocks, output_blocks = split_multiple_shapes_into_blocks(
        output_sample_shape, output_block_shape, halo=output_halo
    )
    assert n_input_blocks == n_output_blocks
    return n_input_blocks, (
        IO_SampleBlockMeta(ipt, out)
        for ipt, out in zip(
            sample_block_meta_generator(input_blocks, origin=input_sample_shape),
            sample_block_meta_generator(output_blocks, origin=output_sample_shape),
        )
    )
