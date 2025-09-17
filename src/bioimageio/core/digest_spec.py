from __future__ import annotations

import collections.abc
import importlib.util
import sys
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from zipfile import ZipFile, is_zipfile

import numpy as np
import xarray as xr
from loguru import logger
from numpy.typing import NDArray
from typing_extensions import Unpack, assert_never

from bioimageio.spec._internal.io import HashKwargs
from bioimageio.spec.common import FileDescr, FileSource, ZipPath
from bioimageio.spec.model import AnyModelDescr, v0_4, v0_5
from bioimageio.spec.model.v0_4 import CallableFromDepencency, CallableFromFile
from bioimageio.spec.model.v0_5 import (
    ArchitectureFromFileDescr,
    ArchitectureFromLibraryDescr,
    ParameterizedSize_N,
)
from bioimageio.spec.utils import load_array

from .axis import Axis, AxisId, AxisInfo, AxisLike, PerAxis
from .block_meta import split_multiple_shapes_into_blocks
from .common import Halo, MemberId, PerMember, SampleId, TotalNumberOfBlocks
from .io import load_tensor
from .sample import (
    LinearSampleAxisTransform,
    Sample,
    SampleBlockMeta,
    sample_block_meta_generator,
)
from .stat_measures import Stat
from .tensor import Tensor

TensorSource = Union[Tensor, xr.DataArray, NDArray[Any], Path]


def import_callable(
    node: Union[
        ArchitectureFromFileDescr,
        ArchitectureFromLibraryDescr,
        CallableFromDepencency,
        CallableFromFile,
    ],
    /,
    **kwargs: Unpack[HashKwargs],
) -> Callable[..., Any]:
    """import a callable (e.g. a torch.nn.Module) from a spec node describing it"""
    if isinstance(node, CallableFromDepencency):
        module = importlib.import_module(node.module_name)
        c = getattr(module, str(node.callable_name))
    elif isinstance(node, ArchitectureFromLibraryDescr):
        module = importlib.import_module(node.import_from)
        c = getattr(module, str(node.callable))
    elif isinstance(node, CallableFromFile):
        c = _import_from_file_impl(node.source_file, str(node.callable_name), **kwargs)
    elif isinstance(node, ArchitectureFromFileDescr):
        c = _import_from_file_impl(node.source, str(node.callable), sha256=node.sha256)
    else:
        assert_never(node)

    if not callable(c):
        raise ValueError(f"{node} (imported: {c}) is not callable")

    return c


tmp_dirs_in_use: List[TemporaryDirectory[str]] = []
"""keep global reference to temporary directories created during import to delay cleanup"""


def _import_from_file_impl(
    source: FileSource, callable_name: str, **kwargs: Unpack[HashKwargs]
):
    src_descr = FileDescr(source=source, **kwargs)
    # ensure sha is valid even if perform_io_checks=False
    # or the source has changed since last sha computation
    src_descr.validate_sha256(force_recompute=True)
    assert src_descr.sha256 is not None
    source_sha = src_descr.sha256

    reader = src_descr.get_reader()
    # make sure we have unique module name
    module_name = f"{reader.original_file_name.split('.')[0]}_{source_sha}"

    # make sure we have a unique and valid module name
    if not module_name.isidentifier():
        module_name = f"custom_module_{source_sha}"
        assert module_name.isidentifier(), module_name

    source_bytes = reader.read()

    module = sys.modules.get(module_name)
    if module is None:
        try:
            td_kwargs: Dict[str, Any] = (
                dict(ignore_cleanup_errors=True) if sys.version_info >= (3, 10) else {}
            )
            if sys.version_info >= (3, 12):
                td_kwargs["delete"] = False

            tmp_dir = TemporaryDirectory(**td_kwargs)
            # keep global ref to tmp_dir to delay cleanup until program exit
            # TODO: remove for py >= 3.12, when delete=False works
            tmp_dirs_in_use.append(tmp_dir)

            module_path = Path(tmp_dir.name) / module_name
            if reader.original_file_name.endswith(".zip") or is_zipfile(reader):
                module_path.mkdir()
                ZipFile(reader).extractall(path=module_path)
            else:
                module_path = module_path.with_suffix(".py")
                _ = module_path.write_bytes(source_bytes)

            importlib_spec = importlib.util.spec_from_file_location(
                module_name, str(module_path)
            )

            if importlib_spec is None:
                raise ImportError(f"Failed to import {source}")

            module = importlib.util.module_from_spec(importlib_spec)

            sys.modules[module_name] = module  # cache this module

            assert importlib_spec.loader is not None
            importlib_spec.loader.exec_module(module)

        except Exception as e:
            del sys.modules[module_name]
            raise ImportError(f"Failed to import {source}") from e

    try:
        callable_attr = getattr(module, callable_name)
    except AttributeError as e:
        raise AttributeError(
            f"Imported custom module from {source} has no `{callable_name}` attribute."
        ) from e
    except Exception as e:
        raise AttributeError(
            f"Failed to access `{callable_name}` attribute from custom module imported from {source} ."
        ) from e

    else:
        return callable_attr


def get_axes_infos(
    io_descr: Union[
        v0_4.InputTensorDescr,
        v0_4.OutputTensorDescr,
        v0_5.InputTensorDescr,
        v0_5.OutputTensorDescr,
    ],
) -> List[AxisInfo]:
    """get a unified, simplified axis representation from spec axes"""
    ret: List[AxisInfo] = []
    for a in io_descr.axes:
        if isinstance(a, v0_5.AxisBase):
            ret.append(AxisInfo.create(Axis(id=a.id, type=a.type)))
        else:
            assert a in ("b", "i", "t", "c", "z", "y", "x")
            ret.append(AxisInfo.create(a))

    return ret


def get_member_id(
    tensor_description: Union[
        v0_4.InputTensorDescr,
        v0_4.OutputTensorDescr,
        v0_5.InputTensorDescr,
        v0_5.OutputTensorDescr,
    ],
) -> MemberId:
    """get the normalized tensor ID, usable as a sample member ID"""

    if isinstance(tensor_description, (v0_4.InputTensorDescr, v0_4.OutputTensorDescr)):
        return MemberId(tensor_description.name)
    elif isinstance(
        tensor_description, (v0_5.InputTensorDescr, v0_5.OutputTensorDescr)
    ):
        return tensor_description.id
    else:
        assert_never(tensor_description)


def get_member_ids(
    tensor_descriptions: Sequence[
        Union[
            v0_4.InputTensorDescr,
            v0_4.OutputTensorDescr,
            v0_5.InputTensorDescr,
            v0_5.OutputTensorDescr,
        ]
    ],
) -> List[MemberId]:
    """get normalized tensor IDs to be used as sample member IDs"""
    return [get_member_id(descr) for descr in tensor_descriptions]


def get_test_input_sample(model: AnyModelDescr) -> Sample:
    return _get_test_sample(
        model.inputs,
        model.test_inputs if isinstance(model, v0_4.ModelDescr) else model.inputs,
    )


get_test_inputs = get_test_input_sample
"""DEPRECATED: use `get_test_input_sample` instead"""


def get_test_output_sample(model: AnyModelDescr) -> Sample:
    """returns a model's test output sample"""
    return _get_test_sample(
        model.outputs,
        model.test_outputs if isinstance(model, v0_4.ModelDescr) else model.outputs,
    )


get_test_outputs = get_test_output_sample
"""DEPRECATED: use `get_test_input_sample` instead"""


def _get_test_sample(
    tensor_descrs: Sequence[
        Union[
            v0_4.InputTensorDescr,
            v0_4.OutputTensorDescr,
            v0_5.InputTensorDescr,
            v0_5.OutputTensorDescr,
        ]
    ],
    test_sources: Sequence[Union[FileSource, v0_5.TensorDescr]],
) -> Sample:
    """returns a model's input/output test sample"""
    member_ids = get_member_ids(tensor_descrs)
    arrays: List[NDArray[Any]] = []
    for src in test_sources:
        if isinstance(src, (v0_5.InputTensorDescr, v0_5.OutputTensorDescr)):
            if src.test_tensor is None:
                raise ValueError(
                    f"Model input '{src.id}' has no test tensor defined, cannot create test sample."
                )
            arrays.append(load_array(src.test_tensor))
        else:
            arrays.append(load_array(src))

    axes = [get_axes_infos(t) for t in tensor_descrs]
    return Sample(
        members={
            m: Tensor.from_numpy(arr, dims=ax)
            for m, arr, ax in zip(member_ids, arrays, axes)
        },
        stat={},
        id="test-sample",
    )


class IO_SampleBlockMeta(NamedTuple):
    input: SampleBlockMeta
    output: SampleBlockMeta


def get_input_halo(model: v0_5.ModelDescr, output_halo: PerMember[PerAxis[Halo]]):
    """returns which halo input tensors need to be divided into blocks with, such that
    `output_halo` can be cropped from their outputs without introducing gaps."""
    input_halo: Dict[MemberId, Dict[AxisId, Halo]] = {}
    outputs = {t.id: t for t in model.outputs}
    all_tensors = {**{t.id: t for t in model.inputs}, **outputs}

    for t, th in output_halo.items():
        axes = {a.id: a for a in outputs[t].axes}

        for a, ah in th.items():
            s = axes[a].size
            if not isinstance(s, v0_5.SizeReference):
                raise ValueError(
                    f"Unable to map output halo for {t}.{a} to an input axis"
                )

            axis = axes[a]
            ref_axis = {a.id: a for a in all_tensors[s.tensor_id].axes}[s.axis_id]

            total_output_halo = sum(ah)
            total_input_halo = total_output_halo * axis.scale / ref_axis.scale
            assert (
                total_input_halo == int(total_input_halo) and total_input_halo % 2 == 0
            )
            input_halo.setdefault(s.tensor_id, {})[a] = Halo(
                int(total_input_halo // 2), int(total_input_halo // 2)
            )

    return input_halo


def get_block_transform(
    model: v0_5.ModelDescr,
) -> PerMember[PerAxis[Union[LinearSampleAxisTransform, int]]]:
    """returns how a model's output tensor shapes relates to its input shapes"""
    ret: Dict[MemberId, Dict[AxisId, Union[LinearSampleAxisTransform, int]]] = {}
    batch_axis_trf = None
    for ipt in model.inputs:
        for a in ipt.axes:
            if a.type == "batch":
                batch_axis_trf = LinearSampleAxisTransform(
                    axis=a.id, scale=1, offset=0, member=ipt.id
                )
                break
        if batch_axis_trf is not None:
            break
    axis_scales = {
        t.id: {a.id: a.scale for a in t.axes}
        for t in chain(model.inputs, model.outputs)
    }
    for out in model.outputs:
        new_axes: Dict[AxisId, Union[LinearSampleAxisTransform, int]] = {}
        for a in out.axes:
            if a.size is None:
                assert a.type == "batch"
                if batch_axis_trf is None:
                    raise ValueError(
                        "no batch axis found in any input tensor, but output tensor"
                        + f" '{out.id}' has one."
                    )
                s = batch_axis_trf
            elif isinstance(a.size, int):
                s = a.size
            elif isinstance(a.size, v0_5.DataDependentSize):
                s = -1
            elif isinstance(a.size, v0_5.SizeReference):
                s = LinearSampleAxisTransform(
                    axis=a.size.axis_id,
                    scale=axis_scales[a.size.tensor_id][a.size.axis_id] / a.scale,
                    offset=a.size.offset,
                    member=a.size.tensor_id,
                )
            else:
                assert_never(a.size)

            new_axes[a.id] = s

        ret[out.id] = new_axes

    return ret


def get_io_sample_block_metas(
    model: v0_5.ModelDescr,
    input_sample_shape: PerMember[PerAxis[int]],
    ns: Mapping[Tuple[MemberId, AxisId], ParameterizedSize_N],
    batch_size: int = 1,
) -> Tuple[TotalNumberOfBlocks, Iterable[IO_SampleBlockMeta]]:
    """returns an iterable yielding meta data for corresponding input and output samples"""
    if not isinstance(model, v0_5.ModelDescr):
        raise TypeError(f"get_block_meta() not implemented for {type(model)}")

    block_axis_sizes = model.get_axis_sizes(ns=ns, batch_size=batch_size)
    input_block_shape = {
        t: {aa: s for (tt, aa), s in block_axis_sizes.inputs.items() if tt == t}
        for t in {tt for tt, _ in block_axis_sizes.inputs}
    }
    output_halo = {
        t.id: {
            a.id: Halo(a.halo, a.halo) for a in t.axes if isinstance(a, v0_5.WithHalo)
        }
        for t in model.outputs
    }
    input_halo = get_input_halo(model, output_halo)

    n_input_blocks, input_blocks = split_multiple_shapes_into_blocks(
        input_sample_shape, input_block_shape, halo=input_halo
    )
    block_transform = get_block_transform(model)
    return n_input_blocks, (
        IO_SampleBlockMeta(ipt, ipt.get_transformed(block_transform))
        for ipt in sample_block_meta_generator(
            input_blocks, sample_shape=input_sample_shape, sample_id=None
        )
    )


def get_tensor(
    src: Union[ZipPath, TensorSource],
    ipt: Union[v0_4.InputTensorDescr, v0_5.InputTensorDescr],
):
    """helper to cast/load various tensor sources"""

    if isinstance(src, Tensor):
        return src
    elif isinstance(src, xr.DataArray):
        return Tensor.from_xarray(src)
    elif isinstance(src, np.ndarray):
        return Tensor.from_numpy(src, dims=get_axes_infos(ipt))
    else:
        return load_tensor(src, axes=get_axes_infos(ipt))


def create_sample_for_model(
    model: AnyModelDescr,
    *,
    stat: Optional[Stat] = None,
    sample_id: SampleId = None,
    inputs: Union[PerMember[TensorSource], TensorSource],
) -> Sample:
    """Create a sample from a single set of input(s) for a specific bioimage.io model

    Args:
        model: a bioimage.io model description
        stat: dictionary with sample and dataset statistics (may be updated in-place!)
        inputs: the input(s) constituting a single sample.
    """

    model_inputs = {get_member_id(d): d for d in model.inputs}
    if isinstance(inputs, collections.abc.Mapping):
        inputs = {MemberId(k): v for k, v in inputs.items()}
    elif len(model_inputs) == 1:
        inputs = {list(model_inputs)[0]: inputs}
    else:
        raise TypeError(
            f"Expected `inputs` to be a mapping with keys {tuple(model_inputs)}"
        )

    if unknown := {k for k in inputs if k not in model_inputs}:
        raise ValueError(f"Got unexpected inputs: {unknown}")

    if missing := {
        k
        for k, v in model_inputs.items()
        if k not in inputs and not (isinstance(v, v0_5.InputTensorDescr) and v.optional)
    }:
        raise ValueError(f"Missing non-optional model inputs: {missing}")

    return Sample(
        members={
            m: get_tensor(inputs[m], ipt)
            for m, ipt in model_inputs.items()
            if m in inputs
        },
        stat={} if stat is None else stat,
        id=sample_id,
    )


def load_sample_for_model(
    *,
    model: AnyModelDescr,
    paths: PerMember[Path],
    axes: Optional[PerMember[Sequence[AxisLike]]] = None,
    stat: Optional[Stat] = None,
    sample_id: Optional[SampleId] = None,
):
    """load a single sample from `paths` that can be processed by `model`"""

    if axes is None:
        axes = {}

    # make sure members are keyed by MemberId, not string
    paths = {MemberId(k): v for k, v in paths.items()}
    axes = {MemberId(k): v for k, v in axes.items()}

    model_inputs = {get_member_id(d): d for d in model.inputs}

    if unknown := {k for k in paths if k not in model_inputs}:
        raise ValueError(f"Got unexpected paths for {unknown}")

    if unknown := {k for k in axes if k not in model_inputs}:
        raise ValueError(f"Got unexpected axes hints for: {unknown}")

    members: Dict[MemberId, Tensor] = {}
    for m, p in paths.items():
        if m not in axes:
            axes[m] = get_axes_infos(model_inputs[m])
            logger.debug(
                "loading '{}' from {} with default input axes {} ",
                m,
                p,
                axes[m],
            )
        members[m] = load_tensor(p, axes[m])

    return Sample(
        members=members,
        stat={} if stat is None else stat,
        id=sample_id or tuple(sorted(paths.values())),
    )
