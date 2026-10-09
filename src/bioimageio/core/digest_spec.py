from __future__ import annotations

import collections.abc
import importlib.util
import sys
from collections.abc import Iterable, Mapping, Sequence
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Any,
    Callable,
    Final,
    Literal,
    Union,
)
from zipfile import ZipFile, is_zipfile

import dask.array as da
import numpy as np
import xarray as xr
from loguru import logger
from numpy.typing import NDArray
from typing_extensions import TypeAlias, Unpack, assert_never

from bioimageio.core._utils.type_hints import cast_and_assert
from bioimageio.spec._internal.io import HashKwargs, PermissiveFileSource
from bioimageio.spec.common import FileDescr, FileSource
from bioimageio.spec.model import AnyModelDescr, v0_4, v0_5
from bioimageio.spec.model.v0_4 import CallableFromDepencency, CallableFromFile
from bioimageio.spec.model.v0_5 import (
    ArchitectureFromFileDescr,
    ArchitectureFromLibraryDescr,
)
from bioimageio.spec.utils import load_array

from .axis import CHANNEL_AXIS_ID, Axis, AxisId, AxisInfo, PerAxis
from .block import LinearAxisTransform
from .common import Halo, MemberId, PerMember, SampleId
from .io import load_tensor
from .sample import (
    Sample,
)
from .stat_measures import Stat
from .tensor import Tensor, get_unstacked_dim_order

TensorSource: TypeAlias = Union[
    Tensor, xr.DataArray, NDArray[Any], PermissiveFileSource
]


def import_callable(
    node: ArchitectureFromFileDescr
    | ArchitectureFromLibraryDescr
    | CallableFromDepencency
    | CallableFromFile
    | v0_5.CustomProcessingDescr,
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
    elif isinstance(node, (ArchitectureFromFileDescr, v0_5.CustomProcessingDescr)):
        c = _import_from_file_impl(node.source, str(node.callable), sha256=node.sha256)
    else:
        assert_never(node)

    if not callable(c):
        raise TypeError(f"{node} (imported: {c}) is not callable")

    return c


tmp_dirs_in_use: list[TemporaryDirectory[str]] = []
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
            td_kwargs: dict[str, Any] = (
                {"ignore_cleanup_errors": True} if sys.version_info >= (3, 10) else {}
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
            if module_name in sys.modules:
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
    io_descr: v0_4.InputTensorDescr
    | v0_4.OutputTensorDescr
    | v0_5.InputTensorDescr
    | v0_5.OutputTensorDescr,
) -> list[AxisInfo]:
    """get a unified, simplified axis representation from spec axes"""
    return [AxisInfo.create(a) for a in io_descr.axes]


def get_member_id(
    tensor_description: v0_4.InputTensorDescr
    | v0_4.OutputTensorDescr
    | v0_5.InputTensorDescr
    | v0_5.OutputTensorDescr,
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
    tensor_descriptions: Iterable[
        v0_4.InputTensorDescr
        | v0_4.OutputTensorDescr
        | v0_5.InputTensorDescr
        | v0_5.OutputTensorDescr
    ],
) -> list[MemberId]:
    """get normalized tensor IDs to be used as sample member IDs"""
    return [get_member_id(descr) for descr in tensor_descriptions]


def get_test_input_sample(
    model: AnyModelDescr,
    *,
    repetitions: Sequence[tuple[int, Axis | AxisInfo, Sequence[float]]] = (),
) -> Sample:
    """returns a model's test input sample

    Args:
        model: The model for which to create a test input sample.
        repetitions: A mapping of additional axes (keyed by their index and ID) to sequences of value factors. This adds new dimensions with scaled repetitions of the original test sample that will be stacked into a/the batch dimension.

    Returns:
        A test input sample for the given model.
    """
    if isinstance(model, v0_4.ModelDescr):
        info = {
            MemberId(d.name): (d, t) for d, t in zip(model.inputs, model.test_inputs)
        }
    else:
        info = {d.id: d for d in model.inputs}

    return _get_test_sample(info, repetitions=repetitions)


get_test_inputs = get_test_input_sample
"""DEPRECATED: use `get_test_input_sample` instead"""


def get_test_output_sample(
    model: AnyModelDescr,
    *,
    repetitions: Sequence[tuple[int, Axis | AxisInfo, Sequence[float]]] = (),
) -> Sample:
    """returns a model's test output sample"""
    if isinstance(model, v0_4.ModelDescr):
        info = {
            MemberId(d.name): (d, t) for d, t in zip(model.outputs, model.test_outputs)
        }
    else:
        info = {d.id: d for d in model.outputs}

    return _get_test_sample(info, repetitions=repetitions)


get_test_outputs = get_test_output_sample
"""DEPRECATED: use `get_test_output_sample` instead"""


def _get_test_sample(
    info: Mapping[MemberId, v0_5.InputTensorDescr | v0_5.OutputTensorDescr]
    | Mapping[MemberId, tuple[v0_4.InputTensorDescr, FileSource]]
    | Mapping[MemberId, tuple[v0_4.OutputTensorDescr, FileSource]],
    *,
    repetitions: Sequence[tuple[int, Axis | AxisInfo, Sequence[float]]] = (),
) -> Sample:
    arrays: dict[MemberId, NDArray[Any] | da.Array] = {}
    for m, src in info.items():
        if isinstance(src, tuple):
            arrays[m] = load_array(src[1])
        elif isinstance(src, (v0_5.InputTensorDescr, v0_5.OutputTensorDescr)):
            if src.test_tensor is None:
                raise ValueError(
                    f"Model input '{m}' has no test tensor defined, cannot create test sample."
                )
            arrays[m] = load_array(src.test_tensor)
        else:
            assert_never(src)

    axes: dict[MemberId, list[Axis | AxisInfo]] = {
        m: list(get_axes_infos(t[0] if isinstance(t, tuple) else t))
        for m, t in info.items()
    }
    tensors: dict[MemberId, Tensor] = {}
    for m in info:
        a = arrays[m]
        ax = axes[m]
        if repetitions:
            for rep_idx, rep_ax, factors in repetitions:
                if rep_idx < -1 or rep_idx >= len(ax):
                    raise ValueError(
                        f"Invalid repetition index {rep_idx} for tensor '{m}' with axes {ax}"
                    )
                # Repeat the array along the specified axis according to the factors
                repitition_arrays = {
                    factor: cast_and_assert(da.Array, da.from_array(a) * factor)
                    for factor in set(factors)
                }
                a = cast_and_assert(
                    da.Array,
                    da.stack(
                        [repitition_arrays[factor] for factor in factors], axis=rep_idx
                    ),
                )

                if len(factors) > 100:
                    # max 100 chunks along repition axis
                    chunk_size = max(1, len(factors) // 100)
                    chunk_shape = list(a.shape)
                    chunk_shape.insert(rep_idx, chunk_size)
                    a = cast_and_assert(da.Array, a.rechunk(chunk_shape))  # pyright: ignore[reportArgumentType]

                if isinstance(rep_ax, AxisInfo):
                    if len(factors) < rep_ax.size.min:
                        raise ValueError(
                            f"Invalid number of repetitions {len(factors)} for axis '{rep_ax.id}' with min size {rep_ax.size.min}"
                        )
                    if rep_ax.size.max is not None and len(factors) > rep_ax.size.max:
                        raise ValueError(
                            f"Invalid number of repetitions {len(factors)} for axis '{rep_ax.id}' with max size {rep_ax.size.max}"
                        )

                ax.insert(rep_idx, rep_ax)

        tensors[m] = Tensor.from_array(a, dims=ax)

    return Sample(members=tensors, stat={}, id="test-sample")


def get_unstacked_output_dim_orders_for_model(
    model: v0_4.ModelDescr | v0_5.ModelDescr,
    unstacked_input_order: PerMember[Sequence[AxisId]] | None,
    stacked_output_dims: PerMember[Sequence[AxisId]] | None,
) -> PerMember[Sequence[AxisId]]:
    """Returns the unstacked output dim order for each output tensor of a model,
    given optional unstacked input dim order and optional output stacked dims."""
    output_orders: dict[MemberId, Sequence[AxisId]] = {}
    for out in model.outputs:
        in_order = None
        if isinstance(out, v0_5.OutputTensorDescr):
            out_id = out.id
            stacked_out_order = [ax.id for ax in out.axes]
            if unstacked_input_order is not None:
                for ax in out.axes:
                    if isinstance(ax.size, v0_5.SizeReference):
                        in_order = unstacked_input_order.get(ax.size.tensor_id)
                        break
        elif isinstance(out, v0_4.OutputTensorDescr):
            out_id = MemberId(out.name)
            stacked_out_order = [AxisId(ax) for ax in out.axes]
            if unstacked_input_order is not None:
                for s in out.shape:
                    if isinstance(s, v0_4.ImplicitOutputShape):
                        in_order = unstacked_input_order.get(
                            MemberId(s.reference_tensor)
                        )
                        break
        else:
            assert_never(out)

        output_orders[out_id] = get_unstacked_dim_order(
            stacked_out_order,
            unstacked_reference_order=in_order,
            stacked_dims=stacked_output_dims.get(out_id)
            if stacked_output_dims is not None
            else None,
        )

    return output_orders


def get_input_halo(
    model: v0_5.ModelDescr, output_halo: PerMember[PerAxis[Halo]] | None = None
):
    """returns which halo input tensors need to be divided into blocks with, such that
    `output_halo` can be cropped from their outputs without introducing gaps."""
    input_halo: dict[MemberId, dict[AxisId, Halo]] = {}
    outputs = {t.id: t for t in model.outputs}
    all_tensors = {**{t.id: t for t in model.inputs}, **outputs}
    if output_halo is None:
        output_halo = {
            t.id: {
                a.id: Halo(a.halo, a.halo)
                for a in t.axes
                if isinstance(a, v0_5.WithHalo)
            }
            for t in model.outputs
        }

    for t, th in output_halo.items():
        axes = {a.id: a for a in outputs[t].axes}

        for a, ah in th.items():
            s = axes[a].size
            if not isinstance(s, v0_5.SizeReference):
                raise TypeError(
                    f"Unable to map output halo for {t}.{a} to an input axis"
                )

            axis = axes[a]
            ref_axis = {a.id: a for a in all_tensors[s.tensor_id].axes}[s.axis_id]

            input_halo_left = ah.left * axis.scale / ref_axis.scale
            input_halo_right = ah.right * axis.scale / ref_axis.scale
            assert input_halo_left == int(input_halo_left), f"{input_halo_left} not int"
            assert input_halo_right == int(input_halo_right), (
                f"{input_halo_right} not int"
            )

            input_halo.setdefault(s.tensor_id, {})[a] = Halo(
                int(input_halo_left), int(input_halo_right)
            )

    return input_halo


def get_block_transform(
    model: v0_5.ModelDescr,
) -> PerMember[PerAxis[LinearAxisTransform | int]]:
    """returns how a model's output tensor shapes relates to given input shapes"""
    ret: dict[MemberId, dict[AxisId, LinearAxisTransform | int]] = {}
    batch_axis_trf = None
    for ipt in model.inputs:
        for a in ipt.axes:
            if a.type == "batch":
                batch_axis_trf = LinearAxisTransform(
                    member=ipt.id, axis=a.id, scale=1, offset=0
                )
                break
        if batch_axis_trf is not None:
            break
    axis_scales = {
        t.id: {a.id: a.scale for a in t.axes}
        for t in chain(model.inputs, model.outputs)
    }
    # a common inaccuracy in bioimage.io model specs is to define a fixed output size equal to a correspoding fixed input size, while the output size should actually be a size reference to the fixed input size to encode the relationship between input and output sizes. The following dict is used to patch such cases, by replacing the fixed output size with a size reference to the corresponding fixed input size.
    patched_size_ref_for_fixed_output_size_matching_fixed_input_size: dict[
        MemberId, dict[AxisId, v0_5.SizeReference]
    ] = {}
    inconsistet_outputs: set[MemberId] = set()
    for out in model.outputs:
        for a in out.axes:
            if not isinstance(a.size, int):
                continue

            for ipt in model.inputs:
                for ia in ipt.axes:
                    if (
                        a.type not in ("space", "time")
                        or a.type != ia.type
                        or a.id != ia.id
                        or a.size != ia.size
                        or a.scale != ia.scale
                        or a.unit != ia.unit
                    ):
                        continue

                    previous_size_refs = list(
                        patched_size_ref_for_fixed_output_size_matching_fixed_input_size.get(
                            out.id, {}
                        ).values()
                    )
                    if previous_size_refs and ipt.id != previous_size_refs[0].tensor_id:
                        inconsistet_outputs.add(out.id)
                    else:
                        patched_size_ref_for_fixed_output_size_matching_fixed_input_size.setdefault(
                            out.id, {}
                        )[a.id] = v0_5.SizeReference(tensor_id=ipt.id, axis_id=ia.id)

    # remove inconsistent outputs from the patch dict
    patched_size_ref_for_fixed_output_size_matching_fixed_input_size = {
        m: {a: sr for a, sr in srs.items()}
        for m, srs in patched_size_ref_for_fixed_output_size_matching_fixed_input_size.items()
        if m not in inconsistet_outputs
    }
    if patched_size_ref_for_fixed_output_size_matching_fixed_input_size:
        logger.warning(
            "Patched output size references: {}\nThe fixed output sizes were likely intended to be defined as size references to the corresponding fixed input sizes.",
            {
                (k, kk): (sr.tensor_id, sr.axis_id)
                for k, v in patched_size_ref_for_fixed_output_size_matching_fixed_input_size.items()
                for kk, sr in v.items()
            },
        )

    for out in model.outputs:
        new_axes: dict[AxisId, LinearAxisTransform | int] = {}
        for a in out.axes:
            a_size = (
                patched_size_ref_for_fixed_output_size_matching_fixed_input_size.get(
                    out.id, {}
                ).get(a.id, a.size)
            )
            if a_size is None:
                assert a.type == "batch"
                if batch_axis_trf is None:
                    raise ValueError(
                        "no batch axis found in any input tensor, but output tensor"
                        + f" '{out.id}' has one."
                    )
                s = batch_axis_trf
            elif isinstance(a_size, int):
                if a.type == "batch":
                    if batch_axis_trf is None:
                        raise ValueError(
                            "no batch axis found in any input tensor, but output tensor"
                            + f" '{out.id}' has one."
                        )
                    s = batch_axis_trf
                else:
                    s = a_size
            elif isinstance(a_size, v0_5.DataDependentSize):
                assert a.type != "batch", (
                    "data dependent size cannot be used for batch axis"
                )
                s = -1
            elif isinstance(a_size, v0_5.SizeReference):
                assert a.type != "batch", "size reference cannot be used for batch axis"
                s = LinearAxisTransform(
                    axis=a_size.axis_id,
                    scale=axis_scales[a_size.tensor_id][a_size.axis_id] / a.scale,
                    offset=a_size.offset,
                    member=a_size.tensor_id,
                )
            else:
                assert_never(a_size)

            new_axes[a.id] = s

        # account for postprocessing that changes the nubmer of output channels by
        # overwriting described output shape by the intermediate output shape
        if CHANNEL_AXIS_ID in new_axes:
            for post in out.postprocessing:
                if post.id == "cellpose_flow_dynamics":
                    new_axes[CHANNEL_AXIS_ID] = 3
                    break
                elif post.id == "stardist_postprocessing":
                    new_axes[CHANNEL_AXIS_ID] = post.kwargs.n_rays + 1
                    break

        ret[out.id] = new_axes

    return ret


def get_tensor(
    src: TensorSource,
    descr: v0_4.InputTensorDescr
    | v0_5.InputTensorDescr
    | v0_4.OutputTensorDescr
    | v0_5.OutputTensorDescr
    | Sequence[AxisInfo],
):
    """helper to cast/load various tensor sources

    Args:
        src: the tensor source to load/cast
        descr: the tensor description or a sequence of axis infos

    """

    if isinstance(
        descr,
        (
            v0_4.InputTensorDescr,
            v0_5.InputTensorDescr,
            v0_4.OutputTensorDescr,
            v0_5.OutputTensorDescr,
        ),
    ):
        axes = get_axes_infos(descr)
    else:
        axes = descr

    if isinstance(src, Tensor):
        output = src
    elif isinstance(src, xr.DataArray):
        output = Tensor.from_xarray(src)

    elif isinstance(src, np.ndarray):
        output = Tensor.from_numpy(src, dims=axes)
    else:
        output = load_tensor(src, axes=axes)

    for a in axes:
        if a.id not in output.dims:
            raise ValueError(
                f"Missing axis '{a.id}' in tensor with dims {output.dims}. Expected axes: {axes}"
            )

    return output


def create_sample_for_model(
    model: AnyModelDescr,
    inputs: PerMember[TensorSource] | TensorSource,
    *,
    stat: Stat | None = None,
    sample_id: SampleId = None,
) -> Sample:
    """Create a sample from a single set of input(s) for a specific bioimage.io model

    Args:
        model: a bioimage.io model description
        inputs: the input(s) constituting a single sample, e.g. {'raw': ..., 'mask': ...}.
        stat: dictionary with sample and dataset statistics (may be updated in-place!)
        sample_id: sample ID
    """

    model_inputs = {get_member_id(d): d for d in model.inputs}
    if isinstance(inputs, collections.abc.Mapping):
        inputs = {MemberId(k): v for k, v in inputs.items()}
    elif len(model_inputs) == 1:
        inputs = {next(iter(model_inputs)): inputs}
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

    members: dict[MemberId, Tensor] = {}
    for m, ipt in model_inputs.items():
        if m in inputs:
            tensor = get_tensor(inputs[m], ipt)
            members[m] = tensor

    return Sample(members=members, stat={} if stat is None else stat, id=sample_id)


load_sample_for_model: Final = create_sample_for_model


def transpose_sample_for_model(
    sample: Sample,
    model: AnyModelDescr,
    *,
    extra_dims: Literal["raise", "squeeze", "stack", "squeeze_or_stack"] = "stack",
    missing_dims: Literal[
        "raise", "expand", "unstack", "unstack_or_expand"
    ] = "unstack",
) -> Sample:
    return Sample(
        members=transpose_inputs_for_model(
            sample.members, model, extra_dims=extra_dims, missing_dims=missing_dims
        ),
        stat=sample.stat,
        id=sample.id,
    )


def transpose_inputs_for_model(
    inputs: PerMember[Tensor],
    model: AnyModelDescr,
    *,
    extra_dims: Literal["raise", "squeeze", "stack", "squeeze_or_stack"] = "stack",
    missing_dims: Literal[
        "raise", "expand", "unstack", "unstack_or_expand"
    ] = "unstack",
) -> dict[MemberId, Tensor]:
    """Transpose sample members to the order expected as inputs by the model."""

    axes = {get_member_id(d): [a.id for a in get_axes_infos(d)] for d in model.inputs}
    for m in axes:
        if m not in inputs and not (
            isinstance(model, v0_5.ModelDescr)
            and next(d for d in model.inputs if get_member_id(d) == m).optional
        ):
            raise ValueError(
                f"Sample is missing non-optional member {m} required by model"
            )

    if any(unknown := [m not in inputs for m in axes]):
        raise ValueError(f"Axes specified for unknown members: {unknown}")

    transposed = {
        m: t.transpose(axes[m], extra_dims=extra_dims, missing_dims=missing_dims)
        for m, t in inputs.items()
        if m in axes
    }
    if (
        len(
            batch_lengths := {
                t.sizes[AxisId("batch")]
                for t in transposed.values()
                if AxisId("batch") in t.dims
            }
        )
        > 1
    ):
        raise ValueError(
            f"Transposed tensors have incompatible batch lengths: {batch_lengths}."
        )

    return transposed
