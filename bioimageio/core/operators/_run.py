import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from os import PathLike
from typing import Any, Dict, Generator, IO, List, Optional, Sequence, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr
from marshmallow import missing

from bioimageio.core import load_resource_description
from bioimageio.core.prediction_pipeline import create_prediction_pipeline
from bioimageio.core.prediction_pipeline._combined_processing import CombinedProcessing
from bioimageio.core.prediction_pipeline._model_adapters import ModelAdapter, create_model_adapter
from bioimageio.core.resource_io import nodes
from bioimageio.core.resource_io.utils import resolve_raw_node
from bioimageio.spec import load_raw_resource_description
from bioimageio.spec.model import raw_nodes
from bioimageio.spec.shared.raw_nodes import ResourceDescription as RawResourceDescription

try:
    import torch.multiprocessing as multiprocessing
except ImportError:
    import multiprocessing

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


BoundaryMode = Literal["reflect"]


def transpose_seq(seq, seq_axes, desired_axes, default):
    return np.array([default if ia not in seq_axes else seq[seq_axes.index(ia)] for ia in desired_axes])


def get_chunk(
    chunk, ipt: raw_nodes.InputTensor, outputs: Sequence[raw_nodes.OutputTensor], tensor
) -> Tuple[Dict[str, int], Dict[int, int], Dict[str, Tuple[int, int]]]:
    """correct chunk to account for offset and halo

    Returns:
        corrected chunk: to tile the input array with
        overlap: overlap of corrected chunks (yields original chunks)
    """
    ipt_shape = np.array([chunk[a] for a in ipt.axes], dtype=int)
    referencing_outputs = [
        ot
        for ot in outputs
        if isinstance(ot.shape, raw_nodes.ImplicitOutputShape) and ot.shape.reference_tensor == ipt.name
    ]
    if not referencing_outputs:
        return (
            chunk,
            defaultdict(lambda: 0),
            defaultdict(lambda: (0, 0)),
        )

    if len(referencing_outputs) > 1:
        raise NotImplementedError("more than one output references an input")

    sohs = [
        (
            transpose_seq(ot.shape.scale, ot.axes, ipt.axes, 1.0),
            transpose_seq(ot.shape.offset, ot.axes, ipt.axes, 0.0),
            transpose_seq(ot.halo, ot.axes, ipt.axes, 0.0),
        )
        for ot in referencing_outputs
    ]
    scale, offset, halo = sohs[0]
    if any((s != scale).any() or (off != offset).any() or (h != halo).any() for s, off, h in sohs[1:]):
        # todo: ignore any new dimensions denoted by scale entry of None
        raise ValueError(
            f"Incompatible output specs referencing same input tensor with different scale/offset/halo: {[out.name for out in referencing_outputs]}."
        )

    if any(off > 0 for a, off in zip(offset, ipt.axes) if a in ("x", "y", "z", "t", "time")):
        raise NotImplementedError(
            "offset>0; space/time output is larger than input. todo: cut offset on tiles, but leave at image edge."
        )

    assert all(h >= 0 for h in halo)
    overlap = np.maximum((halo - offset) / scale, 0)  # no negative overlap
    overlap = np.ceil(overlap).astype(int)
    corrected_chunk = ipt_shape - 2 * overlap
    t_shape = np.array(tensor.shape, dtype=int)
    assert len(t_shape) == len(ipt_shape)
    padding_total = (corrected_chunk - (t_shape % corrected_chunk)) % corrected_chunk
    padding = [(0, p) for p in padding_total]

    return (
        dict(zip(ipt.axes, corrected_chunk)),
        dict(enumerate(overlap)),  # xr.DataArray.overlap not yet available: key by index for da.overlap
        dict(zip(ipt.axes, padding)),
    )


def tuple_roi_to_slices(tuple_roi: Sequence[Tuple[int, int]]) -> Tuple[slice, ...]:
    return tuple(np.s_[r0:-r1] if r1 else np.s_[r0:] for r0, r1 in tuple_roi)


def get_default_input_chunk(ipt: raw_nodes.InputTensor) -> List[int]:
    if isinstance(ipt.shape, list):
        shape = ipt.shape
    elif isinstance(ipt.shape, raw_nodes.ParametrizedInputShape):
        is3d = len([a for a in ipt.axes if a not in "bc"]) > 2
        min_len = 64 if is3d else 256
        shape = []
        for ax, min_ax, step_ax in zip(ipt.axes, ipt.shape.min_shape, ipt.shape.step):
            if ax in "zyx" and step_ax > 0:
                len_ax = min_ax
                while len_ax < min_len:
                    len_ax += step_ax
                shape.append(len_ax)
            else:
                shape.append(min_ax)
    else:
        raise TypeError(type(ipt.shape))

    assert len(ipt.axes) == len(shape)
    return shape


def get_asymmetric_halolike(value: float) -> Tuple[int, int]:
    assert value >= 0
    if value % 1:
        assert value % 0.5 == 0
        return math.floor(value), math.ceil(value)
    else:
        return int(value), int(value)


def get_output_rois(
    out: raw_nodes.OutputTensor,
    input_overlaps: Dict[str, Dict[int, int]],
    input_paddings: Dict[str, Dict[str, Tuple[int, int]]],
    ipt_by_name: Dict[str, raw_nodes.InputTensor],
) -> Tuple[Sequence[Tuple[int, int]], Sequence[Tuple[int, int]]]:
    if isinstance(out.shape, raw_nodes.ImplicitOutputShape):
        scale = np.array([1.0 if s is None else s for s in out.shape.scale])
        offset: Sequence[float] = out.shape.offset
        ref_ipt = ipt_by_name[out.shape.reference_tensor]
        eff_halo_float: List[float] = [
            input_overlaps[out.shape.reference_tensor].get(ref_ipt.axes.index(a), 0) * s + off
            for a, s, off in zip(out.axes, scale, offset)
        ]
        ref_input_padding_dict = input_paddings[out.shape.reference_tensor]
    else:
        scale = np.ones(len(out.shape))
        offset = np.zeros(len(out.shape))
        eff_halo_float = [0.0] * len(out.shape)
        ref_input_padding_dict = {}

    # effective halo to be trimmed from output. (only for space and time dims)
    output_chunk_roi: List[Tuple[int, int]] = []
    for i, a in enumerate(out.axes):
        if a in ("b", "batch"):
            errors_in = (["halo"] if eff_halo_float[i] else []) + (["offset"] if offset[i] else [])
            if errors_in:
                raise ValueError(f"invalid {' and '.join(errors_in)} for batch dimension of output {out.name}")
        elif a in ("x", "y", "z", "t", "time"):
            pass
        elif a in ("i", "index", "c", "channel"):
            # ignore offset. As we cannot tile across these dimensions, offsets should be returned, not trimmed.
            eff_halo_float[i] -= offset[i]
            if eff_halo_float[i]:
                warnings.warn(f"Trimming off halo for axis {a} of output {out.name}.")

        else:
            raise NotImplementedError(a)

        output_chunk_roi.append(get_asymmetric_halolike(eff_halo_float[i]))

    # undo input padding for the resulting final output tensor
    # also trim any negative offset, which we padded for each chunk
    output_roi = []
    for a, s, off in zip(out.axes, scale, offset):
        p0, p1 = ref_input_padding_dict.get(a, (0, 0))
        off0, off1 = get_asymmetric_halolike(-min(off, 0))
        output_roi.append((math.ceil(p0 * s + off0), math.ceil(p1 * s + off1)))

    return output_chunk_roi, output_roi


def forward(*tensors, model_adapter: ModelAdapter, output_tile_roi: Tuple[slice, ...]):
    """helper to cast dask array chunks to xr.DataArray and apply a roi to the output"""
    assert len(model_adapter.bioimageio_model.inputs) == len(tensors), (
        len(model_adapter.bioimageio_model.inputs),
        len(tensors),
    )
    tensors = [xr.DataArray(t, dims=tuple(ipt.axes)) for ipt, t, in zip(model_adapter.bioimageio_model.inputs, tensors)]
    output = model_adapter.forward(*tensors)[0]  # todo: allow more than 1 output
    return output[output_tile_roi]


def get_corrected_chunks(chunks: Dict[int, Sequence[int]], shape: Sequence[int], roi: Sequence[Tuple[int, int]]):
    """adapt `chunks` chunking `shape` for `shape[roi]`"""
    corrected_chunks = []
    rechunk = False
    for i, (s, roi) in enumerate(zip(shape, roi)):
        c = chunks[i]
        assert s == sum(c), (s, c)
        if sum(roi):
            c = list(c)
            r0 = roi[0]
            while r0 >= c[0]:
                r0 -= c[0]
                c = c[1:]
                if not c:
                    raise ValueError(f"Trimming too much from output {shape} with roi {roi}")

            c[0] -= r0

            r1 = roi[1]
            while r1 >= c[-1]:
                r1 -= c[-1]
                c = c[:-1]
                if not c:
                    raise ValueError(f"Trimming too much from output {shape} with roi {roi}")

            c[-1] -= r1

        corrected_chunks.append(c)
    return corrected_chunks, rechunk


def run_model_inference(
    rdf_source: Union[dict, PathLike, IO, str, bytes, raw_nodes.URI, RawResourceDescription],
    *tensors: xr.DataArray,
    enable_preprocessing: bool = True,
    enable_postprocessing: bool = True,
    devices: Sequence[str] = ("cpu",),
    tiles: Union[None, Literal["auto"], Sequence[Dict[str, int]]] = "auto",
    boundary_mode: Union[
        BoundaryMode,
        Sequence[BoundaryMode],
    ] = "reflect",
) -> List[xr.DataArray]:
    """run model inference

    Returns:
        list: model outputs
    """
    if tiles is None:
        return run_model_inference_without_tiling(
            rdf_source,
            *tensors,
            enable_preprocessing=enable_preprocessing,
            enable_postprocessing=enable_postprocessing,
            devices=devices,
        )
    model: raw_nodes.Model = load_raw_resource_description(rdf_source, update_to_format="latest")  # noqa
    if len(model.outputs) > 1:
        raise NotImplementedError("More than one model output not yet implemented")

    assert isinstance(model, raw_nodes.Model)
    # always remove pre-/postprocessing, but save it if enabled
    # todo: improve pre- and postprocessing!

    if enable_preprocessing:
        preprocessing = CombinedProcessing.from_tensor_specs(
            [resolve_raw_node(ipt, nodes, root_path=model.root_path) for ipt in model.inputs]
        )
        sample = {ipt.name: t for ipt, t in zip(model.inputs, tensors)}
        preprocessing.apply(sample, {})
        tensors = [sample[ipt.name] for ipt in model.inputs]

    if enable_postprocessing:
        postprocessing = CombinedProcessing.from_tensor_specs(
            [resolve_raw_node(out, nodes, root_path=model.root_path) for out in model.outputs]
        )
    else:
        postprocessing = None

    # transpose tensors to match ipt spec
    assert len(tensors) == len(model.inputs)
    tensors = [t.transpose(*s.axes) for t, s in zip(tensors, model.inputs)]
    if isinstance(boundary_mode, str):
        boundary_mode = [boundary_mode] * len(tensors)

    if tiles == "auto":
        tiles = [get_default_input_chunk(ipt) for ipt in model.inputs]

    # calculate chunking of the input tensors from tiles taking halo and offset into account
    chunks, overlap_depths, paddings = zip(
        *(get_chunk(c, ipt, model.outputs, t) for c, ipt, t in zip(tiles, model.inputs, tensors))
    )
    chunks_by_name = {ipt.name: c for ipt, c in zip(model.inputs, chunks)}
    padded_input_tensor_shapes = {
        ipt.name: [ts + sum(p[a]) for ts, a in zip(t.shape, ipt.axes)]
        for ipt, t, p in zip(model.inputs, tensors, paddings)
    }

    # note: da.overlap.overlap or da.overlap.map_overlap equivalents are not yet available in xarray
    tensors = [
        da.overlap.overlap(t.pad(p, mode=bm).chunk(c).data, depth=d, boundary=bm)
        for t, c, d, p, bm in zip(tensors, chunks, overlap_depths, paddings, boundary_mode)
    ]

    output_tile_roi, output_roi = get_output_rois(
        model.outputs[0],
        input_overlaps={ipt.name: d for ipt, d in zip(model.inputs, overlap_depths)},
        input_paddings={ipt.name: p for ipt, p in zip(model.inputs, paddings)},
        ipt_by_name={ipt.name: ipt for ipt in model.inputs},
    )

    n_batches = tensors[0].npartitions
    assert all(t.npartitions == n_batches for t in tensors[1:]), [t.npartitions for t in tensors]

    model_adapter = create_model_adapter(bioimageio_model=model, devices=devices)

    # todo: generalize to multiple outputs
    out = model.outputs[0]
    if isinstance(out.shape, raw_nodes.ImplicitOutputShape):
        ipt_shape = padded_input_tensor_shapes[out.shape.reference_tensor]
        ipt_by_name = {ipt.name: ipt for ipt in model.inputs}
        ipt_axes = ipt_by_name[out.shape.reference_tensor].axes
        ipt_shape = transpose_seq(ipt_shape, ipt_axes, out.axes, 0)
        out_scale = [0.0 if s is None else s for s in out.shape.scale]
        out_offset = np.array(out.shape.offset)
        out_shape_float = ipt_shape * out_scale + 2 * out_offset
        assert (out_shape_float == out_shape_float.astype(int)).all(), out_shape_float
        out_shape: Sequence[int] = out_shape_float.astype(int)
    else:
        out_shape = out.shape
        out_scale = [1.0] * len(out_shape)
        ipt_axes = []

    # set up da.blockwise to orchestrate tiled forward
    out_ind = []
    new_axes = {}
    adjust_chunks = {}
    for a, s, sc in zip(out.axes, out_shape, out_scale):
        if a in ("b", "batch"):
            out_ind.append(a)
        elif a in ipt_axes:
            axis_name = f"{out.shape.reference_tensor}_{a}"
            out_ind.append(axis_name)
            adjust_chunks[axis_name] = lambda _, aa=a, scc=sc: chunks_by_name[out.shape.reference_tensor][aa] * scc
        else:
            out_ind.append(f"{out.name}_{a}")
            new_axes[f"{out.name}_{a}"] = s

    inputs_sequence = []
    for t, ipt in zip(tensors, model.inputs):
        inputs_sequence.append(t)
        inputs_sequence.append(tuple("b" if a == "b" else f"{ipt.name}_{a}" for a in ipt.axes))

    result = da.blockwise(
        forward,
        tuple(out_ind),
        *inputs_sequence,
        new_axes=new_axes,
        dtype=np.dtype(out.data_type),
        meta=np.empty((), dtype=np.dtype(out.data_type)),
        name=(model.config or {}).get("bioimageio", {}).get("nickname") or f"model_{model.id}",
        adjust_chunks=adjust_chunks,
        **dict(model_adapter=model_adapter, output_tile_roi=tuple_roi_to_slices(output_tile_roi)),
    )

    corrected_chunks, rechunk = get_corrected_chunks(result.chunks, result.shape, output_roi)
    res = result[tuple_roi_to_slices(output_roi)]
    if rechunk:
        res = res.rechunk(corrected_chunks)

    outputs = [xr.DataArray(res, dims=tuple(out.axes))]
    if enable_postprocessing:
        assert postprocessing is not None
        sample = {out.name: t for out, t in zip(model.outputs, outputs)}
        postprocessing.apply(sample, {})
        outputs = [sample[out.name] for out in model.outputs]

    return outputs


def run_model_inference_without_tiling(
    rdf_source: Union[dict, PathLike, IO, str, bytes, raw_nodes.URI, RawResourceDescription],
    *tensors: xr.DataArray,
    enable_preprocessing: bool = True,
    enable_postprocessing: bool = True,
    devices: Optional[Sequence[str]] = ("cpu",),
) -> List[xr.DataArray]:
    """run model inference

    Returns:
        list: model outputs
    """
    model = load_raw_resource_description(rdf_source, update_to_format="latest")
    assert isinstance(model, raw_nodes.Model)
    # remove pre-/postprocessing if not enabled
    if not enable_preprocessing:
        for ipt in model.inputs:
            if ipt.preprocessing:
                ipt.preprocessing = missing
    if not enable_postprocessing:
        for out in model.outputs:
            if out.postprocessing:
                out.postprocessing = missing

    with create_prediction_pipeline(model, devices=devices) as pred_pipeline:
        return pred_pipeline.forward(*tensors)


def run_workflow(
    rdf_source: Union[dict, PathLike, IO, str, bytes, raw_nodes.URI, RawResourceDescription],
    inputs: Sequence = tuple(),
    options: Dict[str, Any] = None,
) -> tuple:
    outputs = tuple()
    for state in _iterate_workflow_steps_impl(rdf_source, test_steps=False, inputs=inputs, options=options):
        outputs = state.outputs

    return outputs


def run_workflow_test(
    rdf_source: Union[dict, PathLike, IO, str, bytes, raw_nodes.URI, RawResourceDescription],
) -> tuple:
    outputs = tuple()
    for state in _iterate_workflow_steps_impl(rdf_source, test_steps=True):
        outputs = state.outputs

    return outputs


@dataclass
class WorkflowState:
    wf_inputs: Dict[str, Any]
    wf_options: Dict[str, Any]
    inputs: tuple
    outputs: tuple
    named_outputs: Dict[str, Any]


def iterate_workflow_steps(
    rdf_source: Union[dict, PathLike, IO, str, bytes, raw_nodes.URI, RawResourceDescription],
    *,
    inputs: Sequence = tuple(),
    options: Dict[str, Any] = None,
) -> Generator[WorkflowState, None, None]:
    yield from _iterate_workflow_steps_impl(rdf_source, inputs=inputs, options=options, test_steps=False)


def iterate_test_workflow_steps(
    rdf_source: Union[dict, PathLike, IO, str, bytes, raw_nodes.URI, RawResourceDescription]
) -> Generator[WorkflowState, None, None]:
    yield from _iterate_workflow_steps_impl(rdf_source, test_steps=True)


def _iterate_workflow_steps_impl(
    rdf_source: Union[dict, PathLike, IO, str, bytes, raw_nodes.URI, RawResourceDescription],
    *,
    test_steps: bool,
    inputs: Sequence = tuple(),
    options: Dict[str, Any] = None,
) -> Generator[WorkflowState, None, None]:
    import bioimageio.core.operators as ops

    workflow = load_resource_description(rdf_source)
    assert isinstance(workflow, nodes.Workflow)
    wf_options: Dict[str, Any] = {opt.name: opt.default for opt in workflow.options_spec}
    if test_steps:
        assert not inputs
        assert not options
        wf_inputs: Dict[str, Any] = {}
        steps = workflow.test_steps
    else:
        if not len(workflow.inputs_spec) == len(inputs):
            raise ValueError(f"Expected {len(workflow.inputs_spec)} inputs, but got {len(inputs)}.")

        wf_inputs = {ipt_spec.name: ipt for ipt_spec, ipt in zip(workflow.inputs_spec, inputs)}
        for k, v in options.items():
            if k not in wf_options:
                raise ValueError(f"Got unknown option {k}, expected one of {set(wf_options)}.")

            wf_options[k] = v

        steps = workflow.steps

    named_outputs = {}  # for later referencing

    def map_ref(value):
        assert isinstance(workflow, nodes.Workflow)
        if isinstance(value, str) and value.startswith("${{") and value.endswith("}}"):
            ref = value[4:-2].strip()
            if ref.startswith("self.inputs."):
                ref = ref[len("self.inputs.") :]
                if ref not in wf_inputs:
                    raise ValueError(f"Invalid workflow input reference {value}.")

                return wf_inputs[ref]
            elif ref.startswith("self.options."):
                ref = ref[len("self.options.") :]
                if ref not in wf_options:
                    raise ValueError(f"Invalid workflow option reference {value}.")

                return wf_options[ref]
            elif ref == "self.rdf_source":
                assert workflow.rdf_source is not missing
                return str(workflow.rdf_source)
            elif ref in named_outputs:
                return named_outputs[ref]
            else:
                raise ValueError(f"Invalid reference {value}.")
        else:
            return value

    # implicit inputs to a step are the outputs of the previous step.
    # For the first step these are the workflow inputs.
    outputs = tuple(inputs)
    for step in steps:
        if not hasattr(ops, step.op):
            raise NotImplementedError(f"{step.op} not implemented in {ops}")

        op = getattr(ops, step.op)
        if step.inputs is missing:
            inputs = outputs
        else:
            inputs = tuple(map_ref(ipt) for ipt in step.inputs)

        assert isinstance(inputs, tuple)
        options = {k: map_ref(v) for k, v in (step.options or {}).items()}
        outputs = op(*inputs, **options)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        if step.outputs:
            assert step.id is not missing
            if len(step.outputs) != len(outputs):
                raise ValueError(
                    f"Got {len(step.outputs)} step output name{'s' if len(step.outputs) > 1 else ''} ({step.id}.outputs), "
                    f"but op {step.op} returned {len(outputs)} outputs."
                )

            named_outputs.update({f"{step.id}.outputs.{out_name}": out for out_name, out in zip(step.outputs, outputs)})

        yield WorkflowState(
            wf_inputs=wf_inputs, wf_options=wf_options, inputs=inputs, outputs=outputs, named_outputs=named_outputs
        )
    if len(workflow.outputs_spec) != len(outputs):
        raise ValueError(f"Expected {len(workflow.outputs_spec)} outputs from last step, but got {len(outputs)}.")

    def tensor_as_xr(tensor, axes: Sequence[nodes.Axis]):
        spec_axes = [a.name or a.type for a in axes]
        if isinstance(tensor, xr.DataArray):
            if list(tensor.dims) != spec_axes:
                raise ValueError(
                    f"Last workflow step returned xarray.DataArray with dims {tensor.dims}, but expected dims {spec_axes}."
                )

            return tensor
        else:
            return xr.DataArray(tensor, dims=tuple(spec_axes))

    outputs = tuple(
        tensor_as_xr(out, out_spec.axes) if out_spec.type == "tensor" else out
        for out_spec, out in zip(workflow.outputs_spec, outputs)
    )
    yield WorkflowState(
        wf_inputs=wf_inputs, wf_options=wf_options, inputs=inputs, outputs=outputs, named_outputs=named_outputs
    )
