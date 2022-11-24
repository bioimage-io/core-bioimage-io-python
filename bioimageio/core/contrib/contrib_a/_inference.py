import collections
from os import PathLike
from typing import Dict, IO, List, Optional, OrderedDict, Sequence, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr

from bioimageio.core.contrib.utils import (
    get_chunk,
    get_corrected_chunks,
    get_default_input_tile,
    get_output_rois,
    transpose_sequence,
    tuple_roi_to_slices,
)
from bioimageio.core.prediction_pipeline._combined_processing import CombinedProcessing
from bioimageio.core.prediction_pipeline._model_adapters import ModelAdapter, create_model_adapter
from bioimageio.core.resource_io import nodes
from bioimageio.core.resource_io.utils import resolve_raw_node
from bioimageio.spec import load_raw_resource_description
from bioimageio.spec.model import raw_nodes
from bioimageio.spec.shared.raw_nodes import ResourceDescription as RawResourceDescription

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


BoundaryMode = Literal["reflect"]


def forward(*tensors, model_adapter: ModelAdapter, output_tile_roi: Tuple[slice, ...]):
    """helper to cast dask array chunks to xr.DataArray and apply a roi to the output"""
    assert len(model_adapter.bioimageio_model.inputs) == len(tensors), (
        len(model_adapter.bioimageio_model.inputs),
        len(tensors),
    )
    tensors = [xr.DataArray(t, dims=tuple(ipt.axes)) for ipt, t, in zip(model_adapter.bioimageio_model.inputs, tensors)]
    output = model_adapter.forward(*tensors)[0]  # todo: allow more than 1 output
    return output[output_tile_roi]


async def run_model_inference_with_dask(
    model_rdf: Union[str, PathLike, dict, IO, bytes, raw_nodes.URI, RawResourceDescription],
    tensors: Sequence[xr.DataArray],
    boundary_mode: Union[
        BoundaryMode,
        Sequence[BoundaryMode],
    ] = "reflect",
    enable_preprocessing: bool = True,
    enable_postprocessing: bool = True,
    devices: Sequence[str] = ("cpu",),
    tiles: Optional[Sequence[Dict[str, int]]] = None,
) -> OrderedDict[str, xr.DataArray]:
    """run model inference using chunked dask arrays for tiling

    To run inference on arbitrary input tensors, they are chunked such that with halo and offset all inputs to the
    model have `tiles` shape.

    .. code-block:: yaml
    authors: [{name: Fynn Beuttenmüller, github_user: fynnbe}]
    cite: [{text: BioImage.IO, url: "https://doi.org/10.1101/2022.06.07.495102"}]

    Args:
        model_rdf: the (source/raw) model RDF that describes the model to be used for inference
        tensors: model input tensors
        boundary_mode: How to pad missing values.
        enable_preprocessing: If true, apply the preprocessing specified in the model RDF
        enable_postprocessing: If true, apply the postprocessing specified in the model RDF
        devices: devices to use for inference (device management is handled by the created model adapter)
        tiles: Defaults to using an estimated tile sizes based on the model RDF.

    Returns:
        outputs. named model outputs
    """
    model: raw_nodes.Model = load_raw_resource_description(model_rdf, update_to_format="latest")  # noqa
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

    if tiles is None:
        tiles = [get_default_input_tile(ipt) for ipt in model.inputs]

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
        ipt_shape = np.array(transpose_sequence(ipt_shape, ipt_axes, out.axes, 0))
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

    outputs = collections.OrderedDict({out.name: xr.DataArray(res, dims=tuple(out.axes))})
    if enable_postprocessing:
        assert postprocessing is not None
        sample = {name: t for name, t in outputs.items()}
        postprocessing.apply(sample, {})
        outputs = collections.OrderedDict({out.name: sample[out.name] for out in model.outputs})

    return outputs
