import math
import warnings
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple, TypeVar

import numpy as np

from bioimageio.spec.model import raw_nodes

TA = TypeVar("TA")
TS = TypeVar("TS")


def transpose_sequence(sequence: Sequence[TS], axes: Sequence[TA], desired_axes: Sequence[TA], default) -> List[TS]:
    """transpose a sequence according to its axes to match a desired axes order,
    filling non-exising entries with default

    Returns
        sequence: the transposed sequence as a list
    """
    return [default if ia not in axes else sequence[axes.index(ia)] for ia in desired_axes]


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
            np.array(transpose_sequence(ot.shape.scale, ot.axes, ipt.axes, 1.0)),
            np.array(transpose_sequence(ot.shape.offset, ot.axes, ipt.axes, 0.0)),
            np.array(transpose_sequence(ot.halo, ot.axes, ipt.axes, 0.0)),
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


def get_default_input_tile(ipt: raw_nodes.InputTensor) -> List[int]:
    """Guess a good"""
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
