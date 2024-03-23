from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import imageio
import numpy as np
from numpy.typing import NDArray

from bioimageio.core.common import Axis, AxisLike, Tensor
from bioimageio.spec.model import v0_4
from bioimageio.spec.model.v0_4 import InputTensorDescr as InputTensorDescr04
from bioimageio.spec.model.v0_4 import OutputTensorDescr as OutputTensorDescr04
from bioimageio.spec.model.v0_5 import (
    AnyAxis,
    AxisId,
    BatchAxis,
    ChannelAxis,
    Identifier,
    InputTensorDescr,
    OutputTensorDescr,
    SizeReference,
    SpaceInputAxis,
    convert_axes,
)
from bioimageio.spec.utils import load_array

InputTensor = Union[InputTensorDescr04, InputTensorDescr]
OutputTensor = Union[OutputTensorDescr04, OutputTensorDescr]


def normalize_axes(
    axes: Union[v0_4.AxesStr, Sequence[Union[AnyAxis, AxisLike]]]
) -> Tuple[Axis, ...]:
    AXIS_TYPE_MAP: Dict[str, Literal["batch", "time", "index", "channel", "space"]] = {
        "b": "batch",
        "t": "time",
        "i": "index",
        "c": "channel",
        "x": "space",
        "y": "space",
        "z": "space",
    }
    AXIS_ID_MAP = {
        "b": "batch",
        "t": "time",
        "i": "index",
        "c": "channel",
    }
    if isinstance(axes, str):
        return tuple(
            Axis(id=AxisId(AXIS_ID_MAP.get(a, a)), type=AXIS_TYPE_MAP[a]) for a in axes
        )
    else:
        return tuple(Axis(id=AxisId(a.id), type=a.type) for a in axes)


def _interprete_array_wo_known_axes(array: NDArray[Any]):
    ndim = array.ndim
    if ndim == 2:
        current_axes = (
            SpaceInputAxis(id=AxisId("y"), size=array.shape[0]),
            SpaceInputAxis(id=AxisId("x"), size=array.shape[1]),
        )
    elif ndim == 3 and any(s <= 3 for s in array.shape):
        current_axes = (
            ChannelAxis(
                channel_names=[Identifier(f"channel{i}") for i in range(array.shape[0])]
            ),
            SpaceInputAxis(id=AxisId("y"), size=array.shape[1]),
            SpaceInputAxis(id=AxisId("x"), size=array.shape[2]),
        )
    elif ndim == 3:
        current_axes = (
            SpaceInputAxis(id=AxisId("z"), size=array.shape[0]),
            SpaceInputAxis(id=AxisId("y"), size=array.shape[1]),
            SpaceInputAxis(id=AxisId("x"), size=array.shape[2]),
        )
    elif ndim == 4:
        current_axes = (
            ChannelAxis(
                channel_names=[Identifier(f"channel{i}") for i in range(array.shape[0])]
            ),
            SpaceInputAxis(id=AxisId("z"), size=array.shape[1]),
            SpaceInputAxis(id=AxisId("y"), size=array.shape[2]),
            SpaceInputAxis(id=AxisId("x"), size=array.shape[3]),
        )
    elif ndim == 5:
        current_axes = (
            BatchAxis(),
            ChannelAxis(
                channel_names=[Identifier(f"channel{i}") for i in range(array.shape[1])]
            ),
            SpaceInputAxis(id=AxisId("z"), size=array.shape[2]),
            SpaceInputAxis(id=AxisId("y"), size=array.shape[3]),
            SpaceInputAxis(id=AxisId("x"), size=array.shape[4]),
        )
    else:
        raise ValueError(f"Could not guess an axis mapping for {array.shape}")

    return Tensor(array, dims=tuple(a.id for a in current_axes))


def interprete_array(
    array: NDArray[Any],
    axes: Optional[Union[v0_4.AxesStr, Sequence[AnyAxis]]],
) -> Tensor:
    if axes is None:
        return _interprete_array_wo_known_axes(array)

    original_shape = tuple(array.shape)
    if len(array.shape) > len(axes):
        # remove singletons
        for i, s in enumerate(array.shape):
            if s == 1:
                array = np.take(array, 0, axis=i)
                if len(array.shape) == len(axes):
                    break

    # add singletons if nececsary
    for a in axes:
        if len(array.shape) >= len(axes):
            break

        if isinstance(a, str) or a.size is None:
            array = array[None]
            continue

        if isinstance(a.size, int):
            if a.size == 1:
                array = array[None]

            continue

        if isinstance(a.size, SizeReference):
            continue  # TODO: check if singleton is ok for a `SizeReference`

        try:
            maybe_size_one = a.size.validate_size(
                1
            )  # TODO: refactor validate_size() to have boolean func here
        except ValueError:
            continue

        if maybe_size_one == 1:
            array = array[None]

    if len(array.shape) != len(axes):
        raise ValueError(f"Array shape {original_shape} does not map to axes {axes}")

    normalized_axes = normalize_axes(axes)
    assert len(normalized_axes) == len(axes)
    return Tensor(array, dims=tuple(a.id for a in normalized_axes))


def transpose_tensor(
    tensor: Tensor,
    axes: Sequence[AxisId],
) -> Tensor:
    """Transpose `array` to `axes` order.

    Args:
        tensor: the input array
        axes: the desired array axes
    """
    # expand the missing image axes
    current_axes = tuple(d if isinstance(d, AxisId) else AxisId(d) for d in tensor.dims)
    missing_axes = tuple(a for a in axes if a not in current_axes)
    tensor = tensor.expand_dims(missing_axes)
    # transpose to the correct axis order
    return tensor.transpose(*map(str, axes))


def convert_v0_4_axes_for_known_shape(axes: v0_4.AxesStr, shape: Sequence[int]):
    return convert_axes(axes, shape=shape, tensor_type="input", halo=None, size_refs={})


def load_tensor(
    path: Path,
    axes: Optional[Sequence[AnyAxis]] = None,
) -> Tensor:

    ext = path.suffix
    if ext == ".npy":
        array = load_array(path)
    else:
        is_volume = True if axes is None else sum(a.type != "channel" for a in axes) > 2
        array = imageio.volread(path) if is_volume else imageio.imread(path)

    return interprete_array(array, axes)
