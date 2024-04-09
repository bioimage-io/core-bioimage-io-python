from pathlib import Path
from typing import Any, Optional, Sequence

import imageio
from loguru import logger
from numpy.typing import NDArray

from bioimageio.spec.model import AnyModelDescr
from bioimageio.spec.utils import load_array

from .axis import Axis, AxisLike
from .digest_spec import create_sample_for_model, get_axes_infos
from .stat_measures import Stat
from .tensor import Tensor


def load_image(path: Path, is_volume: bool) -> NDArray[Any]:
    """load a single image as numpy array"""
    ext = path.suffix
    if ext == ".npy":
        return load_array(path)
    else:
        return imageio.volread(path) if is_volume else imageio.imread(path)


def load_tensor(path: Path, axes: Optional[Sequence[AxisLike]] = None) -> Tensor:
    array = load_image(
        path,
        is_volume=(
            axes is None or sum(Axis.create(a).type != "channel" for a in axes) > 2
        ),
    )

    return Tensor.from_numpy(array, dims=axes)


def load_sample_for_model(
    *paths: Path,
    model: AnyModelDescr,
    axes: Optional[Sequence[Sequence[AxisLike]]] = None,
    stat: Optional[Stat] = None,
):
    """load a single sample from `paths` that can be processed by `model`"""

    if axes is None:
        axes = [get_axes_infos(ipt) for ipt in model.inputs[: len(paths)]]
        logger.warning(
            "loading paths with default input axes: {} (from {})",
            axes,
            model.id or model.name,
        )
    elif len(axes) != len(paths):
        raise ValueError(f"got {len(paths)} paths, but {len(axes)} axes hints!")

    arrays = [load_image(p, is_volume=True) for p in paths]
    return create_sample_for_model(
        arrays,
        model,
        stat={} if stat is None else stat,
    )
