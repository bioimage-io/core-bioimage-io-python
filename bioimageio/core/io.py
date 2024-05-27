from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import imageio
import numpy as np
import xarray as xr
from loguru import logger
from numpy.typing import NDArray
from typing_extensions import assert_never

from bioimageio.spec.model import AnyModelDescr, v0_4, v0_5
from bioimageio.spec.utils import load_array, save_array

from .axis import Axis, AxisLike
from .common import MemberId, PerMember, SampleId
from .digest_spec import get_axes_infos, get_member_id
from .sample import Sample
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
    # TODO: load axis meta data
    array = load_image(
        path,
        is_volume=(
            axes is None or sum(Axis.create(a).type != "channel" for a in axes) > 2
        ),
    )

    return Tensor.from_numpy(array, dims=axes)


def get_tensor(
    src: Union[Tensor, xr.DataArray, NDArray[Any], Path],
    ipt: Union[v0_4.InputTensorDescr, v0_5.InputTensorDescr],
):
    """helper to cast/load various tensor sources"""

    if isinstance(src, Tensor):
        return src

    if isinstance(src, xr.DataArray):
        return Tensor.from_xarray(src)

    if isinstance(src, np.ndarray):
        return Tensor.from_numpy(src, dims=get_axes_infos(ipt))

    if isinstance(src, Path):
        return load_tensor(src, axes=get_axes_infos(ipt))

    assert_never(src)


def save_tensor(path: Path, tensor: Tensor) -> None:
    # TODO: save axis meta data
    data: NDArray[Any] = tensor.data.to_numpy()
    if path.suffix == ".npy":
        save_array(path, data)
    else:
        imageio.volwrite(path, data)


def save_sample(path: Union[Path, str], sample: Sample) -> None:
    """save a sample to path

    `path` must contain `{member_id}` and may contain `{sample_id}`,
    which are resolved with the `sample` object.
    """
    path = str(path).format(sample_id=sample.id)
    if "{member_id}" not in path:
        raise ValueError(f"missing `{{member_id}}` in path {path}")

    for m, t in sample.members.items():
        save_tensor(Path(path.format(member_id=m)), t)


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
            logger.warning(
                "loading paths with {}'s default input axes {} for input '{}'",
                axes[m],
                model.id or model.name,
                m,
            )
        members[m] = load_tensor(p, axes[m])

    return Sample(
        members=members,
        stat={} if stat is None else stat,
        id=sample_id or tuple(sorted(paths.values())),
    )
