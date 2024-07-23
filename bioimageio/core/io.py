from pathlib import Path
from typing import Any, Optional, Sequence, Union

import imageio
from numpy.typing import NDArray

from bioimageio.spec.utils import load_array, save_array

from .axis import Axis, AxisLike
from .sample import Sample
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
    if "{member_id}" not in str(path):
        raise ValueError(f"missing `{{member_id}}` in path {path}")

    path = str(path).format(sample_id=sample.id, member_id="{member_id}")

    for m, t in sample.members.items():
        save_tensor(Path(path.format(member_id=m)), t)
