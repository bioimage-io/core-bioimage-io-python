import collections.abc
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import imageio
from loguru import logger
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, TypeAdapter

from bioimageio.core.common import PerMember
from bioimageio.core.stat_measures import DatasetMeasure, MeasureValue
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
    path.parent.mkdir(exist_ok=True, parents=True)
    if path.suffix == ".npy":
        save_array(path, data)
    else:
        if singleton_axes := [a for a, s in tensor.tagged_shape.items() if s == 1]:
            tensor = tensor[{a: 0 for a in singleton_axes}]
            singleton_axes_msg = f"(without singleton axes {singleton_axes}) "
        else:
            singleton_axes_msg = ""

        # attempt to write a volume or an image with imageio
        error = None
        for d in (data, data.T):
            for write in (  # pyright: ignore[reportUnknownVariableType]
                imageio.volwrite,
                imageio.imwrite,
            ):
                try:
                    write(path, d)
                except ValueError as e:
                    error = e
                else:
                    logger.info(
                        "wrote tensor {} {}to {} using imageio.{}",
                        dict(tensor.tagged_shape),
                        singleton_axes_msg,
                        path,
                        write.__name__,
                    )

        if error is not None:
            raise error


def save_sample(path: Union[Path, str, PerMember[Path]], sample: Sample) -> None:
    """save a sample to path

    If `path` is a pathlib.Path or a string it must contain `{member_id}` and may contain `{sample_id}`,
    which are resolved with the `sample` object.
    """

    if not isinstance(path, collections.abc.Mapping) and "{member_id}" not in str(path):
        raise ValueError(f"missing `{{member_id}}` in path {path}")

    for m, t in sample.members.items():
        if isinstance(path, collections.abc.Mapping):
            p = path[m]
        else:
            p = Path(str(path).format(sample_id=sample.id, member_id=m))

        save_tensor(p, t)


class _SerializedDatasetStatsEntry(
    BaseModel, frozen=True, arbitrary_types_allowed=True
):
    measure: DatasetMeasure
    value: MeasureValue


_stat_adapter = TypeAdapter(
    Sequence[_SerializedDatasetStatsEntry],
    config=ConfigDict(arbitrary_types_allowed=True),
)


def save_dataset_stat(stat: Mapping[DatasetMeasure, MeasureValue], path: Path):
    serializable = [
        _SerializedDatasetStatsEntry(measure=k, value=v) for k, v in stat.items()
    ]
    _ = path.write_bytes(_stat_adapter.dump_json(serializable))


def load_dataset_stat(path: Path):
    seq = _stat_adapter.validate_json(path.read_bytes())
    return {e.measure: e.value for e in seq}
