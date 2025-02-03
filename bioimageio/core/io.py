import collections.abc
import warnings
import zipfile
from io import TextIOWrapper
from pathlib import Path, PurePosixPath
from shutil import copyfileobj
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
from imageio.v3 import imread, imwrite  # type: ignore
from loguru import logger
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, TypeAdapter

from bioimageio.spec.common import FileSource, ZipPath
from bioimageio.spec.utils import download, load_array, save_array

from .axis import AxisLike
from .common import PerMember
from .sample import Sample
from .stat_measures import DatasetMeasure, MeasureValue
from .tensor import Tensor

DEFAULT_H5_DATASET_PATH = "data"


def load_image(path: Path, is_volume: Optional[bool] = None) -> NDArray[Any]:
    """load a single image as numpy array

    Args:
        path: image path
        is_volume: deprecated
    """
    if is_volume is not None:
        warnings.warn("**is_volume** is deprecated and will be removed soon.")

    file_path, subpath = _split_dataset_path(Path(path))

    if file_path.suffix == ".npy":
        if subpath is not None:
            raise ValueError(f"Unexpected subpath {subpath} for .npy path {path}")
        return load_array(path)
    elif file_path.suffix in (".h5", ".hdf", ".hdf5"):
        if subpath is None:
            dataset_path = DEFAULT_H5_DATASET_PATH
        else:
            dataset_path = str(subpath)

        with h5py.File(file_path, "r") as f:
            h5_dataset = f.get(  # pyright: ignore[reportUnknownVariableType]
                dataset_path
            )
            if not isinstance(h5_dataset, h5py.Dataset):
                raise ValueError(
                    f"{path} is not of type {h5py.Dataset}, but has type "
                    + str(
                        type(h5_dataset)  # pyright: ignore[reportUnknownArgumentType]
                    )
                )
            image: NDArray[Any]
            image = h5_dataset[:]  # pyright: ignore[reportUnknownVariableType]
            assert isinstance(image, np.ndarray), type(
                image  # pyright: ignore[reportUnknownArgumentType]
            )
            return image  # pyright: ignore[reportUnknownVariableType]
    else:
        return imread(path)  # pyright: ignore[reportUnknownVariableType]


def load_tensor(path: Path, axes: Optional[Sequence[AxisLike]] = None) -> Tensor:
    # TODO: load axis meta data
    array = load_image(path)

    return Tensor.from_numpy(array, dims=axes)


def _split_dataset_path(path: Path) -> Tuple[Path, Optional[PurePosixPath]]:
    """Split off subpath (e.g. internal  h5 dataset path)
    from a file path following a file extension.

    Examples:
        >>> _split_dataset_path(Path("my_file.h5/dataset"))
        (...Path('my_file.h5'), PurePosixPath('dataset'))

        If no suffix is detected the path is returned with
        >>> _split_dataset_path(Path("my_plain_file"))
        (...Path('my_plain_file'), None)

    """
    if path.suffix:
        return path, None

    for p in path.parents:
        if p.suffix:
            return p, PurePosixPath(path.relative_to(p))

    return path, None


def save_tensor(path: Path, tensor: Tensor) -> None:
    # TODO: save axis meta data

    data: NDArray[Any] = tensor.data.to_numpy()
    file_path, subpath = _split_dataset_path(Path(path))
    if not file_path.suffix:
        raise ValueError(f"No suffix (needed to decide file format) found in {path}")

    file_path.parent.mkdir(exist_ok=True, parents=True)
    if file_path.suffix == ".npy":
        if subpath is not None:
            raise ValueError(f"Unexpected subpath {subpath} found in .npy path {path}")
        save_array(file_path, data)
    elif file_path.suffix in (".h5", ".hdf", ".hdf5"):
        if subpath is None:
            dataset_path = DEFAULT_H5_DATASET_PATH
        else:
            dataset_path = str(subpath)

        with h5py.File(file_path, "a") as f:
            if dataset_path in f:
                del f[dataset_path]

            _ = f.create_dataset(dataset_path, data=data, chunks=True)
    else:
        # if singleton_axes := [a for a, s in tensor.tagged_shape.items() if s == 1]:
        #     tensor = tensor[{a: 0 for a in singleton_axes}]
        #     singleton_axes_msg = f"(without singleton axes {singleton_axes}) "
        # else:
        singleton_axes_msg = ""

        logger.debug(
            "writing tensor {} {}to {}",
            dict(tensor.tagged_shape),
            singleton_axes_msg,
            path,
        )
        imwrite(path, data)


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


def ensure_unzipped(source: Union[FileSource, ZipPath], folder: Path):
    """unzip a (downloaded) **source** to a file in **folder** if source is a zip archive.
    Always returns the path to the unzipped source (maybe source itself)"""
    local_weights_file = download(source).path
    if isinstance(local_weights_file, ZipPath):
        # source is inside a zip archive
        out_path = folder / local_weights_file.filename
        with local_weights_file.open("rb") as src, out_path.open("wb") as dst:
            assert not isinstance(src, TextIOWrapper)
            copyfileobj(src, dst)

        local_weights_file = out_path

    if zipfile.is_zipfile(local_weights_file):
        # source itself is a zipfile
        out_path = folder / local_weights_file.with_suffix(".unzipped").name
        with zipfile.ZipFile(local_weights_file, "r") as f:
            f.extractall(out_path)

        return out_path
    else:
        return local_weights_file
