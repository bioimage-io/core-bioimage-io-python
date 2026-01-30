import collections.abc
import warnings
import zipfile
from pathlib import Path
from shutil import copyfileobj
from typing import (
    Any,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

from imageio.v3 import imread, imwrite  # type: ignore
from loguru import logger
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, TypeAdapter

from bioimageio.spec._internal.io import get_reader, interprete_file_source
from bioimageio.spec._internal.type_guards import is_ndarray
from bioimageio.spec.common import (
    BytesReader,
    FileSource,
    HttpUrl,
    PermissiveFileSource,
    RelativeFilePath,
    ZipPath,
)
from bioimageio.spec.utils import download, load_array, save_array

from .axis import AxisId, AxisLike
from .common import PerMember
from .sample import Sample
from .stat_measures import DatasetMeasure, MeasureValue
from .tensor import Tensor


def load_image(
    source: Union[ZipPath, PermissiveFileSource], is_volume: Optional[bool] = None
) -> NDArray[Any]:
    """load a single image as numpy array

    Args:
        source: image source
        is_volume: deprecated
    """
    if is_volume is not None:
        warnings.warn("**is_volume** is deprecated and will be removed soon.")

    if isinstance(source, ZipPath):
        parsed_source = source
    else:
        parsed_source = interprete_file_source(source)

    if isinstance(parsed_source, RelativeFilePath):
        parsed_source = parsed_source.absolute()

    if parsed_source.suffix == ".npy":
        image = load_array(parsed_source)
    else:
        reader = download(parsed_source)
        image = imread(  # pyright: ignore[reportUnknownVariableType]
            reader.read(), extension=parsed_source.suffix
        )

    assert is_ndarray(image)
    return image


def load_tensor(
    path: Union[ZipPath, Path, str], axes: Optional[Sequence[AxisLike]] = None
) -> Tensor:
    # TODO: load axis meta data
    array = load_image(path)

    return Tensor.from_numpy(array, dims=axes)


_SourceT = TypeVar("_SourceT", Path, HttpUrl, ZipPath)

Suffix = str


def save_tensor(path: Union[Path, str], tensor: Tensor) -> None:
    # TODO: save axis meta data

    data: NDArray[Any] = (  # pyright: ignore[reportUnknownVariableType]
        tensor.data.to_numpy()
    )
    assert is_ndarray(data)
    path = Path(path)
    if not path.suffix:
        raise ValueError(f"No suffix (needed to decide file format) found in {path}")

    extension = path.suffix.lower()
    path.parent.mkdir(exist_ok=True, parents=True)
    if extension == ".npy":
        save_array(path, data)
    elif extension in (".h5", ".hdf", ".hdf5"):
        raise NotImplementedError("Saving to h5 with dataset path is not implemented.")
    else:
        if (
            extension in (".tif", ".tiff")
            and tensor.tagged_shape.get(ba := AxisId("batch")) == 1
        ):
            # remove singleton batch axis for saving
            tensor = tensor[{ba: 0}]
            singleton_axes_msg = f"(without singleton batch axes) "
        else:
            singleton_axes_msg = ""

        logger.debug(
            "writing tensor {} {}to {}",
            dict(tensor.tagged_shape),
            singleton_axes_msg,
            path,
        )
        imwrite(path, data, extension=extension)


def save_sample(
    path: Union[Path, str, PerMember[Union[Path, str]]], sample: Sample
) -> None:
    """Save a **sample** to a **path** pattern
    or all sample members in the **path** mapping.

    If **path** is a pathlib.Path or a string and the **sample** has multiple members,
    **path** it must contain `{member_id}` (or `{input_id}` or `{output_id}`).

    (Each) **path** may contain `{sample_id}` to be formatted with the **sample** object.
    """
    if not isinstance(path, collections.abc.Mapping):
        if len(sample.members) < 2 or any(
            m in str(path) for m in ("{member_id}", "{input_id}", "{output_id}")
        ):
            path = {m: path for m in sample.members}
        else:
            raise ValueError(
                f"path {path} must contain '{{member_id}}' for sample with multiple members {list(sample.members)}."
            )

    for m, p in path.items():
        t = sample.members[m]
        p_formatted = Path(
            str(p).format(sample_id=sample.id, member_id=m, input_id=m, output_id=m)
        )
        save_tensor(p_formatted, t)


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


def ensure_unzipped(
    source: Union[PermissiveFileSource, ZipPath, BytesReader], folder: Path
):
    """unzip a (downloaded) **source** to a file in **folder** if source is a zip archive
    otherwise copy **source** to a file in **folder**."""
    if isinstance(source, BytesReader):
        weights_reader = source
    else:
        weights_reader = get_reader(source)

    out_path = folder / (
        weights_reader.original_file_name or f"file{weights_reader.suffix}"
    )

    if zipfile.is_zipfile(weights_reader):
        out_path = out_path.with_name(out_path.name + ".unzipped")
        out_path.parent.mkdir(exist_ok=True, parents=True)
        # source itself is a zipfile
        with zipfile.ZipFile(weights_reader, "r") as f:
            f.extractall(out_path)

    else:
        out_path.parent.mkdir(exist_ok=True, parents=True)
        with out_path.open("wb") as f:
            copyfileobj(weights_reader, f)

    return out_path


def get_suffix(source: Union[ZipPath, FileSource]) -> Suffix:
    """DEPRECATED: use source.suffix instead."""
    return source.suffix
