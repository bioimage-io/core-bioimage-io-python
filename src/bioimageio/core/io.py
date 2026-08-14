from __future__ import annotations

import collections.abc
import json
import os
import warnings
import zipfile
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from io import BytesIO
from itertools import chain
from pathlib import Path, PosixPath, PurePath
from shutil import copyfileobj
from typing import TYPE_CHECKING, TypedDict, Union

import numpy as np
import xarray as xr
from exceptiongroup import ExceptionGroup
from imageio.v3 import (
    imread,  # pyright: ignore[reportUnknownVariableType]
    imwrite,  # type: ignore
)
from loguru import logger
from pydantic import BaseModel, FilePath, RootModel
from typing_extensions import Literal, TypeAlias, assert_never
from typing_extensions import TypeAliasType as _TypeAliasType

from bioimageio.spec import get_validation_context
from bioimageio.spec._internal.io import get_reader, interprete_file_source
from bioimageio.spec.common import (
    BytesReader,
    FileDescr,
    FileSource,
    HttpUrl,
    PermissiveFileSource,
    RelativeFilePath,
    RootHttpUrl,
    ZipPath,
)
from bioimageio.spec.model import v0_5
from bioimageio.spec.utils import load_array, save_array

from .axis import AxisId, AxisLike, single_letter_dims_if_possible
from .common import PerMember
from .sample import Sample
from .stat_measures import DatasetMeasure, MeasureValue, SampleMeasure, Stat
from .tensor import Tensor
from .utils._io_zarr import open_zarr_multiscale_array

if TYPE_CHECKING:
    JsonValue: TypeAlias = Union[
        bool, int, float, str, None, list["JsonValue"], dict[str, "JsonValue"]
    ]  # note: order relevant for deserializing

else:
    # for pydantic validation we need to use `TypeAliasType`,
    # see https://docs.pydantic.dev/latest/concepts/types/#named-recursive-types
    # however this results in a partially unknown type with the current pyright 1.1.388
    JsonValue: TypeAlias = _TypeAliasType(
        "JsonValue",
        Union[bool, int, float, str, None, list["JsonValue"], dict[str, "JsonValue"]],
    )

JsonValueReadOnly: TypeAlias = Union[
    bool,
    int,
    float,
    str,
    None,
    Sequence["JsonValueReadOnly"],
    Mapping[str, "JsonValueReadOnly"],
]

IO_Lib: TypeAlias = Literal["bioio", "imageio", "clearscale", "numpy"]


def load_tensor(
    source: FileDescr | PermissiveFileSource | ZipPath,
    /,
    axes: Sequence[AxisLike] | None = None,
    io_lib: IO_Lib | None = None,
) -> Tensor:
    """Load an image tensor

    Args:
        io_lib: library to use for loading the image.
        Defaults to trying various libraries depending on file extension.

    """
    file_source, extension, subdir = _interprete_tensor_source(source)
    if io_lib is None:
        try_io_libs = _select_io_libs_based_on_extension(extension)

        exceptions: list[Exception] = []
        for auto_io_lib in try_io_libs:
            try:
                return load_tensor(source, axes=axes, io_lib=auto_io_lib)
            except Exception as e:
                exceptions.append(e)
                logger.opt(exception=e).warning(
                    "Failed to load tensor with io_lib={}", auto_io_lib
                )

        raise ExceptionGroup(
            "Failed to load tensor with any available io_lib", exceptions
        )

    elif io_lib == "bioio":
        t = _load_tensor_bioio(file_source, None if subdir is None else str(subdir))
    elif io_lib == "clearscale":
        t = _load_tensor_clearscale(source)
    elif io_lib == "numpy":
        array = load_array(source)
        t = Tensor.from_array(array, dims=axes)
        axes = None
    elif io_lib == "imageio":
        reader = get_reader(source)
        array = imread(reader.read(), extension=extension)  # pyright: ignore[reportArgumentType]
        t = Tensor.from_array(array, dims=axes)
        axes = None
    else:
        assert_never(io_lib)

    if axes is not None:
        t = t.transpose(axes)

    return t


def _select_io_libs_based_on_extension(extension: str | None) -> tuple[IO_Lib, ...]:
    if extension is None:
        return ("bioio", "imageio", "clearscale")

    extension = extension.lower()
    if extension == ".npy":
        return ("numpy",)
    elif extension == ".zarr":
        return ("bioio", "clearscale")
    else:
        return ("bioio", "imageio", "clearscale")


def _load_tensor_bioio(
    source: HttpUrl | FilePath | ZipPath,
    subdir: str | None,
) -> Tensor:
    """Load an image tensor using bioio"""

    import bioio

    if isinstance(source, FileDescr):
        src = source.source
    else:
        src = source

    if isinstance(src, RelativeFilePath):
        src = src.absolute()

    del source

    img = bioio.BioImage(src)
    if subdir is not None:
        img.set_scene(subdir)

    dataarray_bioio = img.xarray_dask_data.squeeze()  # pyright: ignore[reportUnknownVariableType]
    assert isinstance(dataarray_bioio, xr.DataArray)

    # map bioio TCZYX/YXS axes to AxisId objects
    # (e.g. "T" -> AxisId("time"), "C" -> AxisId("channel"), "S" -> AxisId("channel"))
    axes_map = {
        d: AxisId("channel") if d == "S" else AxisId(d) for d in dataarray_bioio.dims
    }
    dataarray = dataarray_bioio.rename(axes_map)  # pyright: ignore[reportUnknownVariableType]
    assert isinstance(dataarray, xr.DataArray)

    t = Tensor.from_xarray(dataarray)

    # add physical scale
    scale_bioio: dict[str, float | None] = {
        "T": img.scale.T,
        # "C": img.scale.C,  # should be 'dimensionless'
        "Z": img.scale.Z,
        "Y": img.scale.Y,
        "X": img.scale.X,
    }
    unit_bioio: dict[str, str | None] = {
        "T": img.dimension_properties.T.unit,
        # "C": img.dimension_properties.C.unit,  # should be 'dimensionless'
        "Z": img.dimension_properties.Z.unit,
        "Y": img.dimension_properties.Y.unit,
        "X": img.dimension_properties.X.unit,
    }
    scale = {axes_map[k]: v for k, v in scale_bioio.items() if v is not None}
    unit = {axes_map[k]: str(v) for k, v in unit_bioio.items() if v is not None}
    t.set_physical_scale(scale)
    t.set_physical_scale_unit(unit)
    return t


def _load_tensor_clearscale(source: FileDescr | PermissiveFileSource) -> Tensor:
    """load a single image as numpy array

    Args:
        source: image source
        subdir: image name
    """
    import dask.array as da

    if isinstance(source, FileDescr):
        source = source.source.absolute()

    if isinstance(source, RelativeFilePath):
        source = source.absolute()

    if isinstance(source, ZipPath):
        raise NotImplementedError(
            "clearscale: Loading from zip files is not implemented."
        )

    array, ms = open_zarr_multiscale_array(
        source.as_posix() if isinstance(source, Path) else str(source)
    )
    image = da.from_zarr(array)
    axes = tuple(str(a) for a in ms.axes())

    return Tensor.from_xarray(xr.DataArray(image, dims=axes))


Suffix = str


def save_tensor(
    path: Path | str,
    tensor: Tensor,
    io_lib: IO_Lib | None = None,
) -> None:
    output_path, extension, subdir = _interprete_tensor_source(path)
    if isinstance(output_path, RootHttpUrl):
        raise NotImplementedError(
            "Saving tensors to HTTP URLs is not implemented. Please save to a local file path."
        )

    if io_lib is None:
        try_io_libs = _select_io_libs_based_on_extension(extension)
        exceptions: list[Exception] = []
        for try_io_lib in try_io_libs:
            try:
                return save_tensor(path, tensor, io_lib=try_io_lib)
            except Exception as e:
                exceptions.append(e)
                logger.opt(exception=e).warning(
                    "Failed to save tensor with io_lib={}", try_io_lib
                )

        raise ExceptionGroup(
            "Failed to save tensor with any available io_lib", exceptions
        )
    elif io_lib == "bioio":
        return _save_tensor_bioio(output_path, tensor, subdir=subdir)
    elif io_lib == "imageio":
        return _save_tensor_imageio(output_path, tensor)
    elif io_lib == "clearscale":
        return _save_tensor_clearscale(output_path, tensor, subdir=subdir)
    elif io_lib == "numpy":
        return _save_tensor_numpy(output_path, tensor)
    else:
        assert_never(io_lib)


def _save_tensor_bioio(
    path: Path | ZipPath, tensor: Tensor, subdir: PurePath | None
) -> None:
    """Save an image tensor using bioio"""

    bioio_axes_map = {
        "batch": "S",
        "channel": "C",
        "index": "S",
        "time": "T",
    }
    if path.suffix.lower() in (".tif", ".tiff"):
        from bioio_ome_tiff.writers import OmeTiffWriter

        save = OmeTiffWriter.save  # pyright: ignore[reportUnknownVariableType]
        write = None
        bioio_axis_letters = "TCZYX"
    elif path.suffix.lower() == ".zarr":
        import bioio_ome_zarr.writers

        axes_names = single_letter_dims_if_possible(tensor.dims)
        axes_types = [
            {"time": "time", "t": "time", "channel": "channel", "c": "channel"}.get(
                str(a).lower(), "space"
            )
            for a in tensor.dims
        ]
        unit = tensor.get_physical_scale_unit()
        axes_units = [unit.get(a) for a in tensor.dims]

        scale = tensor.get_physical_scale()
        physical_pixel_size = [scale.get(a, 1.0) for a in tensor.dims]

        channel_names = tensor.channel_names
        channel_colors = tensor.channel_colors
        if channel_colors is None and channel_names is not None:
            # use spec's default channel colors
            channel_colors = [
                c.as_hex(format="long")[:6]
                for c in v0_5.ChannelAxis(channel_names=channel_names).channel_colors
            ]

        channels = [
            bioio_ome_zarr.writers.Channel(
                label=f"channel{i}" if channel_names is None else channel_names[i],
                color="#FF0000" if channel_colors is None else channel_colors[i],
            )
            for i in range(tensor.tagged_shape.get(AxisId("channel"), 0))
        ]
        zf = int(os.getenv("ZARR_FORMAT", "3"))
        if zf not in (2, 3):
            raise NotImplementedError(
                f"ZARR_FORMAT={zf} is not supported. (expected 2 or 3)"
            )

        save = None
        write = bioio_ome_zarr.writers.OMEZarrWriter(
            store=path,
            level_shapes=[tensor.shape],
            dtype=tensor.dtype,
            zarr_format=zf,
            image_name="Image" if subdir is None else str(subdir),
            axes_names=axes_names,
            axes_types=axes_types,
            axes_units=axes_units,
            physical_pixel_size=physical_pixel_size,
            channels=channels,
        ).write_full_volume
        bioio_axis_letters = "TCZYX"
    elif path.suffix.lower() in (".gif", ".mp4", ".mkv"):
        from bioio_imageio.writers import TimeseriesWriter

        tensor = tensor.squeeze()
        save = TimeseriesWriter.save  # pyright: ignore[reportUnknownVariableType]
        write = None
        bioio_axes_map["channel"] = "S"
        bioio_axis_letters = "YXS"
    elif path.suffix.lower() in (
        ".png",
        ".bmp",
        ".jpg",
        ".mov",
        ".avi",
        ".mpg",
        ".mpeg",
        ".mp4",
        ".mkv",
        ".wmv",
        ".ogg",
    ):
        from bioio_imageio.writers import TwoDWriter

        tensor = tensor.squeeze()
        save = TwoDWriter.save  # pyright: ignore[reportUnknownVariableType]
        write = None
        # channel dim denoted as 'S'
        bioio_axes_map["channel"] = "S"
        bioio_axis_letters = "YXS"
    else:
        raise RuntimeError(
            f"Failed to identify a suitable writer for {path} with suffix={path.suffix}."
        )

    if save is not None:
        assert write is None
        bioio_data = tensor.data.rename(  # pyright: ignore[reportUnknownVariableType]
            {
                a: bioio_axes_map.get(str(a).lower(), str(a)[:1].upper())
                for a in tensor.dims
            }
        )
        assert isinstance(bioio_data, xr.DataArray)

        if not all(str(d) in bioio_axis_letters for d in bioio_data.dims):
            raise ValueError(
                f"Failed to save tensor with bioio: dimensions {bioio_data.dims} not in '{bioio_axis_letters}'."
            )

        dim_order = "".join(str(d) for d in bioio_data.dims)

        image_name = None if subdir is None else str(subdir)
        save(
            bioio_data.data,
            path,
            dim_order=dim_order,
            image_name=image_name,
            channel_names=tensor.channel_names,
        )

    if write is not None:
        assert save is None
        write(tensor.data.data)


def _save_tensor_numpy(path: Path | ZipPath, tensor: Tensor) -> None:
    if isinstance(path, ZipPath):
        folder = path.filename.parent
    else:
        folder = path.parent

    folder.mkdir(exist_ok=True, parents=True)
    save_array(path, tensor.to_numpy())


def _save_tensor_imageio(path: Path | ZipPath, tensor: Tensor) -> None:
    if isinstance(path, ZipPath):
        raise NotImplementedError(
            "Saving tensors to zip files is not implemented for imageio. Please save to an unzipped file path."
        )

    path.parent.mkdir(exist_ok=True, parents=True)

    extension = path.suffix.lower()
    removed_singleton_axes: list[AxisId] = []
    remove_singletons = {
        AxisId("batch"): [
            ".tif",
            ".tiff",
        ],  # remove singleton batch dim for tiff files
        **{
            a: [".png", ".jpg", ".jpeg"] for a in tensor.dims
        },  # remove any singleton axis for png and jpg files
    }
    for rm_a, rm_ext in remove_singletons.items():
        if extension in rm_ext and tensor.tagged_shape.get(rm_a) == 1:
            tensor = tensor[{rm_a: 0}]
            removed_singleton_axes.append(rm_a)

    if removed_singleton_axes:
        singleton_axes_msg = (
            f"(with removed singleton axes {list(map(str, removed_singleton_axes))}) "
        )
    else:
        singleton_axes_msg = ""

    logger.info(
        "writing tensor {} {}to {}",
        dict(tensor.tagged_shape),
        singleton_axes_msg,
        path,
    )
    if extension in (".png", ".jpg", ".jpeg") and tensor.dtype in (
        "float32",
        "float64",
    ):
        logger.warning(
            "converting tensor of dtype {} to uint8 for saving as {}",
            tensor.dtype,
            extension,
        )
        tensor = (
            (tensor - (t_min := tensor.data.min()))
            / xr.ufuncs.maximum(tensor.data.max() - t_min, 1e-8)
            * 255
        ).astype("uint8")

    imwrite(path, tensor, extension=extension)


class CreateArrayKwargs(TypedDict):
    name: str
    overwrite: bool
    dimension_names: tuple[str, ...] | None
    shards: Literal["auto"] | tuple[int, ...] | None


def _save_tensor_clearscale(
    path: Path | ZipPath, tensor: Tensor, subdir: PurePath | None
) -> None:
    import clearscale
    import dask.array
    import zarr

    ZARR_FORMAT: int = int(os.getenv("ZARR_FORMAT", "3"))
    SCALE_KEY = "s0" if subdir is None else str(subdir)
    multiscale = clearscale.Multiscale(
        {
            SCALE_KEY: clearscale.Scale(
                clearscale.Shape(**{str(k): v for k, v in tensor.tagged_shape.items()})
            )
        }
    )
    if ZARR_FORMAT == 2:
        zarr_attrs: dict[str, JsonValueReadOnly] = {
            "multiscales": [multiscale.to_ome_zarr(version="0.4")]
        }
    elif ZARR_FORMAT == 3:
        zarr_attrs: dict[str, JsonValueReadOnly] = {
            "ome": {
                "version": "0.5",
                "multiscales": [multiscale.to_ome_zarr(version="0.5")],
            }
        }
    else:
        raise NotImplementedError(f"ZARR_FORMAT={ZARR_FORMAT} is not supported.")

    zarr_group = zarr.open_group(
        str(path),
        mode="w",
        zarr_format=ZARR_FORMAT,
        attributes=zarr_attrs,
    )
    create_array_kwargs = CreateArrayKwargs(
        name=SCALE_KEY,
        overwrite=True,
        dimension_names=None
        if ZARR_FORMAT == 2
        else single_letter_dims_if_possible(tensor.dims),
        shards=None if ZARR_FORMAT == 2 else "auto",
    )
    if isinstance(tensor.data.data, np.ndarray):
        _ = zarr_group.create_array(
            **create_array_kwargs, data=tensor.to_numpy(), write_data=True
        )
    elif isinstance(tensor.data.data, dask.array.Array):
        dask.array.to_zarr(
            tensor.data.data, zarr_group, compute=True, **create_array_kwargs
        )
    else:
        assert_never(tensor.data.data)


def _interprete_tensor_source(source: PermissiveFileSource | ZipPath | FileDescr):
    if isinstance(source, FileDescr):
        source = source.source

    if not isinstance(source, ZipPath):
        source = interprete_file_source(source)

    if isinstance(source, RelativeFilePath):
        source = source.absolute()

    subdir = None
    if isinstance(source, HttpUrl):
        original_source_path = PosixPath(source.path or "")
    elif isinstance(source, ZipPath):
        return source, None, None
    else:
        original_source_path = PosixPath(source)

    file_source = source
    for parent in chain([source], source.parents):
        extension = parent.suffix.lower()
        if extension:
            if isinstance(parent, RootHttpUrl):
                with get_validation_context().replace(perform_io_checks=False):
                    file_source = HttpUrl(parent)

                parent_path = PosixPath(parent.path or "")
            else:
                file_source = parent
                parent_path = PosixPath(parent)

            subdir = original_source_path.relative_to(parent_path)
            if subdir == PosixPath("."):
                subdir = None

            break
    else:
        extension = None

    return file_source, extension, subdir


def save_sample(path: Path | str | PerMember[Path | str], sample: Sample) -> None:
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


class _StatEntry(BaseModel, frozen=True, arbitrary_types_allowed=True):
    """Serializable stat entry"""

    measure: DatasetMeasure | SampleMeasure
    value: MeasureValue


class _StatList(RootModel[list[_StatEntry]]):
    """Serializable stat mapping"""


def serialize_stat(
    stat: Mapping[DatasetMeasure | SampleMeasure, MeasureValue],
) -> list[JsonValue]:
    """Serialize a stat mapping to a JSON string"""
    stat_list = _StatList([_StatEntry(measure=k, value=v) for k, v in stat.items()])
    return stat_list.model_dump(mode="json")


def save_stat(
    stat: Mapping[DatasetMeasure | SampleMeasure, MeasureValue],
    output: Path | BytesIO,
) -> None:
    """Save sample and dataset statistics as a JSON file"""

    if isinstance(output, Path):
        ctxt = output.open("wb")
    else:
        ctxt = nullcontext(output)

    with ctxt as out:
        _ = out.write(json.dumps(serialize_stat(stat), indent=2).encode("utf-8"))


def load_stat(source: Path | str | Sequence[JsonValue]) -> Stat:
    """Load sample and dataset statistics from JSON"""
    if isinstance(source, Path):
        source = source.read_text(encoding="utf-8")

    if isinstance(source, str):
        seq = _StatList.model_validate_json(source)
    else:
        seq = _StatList.model_validate(source)

    return {e.measure: e.value for e in seq.root}


def save_dataset_stat(stat: Mapping[DatasetMeasure, MeasureValue], path: Path) -> None:
    """DEPRECATED alias for save_stat(): use `save_stats()` instead."""
    warnings.warn("`save_dataset_stat()` is deprecated, use `save_stats()` instead.")
    save_stat({k: v for k, v in stat.items()}, path)


def load_dataset_stat(path: Path) -> Stat:
    """DEPRECATED alias for `load_stat()`: use `load_stat()` instead."""
    warnings.warn("`load_dataset_stat()` is deprecated, use `load_stats()` instead.")
    return load_stat(path)


def ensure_unzipped(source: PermissiveFileSource | ZipPath | BytesReader, folder: Path):
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


def get_suffix(source: ZipPath | FileSource) -> Suffix:
    """DEPRECATED: use source.suffix instead."""
    return source.suffix
