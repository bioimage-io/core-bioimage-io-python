from __future__ import annotations

import json
import os
import warnings
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from io import BytesIO, TextIOWrapper
from itertools import chain
from pathlib import Path, PosixPath, PurePath, PurePosixPath
from shutil import copyfileobj
from typing import TYPE_CHECKING, TypedDict, Union

import numpy as np
import pydantic
import xarray as xr
from exceptiongroup import ExceptionGroup
from imageio.v3 import (
    imread,  # pyright: ignore[reportUnknownVariableType]
    imwrite,  # type: ignore
)
from loguru import logger
from pydantic import BaseModel, RootModel, TypeAdapter
from typing_extensions import Literal, TypeAlias, assert_never
from typing_extensions import TypeAliasType as _TypeAliasType

from bioimageio.core.utils._tiff_metadata import (
    ImageJMetadata,
    OmeChannelMetadata,
    OmeMetadata,
    create_ome_metadata_from_xml_string,
)
from bioimageio.spec._internal.io import (
    RelativeDirectory,
    ZarrSource,
    ZarrUrl,
    get_reader,
    interprete_file_source,
)
from bioimageio.spec.common import (
    BytesReader,
    FileDescr,
    FileSource,
    FtpUrl,
    HttpUrl,
    PermissiveFileSource,
    RelativeFilePath,
    RootHttpUrl,
    ZipPath,
)
from bioimageio.spec.model import v0_5
from bioimageio.spec.utils import load_array, save_array

from .axis import AxisId, AxisLike, PerAxis, single_letter_dims_if_possible
from .common import PerMember, SliceInfo
from .sample import Sample, SampleBlock
from .stat_measures import DatasetMeasure, MeasureValue, SampleMeasure, Stat
from .tensor import Tensor
from .utils._color import hex_to_rgb
from .utils._io_zarr import open_zarr_multiscale_array
from .utils._type_guards import is_tuple

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

IO_Lib: TypeAlias = Literal["bioio", "imageio", "clearscale", "numpy", "tifffile"]


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
    file_source, extensions, internal_path = _interprete_tensor_source(source)
    if io_lib is None:
        try_io_libs = _select_io_libs_based_on_extension(extensions)

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
        t = _load_tensor_bioio(
            file_source, None if internal_path is None else str(internal_path)
        )
    elif io_lib == "clearscale":
        t = _load_tensor_clearscale(source)
    elif io_lib == "imageio":
        reader = get_reader(source)
        array = imread(
            reader.read(),
            extension=extensions[-1] if extensions else None,  # pyright: ignore[reportArgumentType]
        )
        t = Tensor.from_array(array, dims=axes)
        axes = None
    elif io_lib == "numpy":
        array = load_array(source)
        t = Tensor.from_array(array, dims=axes)
        axes = None
    elif io_lib == "tifffile":
        t = _load_tensor_tifffile(source)
    else:
        assert_never(io_lib)

    if axes is not None:
        t = t.transpose(axes)

    return t


def _select_io_libs_based_on_extension(
    extensions: list[str] | None,
) -> tuple[IO_Lib, ...]:
    if not extensions:
        return ("bioio", "imageio", "clearscale")

    if extensions[-1] == ".npy":
        return ("numpy",)
    elif extensions[-1] == ".zarr":
        return ("bioio", "clearscale")
    elif extensions[-2:] in ([".ome", ".tif"], [".ome", ".tiff"]):
        return ("tifffile", "bioio")
    elif extensions[-1] in (".tif", ".tiff"):
        return ("tifffile", "imageio")
    else:
        return ("bioio", "imageio")


def _load_tensor_bioio(
    source: ZarrSource | FileSource | ZipPath,
    internal_path: str | None,
) -> Tensor:
    """Load an image tensor using bioio"""

    import bioio

    logger.debug("Loading tensor from {} with bioio", source)

    if isinstance(source, FileDescr):
        src = source.source
    else:
        src = source

    if isinstance(src, (RelativeFilePath, RelativeDirectory)):
        src = src.absolute()

    del source

    img = bioio.BioImage(src)
    if internal_path is not None:
        internal_path_found = False
        try:
            import bioio_ome_zarr

        except ImportError:
            pass
        else:
            if isinstance(img.reader, bioio_ome_zarr.Reader):
                multiscale = img.reader._multiscales_metadata[img.current_scene_index]  # pyright: ignore
                datasets = multiscale.get("datasets", [])  # pyright: ignore
                data_paths = [datasets[i].get("path") for i in img.resolution_levels]  # pyright: ignore
                if internal_path not in data_paths:
                    raise ValueError(
                        f"Resolution level {internal_path} not found in multiscales datasets: {data_paths}"
                    )
                resolution_level = data_paths.index(internal_path)
                img.set_resolution_level(resolution_level)
                internal_path_found = True
            else:
                raise NotImplementedError(
                    f"Reading from internal path {internal_path} is not implemented for bioio reader {type(img.reader)}"  # pyright: ignore[reportUnknownArgumentType]
                )

        if not internal_path_found:
            raise ValueError(
                f"Internal path {internal_path} not found with bioio in {src}"
            )

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
        "C": img.scale.C,
        "Z": img.scale.Z,
        "Y": img.scale.Y,
        "X": img.scale.X,
    }
    if (raw_c_unit := img.dimension_properties.C.unit) is None or str(  # pyright: ignore[reportUnknownVariableType]
        raw_c_unit  # pyright: ignore[reportUnknownArgumentType]
    ) == "dimensionless":
        c_unit = None
    else:
        c_unit = str(raw_c_unit)  # pyright: ignore[reportUnknownArgumentType]
    unit_bioio: dict[str, str | None] = {
        "T": img.dimension_properties.T.unit,
        "C": c_unit,
        "Z": img.dimension_properties.Z.unit,
        "Y": img.dimension_properties.Y.unit,
        "X": img.dimension_properties.X.unit,
    }
    scale = {
        axes_map[k]: v
        for k, v in scale_bioio.items()
        if k in axes_map and v is not None
    }
    unit = {
        axes_map[k]: str(v)
        for k, v in unit_bioio.items()
        if k in axes_map and v is not None
    }
    t.set_physical_scale(scale)
    t.set_physical_scale_unit(unit)
    return t


def _load_tensor_clearscale(source: FileDescr | PermissiveFileSource) -> Tensor:
    """load a single image as numpy array"""
    import dask.array as da

    if isinstance(source, FileDescr):
        source = source.source.absolute()

    if isinstance(source, RelativeFilePath):
        source = source.absolute()

    if isinstance(source, ZipPath):
        raise NotImplementedError(
            "clearscale: Loading from zip files is not implemented."
        )

    logger.debug("loading tensor from {} with clearscale", source)
    array, ms = open_zarr_multiscale_array(
        source.as_posix() if isinstance(source, Path) else str(source)
    )
    image = da.from_zarr(array)
    axes = tuple(str(a) for a in ms.axes())

    return Tensor.from_xarray(xr.DataArray(image, dims=axes))


def _load_tensor_tifffile(source: FileDescr | PermissiveFileSource) -> Tensor:
    import tifffile

    logger.debug("loading tensor from {} with tifffile", source)
    if isinstance(source, FileDescr):
        source = source.source.absolute()

    if isinstance(source, RelativeFilePath):
        source = source.absolute()

    if isinstance(source, ZipPath):
        ctxt = source.open("rb")
        name = source.name
        suffixes = source.suffixes
    else:
        if isinstance(source, (RootHttpUrl, pydantic.AnyUrl)):
            source = str(source)

        if isinstance(source, Path):
            suffixes = source.suffixes
        else:
            suffixes = PurePosixPath(source).suffixes

        ctxt = nullcontext(source)
        name = None

    with ctxt as f:
        assert not isinstance(f, TextIOWrapper)

        tif = tifffile.TiffFile(f, name=name)
        try:
            xr_data = tif.asxarray()
            page = tif.pages[0]
            if isinstance(page, tifffile.TiffFrame):
                page = None if page.pages is None else page.pages[0]
                if isinstance(page, tifffile.TiffFrame):
                    page = None

            if page is None:
                x_res = None
                y_res = None
                unit = None
            else:
                x_res = page.tags["XResolution"].value
                y_res = page.tags["YResolution"].value

                try:
                    x_res = x_res[0] / x_res[1] if is_tuple(x_res) else x_res
                    x_res = None if x_res is None else float(x_res)
                except Exception as e:
                    logger.opt(exception=e).warning(
                        "Failed to interprete tiff XResolution tag: {}", x_res
                    )
                    x_res = None

                try:
                    y_res = y_res[0] / y_res[1] if is_tuple(y_res) else y_res
                    y_res = None if y_res is None else float(y_res)
                except Exception as e:
                    logger.opt(exception=e).warning(
                        "Failed to interprete tiff YResolution tag: {}", y_res
                    )
                    y_res = None

                unit_tag = page.tags.get("ResolutionUnit")
                unit = unit_tag.value if unit_tag else None

            ij_metadata = ImageJMetadata(**(tif.imagej_metadata or {}))
            ome_metadata = None
            if (
                suffixes[-2:] in ([".ome", ".tif"], [".ome", ".tiff"])
                or not ij_metadata
            ) and tif.ome_metadata is not None:
                try:
                    ome_metadata = create_ome_metadata_from_xml_string(tif.ome_metadata)
                except Exception as e:
                    logger.opt(exception=e).warning(
                        "Failed to parse OME-XML metadata from tif.ome_metadata"
                    )

            if ome_metadata is None:
                ome_metadata = OmeMetadata()

        finally:
            tif.close()

    t = Tensor.from_xarray(xr_data)

    # set physical scale
    x_scale = ome_metadata.PhysicalSizeX or (
        1 / x_res if x_res is not None and x_res != 0 else None
    )
    y_scale = ome_metadata.PhysicalSizeY or (
        1 / y_res if y_res is not None and y_res != 0 else x_scale
    )
    z_scale = ome_metadata.PhysicalSizeZ or ij_metadata.spacing or x_scale
    t_scale = (
        ome_metadata.TimeIncrement or ij_metadata.finterval or None
        if ij_metadata.fps is None
        else 1 / ij_metadata.fps
    )
    scale = {
        AxisId("x"): x_scale,
        AxisId("y"): y_scale,
        AxisId("z"): z_scale,
        AxisId("time"): t_scale,
    }
    t.set_physical_scale({k: v for k, v in scale.items() if k in t.dims})

    # set physical scale unit
    x_unit = ome_metadata.PhysicalSizeXUnit or ij_metadata.unit
    if x_unit is None and unit is not None:
        try:
            x_unit = tifffile.RESUNIT(unit).name.lower()
        except Exception as e:
            logger.opt(exception=e).warning(
                "Failed to interprete tiff ResolutionUnit tag: {}", unit
            )
            x_unit = None

    y_unit = ome_metadata.PhysicalSizeYUnit or ij_metadata.yunit or x_unit
    z_unit = ome_metadata.PhysicalSizeZUnit or ij_metadata.zunit or x_unit
    t_unit = ome_metadata.TimeIncrementUnit or "s"

    scale_unit = {
        AxisId("x"): x_unit,
        AxisId("y"): y_unit,
        AxisId("z"): z_unit,
        AxisId("time"): t_unit,
    }
    t.set_physical_scale_unit({k: v for k, v in scale_unit.items() if k in t.dims})

    # set channel names
    if ome_metadata.Channel is not None:
        t.channel_names = ome_metadata.Channel.Name

    return t


Suffix = str


def save_tensor(
    path: Path | str,
    tensor: Tensor,
    *,
    io_lib: IO_Lib | None = None,
    roi_and_tensor_shape: tuple[PerAxis[SliceInfo], PerAxis[int]] | None = None,
) -> None:
    try:
        output_path, extensions, internal_path = _interprete_tensor_source(path)
    except Exception:
        output_path, extensions, internal_path = _interprete_tensor_target(path)

    if isinstance(output_path, RootHttpUrl):
        raise NotImplementedError(
            "Saving tensors to HTTP URLs is not implemented. Please save to a local file path."
        )

    if (
        (v0_5.BATCH_AXIS_ID in tensor.dims)
        and tensor.tagged_shape.get(v0_5.BATCH_AXIS_ID, 0) == 1
        and (extensions is None or extensions[-1] != ".npy")
    ):
        # avoid saving tensors with a singleton batch dimension
        tensor = tensor.squeeze(v0_5.BATCH_AXIS_ID)

    if io_lib is None:
        try_io_libs = _select_io_libs_based_on_extension(extensions)
        exceptions: list[Exception] = []
        for try_io_lib in try_io_libs:
            try:
                return save_tensor(
                    path,
                    tensor,
                    io_lib=try_io_lib,
                    roi_and_tensor_shape=roi_and_tensor_shape,
                )
            except Exception as e:
                exceptions.append(e)
                logger.opt(exception=e).warning(
                    "Failed to save tensor with io_lib={}", try_io_lib
                )

        raise ExceptionGroup(
            "Failed to save tensor with any available io_lib", exceptions
        )
    elif io_lib == "bioio":
        return _save_tensor_bioio(
            output_path,
            tensor,
            internal_path=internal_path,
            roi_and_tensor_shape=roi_and_tensor_shape,
        )
    elif io_lib == "clearscale":
        return _save_tensor_clearscale(
            output_path,
            tensor,
            internal_path=internal_path,
            roi_and_tensor_shape=roi_and_tensor_shape,
        )
    elif io_lib == "imageio":
        if roi_and_tensor_shape is not None:
            raise NotImplementedError(
                "Saving a region of interest (roi) is not implemented for imageio."
            )
        return _save_tensor_imageio(output_path, tensor, internal_path=internal_path)
    elif io_lib == "numpy":
        if roi_and_tensor_shape is not None:
            raise NotImplementedError(
                "Saving a region of interest (roi) is not implemented for numpy."
            )
        if internal_path:
            raise NotImplementedError(
                "Saving to an internal path is not implemented for numpy."
            )
        return _save_tensor_numpy(output_path, tensor)
    elif io_lib == "tifffile":
        if roi_and_tensor_shape is not None:
            raise NotImplementedError(
                "Saving a region of interest (roi) is not implemented for tifffile."
            )
        if internal_path is not None:
            raise NotImplementedError(
                "Saving to an internal path is not implemented for tifffile."
            )

        return _save_tensor_tifffile(output_path, extensions, tensor)
    else:
        assert_never(io_lib)


def _save_tensor_bioio(
    path: Path | ZipPath | FtpUrl,
    tensor: Tensor,
    internal_path: PurePath | None,
    roi_and_tensor_shape: tuple[PerAxis[SliceInfo], PerAxis[int]] | None,
) -> None:
    """Save an image tensor using bioio"""

    import bioio_base.types

    logger.debug(
        "Saving tensor to {}{} with bioio",
        path,
        f"/{internal_path}" if internal_path else "",
    )
    bioio_axis_map = {
        "batch": "S",
        "channel": "C",
        "index": "S",
        "time": "T",
    }
    scale = tensor.get_physical_scale()

    if path.suffix.lower() == ".zarr":
        if internal_path and str(internal_path) != "0":
            raise NotImplementedError(
                "Saving to an internal path that is not '0' is not implemented for zarr files with io_lib='bioio'."
            )
        if roi_and_tensor_shape is None:
            tensor_shape = tensor.tagged_shape
        else:
            raise NotImplementedError(
                "Saving a region of interest (roi) is not implemented for saving zarr files with io_lib='bioio'."
            )

        import bioio_ome_zarr.writers

        axes_names = single_letter_dims_if_possible(tensor.dims)
        axes_types = [
            {"time": "time", "t": "time", "channel": "channel", "c": "channel"}.get(
                str(a).lower(), "space"
            )
            for a in tensor.dims
        ]

        channel_names = tensor.channel_names
        channel_colors = tensor.channel_colors

        channels = [
            bioio_ome_zarr.writers.Channel(
                label=f"channel{i}" if channel_names is None else channel_names[i],
                color="#FF0000" if channel_colors is None else channel_colors[i],
            )
            for i in range(tensor_shape.get(AxisId("channel"), 0))
        ]
        zf = int(os.getenv("ZARR_FORMAT", "3"))
        if zf not in (2, 3):
            raise NotImplementedError(
                f"ZARR_FORMAT={zf} is not supported. (expected 2 or 3)"
            )

        unit = tensor.get_physical_scale_unit()
        writer = bioio_ome_zarr.writers.OMEZarrWriter(
            store=path,
            level_shapes=[[tensor_shape[d] for d in tensor.dims]],
            dtype=tensor.dtype,
            zarr_format=zf,
            image_name=tensor.name or "Image",
            axes_names=list(axes_names),
            axes_types=axes_types,
            axes_units=[unit.get(a) for a in tensor.dims],
            physical_pixel_size=[scale.get(a, 1.0) for a in tensor.dims],
            channels=channels,
        )
        logger.info("OME-Zarr metadata: {}", writer.preview_metadata())
        writer.write_full_volume(tensor.data.data)
        return
    elif path.suffix.lower() in (".tif", ".tiff"):
        if roi_and_tensor_shape is not None:
            raise NotImplementedError(
                "Saving a region of interest (roi) is not implemented for saving tiff files with io_lib='bioio'."
            )

        from bioio_ome_tiff.writers import OmeTiffWriter

        save = OmeTiffWriter.save  # pyright: ignore[reportUnknownVariableType]
        bioio_axis_letters = "TCZYX"
    elif path.suffix.lower() in (".gif", ".mp4", ".mkv"):
        if roi_and_tensor_shape is not None:
            raise NotImplementedError(
                "Saving a region of interest (roi) is not implemented for saving video files with io_lib='bioio'."
            )

        from bioio_imageio.writers import TimeseriesWriter

        tensor = tensor.squeeze()
        save = TimeseriesWriter.save  # pyright: ignore[reportUnknownVariableType]
        bioio_axis_map["channel"] = "S"
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
        if roi_and_tensor_shape is not None:
            raise NotImplementedError(
                "Saving a region of interest (roi) is not implemented for saving 2d images with io_lib='bioio'."
            )

        from bioio_imageio.writers import TwoDWriter

        tensor = tensor.squeeze()
        save = TwoDWriter.save  # pyright: ignore[reportUnknownVariableType]
        # channel dim denoted as 'S'
        bioio_axis_map["channel"] = "S"
        bioio_axis_letters = "YXS"
    else:
        raise RuntimeError(
            f"Failed to identify a suitable writer for {path} with suffix={path.suffix}."
        )

    bioio_data = tensor.data
    dim_map = {
        d: bioio_axis_map.get(str(d).lower(), str(d)[:1].upper()) for d in tensor.dims
    }

    bioio_data = bioio_data.rename(dim_map)  # pyright: ignore[reportUnknownVariableType]
    assert isinstance(bioio_data, xr.DataArray)

    if not all(str(d) in bioio_axis_letters for d in bioio_data.dims):
        logger.warning(
            "Attempting to save tensor with dimensions {} not in '{}' using bioio.",
            bioio_data.dims,
            bioio_axis_letters,
        )

    dim_order = "".join(str(d) for d in bioio_data.dims)

    image_name = None if internal_path is None else str(internal_path)
    physical_scale = tensor.get_physical_scale()
    save(
        bioio_data.data,
        path,
        dim_order=dim_order,
        image_name=image_name,
        channel_names=tensor.channel_names,
        channel_colors=None
        if (channel_colors := tensor.channel_colors) is None
        else [list(hex_to_rgb(c)) for c in channel_colors],
        physical_pixel_sizes=bioio_base.types.PhysicalPixelSizes(
            physical_scale.get(AxisId("z")),
            physical_scale.get(AxisId("y")),
            physical_scale.get(AxisId("x")),
        ),
    )


def _save_tensor_clearscale(
    path: Path | ZipPath | FtpUrl,
    tensor: Tensor,
    internal_path: PurePath | None,
    roi_and_tensor_shape: tuple[PerAxis[SliceInfo], PerAxis[int]] | None,
) -> None:
    import clearscale
    import dask.array
    import zarr

    logger.debug("Saving tensor to {} with clearscale", path)

    if roi_and_tensor_shape is None:
        roi = (slice(None),) * len(tensor.dims)
        tensor_shape = tensor.tagged_shape
    else:
        roi_map, tensor_shape = roi_and_tensor_shape
        roi = tuple(
            slice(None) if (s := roi_map.get(d)) is None else slice(s.start, s.stop)
            for d in tensor.dims
        )

    ZARR_FORMAT: int = int(os.getenv("ZARR_FORMAT", "3"))
    SCALE_KEY = "s0" if internal_path is None else str(internal_path)
    # Prefer single dimension names for compatibility with bioio
    dim_map = dict(zip(tensor.dims, single_letter_dims_if_possible(tensor.dims)))

    multiscale = clearscale.Multiscale(
        {
            SCALE_KEY: clearscale.Scale(
                shape=clearscale.Shape(**dict(zip(dim_map.values(), tensor.shape))),
                pixel_size={
                    dim_map[k]: v for k, v in tensor.get_physical_scale().items()
                }
                or None,
                unit={
                    dim_map[k]: v for k, v in tensor.get_physical_scale_unit().items()
                }
                or None,
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

    create_array_kwargs = _CreateArrayKwargs(
        name=SCALE_KEY,
        overwrite=True,
        dimension_names=None if ZARR_FORMAT == 2 else tuple(dim_map.values()),
        shards=None if ZARR_FORMAT == 2 else "auto",
    )
    zarr_array = zarr_group.create_array(
        **create_array_kwargs,
        shape=[tensor_shape[d] for d in tensor.dims],
        dtype=tensor.dtype,
        write_data=True,
    )
    xr_array = tensor.data
    if dm := {k: v for k, v in dim_map.items() if k != v}:
        xr_array = xr_array.rename(dm)  # pyright: ignore[reportUnknownVariableType]
        assert isinstance(xr_array, xr.DataArray)

    if isinstance(xr_array.data, np.ndarray):
        zarr_array[roi] = xr_array.data
    elif isinstance(xr_array.data, dask.array.Array):
        dask.array.to_zarr(xr_array.data, zarr_array, compute=True, region=roi)
    else:
        assert_never(xr_array.data)


def _save_tensor_imageio(
    path: Path | ZipPath | FtpUrl, tensor: Tensor, internal_path: PurePath | None
) -> None:
    if internal_path is not None:
        raise NotImplementedError(
            "Saving tensors to subdirectories is not implemented for imageio. Please save to a file path without internal path."
        )

    if isinstance(path, ZipPath):
        raise NotImplementedError(
            "Saving tensors to zip files is not implemented for imageio. Please save to an unzipped file path."
        )

    logger.debug("Saving tensor to {} with imageio", path)
    if not isinstance(path, FtpUrl):
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


def _save_tensor_numpy(path: Path | ZipPath | FtpUrl, tensor: Tensor) -> None:
    if isinstance(path, FtpUrl):
        raise NotImplementedError(
            "Saving numpy arrays to FTP URLs is not implemented. Please save to a local file path instead."
        )

    logger.debug("Saving tensor to {} with numpy", path)

    if isinstance(path, ZipPath):
        folder = path.filename.parent
    else:
        folder = path.parent

    folder.mkdir(exist_ok=True, parents=True)

    save_array(path, tensor.to_numpy())


def _save_tensor_tifffile(
    path: Path | ZipPath | FtpUrl, extensions: list[str] | None, tensor: Tensor
) -> None:
    if isinstance(path, FtpUrl):
        raise NotImplementedError(
            "Saving tiff files to FTP URLs is not implemented. Please save to a local file path instead."
        )

    import tifffile

    logger.trace("tensor dims {}", tensor.dims)
    tensor = tensor.transpose("TZCYX")  # tifffile expects TZCYX order
    logger.trace("tensor dims transposed {}", tensor.dims)
    if len(tensor.dims) > 5:
        logger.warning(
            "Saving tensor with more than 5 dimensions ({}) will likely fail."
        )

    logger.debug("Saving tensor to {} with tifffile", path)

    scale = tensor.get_physical_scale()
    x_scale = scale.get(AxisId("x"))
    y_scale = scale.get(AxisId("y"))
    z_scale = scale.get(AxisId("z"))
    t_scale = scale.get(AxisId("time"))
    if x_scale is not None and y_scale is not None:
        resolution = (1 / x_scale, 1 / y_scale)
    else:
        resolution = None

    scale_unit = tensor.get_physical_scale_unit()
    x_unit = scale_unit.get(AxisId("x"))
    y_unit = scale_unit.get(AxisId("y"))
    z_unit = scale_unit.get(AxisId("z"))
    t_unit = scale_unit.get(AxisId("time"))
    if x_unit == "inch":
        resolution_unit = tifffile.RESUNIT.INCH
    elif x_unit in ("centimeter", "cm"):
        resolution_unit = tifffile.RESUNIT.CENTIMETER
    elif x_unit in ("millimeter", "mm"):
        resolution_unit = tifffile.RESUNIT.MILLIMETER
    elif x_unit in ("micrometer", "um", "µm"):
        resolution_unit = tifffile.RESUNIT.MICROMETER
    else:
        resolution_unit = None

    if isinstance(path, ZipPath):
        folder = path.filename.parent
    else:
        folder = path.parent

    folder.mkdir(exist_ok=True, parents=True)

    if extensions is not None and extensions[-2:] in (
        [".ome", ".tif"],
        [".ome", ".tiff"],
    ):
        metadata = OmeMetadata(
            # "DimensionOrder": "TZCYX",  # [::-1]?? # TODO: check if this is needed
            TimeIncrement=t_scale,
            TimeIncrementUnit=t_unit,
            PhysicalSizeX=x_scale,
            PhysicalSizeXUnit=x_unit,
            PhysicalSizeY=y_scale,
            PhysicalSizeYUnit=y_unit,
            PhysicalSizeZ=z_scale,
            PhysicalSizeZUnit=z_unit,
            Channel=None
            if tensor.channel_names is None
            else OmeChannelMetadata(Name=tensor.channel_names),
        )
        kind = "ome"
    else:
        if not t_unit or t_scale is None:
            pass
        elif isinstance(t_scale, (int, float)):
            if t_unit in ("nanosecond", "nanosecond(s)", "ns"):
                t_scale /= 1_000_000_000
            elif t_unit in ("microsecond", "microsecond(s)", "us", "µsec", "µs"):
                t_scale /= 1_000_000
            elif t_unit in ("millisecond", "millisecond(s)", "ms"):
                t_scale /= 1_000
            elif t_unit in ("minute", "minute(s)", "min"):
                t_scale *= 60
            elif t_unit in ("hour", "hour(s)", "h"):
                t_scale *= 3_600
            elif t_unit in ("day", "day(s)", "d"):
                t_scale *= 86_400
            elif t_unit in ("week", "week(s)", "wk"):
                t_scale *= 604_800
            elif t_unit in ("month", "month(s)", "mo"):
                t_scale *= 2_629_746  # average month length in seconds
            elif t_unit in ("year", "year(s)", "yr"):
                t_scale *= 31_556_952  # average year length in seconds
            elif t_unit not in ("second", "s"):
                logger.warning(
                    "Unknown time unit '{}' encountered while saving an imagej tiff with tifffile",
                    t_unit,
                )
        else:
            logger.warning(
                "Failed to convert time interval '{}' of type {} (with time unit '{}') to seconds while saving an imagej tiff with tifffile",
                t_scale,
                type(t_scale),
                t_unit,
            )

        metadata = ImageJMetadata(
            unit=x_unit,
            yunit=y_unit,
            zunit=z_unit,
            spacing=z_scale,
            finterval=t_scale,
            fps=None if not isinstance(t_scale, (int, float)) else 1 / t_scale,
        )
        kind = "imagej"

    if isinstance(path, ZipPath):
        ctxt = path.open("wb")
    else:
        ctxt = nullcontext(path)

    with ctxt as f:
        assert not isinstance(f, TextIOWrapper)
        _ = tifffile.imwrite(
            f,
            tensor.data,
            kind=kind,
            software="bioimageio.core",
            resolution=resolution,
            metadata={
                k: v
                for k, v in metadata.model_dump(mode="json", exclude_none=True).items()
            },
            resolutionunit=resolution_unit,
        )


class _CreateArrayKwargs(TypedDict):
    name: str
    overwrite: bool
    dimension_names: tuple[str, ...] | None
    shards: Literal["auto"] | tuple[int, ...] | None


_zarr_url_adapter: TypeAdapter[ZarrUrl] = TypeAdapter(ZarrUrl)


def _interprete_tensor_target(
    source: PermissiveFileSource | ZarrSource | ZipPath | FileDescr,
):
    if isinstance(source, FileDescr):
        source = source.source

    if isinstance(source, (str, pydantic.AnyUrl)):
        try:
            source = _zarr_url_adapter.validate_python(str(source))
        except Exception:
            if (
                str(source).startswith("http://")
                or str(source).startswith("https://")
                or str(source).startswith("ftp://")
            ):
                raise
            else:
                source = Path(str(source))

    if isinstance(source, (RelativeFilePath, RelativeDirectory)):
        source = source.absolute()

    return _get_file_source_extension_and_internal_path(source)


def _interprete_tensor_source(
    source: PermissiveFileSource | ZarrSource | ZipPath | FileDescr,
):

    if isinstance(source, FileDescr):
        source = source.source

    if isinstance(source, (str, pydantic.AnyUrl)):
        source = interprete_file_source(source, allow_zarr=True)

    if isinstance(source, (RelativeFilePath, RelativeDirectory)):
        source = source.absolute()

    return _get_file_source_extension_and_internal_path(source)


def _get_file_source_extension_and_internal_path(
    source: Path | ZipPath | RootHttpUrl | FtpUrl,
):
    internal_path = None
    if isinstance(source, (HttpUrl, RootHttpUrl, FtpUrl)):
        original_source_path = PosixPath(source.path or "")
        parents: Iterable[FtpUrl | RootHttpUrl | Path | ZipPath] = source.parents
    elif isinstance(source, ZipPath):
        original_source_path = PosixPath(source.filename)
        parents = []
        p = source
        while p != p.parent:
            parents.append(p.parent)
            p = p.parent
    else:
        original_source_path = PosixPath(source)
        parents = source.parents

    file_source = source
    for parent in chain([source], parents):
        extensions = [s.lower() for s in parent.suffixes]
        if extensions:
            file_source = parent
            if isinstance(parent, (RootHttpUrl, FtpUrl)):
                parent_path = PosixPath(parent.path or "")
            elif isinstance(parent, ZipPath):
                parent_path = PosixPath(parent.filename)
            else:
                parent_path = PosixPath(parent)

            internal_path = original_source_path.relative_to(parent_path)
            if internal_path == PosixPath("."):
                internal_path = None

            break
    else:
        extensions = None

    return file_source, extensions, internal_path


def save_sample(
    path: Path | str | PerMember[Path | str], sample: Sample | Iterable[SampleBlock]
) -> None:
    """Save a **sample** to a **path** pattern
    or all sample members in the **path** mapping.

    If **path** is a pathlib.Path or a string and the **sample** has multiple members,
    **path** it must contain `{member_id}` (or `{input_id}` or `{output_id}`).

    (Each) **path** may contain `{sample_id}` to be formatted with the **sample** object.
    """

    if isinstance(sample, Sample):
        first_block = sample
        remaining_blocks = ()
    else:
        remaining_blocks = iter(sample)
        first_block = next(remaining_blocks)

    if not isinstance(path, Mapping):
        if len(first_block.members) < 2 or any(
            m in str(path) for m in ("{member_id}", "{input_id}", "{output_id}")
        ):
            path = {m: path for m in first_block.members}
        else:
            raise ValueError(
                f"path {path} must contain '{{member_id}}' for sample with multiple members {list(first_block.members)}."
            )

    for sample_block in chain([first_block], remaining_blocks):
        for m, p in path.items():
            if isinstance(sample_block, Sample):
                sample_id = sample_block.id
                t = sample_block.members[m]
                roi_and_tensor_shape = None
            else:
                sample_id = sample_block.sample_id
                roi_and_tensor_shape = (
                    sample_block.blocks[m].inner_slice,
                    sample_block.sample_shape[m],
                )
                t = sample_block.blocks[m].inner_data

            p_formatted = Path(
                str(p).format(sample_id=sample_id, member_id=m, input_id=m, output_id=m)
            )
            save_tensor(p_formatted, t, roi_and_tensor_shape=roi_and_tensor_shape)


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
