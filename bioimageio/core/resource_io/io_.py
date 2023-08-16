import os
import pathlib
from copy import deepcopy
from tempfile import TemporaryDirectory
from typing import Dict, Literal, Optional, Sequence, Union
from zipfile import ZIP_DEFLATED, ZipFile

from bioimageio.spec._internal._constants import DISCOVER
from bioimageio.spec import load_raw_resource_description
from bioimageio.spec.shared import raw_nodes
from bioimageio.spec.shared.common import (
    BIOIMAGEIO_CACHE_PATH,
    BIOIMAGEIO_USE_CACHE,
    get_class_name_from_type,
    no_cache_tmp_list,
)
from bioimageio.spec import ResourceDescription
from . import nodes
from .utils import resolve_raw_node, resolve_source

serialize_raw_resource_description = spec.io_.serialize_raw_resource_description
save_raw_resource_description = spec.io_.save_raw_resource_description


def get_local_resource_package_content(
    source: ResourceDescription,
    weights_priority_order: Optional[
        Sequence[
            Literal[
                "keras_hdf5",
                "onnx",
                "pytorch_state_dict",
                "tensorflow_js",
                "tensorflow_saved_model_bundle",
                "torchscript",
            ]
        ]
    ],
    format_version: Union[Literal["discover"], Literal["latest"], str] = DISCOVER,
) -> Dict[str, Union[pathlib.Path, str]]:
    """

    Args:
        source: raw resource description
        weights_priority_order: If given only the first weights format present in the model is included.
                                If none of the prioritized weights formats is found all are included.
        update_to_format: update resource to specific major.minor format version; ignoring patch version.

    Returns:
        Package content of local file paths or text content keyed by file names.

    """
    rd = load_resource_description(source, update_to_format=update_to_format)
    package_content = spec.get_resource_package_content(raw_rd, weights_priority_order=weights_priority_order)

    local_package_content = {}
    for k, v in package_content.items():
        if isinstance(v, raw_nodes.URI):
            v = resolve_source(v, raw_rd.root_path)
        elif isinstance(v, pathlib.Path):
            v = raw_rd.root_path / v

        local_package_content[k] = v

    return local_package_content


def export_resource_package(
    source: Union[RawResourceDescription, os.PathLike, str, dict, raw_nodes.URI],
    *,
    compression: int = ZIP_DEFLATED,
    compression_level: int = 1,
    output_path: Optional[os.PathLike] = None,
    update_to_format: Optional[str] = None,
    weights_priority_order: Optional[Sequence[Union[str]]] = None,
) -> pathlib.Path:
    """Package a bioimage.io resource as a zip file.

    Args:
        source: raw resource description, path, URI or raw data as dict
        compression: The numeric constant of compression method.
        compression_level: Compression level to use when writing files to the archive.
                           See https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile
        output_path: file path to write package to
        update_to_format: update resource to specific "major.minor" or "latest" format version; ignoring patch version.
        weights_priority_order: If given only the first weights format present in the model is included.
                                If none of the prioritized weights formats is found all are included.

    Returns:
        path to zipped bioimage.io package in BIOIMAGEIO_CACHE_PATH or 'output_path'
    """
    raw_rd = load_raw_resource_description(source, update_to_format=update_to_format)
    package_content = get_local_resource_package_content(
        raw_rd, weights_priority_order, update_to_format=update_to_format
    )
    if output_path is None:
        package_path = _get_tmp_package_path(raw_rd, weights_priority_order)
    else:
        package_path = output_path

    make_zip(package_path, package_content, compression=compression, compression_level=compression_level)
    return package_path


def _get_package_base_name(raw_rd: RawResourceDescription, weights_priority_order: Optional[Sequence[str]]) -> str:
    package_file_name = raw_rd.name
    if raw_rd.version is not missing:
        package_file_name += f"_{raw_rd.version}"

    package_file_name = package_file_name.replace(" ", "_").replace(".", "_")

    return package_file_name


def _get_tmp_package_path(raw_rd: RawResourceDescription, weights_priority_order: Optional[Sequence[str]]):
    if BIOIMAGEIO_USE_CACHE:
        package_file_name = _get_package_base_name(raw_rd, weights_priority_order)
        cache_folder = BIOIMAGEIO_CACHE_PATH / "packages"
        cache_folder.mkdir(exist_ok=True, parents=True)

        package_path = (cache_folder / package_file_name).with_suffix(".zip")
        max_cached_packages_with_same_name = 100
        for p in range(max_cached_packages_with_same_name):
            if package_path.exists():
                package_path = (cache_folder / f"{package_file_name}p{p}").with_suffix(".zip")
            else:
                break
        else:
            raise FileExistsError(
                f"Already caching {max_cached_packages_with_same_name} versions of {cache_folder / package_file_name}!"
            )
    else:
        tmp_dir = TemporaryDirectory()
        no_cache_tmp_list.append(tmp_dir)
        package_path = pathlib.Path(tmp_dir.name) / "file"

    return package_path


def make_zip(
    path: os.PathLike, content: Dict[str, Union[str, pathlib.Path]], *, compression: int, compression_level: int
) -> None:
    """Write a zip archive.

    Args:
        path: output path to write to.
        content: dict with archive names and local file paths or strings for text files.
        compression: The numeric constant of compression method.
        compression_level: Compression level to use when writing files to the archive.
                           See https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile

    """
    with ZipFile(path, "w", compression=compression, compresslevel=compression_level) as myzip:
        for arc_name, file_or_str_content in content.items():
            if isinstance(file_or_str_content, str):
                myzip.writestr(arc_name, file_or_str_content)
            else:
                myzip.write(file_or_str_content, arcname=arc_name)
