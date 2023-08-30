from __future__ import annotations

import collections.abc
import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Dict, Literal, NamedTuple, Optional, Sequence, Tuple, Union, cast
from zipfile import ZIP_DEFLATED

import pooch
from bioimageio.spec import ResourceDescription, load_description
from bioimageio.spec._internal.base_nodes import ResourceDescriptionBase
from bioimageio.spec._internal.constants import DISCOVER, ERROR, LATEST
from bioimageio.spec.description import dump_description
from bioimageio.spec.model.v0_4 import WeightsFormat
from bioimageio.spec.package import extract_file_name, get_resource_package_content
from bioimageio.spec.summary import ValidationSummary
from bioimageio.spec.types import (
    FileName,
    RawStringMapping,
    RawValue,
    RelativeFilePath,
    ValidationContext,
    WarningLevel,
)
from pydantic import AnyUrl, DirectoryPath, FilePath, HttpUrl, TypeAdapter
from ruamel.yaml import YAML

from bioimageio.core._internal.utils import get_parent_url, write_zip

yaml = YAML(typ="safe")

FileSource = Union[HttpUrl, FilePath]


class ReadRdf(NamedTuple):
    content: RawStringMapping
    root: Union[HttpUrl, DirectoryPath]
    file_name: str


def load_description_from_file(
    source: Union[FileSource, str],
    /,
    *,
    warning_level: WarningLevel = ERROR,
    format_version: Union[Literal["discover"], Literal["latest"], str] = DISCOVER,
) -> Tuple[Optional[ResourceDescription], ValidationSummary]:
    rdf = read_rdf(source)
    return load_description(
        rdf.content,
        context=ValidationContext(root=rdf.root, file_name=rdf.file_name, warning_level=warning_level),
        format_version=format_version,
    )


def read_rdf(
    source: Union[FileSource, str], /, *, known_hash: Optional[str] = None, encoding: Optional[str] = None
) -> ReadRdf:
    if isinstance(source, str):
        source = TypeAdapter(FileSource).validate_python(source)

    src_msg = str(source)
    if isinstance(source, AnyUrl):
        cached_source: FilePath = Path(pooch.retrieve(url=str(source), known_hash=known_hash))  # type: ignore
        src_msg += f" cached at {cached_source}"
        local_source = cached_source
        root: Union[HttpUrl, DirectoryPath] = get_parent_url(source)
    else:
        local_source = source
        root = source.parent

    with local_source.open(encoding=encoding) as f:
        content: RawValue = yaml.load(f)

    if not isinstance(content, collections.abc.Mapping):
        raise TypeError(f"Expected RDF content to be a mapping, but got '{type(content)}'.")

    if non_string_keys := [k for k in content if not isinstance(k, str)]:
        raise TypeError(f"Got non-string keys {non_string_keys} in {src_msg}")

    return ReadRdf(
        content=cast(RawStringMapping, content),
        root=root,
        file_name=extract_file_name(source),
    )


def resolve_source(
    source: Union[HttpUrl, FilePath, RelativeFilePath, str],
    /,
    *,
    known_hash: Optional[str] = None,
    root: Union[DirectoryPath, AnyUrl, None] = None,
) -> FilePath:
    if isinstance(source, str):
        source = TypeAdapter(Union[HttpUrl, FilePath, RelativeFilePath]).validate_python(source)

    if isinstance(source, RelativeFilePath):
        if root is None:
            raise ValueError(f"Cannot resolve relative file path '{source}' without root.")

        source = source.get_absolute(root)

    if isinstance(source, AnyUrl):
        source = Path(pooch.retrieve(source, known_hash=known_hash))  # type: ignore

    return source


def dump_description_to_file(rd: Union[ResourceDescription, RawStringMapping], /, file_path: Path):
    if isinstance(rd, ResourceDescriptionBase):
        content = dump_description(rd)
    else:
        content = rd

    with file_path.open("w", encoding="utf-8") as f:
        yaml.dump(content, f)


def load_description_from_file_and_validate(
    rdf_source: Union[FileSource, str],
    /,
    *,
    warning_level: WarningLevel = ERROR,
) -> Tuple[Optional[ResourceDescription], ValidationSummary]:
    rdf = read_rdf(rdf_source)
    return load_description_and_validate(
        rdf.content, context=ValidationContext(root=rdf.root, file_name=rdf.file_name, warning_level=warning_level)
    )


def load_description_and_validate(
    rdf_content: RawStringMapping,
    /,
    *,
    context: Optional[ValidationContext] = None,
) -> Tuple[Optional[ResourceDescription], ValidationSummary]:
    """load and validate a BioImage.IO description from the content of a resource description file (RDF)"""
    rd, summary = load_description(rdf_content, context=context, format_version=LATEST)
    # todo: add validation
    return rd, summary


def validate(
    rdf_content: RawStringMapping,
    /,
    *,
    context: Optional[ValidationContext] = None,
) -> ValidationSummary:
    _rd, summary = load_description_and_validate(rdf_content, context=context)
    return summary


def validate_rdf(rdf_source: Union[FileSource, str], /, *, warning_level: WarningLevel = ERROR) -> ValidationSummary:
    _rd, summary = load_description_from_file_and_validate(rdf_source, warning_level=warning_level)
    return summary


def validate_rdf_format(
    rdf_source: Union[FileSource, str], /, *, warning_level: WarningLevel = ERROR
) -> ValidationSummary:
    _rd, summary = load_description_from_file(rdf_source, warning_level=warning_level)
    return summary


def prepare_resource_package(
    rd: ResourceDescription,
    /,
    *,
    root: Union[AnyUrl, DirectoryPath],
    output_folder: DirectoryPath,
    weights_priority_order: Optional[Sequence[WeightsFormat]] = None,
) -> Dict[FileName, FilePath]:
    """Prepare to package a resource description; downloads all required files.

    Args:
        rd: bioimage.io resource description
        root: URL or path to resolve relative file paths in `rd`
        weights_priority_order: If given only the first weights format present in the model is included.
                                If none of the prioritized weights formats is found all are included.
    """
    package_content = get_resource_package_content(rd, weights_priority_order=weights_priority_order)

    output_folder.mkdir(parents=True, exist_ok=True)
    local_package_content: Dict[FileName, FilePath] = {}
    for k, v in package_content.items():
        in_package_path = output_folder / k
        if isinstance(v, RelativeFilePath):
            v = v.get_absolute(root)

        if isinstance(v, AnyUrl):
            v = resolve_source(v, root=root)

        if isinstance(v, Path):
            shutil.copy(str(v), str(in_package_path))
        else:
            assert isinstance(v, collections.abc.Mapping)
            dump_description_to_file(v, in_package_path)

        local_package_content[k] = in_package_path

    return local_package_content


def write_package(
    rd: Union[ResourceDescription, FileSource, str],
    /,
    *,
    root: Union[AnyUrl, DirectoryPath] = Path(),
    compression: int = ZIP_DEFLATED,
    compression_level: int = 1,
    output_path: Optional[os.PathLike[str]] = None,
    weights_priority_order: Optional[  # model only
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
    ] = None,
) -> FilePath:
    """Package a bioimage.io resource as a zip file.

    Args:
        rd: bioimage.io resource description
        root: reference for any relative file paths in the bioimage.io resource description
        compression: The numeric constant of compression method.
        compression_level: Compression level to use when writing files to the archive.
                           See https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile
        output_path: file path to write package to
        weights_priority_order: If given only the first weights format present in the model is included.
                                If none of the prioritized weights formats is found all are included.

    Returns:
        path to zipped bioimage.io package in BIOIMAGEIO_CACHE_PATH or 'output_path'
    """
    if isinstance(rd, (AnyUrl, os.PathLike, str)):
        rd_, summary = load_description_from_file(rd)
        if rd_ is None:
            raise ValueError(summary.format())
        else:
            rd = rd_

    with TemporaryDirectory() as tmp_dir:
        package_content = prepare_resource_package(
            rd,
            root=root,
            output_folder=Path(tmp_dir),
            weights_priority_order=weights_priority_order,
        )

    if output_path is None:
        output_path = Path(NamedTemporaryFile(suffix=".bioimageio.zip", delete=False).name)
    else:
        output_path = Path(output_path)

    write_zip(output_path, package_content, compression=compression, compression_level=compression_level)
    return output_path
