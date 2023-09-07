from __future__ import annotations

import collections.abc
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Literal, NamedTuple, Optional, Sequence, Tuple, Union, cast
from zipfile import ZIP_DEFLATED, ZipFile, is_zipfile

import pooch
from bioimageio.spec import ResourceDescription
from bioimageio.spec import load_description as load_description_from_content
from bioimageio.spec._internal.base_nodes import ResourceDescriptionBase
from bioimageio.spec._internal.constants import DISCOVER, LATEST
from bioimageio.spec._internal.types import FileName, RdfContent, RelativeFilePath, ValidationContext, YamlValue
from bioimageio.spec.description import dump_description
from bioimageio.spec.model.v0_4 import WeightsFormat
from bioimageio.spec.package import extract_file_name, get_resource_package_content
from bioimageio.spec.summary import ValidationSummary
from pydantic import AnyUrl, DirectoryPath, FilePath, HttpUrl, TypeAdapter
from ruamel.yaml import YAML

from bioimageio.core._internal.utils import get_parent_url, write_zip

yaml = YAML(typ="safe")

StrictFileSource = Union[HttpUrl, FilePath]
FileSource = Union[StrictFileSource, str]
StrictRdfSource = Union[StrictFileSource, RdfContent, ResourceDescription]
RdfSource = Union[StrictRdfSource, str]


class RawRdf(NamedTuple):
    content: RdfContent
    root: Union[HttpUrl, DirectoryPath]
    file_name: str


def load_description(
    rdf_source: RdfSource,
    /,
    *,
    context: Optional[ValidationContext] = None,
    format_version: Union[Literal["discover"], Literal["latest"], str] = DISCOVER,
) -> Tuple[Optional[ResourceDescription], ValidationSummary]:
    context = context or ValidationContext()
    rdf_content = _get_rdf_content_and_update_context(rdf_source, context)
    return load_description_from_content(
        rdf_content,
        context=context,
        format_version=format_version,
    )


LEGACY_RDF_NAME = "rdf.yaml"


def read_rdf_content(
    rdf_source: FileSource,
    /,
    *,
    known_hash: Optional[str] = None,
    rdf_encoding: str = "utf-8",
) -> RawRdf:
    class FileSourceInterpreter(BaseModel):
        source: StrictFileSource

    rdf_source = FileSourceInterpreter(source=rdf_source).source

    if isinstance(rdf_source, AnyUrl):
        _ls: Any = pooch.retrieve(url=str(rdf_source), known_hash=known_hash)
        local_source = Path(_ls)
        root: Union[HttpUrl, DirectoryPath] = get_parent_url(rdf_source)
    else:
        local_source = rdf_source
        root = rdf_source.parent

    if is_zipfile(local_source):
        out_path = local_source.with_suffix(local_source.suffix + ".unzip")
        with ZipFile(local_source, "r") as f:
            rdfs = [fname for fname in f.namelist() if fname.endswith(".bioimageio.yaml")]
            if len(rdfs) > 1:
                raise ValueError(f"Multiple RDFs in one package not yet supported (found {rdfs}).")
            elif len(rdfs) == 1:
                rdf_file_name = rdfs[0]
            elif LEGACY_RDF_NAME in f.namelist():
                rdf_file_name = LEGACY_RDF_NAME
            else:
                raise ValueError(
                    f"No RDF found in {local_source}. (Looking for any '*.bioimageio.yaml' file or an 'rdf.yaml' file)."
                )

            f.extractall(out_path)
            local_source = out_path / rdf_file_name

    with local_source.open(encoding=rdf_encoding) as f:
        content: YamlValue = yaml.load(f)

    if not isinstance(content, collections.abc.Mapping):
        raise TypeError(f"Expected RDF content to be a mapping, but got '{type(content)}'.")

    return RawRdf(
        content=cast(RdfContent, content),
        root=root,
        file_name=extract_file_name(rdf_source),
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
        _s: Any = pooch.retrieve(str(source), known_hash=known_hash)
        source = Path(_s)

    return source


def write_description(rd: Union[ResourceDescription, RdfContent], /, file_path: FilePath):
    if isinstance(rd, ResourceDescriptionBase):
        content = dump_description(rd)
    else:
        content = rd

    with file_path.open("w", encoding="utf-8") as f:
        yaml.dump(content, f)


def load_description_and_validate(
    rdf_source: RdfSource,
    /,
    *,
    context: Optional[ValidationContext] = None,
) -> Tuple[Optional[ResourceDescription], ValidationSummary]:
    """load and validate a BioImage.IO description from the content of a resource description file (RDF)"""
    context = context or ValidationContext()
    rdf_content = _get_rdf_content_and_update_context(rdf_source, context)
    rd, summary = load_description_from_content(rdf_content, context=context, format_version=LATEST)
    # todo: add dynamic validation
    return rd, summary


# def _get_default_io_context(context: Union[ValidationContext, CompleteValidationContext, None]) -> Union[ValidationContext, CompleteValidationContext]:
#     if context is None:
#         context = ValidationContext()

#     if "warning_level" not in context:
#         context["warning_level"] = INFO

#     return context


def _get_rdf_content_and_update_context(rdf_source: RdfSource, context: ValidationContext) -> RdfContent:
    class RdfSourceInterpreter(BaseModel):
        source: RdfSource

    rdf_source = RdfSourceInterpreter(source=rdf_source).source

    if isinstance(rdf_source, (AnyUrl, Path, str)):
        rdf = read_rdf_content(rdf_source)
        rdf_source = rdf.content
        context.root = rdf.root
        context.file_name = rdf.file_name
    elif isinstance(rdf_source, ResourceDescriptionBase):
        rdf_source = dump_description(rdf_source, exclude_unset=False)

    return rdf_source


def _get_description_and_update_context(rdf_source: RdfSource, context: ValidationContext) -> ResourceDescription:
    if not isinstance(rdf_source, ResourceDescriptionBase):
        descr, summary = load_description(rdf_source, context=context)
        if descr is None:
            rdf_source_msg = (
                f"{{name={rdf_source.get('name', 'missing'), ...}}})"
                if isinstance(rdf_source, collections.abc.Mapping)
                else rdf_source
            )
            raise ValueError(f"Failed to load {rdf_source_msg}:\n{summary.format()}")
        rdf_source = descr

    return rdf_source


def validate(
    rdf_source: RdfSource,
    /,
    *,
    context: Optional[ValidationContext] = None,
) -> ValidationSummary:
    _rd, summary = load_description_and_validate(rdf_source, context=context)
    return summary


def validate_format_only(
    rdf_source: Union[ResourceDescription, RdfContent, FileSource], context: Optional[ValidationContext] = None
) -> ValidationSummary:
    _rd, summary = load_description(rdf_source, context=context)
    return summary


def prepare_resource_package(
    rdf_source: RdfSource,
    /,
    *,
    context: Optional[ValidationContext] = None,
    weights_priority_order: Optional[Sequence[WeightsFormat]] = None,
) -> Dict[FileName, Union[FilePath, RdfContent]]:
    """Prepare to package a resource description; downloads all required files.

    Args:
        rdf_source: A bioimage.io resource description (as file, raw YAML content or description class)
        context: validation context
        weights_priority_order: If given only the first weights format present in the model is included.
                                If none of the prioritized weights formats is found all are included.
    """
    context = context or ValidationContext()
    rd = _get_description_and_update_context(rdf_source, context)
    package_content = get_resource_package_content(rd, weights_priority_order=weights_priority_order)

    local_package_content: Dict[FileName, Union[FilePath, RdfContent]] = {}
    for k, v in package_content.items():
        if not isinstance(v, collections.abc.Mapping):
            v = resolve_source(v, root=context.root)

        local_package_content[k] = v

    return local_package_content

    # output_folder.mkdir(parents=True, exist_ok=True)


def write_package(
    rdf_source: RdfSource,
    /,
    *,
    context: Optional[ValidationContext] = None,
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
        context:
        compression: The numeric constant of compression method.
        compression_level: Compression level to use when writing files to the archive.
                           See https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile
        output_path: file path to write package to
        weights_priority_order: If given only the first weights format present in the model is included.
                                If none of the prioritized weights formats is found all are included.

    Returns:
        path to zipped bioimage.io package in BIOIMAGEIO_CACHE_PATH or 'output_path'
    """
    package_content = prepare_resource_package(
        rdf_source,
        context=context,
        weights_priority_order=weights_priority_order,
    )
    if output_path is None:
        output_path = Path(NamedTemporaryFile(suffix=".bioimageio.zip", delete=False).name)
    else:
        output_path = Path(output_path)

    write_zip(output_path, package_content, compression=compression, compression_level=compression_level)
    return output_path
