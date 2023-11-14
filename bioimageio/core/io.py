from __future__ import annotations

import collections.abc
import io
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, TextIO, TypedDict, Union, cast
from zipfile import ZIP_DEFLATED, ZipFile, is_zipfile

import pooch
from pydantic import AnyUrl, DirectoryPath, FilePath, HttpUrl, TypeAdapter
from ruamel.yaml import YAML
from typing_extensions import NotRequired, Unpack

from bioimageio.spec import ResourceDescription
from bioimageio.spec import load_description as load_description
from bioimageio.spec._internal.base_nodes import ResourceDescriptionBase
from bioimageio.spec._internal.constants import DISCOVER
from bioimageio.spec._internal.types import FileName, RdfContent, RelativeFilePath, Sha256, ValidationContext, YamlValue
from bioimageio.spec.description import InvalidDescription, dump_description
from bioimageio.spec.model.v0_4 import WeightsFormat
from bioimageio.spec.package import extract_file_name, get_resource_package_content
from bioimageio.spec.summary import ValidationSummary


def load_description_and_validate(
    source: FileSource,
    /,
    *,
    format_version: Union[Literal["discover"], Literal["latest"], str] = DISCOVER,
) -> Union[ResourceDescription, InvalidDescription]:
    rdf = download_rdf(source)
    return build_description_and_validate(
        rdf.content,
        context=ValidationContext(root=rdf.original_root, file_name=rdf.original_file_name),
        format_version=format_version,
    )


def build_description_and_validate(
    rdf_content: RdfContent,
    /,
    *,
    context: Optional[ValidationContext] = None,
    format_version: Union[Literal["discover"], Literal["latest"], str] = DISCOVER,
) -> Union[ResourceDescription, InvalidDescription]:
    """load and validate a BioImage.IO description from the content of a resource description file (RDF)"""
    rd = load_description(rdf_content, context=context, format_version=format_version)
    # todo: add dynamic validation
    return rd


def validate(
    source: RdfSource,
    /,
    *,
    context: Optional[ValidationContext] = None,
    format_version: Union[Literal["discover"], Literal["latest"], str] = DISCOVER,
) -> List[ValidationSummary]:
    if isinstance(source, dict):
        rd = build_description_and_validate(source, context=context, format_version=format_version)
    else:
        rd = load_description_and_validate(source, format_version=format_version)

    return rd.validation_summaries
