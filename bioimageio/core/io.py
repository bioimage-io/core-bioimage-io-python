from __future__ import annotations

from contextlib import nullcontext
from typing import Literal, Optional, Union

from bioimageio.core.resource_tests import test_description
from bioimageio.spec import load_description as load_description
from bioimageio.spec._description import ResourceDescr
from bioimageio.spec._internal.constants import DISCOVER
from bioimageio.spec._internal.io_utils import open_bioimageio_yaml
from bioimageio.spec._internal.validation_context import ValidationContext
from bioimageio.spec.common import BioimageioYamlContent, FileSource, InvalidDescr
from bioimageio.spec.summary import ValidationSummary


def load_description_and_test(
    source: FileSource,
    /,
    *,
    format_version: Union[Literal["discover"], Literal["latest"], str] = DISCOVER,
) -> Union[ResourceDescr, InvalidDescr]:
    opened = open_bioimageio_yaml(source)

    return build_description_and_test(
        opened.content,
        context=ValidationContext(root=opened.original_root, file_name=opened.original_file_name),
        format_version=format_version,
    )


def build_description_and_test(
    data: BioimageioYamlContent,
    /,
    *,
    context: Optional[ValidationContext] = None,
    format_version: Union[Literal["discover"], Literal["latest"], str] = DISCOVER,
) -> Union[ResourceDescr, InvalidDescr]:
    """load and validate a BioImage.IO description from the content of a resource description file (RDF)"""
    if context is None:
        val_context = nullcontext()
    else:
        val_context = context

    with val_context:
        rd = test_description(data, format_version=format_version)

    return rd


def validate(
    source: "FileSource | BioimageioYamlContent",
    /,
    *,
    context: Optional[ValidationContext] = None,
    format_version: Union[Literal["discover"], Literal["latest"], str] = DISCOVER,
) -> ValidationSummary:
    if isinstance(source, dict):
        rd = build_description_and_test(source, context=context, format_version=format_version)
    else:
        rd = load_description_and_test(source, format_version=format_version)

    return rd.validation_summary
