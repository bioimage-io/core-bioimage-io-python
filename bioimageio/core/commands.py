"""deprecated,
use the CLI object `bioimageio.core.cli.Bioimageio` programmatically instead.
"""

import sys
from pathlib import Path
from typing import Optional, Sequence, Union

from typing_extensions import Literal

from bioimageio.spec import (
    InvalidDescr,
    ResourceDescr,
    save_bioimageio_package,
    save_bioimageio_package_as_folder,
)
from bioimageio.spec.model.v0_5 import WeightsFormat

from ._resource_tests import test_description

WeightFormatArgAll = Literal[WeightsFormat, "all"]
WeightFormatArgAny = Literal[WeightsFormat, "any"]


def test(
    descr: Union[ResourceDescr, InvalidDescr],
    *,
    weight_format: WeightFormatArgAll = "all",
    devices: Optional[Union[str, Sequence[str]]] = None,
    decimal: int = 4,
):
    """test a bioimageio resource

    Args:
        source: Path or URL to the bioimageio resource description file
                (bioimageio.yaml or rdf.yaml) or to a zipped resource
        weight_format: (model only) The weight format to use
        devices: Device(s) to use for testing
        decimal: Precision for numerical comparisons
    """
    if isinstance(descr, InvalidDescr):
        descr.validation_summary.display()
        sys.exit(1)

    summary = test_description(
        descr,
        weight_format=None if weight_format == "all" else weight_format,
        devices=[devices] if isinstance(devices, str) else devices,
        decimal=decimal,
    )
    summary.display()
    sys.exit(0 if summary.status == "passed" else 1)


def validate_format(
    descr: Union[ResourceDescr, InvalidDescr],
):
    """validate the meta data format of a bioimageio resource

    Args:
        descr: a bioimageio resource description
    """
    descr.validation_summary.display()
    sys.exit(0 if descr.validation_summary.status == "passed" else 1)


def package(
    descr: ResourceDescr, path: Path, *, weight_format: WeightFormatArgAll = "all"
):
    """Save a resource's metadata with its associated files.

    Note: If `path` does not have a `.zip` suffix this command will save the
          package as an unzipped folder instead.

    Args:
        descr: a bioimageio resource description
        path: output path
        weight-format: include only this single weight-format (if not 'all').
    """
    if isinstance(descr, InvalidDescr):
        descr.validation_summary.display()
        raise ValueError("resource description is invalid")

    if weight_format == "all":
        weights_priority_order = None
    else:
        weights_priority_order = (weight_format,)

    if path.suffix == ".zip":
        _ = save_bioimageio_package(
            descr,
            output_path=path,
            weights_priority_order=weights_priority_order,
        )
    else:
        _ = save_bioimageio_package_as_folder(
            descr,
            output_path=path,
            weights_priority_order=weights_priority_order,
        )
