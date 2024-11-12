from pathlib import Path
from typing import Literal, Optional

import pytest

from bioimageio.core import load_model
from bioimageio.core.commands import package, validate_format
from bioimageio.core.commands import test as command_tst
from bioimageio.spec.model import ModelDescr


@pytest.mark.fixture(scope="module")
def model(unet2d_nuclei_broad_model: str):
    return load_model(unet2d_nuclei_broad_model, perform_io_checks=False)


@pytest.mark.parametrize(
    "weight_format",
    [
        "all",
        "pytorch_state_dict",
    ],
)
def test_package(
    weight_format: Literal["all", "pytorch_state_dict"],
    model: ModelDescr,
    tmp_path: Path,
):
    _ = package(model, weight_format=weight_format, path=tmp_path / "out.zip")


def test_validate_format(model: ModelDescr):
    _ = validate_format(model)


@pytest.mark.parametrize(
    "weight_format,devices", [("all", None), ("pytorch_state_dict", "cpu")]
)
def test_test(
    weight_format: Literal["all", "pytorch_state_dict"],
    devices: Optional[str],
    model: ModelDescr,
):
    _ = command_tst(model, weight_format=weight_format, devices=devices)
