from pathlib import Path

import pytest

from bioimageio.core import add_weights, load_model_description
from bioimageio.spec import InvalidDescr
from bioimageio.spec.model.v0_5 import WeightsFormat


@pytest.mark.parametrize(
    ("source_format", "target_format"),
    [
        ("pytorch_state_dict", "torchscript"),
        ("pytorch_state_dict", "onnx"),
        ("torchscript", "onnx"),
    ],
)
def test_add_weights(
    source_format: WeightsFormat,
    target_format: WeightsFormat,
    unet2d_nuclei_broad_model: str,
    tmp_path: Path,
    request: pytest.FixtureRequest,
):
    model = load_model_description(unet2d_nuclei_broad_model, format_version="latest")
    assert source_format in model.weights.available_formats, (
        "source format not found in model"
    )
    if target_format in model.weights.available_formats:
        setattr(model.weights, target_format, None)

    out_path = tmp_path / "converted.zip"
    converted = add_weights(
        model,
        output_path=out_path,
        source_format=source_format,
        target_format=target_format,
    )
    assert not isinstance(converted, InvalidDescr), (
        "conversion resulted in invalid descr",
        converted.validation_summary.display(),
    )
    assert target_format in converted.weights.available_formats
