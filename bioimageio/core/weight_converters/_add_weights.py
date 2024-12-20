from abc import ABC
from typing import Optional, Sequence, Union, assert_never, final

from bioimageio.spec.model import v0_4, v0_5


def increase_available_weight_formats(
    model_description: Union[v0_4.ModelDescr, v0_5.ModelDescr],
    source_format: v0_5.WeightsFormat,
    target_format: v0_5.WeightsFormat,
    *,
    devices: Optional[Sequence[str]] = None,
):
    if not isinstance(model_description, (v0_4.ModelDescr, v0_5.ModelDescr)):
        raise TypeError(
            f"expected v0_4.ModelDescr or v0_5.ModelDescr, but got {type(model_description)}"
        )

    if (source_format, target_format) == ("pytorch_state_dict", "onnx"):
        from .pytorch_to_onnx import convert_pytorch_to_onnx

    else:
        raise NotImplementedError(
            f"Converting from '{source_format}' to '{target_format}' is not yet implemented. Please create an issue at https://github.com/bioimage-io/core-bioimage-io-python/issues/new/choose"
        )
