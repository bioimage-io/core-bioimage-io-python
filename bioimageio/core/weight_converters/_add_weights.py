from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Sequence, Union

from bioimageio.spec.model import v0_4, v0_5


def increase_available_weight_formats(
    model_descr: Union[v0_4.ModelDescr, v0_5.ModelDescr],
    *,
    source_format: Optional[v0_5.WeightsFormat] = None,
    target_format: Optional[v0_5.WeightsFormat] = None,
    output_path: Path,
    devices: Optional[Sequence[str]] = None,
) -> Union[v0_4.ModelDescr, v0_5.ModelDescr]:
    """Convert neural network weights to other formats and add them to the model description"""
    if not isinstance(model_descr, (v0_4.ModelDescr, v0_5.ModelDescr)):
        raise TypeError(
            f"expected v0_4.ModelDescr or v0_5.ModelDescr, but got {type(model_descr)}"
        )

    if source_format is None:
        available = [wf for wf, w in model_descr.weights if w is not None]
        missing = [wf for wf, w in model_descr.weights if w is None]
    else:
        available = [source_format]
        missing = [target_format]

    if "pytorch_state_dict" in available and "onnx" in missing:
        from .pytorch_to_onnx import convert

        onnx = convert(model_descr)

    else:
        raise NotImplementedError(
            f"Converting from '{source_format}' to '{target_format}' is not yet implemented. Please create an issue at https://github.com/bioimage-io/core-bioimage-io-python/issues/new/choose"
        )
