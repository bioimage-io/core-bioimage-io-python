from pathlib import Path

from bioimageio.spec.model.v0_5 import ModelDescr, OnnxWeightsDescr

from ..backends.pytorch_backend import load_torch_model
from ._utils_torch_onnx import export_to_onnx


def convert(
    model_descr: ModelDescr,
    output_path: Path,
    *,
    verbose: bool = False,
    opset_version: int = 18,
) -> OnnxWeightsDescr:
    """
    Convert model weights from the Torchscript state_dict format to the ONNX format.

    Args:
        model_descr:
            The model description object that contains the model and its weights.
        output_path:
            The file path where the ONNX model will be saved.
        verbose:
            If True, will print out detailed information during the ONNX export process. Defaults to False.
        opset_version:
            The ONNX opset version to use for the export. Defaults to 18.

    Raises:
        ValueError:
            If the provided model does not have weights in the PyTorch state_dict format.

    Returns:
        A description of the exported ONNX weights.
    """

    state_dict_weights_descr = model_descr.weights.pytorch_state_dict
    if state_dict_weights_descr is None:
        raise ValueError(
            "The provided model does not have weights in the pytorch state dict format"
        )

    model = load_torch_model(state_dict_weights_descr, load_state=True)

    return export_to_onnx(
        model_descr,
        model,
        output_path,
        verbose,
        opset_version,
        parent="pytorch_state_dict",
    )
