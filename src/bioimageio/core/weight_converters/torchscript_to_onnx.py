from pathlib import Path

import torch.jit
from bioimageio.spec.model.v0_5 import ModelDescr, OnnxWeightsDescr

from ._utils_torch_onnx import export_to_onnx


def convert(
    model_descr: ModelDescr,
    output_path: Path,
    *,
    verbose: bool = False,
    opset_version: int = 18,
) -> OnnxWeightsDescr:
    """
    Convert model weights from the PyTorch state_dict format to the ONNX format.

    Args:
        model_descr (Union[v0_4.ModelDescr, v0_5.ModelDescr]):
            The model description object that contains the model and its weights.
        output_path (Path):
            The file path where the ONNX model will be saved.
        verbose (bool, optional):
            If True, will print out detailed information during the ONNX export process. Defaults to False.
        opset_version (int, optional):
            The ONNX opset version to use for the export. Defaults to 18.
    Raises:
        ValueError:
            If the provided model does not have weights in the torchscript format.

    Returns:
        A description of the exported ONNX weights.
    """

    torchscript_descr = model_descr.weights.torchscript
    if torchscript_descr is None:
        raise ValueError(
            "The provided model does not have weights in the torchscript format"
        )

    weight_reader = torchscript_descr.get_reader()
    model = torch.jit.load(weight_reader)  # pyright: ignore[reportUnknownVariableType]
    model.to("cpu")
    model = model.eval()  # pyright: ignore[reportUnknownVariableType]

    return export_to_onnx(
        model_descr,
        model,  # pyright: ignore[reportUnknownArgumentType]
        output_path,
        verbose,
        opset_version,
        parent="torchscript",
    )
