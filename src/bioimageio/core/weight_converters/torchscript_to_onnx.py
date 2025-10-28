from pathlib import Path

import torch.jit
from bioimageio.spec.model.v0_5 import ModelDescr, OnnxWeightsDescr

from .. import __version__
from ..digest_spec import get_member_id, get_test_input_sample
from ..proc_setup import get_pre_and_postprocessing
from ._utils_onnx import get_dynamic_shapes


def convert(
    model_descr: ModelDescr,
    output_path: Path,
    *,
    verbose: bool = False,
    opset_version: int = 15,
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
            The ONNX opset version to use for the export. Defaults to 15.
    Raises:
        ValueError:
            If the provided model does not have weights in the torchscript format.

    Returns:
        v0_5.OnnxWeightsDescr:
            A descriptor object that contains information about the exported ONNX weights.
    """

    torchscript_descr = model_descr.weights.torchscript
    if torchscript_descr is None:
        raise ValueError(
            "The provided model does not have weights in the torchscript format"
        )

    sample = get_test_input_sample(model_descr)
    procs = get_pre_and_postprocessing(
        model_descr, dataset_for_initial_statistics=[sample]
    )
    procs.pre(sample)
    inputs_numpy = [
        sample.members[get_member_id(ipt)].data.data for ipt in model_descr.inputs
    ]
    inputs_torch = [torch.from_numpy(ipt) for ipt in inputs_numpy]

    weight_reader = torchscript_descr.get_reader()
    model = torch.jit.load(weight_reader)  # type: ignore
    model.to("cpu")
    model = model.eval()  # type: ignore

    with torch.no_grad():
        outputs_original_torch = model(*inputs_torch)  # type: ignore
        if isinstance(outputs_original_torch, torch.Tensor):
            outputs_original_torch = [outputs_original_torch]

        _ = torch.onnx.export(
            model,  # type: ignore
            tuple(inputs_torch),
            str(output_path),
            dynamo=True,
            input_names=[str(d.id) for d in model_descr.inputs],
            output_names=[str(d.id) for d in model_descr.outputs],
            dynamic_shapes=get_dynamic_shapes(model_descr),
            verbose=verbose,
            opset_version=opset_version,
        )

    return OnnxWeightsDescr(
        source=output_path.absolute(),
        parent="torchscript",
        opset_version=opset_version,
        comment=f"Converted with bioimageio.core {__version__}.",
    )
