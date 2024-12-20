from pathlib import Path
from typing import Any, List, Sequence, Union, cast

import numpy as np
import torch
from numpy.testing import assert_allclose

from bioimageio.core.backends.pytorch_backend import load_torch_model
from bioimageio.core.digest_spec import get_member_id, get_test_inputs
from bioimageio.core.proc_setup import get_pre_and_postprocessing
from bioimageio.spec.model import v0_4, v0_5


def convert(
    model_descr: Union[v0_4.ModelDescr, v0_5.ModelDescr],
    *,
    output_path: Path,
    use_tracing: bool = True,
    relative_tolerance: float = 1e-07,
    absolute_tolerance: float = 0,
    verbose: bool = False,
    opset_version: int = 15,
) -> v0_5.OnnxWeightsDescr:
    """
    Convert model weights from the PyTorch state_dict format to the ONNX format.

    # TODO: update Args
    Args:
        model_descr (Union[v0_4.ModelDescr, v0_5.ModelDescr]):
            The model description object that contains the model and its weights.
        output_path (Path):
            The file path where the ONNX model will be saved.
        use_tracing (bool, optional):
            Whether to use tracing or scripting to export the ONNX format. Defaults to True.
        verbose (bool, optional):
            If True, will print out detailed information during the ONNX export process. Defaults to False.
        opset_version (int, optional):
            The ONNX opset version to use for the export. Defaults to 15.
    Raises:
        ValueError:
            If the provided model does not have weights in the PyTorch state_dict format.
        ImportError:
            If ONNX Runtime is not available for checking the exported ONNX model.
        ValueError:
            If the results before and after weights conversion do not agree.
    Returns:
        v0_5.OnnxWeightsDescr:
            A descriptor object that contains information about the exported ONNX weights.
    """

    state_dict_weights_descr = model_descr.weights.pytorch_state_dict
    if state_dict_weights_descr is None:
        raise ValueError(
            "The provided model does not have weights in the pytorch state dict format"
        )

    sample = get_test_inputs(model_descr)
    procs = get_pre_and_postprocessing(
        model_descr, dataset_for_initial_statistics=[sample]
    )
    procs.pre(sample)
    inputs_numpy = [
        sample.members[get_member_id(ipt)].data.data for ipt in model_descr.inputs
    ]
    inputs_torch = [torch.from_numpy(ipt) for ipt in inputs_numpy]
    model = load_torch_model(state_dict_weights_descr)
    with torch.no_grad():
        outputs_original_torch = model(*inputs_torch)
        if isinstance(outputs_original_torch, torch.Tensor):
            outputs_original_torch = [outputs_original_torch]

        outputs_original: List[np.ndarray[Any, Any]] = [
            out.numpy() for out in outputs_original_torch
        ]

        if use_tracing:
            _ = torch.onnx.export(
                model,
                tuple(inputs_torch),
                str(output_path),
                verbose=verbose,
                opset_version=opset_version,
            )
        else:
            raise NotImplementedError

    try:
        import onnxruntime as rt  # pyright: ignore [reportMissingTypeStubs]
    except ImportError:
        raise ImportError(
            "The onnx weights were exported, but onnx rt is not available and weights cannot be checked."
        )

    # check the onnx model
    sess = rt.InferenceSession(str(output_path))
    onnx_input_node_args = cast(
        List[Any], sess.get_inputs()
    )  # FIXME: remove cast, try using rt.NodeArg instead of Any
    inputs_onnx = {
        input_name.name: inp
        for input_name, inp in zip(onnx_input_node_args, inputs_numpy)
    }
    outputs_onnx = cast(
        Sequence[np.ndarray[Any, Any]], sess.run(None, inputs_onnx)
    )  # FIXME: remove cast

    try:
        for out_original, out_onnx in zip(outputs_original, outputs_onnx):
            assert_allclose(
                out_original, out_onnx, rtol=relative_tolerance, atol=absolute_tolerance
            )
    except AssertionError as e:
        raise AssertionError(
            "Inference results of using original and converted weights do not match"
        ) from e

    return v0_5.OnnxWeightsDescr(
        source=output_path, parent="pytorch_state_dict", opset_version=opset_version
    )
