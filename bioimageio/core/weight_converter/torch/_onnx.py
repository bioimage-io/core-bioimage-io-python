# type: ignore  # TODO: type
import warnings
from pathlib import Path
from typing import Any, List, Sequence, cast

import numpy as np
from numpy.testing import assert_array_almost_equal

from bioimageio.spec import load_description
from bioimageio.spec.common import InvalidDescr
from bioimageio.spec.model import v0_4, v0_5

from ...digest_spec import get_member_id, get_test_inputs
from ...weight_converter.torch._utils import load_torch_model

try:
    import torch
except ImportError:
    torch = None


def add_onnx_weights(
    model_spec: "str | Path | v0_4.ModelDescr | v0_5.ModelDescr",
    *,
    output_path: Path,
    use_tracing: bool = True,
    test_decimal: int = 4,
    verbose: bool = False,
    opset_version: "int | None" = None,
):
    """Convert model weights from format 'pytorch_state_dict' to 'onnx'.

    Args:
        source_model: model without onnx weights
        opset_version: onnx opset version
        use_tracing: whether to use tracing or scripting to export the onnx format
        test_decimal: precision for testing whether the results agree
    """
    if isinstance(model_spec, (str, Path)):
        loaded_spec = load_description(Path(model_spec))
        if isinstance(loaded_spec, InvalidDescr):
            raise ValueError(f"Bad resource description: {loaded_spec}")
        if not isinstance(loaded_spec, (v0_4.ModelDescr, v0_5.ModelDescr)):
            raise TypeError(
                f"Path {model_spec} is a {loaded_spec.__class__.__name__}, expected a v0_4.ModelDescr or v0_5.ModelDescr"
            )
        model_spec = loaded_spec

    state_dict_weights_descr = model_spec.weights.pytorch_state_dict
    if state_dict_weights_descr is None:
        raise ValueError(
            "The provided model does not have weights in the pytorch state dict format"
        )

    assert torch is not None
    with torch.no_grad():

        sample = get_test_inputs(model_spec)
        input_data = [sample[get_member_id(ipt)].data.data for ipt in model_spec.inputs]
        input_tensors = [torch.from_numpy(ipt) for ipt in input_data]
        model = load_torch_model(state_dict_weights_descr)

        expected_tensors = model(*input_tensors)
        if isinstance(expected_tensors, torch.Tensor):
            expected_tensors = [expected_tensors]
        expected_outputs: List[np.ndarray[Any, Any]] = [
            out.numpy() for out in expected_tensors
        ]

        if use_tracing:
            torch.onnx.export(
                model,
                tuple(input_tensors) if len(input_tensors) > 1 else input_tensors[0],
                str(output_path),
                verbose=verbose,
                opset_version=opset_version,
            )
        else:
            raise NotImplementedError

    try:
        import onnxruntime as rt  # pyright: ignore [reportMissingTypeStubs]
    except ImportError:
        msg = "The onnx weights were exported, but onnx rt is not available and weights cannot be checked."
        warnings.warn(msg)
        return

    # check the onnx model
    sess = rt.InferenceSession(str(output_path))
    onnx_input_node_args = cast(
        List[Any], sess.get_inputs()
    )  # fixme: remove cast, try using rt.NodeArg instead of Any
    onnx_inputs = {
        input_name.name: inp
        for input_name, inp in zip(onnx_input_node_args, input_data)
    }
    outputs = cast(
        Sequence[np.ndarray[Any, Any]], sess.run(None, onnx_inputs)
    )  # FIXME: remove cast

    try:
        for exp, out in zip(expected_outputs, outputs):
            assert_array_almost_equal(exp, out, decimal=test_decimal)
        return 0
    except AssertionError as e:
        msg = f"The onnx weights were exported, but results before and after conversion do not agree:\n {str(e)}"
        warnings.warn(msg)
        return 1
