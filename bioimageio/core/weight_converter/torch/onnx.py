import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from numpy.testing import assert_array_almost_equal

from bioimageio.spec import load_description
from bioimageio.spec._internal.types import BioimageioYamlSource
from bioimageio.spec.model import v0_4, v0_5

try:
    import onnxruntime as rt
except ImportError:
    rt = None

# def add_converted_onnx_weights(model_spec: AnyModel, *, opset_version: Optional[int] = 12, use_tracing: bool = True,
#     verbose: bool = True,
#     test_decimal: int = 4):


# def add_onnx_weights_from_pytorch_state_dict(model_spec: Union[BioimageioYamlSource, AnyModel], test_decimals: int = 4):


def add_onnx_weights(
    source_model: Union[BioimageioYamlSource, AnyModel],
    *,
    use_tracing: bool = True,
    test_decimal: int = 4,
):
    """Convert model weights from format 'pytorch_state_dict' to 'onnx'.

    Args:
        source_model: model without onnx weights
        opset_version: onnx opset version
        use_tracing: whether to use tracing or scripting to export the onnx format
        test_decimal: precision for testing whether the results agree
    """
    if isinstance(source_model, (str, Path)):
        model = load_description(Path(source_model))
        assert isinstance(model, (v0_4.Model, v0_5.Model))

    with torch.no_grad():
        # load input and expected output data
        input_data = [np.load(ipt).astype("float32") for ipt in model.test_inputs]
        input_tensors = [torch.from_numpy(inp) for inp in input_data]

        # instantiate and generate the expected output
        model = load_model(model_spec)
        expected_outputs = model(*input_tensors)
        if isinstance(expected_outputs, torch.Tensor):
            expected_outputs = [expected_outputs]
        expected_outputs = [out.numpy() for out in expected_outputs]

        if use_tracing:
            torch.onnx.export(
                model,
                input_tensors if len(input_tensors) > 1 else input_tensors[0],
                output_path,
                verbose=verbose,
                opset_version=opset_version,
            )
        else:
            raise NotImplementedError

    if rt is None:
        msg = "The onnx weights were exported, but onnx rt is not available and weights cannot be checked."
        warnings.warn(msg)
        return 1

    # check the onnx model
    sess = rt.InferenceSession(str(output_path))  # does not support Path, so need to cast to str
    onnx_inputs = {input_name.name: inp for input_name, inp in zip(sess.get_inputs(), input_data)}
    outputs = sess.run(None, onnx_inputs)

    try:
        for exp, out in zip(expected_outputs, outputs):
            assert_array_almost_equal(exp, out, decimal=test_decimal)
        return 0
    except AssertionError as e:
        msg = f"The onnx weights were exported, but results before and after conversion do not agree:\n {str(e)}"
        warnings.warn(msg)
        return 1
