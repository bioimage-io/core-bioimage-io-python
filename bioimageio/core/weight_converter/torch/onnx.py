import warnings
from pathlib import Path
from typing import Union, Optional

import numpy as np
import torch
from numpy.testing import assert_array_almost_equal

import bioimageio.spec as spec
from bioimageio.core import load_resource_description
from bioimageio.core.resource_io import nodes
from .utils import load_model

try:
    import onnxruntime as rt
except ImportError:
    rt = None


def convert_weights_to_onnx(
    model_spec: Union[str, Path, spec.model.raw_nodes.Model],
    output_path: Union[str, Path],
    opset_version: Optional[int] = 12,
    use_tracing: bool = True,
    verbose: bool = True,
):
    """Convert model weights from format 'pytorch_state_dict' to 'onnx'.

    Arguments:
        model_yaml: location of the model.yaml file with bioimage.io spec
        output_path: where to save the onnx weights
        opset_version: onnx opset version
        use_tracing: whether to use tracing or scripting to export the onnx format
        verbose: be verbose during the onnx export
    """
    if isinstance(model_spec, (str, Path)):
        model_spec = load_resource_description(Path(model_spec))

    assert isinstance(model_spec, nodes.Model)
    with torch.no_grad():
        # load input and expected output data
        input_data = [np.load(inp).astype("float32") for inp in model_spec.test_inputs]
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
                example_outputs=expected_outputs,
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
                assert_array_almost_equal(exp, out, decimal=4)
            return 0
        except AssertionError as e:
            msg = f"The onnx weights were exported, but results before and after conversion do not agree:\n {str(e)}"
            warnings.warn(msg)
            return 1
