import warnings

from pathlib import Path
from typing import Union

import numpy as np
import torch
from numpy.testing import assert_array_almost_equal

import bioimageio.spec as spec
from bioimageio.core import load_resource_description
from .utils import load_model


def _check_predictions(model, scripted_model, model_spec, input_data):
    assert isinstance(input_data, list)

    def _check(input_):
        # get the expected output to validate the torchscript weights
        expected_outputs = model(*input_)
        if isinstance(expected_outputs, (torch.Tensor)):
            expected_outputs = [expected_outputs]
        expected_outputs = [out.numpy() for out in expected_outputs]

        outputs = scripted_model(*input_)
        if isinstance(outputs, (torch.Tensor)):
            outputs = [outputs]
        outputs = [out.numpy() for out in outputs]

        try:
            for exp, out in zip(expected_outputs, outputs):
                assert_array_almost_equal(exp, out, decimal=4)
            return 0
        except AssertionError as e:
            msg = f"The onnx weights were exported, but results before and after conversion do not agree:\n {str(e)}"
            warnings.warn(msg)
            return 1

    ret = _check(input_data)
    n_inputs = len(model_spec.inputs)
    # check has not passed or we have more tahn one input? then return immediately
    if ret == 1 or n_inputs > 1:
        return ret

    # do we have fixed input size or variable?
    # if variable, we need to check multiple sizes!
    shape_spec = model_spec.inputs[0].shape
    try:  # we have a variable shape
        min_shape = shape_spec.min
        step = shape_spec.step
    except AttributeError:  # we have fixed shape
        return ret

    half_step = [st // 2 for st in step]
    max_steps = 4
    step_factor = 1

    # check that input and output agree for decreasing input sizes
    while True:

        slice_ = tuple(slice(None) if st == 0 else slice(step_factor * st, -step_factor * st) for st in half_step)
        this_input = [inp[slice_] for inp in input_data]
        this_shape = this_input[0].shape
        if any(tsh < msh for tsh, msh in zip(this_shape, min_shape)):
            return ret

        ret = _check(this_input)
        if ret == 1:
            return ret
        step_factor += 1
        if step_factor > max_steps:
            return ret


def convert_weights_to_pytorch_script(
    model_spec: Union[str, Path, spec.model.raw_nodes.Model], output_path: Union[str, Path], use_tracing: bool = True
):
    """Convert model weights from format 'pytorch_state_dict' to 'torchscript'."""
    if isinstance(model_spec, (str, Path)):
        model_spec = load_resource_description(Path(model_spec))

    with torch.no_grad():
        # load input and expected output data
        input_data = [np.load(inp).astype("float32") for inp in model_spec.test_inputs]
        input_data = [torch.from_numpy(inp) for inp in input_data]

        # instantiate model and get reference output
        model = load_model(model_spec)

        # make scripted model
        if use_tracing:
            scripted_model = torch.jit.trace(model, input_data)
        else:
            scripted_model = torch.jit.script(model)

        # check the scripted model
        ret = _check_predictions(model, scripted_model, model_spec, input_data)

    # save the torchscript model
    scripted_model.save(str(output_path))  # does not support Path, so need to cast to str
    return ret
