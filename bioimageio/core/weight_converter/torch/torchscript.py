import argparse
import os
import sys
import warnings

from pathlib import Path
from typing import Union

import numpy as np
import torch
from numpy.testing import assert_array_almost_equal

import bioimageio.spec as spec
from .utils import load_model


def _check_predictions(model, scripted_model, model_spec, input_data):
    def _check(expected_output, output):
        try:
            assert_array_almost_equal(expected_output, output, decimal=4)
            return 0
        except AssertionError as e:
            msg = f"The onnx weights were exported, but results before and after conversion do not agree:\n {str(e)}"
            warnings.warn(msg)
            return 1

    # get the expected output to validate the torchscript weights
    expected_output = model(input_data).numpy()
    output = scripted_model(input_data).numpy()

    ret = _check(expected_output, output)
    # check has not passed? then return immediately
    if ret == 1:
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
        this_input = input_data[slice_]
        this_shape = this_input.shape
        if any(tsh < msh for tsh, msh in zip(this_shape, min_shape)):
            return ret

        expected_output = model(this_input).numpy()
        output = scripted_model(this_input).numpy()

        ret = _check(expected_output, output)
        if ret == 1:
            return ret
        step_factor += 1
        if step_factor > max_steps:
            return ret

    return ret


def convert_weights_to_pytorch_script(
    model_spec: Union[str, Path, spec.model.raw_nodes.Model], output_path: Union[str, Path], use_tracing: bool = True
):
    """ Convert model weights from format 'pytorch_state_dict' to 'torchscript'.
    """
    if isinstance(model_spec, (str, Path)):
        root = os.path.split(model_spec)[0]
        model_spec = spec.load_resource_description(model_spec, root_path=root)

    with torch.no_grad():
        # load input and expected output data
        input_data = np.load(model_spec.test_inputs[0]).astype("float32")
        input_data = torch.from_numpy(input_data)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--tracing", "-t", default=1, type=int)

    args = parser.parse_args()
    return convert_weights_to_pytorch_script(os.path.abspath(args.model), args.output, bool(args.tracing))


if __name__ == "__main__":
    sys.exit(main())
