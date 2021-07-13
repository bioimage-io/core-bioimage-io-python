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
from .utils import get_nn_instance


def convert_weights_to_pytorch_script(
    model_spec: Union[str, Path, spec.raw_nodes.Model],
    output_path: Union[str, Path],
    use_tracing: bool = True
):
    """ Convert model weights from format 'pytorch_state_dict' to 'torchscript'.
    """
    if isinstance(model_spec, (str, Path)):
        # TODO we probably need the root path here
        model_spec = spec.load_model(model_spec)

    with torch.no_grad():
        # load input and expected output data
        input_data = np.load(model_spec.test_inputs[0]).astype('float32')
        input_data = torch.from_numpy(input_data)

        # instantiate model and get reference output
        model = get_nn_instance(model_spec)
        model.eval()
        state = torch.load(model_spec.weights['pytorch_state_dict'].source)
        model.load_state_dict(state)

        # get the expected output to validate the torchscript weights
        expected_output = model(input_data)

        # make scripted model
        if use_tracing:
            scripted_model = torch.jit.trace(model, input_data)
        else:
            scripted_model = torch.jit.script(model)

        # check the scripted model
        output = scripted_model(input_data).numpy()

    # save the torchscript model
    scripted_model.save(output_path)

    try:
        assert_array_almost_equal(expected_output, output, decimal=4)
        return 0
    except AssertionError as e:
        msg = f"The onnx weights were exported, but results before and after conversion do not agree:\n {str(e)}"
        warnings.warn(msg)
        return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--tracing", "-t", default=1, type=int)

    args = parser.parse_args()
    return convert_weights_to_pytorch_script(os.path.abspath(args.model),
                                             args.output,
                                             bool(args.tracing))


if __name__ == '__main__':
    sys.exit(main())
