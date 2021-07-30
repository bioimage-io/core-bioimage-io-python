import argparse
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import xarray as xr
from numpy.testing import assert_array_almost_equal

from bioimageio.core.prediction_pipeline import create_prediction_pipeline, get_weight_formats
from bioimageio.spec import load_resource_description
from bioimageio.spec.model.nodes import Model

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="bioimage model source", required=True)
parser.add_argument(
    "-f", "--weight-format", help="weight format to use for the test", default=None, choices=get_weight_formats()
)
parser.add_argument("-d", "--decimals", help="test precision", default=4, type=int)


def load_array(path_to_npy: Union[Path, str], spec):
    return xr.DataArray(np.load(path_to_npy), dims=tuple(spec.axes))


def model_test(source: Union[Path, str], weight_format: Optional[str], decimals: int = 4):
    """Check if forward(test_input) == test_output (includes pre- and postprocessing). Raises RuntimeError."""

    try:
        model = load_resource_description(source)
        assert isinstance(model, Model), type(model)
    except TypeError as e:
        msg = "Failed to load source"
        if weight_format is not None:
            msg += f" for weight format {weight_format}"
        raise RuntimeError(msg) from e

    pipeline = create_prediction_pipeline(
        bioimageio_model=model, devices=["cuda" if torch.cuda.is_available() else "cpu"]
    )

    try:
        inputs = [load_array(inp, inp_spec) for inp, inp_spec in zip(model.test_inputs, model.inputs)]
    except Exception as e:
        raise RuntimeError("Failed to load test_input") from e

    try:
        expected_outputs = [load_array(out, out_spec) for out, out_spec in zip(model.test_outputs, model.outputs)]
    except Exception as e:
        raise RuntimeError("Failed to load test_output") from e

    try:
        actual_outputs = pipeline.forward(*inputs)
    except Exception as e:
        raise RuntimeError("Failed to process test_inputs") from e

    try:
        for actual, expected in zip(actual_outputs, expected_outputs):
            assert_array_almost_equal(actual, expected, decimals)
    except AssertionError as e:
        raise RuntimeError("Failed to reproduce test_outputs") from e


def main():
    args = parser.parse_args()
    try:
        model_test(args.model, args.weight_format, args.decimals)
    except RuntimeError as e:
        print(e)
        raise e
    else:
        print(f"Successfully reproduced test_outputs for {args.model}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
