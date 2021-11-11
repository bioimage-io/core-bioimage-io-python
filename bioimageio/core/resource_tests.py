import traceback
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import xarray as xr

from bioimageio.core import load_resource_description
from bioimageio.core.prediction import predict
from bioimageio.core.prediction_pipeline import create_prediction_pipeline
from bioimageio.core.resource_io.nodes import Model, ResourceDescription, URI
from bioimageio.spec.model.raw_nodes import WeightsFormat
from bioimageio.spec.shared.raw_nodes import ResourceDescription as RawResourceDescription


def test_model(
    model_rdf: Union[URI, Path, str],
    weight_format: Optional[WeightsFormat] = None,
    devices: Optional[List[str]] = None,
    decimal: int = 4,
) -> dict:
    """Test whether the test output(s) of a model can be reproduced.

    Returns summary dict with "error" and "traceback" key; summary["error"] is None if no errors were encountered.
    """
    model = load_resource_description(model_rdf)
    if isinstance(model, Model):
        return test_resource(model, weight_format=weight_format, devices=devices, decimal=decimal)
    else:
        return {"error": f"Expected RDF type Model, got {type(model)} instead.", "traceback": None}


def test_resource(
    model_rdf: Union[RawResourceDescription, ResourceDescription, URI, Path, str],
    *,
    weight_format: Optional[WeightsFormat] = None,
    devices: Optional[List[str]] = None,
    decimal: int = 4,
):
    """Test RDF dynamically

    Returns summary dict with "error" and "traceback" key; summary["error"] is None if no errors were encountered.
    """
    error: Optional[str] = None
    tb: Optional = None

    try:
        model = load_resource_description(model_rdf)
    except Exception as e:
        error = str(e)
        tb = traceback.format_tb(e.__traceback__)
    else:
        if isinstance(model, Model):
            try:
                prediction_pipeline = create_prediction_pipeline(
                    bioimageio_model=model, devices=devices, weight_format=weight_format
                )
                inputs = [np.load(str(in_path)) for in_path in model.test_inputs]
                results = predict(prediction_pipeline, inputs)
                if isinstance(results, (np.ndarray, xr.DataArray)):
                    results = [results]

                expected = [np.load(str(out_path)) for out_path in model.test_outputs]
                if len(results) != len(expected):
                    error = (
                        f"Number of outputs and number of expected outputs disagree: {len(results)} != {len(expected)}"
                    )
                else:
                    for res, exp in zip(results, expected):
                        try:
                            np.testing.assert_array_almost_equal(res, exp, decimal=decimal)
                        except AssertionError as e:
                            error = f"Output and expected output disagree:\n {e}"
            except Exception as e:
                error = str(e)
                tb = traceback.format_tb(e.__traceback__)

    # todo: add tests for non-model resources

    return {"error": error, "traceback": tb}


def debug_model(
    model_rdf: Union[RawResourceDescription, ResourceDescription, URI, Path, str],
    *,
    weight_format: Optional[WeightsFormat] = None,
    devices: Optional[List[str]] = None,
):
    """Run the model test and return dict with inputs, results, expected results and intermediates.

    Returns dict with tensors "inputs", "inputs_processed", "outputs_raw", "outputs", "expected" and "diff".
    """
    inputs: Optional = None
    inputs_processed: Optional = None
    outputs_raw: Optional = None
    outputs: Optional = None
    expected: Optional = None
    diff: Optional = None

    model = load_resource_description(model_rdf)
    if not isinstance(model, Model):
        raise ValueError(f"Not a bioimageio.model: {model_rdf}")

    prediction_pipeline = create_prediction_pipeline(
        bioimageio_model=model, devices=devices, weight_format=weight_format
    )
    inputs = [
        xr.DataArray(np.load(str(in_path)), dims=input_spec.axes)
        for in_path, input_spec in zip(model.test_inputs, model.inputs)
    ]

    inputs_processed, stats = prediction_pipeline.preprocess(*inputs)
    outputs_raw = prediction_pipeline.predict(*inputs_processed)
    outputs, _ = prediction_pipeline.postprocess(*outputs_raw, input_sample_statistics=stats)
    if isinstance(outputs, (np.ndarray, xr.DataArray)):
        outputs = [outputs]

    expected = [
        xr.DataArray(np.load(str(out_path)), dims=output_spec.axes)
        for out_path, output_spec in zip(model.test_outputs, model.outputs)
    ]
    if len(outputs) != len(expected):
        error = f"Number of outputs and number of expected outputs disagree: {len(outputs)} != {len(expected)}"
        print(error)
    else:
        diff = []
        for res, exp in zip(outputs, expected):
            diff.append(res - exp)

    return {
        "inputs": inputs,
        "inputs_processed": inputs_processed,
        "outputs_raw": outputs_raw,
        "outputs": outputs,
        "expected": expected,
        "diff": diff,
    }
