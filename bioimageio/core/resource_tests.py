import traceback
import warnings
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
