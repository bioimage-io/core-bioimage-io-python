import os
import re
import traceback
import warnings
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy
import numpy as np
import xarray as xr

from bioimageio.core import __version__ as bioimageio_core_version
from bioimageio.core import load_raw_resource_description, load_resource_description
from bioimageio.core._internal.validation_visitors import Sha256NodeChecker, SourceNodeChecker
from bioimageio.core.common import TestSummary
from bioimageio.core.prediction import predict
from bioimageio.core.prediction_pipeline import create_prediction_pipeline
from bioimageio.core.resource_io.nodes import (
    URI,
    ImplicitOutputShape,
    Model,
    ParametrizedInputShape,
    ResourceDescription,
)
from bioimageio.spec import __version__ as bioimageio_spec_version
from bioimageio.spec._internal.io_utils import load_array
from bioimageio.spec.model.raw_nodes import WeightsFormat
from bioimageio.spec.shared import resolve_source
from bioimageio.spec.shared.common import ValidationWarning
from bioimageio.spec.shared.raw_nodes import ResourceDescription as RawResourceDescription


def test_model(
    model_rdf: Union[URI, Path, str],
    weight_format: Optional[WeightsFormat] = None,
    devices: Optional[List[str]] = None,
    decimal: int = 4,
) -> List[TestSummary]:
    """Test whether the test output(s) of a model can be reproduced."""
    return test_resource(
        model_rdf, weight_format=weight_format, devices=devices, decimal=decimal, expected_type="model"
    )


def check_input_shape(shape: Tuple[int, ...], shape_spec) -> bool:
    if isinstance(shape_spec, list):
        if shape != tuple(shape_spec):
            return False
    elif isinstance(shape_spec, ParametrizedInputShape):
        assert len(shape_spec.min) == len(shape_spec.step)
        if len(shape) != len(shape_spec.min):
            return False
        min_shape = shape_spec.min
        step = shape_spec.step
        # check if the shape is valid for all dimension by seeing if it can be reached with an integer number of steps
        # NOTE we allow that the valid shape is reached using a different number of steps for each axis here
        # this is usually valid because dimensions are independent in neural networks
        is_valid = [(sh - minsh) % st == 0 if st > 0 else sh == minsh for sh, st, minsh in zip(shape, step, min_shape)]
        return all(is_valid)
    else:
        raise TypeError(f"Encountered unexpected shape description of type {type(shape_spec)}")

    return True


def check_output_shape(shape: Tuple[int, ...], shape_spec, input_shapes) -> bool:
    if isinstance(shape_spec, list):
        return shape == tuple(shape_spec)
    elif isinstance(shape_spec, ImplicitOutputShape):
        ref_tensor = shape_spec.reference_tensor
        if ref_tensor not in input_shapes:
            raise ValidationError(f"The reference tensor name {ref_tensor} is not in {input_shapes}")
        ipt_shape = numpy.array(input_shapes[ref_tensor])
        scale = numpy.array([0.0 if sc is None else sc for sc in shape_spec.scale])
        offset = numpy.array(shape_spec.offset)
        exp_shape = numpy.round_(ipt_shape * scale) + 2 * offset

        return shape == tuple(exp_shape)
    else:
        raise TypeError(f"Encountered unexpected shape description of type {type(shape_spec)}")


def _test_resource_urls(rd: RawResourceDescription) -> TestSummary:
    assert isinstance(rd, RawResourceDescription), type(rd)
    with warnings.catch_warnings(record=True) as all_warnings:
        try:
            SourceNodeChecker(root_path=rd.root_path).visit(rd)
        except FileNotFoundError as e:
            error = str(e)
            tb = traceback.format_tb(e.__traceback__)
        else:
            error = None
            tb = None

    return dict(
        name="All URLs and paths available",
        status="passed" if error is None else "failed",
        error=error,
        traceback=tb,
        bioimageio_spec_version=bioimageio_spec_version,
        bioimageio_core_version=bioimageio_core_version,
        nested_errors=None,
        source_name=rd.id or rd.id or rd.name if hasattr(rd, "id") else rd.name,
        warnings={"SourceNodeChecker": [str(w.message) for w in all_warnings]} if all_warnings else {},
    )


def _test_resource_integrity(rd: RawResourceDescription) -> TestSummary:
    assert isinstance(rd, RawResourceDescription)
    with warnings.catch_warnings(record=True) as all_warnings:
        if isinstance(rd, ResourceDescription):
            warnings.warn("Testing source file integrity of an already loaded resource!")

        try:
            Sha256NodeChecker(root_path=rd.root_path).visit(rd)
        except FileNotFoundError as e:
            error = str(e)
            tb = traceback.format_tb(e.__traceback__)
        else:
            error = None
            tb = None

    return dict(
        name="Integrity of source files",
        status="passed" if error is None else "failed",
        error=error,
        traceback=tb,
        bioimageio_spec_version=bioimageio_spec_version,
        bioimageio_core_version=bioimageio_core_version,
        nested_errors=None,
        source_name=rd.id or rd.id or rd.name if hasattr(rd, "id") else rd.name,
        warnings={"Sha256NodeChecker": [str(w.message) for w in all_warnings]} if all_warnings else {},
    )


def _test_model_documentation(rd: ResourceDescription) -> TestSummary:
    assert isinstance(rd, Model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        doc_path: Path = resolve_source(rd.documentation, root_path=rd.root_path)
        doc = doc_path.read_text()
        wrn = ""
        if not re.match("#.*[vV]alidation", doc):
            wrn = "No '# Validation' (sub)section found."

        return dict(
            name="Test documentation completeness.",
            status="passed",
            error=None,
            traceback=None,
            bioimageio_spec_version=bioimageio_spec_version,
            bioimageio_core_version=bioimageio_core_version,
            source_name=rd.id or rd.name if hasattr(rd, "id") else rd.name,
            warnings={"documentation": wrn} if wrn else {},
        )


def _test_model_inference(model: Model, weight_format: str, devices: Optional[List[str]], decimal: int) -> TestSummary:
    error: Optional[str] = None
    tb: Optional = None
    with warnings.catch_warnings(record=True) as all_warnings:
        try:
            inputs = [load_array(str(in_path)) for in_path in model.test_inputs]
            expected = [load_array(str(out_path)) for out_path in model.test_outputs]

            assert len(inputs) == len(model.inputs)  # should be checked by validation
            input_shapes = {}
            for idx, (ipt, ipt_spec) in enumerate(zip(inputs, model.inputs)):
                if not check_input_shape(tuple(ipt.shape), ipt_spec.shape):
                    raise ValidationError(
                        f"Shape {tuple(ipt.shape)} of test input {idx} '{ipt_spec.name}' does not match "
                        f"input shape description: {ipt_spec.shape}."
                    )
                input_shapes[ipt_spec.name] = ipt.shape

            assert len(expected) == len(model.outputs)  # should be checked by validation
            for idx, (out, out_spec) in enumerate(zip(expected, model.outputs)):
                if not check_output_shape(tuple(out.shape), out_spec.shape, input_shapes):
                    error = (error or "") + (
                        f"Shape {tuple(out.shape)} of test output {idx} '{out_spec.name}' does not match "
                        f"output shape description: {out_spec.shape}."
                    )

            with create_prediction_pipeline(
                bioimageio_model=model, devices=devices, weight_format=weight_format
            ) as prediction_pipeline:
                results = predict(prediction_pipeline, inputs)

            if len(results) != len(expected):
                error = (error or "") + (
                    f"Number of outputs and number of expected outputs disagree: {len(results)} != {len(expected)}"
                )
            else:
                for res, exp in zip(results, expected):
                    try:
                        np.testing.assert_array_almost_equal(res, exp, decimal=decimal)
                    except AssertionError as e:
                        error = (error or "") + f"Output and expected output disagree:\n {e}"
        except Exception as e:
            error = str(e)
            tb = traceback.format_tb(e.__traceback__)

    return dict(
        name=f"Reproduce test outputs from test inputs (bioimageio.core {bioimageio_core_version})",
        status="passed" if error is None else "failed",
        error=error,
        traceback=tb,
        bioimageio_spec_version=bioimageio_spec_version,
        bioimageio_core_version=bioimageio_core_version,
        warnings=ValidationWarning.get_warning_summary(all_warnings),
        source_name=model.id or model.name,
    )


def _test_load_raw_resource(
    rdf: Union[RawResourceDescription, ResourceDescription, URI, Path, str]
) -> Tuple[Optional[ResourceDescription], TestSummary]:
    if isinstance(rdf, (URI, os.PathLike)):
        source_name = str(rdf)
    elif isinstance(rdf, str):
        source_name = rdf[:120]
    else:
        source_name = rdf.id if hasattr(rdf, "id") else rdf.name

    main_test_warnings = []
    try:
        with warnings.catch_warnings(record=True) as all_warnings:
            rd: Optional[ResourceDescription] = load_raw_resource_description(rdf)

            main_test_warnings += list(all_warnings)
    except Exception as e:
        rd = None
        error: Optional[str] = str(e)
        tb: Optional = traceback.format_tb(e.__traceback__)
    else:
        error = None
        tb = None

    load_summary = TestSummary(
        name="Load raw resource description",
        status="passed" if error is None else "failed",
        error=error,
        nested_errors=None,
        traceback=tb,
        bioimageio_spec_version=bioimageio_spec_version,
        bioimageio_core_version=bioimageio_core_version,
        warnings={},
        source_name=source_name,
    )

    return rd, load_summary


def _test_load_resource(
    raw_rd: RawResourceDescription,
    weight_format: Optional[WeightsFormat] = None,
) -> Tuple[Optional[ResourceDescription], TestSummary]:
    source_name = getattr(raw_rd, "rdf_source", getattr(raw_rd, "id", raw_rd.name))

    main_test_warnings = []
    try:
        with warnings.catch_warnings(record=True) as all_warnings:
            rd: Optional[ResourceDescription] = load_resource_description(
                raw_rd, weights_priority_order=None if weight_format is None else [weight_format]
            )

            main_test_warnings += list(all_warnings)
    except Exception as e:
        rd = None
        error: Optional[str] = str(e)
        tb: Optional = traceback.format_tb(e.__traceback__)
    else:
        error = None
        tb = None

    load_summary = TestSummary(
        name="Load resource description",
        status="passed" if error is None else "failed",
        error=error,
        nested_errors=None,
        traceback=tb,
        bioimageio_spec_version=bioimageio_spec_version,
        bioimageio_core_version=bioimageio_core_version,
        warnings={},
        source_name=source_name,
    )

    return rd, load_summary


def _test_expected_resource_type(rd: RawResourceDescription, expected_type: str) -> TestSummary:
    has_expected_type = rd.type == expected_type
    return dict(
        name="Has expected resource type",
        status="passed" if has_expected_type else "failed",
        error=None if has_expected_type else f"expected type {expected_type}, found {rd.type}",
        traceback=None,
        source_name=rd.id or rd.name if hasattr(rd, "id") else rd.name,
    )


def test_resource(
    rdf: Union[RawResourceDescription, ResourceDescription, URI, Path, str],
    *,
    weight_format: Optional[WeightsFormat] = None,
    devices: Optional[List[str]] = None,
    decimal: int = 4,
    expected_type: Optional[str] = None,
) -> List[TestSummary]:
    """Test RDF dynamically

    Returns: summary dict with keys: name, status, error, traceback, bioimageio_spec_version, bioimageio_core_version
    """
    raw_rd, load_test = _test_load_raw_resource(rdf)
    tests: List[TestSummary] = [load_test]
    if raw_rd is None:
        return tests

    if expected_type is not None:
        tests.append(_test_expected_resource_type(raw_rd, expected_type))

    tests.append(_test_resource_urls(raw_rd))
    if tests[-1]["status"] == "passed":
        tests.append(_test_resource_integrity(raw_rd))

    if tests[-1]["status"] != "passed":
        return tests  # stop testing if resource availability/integrity is an issue

    rd = _test_load_resource(raw_rd, weight_format)
    if isinstance(rd, Model):
        tests.append(_test_model_documentation(rd))
        tests.append(_test_model_inference(rd, weight_format, devices, decimal))

    return tests


def debug_model(
    model_rdf: Union[RawResourceDescription, ResourceDescription, URI, Path, str],
    *,
    weight_format: Optional[WeightsFormat] = None,
    devices: Optional[List[str]] = None,
):
    """Run the model test and return dict with inputs, results, expected results and intermediates.

    Returns dict with tensors "inputs", "inputs_processed", "outputs_raw", "outputs", "expected" and "diff".
    """
    inputs_raw: Optional = None
    inputs_processed: Optional = None
    outputs_raw: Optional = None
    outputs: Optional = None
    expected: Optional = None
    diff: Optional = None

    model = load_resource_description(
        model_rdf, weights_priority_order=None if weight_format is None else [weight_format]
    )
    if not isinstance(model, Model):
        raise ValueError(f"Not a bioimageio.model: {model_rdf}")

    prediction_pipeline = create_prediction_pipeline(
        bioimageio_model=model, devices=devices, weight_format=weight_format
    )
    inputs = [
        xr.DataArray(load_array(str(in_path)), dims=input_spec.axes)
        for in_path, input_spec in zip(model.test_inputs, model.inputs)
    ]
    input_dict = {input_spec.name: input for input_spec, input in zip(model.inputs, inputs)}

    # keep track of the non-processed inputs
    inputs_raw = [deepcopy(input) for input in inputs]

    computed_measures = {}

    prediction_pipeline.apply_preprocessing(input_dict, computed_measures)
    inputs_processed = list(input_dict.values())
    outputs_raw = prediction_pipeline.predict(*inputs_processed)
    output_dict = {output_spec.name: deepcopy(output) for output_spec, output in zip(model.outputs, outputs_raw)}
    prediction_pipeline.apply_postprocessing(output_dict, computed_measures)
    outputs = list(output_dict.values())

    if isinstance(outputs, (np.ndarray, xr.DataArray)):
        outputs = [outputs]

    expected = [
        xr.DataArray(load_array(str(out_path)), dims=output_spec.axes)
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
        "inputs": inputs_raw,
        "inputs_processed": inputs_processed,
        "outputs_raw": outputs_raw,
        "outputs": outputs,
        "expected": expected,
        "diff": diff,
    }
