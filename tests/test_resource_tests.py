from pathlib import Path

import numpy as np
import pytest

from bioimageio.core import Tensor
from bioimageio.core._resource_tests import evaluate_mismatched_elements
from bioimageio.spec import InvalidDescr, ValidationContext


def test_error_for_wrong_shape(stardist_wrong_shape: str):
    from bioimageio.core._resource_tests import test_model

    summary = test_model(stardist_wrong_shape)
    expected_error_message = (
        "Shape (1, 512, 512, 33) of test output 0 'output' does not match output shape description: "
        "ImplicitOutputShape(reference_tensor='input', "
        "scale=[1.0, 1.0, 1.0, 0.0], offset=[1.0, 1.0, 1.0, 33.0])."
    )
    assert summary.details[0].errors[0].msg == expected_error_message


def test_error_for_wrong_shape2(stardist_wrong_shape2: str):
    from bioimageio.core._resource_tests import test_model

    summary = test_model(stardist_wrong_shape2)
    expected_error_message = (
        "Shape (1, 512, 512, 1) of test input 0 'input' does not match input shape description: "
        "ParameterizedInputShape(min=[1, 80, 80, 1], step=[0, 17, 17, 0])."
    )
    assert summary.details[0].errors[0].msg == expected_error_message


def test_test_model(any_model: str):
    from bioimageio.core._resource_tests import test_model

    with ValidationContext(raise_errors=True):
        summary = test_model(any_model)

    assert summary.status == "passed", summary.display()


def test_loading_description_multiple_times(unet2d_nuclei_broad_model: str):
    from bioimageio.core import load_description

    model_descr = load_description(unet2d_nuclei_broad_model)
    assert not isinstance(model_descr, InvalidDescr)

    # load again, which some users might end up doing
    model_descr = load_description(model_descr)  # pyright: ignore[reportArgumentType]
    assert not isinstance(model_descr, InvalidDescr)


def test_test_description_runtime_env(unet2d_nuclei_broad_model_old: str):
    from bioimageio.core._resource_tests import test_description

    summary = test_description(
        unet2d_nuclei_broad_model_old, runtime_env="as-described"
    )

    assert summary.status == "passed", summary.display()


def test_failed_reproducibility(unet2d_nuclei_broad_model: str, tmp_path: str):
    from bioimageio.core import load_model
    from bioimageio.core._resource_tests import test_model
    from bioimageio.spec.common import FileDescr
    from bioimageio.spec.utils import load_array, save_array

    model = load_model(unet2d_nuclei_broad_model, format_version="latest")

    # use corrupted test input to fail the reproducibility test
    test_array_path = Path(tmp_path) / "input.npy"
    assert model.inputs[0].test_tensor is not None
    test_array = load_array(model.inputs[0].test_tensor)
    save_array(test_array_path, np.zeros_like(test_array))
    model.inputs[0].test_tensor = FileDescr(source=test_array_path)

    summary = test_model(model)

    assert summary.status == "valid-format"


@pytest.mark.parametrize(
    "actual, expected, rtol, atol, ppm",
    [
        ([1.0, 2.0], [1.0, 2.0], 0, 0, 0),
        ([1.125, 2.0], [1.0, 2.0], 0, 0.125, 0),
        ([1.0, 2.25], [1.0, 2.0], 0.125, 0, 0),
        ([1.25, 2.0], [1.0, 2.0], 0, 0.125, 500_000),
        ([1.0, 2.5], [1.0, 2.0], 0.125, 0, 500_000),
    ],
)
def test_mismatched_elements_tolerance_boundaries(
    actual: list[float], expected: list[float], rtol: float, atol: float, ppm: float
):
    mismatched, message, error = evaluate_mismatched_elements(
        Tensor(np.asarray(actual), dims=["x"]),
        Tensor(np.asarray(expected), dims=["x"]),
        rtol=rtol,
        atol=atol,
        mismatched_tol=0,
        name="output",
    )
    assert error is None
    assert mismatched == ppm
    assert ("all elements agree" in message) == (ppm == 0)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="evaluate_mismatched_elements reports NaN predictions as matching finite references",
)
@pytest.mark.parametrize("nan_in_reference", [False, True])
def test_mismatched_elements_detects_nan(nan_in_reference: bool):
    actual = Tensor(
        np.array([np.nan if not nan_in_reference else 0.0, 1.0]), dims=["x"]
    )
    expected = Tensor(np.array([np.nan if nan_in_reference else 0.0, 1.0]), dims=["x"])
    mismatched, _, error = evaluate_mismatched_elements(
        actual, expected, rtol=1e-3, atol=1e-3, mismatched_tol=0, name="output"
    )
    assert error is None
    assert mismatched == 500_000
