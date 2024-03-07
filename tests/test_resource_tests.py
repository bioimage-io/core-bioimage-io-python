from pathlib import Path

from bioimageio.spec import InvalidDescr


def test_error_for_wrong_shape(stardist_wrong_shape: Path):
    from bioimageio.core._resource_tests import test_model

    summary = test_model(stardist_wrong_shape)
    expected_error_message = (
        "Shape (1, 512, 512, 33) of test output 0 'output' does not match output shape description: "
        "ImplicitOutputShape(reference_tensor='input', "
        "scale=[1.0, 1.0, 1.0, 0.0], offset=[1.0, 1.0, 1.0, 33.0])."
    )
    assert summary.details[0].errors[0].msg == expected_error_message


def test_error_for_wrong_shape2(stardist_wrong_shape2: Path):
    from bioimageio.core._resource_tests import test_model

    summary = test_model(stardist_wrong_shape2)
    expected_error_message = (
        "Shape (1, 512, 512, 1) of test input 0 'input' does not match input shape description: "
        "ParameterizedInputShape(min=[1, 80, 80, 1], step=[0, 17, 17, 0])."
    )
    assert summary.details[0].errors[0].msg == expected_error_message


def test_test_model(any_model: Path):
    from bioimageio.core._resource_tests import test_model

    summary = test_model(any_model)
    assert summary.status == "passed"


def test_test_resource(any_model: Path):
    from bioimageio.core._resource_tests import test_description

    summary = test_description(any_model)
    assert summary.status == "passed"
