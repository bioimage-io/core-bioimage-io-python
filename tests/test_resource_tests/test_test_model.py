def test_error_for_wrong_shape(stardist_wrong_shape):
    from bioimageio.core.resource_tests import test_model

    summary = test_model(stardist_wrong_shape)[0]
    expected_error_message = (
        "Shape (1, 512, 512, 33) of test output 0 'output' does not match output shape description: "
        "ImplicitOutputShape(reference_tensor='input', "
        "scale=[1.0, 1.0, 1.0, 0.0], offset=[1.0, 1.0, 1.0, 33.0])."
    )
    assert summary["error"] == expected_error_message


def test_error_for_wrong_shape2(stardist_wrong_shape2):
    from bioimageio.core.resource_tests import test_model

    summary = test_model(stardist_wrong_shape2)[0]
    expected_error_message = (
        "Shape (1, 512, 512, 1) of test input 0 'input' does not match input shape description: "
        "ParametrizedInputShape(min=[1, 80, 80, 1], step=[0, 17, 17, 0])."
    )
    assert summary["error"] == expected_error_message


def test_test_model(any_model):
    from bioimageio.core.resource_tests import test_model

    summary = test_model(any_model)[0]
    assert summary["error"] is None


def test_test_resource(any_model):
    from bioimageio.core.resource_tests import test_resource

    summary = test_resource(any_model)[0]
    assert summary["error"] is None
