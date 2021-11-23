def test_error_for_wrong_shape(stardist_wrong_shape):
    from bioimageio.core.resource_tests import test_model

    summary = test_model(stardist_wrong_shape)
    assert (
        summary["error"]
        == "Shape of test input 0 'input' does not match input shape description: ParametrizedInputShape(min=[1, 16, 16, 1], step=[0, 16, 16, 0])"
    )
