from bioimageio.spec.shared import resolve_rdf_source, yaml


def test_error_for_wrong_shape(stardist_wrong_shape):
    from bioimageio.core.resource_tests import test_model

    summary = test_model(stardist_wrong_shape)
    expected_error_message = (
        "Shape (1, 512, 512, 33) of test output 0 'output' does not match output shape description: "
        "ImplicitOutputShape(reference_tensor='input', "
        "scale=[1.0, 1.0, 1.0, 0.0], offset=[1.0, 1.0, 1.0, 33.0])."
    )
    assert summary["error"] == expected_error_message


def test_error_for_wrong_shape2(stardist_wrong_shape2):
    from bioimageio.core.resource_tests import test_model

    summary = test_model(stardist_wrong_shape2)
    expected_error_message = (
        "Shape (1, 512, 512, 1) of test input 0 'input' does not match input shape description: "
        "ParametrizedInputShape(min=[1, 80, 80, 1], step=[0, 17, 17, 0])."
    )
    assert summary["error"] == expected_error_message


def test_test_model(any_model):
    from bioimageio.core.resource_tests import test_model

    summary = test_model(any_model)
    assert summary["error"] is None


def test_test_resource(any_model):
    from bioimageio.core.resource_tests import test_resource

    summary = test_resource(any_model)
    assert summary["error"] is None


def test_output_shape_with_offset():
    from bioimageio.core.resource_tests import test_resource

    url = "https://zenodo.org/record/6383430/files/rdf.yaml"
    data, name, root = resolve_rdf_source(url)
    data["root_path"] = root
    data["outputs"] = yaml.load(
        """
- axes: bczyx
  data_range: [0, 1]
  data_type: float32
  halo: [0, 0, 8, 16, 16]
  name: output0
  shape:
    offset: [0, 0.5, 0, 0, 0]  # fixed channel diff
    reference_tensor: raw
    scale: [1, 1, 1, 1, 1] 
    """
    )
    summary = test_resource(data)
    assert summary["status"] == "passed", summary
