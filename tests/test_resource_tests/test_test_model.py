import pathlib

import pytest


def test_error_for_wrong_shape(stardist_wrong_shape):
    from bioimageio.core.resource_tests import test_model

    summary = test_model(stardist_wrong_shape)[-1]
    expected_error_message = (
        "Shape (1, 512, 512, 33) of test output 0 'output' does not match output shape description: "
        "ImplicitOutputShape(reference_tensor='input', "
        "scale=[1.0, 1.0, 1.0, 0.0], offset=[1.0, 1.0, 1.0, 33.0])."
    )
    assert summary["error"] == expected_error_message


def test_error_for_wrong_shape2(stardist_wrong_shape2):
    from bioimageio.core.resource_tests import test_model

    summary = test_model(stardist_wrong_shape2)[-1]
    expected_error_message = (
        "Shape (1, 512, 512, 1) of test input 0 'input' does not match input shape description: "
        "ParametrizedInputShape(min=[1, 80, 80, 1], step=[0, 17, 17, 0])."
    )
    assert summary["error"] == expected_error_message


def test_test_model(any_model):
    from bioimageio.core.resource_tests import test_model

    summary = test_model(any_model)
    assert all([s["status"] for s in summary])


def test_test_resource(any_model):
    from bioimageio.core.resource_tests import test_resource

    summary = test_resource(any_model)
    assert all([s["status"] for s in summary])


def test_validation_section_warning(unet2d_nuclei_broad_model, tmp_path: pathlib.Path):
    from bioimageio.core.resource_tests import test_resource
    from bioimageio.core import load_resource_description

    model = load_resource_description(unet2d_nuclei_broad_model)

    summary = test_resource(model)[2]
    assert summary["name"] == "Test documentation completeness."
    assert summary["warnings"] == {"documentation": "No '# Validation' (sub)section found."}
    assert summary["status"] == "passed"

    doc_with_validation = tmp_path / "doc.md"
    doc_with_validation.write_text("# Validation\nThis is a section about how to validate the model on new data")
    model.documentation = doc_with_validation
    summary = test_resource(model)[2]
    assert summary["name"] == "Test documentation completeness."
    assert summary["warnings"] == {}
    assert summary["status"] == "passed"


@pytest.mark.skipif(pytest.skip_torch, reason="requires torch")
def test_issue289():
    """test for failure case from https://github.com/bioimage-io/core-bioimage-io-python/issues/289"""
    import bioimageio.core
    from bioimageio.core.resource_tests import test_model

    doi = "10.5281/zenodo.6287342"
    model_resource = bioimageio.core.load_resource_description(doi)
    test_result = test_model(model_resource)
    assert all([t["status"] == "passed" for t in test_result])
