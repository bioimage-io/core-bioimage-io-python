from pathlib import Path

from bioimageio.spec import InvalidDescr


def test_error_for_wrong_shape(stardist_wrong_shape: Path):
    from bioimageio.core.resource_tests import test_model

    summary = test_model(stardist_wrong_shape)
    expected_error_message = (
        "Shape (1, 512, 512, 33) of test output 0 'output' does not match output shape description: "
        "ImplicitOutputShape(reference_tensor='input', "
        "scale=[1.0, 1.0, 1.0, 0.0], offset=[1.0, 1.0, 1.0, 33.0])."
    )
    assert summary.details[0].errors[0].msg == expected_error_message


def test_error_for_wrong_shape2(stardist_wrong_shape2: Path):
    from bioimageio.core.resource_tests import test_model

    summary = test_model(stardist_wrong_shape2)
    expected_error_message = (
        "Shape (1, 512, 512, 1) of test input 0 'input' does not match input shape description: "
        "ParameterizedInputShape(min=[1, 80, 80, 1], step=[0, 17, 17, 0])."
    )
    assert summary.details[0].errors[0].msg == expected_error_message


def test_test_model(any_model: Path):
    from bioimageio.core.resource_tests import test_model

    summary = test_model(any_model)
    assert summary.status == "passed"


def test_test_resource(any_model: Path):
    from bioimageio.core.resource_tests import test_description

    summary = test_description(any_model)
    assert summary.status == "passed"


def test_validation_section_warning(unet2d_nuclei_broad_model: str, tmp_path: Path):
    from bioimageio.core import load_description
    from bioimageio.core.resource_tests import test_description

    model = load_description(unet2d_nuclei_broad_model)
    assert not isinstance(model, InvalidDescr)
    summary = test_description(model)
    assert summary.name == "Test documentation completeness."
    assert summary.warnings == {"documentation": "No '# Validation' (sub)section found."}
    assert summary.status == "passed"

    doc_with_validation = tmp_path / "doc.md"
    _ = doc_with_validation.write_text("# Validation\nThis is a section about how to validate the model on new data")
    model.documentation = doc_with_validation
    summary = test_description(model)
    assert summary.name == "Test documentation completeness."
    assert summary.warnings == {}
    assert summary.status == "passed"


def test_issue289(unet2d_nuclei_broad_model: str):
    """test for failure case from https://github.com/bioimage-io/core-bioimage-io-python/issues/289"""
    # remote model is a pytorch model, needing unet2d_nuclei_broad_model skips the test when needed
    _ = unet2d_nuclei_broad_model

    from bioimageio.core.resource_tests import test_model

    doi = "10.5281/zenodo.6287342"
    summary = test_model(doi)
    assert summary.status == "passed"
