from pathlib import Path

import numpy as np

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


def test_test_description_runtime_env(unet2d_nuclei_broad_model: str):
    from bioimageio.core._resource_tests import test_description

    summary = test_description(unet2d_nuclei_broad_model, runtime_env="as-described")

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
