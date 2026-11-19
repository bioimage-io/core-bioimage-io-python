from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from loguru import logger
from numpy.typing import NDArray

from bioimageio.core import Tensor, __version__
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


def test_test_description_runtime_env(unet2d_nuclei_broad_model: str):
    from bioimageio.core._resource_tests import test_description

    summary = test_description(unet2d_nuclei_broad_model, runtime_env="as-described")

    if summary.status != "passed":
        summary.display()
        msg = f"test_description failed with status: {summary.status}"
        if __version__ == "0.12.0":
            logger.error(
                "Expected failure due to missing pin in last spec release: {}", msg
            )
        else:
            raise AssertionError(msg)


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


def test_reproducibility_rejects_nan_output():
    from bioimageio.core._resource_tests import evaluate_mismatched_elements

    actual = Tensor(np.full((2,), np.nan), dims=("x",))
    expected = Tensor(np.ones((2,)), dims=("x",))

    mismatched_ppm, msg, error_msg = evaluate_mismatched_elements(
        actual, expected, rtol=1e-3, atol=1e-3, mismatched_tol=0, name="output"
    )

    assert error_msg is None
    assert mismatched_ppm == 1_000_000
    assert "2 of 2 elements (100.0%) disagree" in msg


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


@pytest.mark.parametrize(
    "concatenable, corrupt_blocks, corrupt_expected, stop_early, halo, batch_size, too_small, reference_applicable, expected_statuses",
    [
        (
            "reproducible",
            False,
            False,
            False,
            0,
            None,
            False,
            False,
            ["passed", "passed"],
        ),
        (
            "reproducible",
            True,
            False,
            True,
            0,
            None,
            False,
            False,
            ["passed", "failed"],
        ),
        (
            "reproducible",
            False,
            False,
            False,
            2,
            None,
            False,
            False,
            ["passed", "passed"],
        ),
        (
            "reproducible",
            "edges",
            False,
            False,
            2,
            None,
            False,
            False,
            ["passed", "passed"],
        ),
        (
            "reproducible",
            True,
            False,
            False,
            2,
            None,
            False,
            False,
            ["passed", "failed"],
        ),
        ("reproducible", False, "edges", True, 2, None, False, False, ["failed"]),
        (False, False, False, False, 0, None, False, False, ["passed"]),
        (False, False, False, False, 0, 2, False, False, ["passed", "passed"]),
        ("reproducible", False, True, True, 0, None, False, False, ["failed"]),
        (
            "reproducible",
            False,
            False,
            False,
            0,
            None,
            True,
            False,
            ["passed", "failed"],
        ),
        (
            "reproducible",
            False,
            False,
            False,
            2,
            None,
            True,
            False,
            ["passed", "failed"],
        ),
        (False, False, False, False, 0, 1, True, False, ["passed", "passed"]),
        (False, True, False, False, 0, 1, True, False, ["passed", "failed"]),
        ("applicable", False, False, False, 0, 1, True, False, ["passed", "passed"]),
        (False, False, False, False, 0, 1, True, True, ["passed", "passed"]),
        ("reproducible", False, False, False, 0, 1, True, False, ["passed", "failed"]),
        ("reproducible", False, False, False, 0, 2, True, False, ["passed", "passed"]),
        (True, True, False, False, 0, None, False, False, ["passed"]),
        (True, False, False, False, 0, None, True, False, ["passed"]),
        (True, True, False, False, 0, 2, False, False, ["passed", "passed"]),
        (True, False, False, False, 0, 1, True, False, ["passed", "passed"]),
        (True, True, False, False, 0, 1, True, False, ["passed", "failed"]),
        ("applicable", True, False, False, 0, None, False, False, ["passed"]),
        ("applicable", True, False, False, 0, 2, False, False, ["passed", "passed"]),
        (
            "reproducible",
            True,
            False,
            False,
            0,
            None,
            False,
            False,
            ["passed", "failed"],
        ),
        ("reproducible", True, False, False, 0, 2, False, True, ["passed", "passed"]),
    ],
)
def test_recreate_outputs_blockwise(
    monkeypatch: pytest.MonkeyPatch,
    concatenable: bool | Literal["applicable", "reproducible"],
    corrupt_blocks: bool | Literal["edges"],
    corrupt_expected: bool | Literal["edges"],
    stop_early: bool,
    halo: int,
    batch_size: int | None,
    too_small: bool,
    reference_applicable: bool,
    expected_statuses: list[str],
):
    from bioimageio.core import (
        AxisId,
        Sample,
        _resource_tests,  # pyright: ignore[reportPrivateUsage]
    )
    from bioimageio.core._model_adapter import ModelAdapter
    from bioimageio.core._prediction_pipeline import PredictionPipeline
    from bioimageio.core.common import PerMember
    from bioimageio.core.tensor import Array
    from bioimageio.spec.model import v0_5
    from bioimageio.spec.summary import ValidationSummary

    input_id, output_id = v0_5.TensorId("input"), v0_5.TensorId("output")
    x, y = AxisId("x"), AxisId("y")
    dims = (AxisId("batch"), x, y) if batch_size is not None else (x, y)
    model = v0_5.ModelDescr.model_construct(
        name="Identity",
        inputs=[
            v0_5.InputTensorDescr(
                id=input_id,
                axes=[
                    *([v0_5.BatchAxis()] if batch_size is not None else []),
                    v0_5.SpaceInputAxis(
                        id=x,
                        size=v0_5.ParameterizedSize(min=4, step=4),
                        concatenable=concatenable,
                    ),
                    v0_5.SpaceInputAxis(
                        id=y,
                        size=v0_5.ParameterizedSize(min=4, step=4),
                        concatenable=False,
                    ),
                ],
            ),
            *(
                [
                    v0_5.InputTensorDescr(
                        id=v0_5.TensorId("paired"),
                        axes=[
                            v0_5.BatchAxis(),
                            v0_5.SpaceInputAxis(
                                id=x,
                                size=v0_5.SizeReference(tensor_id=input_id, axis_id=x),
                                concatenable="applicable",
                            ),
                            v0_5.SpaceInputAxis(
                                id=y,
                                size=v0_5.SizeReference(tensor_id=input_id, axis_id=y),
                                concatenable=False,
                            ),
                        ],
                    )
                ]
                if reference_applicable
                else []
            ),
        ],
        outputs=[
            v0_5.OutputTensorDescr(
                id=output_id,
                axes=[
                    *([v0_5.BatchAxis()] if batch_size is not None else []),
                    v0_5.SpaceOutputAxisWithHalo(
                        id=x,
                        size=v0_5.SizeReference(tensor_id=input_id, axis_id=x),
                        halo=halo,
                    )
                    if halo
                    else v0_5.SpaceOutputAxis(
                        id=x, size=v0_5.SizeReference(tensor_id=input_id, axis_id=x)
                    ),
                    v0_5.SpaceOutputAxisWithHalo(
                        id=y,
                        size=v0_5.SizeReference(tensor_id=input_id, axis_id=y),
                        halo=halo,
                    )
                    if "edges" in (corrupt_blocks, corrupt_expected)
                    else v0_5.SpaceOutputAxis(
                        id=y, size=v0_5.SizeReference(tensor_id=input_id, axis_id=y)
                    ),
                ],
            )
        ],
        weights=v0_5.WeightsDescr.model_construct(),
    )
    monkeypatch.setattr(
        model,
        "_validation_summary",
        ValidationSummary(
            name="Identity",
            source_name="test",
            type="model",
            format_version="0.5",
            status="passed",
            details=[],
        ),
    )
    shape: tuple[int, ...] = (
        (8 if halo else 4) if too_small else (12 if halo else 8),
        8,
    )
    if batch_size is not None:
        shape = (batch_size, *shape)
    data: NDArray[np.float32] = np.arange(np.prod(shape), dtype=np.float32).reshape(
        shape
    )
    block_shapes: list[tuple[int, ...]] = []

    class Adapter(ModelAdapter):
        def load(self) -> None:
            super().load()

        def unload(self) -> None:
            super().unload()

        def forward(
            self, inputs: PerMember[Tensor | Array | None]
        ) -> PerMember[Tensor]:
            tensor = inputs[input_id]
            assert isinstance(tensor, Tensor)
            block_shapes.append(tensor.shape)
            values: NDArray[np.float32] = np.array(
                tensor.to_numpy(), dtype=np.float32, copy=True
            )
            if corrupt_blocks and (
                tensor.tagged_shape[x] < shape[-2]
                or (
                    batch_size == 1
                    and concatenable in (False, True, "applicable")
                    and len(block_shapes) > 1
                )
            ):
                if corrupt_blocks == "edges":
                    for axis in (x, y):
                        axis_values = np.moveaxis(values, dims.index(axis), 0)
                        axis_values[:halo] += 1
                        axis_values[-halo:] += 1
                else:
                    values += 1
            return {output_id: Tensor(values, dims=dims)}

    def create_pipeline(**kwargs: object) -> PredictionPipeline:
        return PredictionPipeline(
            name=model.name,
            model_description=model,
            preprocessing=[],
            postprocessing=[],
            model_adapter=Adapter(model, devices=None),
        )

    def get_test_input(model: v0_5.ModelDescr) -> Sample:
        members = {input_id: Tensor(data, dims=dims)}
        if reference_applicable:
            members[v0_5.TensorId("paired")] = Tensor(data, dims=dims)
        return Sample(members=members, stat={}, id="test")

    def get_test_output(model: v0_5.ModelDescr) -> Sample:
        expected: NDArray[np.float32] = np.array(data, dtype=np.float32, copy=True)
        if corrupt_expected == "edges":
            axis_values = np.moveaxis(expected, dims.index(x), 0)
            axis_values[:halo] += 1
            axis_values[-halo:] += 1
        elif corrupt_expected:
            expected += 1
        return Sample(
            members={output_id: Tensor(expected, dims=dims)},
            stat={},
            id="test",
        )

    def get_conda_env(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(_resource_tests, "create_prediction_pipeline", create_pipeline)
    monkeypatch.setattr(_resource_tests, "get_test_input_sample", get_test_input)
    monkeypatch.setattr(_resource_tests, "get_test_output_sample", get_test_output)
    monkeypatch.setattr(_resource_tests, "get_conda_env", get_conda_env)

    passed = _resource_tests._test_recreate_test_outputs(  # pyright: ignore[reportPrivateUsage]
        model, "onnx", None, stop_early, working_dir=None, verbose=False
    )

    assert [d.status for d in model.validation_summary.details] == expected_statuses, [
        d.errors for d in model.validation_summary.details
    ]
    assert passed == all(status == "passed" for status in expected_statuses)
    if too_small and batch_size != 2 and concatenable == "reproducible":
        assert block_shapes == [data.shape]
        assert (
            "larger than the minimal block shape"
            in model.validation_summary.details[-1].errors[0].msg
        )
    elif len(expected_statuses) == 2:
        block_shape = (
            (1, *shape[1:]) if batch_size is not None else (8 if halo else 4, 8)
        )
        assert block_shapes == [data.shape, block_shape, block_shape]
    if (
        concatenable == "reproducible"
        and not corrupt_blocks
        and not corrupt_expected
        and not halo
        and not too_small
    ):
        assert _resource_tests._test_parametrized_inference(  # pyright: ignore[reportPrivateUsage]
            model, "onnx", None, stop_early=True
        )
