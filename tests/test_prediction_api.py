from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
from dask.delayed import delayed  # pyright: ignore[reportUnknownVariableType]

from bioimageio.core import AxisId, MemberId, Sample, Tensor, predict
from bioimageio.core._model_adapter import ModelAdapter
from bioimageio.core._prediction_pipeline import PredictionPipeline
from bioimageio.core.common import PerMember
from bioimageio.core.prediction import predict_many
from bioimageio.core.proc_ops import ScaleLinear
from bioimageio.core.tensor import Array
from bioimageio.spec.model import v0_5


class _IncrementAdapter(ModelAdapter):
    def load(self) -> None:
        super().load()

    def unload(self) -> None:
        super().unload()

    def forward(self, inputs: PerMember[Tensor | Array | None]) -> PerMember[Tensor]:
        assert self.is_loaded
        tensor = inputs[MemberId("input")]
        assert isinstance(tensor, Tensor)
        return {MemberId("output"): tensor + 1}


@pytest.fixture
def pipeline():
    model = v0_5.ModelDescr.model_construct(
        name="Increment with processing",
        inputs=[
            v0_5.InputTensorDescr(
                id=MemberId("input"),
                axes=[
                    v0_5.SpaceInputAxis(
                        id=AxisId("x"), size=v0_5.ParameterizedSize(min=2, step=2)
                    )
                ],
            )
        ],
        outputs=[
            v0_5.OutputTensorDescr(
                id=MemberId("output"),
                axes=[
                    v0_5.SpaceOutputAxis(
                        id=AxisId("x"),
                        size=v0_5.SizeReference(
                            tensor_id=MemberId("input"), axis_id=AxisId("x")
                        ),
                    )
                ],
            )
        ],
    )
    return PredictionPipeline(
        name=model.name,
        model_description=model,
        model_adapter=_IncrementAdapter(model, devices=None),
        preprocessing=[ScaleLinear(MemberId("input"), MemberId("input"), gain=2)],
        postprocessing=[
            ScaleLinear(MemberId("output"), MemberId("output"), gain=3, offset=5)
        ],
    )


@pytest.mark.parametrize("blocking", [None, "parameterized", "fixed"])
def test_predict_applies_processing_once(
    pipeline: PredictionPipeline, blocking: str | None
):
    data = np.arange(8, dtype=np.float32)
    with pipeline:
        output = predict(
            model=pipeline,
            inputs=data,
            sample_id="sample",
            blocksize_parameter=1 if blocking == "parameterized" else None,
            input_block_shape={MemberId("input"): {AxisId("x"): 4}}
            if blocking == "fixed"
            else None,
        )
        assert output.id == "sample"
        np.testing.assert_array_equal(
            output.members[MemberId("output")].to_numpy(), (data * 2 + 1) * 3 + 5
        )
    assert not pipeline.is_loaded


@pytest.mark.parametrize("skip_preprocessing", [False, True])
@pytest.mark.parametrize("skip_postprocessing", [False, True])
def test_pipeline_respects_processing_flags(
    pipeline: PredictionPipeline,
    skip_preprocessing: bool,
    skip_postprocessing: bool,
):
    data = np.arange(8, dtype=np.float32)
    sample = Sample(
        id="sample", members={MemberId("input"): Tensor(data, dims=["x"])}, stat={}
    )
    expected = data if skip_preprocessing else data * 2
    expected = expected + 1
    if not skip_postprocessing:
        expected = expected * 3 + 5
    with pipeline:
        output = pipeline.predict_without_blocking(
            sample,
            skip_preprocessing=skip_preprocessing,
            skip_postprocessing=skip_postprocessing,
        )
        np.testing.assert_array_equal(
            output.members[MemberId("output")].to_numpy(), expected
        )


@pytest.mark.parametrize(
    "pre, post, error, message",
    [
        ("cache.zarr", None, ValueError, "Missing"),
        ("{member_id}.npy", None, NotImplementedError, "suffix"),
        (None, "cache.zarr", ValueError, "Missing"),
        (None, "{member_id}.npy", NotImplementedError, "suffix"),
        ("{member_id}.zarr", "{member_id}.zarr", ValueError, "must be different"),
    ],
)
def test_predict_rejects_invalid_cache_paths(
    pipeline: PredictionPipeline,
    pre: str | None,
    post: str | None,
    error: type[Exception],
    message: str,
):
    with pytest.raises(error, match=message):
        _ = predict(
            model=pipeline,
            inputs=np.arange(8),
            cache_path_after_preprocessing=pre,
            cache_path_before_postprocessing=post,
        )
    assert not pipeline.is_loaded


def test_predict_cache_reads_wait_for_lazy_writes(
    pipeline: PredictionPipeline, tmp_path: Path
):
    zarr = pytest.importorskip("zarr")
    calls: list[str] = []
    data = np.arange(8, dtype=np.float32) + 10

    def load():
        calls.append("load")
        return data

    lazy_data = da.from_delayed(delayed(load)(), shape=data.shape, dtype=data.dtype)
    with pipeline:
        output = predict(
            model=pipeline,
            inputs={MemberId("input"): Tensor(lazy_data, dims=["x"])},
            cache_path_after_preprocessing=tmp_path / "pre_{member_id}.zarr",
            cache_path_before_postprocessing=tmp_path / "post_{member_id}.zarr",
        )
        assert calls == []
        np.testing.assert_array_equal(
            output.members[MemberId("output")].to_numpy(), (data * 2 + 1) * 3 + 5
        )
        assert calls == ["load"]
    np.testing.assert_array_equal(
        zarr.open_array(tmp_path / "pre_input.zarr", mode="r")[:], data * 2
    )
    np.testing.assert_array_equal(
        zarr.open_array(tmp_path / "post_output.zarr", mode="r")[:], data * 2 + 1
    )


@pytest.mark.parametrize("save_output", [False, True])
@pytest.mark.parametrize(
    "sample_id, expected_ids",
    [
        ("sample", ["sample000", "sample001"]),
        ("image{i:02}", ["image00", "image01"]),
        (["first", "second"], ["first", "second"]),
    ],
)
def test_predict_many_preserves_sample_ids_and_values(
    pipeline: PredictionPipeline,
    tmp_path: Path,
    sample_id: str | list[str],
    expected_ids: list[str],
    save_output: bool,
):
    arrays = [np.arange(8, dtype=np.float32), np.arange(8, dtype=np.float32) + 10]
    output_path = tmp_path / "{sample_id}_{member_id}.npy" if save_output else None
    with pipeline:
        outputs = list(
            predict_many(
                model=pipeline,
                inputs=iter(arrays),
                sample_id=sample_id,
                output_path=output_path,
            )
        )
        assert len(outputs) == len(arrays)
        for result, data, expected_id in zip(outputs, arrays, expected_ids):
            if save_output:
                assert isinstance(result, Mapping)
                path = result[MemberId("output")]
                assert path == tmp_path / f"{expected_id}_output.npy"
                actual = np.load(path)
            else:
                assert isinstance(result, Sample)
                assert result.id == expected_id
                actual = result.members[MemberId("output")].to_numpy()
            np.testing.assert_array_equal(actual, (data * 2 + 1) * 3 + 5)


def test_predict_many_rejects_output_path_without_sample_id(
    pipeline: PredictionPipeline,
):
    with pytest.raises(ValueError, match="Missing.*sample_id"):
        _ = list(predict_many(model=pipeline, inputs=[], output_path="output.npy"))
    assert not pipeline.is_loaded
