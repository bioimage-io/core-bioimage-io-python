from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bioimageio.core import AxisId, Sample, Tensor, predict
from bioimageio.core._model_adapter import ModelAdapter
from bioimageio.core._prediction_pipeline import PredictionPipeline
from bioimageio.core.common import PerMember
from bioimageio.core.io import load_tensor
from bioimageio.core.tensor import Array
from bioimageio.spec.model import v0_5


class _IdentityAdapter(ModelAdapter):
    def load(self) -> None:
        super().load()

    def unload(self) -> None:
        super().unload()

    def forward(self, inputs: PerMember[Tensor | Array | None]) -> PerMember[Tensor]:
        tensor = inputs[v0_5.TensorId("input")]
        assert isinstance(tensor, Tensor)
        assert all(size >= 4 and size % 4 == 0 for size in tensor.shape)
        return {v0_5.TensorId("output"): tensor.as_dask()}


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_predict_overwrites_zarr_with_different_shape(tmp_path: Path, zarr_format: int):
    zarr = pytest.importorskip("zarr")
    _ = pytest.importorskip("clearscale")
    input_id, output_id = v0_5.TensorId("input"), v0_5.TensorId("output")
    axes = [AxisId("y"), AxisId("x")]
    model = v0_5.ModelDescr.model_construct(
        name="Dummy identity",
        inputs=[
            v0_5.InputTensorDescr(
                id=input_id,
                axes=[
                    v0_5.SpaceInputAxis(
                        id=a, size=v0_5.ParameterizedSize(min=4, step=4)
                    )
                    for a in axes
                ],
            )
        ],
        outputs=[
            v0_5.OutputTensorDescr(
                id=output_id,
                axes=[
                    v0_5.SpaceOutputAxis(
                        id=a, size=v0_5.SizeReference(tensor_id=input_id, axis_id=a)
                    )
                    for a in axes
                ],
            )
        ],
    )
    adapter = _IdentityAdapter(model, devices=None)
    pipeline = PredictionPipeline(
        name="Dummy identity",
        model_description=model,
        preprocessing=[],
        postprocessing=[],
        model_adapter=adapter,
    )
    output_path = tmp_path / "prediction.zarr" / "s0"
    with zarr.config.set({"default_zarr_format": zarr_format}), pipeline:
        for shape in [(6, 10), (10, 14)]:
            data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            sample = Sample(
                members={input_id: Tensor(data, dims=axes).as_dask()},
                stat={},
                id="sample",
            )
            paths = predict(
                model=pipeline,
                inputs=sample,
                blocksize_parameter=2,
                output_path=output_path,
            )
            np.testing.assert_array_equal(
                load_tensor(paths[output_id], io_lib="clearscale").to_numpy(), data
            )
