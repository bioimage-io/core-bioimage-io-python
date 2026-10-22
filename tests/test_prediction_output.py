from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bioimageio.core import AxisId, Sample, Tensor, predict
from bioimageio.core._model_adapter import ModelAdapter
from bioimageio.core._prediction_pipeline import PredictionPipeline
from bioimageio.core.common import PerMember
from bioimageio.core.io import load_tensor
from bioimageio.core.stat_measures import Stat
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


@pytest.mark.parametrize("blocking", [None, "parameterized", "fixed"])
@pytest.mark.parametrize("output_scale", [1, 2])
@pytest.mark.parametrize("use_predict", [True, False])
def test_predict_preserves_tiff_resolution(
    tmp_path: Path, blocking: str | None, output_scale: int, use_predict: bool
):
    import sys

    if sys.version_info < (3, 12):
        pytest.skip("TIFF loading and saving requires Python >=3.12")
    tifffile = pytest.importorskip("tifffile")
    input_id, output_id = v0_5.TensorId("input"), v0_5.TensorId("output")
    axes = [AxisId("y"), AxisId("x")]
    model = v0_5.ModelDescr.model_construct(
        name="Dummy downsampling",
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
                        id=a,
                        scale=output_scale,
                        size=v0_5.SizeReference(tensor_id=input_id, axis_id=a),
                    )
                    for a in axes
                ],
            )
        ],
    )

    class Adapter(_IdentityAdapter):
        def forward(
            self, inputs: PerMember[Tensor | Array | None]
        ) -> PerMember[Tensor]:
            tensor = inputs[input_id]
            assert isinstance(tensor, Tensor)
            assert tensor.get_physical_scale() == {"y": 0.25, "x": 0.2}
            assert tensor.get_physical_scale_unit() == {"y": "um", "x": "um"}
            # Like a real model adapter, return a new tensor without input attributes.
            return {
                output_id: Tensor(
                    tensor.as_dask().data.data[::output_scale, ::output_scale],
                    dims=axes,
                )
            }

    from bioimageio.core.proc_ops import Sigmoid

    class CalibratedSigmoid(Sigmoid):
        def _apply(self, x: Tensor, stat: Stat) -> Tensor:
            scale = output_scale if self.input == output_id else 1
            assert x.get_physical_scale() == pytest.approx(
                {"y": 0.25 * scale, "x": 0.2 * scale}
            )
            assert x.get_physical_scale_unit() == {"y": "um", "x": "um"}
            return super()._apply(x, stat)

    pipeline = PredictionPipeline(
        name=model.name,
        model_description=model,
        preprocessing=[CalibratedSigmoid(input=input_id, output=input_id)],
        postprocessing=[CalibratedSigmoid(input=output_id, output=output_id)],
        model_adapter=Adapter(model, devices=None),
    )
    data = np.zeros((12, 20), dtype=np.float32)
    source, target = tmp_path / "input.tif", tmp_path / "output.tif"
    tifffile.imwrite(
        source, data, imagej=True, resolution=(5, 4), metadata={"unit": "um"}
    )
    with pipeline:
        if use_predict:
            _ = pytest.importorskip("zarr")
            _ = predict(
                model=pipeline,
                inputs=source,
                output_path=target,
                cache_path_after_preprocessing=tmp_path / "pre_{member_id}.zarr",
                cache_path_before_postprocessing=tmp_path / "post_{member_id}.zarr",
                blocksize_parameter=1 if blocking == "parameterized" else None,
                input_block_shape={input_id: {a: 8 for a in axes}}
                if blocking == "fixed"
                else None,
            )
        else:
            from bioimageio.core.io import save_tensor

            sample = Sample(
                members={input_id: load_tensor(source)}, stat={}, id="sample"
            )
            if blocking == "parameterized":
                output = pipeline.predict_with_blocking(sample, ns=1)
            elif blocking == "fixed":
                output = pipeline.predict_with_fixed_blocking(
                    sample, input_block_shape={input_id: {a: 8 for a in axes}}
                )
            else:
                output = pipeline.predict_without_blocking(sample)
            save_tensor(target, output.members[output_id], io_lib="tifffile")

    actual = load_tensor(target, io_lib="tifffile")
    assert actual.get_physical_scale() == pytest.approx(
        {"y": 0.25 * output_scale, "x": 0.2 * output_scale}
    )
    assert actual.get_physical_scale_unit() == {"y": "um", "x": "um"}
    np.testing.assert_allclose(actual.to_numpy(), 1 / (1 + np.exp(-0.5)))
    with tifffile.TiffFile(target) as tif:
        for tag, resolution in [("XResolution", 5), ("YResolution", 4)]:
            numerator, denominator = tif.pages[0].tags[tag].value
            assert numerator / denominator == pytest.approx(resolution / output_scale)


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
