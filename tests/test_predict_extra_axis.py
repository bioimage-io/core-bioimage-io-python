from pathlib import Path

import dask.config
import numpy as np
import pytest

from bioimageio.core import AxisId, Sample, Tensor, predict
from bioimageio.core._settings import settings
from bioimageio.spec.model import v0_5


@pytest.mark.parametrize("depth", [2, 8, 10])
def test_predict_2d_network_on_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, depth: int
):
    torch = pytest.importorskip("torch")
    network = torch.nn.Conv2d(1, 1, kernel_size=1)
    with torch.no_grad():
        network.weight.fill_(2)
        network.bias.fill_(1)
    weights = tmp_path / "network.pt"
    torch.jit.trace(network.eval(), torch.zeros(1, 1, 8, 8)).save(str(weights))

    input_id, output_id = v0_5.TensorId("input"), v0_5.TensorId("output")
    model = v0_5.ModelDescr.model_construct(
        name="Dummy 2D convolution",
        inputs=[
            v0_5.InputTensorDescr(
                id=input_id,
                axes=[
                    v0_5.BatchAxis(),
                    v0_5.ChannelAxis(channel_names=["channel0"]),
                    *[
                        v0_5.SpaceInputAxis(
                            id=AxisId(a), size=v0_5.ParameterizedSize(min=4, step=4)
                        )
                        for a in ("y", "x")
                    ],
                ],
            )
        ],
        outputs=[
            v0_5.OutputTensorDescr(
                id=output_id,
                axes=[
                    v0_5.BatchAxis(),
                    v0_5.ChannelAxis(channel_names=["channel0"]),
                    *[
                        v0_5.SpaceOutputAxis(
                            id=AxisId(a),
                            size=v0_5.SizeReference(
                                tensor_id=input_id, axis_id=AxisId(a)
                            ),
                        )
                        for a in ("y", "x")
                    ],
                ],
            )
        ],
        weights=v0_5.WeightsDescr(
            torchscript=v0_5.TorchscriptWeightsDescr(
                source=weights, pytorch_version=torch.__version__.split("+")[0]
            )
        ),
    )
    # Four full-resolution slices per chunk, so depth=10 leaves two slices.
    monkeypatch.setattr(settings, "processing_chunk_min_bytes", 4 * 8 * 8 * 4)
    data = np.arange(depth * 8 * 8, dtype=np.float32).reshape(depth, 8, 8)
    with dask.config.set(scheduler="synchronous"):
        result = predict(
            model=model,
            inputs={
                input_id: Tensor(
                    data[None, None], dims=["batch", "channel", "z", "y", "x"]
                )
            },
            blocksize_parameter=1,
            devices=["cpu"],
        )
        assert isinstance(result, Sample)
        output = result.members[output_id]
        assert output.tagged_shape[AxisId("z")] == depth
        actual = output.transpose(["z", "y", "x"], extra_dims="squeeze").to_numpy()
    np.testing.assert_array_equal(actual, 2 * data + 1)
