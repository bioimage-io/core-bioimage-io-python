from pathlib import Path

import imageio
import numpy as np
from numpy.testing import assert_array_almost_equal

from bioimageio.core import load_resource_description
from bioimageio.core.resource_io.nodes import Model


def test_test_model(any_model):
    from bioimageio.core.resource_tests import test_model

    assert test_model(any_model)


def test_test_resource(any_model):
    from bioimageio.core.resource_tests import test_resource

    assert test_resource(any_model)


def test_predict_image(any_model, tmpdir):
    from bioimageio.core.prediction import predict_image

    spec = load_resource_description(any_model)
    assert isinstance(spec, Model)
    inputs = spec.test_inputs

    outputs = [Path(tmpdir) / f"out{i}.npy" for i in range(len(spec.test_outputs))]
    predict_image(any_model, inputs, outputs)
    for out_path in outputs:
        assert out_path.exists()

    result = [np.load(str(p)) for p in outputs]
    expected = [np.load(str(p)) for p in spec.test_outputs]
    for res, exp in zip(result, expected):
        assert_array_almost_equal(res, exp, decimal=4)


def test_predict_image_with_weight_format(unet2d_fixed_shape_or_not, tmpdir):
    from bioimageio.core.prediction import predict_image

    spec = load_resource_description(unet2d_fixed_shape_or_not)
    assert isinstance(spec, Model)
    inputs = spec.test_inputs

    outputs = [Path(tmpdir) / f"out{i}.npy" for i in range(len(spec.test_outputs))]
    predict_image(unet2d_fixed_shape_or_not, inputs, outputs, weight_format="pytorch_state_dict")
    for out_path in outputs:
        assert out_path.exists()

    result = [np.load(str(p)) for p in outputs]
    expected = [np.load(str(p)) for p in spec.test_outputs]
    for res, exp in zip(result, expected):
        assert_array_almost_equal(res, exp, decimal=4)


def _test_predict_with_padding(model, tmp_path):
    from bioimageio.core.prediction import predict_image

    spec = load_resource_description(model)
    assert isinstance(spec, Model)

    input_spec, output_spec = spec.inputs[0], spec.outputs[0]
    channel_axis = input_spec.axes.index("c")
    channel_first = channel_axis == 1
    assert output_spec.shape.scale[channel_axis] == 0
    n_channels = int(2 * output_spec.shape.offset[channel_axis])

    if channel_first:
        image = np.load(str(spec.test_inputs[0]))[0, 0]
    else:
        image = np.load(str(spec.test_inputs[0]))[0, ..., 0]
    original_shape = image.shape
    assert image.ndim == 2

    # write the padded image
    image = image[3:-2, 1:-12]
    in_path = tmp_path / "in.tif"
    out_path = tmp_path / "out.tif"
    imageio.imwrite(in_path, image)

    def check_result():
        if n_channels == 1:
            assert out_path.exists()
            res = imageio.imread(out_path)
            assert res.shape == image.shape
        else:
            path = str(out_path)
            for c in range(n_channels):
                channel_out_path = Path(path.replace(".tif", f"-c{c}.tif"))
                assert channel_out_path.exists()
                res = imageio.imread(channel_out_path)
                assert res.shape == image.shape

    # test with dynamic padding
    predict_image(model, in_path, out_path, padding={"x": 16, "y": 16, "mode": "dynamic"})
    check_result()

    # test with fixed padding
    predict_image(
        model, in_path, out_path, padding={"x": original_shape[0], "y": original_shape[1], "mode": "fixed"}
    )
    check_result()

    # test with automated padding
    predict_image(model, in_path, out_path, padding=True)
    check_result()


# prediction with padding with the parameters above may not be suited for any model
# so we only run it for the pytorch unet2d here
def test_predict_image_with_padding(unet2d_fixed_shape_or_not, tmp_path):
    _test_predict_with_padding(unet2d_fixed_shape_or_not, tmp_path)


def test_predict_image_with_padding_channel_last(stardist, tmp_path):
    _test_predict_with_padding(stardist, tmp_path)


def _test_predict_image_with_tiling(model, tmp_path):
    from bioimageio.core.prediction import predict_image

    spec = load_resource_description(model)
    assert isinstance(spec, Model)
    inputs = spec.test_inputs
    assert len(inputs) == 1
    exp = np.load(str(spec.test_outputs[0]))

    out_path = tmp_path.with_suffix(".npy")

    def check_result():
        assert out_path.exists()
        res = np.load(out_path)
        assert res.shape == exp.shape
        # mean deviation should be smaller 0.1
        mean_deviation = np.abs(res - exp).mean()
        assert mean_deviation < 0.1

    # with tiling config
    tiling = {"halo": {"x": 32, "y": 32}, "tile": {"x": 256, "y": 256}}
    predict_image(model, inputs, [out_path], tiling=tiling)
    check_result()

    # with tiling determined from spec
    predict_image(model, inputs, [out_path], tiling=True)
    check_result()


# prediction with tiling with the parameters above may not be suited for any model
# so we only run it for the pytorch unet2d here
def test_predict_image_with_tiling(unet2d_nuclei_broad_model, tmp_path):
    _test_predict_image_with_tiling(unet2d_nuclei_broad_model, tmp_path)


def test_predict_image_with_tiling_channel_last(stardist, tmp_path):
    _test_predict_image_with_tiling(stardist, tmp_path)


def test_predict_images(unet2d_nuclei_broad_model, tmp_path):
    from bioimageio.core.prediction import predict_images

    n_images = 5
    shape = (256, 256)

    in_paths = []
    out_paths = []
    for i in range(n_images):
        in_path = tmp_path / f"in{i}.tif"
        im = np.random.randint(0, 255, size=shape).astype("uint8")
        imageio.imwrite(in_path, im)
        in_paths.append(in_path)
        out_paths.append(tmp_path / f"out{i}.tif")
    predict_images(unet2d_nuclei_broad_model, in_paths, out_paths)

    for outp in out_paths:
        assert outp.exists()
        out = imageio.imread(outp)
        assert out.shape == shape
