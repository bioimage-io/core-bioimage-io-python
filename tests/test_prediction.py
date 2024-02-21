from pathlib import Path

import imageio
import numpy as np
from numpy.testing import assert_array_almost_equal

from bioimageio.core import load_description
from bioimageio.core.resource_io.nodes import Model


def test_predict_image(any_model, tmpdir):
    from bioimageio.core.prediction import predict_image

    spec = load_description(any_model)
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

    spec = load_description(unet2d_fixed_shape_or_not)
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

    spec = load_description(model)
    assert isinstance(spec, Model)

    input_spec, output_spec = spec.inputs[0], spec.outputs[0]
    channel_axis = input_spec.axes.index("c")
    channel_first = channel_axis == 1

    image = np.load(str(spec.test_inputs[0]))
    assert image.shape[channel_axis] == 1
    if channel_first:
        image = image[0, 0]
    else:
        image = image[0, ..., 0]
    original_shape = image.shape
    assert image.ndim == 2

    if isinstance(output_spec.shape, list):
        n_channels = output_spec.shape[channel_axis]
    else:
        scale = output_spec.shape.scale[channel_axis]
        offset = output_spec.shape.offset[channel_axis]
        in_channels = 1
        n_channels = int(2 * offset + scale * in_channels)

    # write the padded image
    image = image[3:-2, 1:-12]
    in_path = tmp_path / "in.tif"
    out_path = tmp_path / "out.tif"
    imageio.imwrite(in_path, image)

    if hasattr(output_spec.shape, "scale"):
        scale = dict(zip(output_spec.axes, output_spec.shape.scale))
        offset = dict(zip(output_spec.axes, output_spec.shape.offset))
        spatial_axes = [ax for ax in output_spec.axes if ax in "xyz"]
        network_resizes = any(sc != 1 for ax, sc in scale.items() if ax in spatial_axes) or any(
            off != 0 for ax, off in offset.items() if ax in spatial_axes
        )
    else:
        network_resizes = False

    if network_resizes:
        exp_shape = tuple(int(sh * scale[ax] + 2 * offset[ax]) for sh, ax in zip(image.shape, spatial_axes))
    else:
        exp_shape = image.shape

    def check_result():
        if n_channels == 1:
            assert out_path.exists()
            res = imageio.imread(out_path)
            assert res.shape == exp_shape
        else:
            path = str(out_path)
            for c in range(n_channels):
                channel_out_path = Path(path.replace(".tif", f"-c{c}.tif"))
                assert channel_out_path.exists()
                res = imageio.imread(channel_out_path)
                assert res.shape == exp_shape

    # test with dynamic padding
    predict_image(model, in_path, out_path, padding={"x": 16, "y": 16, "mode": "dynamic"})
    check_result()

    # test with fixed padding
    predict_image(model, in_path, out_path, padding={"x": original_shape[0], "y": original_shape[1], "mode": "fixed"})
    check_result()

    # test with automated padding
    predict_image(model, in_path, out_path, padding=True)
    check_result()


# prediction with padding with the parameters above may not be suited for any model
# so we only run it for the pytorch unet2d here
def test_predict_image_with_padding(unet2d_fixed_shape_or_not, tmp_path):
    _test_predict_with_padding(unet2d_fixed_shape_or_not, tmp_path)


# and with different output shape
def test_predict_image_with_padding_diff_output_shape(unet2d_diff_output_shape, tmp_path):
    _test_predict_with_padding(unet2d_diff_output_shape, tmp_path)


def test_predict_image_with_padding_channel_last(stardist, tmp_path):
    _test_predict_with_padding(stardist, tmp_path)


def _test_predict_image_with_tiling(model, tmp_path: Path, exp_mean_deviation):
    from bioimageio.core.prediction import predict_image

    spec = load_description(model)
    assert isinstance(spec, Model)
    inputs = spec.test_inputs
    assert len(inputs) == 1
    exp = np.load(str(spec.test_outputs[0]))

    out_path = tmp_path.with_suffix(".npy")

    def check_result():
        assert out_path.exists()
        res = np.load(out_path)
        assert res.shape == exp.shape
        # check that the mean deviation is smaller than the expected value
        # note that we can't use array_almost_equal here, because the numerical differences
        # between tiled and normal prediction are too large
        mean_deviation = np.abs(res - exp).mean()
        assert mean_deviation <= exp_mean_deviation

    # with tiling config
    tiling = {"halo": {"x": 32, "y": 32}, "tile": {"x": 256, "y": 256}}
    predict_image(model, inputs, [out_path], tiling=tiling)
    check_result()

    # with tiling determined from spec
    predict_image(model, inputs, [out_path], tiling=True)
    check_result()


# prediction with tiling with the parameters above may not be suited for any model
# so we only run it for the pytorch unet2d here
def test_predict_image_with_tiling_1(unet2d_nuclei_broad_model, tmp_path: Path):
    _test_predict_image_with_tiling(unet2d_nuclei_broad_model, tmp_path, 0.012)


def test_predict_image_with_tiling_2(unet2d_diff_output_shape, tmp_path: Path):
    _test_predict_image_with_tiling(unet2d_diff_output_shape, tmp_path, 0.06)


def test_predict_image_with_tiling_3(shape_change_model, tmp_path: Path):
    _test_predict_image_with_tiling(shape_change_model, tmp_path, 0.012)


def test_predict_image_with_tiling_channel_last(stardist, tmp_path: Path):
    _test_predict_image_with_tiling(stardist, tmp_path, 0.13)


def test_predict_image_with_tiling_fixed_output_shape(unet2d_fixed_shape, tmp_path: Path):
    _test_predict_image_with_tiling(unet2d_fixed_shape, tmp_path, 0.025)


def test_predict_images(unet2d_nuclei_broad_model, tmp_path: Path):
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
