import imageio
import numpy as np
import pytest

from bioimageio.spec import load_resource_description
from numpy.testing import assert_array_almost_equal


@pytest.mark.skipif(pytest.skip_torch, reason="requires torch")
def test_predict_image(unet2d_nuclei_broad_model, tmp_path):
    from bioimageio.core.prediction import predict_image

    spec = load_resource_description(unet2d_nuclei_broad_model)
    inputs = spec.test_inputs
    assert len(inputs) == 1

    out_path = tmp_path.with_suffix(".npy")
    outputs = [out_path]
    predict_image(unet2d_nuclei_broad_model, inputs, outputs)
    assert out_path.exists()
    res = np.load(out_path)
    exp = np.load(spec.test_outputs[0])
    assert_array_almost_equal(res, exp, decimal=4)


@pytest.mark.skipif(pytest.skip_torch, reason="requires torch")
def test_predict_image_with_padding(unet2d_nuclei_broad_model, tmp_path):
    from bioimageio.core.prediction import predict_image

    spec = load_resource_description(unet2d_nuclei_broad_model)
    image = np.load(spec.test_inputs[0])[0, 0]
    original_shape = image.shape
    assert image.ndim == 2

    # write the padded image
    image = image[3:-2, 1:-12]
    in_path = tmp_path / "in.tif"
    out_path = tmp_path / "out.tif"
    imageio.imwrite(in_path, image)

    def check_result():
        assert out_path.exists()
        res = imageio.imread(out_path)
        assert res.shape == image.shape

    # test with dynamic padding
    predict_image(unet2d_nuclei_broad_model, in_path, out_path, padding={"x": 8, "y": 8, "mode": "dynamic"})
    check_result()

    # test with fixed padding
    predict_image(unet2d_nuclei_broad_model, in_path, out_path,
                  padding={"x": original_shape[0], "y": original_shape[1], "mode": "fixed"})
    check_result()

    # test with automated padding
    predict_image(unet2d_nuclei_broad_model, in_path, out_path, padding=True)
    check_result()


@pytest.mark.skipif(pytest.skip_torch, reason="requires torch")
def test_predict_image_with_tiling(unet2d_nuclei_broad_model, tmp_path):
    from bioimageio.core.prediction import predict_image

    spec = load_resource_description(unet2d_nuclei_broad_model)
    inputs = spec.test_inputs
    assert len(inputs) == 1
    exp = np.load(spec.test_outputs[0])

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
    predict_image(unet2d_nuclei_broad_model, inputs, [out_path], tiling=tiling)
    check_result()

    # with tiling determined from spec
    predict_image(unet2d_nuclei_broad_model, inputs, [out_path], tiling=True)
    check_result()


@pytest.mark.skipif(pytest.skip_torch, reason="requires torch")
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
