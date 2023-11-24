import numpy as np


def test_transform_input_image():
    from bioimageio.core.image_helper import transpose_image

    ax_list = ["yx", "xy", "cyx", "yxc", "bczyx", "xyz", "xyzc", "bzyxc"]
    im = np.random.rand(256, 256)
    for axes in ax_list:
        inp = transpose_image(im, axes)
        assert inp.ndim == len(axes)

    ax_list = ["zyx", "cyx", "yxc", "bczyx", "xyz", "xyzc", "bzyxc"]
    vol = np.random.rand(64, 64, 64)
    for axes in ax_list:
        inp = transpose_image(vol, axes)
        assert inp.ndim == len(axes)


def test_transform_output_tensor():
    from bioimageio.core.image_helper import transform_output_tensor

    tensor = np.random.rand(1, 3, 64, 64, 64)
    tensor_axes = "bczyx"

    out_ax_list = ["bczyx", "cyx", "xyc", "byxc", "zyx", "xyz"]
    for out_axes in out_ax_list:
        out = transform_output_tensor(tensor, tensor_axes, out_axes)
        assert out.ndim == len(out_axes)
