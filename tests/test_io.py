from pathlib import Path
from typing import Literal

import numpy as np
import pytest


@pytest.mark.parametrize(
    "name",
    [
        "img.npy",
    ],
)
@pytest.mark.parametrize(
    "io_lib",
    [
        "numpy",
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (4, 5),
        (3, 4, 5),
        (1, 4, 5),
        (5, 4, 3),
        (5, 3, 4),
    ],
)
def test_tensor_io_numpy(
    name: str,
    io_lib: Literal["numpy"],
    shape: tuple[int, ...],
    tmp_path: Path,
):
    from bioimageio.core import Tensor
    from bioimageio.core.io import load_tensor, save_tensor

    path = tmp_path / name
    expected = Tensor.from_numpy(
        np.arange(np.prod(shape), dtype=np.uint8).reshape(shape), dims=None
    )
    save_tensor(path, expected, io_lib=io_lib)
    actual = load_tensor(path, io_lib=io_lib)
    assert (actual == expected).to_numpy().all()


@pytest.mark.parametrize(
    "name",
    [
        "img.png",
        "img.tiff",
        "img.zarr",
        "img.zarr/0",
        "img.zarr/s2",
    ],
)
@pytest.mark.parametrize(
    "io_lib",
    [
        "bioio",
        "imageio",
        "clearscale",
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (6, 5),
        (1, 6, 5),
        (3, 6, 5),
        (4, 5, 6),
    ],
)
def test_tensor_io(
    name: str,
    io_lib: Literal["bioio", "imageio", "clearscale"],
    shape: tuple[int, ...],
    tmp_path: Path,
):
    from bioimageio.core import Tensor
    from bioimageio.core.io import load_tensor, save_tensor

    if io_lib == "clearscale":
        try:
            import clearscale as _
        except ImportError:
            pytest.skip("clearscale not installed")

        if name.find(".zarr") == -1:
            pytest.skip("clearscale only supports zarr")
    elif io_lib == "imageio":
        if name.find(".zarr") != -1:
            pytest.skip("imageio does not support zarr")

    path = tmp_path / name
    c = "S" if name.endswith(".png") else "C"
    expected = Tensor.from_numpy(
        np.arange(np.prod(shape), dtype=np.uint8).reshape(shape),
        dims=f"{c}YX"[-len(shape) :],
    )
    # if io_lib == "bioio" and path.suffix == ".zarr":
    #     # rename axes to enable roundtrip with bioio
    #     expected._data = expected.data.rename(
    #         {a: str(a)[0].upper() for a in expected.dims}
    #     )

    save_tensor(path, expected, io_lib=io_lib)

    actual = load_tensor(path, io_lib=io_lib)
    assert (actual == expected).to_numpy().all()


def test_load_tensor_zarr():
    source = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr/2"
    from bioimageio.core.io import load_tensor

    img = load_tensor(source)
    assert img.shape


def test_load_tensor_zarr_group():
    source = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr"

    from bioimageio.core.io import load_tensor

    tensor = load_tensor(source)
    assert tensor.dims == ("channel", "z", "y", "x")
