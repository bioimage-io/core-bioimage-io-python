import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pytest


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="various bioio writer modules are only available for py>=3.10.",
)
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


@pytest.mark.parametrize("suffix", [".png", ".tiff", ".zarr"])
def test_save_sample(tmp_path: Path, suffix: str):
    from bioimageio.core import Sample, Tensor
    from bioimageio.core.common import MemberId
    from bioimageio.core.io import load_tensor, save_sample

    members = {
        MemberId("raw"): np.arange(30, dtype=np.uint8).reshape(5, 6),
        MemberId("mask"): np.ones((5, 6), dtype=np.uint8),
    }
    sample = Sample(
        members={
            m: Tensor.from_numpy(data, dims=("y", "x")) for m, data in members.items()
        },
        stat={},
        id="sample",
    )

    save_sample(tmp_path / f"{sample.id}_{{member_id}}{suffix}", sample)

    for member_id, expected in members.items():
        actual = load_tensor(tmp_path / f"sample_{member_id}{suffix}")
        np.testing.assert_array_equal(actual, expected)


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

    def save():
        save_tensor(path, expected, io_lib=io_lib)

    if io_lib == "bioio" and name.endswith(".zarr/s2"):
        with pytest.raises(
            NotImplementedError,
            match="Saving to an internal path that is not '0' is not implemented for zarr files with io_lib='bioio'.",
        ):
            save()
    else:
        save()

        actual = load_tensor(path, io_lib=io_lib)
        assert (actual == expected).to_numpy().all()


@pytest.mark.parametrize("io_lib", ["clearscale", "bioio"])
@pytest.mark.parametrize("group_path", ["", "group"])
@pytest.mark.parametrize("scale_key", ["", "s0", "0", "scalekeyvalue"])
def test_load_zarr_from_group(
    io_lib: Literal["clearscale", "bioio"],
    group_path: str,
    scale_key: str,
    tmp_path: Path,
):
    import zarr

    from bioimageio.core.io import load_tensor

    default_scale_key = "0"
    storage_source = tmp_path / "image.ome.zarr"
    group_source = storage_source / group_path if storage_source else storage_source
    group = zarr.open_group(group_source, mode="w", zarr_format=2)
    expected = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
    _ = group.create_array(scale_key or default_scale_key, data=expected)
    # group = zarr.open_group(storage_source, mode="a", zarr_format=2)
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [
                {
                    "path": scale_key or default_scale_key,
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1, 1, 1, 1]}
                    ],
                }
            ],
        }
    ]
    source = group_source / scale_key

    tensor = load_tensor(source, io_lib=io_lib)
    assert tensor.dims == ("channel", "z", "y", "x")
    np.testing.assert_array_equal(tensor.to_numpy(), expected)
