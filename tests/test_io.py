import sys
from importlib.util import find_spec
from inspect import signature
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


@pytest.mark.parametrize(
    "suffix",
    [
        ".png",
        pytest.param(
            ".tiff",
            marks=pytest.mark.skipif(
                sys.version_info < (3, 12),
                reason="tifffile tensor loading and saving requires Python >=3.12.",
            ),
        ),
        ".zarr",
    ],
)
def test_save_sample(tmp_path: Path, suffix: str):
    from bioimageio.core import Sample, Tensor
    from bioimageio.core.common import MemberId
    from bioimageio.core.io import load_tensor, save_sample

    if suffix == ".zarr" and find_spec("clearscale") is None:
        pytest.importorskip("bioio_ome_zarr")

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

    _ = save_sample(tmp_path / f"{sample.id}_{{member_id}}{suffix}", sample)

    for member_id, expected in members.items():
        actual = load_tensor(tmp_path / f"sample_{member_id}{suffix}")
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "name",
    [
        "img.png",
        pytest.param(
            "img.tiff",
            marks=pytest.mark.skipif(
                sys.version_info < (3, 12),
                reason="tifffile tensor loading requires Python >=3.12.",
            ),
        ),
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
        _ = pytest.importorskip("clearscale")

        if name.find(".zarr") == -1:
            pytest.skip("clearscale only supports zarr")
    elif io_lib == "bioio" and (
        name.endswith(".zarr") or name.endswith(".zarr/0")
    ):
        pytest.importorskip("bioio_ome_zarr")
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


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("io_lib", ["clearscale", "bioio"])
@pytest.mark.parametrize("internal_path", ["", "0"])
def test_tensor_ome_zarr_metadata_roundtrip(
    zarr_format: int,
    io_lib: Literal["clearscale", "bioio"],
    internal_path: str,
    tmp_path: Path,
):
    zarr = pytest.importorskip("zarr")
    _ = pytest.importorskip(
        "clearscale" if io_lib == "clearscale" else "bioio_ome_zarr"
    )

    from bioimageio.core import AxisId, Tensor
    from bioimageio.core.io import load_tensor, save_tensor

    expected = Tensor.from_numpy(
        np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5),
        dims=("z", "y", "x"),
    )
    pixel_size = {AxisId("z"): 2.5, AxisId("y"): 0.5, AxisId("x"): 0.25}
    units = {axis: "micrometer" for axis in pixel_size}
    expected.set_physical_scale(pixel_size)
    expected.set_physical_scale_unit(units)
    store = tmp_path / "image.ome.zarr"
    path = store / internal_path

    with zarr.config.set({"default_zarr_format": zarr_format}):
        save_tensor(path, expected, io_lib=io_lib)

    group = zarr.open_group(store, mode="r")
    assert group.metadata.zarr_format == zarr_format
    metadata = group.attrs if zarr_format == 2 else group.attrs["ome"]
    multiscale = metadata["multiscales"][0]
    assert [axis["name"] for axis in multiscale["axes"]] == ["z", "y", "x"]
    assert [axis["unit"] for axis in multiscale["axes"]] == ["micrometer"] * 3
    dataset = multiscale["datasets"][0]
    assert dataset["coordinateTransformations"] == [
        {"type": "scale", "scale": list(pixel_size.values())}
    ]
    array = group[dataset["path"]]
    assert array.metadata.zarr_format == zarr_format

    actual = load_tensor(path, io_lib=io_lib)
    assert actual.dims == expected.dims
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.to_numpy(), expected.to_numpy())
    assert actual.get_physical_scale() == pytest.approx(pixel_size)
    assert actual.get_physical_scale_unit() == units


@pytest.mark.parametrize("io_lib", ["clearscale", "bioio"])
@pytest.mark.parametrize("group_path", ["", "group"])
@pytest.mark.parametrize("scale_key", ["", "s0", "0", "scalekeyvalue"])
def test_load_zarr_from_group(
    io_lib: Literal["clearscale", "bioio"],
    group_path: str,
    scale_key: str,
    tmp_path: Path,
):
    if io_lib == "clearscale":
        _ = pytest.importorskip("clearscale")
    else:
        pytest.importorskip("bioio_ome_zarr")

    import zarr

    from bioimageio.core.io import load_tensor

    default_scale_key = "0"
    storage_source = tmp_path / "image.ome.zarr"
    group_source = storage_source / group_path if storage_source else storage_source
    uses_zarr_format = "zarr_format" in signature(zarr.open_group).parameters
    if uses_zarr_format:
        group = zarr.open_group(group_source, mode="w", zarr_format=2)
    else:
        group = zarr.open_group(group_source, mode="w")
    expected = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
    if uses_zarr_format:
        _ = group.create_array(scale_key or default_scale_key, data=expected)
    else:
        _ = group.create_dataset(scale_key or default_scale_key, data=expected)
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
