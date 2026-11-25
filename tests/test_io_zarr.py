from inspect import signature
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest

_ = pytest.importorskip("zarr")
_ = pytest.importorskip("clearscale")

import zarr

from bioimageio.core._utils import io_zarr

# pyright: reportPrivateUsage=false


def multiscale(*paths: str):
    return {
        "version": "0.4",
        "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
        "datasets": [
            {
                "path": path,
                "coordinateTransformations": [{"type": "scale", "scale": [1, 1]}],
            }
            for path in paths
        ],
    }


@pytest.fixture
def group(tmp_path: Path):
    group = zarr.open_group(str(tmp_path / "image.zarr"), mode="w")
    if "zarr_format" in signature(cast(Any, zarr.open_group)).parameters:
        create_array = group.create_array
    else:
        create_array = cast(Any, group).create_dataset
    _ = create_array("0", data=np.arange(12).reshape(3, 4))
    _ = create_array("1", data=np.arange(4).reshape(2, 2))
    return group


@pytest.mark.parametrize("nested", [False, True])
def test_multiscale_metadata(group: zarr.Group, nested: bool):
    metadata = [multiscale("0", "1")]
    group.attrs.update(
        {"ome": {"multiscales": metadata}} if nested else {"multiscales": metadata}
    )
    ms = io_zarr._ms_from_group(group)
    assert list(ms) == ["0", "1"]
    assert io_zarr._get_shape_callable(group)("0") == (3, 4)
    assert list(io_zarr._ms_from_group(group, "1")) == ["0", "1"]
    with pytest.raises(ValueError, match="No Multiscale containing"):
        _ = io_zarr._ms_from_group(group, "missing")


@pytest.mark.parametrize(
    "attrs,error,message",
    [
        ({}, ValueError, "No multiscale metadata"),
        ({"multiscales": "invalid"}, TypeError, "valid sequence"),
        ({"multiscales": {}}, TypeError, "valid sequence"),
        ({"multiscales": []}, ValueError, "all invalid"),
        ({"multiscales": [None, dict[str, Any]()]}, ValueError, "all invalid"),
    ],
)
def test_invalid_multiscale_metadata(
    group: zarr.Group, attrs: dict[str, Any], error: type[Exception], message: str
):
    group.attrs.update(attrs)
    with pytest.raises(error, match=message):
        _ = io_zarr._ms_from_group(group)


def test_multiple_multiscales_require_selection(group: zarr.Group):
    group.attrs["multiscales"] = [None, multiscale("0"), multiscale("1")]
    with pytest.raises(ValueError, match="2 Multiscale definitions"):
        _ = io_zarr._ms_from_group(group)
    assert list(io_zarr._ms_from_group(group, "1")) == ["1"]


@pytest.mark.parametrize("single_scale", [False, True])
def test_open_multiscale(tmp_path: Path, group: zarr.Group, single_scale: bool):
    group.attrs["multiscales"] = [multiscale(*(["0"] if single_scale else ["0", "1"]))]
    uri = str(tmp_path / "image.zarr")
    if not single_scale:
        with pytest.raises(ValueError, match="Append '/<key>'"):
            _ = io_zarr.open_zarr_multiscale_array(uri)
        uri += "/0"
    array, scale = io_zarr.open_zarr_multiscale_array(uri + "/")
    np.testing.assert_array_equal(array[:], np.arange(12).reshape(3, 4))
    assert scale is not None


def test_missing_zarr_root_stops_search(monkeypatch: pytest.MonkeyPatch):
    open_zarr = Mock(side_effect=FileNotFoundError)
    monkeypatch.setattr(zarr, "open", open_zarr)
    with pytest.raises(ValueError, match="No multiscale meta"):
        _ = io_zarr._recurse_url_to_find_group("/missing/image.zarr/0")
    assert [call.args[0] for call in open_zarr.call_args_list] == [
        "/missing/image.zarr/0",
        "/missing/image.zarr",
    ]


def test_missing_relative_zarr_root():
    with pytest.raises(ValueError, match="No multiscale meta"):
        _ = io_zarr._recurse_url_to_find_group("missing.zarr")
