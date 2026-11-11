# pyright: reportPrivateUsage=false

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from bioimageio.core import bench


@pytest.mark.parametrize("width", [1, 3, 4, 8])
def test_preview_preserves_spatial_dimensions(width: int):
    array = np.arange(5 * width).reshape(5, width)
    preview = bench._preview(array, dims=("y", "x"))

    assert preview.shape == array.shape
    np.testing.assert_allclose(preview, array / array.max())


def test_preview_channel_first_rgb():
    array = np.arange(3 * 5 * 8).reshape(3, 5, 8)
    preview = bench._preview(array, dims=("channel", "y", "x"))

    assert preview.shape == (5, 8, 3)
    np.testing.assert_allclose(preview, np.transpose(array, (1, 2, 0)) / array.max())


@pytest.mark.parametrize(
    "dims", [("z", "y", "x"), ("t", "y", "x"), ("t", "z", "y", "x")]
)
def test_slice_controls(
    dims: tuple[str, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _ = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    shape = tuple(range(12, 12 + len(dims)))
    array = np.arange(np.prod(shape)).reshape(shape)
    source = tmp_path / "stack.npy"
    np.save(source, array)
    monkeypatch.setattr(sys, "argv", ["bench"])
    previews: list[NDArray[Any]] = []
    original_preview = bench._preview

    def capture_preview(
        array: NDArray[Any], *, dims: tuple[str, ...] | None = None
    ) -> NDArray[Any]:
        previews.append(np.copy(array))
        return original_preview(array, dims=dims)

    monkeypatch.setattr(bench, "_preview", capture_preview)
    app = AppTest.from_string("from bioimageio.core.bench import main\nmain()")
    app.session_state["bench_data_source_0"] = str(source)
    _ = app.run(timeout=20)
    assert not app.exception
    _ = app.text_input(key="bench_data_dims_0").set_value(", ".join(dims)).run()
    assert not app.exception

    slice_dims = [dim for dim in dims if dim in ("t", "z")]
    for dim in slice_dims:
        axis = dims.index(dim)
        slider = app.slider(key=f"bench_data_slice_0_{axis}")
        assert slider.label == ("Z slice" if dim == "z" else "Time frame")
        _ = slider.set_value(5).run()
        assert not app.exception

    np.testing.assert_array_equal(previews[-1], array[(5,) * len(slice_dims)])

    # Moving the ROI past the current slice clamps the view to its first slice.
    _ = app.slider(key="bench_data_roi_0_0").set_range(7, 9).run()
    assert not app.exception
    assert app.slider(key="bench_data_slice_0_0").value == 7
    expected = array[(7, *((5,) * (len(slice_dims) - 1)))]
    np.testing.assert_array_equal(previews[-1], expected)

    # A single-slice ROI needs no slider and still displays the correct plane.
    _ = app.slider(key="bench_data_roi_0_0").set_range(8, 8).run()
    assert not app.exception
    assert "bench_data_slice_0_0" not in [slider.key for slider in app.slider]
    np.testing.assert_array_equal(
        previews[-1],
        array[(8, *((5,) * (len(slice_dims) - 1)))],
    )
