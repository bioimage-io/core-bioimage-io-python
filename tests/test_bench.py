# pyright: reportPrivateUsage=false

import io
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from bioimageio.core import Sample, bench


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


def test_preview_shared_slice_indices():
    array = np.arange(3 * 5 * 8).reshape(3, 5, 8) ** 2
    expected = bench._preview(array[2], dims=("y", "x"))
    np.testing.assert_allclose(
        bench._preview(array, dims=("z", "y", "x"), indices={"z": 9, "t": 2}),
        expected,
    )
    np.testing.assert_allclose(
        bench._preview(
            array.transpose(1, 2, 0), dims=("y", "x", "z"), indices={"z": 2}
        ),
        expected,
    )


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
    monkeypatch.setattr(sys, "argv", ["bench", "--data", str(source)])
    previews: list[NDArray[Any]] = []
    original_preview = bench._preview

    def capture_preview(
        array: NDArray[Any], *, dims: tuple[str, ...] | None = None
    ) -> NDArray[Any]:
        previews.append(np.copy(array))
        return original_preview(array, dims=dims)

    monkeypatch.setattr(bench, "_preview", capture_preview)
    app = AppTest.from_string("from bioimageio.core.bench import main\nmain()")
    sample_key = bench._sample_key(0, str(source.resolve()))
    _ = app.run(timeout=20)
    assert not app.exception
    _ = (
        app.text_input(key=f"bench_data_dims_{sample_key}")
        .set_value(", ".join(dims))
        .run()
    )
    assert not app.exception

    slice_dims = [dim for dim in dims if dim in ("t", "z")]
    for dim in slice_dims:
        axis = dims.index(dim)
        slider = app.slider(key=f"bench_data_slice_{sample_key}_{axis}")
        assert slider.label == ("Z slice" if dim == "z" else "Time frame")
        _ = slider.set_value(5).run()
        assert not app.exception

    np.testing.assert_array_equal(previews[-1], array[(5,) * len(slice_dims)])

    # Full-screen selection preserves the ROI and preview widget state.
    _ = app.button(key=f"bench_selection_expand_{sample_key}").click().run()
    assert not app.exception
    assert app.toggle(key=f"bench_selection_fullscreen_{sample_key}").value is True
    assert app.slider(key=f"bench_data_slice_{sample_key}_0").value == 5
    np.testing.assert_array_equal(previews[-1], array[(5,) * len(slice_dims)])

    _ = app.button(key=f"bench_selection_expand_{sample_key}").click().run()
    assert not app.exception
    assert app.toggle(key=f"bench_selection_fullscreen_{sample_key}").value is False
    assert app.slider(key=f"bench_data_slice_{sample_key}_0").value == 5
    _ = app.toggle(key=f"bench_selection_fullscreen_{sample_key}").set_value(True).run()
    assert not app.exception

    # Moving the ROI past the current slice clamps the view to its first slice.
    _ = app.slider(key=f"bench_data_roi_{sample_key}_0").set_range(7, 9).run()
    assert not app.exception
    assert app.slider(key=f"bench_data_slice_{sample_key}_0").value == 7
    expected = array[(7, *((5,) * (len(slice_dims) - 1)))]
    np.testing.assert_array_equal(previews[-1], expected)

    # A single-slice ROI needs no slider and still displays the correct plane.
    _ = app.slider(key=f"bench_data_roi_{sample_key}_0").set_range(8, 8).run()
    assert not app.exception
    assert f"bench_data_slice_{sample_key}_0" not in [
        slider.key for slider in app.slider
    ]
    np.testing.assert_array_equal(
        previews[-1],
        array[(8, *((5,) * (len(slice_dims) - 1)))],
    )


def test_resolve_data_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in ("b.npy", "a.npy", "other.TIF", "notes.txt"):
        _ = (tmp_path / name).touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    _ = (nested / "c.npy").touch()
    expected = [str(tmp_path / name) for name in ("a.npy", "b.npy", "other.TIF")]
    assert bench._resolve_data_sources(str(tmp_path)) == expected
    assert bench._resolve_data_sources(f"{tmp_path}/*.npy") == expected[:2]
    assert bench._resolve_data_sources(f"{tmp_path}/**/*.npy") == [
        *expected[:2],
        str(nested / "c.npy"),
    ]
    # Overlapping entries include each local file once, keeping input order.
    monkeypatch.chdir(tmp_path)
    assert bench._resolve_data_sources(f"b.npy\n{tmp_path}/*.npy\n\n") == [
        expected[1],
        expected[0],
    ]
    url = "https://example.org/image.npy?token=a[1]"
    assert bench._resolve_data_sources(url) == [url]


@pytest.mark.parametrize("source", ["*.png", "empty"])
def test_resolve_data_sources_no_matches(tmp_path: Path, source: str):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No supported data files"):
        _ = bench._resolve_data_sources(str(tmp_path / source))


def test_dataset_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _ = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(sys, "argv", ["bench"])
    for name in ("a.npy", "b.npy"):
        np.save(tmp_path / name, np.arange(20).reshape(4, 5))
    app = AppTest.from_string("from bioimageio.core.bench import main\nmain()")
    _ = app.run(timeout=20)
    assert not app.exception
    assert app.session_state["bench_data_ids"] == []
    assert app.get("popover")[0].proto.popover.label == "Add dataset"
    _ = app.text_input(key="bench_new_dataset_name").set_value("Cells")
    _ = app.text_area(key="bench_new_dataset_sources").set_value(str(tmp_path))
    _ = app.button(key="bench_add_dataset").click().run()
    assert not app.exception
    assert app.session_state["bench_data_ids"] == [0]
    assert app.text_input(key="bench_dataset_name_0").value == "Cells"
    assert any(c.value == "2 samples" for c in app.caption)

    sample_a = bench._sample_key(0, str(tmp_path / "a.npy"))
    sample_b = bench._sample_key(0, str(tmp_path / "b.npy"))
    _ = app.slider(key=f"bench_data_roi_{sample_a}_0").set_range(1, 2).run()
    assert not app.exception
    assert app.slider(key=f"bench_data_roi_{sample_b}_0").value == (0, 3)
    _ = app.text_input(key="bench_dataset_name_0").set_value("Renamed cells").run()
    assert not app.exception
    assert app.slider(key=f"bench_data_roi_{sample_a}_0").value == (1, 2)
    assert app.expander[0].label == "Renamed cells"

    # Removing another sample from the sources must not reset this sample's ROI.
    _ = (
        app.text_area(key="bench_data_source_0")
        .set_value(str(tmp_path / "a.npy"))
        .run()
    )
    assert not app.exception
    assert app.slider(key=f"bench_data_roi_{sample_a}_0").value == (1, 2)
    assert any(c.value == "1 sample" for c in app.caption)

    _ = (
        app.text_area(key="bench_data_source_0")
        .set_value(str(tmp_path / "missing*.tif"))
        .run()
    )
    assert not app.exception
    assert any("No supported data files" in error.value for error in app.error)
    _ = app.text_area(key="bench_data_source_0").set_value(str(tmp_path)).run()
    assert not app.exception
    assert not app.error
    assert app.text_input(key=f"bench_data_dims_{sample_a}").value == "y, x"

    _ = app.button(key="bench_delete_data_0").click().run()
    assert not app.exception
    assert app.session_state["bench_data_ids"] == []
    # Dataset IDs are never reused after deletion.
    _ = app.button(key="bench_add_dataset").click().run()
    assert not app.exception
    assert app.session_state["bench_data_ids"] == [1]


def test_dataset_uploads(monkeypatch: pytest.MonkeyPatch):
    _ = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(sys, "argv", ["bench"])
    app = AppTest.from_string("from bioimageio.core.bench import main\nmain()")
    _ = app.run(timeout=20)
    # Use .npy data, including duplicate file names, to exercise upload identity.
    buffer = io.BytesIO()
    np.save(buffer, np.arange(20).reshape(4, 5))
    content = buffer.getvalue()
    uploader = app.file_uploader(key="bench_new_dataset_uploads")
    uploader.upload("sample.npy", content)
    uploader.upload("sample.npy", content)
    _ = app.button(key="bench_add_dataset").click().run()
    assert not app.exception
    assert len(app.session_state["bench_dataset_uploads_0"]) == 2
    assert any(c.value == "2 samples" for c in app.caption)

    app.file_uploader(key="bench_data_0_0").upload("extra.npy", content).run()
    assert not app.exception
    assert len(app.session_state["bench_dataset_uploads_0"]) == 3
    assert app.file_uploader(key="bench_data_0_1").value == []
    uploads = app.session_state["bench_dataset_uploads_0"]
    sample_key = bench._sample_key(0, uploads[0].id)
    _ = app.button(key=f"bench_remove_sample_{sample_key}").click().run()
    assert not app.exception
    assert len(app.session_state["bench_dataset_uploads_0"]) == 2
    _ = app.text_input(key="bench_dataset_name_0").set_value("Uploads").run()
    assert not app.exception
    assert len(app.session_state["bench_dataset_uploads_0"]) == 2


@pytest.mark.parametrize("depth", [1, 3])
def test_dataset_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, depth: int
):
    _ = pytest.importorskip("streamlit")
    from unittest.mock import MagicMock, Mock

    from streamlit.testing.v1 import AppTest

    from bioimageio.core.axis import AxisId
    from bioimageio.core.common import MemberId
    from bioimageio.spec.model import v0_5

    model = v0_5.ModelDescr.model_construct(
        name="Identity",
        format_version="0.5.10",
        inputs=[
            v0_5.InputTensorDescr(
                id=MemberId("input"),
                axes=[
                    v0_5.SpaceInputAxis(id=AxisId("z"), size=depth),
                    v0_5.SpaceInputAxis(id=AxisId("y"), size=4),
                    v0_5.SpaceInputAxis(id=AxisId("x"), size=5),
                ],
            )
        ],
        outputs=[
            v0_5.OutputTensorDescr.model_construct(
                id=MemberId("input"),
                axes=[
                    v0_5.SpaceOutputAxis(id=AxisId("z"), size=depth),
                    v0_5.SpaceOutputAxis(id=AxisId("y"), size=4),
                    v0_5.SpaceOutputAxis(id=AxisId("x"), size=5),
                ],
            )
        ],
        weights=v0_5.WeightsDescr.model_construct(),
    )
    pipeline = MagicMock()
    pipeline.__enter__.return_value = pipeline

    def predict_sample(sample: Sample) -> Sample:
        return sample

    pipeline.predict_without_blocking.side_effect = predict_sample
    monkeypatch.setattr(bench, "load_model", Mock(return_value=model))
    monkeypatch.setattr(
        bench, "create_prediction_pipeline", Mock(return_value=pipeline)
    )
    monkeypatch.setattr(
        sys, "argv", ["bench", "--data", str(tmp_path), "--model", "identity-test"]
    )
    for index, name in enumerate(("a.npy", "b.npy")):
        np.save(tmp_path / name, np.full((depth, 4, 5), index))
    app = AppTest.from_string("from bioimageio.core.bench import main\nmain()")
    _ = app.run(timeout=20)
    assert not app.exception
    for name in ("a.npy", "b.npy"):
        sample_key = bench._sample_key(0, str((tmp_path / name).resolve()))
        _ = (
            app.text_input(key=f"bench_data_dims_{sample_key}")
            .set_value("z, y, x")
            .run()
        )
    assert not app.exception
    assert not app.error
    _ = app.checkbox[1].check().run()
    assert not app.exception
    assert pipeline.predict_without_blocking.call_count == 2
    names = [heading.value for heading in app.subheader]
    assert "Dataset 1 · a.npy" in names
    assert "Dataset 1 · b.npy" in names
    for index, call in enumerate(pipeline.predict_without_blocking.call_args_list):
        np.testing.assert_array_equal(
            call.args[0].members[MemberId("input")].to_numpy(),
            np.full((depth, 4, 5), index),
        )

    assert sum(m.value == "**Input**" for m in app.markdown) == 2
    comparison_sliders = [
        s for s in app.slider if s.key.startswith("bench_comparison_slice_")
    ]
    assert len(comparison_sliders) == (2 if depth > 1 else 0)
    if depth > 1:
        _ = comparison_sliders[0].set_value(depth - 1).run()
        assert not app.exception
        assert pipeline.predict_without_blocking.call_count == 2
        assert app.slider(key=comparison_sliders[1].key).value == 0


def test_sync_roi_from_any_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _ = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    first, second = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(first, np.arange(48).reshape(6, 8))
    np.save(second, np.arange(48).reshape(8, 6))
    monkeypatch.setattr(
        sys, "argv", ["bench", "--data", f"{first}\n{second}", "--data", str(first)]
    )
    app = AppTest.from_string("from bioimageio.core.bench import main\nmain()")
    _ = app.run(timeout=20)
    assert not app.exception
    first_key = bench._sample_key(0, str(first.resolve()))
    second_key = bench._sample_key(0, str(second.resolve()))
    other_dataset_key = bench._sample_key(1, str(first.resolve()))
    assert f"bench_sync_roi_{other_dataset_key}" not in [
        button.key for button in app.button
    ]

    # Also sync to uploaded data, including an axis of size one.
    buffer = io.BytesIO()
    np.save(buffer, np.zeros((1, 3)))
    app.file_uploader(key="bench_data_0_0").upload("small.npy", buffer.getvalue()).run()
    assert not app.exception
    upload = app.session_state["bench_dataset_uploads_0"][0]
    upload_key = bench._sample_key(0, upload.id)
    _ = app.text_input(key=f"bench_data_dims_{second_key}").set_value("x, y").run()
    assert not app.exception
    _ = app.slider(key=f"bench_data_roi_{second_key}_0").set_range(2, 6)
    _ = app.slider(key=f"bench_data_roi_{second_key}_1").set_range(3, 5).run()
    assert not app.exception
    _ = app.button(key=f"bench_sync_roi_{second_key}").click().run()
    assert not app.exception
    assert app.slider(key=f"bench_data_roi_{first_key}_0").value == (3, 5)
    assert app.slider(key=f"bench_data_roi_{first_key}_1").value == (2, 6)
    assert app.slider(key=f"bench_data_roi_{upload_key}_1").value == (2, 2)
    assert app.slider(key=f"bench_data_roi_{other_dataset_key}_0").value == (0, 5)
    assert app.slider(key=f"bench_data_roi_{other_dataset_key}_1").value == (0, 7)

    # A different image can become the source; existing ranges are overwritten.
    _ = app.slider(key=f"bench_data_roi_{first_key}_0").set_range(1, 4).run()
    assert not app.exception
    _ = app.button(key=f"bench_sync_roi_{first_key}").click().run()
    assert not app.exception
    assert app.slider(key=f"bench_data_roi_{second_key}_1").value == (1, 4)
    _ = app.run()
    assert not app.exception
    assert app.slider(key=f"bench_data_roi_{second_key}_1").value == (1, 4)


def test_sync_roi_axes_and_slice_clamping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _ = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    paths = [tmp_path / name for name in ("stack.npy", "target.npy", "plane.npy")]
    for path, shape in zip(paths, ((6, 4, 5), (6, 4, 5), (4, 5))):
        np.save(path, np.zeros(shape))
    monkeypatch.setattr(
        sys, "argv", ["bench", "--data", "\n".join(str(path) for path in paths)]
    )
    app = AppTest.from_string("from bioimageio.core.bench import main\nmain()")
    _ = app.run(timeout=20)
    assert not app.exception
    source_key, target_key, plane_key = [
        bench._sample_key(0, str(path.resolve())) for path in paths
    ]
    for sample_key in (source_key, target_key):
        _ = (
            app.text_input(key=f"bench_data_dims_{sample_key}")
            .set_value("z, y, x")
            .run()
        )
        assert not app.exception
    _ = app.slider(key=f"bench_data_slice_{target_key}_0").set_value(5).run()
    assert not app.exception
    _ = app.slider(key=f"bench_data_roi_{source_key}_0").set_range(1, 2)
    _ = app.slider(key=f"bench_data_roi_{source_key}_1").set_range(1, 3).run()
    assert not app.exception
    _ = app.button(key=f"bench_sync_roi_{source_key}").click().run()
    assert not app.exception
    assert app.slider(key=f"bench_data_slice_{target_key}_0").value == 2
    assert app.slider(key=f"bench_data_roi_{target_key}_0").value == (1, 2)
    assert app.slider(key=f"bench_data_roi_{plane_key}_0").value == (1, 3)
    assert app.slider(key=f"bench_data_roi_{plane_key}_1").value == (0, 4)

    # Syncing a plane back to a stack leaves the stack's unmatched z ROI alone.
    _ = app.slider(key=f"bench_data_roi_{plane_key}_0").set_range(2, 2).run()
    assert not app.exception
    _ = app.button(key=f"bench_sync_roi_{plane_key}").click().run()
    assert not app.exception
    assert app.slider(key=f"bench_data_roi_{target_key}_0").value == (1, 2)
    assert app.slider(key=f"bench_data_roi_{target_key}_1").value == (2, 2)


def test_model_selection(monkeypatch: pytest.MonkeyPatch):
    _ = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from bioimageio.spec.model import v0_5

    model = v0_5.ModelDescr.model_construct(
        name="Selected model",
        format_version="0.5.10",
        weights=v0_5.WeightsDescr.model_construct(),
    )
    monkeypatch.setattr(bench, "load_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(sys, "argv", ["bench"])
    app = AppTest.from_string("from bioimageio.core.bench import main\nmain()")
    _ = app.run(timeout=20)
    assert not app.exception
    assert app.session_state["bench_model_ids"] == []
    assert app.get("form")[0].proto.form.enter_to_submit
    _ = app.button(key="bench_add_model").click().run()
    assert not app.exception
    assert app.session_state["bench_model_ids"] == []

    _ = app.text_input(key="bench_new_models").set_value(" first, , second, ")
    _ = app.button(key="bench_add_model").click().run()
    assert not app.exception
    assert app.session_state["bench_model_ids"] == [0, 1]
    assert app.text_input(key="bench_model_0").value == "first"
    assert app.text_input(key="bench_model_1").value == "second"
    assert app.text_input(key="bench_new_models").value == ""
    assert sum(m.value == "**Selected model**" for m in app.markdown) == 2
    _ = app.checkbox(key="bench_model_blockwise_0").check().run()
    _ = app.text_input(key="bench_new_models").set_value("third")
    _ = app.button(key="bench_add_model").click().run()
    assert not app.exception
    assert app.session_state["bench_model_ids"] == [0, 1, 2]
    assert app.checkbox(key="bench_model_blockwise_0").value
    _ = app.button(key="bench_delete_model_1").click().run()
    assert not app.exception
    assert app.session_state["bench_model_ids"] == [0, 2]
    assert app.checkbox(key="bench_model_blockwise_0").value


@pytest.mark.parametrize("cover_kind", ["png", "svg", "missing", "none"])
def test_model_cards(cover_kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _ = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from bioimageio.spec.common import FileDescr
    from bioimageio.spec.model import v0_5

    covers: list[FileDescr] = []
    if cover_kind != "none":
        cover = tmp_path / f"cover.{cover_kind}"
        if cover_kind == "png":
            from PIL import Image

            Image.new("RGB", (8, 8), "blue").save(cover)
        elif cover_kind == "svg":
            _ = cover.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8">'
                '<rect width="8" height="8" fill="blue"/></svg>'
            )
        covers = [FileDescr.model_construct(source=cover)]

    model = v0_5.ModelDescr.model_construct(
        name="Model with a card",
        description="Segments cells in microscopy images.",
        covers=covers,
        format_version="0.5.10",
        weights=v0_5.WeightsDescr.model_construct(),
    )
    monkeypatch.setattr(bench, "load_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(sys, "argv", ["bench"])
    app = AppTest.from_string("from bioimageio.core.bench import main\nmain()")
    _ = app.run(timeout=20)
    _ = app.text_input(key="bench_new_models").set_value(str(tmp_path / "model"))
    _ = app.button(key="bench_add_model").click().run()
    assert not app.exception
    assert not app.error
    assert any(m.value == "**Model with a card**" for m in app.markdown)
    assert any(m.value == model.description for m in app.markdown)
    assert len(app.get("image")) == (1 if cover_kind in ("png", "svg") else 0)
    assert any(c.value == "Model cover could not be loaded." for c in app.caption) == (
        cover_kind == "missing"
    )
    assert not app.checkbox(key="bench_model_blockwise_0").value
    _ = app.button(key="bench_delete_model_0").click().run()
    assert not app.exception
    assert not app.get("image")
    assert not any(m.value == model.description for m in app.markdown)
