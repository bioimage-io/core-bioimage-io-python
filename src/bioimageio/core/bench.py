from __future__ import annotations

import io
import json
import sys
from argparse import ArgumentParser
from collections.abc import Hashable, Mapping, Sequence
from glob import glob, has_magic
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, NamedTuple, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from bioimageio.core import (
    Sample,
    Tensor,
    create_prediction_pipeline,
    create_remote_prediction_pipeline,
    load_model,
)
from bioimageio.core.axis import AxisId
from bioimageio.core.digest_spec import (
    create_sample_for_model,
    get_axes_infos,
    get_member_id,
)
from bioimageio.core.io import load_tensor
from bioimageio.spec.model import v0_5

Prediction: TypeAlias = dict[str, NDArray[Any]]
PredictionResult: TypeAlias = Prediction | Exception
DataSignature: TypeAlias = tuple[
    str, str, tuple[str, ...], tuple[tuple[str, int, int], ...]
]
RunSignature: TypeAlias = tuple[
    str, tuple[tuple[str, bool], ...], tuple[DataSignature, ...]
]
StoredResults: TypeAlias = tuple[
    RunSignature, list[str], dict[tuple[int, int], PredictionResult]
]
DATA_EXTENSIONS = ("npy", "tif", "tiff", "png", "jpg", "jpeg", "h5", "hdf5")


class _UploadedSample(NamedTuple):
    id: str
    name: str
    content: bytes


def _resolve_data_sources(text: str) -> list[str]:
    """Expand local directories and globs, preserving URLs and input order."""
    sources: dict[str, None] = {}
    for line in text.splitlines():
        source = line.strip()
        if not source:
            continue
        if "://" in source:
            sources[source] = None
            continue

        path = Path(source).expanduser()
        if path.is_file():
            matches = [str(path.resolve())]
        elif path.is_dir() or has_magic(str(path)):
            candidates = (
                path.iterdir()
                if path.is_dir()
                else (Path(match) for match in glob(str(path), recursive=True))
            )
            matches = sorted(
                {
                    str(candidate.resolve())
                    for candidate in candidates
                    if candidate.is_file()
                    and candidate.suffix.lower().lstrip(".") in DATA_EXTENSIONS
                }
            )
            if not matches:
                raise ValueError(f"No supported data files found for '{source}'")
        else:
            # Keep tensor selectors (for example, HDF5 internal paths) intact.
            matches = [str(path)]

        sources.update(dict.fromkeys(matches))

    return list(sources)


def _sample_key(dataset_id: int, source_id: str) -> str:
    return f"{dataset_id}_{sha256(source_id.encode()).hexdigest()[:16]}"


class _BenchArgs(NamedTuple):
    server: str
    data: tuple[str, ...]
    model: tuple[str, ...]
    state_path: str | None


def _parse_args(args: Sequence[str]) -> _BenchArgs:
    parser = ArgumentParser(add_help=False)
    _ = parser.add_argument("--server", default="")
    _ = parser.add_argument("--data", action="append", default=[])
    _ = parser.add_argument("--model", action="append", default=[])
    _ = parser.add_argument("--state-path")
    parsed, _ = parser.parse_known_args(args)
    return _BenchArgs(
        server=cast(str, parsed.server),
        data=tuple(cast(list[str], parsed.data)),
        model=tuple(cast(list[str], parsed.model)),
        state_path=cast(str | None, parsed.state_path),
    )


def _with_trailing_empty_row(
    row_ids: Sequence[int], used_ids: set[int], next_id: int
) -> tuple[list[int], int]:
    populated = [row_id for row_id in row_ids if row_id in used_ids]
    empty = next((row_id for row_id in row_ids if row_id not in used_ids), None)
    if empty is None:
        empty = next_id
        next_id += 1

    return [*populated, empty], next_id


def _preview(
    array: NDArray[Any],
    *,
    dims: Sequence[str] | None = None,
    indices: Mapping[str, int] | None = None,
) -> NDArray[Any]:
    if dims is not None and indices:
        selection = tuple(
            min(max(indices[dim], 0), size - 1) if dim in indices else slice(None)
            for dim, size in zip(dims, array.shape)
        )
        array = array[selection]
        dims = tuple(dim for dim in dims if dim not in indices)
    if dims is not None and "y" in dims and "x" in dims:
        display_dims = ["y", "x"]
        if "channel" in dims:
            display_dims.append("channel")
        order = [dim for dim in dims if dim not in display_dims] + display_dims
        image = np.transpose(array, tuple(dims.index(dim) for dim in order))
        while image.ndim > len(display_dims):
            image = image[0]
        if "channel" in display_dims and image.shape[-1] not in (3, 4):
            image = image.max(axis=-1)
    else:
        image = np.squeeze(array)
        while image.ndim > 3:
            image = image[0]
        if image.ndim == 3 and image.shape[-1] not in (3, 4):
            image = image.max(axis=0)
    if image.ndim < 2:
        image = np.atleast_2d(image)

    image = image.astype(np.float32, copy=False)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image)

    low, high = float(np.nanmin(image)), float(np.nanmax(image))
    if high <= low:
        return np.zeros_like(image)

    normalized = cast(NDArray[Any], (image - low) / (high - low))
    return np.nan_to_num(normalized)


def _result_archive(outputs: Mapping[str, NDArray[Any]]) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **cast(Any, dict(outputs)))
    return buffer.getvalue()


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit(
            "The benchmark app requires Streamlit. Install it with `pip install "
            + "'bioimageio.core[bench]'`."
        ) from exc

    initial = _parse_args(sys.argv[1:])

    @st.cache_data(show_spinner=False)
    def load_uploaded_tensor(
        name: str, content: bytes
    ) -> tuple[NDArray[Any], tuple[str, ...]]:
        with TemporaryDirectory(prefix="bioimageio-bench-") as temp_dir:
            path = Path(temp_dir) / Path(name).name
            _ = path.write_bytes(content)
            tensor = load_tensor(path)

        return tensor.to_numpy(), tuple(str(dim) for dim in tensor.dims)

    @st.cache_data(show_spinner=False)
    def load_source_tensor(source: str) -> tuple[NDArray[Any], tuple[str, ...]]:
        tensor = load_tensor(source)
        return tensor.to_numpy(), tuple(str(dim) for dim in tensor.dims)

    @st.cache_resource(show_spinner=False)
    def load_model_cached(source: str) -> v0_5.ModelDescr:
        model: v0_5.ModelDescr = load_model(
            source, format_version="latest", perform_io_checks=False
        )
        if model.weights.torchscript is not None:
            # shortcut: portable weights avoid serializing remote Python files.
            model.weights.torchscript.parent = None
            for weights_format in model.weights.available_formats:
                if weights_format != "torchscript":
                    setattr(model.weights, weights_format, None)

        return model

    @st.cache_data(show_spinner=False)
    def load_model_cover_cached(source: str) -> bytes | str | None:
        model = load_model_cached(source)
        if not model.covers:
            return None

        cover = model.covers[0]
        content = cover.get_reader(progressbar=False).read()
        return (
            content.decode("utf-8-sig") if cover.suffix.lower() == ".svg" else content
        )

    def create_compatible_sample(
        model: v0_5.ModelDescr,
        tensor: Tensor,
        sample_id: Hashable,
        *,
        allow_shape_mismatch: bool,
    ) -> tuple[Sample, tuple[str, ...]]:
        if len(model.inputs) != 1:
            input_ids = ", ".join(str(get_member_id(i)) for i in model.inputs)
            raise ValueError(
                f"model requires {len(model.inputs)} input tensors ({input_ids}), "
                + "but each data entry provides one tensor"
            )

        input_descr = model.inputs[0]
        input_id = get_member_id(input_descr)
        axis_infos = get_axes_infos(input_descr)
        expected_axes = [axis.id for axis in axis_infos]
        minimum_sizes = [axis.size.min for axis in axis_infos]
        missing_required_axes = [
            str(axis)
            for axis, minimum in zip(expected_axes, minimum_sizes)
            if axis not in tensor.dims and minimum > 1
        ]
        if missing_required_axes:
            raise ValueError(
                "missing required non-singleton axes "
                + ", ".join(missing_required_axes)
            )

        aligned = tensor.transpose(
            expected_axes, extra_dims="raise", missing_dims="expand"
        )

        invalid_sizes: list[str] = []
        unblockable_sizes: list[str] = []
        for axis, actual, info in zip(expected_axes, aligned.shape, axis_infos):
            size = info.size
            if (
                actual < size.min
                or (size.max is not None and actual > size.max)
                or (size.step is not None and (actual - size.min) % size.step != 0)
            ):
                expected = f">= {size.min}"
                if size.max is not None:
                    expected += f" and <= {size.max}"
                if size.step is not None:
                    expected += f" in steps of {size.step}"
                message = f"axis '{axis}' has size {actual}, expected {expected}"
                invalid_sizes.append(message)
                if info.type not in ("space", "time"):
                    unblockable_sizes.append(message)

        if unblockable_sizes:
            raise ValueError("; ".join(unblockable_sizes))

        if invalid_sizes and not allow_shape_mismatch:
            raise ValueError(
                "; ".join(invalid_sizes) + ". Enable 'Apply blockwise' for this model"
            )

        return (
            create_sample_for_model(
                model, inputs={input_id: aligned}, sample_id=sample_id
            ),
            tuple(invalid_sizes),
        )

    _ = st.set_page_config(
        page_title="BioImage.IO prediction comparison", layout="wide"
    )
    _ = st.title("BioImage.IO prediction comparison")
    _ = st.caption("Run several BioImage.IO models on your own arrays and images.")
    if "bench_server" not in st.session_state:
        st.session_state.bench_server = initial.server
    if "bench_data_ids" not in st.session_state:
        data_count = cast(
            int,
            st.session_state.get("bench_data_count", len(initial.data)),
        )
        st.session_state.bench_data_ids = list(range(data_count))
        st.session_state.bench_next_data_id = data_count
        for index, source in enumerate(initial.data):
            if f"bench_data_source_{index}" not in st.session_state:
                st.session_state[f"bench_data_source_{index}"] = source
    if "bench_model_ids" not in st.session_state:
        model_count = cast(
            int,
            st.session_state.get("bench_model_count", len(initial.model)),
        )
        st.session_state.bench_model_ids = list(range(model_count))
        st.session_state.bench_next_model_id = model_count
        for index, source in enumerate(initial.model):
            if f"bench_model_{index}" not in st.session_state:
                st.session_state[f"bench_model_{index}"] = source

    data_ids = list(cast(list[int], st.session_state.bench_data_ids))

    model_ids = list(cast(list[int], st.session_state.bench_model_ids))

    server = st.text_input(
        "Remote inference server (optional)",
        key="bench_server",
        placeholder="Leave blank for local inference",
        help=(
            "Provide a BioImage.IO Gradio server URL or Hugging Face Space name to "
            "run remotely. Leave blank to run locally."
        ),
    ).strip()

    compatibility_errors: list[str] = []
    data_entries: list[tuple[str, Tensor]] = []
    data_signature: list[DataSignature] = []
    selected_data_sources: list[str] = []

    def collect_uploads(row_id: int, uploader_key: str) -> None:
        from streamlit.typing import UploadedFile

        uploads = cast(list[UploadedFile], st.session_state[uploader_key])
        stored_uploads = cast(
            list[_UploadedSample],
            st.session_state.get(f"bench_dataset_uploads_{row_id}", []),
        )
        known_ids = {upload.id for upload in stored_uploads}
        st.session_state[f"bench_dataset_uploads_{row_id}"] = [
            *stored_uploads,
            *(
                _UploadedSample(upload.file_id, upload.name, upload.getvalue())
                for upload in uploads
                if upload.file_id not in known_ids
            ),
        ]
        # Clear the staging uploader after copying files into dataset state.
        st.session_state[f"bench_dataset_upload_version_{row_id}"] = (
            cast(int, st.session_state.get(f"bench_dataset_upload_version_{row_id}", 0))
            + 1
        )

    _ = st.subheader("Data")
    with (
        st.popover("Add dataset", icon=":material/add:"),
        st.form("bench_add_dataset_form", clear_on_submit=True),
    ):
        new_name = st.text_input("Dataset name", key="bench_new_dataset_name")
        new_uploads = st.file_uploader(
            "Upload data",
            type=DATA_EXTENSIONS,
            accept_multiple_files=True,
            key="bench_new_dataset_uploads",
        )
        new_sources = st.text_area(
            "Data paths, URLs, globs, or directories",
            key="bench_new_dataset_sources",
            placeholder="/path/to/images/*.tif",
            help="One entry per line. Use **/*.tif to include subdirectories.",
        )
        if st.form_submit_button("Add", key="bench_add_dataset"):
            row_id = cast(int, st.session_state.bench_next_data_id)
            st.session_state.bench_data_ids = [*data_ids, row_id]
            st.session_state.bench_next_data_id = row_id + 1
            st.session_state[f"bench_dataset_name_{row_id}"] = (
                new_name.strip() or f"Dataset {row_id + 1}"
            )
            st.session_state[f"bench_data_source_{row_id}"] = new_sources
            st.session_state[f"bench_dataset_uploads_{row_id}"] = [
                _UploadedSample(upload.file_id, upload.name, upload.getvalue())
                for upload in new_uploads
            ]
            st.rerun()

    for row_id in data_ids:
        name_key = f"bench_dataset_name_{row_id}"
        st.session_state.setdefault(name_key, f"Dataset {row_id + 1}")
        dataset_name = (
            cast(str, st.session_state[name_key]).strip() or f"Dataset {row_id + 1}"
        )
        with st.expander(dataset_name, expanded=True):
            data_column, delete_column = st.columns(
                [10, 1], vertical_alignment="bottom"
            )
            dataset_name = (
                data_column.text_input("Dataset name", key=name_key).strip()
                or f"Dataset {row_id + 1}"
            )
            if delete_column.button(
                "Delete",
                key=f"bench_delete_data_{row_id}",
                icon=":material/delete:",
                help=f"Delete {dataset_name}",
                width="stretch",
            ):
                st.session_state.bench_data_ids = [
                    existing for existing in data_ids if existing != row_id
                ]
                st.session_state.pop(f"bench_dataset_uploads_{row_id}", None)
                st.rerun()

            version = st.session_state.get(f"bench_dataset_upload_version_{row_id}", 0)
            uploader_key = f"bench_data_{row_id}_{version}"
            _ = st.file_uploader(
                "Upload more data",
                type=DATA_EXTENSIONS,
                accept_multiple_files=True,
                key=uploader_key,
                on_change=collect_uploads,
                args=(row_id, uploader_key),
            )
            source_text = st.text_area(
                "Data paths, URLs, globs, or directories",
                key=f"bench_data_source_{row_id}",
                help=(
                    "One entry per line. Directories include supported files directly "
                    "inside them; use **/*.tif for subdirectories."
                ),
            ).strip()
            if source_text:
                selected_data_sources.append(source_text)
            try:
                sources = _resolve_data_sources(source_text)
            except Exception as exc:
                message = f"Could not select data for '{dataset_name}': {exc}"
                compatibility_errors.append(message)
                _ = st.error(message)
                sources = []

            uploads = cast(
                list[_UploadedSample],
                st.session_state.get(f"bench_dataset_uploads_{row_id}", []),
            )
            samples: list[tuple[str, str, _UploadedSample | None]] = [
                (source, Path(source).name or source, None) for source in sources
            ]
            samples.extend((upload.id, upload.name, upload) for upload in uploads)
            _ = st.caption(
                f"{len(samples)} {'sample' if len(samples) == 1 else 'samples'}"
            )
            if not samples:
                _ = st.info(
                    "Upload data or enter paths to add samples to this dataset."
                )
            pending_roi = cast(
                dict[str, tuple[int, int]] | None,
                st.session_state.pop(f"bench_dataset_pending_roi_{row_id}", None),
            )
            for data_id, data_name, upload in samples:
                sample_key = _sample_key(row_id, data_id)
                data_column = st.expander(data_name, expanded=len(samples) == 1)
                if upload is not None and data_column.button(
                    "Remove sample",
                    key=f"bench_remove_sample_{sample_key}",
                    icon=":material/delete:",
                ):
                    st.session_state[f"bench_dataset_uploads_{row_id}"] = [
                        existing for existing in uploads if existing.id != data_id
                    ]
                    st.rerun()
                try:
                    if upload is None:
                        array, inferred_dims = load_source_tensor(data_id)
                    else:
                        array, inferred_dims = load_uploaded_tensor(
                            upload.name, upload.content
                        )
                except Exception as exc:
                    message = f"Could not read '{data_name}' in '{dataset_name}': {exc}"
                    compatibility_errors.append(message)
                    _ = data_column.error(message)
                    continue

                dims_key = f"bench_data_dims_{sample_key}"
                data_key = f"bench_data_file_{sample_key}"
                if (
                    st.session_state.get(data_key) != (data_id, array.shape)
                    or dims_key not in st.session_state
                ):
                    st.session_state[data_key] = (data_id, array.shape)
                    st.session_state[dims_key] = ", ".join(inferred_dims)
                    for dim_index in range(array.ndim):
                        st.session_state.pop(
                            f"bench_data_roi_{sample_key}_{dim_index}", None
                        )
                        st.session_state.pop(
                            f"bench_data_slice_{sample_key}_{dim_index}", None
                        )

                _ = data_column.caption(f"Shape: {array.shape} · dtype: {array.dtype}")
                dims_text = data_column.text_input(
                    "Dimension names",
                    key=dims_key,
                    help=(
                        "Comma- or space-separated BioImage.IO axis names, in array order. "
                        "Examples: `y, x` or `batch, channel, y, x`."
                    ),
                )
                dims = tuple(
                    part.strip()
                    for part in dims_text.replace(",", " ").split()
                    if part.strip()
                )
                if len(dims) != array.ndim:
                    message = (
                        f"'{data_name}' has {array.ndim} dimensions, but {len(dims)} "
                        f"dimension names were provided: {dims}"
                    )
                    compatibility_errors.append(message)
                    _ = data_column.error(message)
                    continue

                try:
                    tensor = Tensor.from_numpy(array, dims=dims)
                except Exception as exc:
                    message = f"Invalid dimension names for '{data_name}': {exc}"
                    compatibility_errors.append(message)
                    _ = data_column.error(message)
                    continue

                normalized_dims = tuple(str(dim) for dim in tensor.dims)
                roi_slices: dict[AxisId, slice] = {}
                roi_signature: list[tuple[str, int, int]] = []
                browser = data_column.expander("Browse and select ROI", expanded=True)
                selection_key = f"bench_selection_{sample_key}"
                selection = browser.container(key=selection_key)
                fullscreen = selection.toggle(
                    "Full-screen selection",
                    key=f"bench_selection_fullscreen_{sample_key}",
                    help="Enlarge the image with ROI and viewer controls beside it.",
                )
                if fullscreen:
                    # Use an explicit opaque surface: Streamlit does not expose a
                    # --st-background-color CSS variable on native containers.
                    background = st.get_option("theme.backgroundColor") or (
                        "#0e1117" if st.context.theme.type == "dark" else "#ffffff"
                    )
                    # Full-screen the whole selection surface, including its controls.
                    _ = st.html(
                        f"""<style>
                        .st-key-{selection_key} {{
                            position: fixed;
                            inset: 0;
                            z-index: 999990;
                            background-color: {background};
                            padding: 1.5rem;
                            overflow: auto;
                        }}
                        </style>"""
                    )
                    image_column, controls = selection.columns([3, 1])
                else:
                    controls = selection.container()
                    image_column = selection.container()
                roi_controls = controls.container(border=True)
                _ = roi_controls.markdown("**ROI selection · prediction input**")
                _ = roi_controls.caption(
                    "Range sliders crop the data used for prediction. "
                    "Both endpoints are included."
                )
                for dim_index, (dim, size) in enumerate(zip(tensor.dims, tensor.shape)):
                    roi_key = f"bench_data_roi_{sample_key}_{dim_index}"
                    if pending_roi is not None and str(dim) in pending_roi and size > 1:
                        start, stop = pending_roi[str(dim)]
                        start = min(max(start, 0), size - 1)
                        last = min(max(stop - 1, start), size - 1)
                        st.session_state[roi_key] = (start, last)
                    if size == 1:
                        start, stop = 0, 1
                        _ = roi_controls.caption(f"{dim}: singleton")
                    else:
                        st.session_state.setdefault(roi_key, (0, size - 1))
                        start, last = cast(
                            tuple[int, int],
                            roi_controls.slider(
                                f"{dim} ROI range",
                                min_value=0,
                                max_value=size - 1,
                                key=roi_key,
                                help="Crop the prediction input along this axis, including both endpoints.",
                            ),
                        )
                        stop = last + 1

                    roi_slices[dim] = slice(start, stop)
                    roi_signature.append((str(dim), start, stop))

                if len(samples) > 1 and roi_controls.button(
                    "Sync ROI selection",
                    key=f"bench_sync_roi_{sample_key}",
                    icon=":material/sync:",
                    help=(
                        "Copy this image's current ROI ranges to all samples in this "
                        "dataset. Axes are matched by name and ranges are clipped to "
                        "each image's size."
                    ),
                ):
                    # Apply on the next pass, before any sample's sliders render.
                    st.session_state[f"bench_dataset_pending_roi_{row_id}"] = {
                        dim: (start, stop) for dim, start, stop in roi_signature
                    }
                    st.rerun()

                view = tensor[roi_slices]
                viewer_controls = controls.container(border=True)
                _ = viewer_controls.markdown("**Viewer · preview only**")
                _ = viewer_controls.caption(
                    "Slice controls change the displayed plane only; "
                    "the selected ROI is used for prediction."
                )
                preview_indices: dict[AxisId, int] = {}
                slice_labels: list[str] = []
                for dim_index, (dim, start, stop) in enumerate(roi_signature):
                    if dim not in ("z", "time"):
                        continue

                    label = "Z slice" if dim == "z" else "Time frame"
                    if stop - start == 1:
                        index = start
                        _ = viewer_controls.caption(f"{label}: {index}")
                    else:
                        slice_key = f"bench_data_slice_{sample_key}_{dim_index}"
                        previous = cast(int, st.session_state.get(slice_key, start))
                        st.session_state[slice_key] = min(
                            max(previous, start), stop - 1
                        )
                        index = viewer_controls.slider(
                            label,
                            min_value=start,
                            max_value=stop - 1,
                            step=1,
                            key=slice_key,
                            help="Preview a single slice within the ROI without changing the prediction input.",
                        )

                    preview_indices[AxisId(dim)] = index - start
                    slice_labels.append(f"{dim}={index}")

                preview = view[preview_indices] if preview_indices else view
                caption = f"Selected ROI · shape {view.shape}"
                if slice_labels:
                    caption += " · " + ", ".join(slice_labels)
                preview_key = f"bench_selection_preview_{sample_key}"
                fullscreen_button_key = f"bench_selection_expand_{sample_key}"
                image_surface = image_column.container(key=preview_key)
                # Replace the image-only fullscreen action with the selection action.
                # Scope the toolbar override to this dataset preview.
                _ = st.html(
                    f"""<style>
                    .st-key-{preview_key} {{ position: relative; }}
                    .st-key-{preview_key} button[aria-label^="Fullscreen"] {{
                        display: none;
                    }}
                    .st-key-{fullscreen_button_key} {{
                        position: absolute;
                        top: 0.5rem;
                        right: 0.5rem;
                        z-index: 1;
                        opacity: 0;
                    }}
                    .st-key-{preview_key}:hover .st-key-{fullscreen_button_key},
                    .st-key-{preview_key}:focus-within .st-key-{fullscreen_button_key} {{
                        opacity: 1;
                    }}
                    @media (hover: none) {{
                        .st-key-{fullscreen_button_key} {{ opacity: 1; }}
                    }}
                    </style>"""
                )
                _ = image_surface.image(
                    _preview(
                        preview.to_numpy(), dims=tuple(str(dim) for dim in preview.dims)
                    ),
                    caption=caption,
                    alt=f"Preview of {data_name}"
                    + (" at " + ", ".join(slice_labels) if slice_labels else ""),
                    width="stretch",
                )

                _ = image_surface.button(
                    "Exit full-screen selection"
                    if fullscreen
                    else "Full-screen selection",
                    icon=":material/fullscreen_exit:"
                    if fullscreen
                    else ":material/fullscreen:",
                    key=fullscreen_button_key,
                    help="Show the image with ROI and slice controls, or return to the page.",
                    on_click=st.session_state.__setitem__,
                    args=(f"bench_selection_fullscreen_{sample_key}", not fullscreen),
                )

                display_name = f"{dataset_name} · {data_name}"
                data_entries.append((display_name, view))
                data_signature.append(
                    (
                        f"{row_id}:{data_id}",
                        display_name,
                        normalized_dims,
                        tuple(roi_signature),
                    )
                )

    _ = st.subheader("Models")
    models: list[tuple[int, str, v0_5.ModelDescr, bool]] = []
    selected_model_sources: list[str] = []

    def add_models() -> None:
        sources = [
            source.strip()
            for source in str(st.session_state.bench_new_models).split(",")
            if source.strip()
        ]
        for source in sources:
            row_id = cast(int, st.session_state.bench_next_model_id)
            st.session_state[f"bench_model_{row_id}"] = source
            st.session_state.bench_model_ids.append(row_id)
            st.session_state.bench_next_model_id = row_id + 1
        st.session_state.bench_new_models = ""

    with st.form("bench_add_models", border=False, enter_to_submit=True):
        input_column, add_column = st.columns([10, 1], vertical_alignment="bottom")
        _ = input_column.text_input(
            "Add models by ID, URL, or path",
            key="bench_new_models",
            placeholder="Enter one or more models, separated by commas",
        )
        with add_column:
            _ = st.form_submit_button(
                "Add model",
                key="bench_add_model",
                on_click=add_models,
                width="stretch",
            )
    if model_ids:
        for column, label in zip(
            st.columns([4, 4, 2, 1]),
            ("Model ID, URL, or path", "Model card", "Blockwise", ""),
        ):
            _ = column.markdown(f"**{label}**" if label else "")
    else:
        _ = st.caption("No models selected. Add a model to get started.")

    for row_id in model_ids:
        model_column, name_column, blockwise_column, delete_column = st.columns(
            [4, 4, 2, 1], vertical_alignment="top"
        )
        card = name_column.container(border=True)
        name_slot = card.empty()
        source = model_column.text_input(
            "Model ID, URL, or path",
            key=f"bench_model_{row_id}",
            label_visibility="collapsed",
            placeholder="BioImage.IO model ID, URL, or local bioimageio.yaml/package path",
        ).strip()
        _ = name_slot.markdown(f"**{source or 'New model'}**")
        if delete_column.button(
            "Delete",
            key=f"bench_delete_model_{row_id}",
            icon=":material/delete:",
            help=f"Delete {source or 'new model'}",
            disabled=not source,
            width="stretch",
        ):
            st.session_state.bench_model_ids = [
                existing for existing in model_ids if existing != row_id
            ]
            st.rerun()

        if not source:
            continue

        selected_model_sources.append(source)
        try:
            model = load_model_cached(source)
        except Exception as exc:
            message = f"Could not load model '{source}': {exc}"
            compatibility_errors.append(message)
            _ = model_column.error(message)
            continue

        _ = name_slot.markdown(f"**{model.name}**")
        try:
            cover = load_model_cover_cached(source)
            if cover is not None:
                _ = card.image(cover, width="stretch", alt=f"Cover for {model.name}")
        except Exception:
            _ = card.caption("Model cover could not be loaded.")
        description = getattr(model, "description", "")
        if description:
            _ = card.markdown(description)
        _ = card.caption(f"Format {model.format_version}")
        shape_mismatch = False
        for data_name, tensor in data_entries:
            try:
                _, invalid_sizes = create_compatible_sample(
                    model,
                    tensor,
                    data_name,
                    allow_shape_mismatch=True,
                )
            except Exception:
                invalid_sizes = ()
            else:
                shape_mismatch = shape_mismatch or bool(invalid_sizes)

        blockwise_key = f"bench_model_blockwise_{row_id}"
        auto_key = f"bench_model_blockwise_auto_{row_id}"
        manual_key = f"bench_model_blockwise_manual_{row_id}"
        pending_key = f"bench_model_blockwise_pending_{row_id}"
        auto_signature = (source, tuple(data_signature), shape_mismatch)
        if st.session_state.get(auto_key) != auto_signature:
            st.session_state[auto_key] = auto_signature
            st.session_state[blockwise_key] = shape_mismatch
            st.session_state[manual_key] = False

        if st.session_state.pop(pending_key, False):
            st.session_state[blockwise_key] = True
            st.session_state[manual_key] = False

        def record_manual_blockwise_choice(
            blockwise_key: str = blockwise_key, manual_key: str = manual_key
        ) -> None:
            st.session_state[manual_key] = not bool(st.session_state[blockwise_key])

        apply_blockwise = blockwise_column.checkbox(
            "Apply blockwise",
            key=blockwise_key,
            on_change=record_manual_blockwise_choice,
            help=(
                "Automatically enabled when a current data view does not match the "
                "model input shape or non-blockwise prediction fails, unless you "
                "manually disable it."
            ),
        )
        models.append((row_id, source, model, apply_blockwise))

    if initial.state_path is not None:
        _ = Path(initial.state_path).write_text(
            json.dumps(
                {
                    "server": server or None,
                    "data": selected_data_sources,
                    "model": selected_model_sources,
                }
            ),
            encoding="utf-8",
        )

    compatible_samples: dict[tuple[int, int], Sample] = {}
    for model_index, (_row_id, source, model, apply_blockwise) in enumerate(models):
        for data_index, (data_name, tensor) in enumerate(data_entries):
            try:
                compatible_samples[model_index, data_index], _ = (
                    create_compatible_sample(
                        model,
                        tensor,
                        data_name,
                        allow_shape_mismatch=apply_blockwise,
                    )
                )
            except Exception as exc:
                axes = ", ".join(str(dim) for dim in tensor.dims)
                compatibility_errors.append(
                    f"Data '{data_name}' with shape {tensor.shape} and axes "
                    + f"({axes}) is incompatible with model '{source}': {exc}"
                )

    if compatibility_errors:
        _ = st.error(
            "The current data and model selection is incompatible:\n\n"
            + "\n".join(f"- {error}" for error in compatibility_errors)
        )
        _ = st.info(
            "Edit the dimension names beneath each data file so they describe the "
            + "array correctly and match every model's required axes and sizes."
        )

    selection_is_compatible = (
        bool(data_entries) and bool(models) and not compatibility_errors
    )
    run_comparisons = st.checkbox(
        "Run comparisons",
        disabled=not selection_is_compatible,
        help=(
            "While active, predictions update automatically when the current ROI, "
            "dimension names, data, models, or server changes."
        ),
    )

    signature: RunSignature = (
        server,
        tuple(
            (source, apply_blockwise)
            for _row_id, source, _model, apply_blockwise in models
        ),
        tuple(data_signature),
    )
    stored = cast(StoredResults | None, st.session_state.get("bench_results"))
    if (
        run_comparisons
        and selection_is_compatible
        and (stored is None or stored[0] != signature)
    ):
        results: dict[tuple[int, int], PredictionResult] = {}
        progress = st.progress(0, text="Running predictions…")
        total = len(models) * len(data_entries)
        completed = 0
        for model_index, (row_id, _source, model, apply_blockwise) in enumerate(models):
            try:
                pipeline_context = (
                    create_remote_prediction_pipeline(model, server=server)
                    if server
                    else create_prediction_pipeline(model)
                )
                with pipeline_context as pipeline:
                    for data_index, _data in enumerate(data_entries):
                        try:
                            input_sample = compatible_samples[model_index, data_index]
                            sample = (
                                pipeline.predict_with_blocking(input_sample)
                                if apply_blockwise
                                else pipeline.predict_without_blocking(input_sample)
                            )
                            results[model_index, data_index] = {
                                str(member): tensor.to_numpy()
                                for member, tensor in sample.members.items()
                            }
                        except Exception as exc:
                            if not apply_blockwise and not st.session_state.get(
                                f"bench_model_blockwise_manual_{row_id}", False
                            ):
                                st.session_state[
                                    f"bench_model_blockwise_pending_{row_id}"
                                ] = True
                                st.rerun()

                            results[model_index, data_index] = exc
                        completed += 1
                        _ = progress.progress(
                            completed / total, text="Running predictions…"
                        )
            except Exception as exc:
                for data_index in range(len(data_entries)):
                    results[model_index, data_index] = exc
                    completed += 1
                _ = progress.progress(completed / total, text="Running predictions…")

        _ = progress.empty()
        stored = (
            signature,
            [name for name, _ in data_entries],
            results,
        )
        st.session_state.bench_results = stored

    if (
        not run_comparisons
        or not selection_is_compatible
        or stored is None
        or stored[0] != signature
    ):
        return

    _, data_names, results = stored
    _ = st.divider()
    _ = st.header("Predictions")
    for data_index, data_name in enumerate(data_names):
        _ = st.subheader(data_name)
        input_tensor = data_entries[data_index][1]
        input_dims = tuple(str(dim) for dim in input_tensor.dims)
        output_dims = {
            (model_index, str(get_member_id(output))): tuple(
                str(axis.id) for axis in output.axes
            )
            for model_index, (_, _, model, _) in enumerate(models)
            for output in model.outputs
        }
        axis_sizes: dict[str, int] = {}
        tensors = [(input_tensor.to_numpy(), input_dims)]
        for model_index in range(len(models)):
            result = results[model_index, data_index]
            if not isinstance(result, Exception):
                tensors.extend(
                    (array, output_dims[model_index, name])
                    for name, array in result.items()
                )
        for array, dims in tensors:
            for dim, size in zip(dims, array.shape):
                if dim not in ("y", "x", "channel") and size > 1:
                    axis_sizes[dim] = max(axis_sizes.get(dim, 1), size)

        indices: dict[str, int] = {}
        for dim, size in axis_sizes.items():
            key = f"bench_comparison_slice_{data_signature[data_index][0]}_{dim}"
            st.session_state[key] = min(
                max(cast(int, st.session_state.get(key, 0)), 0), size - 1
            )
            indices[dim] = st.slider(
                f"{dim} slice",
                min_value=0,
                max_value=size - 1,
                step=1,
                key=key,
                help=(
                    "Shared by the input and outputs with this dimension. "
                    "Shorter tensors use their last slice. Indices are relative to the ROI."
                ),
            )

        columns = st.columns(len(models) + 1)
        _ = columns[0].markdown("**Input**")
        _ = columns[0].image(
            _preview(tensors[0][0], dims=input_dims, indices=indices),
            caption=f"Input · shape {input_tensor.shape} · {tensors[0][0].dtype}",
            alt=f"Input for {data_name}",
            width="stretch",
        )
        for model_index, (
            column,
            (_row_id, source, _model, _blockwise),
        ) in enumerate(zip(columns[1:], models)):
            _ = column.markdown(f"**{source}**")
            result = results[model_index, data_index]
            if isinstance(result, Exception):
                _ = column.error(str(result))
                continue

            for output_name, array in result.items():
                _ = column.image(
                    _preview(
                        array,
                        dims=output_dims[model_index, output_name],
                        indices=indices,
                    ),
                    caption=f"{output_name} · shape {array.shape} · {array.dtype}",
                    alt=f"{output_name} prediction for {data_name}",
                    width="stretch",
                )
            _ = column.download_button(
                "Download outputs",
                _result_archive(result),
                file_name=f"{Path(data_name).name}-{data_index + 1}-{model_index + 1}-outputs.npz",
                mime="application/octet-stream",
                key=f"bench_download_{data_index}_{model_index}",
                width="stretch",
            )


if __name__ == "__main__":
    main()
