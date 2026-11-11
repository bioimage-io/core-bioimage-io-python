from __future__ import annotations

import io
import json
import sys
from argparse import ArgumentParser
from collections.abc import Hashable, Mapping, Sequence
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


def _preview(array: NDArray[Any]) -> NDArray[Any]:
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
            st.session_state.get("bench_data_count", max(1, len(initial.data))),
        )
        st.session_state.bench_data_ids = list(range(data_count))
        st.session_state.bench_next_data_id = data_count
        for index, source in enumerate(initial.data):
            if f"bench_data_source_{index}" not in st.session_state:
                st.session_state[f"bench_data_source_{index}"] = source
    if "bench_model_ids" not in st.session_state:
        model_count = cast(
            int,
            st.session_state.get("bench_model_count", max(1, len(initial.model))),
        )
        st.session_state.bench_model_ids = list(range(model_count))
        st.session_state.bench_next_model_id = model_count
        for index, source in enumerate(initial.model):
            if f"bench_model_{index}" not in st.session_state:
                st.session_state[f"bench_model_{index}"] = source

    data_ids = list(cast(list[int], st.session_state.bench_data_ids))
    data_ids, next_data_id = _with_trailing_empty_row(
        data_ids,
        {
            row_id
            for row_id in data_ids
            if st.session_state.get(f"bench_data_source_{row_id}")
            or st.session_state.get(f"bench_data_{row_id}") is not None
        },
        cast(int, st.session_state.bench_next_data_id),
    )
    st.session_state.bench_data_ids = data_ids
    st.session_state.bench_next_data_id = next_data_id

    model_ids = list(cast(list[int], st.session_state.bench_model_ids))
    model_ids, next_model_id = _with_trailing_empty_row(
        model_ids,
        {
            row_id
            for row_id in model_ids
            if st.session_state.get(f"bench_model_{row_id}")
        },
        cast(int, st.session_state.bench_next_model_id),
    )
    st.session_state.bench_model_ids = model_ids
    st.session_state.bench_next_model_id = next_model_id

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

    _ = st.subheader("Data")
    for row_id in data_ids:
        data_column, delete_column = st.columns([10, 1], vertical_alignment="top")
        name_slot = data_column.empty()
        upload = data_column.file_uploader(
            "Upload data",
            type=[
                "npy",
                "tif",
                "tiff",
                "png",
                "jpg",
                "jpeg",
                "h5",
                "hdf5",
            ],
            key=f"bench_data_{row_id}",
        )
        data_source = (
            data_column.text_input(
                "Data path or URL",
                key=f"bench_data_source_{row_id}",
                placeholder="Optional path or URL",
            ).strip()
            if upload is None
            else ""
        )
        display_name = (
            upload.name
            if upload is not None
            else (Path(data_source).name or data_source)
            if data_source
            else "New data"
        )
        _ = name_slot.markdown(f"**{display_name}**")
        in_use = bool(data_source or upload is not None)
        if delete_column.button(
            "Delete",
            key=f"bench_delete_data_{row_id}",
            icon=":material/delete:",
            help=f"Delete {display_name}",
            disabled=not in_use,
            width="stretch",
        ):
            st.session_state.bench_data_ids = [
                existing for existing in data_ids if existing != row_id
            ]
            st.rerun()

        if data_source:
            selected_data_sources.append(data_source)

        if not in_use:
            continue

        try:
            if data_source:
                array, inferred_dims = load_source_tensor(data_source)
                data_name = Path(data_source).name or data_source
                data_id = data_source
            else:
                assert upload is not None
                array, inferred_dims = load_uploaded_tensor(
                    upload.name, upload.getvalue()
                )
                data_name = upload.name
                data_id = upload.file_id
        except Exception as exc:
            source = data_source or (upload.name if upload is not None else "")
            message = f"Could not read '{source}': {exc}"
            compatibility_errors.append(message)
            _ = data_column.error(message)
            continue

        dims_key = f"bench_data_dims_{row_id}"
        data_key = f"bench_data_file_{row_id}"
        if st.session_state.get(data_key) != data_id:
            st.session_state[data_key] = data_id
            st.session_state[dims_key] = ", ".join(inferred_dims)
            for dim_index in range(array.ndim):
                st.session_state.pop(f"bench_data_roi_{row_id}_{dim_index}", None)

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
            part.strip() for part in dims_text.replace(",", " ").split() if part.strip()
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
        for dim_index, (dim, size) in enumerate(zip(tensor.dims, tensor.shape)):
            if size == 1:
                start, stop = 0, 1
                _ = browser.caption(f"{dim}: singleton")
            else:
                start, last = browser.slider(
                    f"{dim} range",
                    min_value=0,
                    max_value=size - 1,
                    value=(0, size - 1),
                    key=f"bench_data_roi_{row_id}_{dim_index}",
                    help="Both displayed indices are included in the view.",
                )
                stop = last + 1

            roi_slices[dim] = slice(start, stop)
            roi_signature.append((str(dim), start, stop))

        view = tensor[roi_slices]
        _ = browser.image(
            _preview(view.to_numpy()),
            caption=f"Current view · shape {view.shape}",
            width="stretch",
        )

        data_entries.append((data_name, view))
        data_signature.append(
            (data_id, data_name, normalized_dims, tuple(roi_signature))
        )

    _ = st.subheader("Models")
    models: list[tuple[int, str, v0_5.ModelDescr, bool]] = []
    selected_model_sources: list[str] = []
    for row_id in model_ids:
        model_column, delete_column = st.columns([10, 1], vertical_alignment="top")
        name_slot = model_column.empty()
        source = model_column.text_input(
            "Model ID, URL, or path",
            key=f"bench_model_{row_id}",
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
        _ = model_column.caption(f"Format {model.format_version}")
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

        apply_blockwise = model_column.checkbox(
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
        columns = st.columns(len(models))
        for model_index, (
            column,
            (_row_id, source, _model, _blockwise),
        ) in enumerate(zip(columns, models)):
            _ = column.markdown(f"**{source}**")
            result = results[model_index, data_index]
            if isinstance(result, Exception):
                _ = column.error(str(result))
                continue

            for output_name, array in result.items():
                _ = column.image(
                    _preview(array),
                    caption=f"{output_name} · shape {array.shape} · {array.dtype}",
                    width="stretch",
                )
            _ = column.download_button(
                "Download outputs",
                _result_archive(result),
                file_name=f"{data_name}-{model_index + 1}-outputs.npz",
                mime="application/octet-stream",
                key=f"bench_download_{data_index}_{model_index}",
                width="stretch",
            )


if __name__ == "__main__":
    main()
