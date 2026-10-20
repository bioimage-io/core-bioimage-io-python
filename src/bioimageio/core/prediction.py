from __future__ import annotations

import atexit
import itertools
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence, Sized
from pathlib import Path
from typing import overload

import dask.array as da
from loguru import logger
from tqdm import tqdm

from bioimageio.core._utils.type_hints import cast_and_assert
from bioimageio.core.tensor import Tensor
from bioimageio.spec import load_description
from bioimageio.spec.common import PermissiveFileSource
from bioimageio.spec.model import v0_4, v0_5

from ._prediction_pipeline import PredictionPipeline, create_prediction_pipeline
from .axis import AxisId
from .common import BlocksizeParameter, MemberId, PerMember
from .digest_spec import TensorSource, create_sample_for_model, get_member_id
from .io import save_sample
from .sample import Sample


@overload
def predict(
    *,
    model: PermissiveFileSource
    | v0_4.ModelDescr
    | v0_5.ModelDescr
    | PredictionPipeline,
    inputs: Sample | PerMember[TensorSource] | TensorSource,
    sample_id: Hashable = "sample",
    blocksize_parameter: BlocksizeParameter | None = None,
    input_block_shape: Mapping[MemberId, Mapping[AxisId, int]] | None = None,
    skip_preprocessing: bool = False,
    skip_postprocessing: bool = False,
    output_path: None = None,
    devices: Sequence[str] | None = None,
) -> Sample: ...


@overload
def predict(
    *,
    model: PermissiveFileSource
    | v0_4.ModelDescr
    | v0_5.ModelDescr
    | PredictionPipeline,
    inputs: Sample | PerMember[TensorSource] | TensorSource,
    sample_id: Hashable = "sample",
    blocksize_parameter: BlocksizeParameter | None = None,
    input_block_shape: Mapping[MemberId, Mapping[AxisId, int]] | None = None,
    skip_preprocessing: bool = False,
    skip_postprocessing: bool = False,
    output_path: Path | str,
    devices: Sequence[str] | None = None,
) -> PerMember[Path]: ...


@overload
def predict(
    *,
    model: PermissiveFileSource
    | v0_4.ModelDescr
    | v0_5.ModelDescr
    | PredictionPipeline,
    inputs: Sample | PerMember[TensorSource] | TensorSource,
    sample_id: Hashable = "sample",
    blocksize_parameter: BlocksizeParameter | None = None,
    input_block_shape: Mapping[MemberId, Mapping[AxisId, int]] | None = None,
    skip_preprocessing: bool = False,
    skip_postprocessing: bool = False,
    output_path: Path | str | None = None,
    devices: Sequence[str] | None = None,
) -> Sample | PerMember[Path]: ...


def predict(
    *,
    model: PermissiveFileSource
    | v0_4.ModelDescr
    | v0_5.ModelDescr
    | PredictionPipeline,
    inputs: Sample | PerMember[TensorSource] | TensorSource,
    sample_id: Hashable = "sample",
    blocksize_parameter: BlocksizeParameter | None = None,
    input_block_shape: Mapping[MemberId, Mapping[AxisId, int]] | None = None,
    output_path: Path | str | None = None,
    devices: Sequence[str] | None = None,
    cache_path_after_preprocessing: Path | str | None = None,
    cache_path_before_postprocessing: Path | str | None = None,
) -> Sample | PerMember[Path]:
    """Run prediction for a single set of input(s) with a bioimage.io model

    Args:
        model: Model to predict with.
            May be given as RDF source, model description or prediction pipeline.
        inputs: the input sample or the named input(s) for this model as a dictionary
        sample_id: the sample id.
            The **sample_id** is used to format **output_path**
            and to distinguish sample specific log messages.
        blocksize_parameter: (optional) Tile the input into blocks parametrized by
            **blocksize_parameter** according to any parametrized axis sizes defined
            by the **model**.
            See `bioimageio.spec.model.v0_5.ParameterizedSize` for details.
            Note: For a predetermined, fixed block shape use **input_block_shape**.
        input_block_shape: (optional) Tile the input sample tensors into blocks.
            Note: Use **blocksize_parameter** for a parameterized block shape to
                run prediction independent of the exact block shape.
        cache_path_after_preprocessing: (experimental)
            A path to cache the input after preprocessing.
            Must have a '.zarr' suffix.
            Must include `{member_id}`.
            If the model has no preprocessing this will be ignored.
        cache_path_before_postprocessing: (experimental)
            A path to cache the output before postprocessing.
            Must have a '.zarr' suffix.
            Must include `{member_id}`.
            If the model has no postprocessing this will be ignored.
        output_path: A path with to save the output to.
            Must contain:
            - `{output_id}` (or `{member_id}`) if the model has multiple output tensors
            May contain:
            - `{sample_id}` to avoid overwriting recurrent calls

    Returns:
        The eager (numpy-based) output sample if **model** was not given as a prediction pipeline and **output_path** was None.
        The lazy (dask-based) output sample if **model** was given as a prediction pipeline and **output_path** was None.
        The mapping of output tensor id to output file path if **output_path** was given.
    """
    if cache_path_after_preprocessing is not None:
        if "{member_id}" not in str(cache_path_after_preprocessing):
            raise ValueError(
                f"Missing `{{member_id}}` in cache_path_after_preprocessing={cache_path_after_preprocessing}"
                + " to differentiate cached sample members."
            )
        if Path(cache_path_after_preprocessing).suffix != ".zarr":
            raise NotImplementedError(
                f"cache_path_after_preprocessing={cache_path_after_preprocessing} must have a '.zarr' suffix"
            )
    if cache_path_before_postprocessing is not None:
        if "{member_id}" not in str(cache_path_before_postprocessing):
            raise ValueError(
                f"Missing `{{member_id}}` in cache_path_before_postprocessing={cache_path_before_postprocessing}"
                + " to differentiate cached sample members."
            )
        if Path(cache_path_before_postprocessing).suffix != ".zarr":
            raise NotImplementedError(
                f"cache_path_before_postprocessing={cache_path_before_postprocessing} must have a '.zarr' suffix"
            )
        if cache_path_before_postprocessing == cache_path_after_preprocessing:
            raise ValueError(
                f"cache_path_before_postprocessing={cache_path_before_postprocessing} must be different from "
                + f"cache_path_after_preprocessing={cache_path_after_preprocessing}"
            )

    if isinstance(model, PredictionPipeline):
        pp = model
        model = pp.model_descr
        unload_pp = False
        if not pp.is_loaded:
            pp.load()
            _ = atexit.register(pp.unload)
    else:
        if not isinstance(model, (v0_4.ModelDescr, v0_5.ModelDescr)):
            loaded = load_description(model)
            if not isinstance(loaded, (v0_4.ModelDescr, v0_5.ModelDescr)):
                raise TypeError(f"expected model description, but got {loaded}")
            model = loaded

        pp = create_prediction_pipeline(
            model,
            fixed_dataset_statistics=inputs.stat if isinstance(inputs, Sample) else {},
            devices=devices,
        )
        pp.load()
        unload_pp = True

    try:
        model_descr = pp.model_descr
        if output_path is not None and (
            "{output_id}" not in str(output_path)
            and "{member_id}" not in str(output_path)
            and len(model_descr.outputs) > 1
        ):
            raise ValueError(
                f"Missing `{{output_id}}` in output_path={output_path} to "
                + "distinguish model outputs "
                + str([get_member_id(d) for d in model_descr.outputs])
            )

        if isinstance(inputs, Sample):
            sample = inputs
        else:
            sample = create_sample_for_model(
                pp.model_descr, inputs=inputs, sample_id=sample_id
            )

        pp.apply_input_padding(sample)
        pp.apply_preprocessing(sample)
        if cache_path_after_preprocessing is not None:
            for m in list(sample.members):
                cache_path = Path(
                    str(cache_path_after_preprocessing).format(member_id=m)
                )
                assert str(cache_path) != str(output_path)
                assert str(cache_path) != str(cache_path_after_preprocessing)
                assert str(cache_path) != str(cache_path_before_postprocessing)

                data = cast_and_assert(da.Array, sample.members[m].as_dask().data.data)
                logger.info(
                    "caching input sample member {} to {}",
                    m,
                    cache_path,
                )
                cached_data = da.to_zarr(
                    data,
                    cache_path,
                    compute=False,
                    return_stored=True,
                    overwrite=True,
                    mode="a",
                )
                cached_tensor = Tensor(
                    da.from_zarr(cached_data), sample.members[m].dims
                )
                sample.members[m] = cached_tensor

        if input_block_shape is not None:
            if blocksize_parameter is not None:
                logger.warning(
                    "ignoring blocksize_parameter={} in favor of input_block_shape={}",
                    blocksize_parameter,
                    input_block_shape,
                )

            output = pp.predict_with_fixed_blocking(
                sample,
                input_block_shape=input_block_shape,
                skip_input_padding=True,
                skip_preprocessing=True,
                skip_postprocessing=True,
                skip_output_cropping=True,
            )
        elif blocksize_parameter is not None:
            output = pp.predict_with_blocking(
                sample,
                ns=blocksize_parameter,
                skip_input_padding=True,
                skip_preprocessing=True,
                skip_postprocessing=True,
                skip_output_cropping=True,
            )
        else:
            output = pp.predict_without_blocking(
                sample,
                skip_input_padding=True,
                skip_preprocessing=True,
                skip_postprocessing=True,
                skip_output_cropping=True,
            )

        if cache_path_before_postprocessing is not None:
            for m in list(output.members):
                cache_path = Path(
                    str(cache_path_before_postprocessing).format(member_id=m)
                )
                assert str(cache_path) != str(output_path)
                assert str(cache_path) != str(cache_path_before_postprocessing)

                data = cast_and_assert(da.Array, output.members[m].as_dask().data.data)
                logger.info(
                    "caching output sample member {} to {}",
                    m,
                    cache_path,
                )
                cached_data = da.to_zarr(
                    data,
                    cache_path,
                    compute=False,
                    return_stored=True,
                    overwrite=True,
                    mode="a",
                )
                cached_tensor = Tensor(
                    da.from_zarr(cached_data), output.members[m].dims
                )
                output.members[m] = cached_tensor

        pp.apply_postprocessing(output)
        pp.apply_output_cropping(output)

        if output_path:
            return save_sample(output_path, output)
        elif unload_pp:
            return output.as_numpy()
        else:
            return output
    finally:
        if unload_pp:
            pp.unload()


@overload
def predict_many(
    *,
    model: PermissiveFileSource
    | v0_4.ModelDescr
    | v0_5.ModelDescr
    | PredictionPipeline,
    inputs: Iterable[PerMember[TensorSource]] | Iterable[TensorSource],
    sample_id: str | Iterable[str] = "sample{i:03}",
    blocksize_parameter: v0_5.ParameterizedSize_N
    | Mapping[tuple[MemberId, AxisId], v0_5.ParameterizedSize_N]
    | None = None,
    skip_preprocessing: bool = False,
    skip_postprocessing: bool = False,
    output_path: Path | str,
) -> Iterator[PerMember[Path]]: ...


@overload
def predict_many(
    *,
    model: PermissiveFileSource
    | v0_4.ModelDescr
    | v0_5.ModelDescr
    | PredictionPipeline,
    inputs: Iterable[PerMember[TensorSource]] | Iterable[TensorSource],
    sample_id: str | Iterable[str] = "sample{i:03}",
    blocksize_parameter: v0_5.ParameterizedSize_N
    | Mapping[tuple[MemberId, AxisId], v0_5.ParameterizedSize_N]
    | None = None,
    skip_preprocessing: bool = False,
    skip_postprocessing: bool = False,
    output_path: None = None,
) -> Iterator[Sample]: ...


def predict_many(
    *,
    model: PermissiveFileSource
    | v0_4.ModelDescr
    | v0_5.ModelDescr
    | PredictionPipeline,
    inputs: Iterable[PerMember[TensorSource]] | Iterable[TensorSource],
    sample_id: str | Iterable[str] = "sample{i:03}",
    blocksize_parameter: v0_5.ParameterizedSize_N
    | Mapping[tuple[MemberId, AxisId], v0_5.ParameterizedSize_N]
    | None = None,
    skip_preprocessing: bool = False,
    skip_postprocessing: bool = False,
    output_path: Path | str | None = None,
) -> Iterator[Sample] | Iterator[PerMember[Path]]:
    """Run prediction for a multiple sets of inputs with a bioimage.io model

    Args:
        model: Model to predict with.
            May be given as RDF source, model description or prediction pipeline.
        inputs: An iterable of the (named) input(s) for this model.
        sample_id: The sample id pattern or iterable that is not a string (specify samples a,b,c as ['a', 'b', 'c'], not 'abc').
            Note:  If the `sample_id` (item) contains `{i}`, it will be formatted as the i-th sample starting with i=0.
            If `{i}` (or `{i:`) is not present and `sample_id` is a single string, `{i:03}` is appended.
        blocksize_parameter: (optional) Tile the input into blocks parametrized by
            blocksize according to any parametrized axis sizes defined in the model RDF.
        skip_preprocessing: Flag to skip the model's preprocessing.
        skip_postprocessing: Flag to skip the model's postprocessing.
        output_path: A path to save the output to.
            Must contain:
            - `{sample_id}` to differentiate predicted samples
            - `{output_id}` (or `{member_id}`) if the model has multiple outputs

    Returns:
        An iterator of the mapping of output tensor id to output file path if **output_path** was given.
        An iterator of the lazy output samples if **output_path** was None and **model** was given as a prediction pipeline
        An iterator of the eager output samples if **output_path** was None and **model** was not given as a prediction pipeline
    """
    if output_path is not None and "{sample_id}" not in str(output_path):
        raise ValueError(
            f"Missing `{{sample_id}}` in output_path={output_path}"
            + " to differentiate predicted samples."
        )

    if isinstance(model, PredictionPipeline):
        pp = model
    else:
        if not isinstance(model, (v0_4.ModelDescr, v0_5.ModelDescr)):
            loaded = load_description(model)
            if not isinstance(loaded, (v0_4.ModelDescr, v0_5.ModelDescr)):
                raise TypeError(f"expected model description, but got {loaded}")
            model = loaded

        pp = create_prediction_pipeline(model)

    if isinstance(sample_id, str):
        if "{i}" not in sample_id and "{i:" not in sample_id:
            sample_id += "{i:03}"

        sample_ids = itertools.repeat(sample_id)
    else:
        sample_ids = sample_id

    total = (
        len(inputs)
        if isinstance(inputs, Sized)
        else len(sample_ids)
        if not isinstance(sample_ids, str) and isinstance(sample_ids, Sized)
        else None
    )

    for i, (s_id, ipts) in tqdm(enumerate(zip(sample_ids, inputs)), total=total):
        output = predict(
            model=pp,
            inputs=ipts,
            sample_id=s_id.format(i=i),
            blocksize_parameter=blocksize_parameter,
            skip_preprocessing=True,
            skip_postprocessing=True,
            output_path=output_path,
        )
        if output_path is None:
            assert isinstance(output, Sample)
            yield output
        else:
            assert isinstance(output, Mapping)
            yield output
