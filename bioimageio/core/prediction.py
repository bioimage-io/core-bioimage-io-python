import collections.abc
from pathlib import Path
from typing import (
    Any,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import xarray as xr
from loguru import logger
from numpy.typing import NDArray
from tqdm import tqdm

from bioimageio.core.axis import AxisId
from bioimageio.core.io import save_sample
from bioimageio.spec import load_description
from bioimageio.spec.common import PermissiveFileSource
from bioimageio.spec.model import v0_4, v0_5

from ._prediction_pipeline import PredictionPipeline, create_prediction_pipeline
from .common import MemberId, PerMember
from .digest_spec import create_sample_for_model
from .sample import Sample
from .tensor import Tensor


def predict(
    *,
    model: Union[
        PermissiveFileSource, v0_4.ModelDescr, v0_5.ModelDescr, PredictionPipeline
    ],
    inputs: Union[Sample, PerMember[Union[Tensor, xr.DataArray, NDArray[Any], Path]]],
    sample_id: Hashable = "sample",
    blocksize_parameter: Optional[
        Union[
            v0_5.ParameterizedSize_N,
            Mapping[Tuple[MemberId, AxisId], v0_5.ParameterizedSize_N],
        ]
    ] = None,
    input_block_shape: Optional[Mapping[MemberId, Mapping[AxisId, int]]] = None,
    skip_preprocessing: bool = False,
    skip_postprocessing: bool = False,
    save_output_path: Optional[Union[Path, str]] = None,
) -> Sample:
    """Run prediction for a single set of input(s) with a bioimage.io model

    Args:
        model: model to predict with.
            May be given as RDF source, model description or prediction pipeline.
        inputs: the input sample or the named input(s) for this model as a dictionary
        sample_id: the sample id.
        blocksize_parameter: (optional) tile the input into blocks parametrized by
            blocksize according to any parametrized axis sizes defined in the model RDF.
            Note: For a predetermined, fixed block shape use `input_block_shape`
        input_block_shape: (optional) tile the input sample tensors into blocks.
            Note: For a parameterized block shape, not dealing with the exact block shape,
            use `blocksie_parameter`.
        skip_preprocessing: flag to skip the model's preprocessing
        skip_postprocessing: flag to skip the model's postprocessing
        save_output_path: A path with `{member_id}` `{sample_id}` in it
            to save the output to.
    """
    if save_output_path is not None:
        if "{member_id}" not in str(save_output_path):
            raise ValueError(
                f"Missing `{{member_id}}` in save_output_path={save_output_path}"
            )

    if isinstance(model, PredictionPipeline):
        pp = model
    else:
        if not isinstance(model, (v0_4.ModelDescr, v0_5.ModelDescr)):
            loaded = load_description(model)
            if not isinstance(loaded, (v0_4.ModelDescr, v0_5.ModelDescr)):
                raise ValueError(f"expected model description, but got {loaded}")
            model = loaded

        pp = create_prediction_pipeline(model)

    if isinstance(inputs, Sample):
        sample = inputs
    else:
        sample = create_sample_for_model(
            pp.model_description, inputs=inputs, sample_id=sample_id
        )

    if input_block_shape is not None:
        if blocksize_parameter is not None:
            logger.warning(
                "ignoring blocksize_parameter={} in favor of input_block_shape={}",
                blocksize_parameter,
                input_block_shape,
            )

        output = pp.predict_sample_with_fixed_blocking(
            sample,
            input_block_shape=input_block_shape,
            skip_preprocessing=skip_preprocessing,
            skip_postprocessing=skip_postprocessing,
        )
    elif blocksize_parameter is not None:
        output = pp.predict_sample_with_blocking(
            sample,
            skip_preprocessing=skip_preprocessing,
            skip_postprocessing=skip_postprocessing,
            ns=blocksize_parameter,
        )
    else:
        output = pp.predict_sample_without_blocking(
            sample,
            skip_preprocessing=skip_preprocessing,
            skip_postprocessing=skip_postprocessing,
        )
    if save_output_path:
        save_sample(save_output_path, output)

    return output


def predict_many(
    *,
    model: Union[
        PermissiveFileSource, v0_4.ModelDescr, v0_5.ModelDescr, PredictionPipeline
    ],
    inputs: Iterable[PerMember[Union[Tensor, xr.DataArray, NDArray[Any], Path]]],
    sample_id: str = "sample{i:03}",
    blocksize_parameter: Optional[
        Union[
            v0_5.ParameterizedSize_N,
            Mapping[Tuple[MemberId, AxisId], v0_5.ParameterizedSize_N],
        ]
    ] = None,
    skip_preprocessing: bool = False,
    skip_postprocessing: bool = False,
    save_output_path: Optional[Union[Path, str]] = None,
) -> Iterator[Sample]:
    """Run prediction for a multiple sets of inputs with a bioimage.io model

    Args:
        model: model to predict with.
            May be given as RDF source, model description or prediction pipeline.
        inputs: An iterable of the named input(s) for this model as a dictionary.
        sample_id: the sample id.
            note: `{i}` will be formatted as the i-th sample.
            If `{i}` (or `{i:`) is not present and `inputs` is an iterable `{i:03}` is appended.
        blocksize_parameter: (optional) tile the input into blocks parametrized by
            blocksize according to any parametrized axis sizes defined in the model RDF
        skip_preprocessing: flag to skip the model's preprocessing
        skip_postprocessing: flag to skip the model's postprocessing
        save_output_path: A path with `{member_id}` `{sample_id}` in it
            to save the output to.
    """
    if save_output_path is not None:
        if "{member_id}" not in str(save_output_path):
            raise ValueError(
                f"Missing `{{member_id}}` in save_output_path={save_output_path}"
            )

        if not isinstance(inputs, collections.Mapping) and "{sample_id}" not in str(
            save_output_path
        ):
            raise ValueError(
                f"Missing `{{sample_id}}` in save_output_path={save_output_path}"
            )

    if isinstance(model, PredictionPipeline):
        pp = model
    else:
        if not isinstance(model, (v0_4.ModelDescr, v0_5.ModelDescr)):
            loaded = load_description(model)
            if not isinstance(loaded, (v0_4.ModelDescr, v0_5.ModelDescr)):
                raise ValueError(f"expected model description, but got {loaded}")
            model = loaded

        pp = create_prediction_pipeline(model)

    if not isinstance(inputs, collections.Mapping):
        sample_id = str(sample_id)
        if "{i}" not in sample_id and "{i:" not in sample_id:
            sample_id += "{i:03}"

        total = len(inputs) if isinstance(inputs, collections.abc.Sized) else None

        for i, ipts in tqdm(enumerate(inputs), total=total):
            yield predict(
                model=pp,
                inputs=ipts,
                sample_id=sample_id.format(i=i),
                blocksize_parameter=blocksize_parameter,
                skip_preprocessing=skip_preprocessing,
                skip_postprocessing=skip_postprocessing,
                save_output_path=save_output_path,
            )
