from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from itertools import chain
from types import MappingProxyType
from typing import (
    Any,
    Literal,
    NamedTuple,
    TypeVar,
)

from loguru import logger
from tqdm import tqdm
from typing_extensions import assert_never

from bioimageio.spec import load_model_description
from bioimageio.spec.model import AnyModelDescr, v0_4, v0_5

from ._model_adapter import ModelAdapter
from ._op_base import BlockwiseOperator, SamplewiseOperator
from .axis import AxisId, PerAxis
from .backends import create_model_adapter
from .common import (
    BlocksizeParameter,
    Halo,
    MemberId,
    PerMember,
    SampleId,
    SupportedWeightsFormat,
)
from .digest_spec import (
    get_block_transform,
    get_input_halo,
    get_member_ids,
)
from .proc_ops import Processing
from .proc_setup import setup_pre_and_postprocessing
from .sample import Sample, SampleBlock
from .stat_measures import Measure, MeasureValue, Stat
from .tensor import Tensor

Predict_IO = TypeVar(
    "Predict_IO",
    Sample,
    Iterable[Sample],
)


class IntermediatePrediction(NamedTuple):
    """Represents an intermediate prediction of a sample with blocking, including the predicted sample so far and the last predicted block.

    The final `IntermediatePrediction` in a sequence holds the complete predicted (and postprocessed if applicable) sample."""

    sample: Sample
    last_block: SampleBlock


class _PredictionPipelineBase(ABC):
    def __init__(
        self,
        model_descr: AnyModelDescr,
        *,
        default_blocksize_parameter: BlocksizeParameter,
        default_batch_size: int,
        preceding_prediction_pipelines: Sequence[
            PredictionPipeline | RemotePredictionPipeline
        ]
        | None,
    ) -> None:
        super().__init__()
        self._model_descr = model_descr
        self._default_blocksize_parameter = default_blocksize_parameter
        self._default_batch_size = default_batch_size
        # TODO: Improve parallelization of blockwise predictions with preceding prediction pipelines.
        self._preceding_prediction_pipelines = preceding_prediction_pipelines
        if isinstance(model_descr, v0_4.ModelDescr):
            self._default_output_halo: PerMember[PerAxis[Halo]] = {}
            self._default_input_halo: PerMember[PerAxis[Halo]] = {}
            self._block_transform = None
        else:
            self._default_output_halo = {
                t.id: {
                    a.id: Halo(a.halo, a.halo)
                    for a in t.axes
                    if isinstance(a, v0_5.WithHalo)
                }
                for t in model_descr.outputs
            }
            self._default_input_halo = get_input_halo(
                model_descr, self._default_output_halo
            )
            self._block_transform = get_block_transform(model_descr)

        self.pad_mode = (
            {}
            if isinstance(model_descr, v0_4.ModelDescr)
            else {
                descr.id: descr.pad or v0_5.SymmetricPadding()
                for descr in model_descr.inputs
            }
        )

        self._input_ids = tuple(get_member_ids(model_descr.inputs))
        self._output_ids = tuple(get_member_ids(model_descr.outputs))

    @property
    def input_ids(self) -> Sequence[MemberId]:
        return self._input_ids

    @property
    def output_ids(self) -> Sequence[MemberId]:
        return self._output_ids

    @property
    def model_descr(self) -> AnyModelDescr:
        return self._model_descr

    @property
    def model_description(self) -> AnyModelDescr:
        return self._model_descr

    def _get_preceding_prediction_pipelines_for_sample(
        self, sample: Sample
    ) -> Sequence[PredictionPipeline | RemotePredictionPipeline]:
        """Get preceding prediction pipelines for a sample based on the sample's input member ids."""
        if not self._preceding_prediction_pipelines:
            return ()

        required_inputs = set(self.input_ids)
        sample_members = set(sample.members.keys())
        preceding_pipelines: list[PredictionPipeline | RemotePredictionPipeline] = []
        for pp in self._preceding_prediction_pipelines[::-1]:
            preceding_pipelines.insert(0, pp)
            sample_members.update(pp.output_ids)
            required_inputs.update(pp.input_ids)
            required_inputs.difference_update(sample_members)

            if not required_inputs:
                return preceding_pipelines

        raise KeyError(
            f"Sample is missing required inputs {required_inputs} for the prediction pipeline or its preceding pipelines."
        )

    def predict_sample_without_blocking(
        self,
        sample: Sample,
        skip_preprocessing: bool = False,
        skip_postprocessing: bool = False,
        skip_input_padding: bool = False,
        skip_output_cropping: bool = False,
    ) -> Sample:
        """Predict a whole sample at once.

        Note:
            The sample's tensor shapes have to match the model's input tensor description.
            If that is not the case, consider `predict_sample_with_blocking`

        Args:
            sample: input sample
            skip_preprocessing: if `True`, skip all preprocessing steps (except for any preceding prediction pipeline).
            skip_postprocessing: if `True`, skip all postprocessing steps (except for any preceding prediction pipeline).
            skip_input_padding: if `True`, skip padding the input sample according to the model's (optional) output halos.
            skip_output_cropping: if `True`, skip cropping any output halos from the model output.
        """
        for pp in self._get_preceding_prediction_pipelines_for_sample(sample):
            sample = pp._predict_sample_without_blocking_impl(
                sample,
                skip_input_padding=skip_input_padding,
                skip_output_cropping=skip_output_cropping,
            )

        return self._predict_sample_without_blocking_impl(
            sample,
            skip_preprocessing=skip_preprocessing,
            skip_postprocessing=skip_postprocessing,
            skip_input_padding=skip_input_padding,
            skip_output_cropping=skip_output_cropping,
        )

    @abstractmethod
    def _predict_sample_without_blocking_impl(
        self,
        sample: Sample,
        skip_preprocessing: bool = False,
        skip_postprocessing: bool = False,
        skip_input_padding: bool = False,
        skip_output_cropping: bool = False,
    ) -> Sample:
        """Predict a whole sample at once.

        Note:
            The sample's tensor shapes have to match the model's input tensor description.
            If that is not the case, consider `predict_sample_with_blocking`

        Args:
            sample: input sample
            skip_preprocessing: if `True`, skip all preprocessing steps.
            skip_postprocessing: if `True`, skip all postprocessing steps.
            skip_input_padding: if `True`, skip padding the input sample according to the model's (optional) output halos.
            skip_output_cropping: if `True`, skip cropping any output halos from the model output.
        """

    def predict_sample_with_blocking(
        self,
        sample: Sample,
        skip_preprocessing: bool = False,
        skip_postprocessing: bool = False,
        ns: v0_5.ParameterizedSize_N
        | Mapping[tuple[MemberId, AxisId], v0_5.ParameterizedSize_N]
        | None = None,
        batch_size: int | None = None,
    ) -> Sample:
        """Predict a sample by predicting sample blocks.

        Note: For fixed/known blocksizes use `predict_sample_with_fixed_blocking`.

        Args:
            sample: The sample to predict on.
            skip_preprocessing: If `True`, skip all preprocessing steps.
            skip_postprocessing: If `True`, skip all postprocessing steps.
            ns: Block size parameter(s) allows scaling the model's default input block size.
              Blocksize parameters are only applied to parameterized input axes, all other axis sizes are fixed/derived or (for output axes) data dependent.
              Unapplicable blocksize parameters are ignored.
            batch_size: Batch size to use for prediction.
        """

        output = None
        for output in self.predict_sample_with_blocking_yield_intermediates(
            sample,
            skip_preprocessing=skip_preprocessing,
            skip_postprocessing=skip_postprocessing,
            ns=ns,
            batch_size=batch_size,
        )[1]:
            pass

        assert output is not None, (
            "No blocks were predicted, cannot return final sample."
        )
        return output.sample

    def predict_sample_with_fixed_blocking(
        self,
        sample: Sample,
        input_block_shape: PerMember[PerAxis[int]],
        skip_preprocessing: bool = False,
        skip_postprocessing: bool = False,
    ) -> Sample:
        """Predict `sample` with given `input_block_shape`.

        Note:
            - `input_block_shape` is expected to be a valid input shape for the model.
            - Use `predict_sample_with_blocking` if you want to control block sizes via generic block size parameters rather than fixed block shapes.

        Args:
            sample: The sample to predict on.
            input_block_shape: Mapping of input member id to mapping of axis id to block size for that axis.
            skip_preprocessing: If `True`, skip all preprocessing steps.
            skip_postprocessing: If `True`, skip all postprocessing steps.
        """
        intermediate = None
        for (
            intermediate
        ) in self._predict_sample_with_fixed_blocking_yield_intermediates_impl(
            sample,
            input_block_shape=input_block_shape,
            skip_preprocessing=skip_preprocessing,
            skip_postprocessing=skip_postprocessing,
        )[1]:
            pass

        assert intermediate is not None, (
            "No blocks were predicted, cannot return final sample."
        )
        return intermediate.sample

    def predict_sample_with_blocking_yield_intermediates(
        self,
        sample: Sample,
        skip_preprocessing: bool = False,
        skip_postprocessing: bool = False,
        ns: v0_5.ParameterizedSize_N
        | Mapping[tuple[MemberId, AxisId], v0_5.ParameterizedSize_N]
        | None = None,
        batch_size: int | None = None,
    ) -> tuple[int, Iterable[IntermediatePrediction]]:
        """Predict `sample` by predicting sample blocks and yield intermediate predictions if no samplewise postprocessing is included.
        Also yields intermediate predictions if there are preceding prediction pipelines (model inputs depend on another model's outputs).
        For preceding prediction pipelines `ns` and `batch_size` are shared, but pre- and postprocessing are never skipped in preceding pipelines.

        Returns:
            Tuple of number of prediction steps and an iterator of predicted intermediate samples with the last predicted block,
            All samples, but the last one, are intermediate samples with more and more blocks predicted.
            In case samplewise postprocessing needs to be applied, no intermediate results are yielded, but only the final sample after all blocks are predicted and postprocessed.
            In case of preceding prediction pipelines (model inputs depend on another model's outputs), intermediate results initially do not include the final output tensors at all.
        """

        total_prediction_steps = 0
        iterable_intermediates = ()
        for pp in self._get_preceding_prediction_pipelines_for_sample(sample):
            pp_steps, pp_intermediates = (
                pp.predict_sample_with_blocking_yield_intermediates(
                    sample,
                    ns=ns,
                    batch_size=batch_size,
                    skip_preprocessing=False,
                    skip_postprocessing=False,
                )
            )
            total_prediction_steps += pp_steps
            iterable_intermediates = chain(iterable_intermediates, pp_intermediates)

        if isinstance(self._model_descr, v0_4.ModelDescr):
            raise NotImplementedError(
                "`predict_sample_with_blocking` not implemented for v0_4.ModelDescr"
                + f" {self._model_descr.name}."
                + " Consider using `predict_sample_with_fixed_blocking`"
            )

        ns = ns or self._default_blocksize_parameter
        if isinstance(ns, int):
            ns = {
                (ipt.id, a.id): ns
                for ipt in self._model_descr.inputs
                for a in ipt.axes
                if isinstance(a.size, v0_5.ParameterizedSize)
            }
        input_block_shape = self._model_descr.get_tensor_sizes(
            ns, batch_size or self._default_batch_size, sample.shape
        ).inputs

        steps, intermediates = (
            self._predict_sample_with_fixed_blocking_yield_intermediates_impl(
                sample,
                input_block_shape=input_block_shape,
                skip_preprocessing=skip_preprocessing,
                skip_postprocessing=skip_postprocessing,
            )
        )
        total_prediction_steps += steps
        iterable_intermediates = chain(iterable_intermediates, intermediates)
        return total_prediction_steps, iterable_intermediates

    def predict_sample_with_fixed_blocking_yield_intermediates(
        self,
        sample: Sample,
        input_block_shape: PerMember[PerAxis[int]],
        *,
        skip_preprocessing: bool = False,
        skip_postprocessing: bool = False,
        fill_value: float = float("nan"),
    ) -> tuple[int, Iterable[IntermediatePrediction]]:
        """Predict `sample` by predicting sample blocks of `input_block_shape` and yield intermediate predictions if no samplewise postprocessing is included.
        Also yields intermediate predictions if there are preceding prediction pipelines (model inputs depend on another model's outputs).
        For preceding prediction pipelines `input_block_shape` and `fill_value` are shared, but pre- and postprocessing are never skipped in preceding pipelines.

        Returns:
            Tuple of number of prediction steps and an iterator of predicted intermediate samples with the last predicted block,
            All samples, but the last one, are intermediate samples with more and more blocks predicted.
            In case samplewise postprocessing needs to be applied, no intermediate results are yielded, but only the final sample after all blocks are predicted and postprocessed.
            In case of preceding prediction pipelines (model inputs depend on another model's outputs), intermediate results initially do not include the final output tensors at all.
        """
        total_prediction_steps = 0
        iterable_intermediates = ()
        for pp in self._get_preceding_prediction_pipelines_for_sample(sample):
            pp_steps, pp_intermediates = (
                pp._predict_sample_with_fixed_blocking_yield_intermediates_impl(
                    sample,
                    input_block_shape=input_block_shape,
                    skip_preprocessing=False,
                    skip_postprocessing=False,
                    fill_value=fill_value,
                )
            )
            total_prediction_steps += pp_steps
            iterable_intermediates = chain(iterable_intermediates, pp_intermediates)

        pp_steps, pp_intermediates = (
            self._predict_sample_with_fixed_blocking_yield_intermediates_impl(
                sample,
                input_block_shape=input_block_shape,
                skip_preprocessing=skip_preprocessing,
                skip_postprocessing=skip_postprocessing,
                fill_value=fill_value,
            )
        )
        total_prediction_steps += pp_steps
        iterable_intermediates = chain(iterable_intermediates, pp_intermediates)
        return total_prediction_steps, iterable_intermediates

    @abstractmethod
    def _predict_sample_with_fixed_blocking_yield_intermediates_impl(
        self,
        sample: Sample,
        input_block_shape: PerMember[PerAxis[int]],
        *,
        skip_preprocessing: bool = False,
        skip_postprocessing: bool = False,
        fill_value: float = float("nan"),
    ) -> tuple[int, Iterable[IntermediatePrediction]]: ...

    @abstractmethod
    def predict_sample_block(
        self,
        sample_block: SampleBlock,
        skip_preprocessing: bool = False,
        skip_postprocessing: bool = False,
    ) -> SampleBlock:
        """Predict a single sample block.

        Note that this does not apply samplewise preprocessing or postprocessing steps, but only blockwise ones.

        Args:
            sample_block: The sample block to predict on.
            skip_preprocessing: If `True`, skip blockwise preprocessing steps.
            skip_postprocessing: If `True`, skip blockwise postprocessing steps.
        """


class PredictionPipeline(_PredictionPipelineBase):
    """
    Represents model computation including preprocessing and postprocessing
    Note: Ideally use the `PredictionPipeline` in a with statement
        (as a context manager).
    """

    def __init__(
        self,
        *,
        name: str,
        model_description: AnyModelDescr,
        preprocessing: list[Processing],
        postprocessing: list[Processing],
        model_adapter: ModelAdapter,
        default_blocksize_parameter: BlocksizeParameter = 10,
        default_batch_size: int = 1,
        preceding_prediction_pipelines: Sequence[
            PredictionPipeline | RemotePredictionPipeline
        ]
        | None = None,
    ) -> None:
        """Consider using `create_prediction_pipeline` to create a `PredictionPipeline` with sensible defaults."""
        super().__init__(
            model_descr=model_description,
            default_blocksize_parameter=default_blocksize_parameter,
            default_batch_size=default_batch_size,
            preceding_prediction_pipelines=preceding_prediction_pipelines,
        )

        if model_description.run_mode:
            warnings.warn(
                f"Not yet implemented inference for run mode '{model_description.run_mode.name}'"
            )

        self.name = name
        # split preprocessing into samplewise and blockwise. samplewise preprocessing is all preprocessing up to including the last samplewise operator, blockwise preprocessing are the remaining blockwise operators.
        # I.e. some samplewise preprocessing may be a blockwise op (at some point followed by a samplewise op).
        self._samplewise_preprocessing: list[
            SamplewiseOperator | BlockwiseOperator
        ] = []
        self._blockwise_preprocessing: list[BlockwiseOperator] = []
        for op in preprocessing[::-1]:
            if isinstance(op, BlockwiseOperator) and not self._samplewise_preprocessing:
                self._blockwise_preprocessing.insert(0, op)
            else:
                self._samplewise_preprocessing.insert(0, op)
        # split postprocessing analougly, but here we start blockwise and switch to samplewise at the first samplewise operator.
        self._blockwise_postprocessing: list[BlockwiseOperator] = []
        self._samplewise_postprocessing: list[
            BlockwiseOperator | SamplewiseOperator
        ] = []
        for op in postprocessing:
            if (
                isinstance(op, BlockwiseOperator)
                and not self._samplewise_postprocessing
            ):
                self._blockwise_postprocessing.append(op)
            else:
                self._samplewise_postprocessing.append(op)

        self._adapter = model_adapter

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore
        self.unload()
        return False

    @property
    def has_non_blockwise_preprocessing(self) -> bool:
        """`True` if any preprocessing operators in the pipeline are not applicable blockwise."""
        return bool(self._samplewise_preprocessing)

    @property
    def has_non_blockwise_postprocessing(self) -> bool:
        """`True` if any postprocessing operators in the pipeline are not applicable blockwise."""
        return bool(self._samplewise_postprocessing)

    def _raise_for_non_blockwise_processing(
        self, proc_type: Literal["preprocessing", "postprocessing"]
    ):
        ops = (
            self._samplewise_preprocessing
            if proc_type == "preprocessing"
            else self._samplewise_postprocessing
        )
        non_blockwise = [
            op.__class__.__name__ for op in ops if not isinstance(op, BlockwiseOperator)
        ]
        if non_blockwise:
            raise NotImplementedError(
                f"Blockwise {proc_type} for {non_blockwise} not implemented."
            )

    def raise_for_non_blockwise_preprocessing(self):
        """
        Raises:
            NotImplementedError: if there are any non-blockwise preprocessing operators in the pipeline
        """
        self._raise_for_non_blockwise_processing("preprocessing")

    def raise_for_non_blockwise_postprocessing(self):
        """
        Raises:
            NotImplementedError: if there are any non-blockwise postprocessing operators in the pipeline
        """
        self._raise_for_non_blockwise_processing("postprocessing")

    def predict_sample_block(
        self,
        sample_block: SampleBlock,
        skip_preprocessing: bool = False,
        skip_postprocessing: bool = False,
    ) -> SampleBlock:
        if isinstance(self._model_descr, v0_4.ModelDescr):
            raise NotImplementedError(
                f"predict_sample_block not implemented for model {self._model_descr.format_version}"
            )
        else:
            assert self._block_transform is not None

        if not skip_preprocessing:
            self._apply_blockwise_preprocessing(sample_block)

        output_meta = sample_block.get_transformed_meta(self._block_transform)
        local_output = self._adapter.forward(sample_block.members)

        output = output_meta.with_data(
            {k: v for k, v in local_output.items() if v is not None},
            stat=sample_block.stat,
        )
        if not skip_postprocessing:
            self._apply_blockwise_postprocessing(output)

        return output

    def _predict_sample_without_blocking_impl(
        self,
        sample: Sample,
        skip_preprocessing: bool = False,
        skip_postprocessing: bool = False,
        skip_input_padding: bool = False,
        skip_output_cropping: bool = False,
    ) -> Sample:
        if not skip_input_padding:
            sample = sample.pad(pad_width=self._default_input_halo, mode=self.pad_mode)

        if not skip_preprocessing:
            self.apply_preprocessing(sample)

        output = Sample(
            members={
                k: v
                for k, v in self._adapter.forward(sample.members).items()
                if v is not None
            },
            stat=sample.stat,
            id=sample.id,
        )
        if not skip_postprocessing:
            self.apply_postprocessing(output)

        if not skip_output_cropping:
            output.members = {
                m: t
                if m not in self._default_output_halo
                else t[
                    {
                        a: slice(h.left, None if h.right == 0 else -h.right)
                        for a, h in self._default_output_halo[m].items()
                    }
                ]
                for m, t in output.members.items()
            }

        return output

    def get_output_sample_id(self, input_sample_id: SampleId):
        warnings.warn(
            "`PredictionPipeline.get_output_sample_id()` is deprecated and will be"
            + " removed soon. Output sample id is equal to input sample id, hence this"
            + " function is not needed."
        )
        return input_sample_id

    def predict_sample_with_blocking(
        self,
        sample: Sample,
        skip_preprocessing: bool = False,
        skip_postprocessing: bool = False,
        ns: v0_5.ParameterizedSize_N
        | Mapping[tuple[MemberId, AxisId], v0_5.ParameterizedSize_N]
        | None = None,
        batch_size: int | None = None,
    ) -> Sample:
        output = None
        for output in self.predict_sample_with_blocking_yield_intermediates(
            sample,
            skip_preprocessing=skip_preprocessing,
            skip_postprocessing=skip_postprocessing,
            ns=ns,
            batch_size=batch_size,
        )[1]:
            pass

        assert output is not None, (
            "No blocks were predicted, cannot return final sample."
        )
        return output.sample

    def _predict_sample_with_fixed_blocking_yield_intermediates_impl(
        self,
        sample: Sample,
        input_block_shape: Mapping[MemberId, Mapping[AxisId, int]],
        *,
        skip_preprocessing: bool = False,
        skip_postprocessing: bool = False,
        fill_value: float = float("nan"),
    ) -> tuple[int, Iterable[IntermediatePrediction]]:
        """Predict `sample` with given `input_block_shape` and yield the full sample with intermediate results.

        Note:
            - `input_block_shape` is expected to be a valid input shape for the model.
            - Use `predict_sample_with_blocking` if you want to control block sizes via generic block size parameters
              rather than fixed block shapes.
            - Postprocessing may only be complete for the final sample (if samplewise postprocessing steps are included
              in the pipeline), intermediate samples may have some (blockwise applicable) postprocessing steps applied.

        Args:
            sample: The sample to predict on.
            input_block_shape: Mapping of input member id to mapping of axis id to block size for that axis.
            skip_preprocessing: If `True`, skip all preprocessing steps.
            skip_postprocessing: If `True`, skip all postprocessing steps.

        Returns:
            Tuple of number of blocks and an iterable of predicted intermediate samples with the last predicted block,
            All samples, but the last one, are intermediate samples with more and more blocks predicted.
        """
        if not skip_preprocessing:
            self._apply_samplewise_preprocessing(sample)

        n_blocks, input_blocks = sample.split_into_blocks(
            input_block_shape,
            halo=self._default_input_halo,
            pad_mode=self.pad_mode,
        )
        logger.info(
            "split sample shape {} into {} blocks of {}.",
            {k: dict(v) for k, v in sample.shape.items()},
            n_blocks,
            {k: dict(v) for k, v in input_block_shape.items()},
        )

        def _predict_blocks():
            predicted_sample = None
            for i, b in enumerate(
                tqdm(
                    input_blocks,
                    desc=f"predict sample '{sample.id or ''}' with {self._model_descr.id or self._model_descr.name}",
                    unit="block",
                    unit_divisor=1,
                    total=n_blocks,
                )
            ):
                if not skip_preprocessing:
                    self._apply_blockwise_preprocessing(b)

                predicted_block = self.predict_sample_block(
                    b, skip_preprocessing=True, skip_postprocessing=True
                )

                if not skip_postprocessing:
                    self._apply_blockwise_postprocessing(predicted_block)

                if predicted_sample is None:
                    predicted_sample = Sample.from_blocks(
                        [predicted_block], fill_value=fill_value
                    )
                else:
                    predicted_sample.set_block(predicted_block)

                if not skip_postprocessing and i == n_blocks - 1:
                    self._apply_samplewise_postprocessing(predicted_sample)

                yield IntermediatePrediction(predicted_sample, predicted_block)

        return n_blocks, _predict_blocks()

    def _apply_samplewise_preprocessing(self, sample: Sample, /) -> None:
        """Apply preprocessing operators up to and including the last samplewise operator in-place.

        Note: This skips all blockwise preprocessing steps after the last samplewise operator.
        """
        if isinstance(sample, SampleBlock):
            self.raise_for_non_blockwise_preprocessing()

        for op in self._samplewise_preprocessing:
            op(sample)

    def _apply_blockwise_preprocessing(
        self, sample_block: Sample | SampleBlock, /
    ) -> None:
        """Apply blockwise preprocessing operators in-place.

        Note: This skips all preprocessing operators up to and including the last samplewise one.
        """
        for op in self._blockwise_preprocessing:
            op(sample_block)

    def apply_preprocessing(self, sample: Sample | SampleBlock) -> None:
        """Apply preprocessing in-place, also may updates sample stats"""

        if isinstance(sample, Sample):
            self._apply_samplewise_preprocessing(sample)
        else:
            self.raise_for_non_blockwise_preprocessing()

        self._apply_blockwise_preprocessing(sample)

    def _apply_blockwise_postprocessing(
        self, sample_block: Sample | SampleBlock, /
    ) -> None:
        """Apply in-place blockwise postprocessing operators

        Note: This does not apply all postprocessing operators from the first samplewise one onwards.
        """
        for op in self._blockwise_postprocessing:
            op(sample_block)

    def _apply_samplewise_postprocessing(self, sample: Sample, /) -> None:
        """Apply in-place postprocessing operators starting from and including the first samplewise operator.

        Note: This skips all blockwise postprocessing steps before the first samplewise one.
        """
        if isinstance(sample, SampleBlock):
            self.raise_for_non_blockwise_postprocessing()

        for op in self._samplewise_postprocessing:
            op(sample)

    def apply_postprocessing(self, sample: Sample | SampleBlock) -> None:
        """apply postprocessing in-place, also may updates samples stats"""
        self._apply_blockwise_postprocessing(sample)
        if isinstance(sample, Sample):
            self._apply_samplewise_postprocessing(sample)
        else:
            self.raise_for_non_blockwise_postprocessing()

    def load(self):
        """Prepare prediction pipeline for use.

        Reusable model adapters may be loaded and unloaded multiple times, but currently not all model adapters
        cleanly unload and reload.

        Note:
            For some model adapters loading is currently part of the constructor making them unusable after unloading.
        """
        self._adapter.load()

    def unload(self):
        """Free any device memory in use.

        Note:
            Currently prediction pipeline becomes unusable after unloading."""
        self._adapter.unload()

    def close(self):
        """Permanently close the prediction pipeline and free any device memory in use.
        This makes the prediction pipeline unusable afterwards."""
        self.unload()


class RemotePredictionPipeline(_PredictionPipelineBase):
    """Abstract base class for fully remote prediction pipelines.

    Note: A ("local") `PredictionPipeline` may also use a `RemoteModelAdapter` for remote model inference, but it may
        still apply local preprocessing and postprocessing steps.
        In contrast, a `RemotePredictionPipeline` is designed for the case where all steps including preprocessing and
        postprocessing are performed remotely.
    """

    def __init__(
        self,
        model_descr: AnyModelDescr,
        *,
        server: str,
        default_blocksize_parameter: BlocksizeParameter,
        default_batch_size: int,
        preceding_prediction_pipelines: Sequence[
            PredictionPipeline | RemotePredictionPipeline
        ]
        | None = None,
    ) -> None:
        super().__init__(
            model_descr,
            default_blocksize_parameter=default_blocksize_parameter,
            default_batch_size=default_batch_size,
            preceding_prediction_pipelines=preceding_prediction_pipelines,
        )
        self._server = server

    @property
    def server(self) -> str:
        return self._server


def create_prediction_pipeline(
    bioimageio_model: AnyModelDescr,
    *,
    devices: Sequence[str] | None = None,
    weight_format: SupportedWeightsFormat | None = None,
    weights_format: SupportedWeightsFormat | None = None,
    dataset_for_initial_statistics: Iterable[Sample | Sequence[Tensor]] = (),
    keep_updating_initial_dataset_statistics: bool = False,
    fixed_dataset_statistics: Mapping[Measure, MeasureValue] = MappingProxyType({}),
    model_adapter: ModelAdapter | None = None,
    ns: BlocksizeParameter | None = None,
    default_blocksize_parameter: BlocksizeParameter = 10,  # TODO: default to None and find smart blocksize params per axis to reduce overlap of blocks with large halo
    preceding_prediction_pipelines: Sequence[
        PredictionPipeline | RemotePredictionPipeline
    ]
    | None = None,
    **deprecated_kwargs: Any,
) -> PredictionPipeline:
    """
    Creates prediction pipeline which includes:
    * computation of input statistics
    * preprocessing
    * model prediction
    * computation of output statistics
    * postprocessing

    Args:
        bioimageio_model: A bioimageio model description.
        devices: (optional)
        weight_format: deprecated in favor of **weights_format**
        weights_format: (optional) Use a specific **weights_format** rather than
            choosing one automatically.
            A corresponding `bioimageio.core.model_adapters.ModelAdapter` will be
            created to run inference with the **bioimageio_model**.
        dataset_for_initial_statistics: (optional) If preprocessing steps require input
            dataset statistics, **dataset_for_initial_statistics** allows you to
            specifcy a dataset from which these statistics are computed.
        keep_updating_initial_dataset_statistics: (optional) Set to `True` if you want
            to update dataset statistics with each processed sample.
        fixed_dataset_statistics: (optional) Precomputed dataset (and optionally sample) statistics.
            Any included sample statistics will not be calculated on the fly and it is the callers
            responsibility to use samples with the corresponding statistics availble in `sample.stat`.
        model_adapter: (optional) Allows you to use a custom **model_adapter** instead
            of creating one according to the present/selected **weights_format**.
        ns: deprecated in favor of **default_blocksize_parameter**
        default_blocksize_parameter: Allows to control the default block size for
            blockwise predictions, see `BlocksizeParameter`.
        preceding_prediction_pipelines: (optional) If the model has inputs that are
            outputs of other models (input field 'output_of'), you can provide a sequence
            of preceding prediction pipelines. The prediction pipeline will then automatically
            use the outputs of those preceding pipelines as inputs for the current model.
            If no preceding prediction pipelines for a model are provided, prediction pipelines using the
            same devices and weight format as for the current model will be created for any required preceding models.
    """
    weights_format = weight_format or weights_format
    del weight_format
    default_blocksize_parameter = ns or default_blocksize_parameter
    del ns
    if deprecated_kwargs:
        warnings.warn(
            f"deprecated create_prediction_pipeline kwargs: {set(deprecated_kwargs)}"
        )

    model_adapter = model_adapter or create_model_adapter(
        model_description=bioimageio_model,
        devices=devices,
        weight_format_priority_order=weights_format and (weights_format,),
    )

    input_ids = get_member_ids(bioimageio_model.inputs)

    def dataset():
        common_stat: Stat = {}
        for i, x in enumerate(dataset_for_initial_statistics):
            if isinstance(x, Sample):
                yield x
            else:
                yield Sample(members=dict(zip(input_ids, x)), stat=common_stat, id=i)

    preprocessing, postprocessing = setup_pre_and_postprocessing(
        bioimageio_model,
        dataset(),
        keep_updating_initial_dataset_stats=keep_updating_initial_dataset_statistics,
        fixed_dataset_stats=fixed_dataset_statistics,
    )

    def _get_preceding_model_ids(model: AnyModelDescr) -> set[v0_5.ModelId]:
        return {
            input_descr.output_of
            for input_descr in model.inputs
            if isinstance(input_descr, v0_5.InputTensorDescr)
            and input_descr.output_of is not None
        }

    preceding_model_ids = _get_preceding_model_ids(bioimageio_model)
    if preceding_prediction_pipelines is None:
        preceding_prediction_pipelines = []
    else:
        preceding_prediction_pipelines = list(preceding_prediction_pipelines)

    for preceding_model_id in preceding_model_ids:
        if preceding_model_id in {
            pp.model_description.id for pp in preceding_prediction_pipelines
        }:
            continue

        preceding_model = load_model_description(preceding_model_id)
        preceding_prediction_pipelines.insert(
            0,
            create_prediction_pipeline(
                preceding_model,
                devices=devices,
                weights_format=weights_format,
                default_blocksize_parameter=default_blocksize_parameter,
                dataset_for_initial_statistics=dataset_for_initial_statistics,
                keep_updating_initial_dataset_statistics=keep_updating_initial_dataset_statistics,
                fixed_dataset_statistics=fixed_dataset_statistics,
            ),
        )

    pp = PredictionPipeline(
        name=bioimageio_model.name,
        model_description=bioimageio_model,
        model_adapter=model_adapter,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
        default_blocksize_parameter=default_blocksize_parameter,
        preceding_prediction_pipelines=preceding_prediction_pipelines,
    )
    logger.info(
        "Created prediction pipeline for '{}' with {} adapter",
        bioimageio_model.name,
        model_adapter.__class__.__name__,
    )
    return pp


def create_remote_prediction_pipeline(
    model_description: AnyModelDescr,
    *,
    server: str | None = None,
    server_type: Literal["gradio"] | None = "gradio",
    precomputed_statistics: Mapping[Measure, MeasureValue] = MappingProxyType({}),
    default_blocksize_parameter: BlocksizeParameter = 10,  # TODO: default to None and find smart blocksize params per axis to reduce overlap of blocks with large halo
    default_batch_size: int = 1,
) -> RemotePredictionPipeline:
    """Create a `RemotePredictionPipeline` for the given `model_description`.

    Args:
        model_description: The model to run inference with.
        server: The URL or Hugging Face space name of a running bioimageio server instance
        server_type: The type of the remote server to connect to. Currently only "gradio" is supported.
        precomputed_statistics: Precomputed dataset (and optionally sample) statistics.
            Any included sample statistics will not be calculated on the fly and it is the callers
            responsibility to use samples with the corresponding statistics availble in `sample.stat`.
        default_blocksize_parameter: Allows to control the default block size with a single parameter for blockwise predictions. (not all models support this)
        default_batch_size: Default batch size to use
    """

    if server_type is None:
        server_type = "gradio"

    try:
        if server_type == "gradio":
            from .remote_backends.gradio.client import (
                GradioPredictionPipeline as RemotePredictionPipelineImpl,
            )
        else:
            assert_never(server_type)
    except ImportError as e:
        raise ImportError(
            f"Failed to import {server_type.capitalize()}PredictionPipeline. Make sure to install the '{server_type}-client' extra,"
            + f" e.g. with `pip install bioimageio.core[{server_type}-client]`."
        ) from e

    return RemotePredictionPipelineImpl(
        model_description,
        server=server,
        precomputed_statistics=precomputed_statistics,
        default_blocksize_parameter=default_blocksize_parameter,
        default_batch_size=default_batch_size,
    )
