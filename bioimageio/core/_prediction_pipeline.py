import warnings
from types import MappingProxyType
from typing import (
    Any,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from tqdm import tqdm

from bioimageio.spec.model import AnyModelDescr, v0_4, v0_5
from bioimageio.spec.model.v0_5 import WeightsFormat

from ._op_base import BlockedOperator
from .axis import AxisId, PerAxis
from .common import Halo, MemberId, PerMember, SampleId
from .digest_spec import (
    get_block_transform,
    get_input_halo,
    get_member_ids,
)
from .model_adapters import ModelAdapter, create_model_adapter
from .model_adapters import get_weight_formats as get_weight_formats
from .proc_ops import Processing
from .proc_setup import setup_pre_and_postprocessing
from .sample import Sample, SampleBlock, SampleBlockWithOrigin
from .stat_measures import DatasetMeasure, MeasureValue, Stat
from .tensor import Tensor

Predict_IO = TypeVar(
    "Predict_IO",
    Sample,
    Iterable[Sample],
)


class PredictionPipeline:
    """
    Represents model computation including preprocessing and postprocessing
    Note: Ideally use the PredictionPipeline as a context manager
    """

    def __init__(
        self,
        *,
        name: str,
        model_description: AnyModelDescr,
        preprocessing: List[Processing],
        postprocessing: List[Processing],
        model_adapter: ModelAdapter,
        default_ns: Union[
            v0_5.ParameterizedSize.N,
            Mapping[Tuple[MemberId, AxisId], v0_5.ParameterizedSize.N],
        ] = 10,
        default_batch_size: int = 1,
    ) -> None:
        super().__init__()
        if model_description.run_mode:
            warnings.warn(
                f"Not yet implemented inference for run mode '{model_description.run_mode.name}'"
            )

        self.name = name
        self._preprocessing = preprocessing
        self._postprocessing = postprocessing

        self.model_description = model_description
        if isinstance(model_description, v0_4.ModelDescr):
            self._default_input_halo: PerMember[PerAxis[Halo]] = {}
            self._block_transform = None
        else:
            default_output_halo = {
                t.id: {
                    a.id: Halo(a.halo, a.halo)
                    for a in t.axes
                    if isinstance(a, v0_5.WithHalo)
                }
                for t in model_description.outputs
            }
            self._default_input_halo = get_input_halo(
                model_description, default_output_halo
            )
            self._block_transform = get_block_transform(model_description)

        self._default_ns = default_ns
        self._default_batch_size = default_batch_size

        self._input_ids = get_member_ids(model_description.inputs)
        self._output_ids = get_member_ids(model_description.outputs)

        self._adapter: ModelAdapter = model_adapter

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore
        self.unload()
        return False

    def predict_sample_block(
        self,
        sample_block: SampleBlockWithOrigin,
        skip_preprocessing: bool = False,
        skip_postprocessing: bool = False,
    ) -> SampleBlock:
        if isinstance(self.model_description, v0_4.ModelDescr):
            raise NotImplementedError(
                f"predict_sample_block not implemented for model {self.model_description.format_version}"
            )
        else:
            assert self._block_transform is not None

        if not skip_preprocessing:
            self.apply_preprocessing(sample_block)

        output_meta = sample_block.get_transformed_meta(self._block_transform)
        output = output_meta.with_data(
            {
                tid: out
                for tid, out in zip(
                    self._output_ids,
                    self._adapter.forward(
                        *(sample_block.members.get(t) for t in self._input_ids)
                    ),
                )
                if out is not None
            },
            stat=sample_block.stat,
        )
        if not skip_postprocessing:
            self.apply_postprocessing(output)

        return output

    def predict_sample_without_blocking(
        self,
        sample: Sample,
        skip_preprocessing: bool = False,
        skip_postprocessing: bool = False,
    ) -> Sample:
        """predict a sample.
        The sample's tensor shapes have to match the model's input tensor description.
        If that is not the case, consider `predict_sample_with_blocking`"""

        if not skip_preprocessing:
            self.apply_preprocessing(sample)

        output = Sample(
            members={
                out_id: out
                for out_id, out in zip(
                    self._output_ids,
                    self._adapter.forward(
                        *(sample.members.get(in_id) for in_id in self._input_ids)
                    ),
                )
                if out is not None
            },
            stat=sample.stat,
            id=self.get_output_sample_id(sample.id),
        )
        if not skip_postprocessing:
            self.apply_postprocessing(output)

        return output

    def get_output_sample_id(self, input_sample_id: SampleId):
        if input_sample_id is None:
            return None
        else:
            return f"{input_sample_id}_" + (
                self.model_description.id or self.model_description.name
            )

    def predict_sample_with_blocking(
        self,
        sample: Sample,
        skip_preprocessing: bool = False,
        skip_postprocessing: bool = False,
        ns: Optional[
            Union[
                v0_5.ParameterizedSize.N,
                Mapping[Tuple[MemberId, AxisId], v0_5.ParameterizedSize.N],
            ]
        ] = None,
        batch_size: Optional[int] = None,
    ) -> Sample:
        """predict a sample by splitting it into blocks according to the model and the `ns` parameter"""
        if not skip_preprocessing:
            self.apply_preprocessing(sample)

        if isinstance(self.model_description, v0_4.ModelDescr):
            raise NotImplementedError(
                "predict with blocking not implemented for v0_4.ModelDescr {self.model_description.name}"
            )

        ns = ns or self._default_ns
        if isinstance(ns, int):
            ns = {
                (ipt.id, a.id): ns
                for ipt in self.model_description.inputs
                for a in ipt.axes
                if isinstance(a.size, v0_5.ParameterizedSize)
            }
        input_block_shape = self.model_description.get_tensor_sizes(
            ns, batch_size or self._default_batch_size
        ).inputs

        n_blocks, input_blocks = sample.split_into_blocks(
            input_block_shape,
            halo=self._default_input_halo,
            pad_mode="reflect",
        )
        input_blocks = list(input_blocks)
        predicted_blocks: List[SampleBlock] = []
        for b in tqdm(
            input_blocks,
            desc=f"predict sample {sample.id or ''} with {self.model_description.id or self.model_description.name}",
            unit="block",
            unit_divisor=1,
            total=n_blocks,
        ):
            predicted_blocks.append(
                self.predict_sample_block(
                    b, skip_preprocessing=True, skip_postprocessing=True
                )
            )

        predicted_sample = Sample.from_blocks(predicted_blocks)
        if not skip_postprocessing:
            self.apply_postprocessing(predicted_sample)

        return predicted_sample

    # def predict(
    #     self,
    #     inputs: Predict_IO,
    #     skip_preprocessing: bool = False,
    #     skip_postprocessing: bool = False,
    # ) -> Predict_IO:
    #     """Run model prediction **including** pre/postprocessing."""

    #     if isinstance(inputs, Sample):
    #         return self.predict_sample_with_blocking(
    #             inputs,
    #             skip_preprocessing=skip_preprocessing,
    #             skip_postprocessing=skip_postprocessing,
    #         )
    #     elif isinstance(inputs, collections.abc.Iterable):
    #         return (
    #             self.predict(
    #                 ipt,
    #                 skip_preprocessing=skip_preprocessing,
    #                 skip_postprocessing=skip_postprocessing,
    #             )
    #             for ipt in inputs
    #         )
    #     else:
    #         assert_never(inputs)

    def apply_preprocessing(self, sample: Union[Sample, SampleBlockWithOrigin]) -> None:
        """apply preprocessing in-place, also updates sample stats"""
        for op in self._preprocessing:
            op(sample)

    def apply_postprocessing(
        self, sample: Union[Sample, SampleBlock, SampleBlockWithOrigin]
    ) -> None:
        """apply postprocessing in-place, also updates samples stats"""
        for op in self._postprocessing:
            if isinstance(sample, (Sample, SampleBlockWithOrigin)):
                op(sample)
            elif not isinstance(op, BlockedOperator):
                raise NotImplementedError(
                    "block wise update of output statistics not yet implemented"
                )
            else:
                op(sample)

    def load(self):
        """
        optional step: load model onto devices before calling forward if not using it as context manager
        """
        pass

    def unload(self):
        """
        free any device memory in use
        """
        self._adapter.unload()


def create_prediction_pipeline(
    bioimageio_model: AnyModelDescr,
    *,
    devices: Optional[Sequence[str]] = None,
    weight_format: Optional[WeightsFormat] = None,
    weights_format: Optional[WeightsFormat] = None,
    dataset_for_initial_statistics: Iterable[Union[Sample, Sequence[Tensor]]] = tuple(),
    keep_updating_initial_dataset_statistics: bool = False,
    fixed_dataset_statistics: Mapping[DatasetMeasure, MeasureValue] = MappingProxyType(
        {}
    ),
    model_adapter: Optional[ModelAdapter] = None,
    ns: Union[
        v0_5.ParameterizedSize.N,
        Mapping[Tuple[MemberId, AxisId], v0_5.ParameterizedSize.N],
    ] = 10,
    **deprecated_kwargs: Any,
) -> PredictionPipeline:
    """
    Creates prediction pipeline which includes:
    * computation of input statistics
    * preprocessing
    * model prediction
    * computation of output statistics
    * postprocessing
    """
    weights_format = weight_format or weights_format
    del weight_format
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

    return PredictionPipeline(
        name=bioimageio_model.name,
        model_description=bioimageio_model,
        model_adapter=model_adapter,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
        default_ns=ns,
    )
