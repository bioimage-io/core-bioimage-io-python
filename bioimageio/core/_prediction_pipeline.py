import collections.abc
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
from typing_extensions import assert_never

from bioimageio.spec.model import AnyModelDescr, v0_4, v0_5
from bioimageio.spec.model.v0_5 import WeightsFormat

from .axis import AxisId, PerAxis
from .common import Halo, MemberId, PerMember
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
from .stat_measures import DatasetMeasure, MeasureValue
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
            self._default_input_block_shape = {}
            default_ns = {}
            self._default_input_halo: PerMember[PerAxis[Halo]] = {}
            self._block_transform = {}
        else:
            if isinstance(default_ns, int):
                default_ns = {
                    (ipt.id, a.id): default_ns
                    for ipt in model_description.inputs
                    for a in ipt.axes
                    if isinstance(a.size, v0_5.ParameterizedSize)
                }

            self._default_input_block_shape = model_description.get_tensor_sizes(
                default_ns, default_batch_size
            ).inputs

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

        self._input_ids = get_member_ids(model_description.inputs)
        self._output_ids = get_member_ids(model_description.outputs)

        self._adapter: ModelAdapter = model_adapter

    def __call__(self, data: Predict_IO) -> Predict_IO:
        return self.predict(data)

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore
        self.unload()
        return False

    def _predict_sample_block_wo_procs(
        self, sample_block: SampleBlockWithOrigin
    ) -> SampleBlock:
        output_meta = sample_block.get_transformed_meta(self._block_transform)
        output = output_meta.with_data(
            {
                tid: out
                for tid, out in zip(
                    self._output_ids,
                    self._adapter.forward(
                        *(sample_block.members[t] for t in self._input_ids)
                    ),
                )
                if out is not None
            },
            stat=sample_block.stat,
        )
        return output

    def predict_sample(self, sample: Sample) -> Sample:
        self.apply_preprocessing(sample)
        n_blocks, input_blocks = sample.split_into_blocks(
            self._default_input_block_shape,
            halo=self._default_input_halo,
            pad_mode="reflect",
        )
        input_blocks = tqdm(
            input_blocks,
            desc=f"predict sample {sample.id or ''} with {self.model_description.id or self.model_description.name}",
            unit="block",
            total=n_blocks,
        )
        predicted_blocks = map(self._predict_sample_block_wo_procs, input_blocks)
        predicted_sample = Sample.from_blocks(predicted_blocks)
        self.apply_postprocessing(predicted_sample)
        return predicted_sample

    def predict(
        self,
        inputs: Predict_IO,
    ) -> Predict_IO:
        """Run model prediction **including** pre/postprocessing."""

        if isinstance(inputs, Sample):
            return self.predict_sample(inputs)
        elif isinstance(inputs, collections.abc.Iterable):
            return (self.predict(ipt) for ipt in inputs)
        else:
            assert_never(inputs)

    def apply_preprocessing(
        self, sample_block: Union[Sample, SampleBlockWithOrigin]
    ) -> None:
        """apply preprocessing in-place, also updates sample stats"""
        for op in self._preprocessing:
            op(sample_block)

    def apply_postprocessing(
        self, sample_block: Union[Sample, SampleBlockWithOrigin]
    ) -> None:
        """apply postprocessing in-place, also updates samples stats"""
        for op in self._postprocessing:
            op(sample_block)

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
        for x in dataset_for_initial_statistics:
            if isinstance(x, Sample):
                yield x
            else:
                yield Sample(members=dict(zip(input_ids, x)))

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
