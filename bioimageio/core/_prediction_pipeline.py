import collections.abc
import warnings
from dataclasses import dataclass
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

from .axis import AxisId, AxisInfo
from .block import Block
from .common import MemberId, PadMode, PerMember
from .digest_spec import get_axes_infos, get_block_meta
from .model_adapters import ModelAdapter, create_model_adapter
from .model_adapters import get_weight_formats as get_weight_formats
from .proc_ops import Processing
from .proc_setup import setup_pre_and_postprocessing
from .sample import Sample, SampleBlock
from .stat_measures import DatasetMeasure, MeasureValue
from .tensor import Tensor


@dataclass
class MemberDescr:
    id: MemberId
    axes: Sequence[AxisInfo]
    optional: bool


Predict_IO = TypeVar(
    "Predict_IO",
    Sample,
    SampleBlock,
    Iterable[Sample],
    Iterable[SampleBlock],
)

# NDArray[Any],
# Sequence[Optional[NDArray[Any]]],
# Mapping[Union[MemberId, str], Optional[NDArray[Any]]],


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
        ns: Union[
            v0_5.ParameterizedSize.N,
            Mapping[Tuple[MemberId, AxisId], v0_5.ParameterizedSize.N],
        ],
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
        if isinstance(ns, int):
            if isinstance(model_description, v0_4.ModelDescr):
                self.ns = None
            else:
                self.ns = {
                    (ipt.id, a.id): ns
                    for ipt in model_description.inputs
                    for a in ipt.axes
                    if isinstance(a.size, v0_5.ParameterizedSize)
                }
        else:
            self.ns = ns
        # if isinstance(model_description, v0_4.ModelDescr):
        #     self.default_sample_block_shape = None
        # else:

        #     self.default_sample_block_shape = model_description.get_tensor_sizes(
        #         ns, 1
        #     ).inputs

        self.input_ids = tuple(
            (MemberId(str(t.name)) if isinstance(t, v0_4.InputTensorDescr) else t.id)
            for t in model_description.inputs
        )
        self.inputs = collections.OrderedDict(
            (
                tid,
                MemberDescr(
                    id=tid,
                    axes=get_axes_infos(t),
                    optional=not isinstance(t, v0_4.InputTensorDescr) and t.optional,
                ),
            )
            for tid, t in zip(self.input_ids, model_description.inputs)
        )
        self.output_ids = tuple(
            (MemberId(str(t.name)) if isinstance(t, v0_4.OutputTensorDescr) else t.id)
            for t in model_description.outputs
        )
        self.outputs = collections.OrderedDict(
            (
                tid,
                MemberDescr(
                    id=tid,
                    axes=get_axes_infos(t),
                    optional=False,
                ),
            )
            for tid, t in zip(self.output_ids, model_description.outputs)
        )

        self._adapter: ModelAdapter = model_adapter

    def __call__(self, data: Predict_IO) -> Predict_IO:
        return self.predict(data)

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore
        self.unload()
        return False

    # def predict_sample(
    #     self,
    #     sample: Sample,
    #     parameterized_size_n: Optional[int] = None,
    #     pad_mode: PadMode = "reflect",
    # ) -> Sample:
    #     if parameterized_size_n is None:
    #         # TODO guess n
    #         parameterized_size_n = 10

    #     return Sample.from_blocks(
    #         map(
    #             self.predict_sample_block,
    #             sample.split_into_blocks(
    #                 block_shapes={m: ipt.axes for m, ipt in self.inputs.items()},
    #                 halo={
    #                     m: ipt.axes.halo
    #                     for m, ipt in self.inputs.items()
    #                     if isinstance(ipt.axes, v0_5.WithHalo)
    #                 },
    #                 pad_mode=pad_mode,
    #             ),
    #         )
    #     )

    # def predict_sample_block(self, inputs: SampleBlock) -> SampleBlock:
    #     self.apply_preprocessing(inputs)
    #     output = Block(
    #         data={
    #             tid: out
    #             for tid, out in zip(
    #                 self.output_ids,
    #                 self._adapter.forward(
    #                     *(inputs.data[t] for t in self.input_ids)
    #                 ),
    #             )
    #             if out is not None
    #         }
    #     )
    #     self.apply_postprocessing(output)
    #     return output

    #     else:
    #         assert_never(inputs)

    #     return output

    def predict(self, inputs: Predict_IO) -> Predict_IO:
        """Run model prediction **including** pre/postprocessing."""

        if isinstance(inputs, Sample):
            if isinstance(self.model_description, v0_4.ModelDescr):
                raise NotImplementedError(
                    "predicting `Sample`s no implemented for model"
                    + f" {self.model_description.format_version}."
                    + " Please divide the sample into block. using `sample.split_into_blocks()`."
                )

            assert self.ns is not None
            n_blocks, block_metas = get_block_meta(
                self.model_description, input_sample_shape=inputs.shape, ns=self.ns
            )

            # for block_meta in tqdm(block_metas, desc=f"predict sample {inputs.id or ''} with {self.model_description.id or self.model_description.name}", unit="block", total=n_blocks):
            input_halo =
            Sample.from_blocks(inputs.split_into_blocks())
            # return Sample.from_blocks(
            #     map(
            #         self.predict,
            n_blocks, blocks = inputs.split_into_blocks(
                block_shapes=self.default_sample_block_shape,
                halo={
                    m: ipt.axes.halo
                    for m, ipt in self.inputs.items()
                    if isinstance(ipt.axes, v0_5.WithHalo)
                },
                pad_mode="reflect",
            )
        #     )
        # )
        else:
            return self.predict_sample_block(inputs)

        # if isinstance(inputs, collections.abc.Mapping):
        #     data = {
        #         tid: d
        #         for tid in self.input_ids
        #         if (d := inputs.get(tid, inputs.get(str(tid)))) is not None
        #     }
        # else:
        #     if isinstance(inputs, (Tensor, np.ndarray)):
        #         inputs_seq = [inputs]
        #     else:
        #         inputs_seq = inputs

        #     assert len(inputs_seq) == len(self.input_ids)
        #     data = {
        #         tid: d for tid, d in zip(self.input_ids, inputs_seq) if d is not None
        #     }

        # sample = Sample(
        #     data={
        #         tid: Tensor.from_numpy(d, dims=self.inputs[tid].axes)
        #         for tid, d in data.items()
        #     }
        # )
        # output = self.predict(sample)
        # return {tid: out.data.data for }

    def apply_preprocessing(self, sample_block: SampleBlock) -> None:
        """apply preprocessing in-place, also updates sample stats"""
        for op in self._preprocessing:
            op(sample_block)

    def apply_postprocessing(self, sample_block: SampleBlock) -> None:
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
        Mapping[Tuple[TensorId, AxisId], v0_5.ParameterizedSize.N],
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

    if isinstance(bioimageio_model, v0_4.ModelDescr):
        input_ids = [MemberId(str(ipt.name)) for ipt in bioimageio_model.inputs]
    else:
        input_ids = [ipt.id for ipt in bioimageio_model.inputs]

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
        ns=ns,
    )
