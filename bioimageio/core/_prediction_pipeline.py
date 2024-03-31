import collections
import warnings
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Union

from bioimageio.core.axis import AxisInfo
from bioimageio.spec.model import AnyModelDescr, v0_4
from bioimageio.spec.model.v0_5 import WeightsFormat

from .model_adapters import ModelAdapter, create_model_adapter
from .model_adapters import get_weight_formats as get_weight_formats
from .proc_ops import Processing
from .proc_setup import setup_pre_and_postprocessing
from .sample import Sample
from .stat_measures import DatasetMeasure, MeasureValue
from .tensor import Tensor, TensorId
from .utils import get_axes_infos


@dataclass
class CoreTensorDescr:
    id: TensorId
    axes: Sequence[AxisInfo]
    optional: bool


class PredictionPipeline:
    """
    Represents model computation including preprocessing and postprocessing
    Note: Ideally use the PredictionPipeline as a context manager
    """

    def __init__(
        self,
        *,
        name: str,
        bioimageio_model: AnyModelDescr,
        preprocessing: List[Processing],
        postprocessing: List[Processing],
        model: ModelAdapter,
    ) -> None:
        super().__init__()
        if bioimageio_model.run_mode:
            warnings.warn(
                f"Not yet implemented inference for run mode '{bioimageio_model.run_mode.name}'"
            )

        self.name = name
        self._preprocessing = preprocessing
        self._postprocessing = postprocessing

        self.input_ids = tuple(
            (TensorId(str(t.name)) if isinstance(t, v0_4.InputTensorDescr) else t.id)
            for t in bioimageio_model.inputs
        )
        self.inputs = collections.OrderedDict(
            (
                tid,
                CoreTensorDescr(
                    id=tid,
                    axes=get_axes_infos(t),
                    optional=not isinstance(t, v0_4.InputTensorDescr) and t.optional,
                ),
            )
            for tid, t in zip(self.input_ids, bioimageio_model.inputs)
        )
        self.output_ids = tuple(
            (TensorId(str(t.name)) if isinstance(t, v0_4.OutputTensorDescr) else t.id)
            for t in bioimageio_model.outputs
        )
        self.outputs = collections.OrderedDict(
            (
                tid,
                CoreTensorDescr(
                    id=tid,
                    axes=get_axes_infos(t),
                    optional=False,
                ),
            )
            for tid, t in zip(self.output_ids, bioimageio_model.outputs)
        )

        self._adapter: ModelAdapter = model

    def __call__(self, sample: Sample) -> Sample:
        return self.predict(sample)

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore
        self.unload()
        return False

    def predict(self, sample: Sample) -> Sample:
        """Run model prediction **including** pre/postprocessing."""
        self.apply_preprocessing(sample)
        output = Sample(
            data={
                tid: out
                for tid, out in zip(
                    self.output_ids,
                    self._adapter.forward(*(sample.data[t] for t in self.input_ids)),
                )
                if out is not None
            }
        )
        self.apply_postprocessing(output)
        return output

    def apply_preprocessing(self, sample: Sample) -> None:
        """apply preprocessing in-place, also updates sample stats"""
        for op in self._preprocessing:
            op(sample)

    def apply_postprocessing(self, sample: Sample) -> None:
        """apply postprocessing in-place, also updates samples stats"""
        for op in self._postprocessing:
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
        input_ids = [TensorId(str(ipt.name)) for ipt in bioimageio_model.inputs]
    else:
        input_ids = [ipt.id for ipt in bioimageio_model.inputs]

    def dataset():
        for x in dataset_for_initial_statistics:
            if isinstance(x, Sample):
                yield x
            else:
                yield Sample(data=dict(zip(input_ids, x)))

    preprocessing, postprocessing = setup_pre_and_postprocessing(
        bioimageio_model,
        dataset(),
        keep_updating_initial_dataset_stats=keep_updating_initial_dataset_statistics,
        fixed_dataset_stats=fixed_dataset_statistics,
    )

    return PredictionPipeline(
        name=bioimageio_model.name,
        bioimageio_model=bioimageio_model,
        model=model_adapter,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
    )
