import warnings
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import xarray as xr

from bioimageio.core.common import Sample, TensorId
from bioimageio.core.model_adapters import ModelAdapter, create_model_adapter
from bioimageio.core.model_adapters import get_weight_formats as get_weight_formats
from bioimageio.core.proc_ops import Processing
from bioimageio.core.proc_setup import setup_pre_and_postprocessing
from bioimageio.core.stat_measures import DatasetMeasure, MeasureValue
from bioimageio.spec.model import AnyModelDescr, v0_4
from bioimageio.spec.model.v0_5 import WeightsFormat


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
                "Not yet implemented inference for run mode " +
                f"'{bioimageio_model.run_mode.name}'"
            )

        self.name = name
        self.input_specs = bioimageio_model.inputs
        self._preprocessing = preprocessing
        self._postprocessing = postprocessing
        if isinstance(bioimageio_model, v0_4.ModelDescr):
            self.input_ids = [TensorId(str(d.name)) for d in bioimageio_model.inputs]
            self.output_ids = [TensorId(str(d.name)) for d in bioimageio_model.outputs]
        else:
            self.input_ids = [d.id for d in bioimageio_model.inputs]
            self.output_ids = [d.id for d in bioimageio_model.outputs]

        self._adapter: ModelAdapter = model

    def __call__(
        self, *input_tensors: xr.DataArray, **named_input_tensors: xr.DataArray
    ) -> List[xr.DataArray]:
        return self.forward(*input_tensors, **named_input_tensors)

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore
        self.unload()
        return False

    def predict(
        self, *input_tensors: xr.DataArray, **named_input_tensors: xr.DataArray
    ) -> List[xr.DataArray]:
        """Predict input_tensor with the model without applying pre/postprocessing."""
        named_tensors = [
            named_input_tensors[str(k)] for k in self.input_ids[len(input_tensors):]
        ]
        return self._adapter.forward(*input_tensors, *named_tensors)

    def apply_preprocessing(self, sample: Sample) -> None:
        """apply preprocessing in-place, also updates sample stats"""
        for op in self._preprocessing:
            op(sample)

    def apply_postprocessing(self, sample: Sample) -> None:
        """apply postprocessing in-place, also updates samples stats"""
        for op in self._postprocessing:
            op(sample)

    def forward_sample(self, input_sample: Sample) -> Sample:
        """Apply preprocessing, run prediction and apply postprocessing."""
        self.apply_preprocessing(input_sample)

        prediction_tensors = self.predict(
            **{str(k): v for k, v in input_sample.data.items()}
        )
        prediction = Sample(
            data=dict(zip(self.output_ids, prediction_tensors)), stat=input_sample.stat
        )
        self.apply_postprocessing(prediction)
        return prediction

    def forward_tensors(
        self, *input_tensors: xr.DataArray, **named_input_tensors: xr.DataArray
    ) -> Dict[TensorId, xr.DataArray]:
        """Apply preprocessing, run prediction and apply postprocessing."""
        input_sample = Sample(
            data={
                **dict(zip(self.input_ids, input_tensors)),
                **{TensorId(k): v for k, v in named_input_tensors.items()},
            }
        )
        return self.forward_sample(input_sample).data

    def forward(
        self, *input_tensors: xr.DataArray, **named_input_tensors: xr.DataArray
    ) -> List[xr.DataArray]:
        """Apply preprocessing, run prediction and apply postprocessing."""
        named_outputs = self.forward_tensors(*input_tensors, **named_input_tensors)
        return [named_outputs[x] for x in self.output_ids]

    def load(self):
        """
        optional step: load model onto devices before calling forward
        if not using it as context manager
        """
        self._adapter.load()

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
    dataset_for_initial_statistics: Iterable[
        Union[Sample, Sequence[xr.DataArray]]
    ] = tuple(),
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
