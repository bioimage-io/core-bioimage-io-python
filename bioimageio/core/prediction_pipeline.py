import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence

import xarray as xr

from bioimageio.core.common import Sample, TensorId
from bioimageio.core.model_adapters import ModelAdapter, create_model_adapter
from bioimageio.core.model_adapters import get_weight_formats as get_weight_formats
from bioimageio.core.proc_ops import Processing
from bioimageio.core.proc_setup import setup_pre_and_postprocessing
from bioimageio.core.stat_calculators import StatsCalculator
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
        ipt_stats: StatsCalculator,
        out_stats: StatsCalculator,
        model: ModelAdapter,
    ) -> None:
        super().__init__()
        if bioimageio_model.run_mode:
            warnings.warn(f"Not yet implemented inference for run mode '{bioimageio_model.run_mode.name}'")

        self.name = name
        self._preprocessing = preprocessing
        self._postprocessing = postprocessing
        self._ipt_stats = ipt_stats
        self._out_stats = out_stats
        if isinstance(bioimageio_model, v0_4.ModelDescr):
            self._input_ids = [TensorId(d.name) for d in bioimageio_model.inputs]
            self._output_ids = [TensorId(d.name) for d in bioimageio_model.outputs]
        else:
            self._input_ids = [d.id for d in bioimageio_model.inputs]
            self._output_ids = [d.id for d in bioimageio_model.outputs]

        self._adapter: ModelAdapter = model

    def __call__(self, *input_tensors: xr.DataArray, **named_input_tensors: xr.DataArray) -> List[xr.DataArray]:
        return self.forward(*input_tensors, **named_input_tensors)

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore
        self.unload()
        return False

    def predict(self, *input_tensors: xr.DataArray, **named_input_tensors: xr.DataArray) -> List[xr.DataArray]:
        """Predict input_tensor with the model without applying pre/postprocessing."""
        named_tensors = [named_input_tensors[k] for k in self._input_ids[len(input_tensors) :]]
        return self._adapter.forward(*input_tensors, *named_tensors)

    def apply_preprocessing(self, sample: Sample) -> None:
        """apply preprocessing in-place, also updates sample stats"""
        sample.stat.update(self._ipt_stats.update_and_get_all(sample))
        for op in self._preprocessing:
            op(sample)

    def apply_postprocessing(self, sample: Sample) -> None:
        """apply postprocessing in-place, also updates samples stats"""
        sample.stat.update(self._out_stats.update_and_get_all(sample))
        for op in self._postprocessing:
            op(sample)

    def forward_sample(self, input_sample: Sample):
        """Apply preprocessing, run prediction and apply postprocessing."""
        self.apply_preprocessing(input_sample)

        prediction_tensors = self.predict(**input_sample.data)
        prediction = Sample(data=dict(zip(self._output_ids, prediction_tensors)), stat=input_sample.stat)
        self.apply_postprocessing(prediction)
        return prediction

    def forward_named(
        self, *input_tensors: xr.DataArray, **named_input_tensors: xr.DataArray
    ) -> Dict[TensorId, xr.DataArray]:
        """Apply preprocessing, run prediction and apply postprocessing."""
        input_sample = Sample(
            data={
                **dict(zip(self._input_ids, input_tensors)),
                **{TensorId(k): v for k, v in named_input_tensors.items()},
            }
        )
        return self.forward_sample(input_sample).data

    def forward(self, *input_tensors: xr.DataArray, **named_input_tensors: xr.DataArray) -> List[xr.DataArray]:
        """Apply preprocessing, run prediction and apply postprocessing."""
        named_outputs = self.forward_named(*input_tensors, **named_input_tensors)
        return [named_outputs[x] for x in self._output_ids]

    def load(self):
        """
        optional step: load model onto devices before calling forward if not using it as context manager
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
    dataset_for_initial_statistics: Iterable[Sequence[xr.DataArray]] = tuple(),
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
    if deprecated_kwargs:
        warnings.warn(f"deprecated create_prediction_pipeline kwargs: {set(deprecated_kwargs)}")

    model_adapter = model_adapter or create_model_adapter(
        model_description=bioimageio_model,
        devices=devices,
        weight_format_priority_order=weight_format and (weight_format,),
    )

    if isinstance(bioimageio_model, v0_4.ModelDescr):
        input_ids = [TensorId(ipt.name) for ipt in bioimageio_model.inputs]
    else:
        input_ids = [ipt.id for ipt in bioimageio_model.inputs]

    preprocessing, postprocessing, pre_req_meas, post_req_meas = setup_pre_and_postprocessing(bioimageio_model)
    ipt_stats = StatsCalculator(pre_req_meas)
    out_stats = StatsCalculator(post_req_meas)
    for tensors in dataset_for_initial_statistics:
        sample = Sample(data=dict(zip(input_ids, tensors)))
        ipt_stats.update(sample)
        out_stats.update(sample)

    return PredictionPipeline(
        name=bioimageio_model.name,
        bioimageio_model=bioimageio_model,
        model=model_adapter,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
        ipt_stats=ipt_stats,
        out_stats=out_stats,
    )
