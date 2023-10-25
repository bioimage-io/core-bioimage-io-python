import abc
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import xarray as xr

from bioimageio.core.model_adapters import ModelAdapter, create_model_adapter
from bioimageio.core.validation_visitors import resolve_raw_node
from bioimageio.spec.model import AnyModel, raw_nodes

from ._combined_processing import CombinedProcessing
from ._stat_state import StatsState
from ._utils import ComputedMeasures, Sample, TensorName


@dataclass
class NamedImplicitOutputShape:
    reference_input: TensorName = missing
    scale: List[Tuple[str, float]] = missing
    offset: List[Tuple[str, int]] = missing

    def __len__(self):
        return len(self.scale)


class PredictionPipeline(abc.ABC):
    """
    Represents model computation including preprocessing and postprocessing
    Note: Ideally use the PredictionPipeline as a context manager
    """

    @abc.abstractmethod
    def __enter__(self):
        ...

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        ...

    @abc.abstractmethod
    def forward(self, *input_tensors: xr.DataArray) -> List[xr.DataArray]:
        """
        Compute predictions
        """
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Name of the pipeline
        """
        ...

    @property
    @abc.abstractmethod
    def input_specs(self) -> List[nodes.InputTensor]:
        """
        specs of inputs
        """
        ...

    @property
    @abc.abstractmethod
    def output_specs(self) -> List[nodes.OutputTensor]:
        """
        specs of outputs
        """
        ...

    @abc.abstractmethod
    def load(self) -> None:
        """
        optional step: load model onto devices before calling forward if not using it as context manager
        """
        ...

    @abc.abstractmethod
    def unload(self) -> None:
        """
        free any device memory in use
        """
        ...


class _PredictionPipelineImpl(PredictionPipeline):
    def __init__(
        self,
        *,
        name: str,
        bioimageio_model: AnyModel,
        preprocessing: CombinedProcessing,
        postprocessing: CombinedProcessing,
        ipt_stats: StatsState,
        out_stats: StatsState,
        model: ModelAdapter,
    ) -> None:
        if bioimageio_model.run_mode:
            warnings.warn(f"Not yet implemented inference for run mode '{bioimageio_model.run_mode.name}'")

        self._name = name
        self._input_specs = bioimageio_model.inputs
        self._output_specs = bioimageio_model.outputs

        self._preprocessing = preprocessing
        self._postprocessing = postprocessing
        self._ipt_stats = ipt_stats
        self._out_stats = out_stats
        self._model: ModelAdapter = model

    def __call__(self, *input_tensors: xr.DataArray) -> List[xr.DataArray]:
        return self.forward(*input_tensors)

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False

    @property
    def name(self):
        return self._name

    @property
    def input_specs(self):
        return self._input_specs

    @property
    def output_specs(self):
        return self._output_specs

    def predict(self, *input_tensors: xr.DataArray) -> List[xr.DataArray]:
        """Predict input_tensor with the model without applying pre/postprocessing."""
        return self._model.forward(*input_tensors)

    def apply_preprocessing(self, sample: Sample, computed_measures: ComputedMeasures) -> None:
        """apply preprocessing in-place, also updates given computed_measures"""
        self._ipt_stats.update_with_sample(sample)
        for mode, stats in self._ipt_stats.compute_measures().items():
            if mode not in computed_measures:
                computed_measures[mode] = {}
            computed_measures[mode].update(stats)

        self._preprocessing.apply(sample, computed_measures)

    def apply_postprocessing(self, sample: Sample, computed_measures: ComputedMeasures) -> None:
        """apply postprocessing in-place, also updates given computed_measures"""
        self._out_stats.update_with_sample(sample)
        for mode, stats in self._out_stats.compute_measures().items():
            if mode not in computed_measures:
                computed_measures[mode] = {}
            computed_measures[mode].update(stats)

        self._postprocessing.apply(sample, computed_measures)

    def forward(self, *input_tensors: xr.DataArray) -> List[xr.DataArray]:
        """Apply preprocessing, run prediction and apply postprocessing.
        Note: The preprocessing might change input_tensors in-pace.
        """
        input_sample = dict(zip([ipt.name for ipt in self.input_specs], input_tensors))
        computed_measures = {}
        self.apply_preprocessing(input_sample, computed_measures)

        prediction_tensors = self.predict(*list(input_sample.values()))
        prediction = dict(zip([out.name for out in self.output_specs], prediction_tensors))
        self.apply_postprocessing(prediction, computed_measures)

        return [prediction[tn] for tn in [out.name for out in self.output_specs]]

    def load(self):
        self._model.load()

    def unload(self):
        self._model.unload()


def create_prediction_pipeline(
    bioimageio_model: AnyModel,
    *,
    devices: Optional[Sequence[str]] = None,
    weight_format: Optional[str] = None,
    dataset_for_initial_statistics: Iterable[Sequence[xr.DataArray]] = tuple(),
    update_dataset_stats_after_n_samples: Optional[int] = None,
    update_dataset_stats_for_n_samples: int = float("inf"),
    model_adapter: Optional[ModelAdapter] = None,
) -> PredictionPipeline:
    """
    Creates prediction pipeline which includes:
    * computation of input statistics
    * preprocessing
    * model prediction
    * computation of output statistics
    * postprocessing
    """
    model_adapter: ModelAdapter = model_adapter or create_model_adapter(
        bioimageio_model=bioimageio_model, devices=devices, weight_format=weight_format
    )
    if isinstance(bioimageio_model, nodes.Model):
        ipts = bioimageio_model.inputs
        outs = bioimageio_model.outputs

    else:
        assert isinstance(bioimageio_model, raw_nodes.Model)
        ipts = [resolve_raw_node(s, nodes) for s in bioimageio_model.inputs]
        outs = [resolve_raw_node(s, nodes) for s in bioimageio_model.outputs]

    preprocessing = CombinedProcessing.from_tensor_specs(ipts)

    def sample_dataset():
        for tensors in dataset_for_initial_statistics:
            yield dict(zip([ipt.name for ipt in bioimageio_model.inputs], tensors))

    ipt_stats = StatsState(
        preprocessing.required_measures,
        dataset=sample_dataset(),
        update_dataset_stats_after_n_samples=update_dataset_stats_after_n_samples,
        update_dataset_stats_for_n_samples=update_dataset_stats_for_n_samples,
    )
    postprocessing = CombinedProcessing.from_tensor_specs(outs)
    out_stats = StatsState(
        postprocessing.required_measures,
        dataset=tuple(),
        update_dataset_stats_after_n_samples=0,
        update_dataset_stats_for_n_samples=ipt_stats.sample_count + update_dataset_stats_for_n_samples,
    )

    return _PredictionPipelineImpl(
        name=bioimageio_model.name,
        bioimageio_model=bioimageio_model,
        model=model_adapter,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
        ipt_stats=ipt_stats,
        out_stats=out_stats,
    )
