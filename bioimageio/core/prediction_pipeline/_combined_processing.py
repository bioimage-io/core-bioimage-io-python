import collections
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, Type

import xarray as xr
from marshmallow import missing

from bioimageio.core.resource_io import nodes
from bioimageio.core.statistical_measures import Mean, Measure, Percentile, Std
from bioimageio.spec.model.raw_nodes import PostprocessingName, PreprocessingName
from ._processing import (
    Binarize,
    Clip,
    EnsureDtype,
    Processing,
    ScaleLinear,
    ScaleMeanVariance,
    ScaleRange,
    Sigmoid,
    ZeroMeanUnitVariance,
)


KNOWN_PREPROCESSING: Dict[PreprocessingName, Type[Processing]] = {
    "binarize": Binarize,
    "clip": Clip,
    "scale_linear": ScaleLinear,
    "scale_range": ScaleRange,
    "sigmoid": Sigmoid,
    "zero_mean_unit_variance": ZeroMeanUnitVariance,
}

KNOWN_POSTPROCESSING: Dict[PostprocessingName, Type[Processing]] = {
    "binarize": Binarize,
    "clip": Clip,
    "scale_linear": ScaleLinear,
    "scale_mean_variance": ScaleMeanVariance,
    "scale_range": ScaleRange,
    "sigmoid": Sigmoid,
    "zero_mean_unit_variance": ZeroMeanUnitVariance,
}


Scope = Literal["sample", "dataset"]
SAMPLE: Literal["sample"] = "sample"
DATASET: Literal["dataset"] = "dataset"
SCOPES: Set[Scope] = {SAMPLE, DATASET}


class CombinedProcessing:
    def __init__(self, inputs: List[nodes.InputTensor], outputs: List[nodes.OutputTensor]):
        self._prep = [
            KNOWN_PREPROCESSING[step.name](tensor_name=ipt.name, **step.kwargs)
            for ipt in inputs
            for step in ipt.preprocessing or []
        ]
        self._post = [
            KNOWN_POSTPROCESSING.get(step.name)(tensor_name=out.name, **step.kwargs)
            for out in outputs
            for step in out.postprocessing or []
        ]

        # There is a difference between pre-and-postprocessing:
        # Pre-processing always returns float32, because its output is consumed by the model.
        # Post-processing, however, should return the dtype that is specified in the model spec.
        # todo: cast dtype for inputs before preprocessing? or check dtype?
        for out in outputs:
            self._post.append(EnsureDtype(tensor_name=out.name, dtype=out.data_type))

        self._req_input_stats = {s: self._collect_required_stats(self._prep, s) for s in SCOPES}
        self._req_output_stats = {s: self._collect_required_stats(self._post, s) for s in SCOPES}
        if any(self._req_output_stats[s] for s in SCOPES):
            raise NotImplementedError("computing statistics for output tensors not yet implemented")

        self._computed_dataset_stats: Optional[Dict[str, Dict[Measure, Any]]] = None

        self.input_tensor_names = [ipt.name for ipt in inputs]
        self.output_tensor_names = [out.name for out in outputs]
        assert not any(name in self.output_tensor_names for name in self.input_tensor_names)
        assert not any(name in self.input_tensor_names for name in self.output_tensor_names)

    @property
    def required_input_dataset_statistics(self) -> Dict[str, Set[Measure]]:
        return self._req_input_stats[DATASET]

    @property
    def required_output_dataset_statistics(self) -> Dict[str, Set[Measure]]:
        return self._req_output_stats[DATASET]

    @property
    def computed_dataset_statistics(self) -> Dict[str, Dict[Measure, Any]]:
        return self._computed_dataset_stats

    def apply_preprocessing(
        self, *input_tensors: xr.DataArray
    ) -> Tuple[List[xr.DataArray], Dict[str, Dict[Measure, Any]]]:
        assert len(input_tensors) == len(self.input_tensor_names)
        tensors = dict(zip(self.input_tensor_names, input_tensors))
        sample_stats = self.compute_sample_statistics(tensors, self._req_input_stats[SAMPLE])
        for proc in self._prep:
            proc.set_computed_sample_statistics(sample_stats)
            tensors[proc.tensor_name] = proc.apply(tensors[proc.tensor_name])

        return [tensors[tn] for tn in self.input_tensor_names], sample_stats

    def apply_postprocessing(
        self, *output_tensors: xr.DataArray, input_sample_statistics: Dict[str, Dict[Measure, Any]]
    ) -> Tuple[List[xr.DataArray], Dict[str, Dict[Measure, Any]]]:
        assert len(output_tensors) == len(self.output_tensor_names)
        tensors = dict(zip(self.output_tensor_names, output_tensors))
        sample_stats = input_sample_statistics
        sample_stats.update(self.compute_sample_statistics(tensors, self._req_output_stats[SAMPLE]))
        for proc in self._prep:
            proc.set_computed_sample_statistics(sample_stats)
            tensors[proc.tensor_name] = proc.apply(tensors[proc.tensor_name])

        return [tensors[tn] for tn in self.output_tensor_names], sample_stats

    def set_computed_dataset_statistics(self, computed: Dict[str, Dict[Measure, Any]]):
        """
        This method sets the externally computed dataset statistics.
        Which statistics are expected is specified by the `required_dataset_statistics` property.
        """
        # always expect input tensor statistics
        for tensor_name, req_measures in self.required_input_dataset_statistics:
            comp_measures = computed.get(tensor_name, {})
            for req_measure in req_measures:
                if req_measure not in comp_measures:
                    raise ValueError(f"Missing required measure {req_measure} for input tensor {tensor_name}")

        # as output tensor statistics may initially not be available, we only warn about their absence
        output_statistics_missing = False
        for tensor_name, req_measures in self.required_output_dataset_statistics:
            comp_measures = computed.get(tensor_name, {})
            for req_measure in req_measures:
                if req_measure not in comp_measures:
                    output_statistics_missing = True
                    warnings.warn(f"Missing required measure {req_measure} for output tensor {tensor_name}")

        self._computed_dataset_stats = computed

        # set dataset statistics for each processing step
        for proc in self._prep:
            proc.set_computed_dataset_statistics(self.computed_dataset_statistics)

    @classmethod
    def compute_sample_statistics(
        cls, tensors: Dict[str, xr.DataArray], measures: Dict[str, Set[Measure]]
    ) -> Dict[str, Dict[Measure, Any]]:
        return {tname: cls._compute_tensor_statistics(tensors[tname], ms) for tname, ms in measures.items()}

    @staticmethod
    def _compute_tensor_statistics(tensor: xr.DataArray, measures: Set[Measure]) -> Dict[Measure, Any]:
        ret = {}
        for measure in measures:
            if isinstance(measure, Mean):
                v = tensor.mean(dim=measure.axes)
            elif isinstance(measure, Std):
                v = tensor.std(dim=measure.axes)
            elif isinstance(measure, Percentile):
                v = tensor.quantile(measure.n / 100.0, dim=measure.axes)
            else:
                raise NotImplementedError(measure)

            ret[measure] = v

        return ret

    @staticmethod
    def _collect_required_stats(proc: Sequence[Processing], scope: Literal["sample", "dataset"]):
        stats = defaultdict(set)
        for p in proc:
            if scope == SAMPLE:
                req = p.get_required_sample_statistics()
            elif scope == DATASET:
                req = p.get_required_dataset_statistics()
            else:
                raise ValueError(scope)
            for tn, ms in req.items():
                stats[tn].update(ms)

        return dict(stats)
