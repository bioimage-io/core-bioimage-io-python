import typing
import warnings
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Sequence, Set, Tuple

import xarray as xr

from bioimageio.core.measure_groups import MeanVarStd, PercentileGroup
from bioimageio.core.resource_io import nodes
from bioimageio.core.statistical_measures import Mean, Measure, MeasureValue, Percentile, Std, Var
from ._processing import (
    DatasetMode,
    EnsureDtype,
    KNOWN_POSTPROCESSING,
    KNOWN_PREPROCESSING,
    PER_DATASET,
    PER_SAMPLE,
    Processing,
    SampleMode,
    TensorName,
)

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore


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

        self.req_input_stats: Dict[Literal[SampleMode, DatasetMode], Dict[TensorName, Set[Measure]]] = {
            s: self._collect_required_stats(self._prep, s) for s in [PER_SAMPLE, PER_DATASET]
        }
        self.req_output_stats: Dict[Literal[SampleMode, DatasetMode], Dict[TensorName, Set[Measure]]] = {
            s: self._collect_required_stats(self._post, s) for s in [PER_SAMPLE, PER_DATASET]
        }
        if self.req_output_stats[PER_DATASET]:
            raise NotImplementedError("computing statistics for output tensors per dataset is not yet implemented")

        if self.req_input_stats[PER_DATASET] or self.req_output_stats[PER_DATASET]:
            self._computed_dataset_statistics: Optional[Dict[TensorName, Dict[Measure, MeasureValue]]] = None
        else:
            self._computed_dataset_statistics = {}

        self.input_tensor_names = [ipt.name for ipt in inputs]
        self.output_tensor_names = [out.name for out in outputs]
        assert not any(name in self.output_tensor_names for name in self.input_tensor_names)
        assert not any(name in self.input_tensor_names for name in self.output_tensor_names)

    def _require_computed_dataset_statistics(self):
        if self._computed_dataset_statistics is None:
            raise RuntimeError(
                "Missing computed dataset statistics, call 'set_computed_dataset_statistics or "
                "'compute_and_set_required_dataset_statistics' first."
            )

    def apply_preprocessing(
        self, *input_tensors: xr.DataArray
    ) -> Tuple[List[xr.DataArray], Dict[TensorName, Dict[Measure, MeasureValue]]]:
        assert len(input_tensors) == len(self.input_tensor_names)
        self._require_computed_dataset_statistics()

        tensors = dict(zip(self.input_tensor_names, input_tensors))
        sample_stats = self.compute_sample_statistics(tensors, self.req_input_stats[PER_SAMPLE])
        for proc in self._prep:
            proc.set_computed_statistics(sample_stats, mode=PER_SAMPLE)
            tensors[proc.tensor_name] = proc.apply(tensors[proc.tensor_name])

        return [tensors[tn] for tn in self.input_tensor_names], sample_stats

    def apply_postprocessing(
        self, *output_tensors: xr.DataArray, input_sample_statistics: Dict[TensorName, Dict[Measure, MeasureValue]]
    ) -> Tuple[List[xr.DataArray], Dict[TensorName, Dict[Measure, MeasureValue]]]:
        assert len(output_tensors) == len(self.output_tensor_names)
        self._require_computed_dataset_statistics()

        tensors = dict(zip(self.output_tensor_names, output_tensors))
        sample_stats = {
            **input_sample_statistics,
            **self.compute_sample_statistics(tensors, self.req_output_stats[PER_SAMPLE]),
        }
        for proc in self._post:
            proc.set_computed_statistics(sample_stats, mode=PER_SAMPLE)
            tensors[proc.tensor_name] = proc.apply(tensors[proc.tensor_name])

        return [tensors[tn] for tn in self.output_tensor_names], sample_stats

    def compute_and_set_required_dataset_statistics(
        self, dataset: typing.Iterable[Dict[TensorName, xr.DataArray]]
    ) -> None:
        if self.req_output_stats[PER_DATASET]:
            raise NotImplementedError("computing statistics for output tensors per dataset is not yet implemented")

        computed = self.compute_dataset_statistics(dataset, self.req_input_stats[PER_DATASET])
        self.set_computed_dataset_statistics(computed)

    def set_computed_dataset_statistics(self, computed: Dict[TensorName, Dict[Measure, MeasureValue]]):
        """
        This method sets the externally computed dataset statistics.
        Which statistics are expected is specified by the `required_dataset_statistics` property.
        """
        # always expect input tensor statistics
        for tensor_name, req_measures in self.req_input_stats[PER_DATASET]:
            comp_measures = computed.get(tensor_name, {})
            for req_measure in req_measures:
                if req_measure not in comp_measures:
                    raise ValueError(f"Missing required measure {req_measure} for input tensor {tensor_name}")

        # as output tensor statistics may initially not be available, we only warn about their absence
        output_statistics_missing = False
        for tensor_name, req_measures in self.req_output_stats[PER_DATASET]:
            comp_measures = computed.get(tensor_name, {})
            for req_measure in req_measures:
                if req_measure not in comp_measures:
                    output_statistics_missing = True
                    warnings.warn(f"Missing required measure {req_measure} for output tensor {tensor_name}")

        # set dataset statistics for each processing step
        for proc in self._prep:
            proc.set_computed_statistics(computed, mode=PER_DATASET)

        self._computed_dataset_statistics = computed

    @classmethod
    def compute_dataset_statistics(
        cls, dataset: typing.Iterable[Dict[TensorName, xr.DataArray]], measures: Dict[TensorName, Set[Measure]]
    ) -> Dict[TensorName, Dict[Measure, MeasureValue]]:

        # find MeasureGroups to compute dataset statistics in batches
        mean_var_std_groups = set()
        percentile_groups = defaultdict(list)
        for tn, ms in measures.items():
            for m in ms:
                if isinstance(m, (Mean, Var, Std)):
                    mean_var_std_groups.add((tn, m.axes))
                elif isinstance(m, Percentile):
                    percentile_groups[(tn, m.axes)].append(m.n)
                else:
                    raise NotImplementedError(f"Computing datasets statistics for {m} not yet implemented")

        measure_groups = []
        for (tn, axes) in mean_var_std_groups:
            measure_groups.append(MeanVarStd(tensor_name=tn, axes=axes))

        for (tn, axes), ns in percentile_groups.items():
            measure_groups.append(PercentileGroup(tensor_name=tn, axes=axes, ns=ns))

        for s in dataset:
            for mg in measure_groups:
                mg.update_with_sample(s)

        ret = {}
        for mg in measure_groups:
            ret.update(mg.finalize())

        return ret

    @classmethod
    def compute_sample_statistics(
        cls, tensors: Dict[TensorName, xr.DataArray], measures: Dict[TensorName, Set[Measure]]
    ) -> Dict[TensorName, Dict[Measure, MeasureValue]]:
        return {tname: cls._compute_tensor_statistics(tensors[tname], ms) for tname, ms in measures.items()}

    @staticmethod
    def _compute_tensor_statistics(tensor: xr.DataArray, measures: Set[Measure]) -> Dict[Measure, MeasureValue]:
        return {m: m.compute(tensor) for m in measures}

    @staticmethod
    def _collect_required_stats(
        proc: Sequence[Processing], mode: Literal[SampleMode, DatasetMode]
    ) -> Dict[TensorName, Set[Measure]]:
        stats: DefaultDict[TensorName, Set[Measure]] = defaultdict(set)
        for p in proc:
            req = p.get_required_statistics().get(mode, {})
            for tn, ms in req.items():
                stats[tn].update(ms)

        return dict(stats)
