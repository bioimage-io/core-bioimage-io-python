from typing import Dict, List, Optional, Sequence

from bioimageio.core.resource_io import nodes
from bioimageio.core.statistical_measures import Measure, MeasureValue
from ._processing import EnsureDtype, KNOWN_POSTPROCESSING, KNOWN_PREPROCESSING, Processing
from ._utils import ComputedMeasures, PER_DATASET, PER_SAMPLE, RequiredMeasures, Sample, TensorName

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

        self.req_input_stats: RequiredMeasures = self._collect_required_stats(self._prep)
        self.req_output_stats: RequiredMeasures = self._collect_required_stats(self._post)
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

    def apply_preprocessing(self, tensors: Sample, computed_measures: ComputedMeasures) -> None:
        for proc in self._prep:
            proc.set_computed_measures(computed_measures)
            tensors[proc.tensor_name] = proc.apply(tensors[proc.tensor_name])

    def apply_postprocessing(self, tensors: Sample, computed_measures: ComputedMeasures) -> None:
        for proc in self._post:
            proc.set_computed_measures(computed_measures)
            tensors[proc.tensor_name] = proc.apply(tensors[proc.tensor_name])

    # def compute_and_set_required_dataset_statistics(
    #     self, dataset: Iterable[Sample]
    # ) -> None:
    #     if self.req_output_stats[PER_DATASET]:
    #         raise NotImplementedError("computing statistics for output tensors per dataset is not yet implemented")
    #
    #     computed = self.compute_dataset_statistics(dataset, self.req_input_stats[PER_DATASET])
    #     self.set_computed_dataset_statistics(computed)

    # def set_computed_dataset_statistics(self, computed: Dict[TensorName, Dict[Measure, MeasureValue]]):
    #     """
    #     This method sets the externally computed dataset statistics.
    #     Which statistics are expected is specified by the `required_dataset_statistics` property.
    #     """
    #     # always expect input tensor statistics
    #     for tensor_name, req_measures in self.req_input_stats[PER_DATASET]:
    #         comp_measures = computed.get(tensor_name, {})
    #         for req_measure in req_measures:
    #             if req_measure not in comp_measures:
    #                 raise ValueError(f"Missing required measure {req_measure} for input tensor {tensor_name}")
    #
    #     # as output tensor statistics may initially not be available, we only warn about their absence
    #     output_statistics_missing = False
    #     for tensor_name, req_measures in self.req_output_stats[PER_DATASET]:
    #         comp_measures = computed.get(tensor_name, {})
    #         for req_measure in req_measures:
    #             if req_measure not in comp_measures:
    #                 output_statistics_missing = True
    #                 warnings.warn(f"Missing required measure {req_measure} for output tensor {tensor_name}")
    #
    #     # set dataset statistics for each processing step
    #     for proc in self._prep:
    #         proc.set_computed_measures(computed, mode=PER_DATASET)
    #
    #     self._computed_dataset_statistics = computed

    @staticmethod
    def _collect_required_stats(proc: Sequence[Processing]) -> RequiredMeasures:
        ret: RequiredMeasures = {PER_SAMPLE: {}, PER_DATASET: {}}
        for p in proc:
            for mode, ms_per_mode in p.get_required_measures().items():
                for tn, ms_per_tn in ms_per_mode.items():
                    if tn not in ret[mode]:
                        ret[mode][tn] = set()

                    ret[mode][tn].update(ms_per_tn)

        return ret
