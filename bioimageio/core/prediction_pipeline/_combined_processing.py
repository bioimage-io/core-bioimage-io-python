import dataclasses
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, TypedDict, Union

from bioimageio.core.resource_io import nodes
from ._processing import EnsureDtype, KNOWN_PROCESSING, Processing, TensorName
from ._utils import ComputedMeasures, PER_DATASET, PER_SAMPLE, RequiredMeasures, Sample

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore


@dataclasses.dataclass
class TensorProcessingInfo:
    processing_steps: Union[List[nodes.Preprocessing], List[nodes.Postprocessing]]
    data_type_before: Optional[str] = None
    data_type_after: Optional[str] = None


class CombinedProcessing:
    def __init__(self, combine_tensors: Dict[TensorName, TensorProcessingInfo]):
        self._procs = []
        known = dict(KNOWN_PROCESSING["pre"])
        known.update(KNOWN_PROCESSING["post"])

        # ensure all tensors have correct data type before any processing
        for tensor_name, info in combine_tensors.items():
            if info.data_type_before is not None:
                self._procs.append(EnsureDtype(tensor_name=tensor_name, dtype=info.data_type_before))

        for tensor_name, info in combine_tensors.items():
            for step in info.processing_steps:
                self._procs.append(known[step.name](tensor_name=tensor_name, **step.kwargs))

            # ensure tensor has correct data type right after its processing
            if info.data_type_after is not None:
                self._procs.append(EnsureDtype(tensor_name=tensor_name, dtype=info.data_type_after))

        self.required_measures: RequiredMeasures = self._collect_required_measures(self._procs)
        self.tensor_names = list(combine_tensors)

    @classmethod
    def from_tensor_specs(cls, tensor_specs: List[Union[nodes.InputTensor, nodes.OutputTensor]]):
        combine_tensors = {}
        for ts in tensor_specs:
            # There is a difference between pre-and postprocessing:
            # Preprocessing always returns float32, because its output is consumed by the model.
            # Postprocessing, however, should return the dtype that is specified in the model spec.
            # todo: cast dtype for inputs before preprocessing? or check dtype?
            assert ts.name not in combine_tensors
            if isinstance(ts, nodes.InputTensor):
                # todo: move preprocessing ensure_dtype here as data_type_after
                combine_tensors[ts.name] = TensorProcessingInfo(ts.preprocessing)
            elif isinstance(ts, nodes.OutputTensor):
                combine_tensors[ts.name] = TensorProcessingInfo(ts.postprocessing, None, ts.data_type)
            else:
                raise NotImplementedError(type(ts))

        inst = cls(combine_tensors)
        for ts in tensor_specs:
            if isinstance(ts, nodes.OutputTensor) and ts.name in inst.required_measures[PER_DATASET]:
                raise NotImplementedError("computing statistics for output tensors per dataset is not yet implemented")

        return inst

    def apply(self, sample: Sample, computed_measures: ComputedMeasures) -> None:
        for proc in self._procs:
            proc.set_computed_measures(computed_measures)
            sample[proc.tensor_name] = proc.apply(sample[proc.tensor_name])

    @staticmethod
    def _collect_required_measures(proc: Sequence[Processing]) -> RequiredMeasures:
        ret: RequiredMeasures = {PER_SAMPLE: {}, PER_DATASET: {}}
        for p in proc:
            for mode, ms_per_mode in p.get_required_measures().items():
                for tn, ms_per_tn in ms_per_mode.items():
                    if tn not in ret[mode]:
                        ret[mode][tn] = set()

                    ret[mode][tn].update(ms_per_tn)

        return ret
