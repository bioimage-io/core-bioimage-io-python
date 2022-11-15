import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Union

from bioimageio.core.resource_io import nodes
from ._processing import AssertDtype, EnsureDtype, KNOWN_PROCESSING, Processing, TensorName
from ._utils import ComputedMeasures, PER_DATASET, PER_SAMPLE, RequiredMeasures, Sample

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore


@dataclasses.dataclass
class ProcessingInfoStep:
    name: str
    kwargs: Dict[str, Any]


@dataclasses.dataclass
class ProcessingInfo:
    steps: List[ProcessingInfoStep]
    assert_dtype_before: Optional[Union[str, Sequence[str]]] = None  # throw AssertionError if data type doesn't match
    ensure_dtype_before: Optional[str] = None  # cast data type if needed
    assert_dtype_after: Optional[Union[str, Sequence[str]]] = None  # throw AssertionError if data type doesn't match
    ensure_dtype_after: Optional[str] = None  # throw AssertionError if data type doesn't match


class CombinedProcessing:
    def __init__(self, combine_tensors: Dict[TensorName, ProcessingInfo]):
        self.procs = []
        known = dict(KNOWN_PROCESSING["pre"])
        known.update(KNOWN_PROCESSING["post"])

        # ensure all tensors have correct data type before any processing
        for tensor_name, info in combine_tensors.items():
            if info.assert_dtype_before is not None:
                self.procs.append(AssertDtype(tensor_name=tensor_name, dtype=info.assert_dtype_before))

            if info.ensure_dtype_before is not None:
                self.procs.append(EnsureDtype(tensor_name=tensor_name, dtype=info.ensure_dtype_before))

        for tensor_name, info in combine_tensors.items():
            for step in info.steps:
                self.procs.append(known[step.name](tensor_name=tensor_name, **step.kwargs))

            if info.assert_dtype_after is not None:
                self.procs.append(AssertDtype(tensor_name=tensor_name, dtype=info.assert_dtype_after))

            # ensure tensor has correct data type right after its processing
            if info.ensure_dtype_after is not None:
                self.procs.append(EnsureDtype(tensor_name=tensor_name, dtype=info.ensure_dtype_after))

        self.required_measures: RequiredMeasures = self._collect_required_measures(self.procs)
        self.tensor_names = list(combine_tensors)

    @classmethod
    def from_tensor_specs(cls, tensor_specs: List[Union[nodes.InputTensor, nodes.OutputTensor]]):
        combine_tensors = {}
        for ts in tensor_specs:
            # There is a difference between pre-and postprocessing:
            # After preprocessing we ensure float32, because the output is consumed by the model.
            # After postprocessing the dtype that is specified in the model spec needs to be ensured.
            assert ts.name not in combine_tensors
            if isinstance(ts, nodes.InputTensor):
                # todo: assert nodes.InputTensor.dtype with assert_dtype_before?
                # todo: in the long run we do not want to limit model inputs to float32...
                combine_tensors[ts.name] = ProcessingInfo(
                    [ProcessingInfoStep(p.name, kwargs=p.kwargs) for p in ts.preprocessing or []],
                    ensure_dtype_after="float32",
                )
            elif isinstance(ts, nodes.OutputTensor):
                combine_tensors[ts.name] = ProcessingInfo(
                    [ProcessingInfoStep(p.name, kwargs=p.kwargs) for p in ts.postprocessing or []],
                    ensure_dtype_after=ts.data_type,
                )
            else:
                raise NotImplementedError(type(ts))

        inst = cls(combine_tensors)
        for ts in tensor_specs:
            if isinstance(ts, nodes.OutputTensor) and ts.name in inst.required_measures[PER_DATASET]:
                raise NotImplementedError("computing statistics for output tensors per dataset is not yet implemented")

        return inst

    def apply(self, sample: Sample, computed_measures: ComputedMeasures) -> None:
        for proc in self.procs:
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
