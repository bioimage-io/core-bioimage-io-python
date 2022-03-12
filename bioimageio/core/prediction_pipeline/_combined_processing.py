from typing import List, Optional, Sequence, Union

from bioimageio.core.resource_io import nodes
from ._processing import EnsureDtype, KNOWN_PROCESSING, Processing
from ._utils import ComputedMeasures, PER_DATASET, PER_SAMPLE, RequiredMeasures, Sample

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore


class CombinedProcessing:
    def __init__(self, tensor_specs: Union[List[nodes.InputTensor], List[nodes.OutputTensor]]):
        PRE: Literal["pre"] = "pre"
        POST: Literal["post"] = "post"
        proc_prefix: Optional[Literal["pre", "post"]] = None
        self._procs = []
        for t in tensor_specs:
            if isinstance(t, nodes.InputTensor):
                steps = t.preprocessing or []
                if proc_prefix is not None and proc_prefix != PRE:
                    raise ValueError(f"Invalid mixed input/output tensor specs: {tensor_specs}")

                proc_prefix = PRE
            elif isinstance(t, nodes.OutputTensor):
                steps = t.postprocessing or []
                if proc_prefix is not None and proc_prefix != POST:
                    raise ValueError(f"Invalid mixed input/output tensor specs: {tensor_specs}")

                proc_prefix = POST
            else:
                raise NotImplementedError(t)

            for step in steps:
                KNOWN_PROCESSING[proc_prefix][step.name](tensor_name=t.name, **step.kwargs)

        # There is a difference between pre-and-postprocessing:
        # Pre-processing always returns float32, because its output is consumed by the model.
        # Post-processing, however, should return the dtype that is specified in the model spec.
        # todo: cast dtype for inputs before preprocessing? or check dtype?
        if proc_prefix == POST:
            for t in tensor_specs:
                self._procs.append(EnsureDtype(tensor_name=t.name, dtype=t.data_type))

        self.required_measures: RequiredMeasures = self._collect_required_measures(self._procs)
        if proc_prefix == POST and self.required_measures[PER_DATASET]:
            raise NotImplementedError("computing statistics for output tensors per dataset is not yet implemented")

        self.tensor_names = [t.name for t in tensor_specs]

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
