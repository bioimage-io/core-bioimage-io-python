from typing import (
    Any,
    Iterator,
    List,
    NamedTuple,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    cast,
)

from typing_extensions import assert_never

from bioimageio.core.common import ProcessingKwargs, Sample
from bioimageio.core.proc_ops import (
    Processing,
    get_proc_class,
)
from bioimageio.core.stat_calculators import compute_measures
from bioimageio.core.stat_measures import Measure
from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.model.v0_5 import TensorId

ModelDescr = Union[v0_4.ModelDescr, v0_5.ModelDescr]
TensorDescr = Union[v0_4.InputTensorDescr, v0_4.OutputTensorDescr, v0_5.InputTensorDescr, v0_5.OutputTensorDescr]


class _SetupProcessing(NamedTuple):
    preprocessing: List[Processing]
    postprocessing: List[Processing]


def setup_pre_and_postprocessing(model: ModelDescr, dataset: Iterator[Sample]) -> _SetupProcessing:
    Prepared = List[Tuple[Type[Processing], ProcessingKwargs, TensorId]]

    required_measures: Set[Measure] = set()

    def prepare_procs(tensor_descrs: Sequence[TensorDescr]):
        prepared: Prepared = []
        for t_descr in tensor_descrs:
            if isinstance(t_descr, (v0_4.InputTensorDescr, v0_5.InputTensorDescr)):
                proc_descrs = t_descr.preprocessing
            elif isinstance(
                t_descr,
                (v0_4.OutputTensorDescr, v0_5.OutputTensorDescr),
            ):
                proc_descrs = t_descr.postprocessing
            else:
                assert_never(t_descr)

            for proc_d in proc_descrs:
                proc_class = get_proc_class(proc_d)
                tensor_id = cast(TensorId, t_descr.name) if isinstance(t_descr, v0_4.TensorDescrBase) else t_descr.id
                req = proc_class.from_proc_descr(proc_d, tensor_id)
                required_measures.update(req.get_set())
                prepared.append((proc_class, proc_d.kwargs, tensor_id))

        return prepared

    prepared_preps = prepare_procs(model.inputs)
    prepared_posts = prepare_procs(model.outputs)

    computed_measures = compute_measures(required_measures, dataset=dataset)

    def init_procs(prepared: Prepared):
        initialized: List[ProcessingImpl] = []
        for impl_class, kwargs, tensor_id in prepared:
            impl = impl_class(tensor_id=tensor_id, kwargs=kwargs, computed_measures=computed_measures)
            initialized.append(impl)

        return initialized

    return _SetupProcessing(init_procs(prepared_preps), init_procs(prepared_posts))
