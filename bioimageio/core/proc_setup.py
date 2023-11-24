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

from bioimageio.core.common import ProcessingKwargs, RequiredMeasure, Sample
from bioimageio.core.proc_impl import (
    ProcessingImpl,
    ProcessingImplBase,
    get_impl_class,
)
from bioimageio.core.stat_calculators import compute_measures
from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.model.v0_5 import TensorId

ModelDescr = Union[v0_4.ModelDescr, v0_5.ModelDescr]
TensorDescr = Union[v0_4.InputTensorDescr, v0_4.OutputTensorDescr, v0_5.InputTensorDescr, v0_5.OutputTensorDescr]


class _SetupProcessing(NamedTuple):
    preprocessing: List[ProcessingImpl]
    postprocessing: List[ProcessingImpl]


def setup_pre_and_postprocessing(model: ModelDescr, dataset: Iterator[Sample]) -> _SetupProcessing:
    Prepared = List[Tuple[Type[ProcessingImplBase[Any, Any, Any]], ProcessingKwargs, TensorId]]

    required_measures: Set[RequiredMeasure] = set()

    def prepare_procs(tensor_descrs: Sequence[TensorDescr]):
        prepared: Prepared = []
        for t_descr in tensor_descrs:
            if isinstance(t_descr, (v0_4.InputTensorDescr, v0_5.InputTensorDescr)):
                proc_specs = t_descr.preprocessing
            elif isinstance(
                t_descr,  # pyright: ignore[reportUnnecessaryIsInstance]
                (v0_4.OutputTensorDescr, v0_5.OutputTensorDescr),
            ):
                proc_specs = t_descr.postprocessing
            else:
                assert_never(t_descr)

            for proc_spec in proc_specs:
                impl_class = get_impl_class(proc_spec)
                tensor_id = cast(TensorId, t_descr.name) if isinstance(t_descr, v0_4.TensorDescrBase) else t_descr.id
                req = impl_class.get_required_measures(tensor_id, proc_spec.kwargs)  # type: ignore
                required_measures.update(req.get_set())
                prepared.append((impl_class, proc_spec.kwargs, tensor_id))

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
