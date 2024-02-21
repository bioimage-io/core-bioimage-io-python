from typing import (
    List,
    NamedTuple,
    Sequence,
    Set,
    Union,
    cast,
)

from typing_extensions import assert_never

from bioimageio.core.proc_ops import Processing, get_proc_class
from bioimageio.core.stat_measures import Measure
from bioimageio.spec.model import AnyModelDescr, v0_4, v0_5
from bioimageio.spec.model.v0_5 import TensorId

TensorDescr = Union[v0_4.InputTensorDescr, v0_4.OutputTensorDescr, v0_5.InputTensorDescr, v0_5.OutputTensorDescr]


class _SetupProcessing(NamedTuple):
    preprocessing: List[Processing]
    postprocessing: List[Processing]
    preprocessing_req_measures: Set[Measure]
    postprocessing_req_measures: Set[Measure]


def setup_pre_and_postprocessing(model: AnyModelDescr) -> _SetupProcessing:
    pre_measures: Set[Measure] = set()
    post_measures: Set[Measure] = set()

    if isinstance(model, v0_4.ModelDescr):
        input_ids = {TensorId(d.name) for d in model.inputs}
        output_ids = {TensorId(d.name) for d in model.outputs}
    else:
        input_ids = {d.id for d in model.inputs}
        output_ids = {d.id for d in model.outputs}

    def prepare_procs(tensor_descrs: Sequence[TensorDescr]):
        procs: List[Processing] = []
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
                req = proc_class.from_proc_descr(proc_d, tensor_id)  # pyright: ignore[reportArgumentType]
                for m in req.required_measures:
                    if m.tensor_id in input_ids:
                        pre_measures.add(m)
                    elif m.tensor_id in output_ids:
                        post_measures.add(m)
                    else:
                        raise ValueError("When to raise ")
                procs.append(req)
        return procs

    pre_procs = prepare_procs(model.inputs)
    post_procs = prepare_procs(model.outputs)

    return _SetupProcessing(
        preprocessing=pre_procs,
        postprocessing=post_procs,
        preprocessing_req_measures=pre_measures,
        postprocessing_req_measures=post_measures,
    )
