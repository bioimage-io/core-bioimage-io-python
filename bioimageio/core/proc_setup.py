from types import MappingProxyType
from typing import (
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Sequence,
    Set,
    Union,
)

from typing_extensions import assert_never

from bioimageio.core.proc_ops import (
    AddKnownDatasetStats,
    Processing,
    UpdateStats,
    get_proc_class,
)
from bioimageio.core.sample import Sample
from bioimageio.core.stat_calculators import StatsCalculator
from bioimageio.core.stat_measures import DatasetMeasure, Measure, MeasureValue
from bioimageio.spec.model import AnyModelDescr, v0_4, v0_5
from bioimageio.spec.model.v0_5 import TensorId

TensorDescr = Union[
    v0_4.InputTensorDescr,
    v0_4.OutputTensorDescr,
    v0_5.InputTensorDescr,
    v0_5.OutputTensorDescr,
]


class PreAndPostprocessing(NamedTuple):
    pre: List[Processing]
    post: List[Processing]


class _SetupProcessing(NamedTuple):
    pre: List[Processing]
    post: List[Processing]
    pre_measures: Set[Measure]
    post_measures: Set[Measure]


def setup_pre_and_postprocessing(
    model: AnyModelDescr,
    dataset_for_initial_statistics: Iterable[Sample],
    keep_updating_initial_dataset_stats: bool = False,
    fixed_dataset_stats: Mapping[DatasetMeasure, MeasureValue] = MappingProxyType({}),
) -> PreAndPostprocessing:
    """
    Get pre- and postprocessing operators for a `model` description.
    userd in `bioimageio.core.create_prediction_pipeline"""
    prep, post, prep_meas, post_meas = _prepare_setup_pre_and_postprocessing(model)

    missing_dataset_stats = {
        m for m in prep_meas | post_meas if m not in fixed_dataset_stats
    }
    initial_stats_calc = StatsCalculator(missing_dataset_stats)
    for sample in dataset_for_initial_statistics:
        initial_stats_calc.update(sample)

    initial_stats = initial_stats_calc.finalize()
    prep.insert(
        0,
        UpdateStats(
            StatsCalculator(prep_meas, initial_stats),
            keep_updating_initial_dataset_stats=keep_updating_initial_dataset_stats,
        ),
    )
    post.insert(
        0,
        UpdateStats(
            StatsCalculator(post_meas, initial_stats),
            keep_updating_initial_dataset_stats=keep_updating_initial_dataset_stats,
        ),
    )
    if fixed_dataset_stats:
        prep.insert(0, AddKnownDatasetStats(fixed_dataset_stats))
        post.insert(0, AddKnownDatasetStats(fixed_dataset_stats))

    return PreAndPostprocessing(prep, post)


def _prepare_setup_pre_and_postprocessing(model: AnyModelDescr) -> _SetupProcessing:
    pre_measures: Set[Measure] = set()
    post_measures: Set[Measure] = set()

    if isinstance(model, v0_4.ModelDescr):
        input_ids = {TensorId(str(d.name)) for d in model.inputs}
        output_ids = {TensorId(str(d.name)) for d in model.outputs}
    else:
        input_ids = {d.id for d in model.inputs}
        output_ids = {d.id for d in model.outputs}

    def prepare_procs(tensor_descrs: Sequence[TensorDescr]):
        procs: List[Processing] = []
        for t_descr in tensor_descrs:
            if isinstance(t_descr, (v0_4.InputTensorDescr, v0_5.InputTensorDescr)):
                proc_descrs: List[
                    Union[
                        v0_4.PreprocessingDescr,
                        v0_5.PreprocessingDescr,
                        v0_4.PostprocessingDescr,
                        v0_5.PostprocessingDescr,
                    ]
                ] = list(t_descr.preprocessing)
            elif isinstance(
                t_descr,
                (v0_4.OutputTensorDescr, v0_5.OutputTensorDescr),
            ):
                proc_descrs = list(t_descr.postprocessing)
            else:
                assert_never(t_descr)

            if isinstance(t_descr, (v0_4.InputTensorDescr, v0_4.OutputTensorDescr)):
                ensure_dtype = v0_5.EnsureDtypeDescr(
                    kwargs=v0_5.EnsureDtypeKwargs(dtype=t_descr.data_type)
                )
                if isinstance(t_descr, v0_4.InputTensorDescr) and proc_descrs:
                    proc_descrs.insert(0, ensure_dtype)

                proc_descrs.append(ensure_dtype)

            for proc_d in proc_descrs:
                proc_class = get_proc_class(proc_d)
                tensor_id = (
                    TensorId(str(t_descr.name))
                    if isinstance(t_descr, v0_4.TensorDescrBase)
                    else t_descr.id
                )
                req = proc_class.from_proc_descr(
                    proc_d, tensor_id  # pyright: ignore[reportArgumentType]
                )
                for m in req.required_measures:
                    if m.tensor_id in input_ids:
                        pre_measures.add(m)
                    elif m.tensor_id in output_ids:
                        post_measures.add(m)
                    else:
                        raise ValueError("When to raise ")
                procs.append(req)
        return procs

    return _SetupProcessing(
        pre=prepare_procs(model.inputs),
        post=prepare_procs(model.outputs),
        pre_measures=pre_measures,
        post_measures=post_measures,
    )
