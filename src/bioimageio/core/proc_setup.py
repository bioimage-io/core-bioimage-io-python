from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import (
    NamedTuple,
)

from typing_extensions import assert_never

from bioimageio.spec.model import AnyModelDescr, v0_4, v0_5

from ._op_base import Operator
from .digest_spec import get_member_id
from .proc_ops import (
    AddKnownDatasetStats,
    EnsureDtype,
    Processing,
    UpdateStats,
    get_proc,
)
from .sample import Sample
from .stat_calculators import StatsCalculator, compute_dataset_measures
from .stat_measures import (
    DatasetMeasure,
    DatasetMeasureBase,
    Measure,
    MeasureValue,
    SampleMeasure,
    SampleMeasureBase,
)

TensorDescr = (
    v0_4.InputTensorDescr
    | v0_4.OutputTensorDescr
    | v0_5.InputTensorDescr
    | v0_5.OutputTensorDescr
)


class PreAndPostprocessing(NamedTuple):
    pre: list[Processing]
    post: list[Processing]


class _ProcessingCallables(NamedTuple):
    pre: Callable[[Sample], None]
    post: Callable[[Sample], None]


class _ApplyProcs:
    def __init__(self, procs: Sequence[Processing]):
        super().__init__()
        self._procs = procs

    def __call__(self, sample: Sample) -> None:
        for op in self._procs:
            op(sample)


def get_pre_and_postprocessing(
    model: AnyModelDescr,
    *,
    dataset_for_initial_statistics: Iterable[Sample],
    keep_updating_initial_dataset_stats: bool = False,
    fixed_dataset_stats: Mapping[Measure, MeasureValue] | None = None,
    dataset_fits_in_memory: bool = False,
) -> _ProcessingCallables:
    """Creates callables to apply pre- and postprocessing in-place to a sample"""

    setup = setup_pre_and_postprocessing(
        model=model,
        dataset_for_initial_statistics=dataset_for_initial_statistics,
        keep_updating_initial_dataset_stats=keep_updating_initial_dataset_stats,
        fixed_dataset_stats=fixed_dataset_stats,
        dataset_fits_in_memory=dataset_fits_in_memory,
    )
    return _ProcessingCallables(_ApplyProcs(setup.pre), _ApplyProcs(setup.post))


def setup_pre_and_postprocessing(
    model: AnyModelDescr,
    dataset_for_initial_statistics: Iterable[Sample],
    keep_updating_initial_dataset_stats: bool = False,
    fixed_dataset_stats: Mapping[Measure, MeasureValue] | None = None,
    dataset_fits_in_memory: bool = False,
) -> PreAndPostprocessing:
    """Get pre- and postprocessing operators for a `model` description.

    Used in `bioimageio.core.create_prediction_pipeline
    """

    prep = _get_described_procs(model.inputs)
    post = _get_described_procs(model.outputs, preceding_ops=prep)
    prep_required = _get_required_measures(prep)
    post_required = _get_required_measures(post) - prep_required
    fixed_stats = fixed_dataset_stats or {}

    stages: dict[tuple[Operator, ...], tuple[int, set[Measure]]] = {}
    for i, proc in enumerate(prep):
        for measure in proc.get_required_measures():
            if measure in fixed_stats:
                continue
            applied_ops = measure.preceding_ops
            if applied_ops in stages:
                first_i, measures = stages[applied_ops]
                measures.add(measure)
                stages[applied_ops] = (min(first_i, i), measures)
            else:
                stages[applied_ops] = (i, {measure})

    missing_post_stats = {
        measure for measure in post_required if measure not in fixed_stats
    }
    initial_dataset_measures = {
        measure
        for _, measures in stages.values()
        for measure in measures
        if isinstance(measure, DatasetMeasureBase)
    } | {
        measure
        for measure in missing_post_stats
        if isinstance(measure, DatasetMeasureBase)
    }
    initial_dataset = (
        list(dataset_for_initial_statistics) if initial_dataset_measures else []
    )

    fixed_stats_op = AddKnownDatasetStats(fixed_stats) if fixed_stats else None
    initial_pre: list[Processing] = [] if fixed_stats_op is None else [fixed_stats_op]
    pre: list[Processing] = list(initial_pre)
    initial_stats: dict[Measure, MeasureValue] = {}

    stages_by_index: dict[int, list[tuple[tuple[Operator, ...], set[Measure]]]] = {}
    for applied_ops, (i, measures) in stages.items():
        stages_by_index.setdefault(i, []).append((applied_ops, measures))

    for i, proc in enumerate(prep):
        for applied_ops, measures in stages_by_index.get(i, []):
            dataset_measures = {
                measure
                for measure in measures
                if isinstance(measure, DatasetMeasureBase)
            }
            if dataset_measures:
                initial_stats_calc = StatsCalculator(
                    dataset_measures,
                    dataset_fits_in_memory=dataset_fits_in_memory,
                )
                for original_sample in initial_dataset:
                    sample = original_sample.copy(deep=False)
                    sample.stat = dict(sample.stat)
                    for op in initial_pre:
                        op(sample)
                    initial_stats_calc.update(sample)

                stage_stats = initial_stats_calc.finalize()
                missing_initial_dataset_stats = dataset_measures - stage_stats.keys()
                if (
                    missing_initial_dataset_stats
                    and not keep_updating_initial_dataset_stats
                ):
                    raise ValueError(
                        "Model requires dataset statistics that were not provided: "
                        + f"{missing_initial_dataset_stats}. Provide "
                        + "`dataset_for_initial_statistics` or "
                        + "`fixed_dataset_statistics`, or set "
                        + "`keep_updating_initial_dataset_statistics=True` to "
                        + "update them during prediction."
                    )

                initial_stats.update(stage_stats.items())

            runtime_stats = {
                measure: value
                for measure, value in {
                    **fixed_stats,
                    **initial_stats,
                }.items()
                if measure in measures and isinstance(measure, DatasetMeasureBase)
            }
            runtime_updater = UpdateStats(
                StatsCalculator(
                    measures,
                    runtime_stats or None,
                    dataset_fits_in_memory=dataset_fits_in_memory,
                ),
                keep_updating_initial_dataset_stats=keep_updating_initial_dataset_stats,
            )
            pre.append(runtime_updater)
            initial_pre.append(
                UpdateStats(
                    StatsCalculator(
                        measures,
                        runtime_stats or None,
                        dataset_fits_in_memory=dataset_fits_in_memory,
                    ),
                )
            )

        pre.append(proc)
        initial_pre.append(proc)

    if missing_post_stats:
        post_dataset_measures = {
            measure
            for measure in missing_post_stats
            if isinstance(measure, DatasetMeasureBase)
        }
        output_member_ids = {
            get_member_id(tensor)
            for tensor in model.outputs
            if isinstance(tensor, (v0_4.OutputTensorDescr, v0_5.OutputTensorDescr))
        }
        output_dataset_measures = {
            measure
            for measure in post_dataset_measures
            if measure.member_id in output_member_ids
        }
        if output_dataset_measures:
            raise NotImplementedError(
                "Dataset statistics on model outputs are not supported: "
                + f"{output_dataset_measures}"
            )

        post_stats = compute_dataset_measures(
            post_dataset_measures,
            initial_dataset,
            dataset_fits_in_memory=dataset_fits_in_memory,
            initial_dataset_measures={
                measure: value
                for measure, value in {**fixed_stats, **initial_stats}.items()
                if isinstance(measure, DatasetMeasureBase)
            },
        )
        missing_initial_dataset_stats = post_dataset_measures - post_stats.keys()
        if missing_initial_dataset_stats and not keep_updating_initial_dataset_stats:
            raise ValueError(
                "Model requires dataset statistics that were not provided: "
                + f"{missing_initial_dataset_stats}. Provide "
                + "`dataset_for_initial_statistics` or `fixed_dataset_statistics`, "
                + "or set `keep_updating_initial_dataset_statistics=True` to update "
                + "them during prediction."
            )

        post_with_stats: list[Processing] = []
        added: set[Measure] = set()
        for proc in post:
            required = (set(proc.get_required_measures()) & missing_post_stats) - added
            by_applied_ops: dict[tuple[Operator, ...], set[Measure]] = {}
            for measure in required:
                by_applied_ops.setdefault(measure.preceding_ops, set()).add(measure)
            for applied_ops, measures in by_applied_ops.items():
                post_with_stats.append(
                    UpdateStats(
                        StatsCalculator(
                            measures,
                            {
                                measure: value
                                for measure, value in post_stats.items()
                                if measure in measures
                            },
                            dataset_fits_in_memory=dataset_fits_in_memory,
                        ),
                        keep_updating_initial_dataset_stats=keep_updating_initial_dataset_stats,
                    )
                )
            post_with_stats.append(proc)
            added.update(required)
        post = post_with_stats

    return PreAndPostprocessing(pre, post)


class RequiredMeasures(NamedTuple):
    pre: set[Measure]
    post: set[Measure]


class RequiredDatasetMeasures(NamedTuple):
    pre: set[DatasetMeasure]
    post: set[DatasetMeasure]


class RequiredSampleMeasures(NamedTuple):
    pre: set[SampleMeasure]
    post: set[SampleMeasure]


def get_requried_measures(model: AnyModelDescr) -> RequiredMeasures:
    pre = _get_described_procs(model.inputs)
    post = _get_described_procs(model.outputs, preceding_ops=pre)
    return RequiredMeasures(
        _get_required_measures(pre),
        _get_required_measures(post),
    )


def get_required_dataset_measures(model: AnyModelDescr) -> RequiredDatasetMeasures:
    req = get_requried_measures(model)
    return RequiredDatasetMeasures(
        {m for m in req.pre if isinstance(m, DatasetMeasureBase)},
        {m for m in req.post if isinstance(m, DatasetMeasureBase)},
    )


def get_requried_sample_measures(model: AnyModelDescr) -> RequiredSampleMeasures:
    req = get_requried_measures(model)
    return RequiredSampleMeasures(
        {m for m in req.pre if isinstance(m, SampleMeasureBase)},
        {m for m in req.post if isinstance(m, SampleMeasureBase)},
    )


def _get_described_procs(
    tensor_descrs: Iterable[TensorDescr],
    *,
    preceding_ops: Sequence[Processing] = (),
) -> list[Processing]:
    tensor_descrs = list(tensor_descrs)
    procs: list[Processing] = []
    preceding_ops = list(preceding_ops)

    def append(proc: Processing) -> None:
        procs.append(proc)
        preceding_ops.append(proc)

    for t_descr in tensor_descrs:
        first_proc = len(procs)
        if isinstance(t_descr, (v0_4.InputTensorDescr, v0_4.OutputTensorDescr)):
            member_id = get_member_id(t_descr)
            append(
                EnsureDtype(input=member_id, output=member_id, dtype=t_descr.data_type)
            )

        if isinstance(t_descr, (v0_4.InputTensorDescr, v0_5.InputTensorDescr)):
            for proc_d in t_descr.preprocessing:
                append(get_proc(proc_d, t_descr, preceding_ops=preceding_ops))
        elif isinstance(t_descr, (v0_4.OutputTensorDescr, v0_5.OutputTensorDescr)):
            for proc_d in t_descr.postprocessing:
                append(get_proc(proc_d, t_descr, preceding_ops=preceding_ops))
        else:
            assert_never(t_descr)

        if isinstance(t_descr, (v0_4.InputTensorDescr, v0_4.OutputTensorDescr)):
            if len(procs) == first_proc + 1:
                # remove initial ensure_dtype if there are no other proccessing steps
                assert isinstance(procs[-1], EnsureDtype)
                _ = procs.pop()
                _ = preceding_ops.pop()

            # ensure 0.4 models get float32 input
            # which has been the implicit assumption for 0.4
            member_id = get_member_id(t_descr)
            append(EnsureDtype(input=member_id, output=member_id, dtype="float32"))

    return procs


def _get_required_measures(procs: Iterable[Processing]) -> set[Measure]:
    return {measure for proc in procs for measure in proc.get_required_measures()}
