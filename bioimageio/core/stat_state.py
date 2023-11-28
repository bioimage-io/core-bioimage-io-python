from dataclasses import dataclass, field
from typing import Dict, Iterable, Literal, Optional, Union

from tqdm import tqdm

from bioimageio.core.common import PER_DATASET, PER_SAMPLE, RequiredMeasure, Sample, TensorId
from bioimageio.core.stat_calculators import MeasureGroups, MeasureValue, get_measure_calculators
from bioimageio.core.stat_measures import MeasureBase, MeasureValue


@dataclass
class StatsState:
    """class to compute, hold and update dataset and sample statistics"""

    required_measures: Iterable[RequiredMeasure]


def compute_statistics():
    dataset: Iterable[Sample]
    update_dataset_stats_after_n_samples: Optional[int] = None
    update_dataset_stats_for_n_samples: Union[int, float] = float("inf")

def
    """iterates over dataset to compute dataset statistics (if required). The resulting dataset statistics are further updated with each new sample. A sample in this context may be a mini-batch.

    Args:
        required_measures: measures to be computed
        dataset: (partial) dataset to initialize dataset statistics with
        update_dataset_stats_after_n_samples: Update dataset statistics for new samples S_i if i > n.
                                                (default: len(dataset))
                                                This parameter allows to avoid weighting the first n processed
                                                samples to count twice if they make up the given 'dataset'.
        update_dataset_stats_for_n_samples: stop updating dataset statistics with new samples S_i if
                                            i > for_n_samples (+ update_dataset_stats_after_n_samples)
    """
    sample_count: int = field(init=False)
    last_sample: Optional[Sample] = field(init=False)
    measure_groups: MeasureGroups = field(init=False)
    _n_start: Union[int, float] = field(init=False)
    _n_stop: Union[int, float] = field(init=False)
    _final_dataset_stats: Optional[Dict[RequiredMeasure, MeasureValue]] = field(init=False)

    def __init__(
        self,
        *,
    ):
        super().__init__()
        self.required_measures = required_measures
        self.update_dataset_stats_after_n_samples = update_dataset_stats_after_n_samples
        self.update_dataset_stats_for_n_samples = update_dataset_stats_for_n_samples
        self.reset(dataset)

    def reset(self, dataset: Iterable[Sample]):
        self.sample_count = 0
        self.last_sample = None
        self._final_dataset_stats = None
        self.measure_groups = get_measure_calculators(self.required_measures)

        len_dataset = 0
        if self.measure_groups[PER_DATASET]:
            for sample in tqdm(dataset, "computing dataset statistics"):
                len_dataset += 1
                self._update_dataset_measure_groups(sample)

        if self.update_dataset_stats_after_n_samples is None:
            self._n_start = len_dataset
        else:
            self._n_start = self.update_dataset_stats_after_n_samples

        self._n_stop = self._n_start + self.update_dataset_stats_for_n_samples

    def update_with_sample(self, sample: Sample):
        self.last_sample = sample
        self.sample_count += 1
        if self._n_start < self.sample_count <= self._n_stop:
            self._update_dataset_measure_groups(sample)

    def _update_dataset_measure_groups(self, sample: Sample):
        for mg in self.measure_groups[PER_DATASET]:
            mg.update_with_sample(sample)

    def compute_measures(self) -> ComputedMeasures:
        ret = {PER_SAMPLE: {}, PER_DATASET: {}}
        if self.last_sample is not None:
            for mg in self.measure_groups[PER_SAMPLE]:
                ret[PER_SAMPLE].update(mg.compute(self.last_sample))

        if self._final_dataset_stats is None:
            dataset_stats = {}
            for mg in self.measure_groups[PER_DATASET]:
                dataset_stats.update(mg.finalize())

            if self.sample_count > self._n_stop:
                # stop recomputing final dataset statistics
                self._final_dataset_stats = dataset_stats
        else:
            dataset_stats = self._final_dataset_stats

        ret[PER_DATASET] = dataset_stats
        return ret
