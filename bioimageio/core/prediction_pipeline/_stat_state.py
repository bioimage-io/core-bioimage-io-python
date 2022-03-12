from typing import Iterable, Optional, Union

from tqdm import tqdm

from ._measure_groups import MeasureGroups, get_measure_groups
from ._utils import ComputedMeasures, PER_DATASET, PER_SAMPLE, RequiredMeasures, Sample

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore


class StatState:
    """class to compute, hold and update dataset and sample statistics"""

    sample_count: int
    last_sample: Optional[Sample]
    measure_groups: MeasureGroups
    _n: Union[float, int]

    def __init__(
        self,
        required_measures: RequiredMeasures,
        dataset: Iterable[Sample],
        keep_updating_dataset_stats: bool = False,
        after_n_samples: Optional[int] = None,
    ):
        """iterates over dataset to compute dataset statistics (if required). The resulting dataset statistics are further updated with each new sample. A sample in this context may be a mini-batch.

        Args:
            required_measures: measures to be computed
            dataset: (partial) dataset to initialize dataset statistics with
            keep_updating_dataset_stats: weather or not to keep updating dataset statistics on 'update_with_sample'
            after_n_samples: only update for new samples S_i > after_n_samples (default: len(dataset))
        """
        self.required_measures = required_measures
        self.keep_updating_dataset_stats = keep_updating_dataset_stats
        self.after_n_samples = after_n_samples
        self.reset(dataset)

    def reset(self, dataset: Iterable[Sample]):
        self.sample_count = 0
        self.last_sample = None
        self.measure_groups = get_measure_groups(self.required_measures)

        len_dataset = 0
        if self.measure_groups[PER_DATASET]:
            for sample in tqdm(dataset, "computing dataset statistics"):
                len_dataset += 1
                self._update_dataset_measure_groups(sample)

        if not self.keep_updating_dataset_stats:
            self._n = float("inf")
        elif self.after_n_samples is None:
            self._n = len_dataset
        else:
            self._n = self.after_n_samples

    def update_with_sample(self, sample: Sample):
        self.last_sample = sample
        self.sample_count += 1
        if self.sample_count > self._n:
            self._update_dataset_measure_groups(sample)

    def _update_dataset_measure_groups(self, sample: Sample):
        for mg in self.measure_groups[PER_DATASET]:
            mg.update_with_sample(sample)

    def compute_measures(self) -> ComputedMeasures:
        ret = {PER_SAMPLE: {}, PER_DATASET: {}}
        for mg in self.measure_groups:
            ret[PER_SAMPLE].update(mg.compute(self.last_sample))

        for mg in self.measure_groups[PER_DATASET]:
            ret[PER_DATASET].update(mg.finalize(self.last_sample))

        return ret
