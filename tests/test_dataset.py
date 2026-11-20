from collections.abc import Iterable

from bioimageio.core.dataset import Dataset
from bioimageio.core.sample import Sample


def test_dataset_accepts_single_pass_iterables():
    sample = Sample(id="sample", members={}, stat={})
    dataset: Dataset = (item for item in [sample])
    assert isinstance(dataset, Iterable)
    assert list(dataset) == [sample]
    assert list(dataset) == []
