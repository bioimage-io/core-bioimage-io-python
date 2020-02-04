from math import ceil

import pytest

from pybio.core.readers.dummy import DummyReader
from pybio.core.samplers.sequential import SequentialSamplerAlongDimension


@pytest.fixture
def dummy_reader():
    return DummyReader()


def test_SequentialSamplerAlongDimension(dummy_reader):
    sampler = SequentialSamplerAlongDimension(sample_dimensions=[0, 0], readers=[dummy_reader])
    for i, s in enumerate(sampler):
        assert s

    assert i == 14  # hard coded in DummyReader


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_SequentialSamplerAlongDimension_with_batch_size_drop_last(dummy_reader, batch_size):
    sampler = SequentialSamplerAlongDimension(
        sample_dimensions=[0, 0], readers=[dummy_reader], batch_size=batch_size, drop_last=True
    )
    for i, s in enumerate(sampler):
        assert s

    assert i + 1 == 15 // batch_size  # 15 hard coded in DummyReader


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_SequentialSamplerAlongDimension_with_batch_size_dont_drop_last(dummy_reader, batch_size):
    sampler = SequentialSamplerAlongDimension(
        sample_dimensions=[0, 0], readers=[dummy_reader], batch_size=batch_size, drop_last=False
    )
    for i, s in enumerate(sampler):
        assert s

    assert i + 1 == ceil(15 / batch_size)  # 15 hard coded in DummyReader
