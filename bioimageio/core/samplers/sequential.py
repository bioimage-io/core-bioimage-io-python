from math import ceil
from typing import Optional, Sequence, Tuple, Union

import numpy

from bioimageio.core.readers.base import PyBioReader
from bioimageio.core.samplers.base import PyBioSampler


class SequentialSamplerAlongDimension(PyBioSampler):
    def __init__(
        self,
        readers: Sequence[PyBioReader],
        sample_dimensions: Sequence[int],
        batch_size: int = 1,
        drop_last: bool = True,
    ):
        # first dimension to sample along is also used to concatenate readers (ideally all sample dimensions would be
        assert len(readers) == 1
        reader = readers[0]
        assert len(sample_dimensions) == len(reader.shape)
        self.sample_dimensions = sample_dimensions
        # axes/shape: sample dimension is 'consumed' and a batch dimension added
        self._axes = tuple("b" + a[:sd] + a[sd + 1 :] for a, sd in zip(reader.axes, sample_dimensions))
        shape = [list(s) for s in reader.shape]
        for s, sd in zip(shape, sample_dimensions):
            sd_len = s.pop(sd)
            if drop_last:
                batch_len = sd_len // batch_size
            else:
                batch_len = ceil(sd_len / batch_size)

            s.insert(0, batch_len)

        self._shape = tuple(tuple(s) for s in shape)
        self.None_slices_before = [(slice(None),) * sd for sd in sample_dimensions]
        self.None_slices_after = [(slice(None),) * (len(s) - sd - 1) for s, sd in zip(shape, sample_dimensions)]
        self.min_sample_dim_length = min(s[dim] for s, dim in zip(reader.shape, sample_dimensions))
        super().__init__(reader=reader, batch_size=batch_size, drop_last=drop_last)

    def get_rois(self, index: int, batch_size: int) -> Tuple[Tuple[slice]]:
        return tuple(
            before + (slice(index, index + batch_size),) + after
            for before, after in zip(self.None_slices_before, self.None_slices_after)
        )

    def __getitem__(self, index_and_optionally_batch: Union[int, Tuple[int, Optional[int]]]) -> Sequence[numpy.ndarray]:
        if isinstance(index_and_optionally_batch, int):
            index = index_and_optionally_batch
            batch_size = None
        else:
            index, batch_size = index_and_optionally_batch

        batch_size = batch_size or self.batch_size
        max_batch_size = self.min_sample_dim_length - index
        if max_batch_size < 1 or max_batch_size < batch_size and self.drop_last:
            raise IndexError(
                f"Can't return mini-batch of size {batch_size} at index {index} when sampling reader {self.reader} "
                f"along dimensions {self.sample_dimensions} (min dim: {self.min_sample_dim_length}) with `drop_last`="
                f"{self.drop_last}"
            )
        batch_size = min(max_batch_size, batch_size)

        return self.reader[self.get_rois(index, batch_size)]

    def __iter__(self):
        end = self.min_sample_dim_length
        if self.drop_last:
            end -= self.min_sample_dim_length % self.batch_size

        return (self[i] for i in range(0, end, self.batch_size))

    def __len__(self) -> int:
        return self.shape[0][0]
