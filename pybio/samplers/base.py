from typing import Tuple, Optional, Sequence, Union

import numpy

from pybio.readers.base import PyBioReader


class PyBioSampler:
    _axes: Tuple[str, ...]
    _shape: Tuple[Optional[Tuple[Optional[int], ...]], ...]
    # None implies unknown (part of) shape, e.g.
    # >>> _shape = (
    # >>>   (1, 2, 3),  # fully known shape
    # >>>   (2, None),  # partially known shape
    # >>>   None        # unknown shape
    # >>> )

    def __init__(self, reader: PyBioReader, batch_size: int = 1, drop_last: bool = True):
        self.reader = reader
        self.batch_size = batch_size
        self.drop_last = drop_last
        assert all(len(a) == len(s) for a, s in zip(self.axes, self.shape))

    @property
    def axes(self) -> Tuple[str, ...]:
        return self._axes

    @property
    def shape(self) -> Tuple[Optional[Tuple[Optional[int], ...]], ...]:
        return self._shape

    def __getitem__(self, index_and_optionally_batch: Union[int, Tuple[int, Optional[int]]]) -> Sequence[numpy.ndarray]:
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
