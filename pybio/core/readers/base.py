from typing import Tuple, Sequence

import numpy


class PyBioReader:
    _axes: Tuple[str, ...]
    _shape: Tuple[Tuple[int, ...], ...]

    def __init__(self):
        assert all(len(a) == len(s) for a, s in zip(self.axes, self.shape))

    @property
    def axes(self) -> Tuple[str, ...]:
        return self._axes

    @property
    def shape(self) -> Tuple[Tuple[int, ...], ...]:
        return self._shape

    def __getitem__(self, rois: Tuple[Tuple[slice, ...], ...]) -> Sequence[numpy.ndarray]:
        raise NotImplementedError
