from typing import Tuple, Sequence

import numpy

from pybio.core.readers.base import PyBioReader


class DummyReader(PyBioReader):
    _axes = "bx", "b"
    _shape = (15, 4), (15,)
    X = numpy.arange(60).reshape(15, 4)
    Y = numpy.array([i < 7 for i in range(15)])

    def __getitem__(self, rois: Tuple[Tuple[slice, ...], ...]) -> Sequence[numpy.ndarray]:
        assert len(rois) == 2
        return tuple(data[roi] for data, roi in zip([self.X, self.Y], rois))
