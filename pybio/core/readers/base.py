from typing import Sequence, Tuple

from pybio.core.array import PyBioArray
from pybio.core.transformations import apply_transformations
from pybio.spec import utils
from pybio.spec.node import OutputArray, Transformation


class PyBioReader:
    def __init__(self, outputs: Sequence[OutputArray], transformations: Sequence[Transformation] = tuple()):
        self.transformations = [utils.get_instance(trf) for trf in transformations]

        self._output = tuple(outputs)
        assert len(self.axes) == len(self.shape), (self.axes, self.shape)
        assert all(len(a) == len(s) for a, s in zip(self.axes, self.shape)), (self.axes, self.shape)

    @property
    def axes(self) -> Tuple[str, ...]:
        return tuple(out.axes for out in self._output)

    @property
    def shape(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(out.shape for out in self._output)

    @property
    def output(self) -> Tuple[OutputArray]:
        return self._output

    def __getitem__(self, rois: Tuple[Tuple[slice, ...], ...]) -> Sequence[PyBioArray]:
        raise NotImplementedError

    def apply_transformations(self, *arrays: PyBioArray) -> Sequence[PyBioArray]:
        if self.transformations is None:
            return arrays
        else:
            return apply_transformations(self.transformations, *arrays)
