from typing import Optional, Sequence, Tuple

from pybio.core.array import PyBioTensor
from pybio.core.transformations import PyBioTransformation
from pybio.spec.nodes import OutputTensor


class PyBioDataset:
    def __init__(self, outputs: Sequence[OutputTensor], transformation: Optional[PyBioTransformation] = None):
        self.transformation = transformation

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
    def output(self) -> Tuple[OutputTensor]:
        return self._output

    def __getitem__(self, rois: Tuple[Tuple[slice, ...], ...]) -> Sequence[PyBioTensor]:
        raise NotImplementedError

    def apply_transformation(self, *arrays: PyBioTensor) -> Sequence[PyBioTensor]:
        if self.transformation is None:
            return arrays
        else:
            return self.transformation.apply(*arrays)
