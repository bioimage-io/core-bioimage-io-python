from typing import Optional, Sequence, Tuple

import torch.utils.data

from pybio.core.array import PyBioArray
from pybio.core.transformations import PyBioTransformation
from pybio.spec.nodes import MagicTensorsValue, OutputArray


class PyBioDataset(torch.utils.data.Dataset):
    def __init__(self, outputs: Sequence[OutputArray], transformation: Optional[PyBioTransformation] = None):
        if isinstance(outputs, MagicTensorsValue):
            raise ValueError(f"unresolved MagicTensorsValue: {outputs}")

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
    def output(self) -> Tuple[OutputArray]:
        return self._output

    def __getitem__(self, rois: Tuple[Tuple[slice, ...], ...]) -> Sequence[PyBioArray]:
        raise NotImplementedError

    def apply_transformation(self, *arrays: PyBioArray) -> Sequence[PyBioArray]:
        if self.transformation is None:
            return arrays
        else:
            return self.transformation.apply(*arrays)
