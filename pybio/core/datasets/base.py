import collections
from typing import Dict, Optional, OrderedDict, Sequence, Tuple

from pybio.core.protocols import Tensor
from pybio.core.transformations import BatchTransformation
from pybio.spec.nodes import OutputTensor


class Dataset:
    def __init__(self, outputs: Sequence[OutputTensor], transformation: Optional[BatchTransformation] = None):
        self.transformation = transformation

        self._output = tuple(outputs)
        assert len(self.axes) == len(self.shape), (self.axes, self.shape)
        assert all(len(a) == len(s) for a, s in zip(self.axes, self.shape)), (self.axes, self.shape)

    @property
    def axes(self) -> OrderedDict[str, str]:
        return collections.OrderedDict([(out.name, out.axes) for out in self._output])

    @property
    def shape(self) -> OrderedDict[str, Tuple[int, ...]]:
        return collections.OrderedDict([(out.name, out.shape) for out in self._output])

    @property
    def output(self) -> Tuple[OutputTensor]:
        return self._output

    def __getitem__(self, rois: Dict[str, Tuple[slice, ...]]) -> OrderedDict[str, Tensor]:
        raise NotImplementedError

    def apply_transformation(self, batch: OrderedDict[str, Tensor]) -> None:
        if self.transformation is not None:
            return self.transformation.apply(batch)
