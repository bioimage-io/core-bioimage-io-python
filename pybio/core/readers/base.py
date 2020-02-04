from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy

from pybio.spec import utils
from pybio.spec.node import OutputArray


class PyBioReader:
    def __init__(self, output: Union[Sequence[OutputArray], Path]):
        if isinstance(output, Path):
            output = utils.load_spec_and_kwargs(uri=str(output)).spec.outputs

        assert all(isinstance(out, OutputArray) for out in output)
        self._output = tuple(output)
        assert all(len(a) == len(s) for a, s in zip(self.axes, self.shape))

    @property
    def axes(self) -> Tuple[str, ...]:
        return tuple(out.axes for out in self._output)

    @property
    def shape(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(out.shape for out in self._output)

    @property
    def output(self) -> Tuple[OutputArray]:
        return self._output

    def __getitem__(self, rois: Tuple[Tuple[slice, ...], ...]) -> Sequence[numpy.ndarray]:
        raise NotImplementedError
