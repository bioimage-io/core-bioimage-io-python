from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from pybio.core.array import PyBioArray
from pybio.core.transformations import apply_transformations
from pybio.spec import utils
from pybio.spec.node import MagicShapeValue, OutputArray, Transformation


class PyBioReader:
    def __init__(
        self,
        output: Union[Sequence[OutputArray], Path],
        dynamic_shape: Optional[Tuple[Optional[Tuple[int, ...]], ...]] = None,
        transformations: Sequence[Transformation] = tuple(),
    ):
        self.transformations = [utils.get_instance(trf) for trf in transformations]
        if isinstance(output, Path):
            output = utils.load_spec_and_kwargs(uri=str(output)).spec.outputs

        assert all(isinstance(out, OutputArray) for out in output)
        if dynamic_shape is not None:
            assert len(dynamic_shape) == len(output)
            output = list(output)
            for out, s in zip(output, dynamic_shape):
                if s is None:
                    assert isinstance(out.shape, tuple), type(out)
                else:
                    assert out.shape == MagicShapeValue.dynamic
                    out.shape = s

        self._output = tuple(output)
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
