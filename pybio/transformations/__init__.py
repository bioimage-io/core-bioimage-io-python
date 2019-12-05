from typing import List, Sequence, Tuple

from pybio.array import PyBioArray
from pybio_spec.spec_types import InputArray, OutputArray


class Transformation:
    def __init__(self, apply_to: Sequence[int] = (0,)):
        self.apply_to = apply_to

    def apply_to_array(self, array: PyBioArray) -> PyBioArray:
        raise NotImplementedError

    def apply(self, tensors: List[PyBioArray]) -> List[PyBioArray]:
        return [self.apply_to_array(t) if i in self.apply_to else t for i, t in enumerate(tensors)]

    def dynamic_output_shape(self, input_shape: List[Tuple[int]]) -> List[Tuple[int]]:
        raise NotImplementedError

    def dynamic_input_shape(self, output_shpe: List[Tuple[int]]) -> List[Tuple[int]]:
        raise NotImplementedError

    def dynamic_outputs(self, inputs: List[InputArray]) -> List[OutputArray]:
        raise NotImplementedError

    def dynamic_inputs(self, outputs: List[OutputArray]) -> List[InputArray]:
        raise NotImplementedError
