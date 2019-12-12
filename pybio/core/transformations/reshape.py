from typing import Sequence, List, Tuple

from pybio.core.array import PyBioArray
from pybio.core.transformations import Transformation


class Reshape(Transformation):
    def __init__(self, shape: Sequence[int], **super_kwargs):
        self.shape = shape
        super().__init__(**super_kwargs)

    def apply_to_one(self, array: PyBioArray) -> PyBioArray:
        if -2 in self.shape and self.shape.index(-2) >= len(array.shape):
            raise ValueError(f"transformation shape {self.shape} incompatible with array shape {array.shape}")

        out_shape = tuple([array.shape[i] if s == -2 else s for i, s in enumerate(self.shape)])

        return array.reshape(out_shape)

    def dynamic_output_shape(self, input_shape: List[Tuple[int]]) -> List[Tuple[int]]:
        raise

    def dynamic_input_shape(self, output_shpe: List[Tuple[int]]) -> List[Tuple[int]]:
        raise NotImplementedError
