from typing import Sequence

import numpy

from pybio.transformations import Transformation


class Reshape(Transformation):
    def __init__(self, shape: Sequence[int], **super_kwargs):
        self.shape = shape
        super().__init__(**super_kwargs)

    def apply_to_ndarray(self, array: numpy.ndarray) -> numpy.ndarray:
        if -2 in self.shape and self.shape.index(-2) >= len(array.shape):
            raise ValueError(f"transformation shape {self.shape} incompatible with array shape {array.shape}")

        out_shape = tuple([array.shape[i] if s == -2 else s for i, s in enumerate(self.shape)])

        return numpy.reshape(array, out_shape)
