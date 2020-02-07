from typing import Sequence, List, Tuple

import numpy

from pybio.core.array import PyBioArray
from pybio.core.transformations import PyBioTransformation


class Reshape(PyBioTransformation):
    def __init__(self, shape: Sequence[int], **super_kwargs):
        self.shape = shape
        super().__init__(**super_kwargs)

    def apply_to_chosen(self, array: PyBioArray) -> PyBioArray:
        if -2 in self.shape and self.shape.index(-2) >= len(array.shape):
            raise ValueError(f"transformation shape {self.shape} incompatible with array shape {array.shape}")

        out_shape = tuple([array.shape[i] if s == -2 else s for i, s in enumerate(self.shape)])

        return array.reshape(out_shape)

    def dynamic_output_shape(self, input_shape: List[Tuple[int]]) -> List[Tuple[int]]:
        output_shape = []
        for i, ipt_shape in enumerate(input_shape):
            if i in self.apply_to:
                s = numpy.prod(ipt_shape)
                rest_dim = None
                out_shape = list(ipt_shape)
                for out_idx, out in enumerate(self.shape):
                    if out == -1:
                        rest_dim = out_idx
                        continue

                    if out == -2:
                        out = ipt_shape[self.shape.index(-2)]
                        out_shape[out_idx] = out

                    if s / out != s // out:
                        raise ValueError(f"Cannot reshape {ipt_shape} to {self.shape}")

                    s //= out

                if rest_dim is not None:
                    out_shape[rest_dim] = s
                elif s != 1:
                    raise ValueError(f"Cannot reshape {ipt_shape} to {self.shape}")

                output_shape.append(tuple(out_shape))
            else:
                output_shape.append(ipt_shape)

        return output_shape

    def dynamic_input_shape(self, output_shape: List[Tuple[int]]) -> List[Tuple[int]]:
        raise NotImplementedError
