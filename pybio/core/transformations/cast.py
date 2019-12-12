from typing import Optional, Tuple

import numpy

from pybio.core.transformations import Transformation


class Cast(Transformation):
    def __init__(self, dtype: str, **super_kwargs):
        super().__init__(**super_kwargs)
        self.dtype = dtype

    def apply_to_one(self, array: numpy.ndarray) -> numpy.ndarray:
        return array.astype()


class Reshape(Transformation):
    def __init__(self, shape: Tuple[int], **super_kwargs):
        super().__init__(**super_kwargs)
        self.shape = shape

    def apply_to_one(self, array: numpy.ndarray) -> numpy.ndarray:
        return array.reshape(self.shape)


class Transpose(Transformation):
    def __init__(self, axes: Optional[Tuple[int]] = None, **super_kwargs):
        super().__init__(**super_kwargs)
        self.axes = axes or []

    def apply_to_one(self, array: numpy.ndarray) -> numpy.ndarray:
        return array.transpose(*self.axes)


