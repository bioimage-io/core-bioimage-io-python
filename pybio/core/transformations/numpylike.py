from typing import Optional, Tuple

import numpy

from pybio.core.transformations import Transformation


# class NumpylikeTransformation(Transformation):
#     def __init__(self, apply_to: Optional[Sequence[int]] = None, **kwargs):
#         super().__init__(apply_to=apply_to)
#         self.kwargs = kwargs


# tdo: remove commented code
# def make_numpy_like_transformation(function_name: str) -> NumpylikeTransformation:
#     numpy_func = getattr(numpy, function_name)
#
#     def apply_to_ndarray(self, array: numpy.ndarray) -> numpy.ndarray:
#         return numpy_func(array, **self.kwargs)
#
#     return type(
#         function_name.title().replace("_", ""), (NumpylikeTransformation,), {"apply_to_ndarray": apply_to_ndarray}
#     )
# __all__ = [make_numpy_like_transformation(function_name) for function_name in ["reshape", "transpose"]]

class AsType(Transformation):
    def __init__(self, dtype: str, order: str, casting: str, subok: bool, copy: bool, **super_kwargs):
        super().__init__(**super_kwargs)
        self.kwargs = {"dtype": dtype, "order": order, "casting": casting, "subok": subok, "copy": copy}

    def apply_to_one(self, array: numpy.ndarray) -> numpy.ndarray:
        return array.astype(**self.kwargs)


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


