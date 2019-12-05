from typing import Sequence, Optional

import numpy

from pybio.transformations import Transformation


class NumpylikeTransformation(Transformation):
    def __init__(self, apply_to: Optional[Sequence[int]] = None, **kwargs):
        super().__init__(apply_to=apply_to)
        self.kwargs = kwargs


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


class Reshape(NumpylikeTransformation):
    def apply_to_array(self, array: numpy.ndarray) -> numpy.ndarray:
        return numpy.reshape(array, **self.kwargs)


class Transpose(NumpylikeTransformation):
    def apply_to_array(self, array: numpy.ndarray) -> numpy.ndarray:
        return numpy.transpose(array, **self.kwargs)
