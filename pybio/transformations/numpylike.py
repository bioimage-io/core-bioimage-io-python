from typing import Tuple, Sequence

import numpy

from pybio.transformations import Transformation


# class PixelsAsBatch(Transformation):
#     def __init__(self, axes: str, **super_kwargs):
#
#         c_pos = axes.find("c")
#         if c_pos == -1:
#
#             def order(array: numpy.ndarray):
#                 return array[..., None]
#
#         else:
#
#             def order(array: numpy.ndarray):
#                 return numpy.moveaxis(array, c_pos, -1)
#
#         def apply_to_ndarray(array: numpy.ndarray) -> numpy.ndarray:
#             assert len(array.shape) == len(axes), (array.shape, axes)
#             ordered = order(array)
#             return ordered.reshape((-1, ordered.shape[-1]))
#
#         self.apply_to_ndarray = apply_to_ndarray
#         super().__init__(**super_kwargs)




class NumpylikeTransformation(Transformation):
    def __init__(self, apply_to: Sequence[int] = (0,), **kwargs):
        self.kwargs = kwargs
        super().__init__(apply_to=apply_to)


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
    def apply_to_ndarray(self, array: numpy.ndarray) -> numpy.ndarray:
        return numpy.reshape(array, **self.kwargs)

class Transpose(NumpylikeTransformation):
    def apply_to_ndarray(self, array: numpy.ndarray) -> numpy.ndarray:
        return numpy.transpose(array, **self.kwargs)

