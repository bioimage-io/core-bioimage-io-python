import numpy
import re

from typing import Optional, Sequence
from pybio.core.transformations import Transformation


class NumpylikeTransformation(Transformation):
    def __init__(self, apply_to: Optional[Sequence[int]] = None, **kwargs):
        super().__init__(apply_to=apply_to)
        self.kwargs = kwargs


special_numpy_name = {
    "AsType": "astype",
}
def make_numpy_like_transformation(class_name: str) -> NumpylikeTransformation:
    function_name = special_numpy_name.get(class_name, re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower())
    numpy_func = getattr(numpy, function_name, None)

    def apply_to_ndarray(self, array: numpy.ndarray) -> numpy.ndarray:
        if numpy_func is None:
            return getattr(array, function_name)(**self.kwargs)
        else:
            return numpy_func(array, **self.kwargs)

    return type(class_name, (NumpylikeTransformation,), {"apply_to_one": apply_to_ndarray})


__all__ = ["AsType", "Clip", "Reshape", "Transpose"]


for name in __all__:
    NumpyLikeClass = make_numpy_like_transformation(name)
    assert NumpyLikeClass.__name__ == name, (NumpyLikeClass.__name__, name)
    globals()[name] = NumpyLikeClass

del special_numpy_name, make_numpy_like_transformation
