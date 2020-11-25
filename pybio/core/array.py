from typing import Callable, TypeVar, Tuple, Union, NewType

import numpy

PyBioTensorDerived = TypeVar("PyBioTensorDerived", bound="PyBioTensorBase")


class PyBioTensorBase:
    reshape: Callable[[Tuple[int, ...]], PyBioTensorDerived]
    shape: Tuple[int, ...]


PyBioTensor = NewType("PyBioTensor", Union[PyBioTensorBase, numpy.ndarray])

# Same as pybio array, but should only hold a single scalar (e.g. a loss value)
PyBioScalar = NewType("PyBioScalar", Union[PyBioTensorBase, numpy.ndarray])
