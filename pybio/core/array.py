from typing import Callable, TypeVar, Tuple, Union, NewType

import numpy

PyBioArrayDerived = TypeVar("PyBioArrayDerived", bound="PyBioArrayBase")

class PyBioArrayBase:
    reshape: Callable[[Tuple[int, ...]], PyBioArrayDerived]
    shape: Tuple[int, ...]


PyBioArray = NewType("PyBioArray", Union[PyBioArrayBase, numpy.ndarray])
