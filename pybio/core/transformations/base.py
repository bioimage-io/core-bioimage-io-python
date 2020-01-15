from typing import List, Sequence, Tuple, Optional

from pybio.core.array import PyBioArray
from pybio.spec.spec_types import InputArray, OutputArray


class ApplyToAll:
    def __contains__(self, item):
        return True


class Transformation:
    def __init__(self, apply_to: Optional[Sequence[int]] = None):
        self.apply_to = ApplyToAll() if apply_to is None else apply_to

    # todo: with python 3.8 add / to make array argument purely positional
    #       (might be called tensor or similar in derived classes)
    def apply_to_one(self, array: PyBioArray) -> PyBioArray:
        raise NotImplementedError

    def apply(self, *arrays: PyBioArray) -> List[PyBioArray]:
        return [self.apply_to_one(a) if i in self.apply_to else a for i, a in enumerate(arrays)]

    def dynamic_output_shape(self, input_shape: List[Tuple[int]]) -> List[Tuple[int]]:
        raise NotImplementedError

    def dynamic_input_shape(self, output_shpe: List[Tuple[int]]) -> List[Tuple[int]]:
        raise NotImplementedError

    def dynamic_outputs(self, inputs: List[InputArray]) -> List[OutputArray]:
        raise NotImplementedError

    def dynamic_inputs(self, outputs: List[OutputArray]) -> List[InputArray]:
        raise NotImplementedError


def apply_transformations(transformations: Sequence[Transformation], *tensors: PyBioArray) -> List[PyBioArray]:
    """ Helper function to apply a list of transformations to input tensors.
    """
    if not all(isinstance(trafo, Transformation) for trafo in transformations):
        raise ValueError("Expect iterable of transformations")
    for trafo in transformations:
        tensors = trafo.apply(*tensors)

    return tensors
