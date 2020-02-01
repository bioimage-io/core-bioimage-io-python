from typing import List, Sequence, Tuple, Optional, Union

from pybio.core.array import PyBioArray
from pybio.spec.node import InputArray, OutputArray


class ApplyToAll:
    def __contains__(self, item):
        return True


class Transformation:
    def __init__(self, apply_to: Optional[Sequence[int]] = None):
        self.apply_to = ApplyToAll() if apply_to is None else apply_to

    # todo: with python 3.8 add / to make array argument purely positional
    #       (might be called tensor or similar in derived classes)
    def apply_to_chosen(self, array: PyBioArray) -> PyBioArray:
        raise NotImplementedError

    def apply(self, *arrays: PyBioArray) -> List[PyBioArray]:
        return [self.apply_to_chosen(a) if i in self.apply_to else a for i, a in enumerate(arrays)]

    def dynamic_output_shape(self, input_shape: List[Tuple[int]]) -> List[Tuple[int]]:
        raise NotImplementedError

    def dynamic_input_shape(self, output_shape: List[Tuple[int]]) -> List[Tuple[int]]:
        raise NotImplementedError

    def dynamic_outputs(self, inputs: List[InputArray]) -> List[OutputArray]:
        raise NotImplementedError

    def dynamic_inputs(self, outputs: List[OutputArray]) -> List[InputArray]:
        raise NotImplementedError


class CombinedTransformation(Transformation):
    def apply_to_chosen(self, *arrays: PyBioArray) -> List[PyBioArray]:
        raise NotImplementedError

    def apply(self, *arrays: PyBioArray) -> List[PyBioArray]:
        if isinstance(self.apply_to, ApplyToAll):
            return self.apply_to_chosen(*arrays)
        else:
            return self.apply_to_chosen(*[arrays[i] for i in self.apply_to])


class SynchronizedTransformation(Transformation):
    """ Transformation for which application to all tensors is synchronized.
    This means, some state must be known before applying it to the tensors,
    e.g. the degree before a random rotation
    """

    def set_next_state(self):
        raise NotImplementedError

    def apply(self, *tensors):
        # TODO the state might depend on some tensor properties (esp. shape)
        # inferno solves this with the 'set_random_state' and 'get_random_state' construction
        # here, we could just pass *tensors to set_next_state
        self.set_next_state()
        return super().apply(*tensors)


def apply_transformations(transformations: Sequence[Transformation], *tensors: PyBioArray) -> List[PyBioArray]:
    """ Helper function to apply a list of transformations to input tensors.
    """
    if not all(isinstance(trafo, Transformation) for trafo in transformations):
        raise ValueError("Expect iterable of transformations")
    for trafo in transformations:
        tensors = trafo.apply(*tensors)

    return tensors
