from typing import List, Sequence, Tuple, Optional, Union, Callable

from pybio.array import PyBioArray, PyBioScalar
from pybio_spec.spec_types import InputArray, OutputArray


class ApplyToAll:
    def __contains__(self, item):
        return True


class BaseTransformation:
    def apply(self, *arrays: PyBioArray):
        raise NotImplementedError

    def dynamic_output_shape(self, input_shape: List[Tuple[int]]) -> List[Tuple[int]]:
        raise NotImplementedError

    def dynamic_input_shape(self, output_shpe: List[Tuple[int]]) -> List[Tuple[int]]:
        raise NotImplementedError

    def dynamic_outputs(self, inputs: List[InputArray]) -> List[OutputArray]:
        raise NotImplementedError

    def dynamic_inputs(self, outputs: List[OutputArray]) -> List[InputArray]:
        raise NotImplementedError


class Transformation(BaseTransformation):
    def __init__(self, apply_to: Optional[Sequence[int]] = None):
        self.apply_to = ApplyToAll() if apply_to is None else apply_to

    def apply_to_one(self, array: PyBioArray) -> PyBioArray:
        raise NotImplementedError

    def apply(self, *arrays: PyBioArray) -> List[PyBioArray]:
        return [self.apply_to_one(a) if i in self.apply_to else a for i, a in enumerate(arrays)]


class Loss(BaseTransformation):
    loss_callable: Callable

    def __init__(self, apply_to: Optional[Sequence[int]] = None):
        self.apply_to = (0, 1) if apply_to is None else apply_to

    def apply(self, *arrays: PyBioArray, losses: List[PyBioScalar]) -> Tuple[Sequence[PyBioArray], List[PyBioScalar]]:
        losses.append(self.loss_callable(*[arrays[i] for i in self.apply_to]))
        return arrays, losses


def apply_transformations(transformations: Sequence[Transformation], *tensors: PyBioArray) -> List[PyBioArray]:
    """ Helper function to apply a list of transformations to input tensors.
    """
    if not all(isinstance(trafo, Transformation) for trafo in transformations):
        raise ValueError("Expect iterable of transformations")
    for trafo in transformations:
        tensors = trafo.apply(*tensors)

    return tensors


def apply_transformations_and_losses(
    transformations: Sequence[Union[Transformation, Loss]], *tensors: PyBioArray
) -> Tuple[List[PyBioArray], List[PyBioScalar]]:
    """ Helper function to apply a list of transformations to input tensors.
    """
    if not all(isinstance(trafo, Transformation) or isinstance(trafo, Loss) for trafo in transformations):
        raise ValueError("Expect iterable of transformations and losses")

    losses = []
    for trafo in transformations:
        if isinstance(trafo, Transformation):
            tensors = trafo.apply(*tensors)
        elif isinstance(trafo, Loss):
            tensors, losses = trafo.apply(*tensors, losses=losses)
        else:
            raise NotImplementedError(type(trafo))

    return tensors, losses
