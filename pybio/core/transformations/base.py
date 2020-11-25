from typing import Optional, OrderedDict, Sequence, Tuple, Union

from pybio.core.array import PyBioTensor


class PyBioTransformation:
    def __init__(self, apply_to: Union[str, Sequence[str]], output_names: Optional[Union[str, Sequence[str]]] = None):
        self.apply_to: Tuple[str] = (apply_to,) if isinstance(apply_to, str) else tuple(apply_to)
        self.output_names = output_names or self.apply_to

    # todo: with python 3.8 add / to make array argument purely positional
    #       (might be called tensor or similar in derived classes)
    def apply_to_chosen(self, tensor: PyBioTensor) -> PyBioTensor:
        raise NotImplementedError

    def apply(self, tensors: OrderedDict[str, PyBioTensor]) -> None:
        updates = [self.apply_to_chosen(t) for name, t in tensors.items() if name in self.apply_to]
        assert len(self.output_names) == len(updates)
        for name, update in zip(self.output_names, updates):
            tensors[name] = update

    # def dynamic_output_shape(self, input_shape: OrderedDict[Tuple[int]]) -> OrderedDict[Tuple[int]]:
    #     raise NotImplementedError
    #
    # def dynamic_input_shape(self, output_shape: OrderedDict[Tuple[int]]) -> OrderedDict[Tuple[int]]:
    #     raise NotImplementedError

    # def dynamic_outputs(self, inputs: Tuple[InputTensor]) -> Tuple[OutputTensor]:
    #     raise NotImplementedError
    #
    # def dynamic_inputs(self, outputs: Tuple[OutputTensor]) -> Tuple[InputTensor]:
    #     raise NotImplementedError


# class CombinedPyBioTransformation(PyBioTransformation):
#     def apply_to_chosen(self, *tensors: PyBioTensor) -> Sequence[PyBioTensor]:
#         raise NotImplementedError
#
#     def apply(self, tensors: OrderedDict[str, PyBioTensor]) -> None:
#         updates = self.apply_to_chosen(*[tensors[name] for name in self.apply_to])
#         assert len(self.output_names) == len(updates)
#         for name, update in zip(self.output_names, updates):
#             tensors[name] = update

#
# class SynchronizedPyBioTransformation(PyBioTransformation):
#     """ Transformation for which application to all tensors is synchronized.
#     This means, some state must be known before applying it to the tensors,
#     e.g. the degree before a random rotation
#     """
#
#     def set_next_state(self):
#         raise NotImplementedError
#
#     def apply(self, *tensors):
#         # TODO the state might depend on some tensor properties (esp. shape)
#         # inferno solves this with the 'set_random_state' and 'get_random_state' construction
#         # here, we could just pass *tensors to set_next_state
#         self.set_next_state()
#         return super().apply(*tensors)
