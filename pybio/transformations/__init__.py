from typing import List, Sequence, Callable, Any, Tuple

import numpy


class Transformation:
    expected_input_axes: List[str]
    expected_input_shapes: List[Tuple[int]]

    apply_to_ndarray: Callable[[numpy.ndarray], numpy.ndarray]

    def __init__(self, apply_to: Sequence[int] = (0,)):
        self.apply_to = apply_to

        if hasattr(self, "apply_to_ndarray"):
            self._apply_single = self.apply_to_ndarray
        else:
            raise NotImplementedError

    def __call__(self, tensors: List[numpy.ndarray]):
        return [self._apply_single(t) if i in self.apply_to else t for i, t in enumerate(tensors)]
