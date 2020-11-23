from typing import List, Optional, Sequence, Tuple

from pybio.core.array import PyBioArray
from pybio.core.transformations import ApplyToAll, PyBioTransformation
from pybio.spec.nodes import InputTensor, OutputTensor


class NormalizeZeroMeanUnitVariance(PyBioTransformation):
    def __init__(
        self,
        eps=1.0e-6,
        means: Sequence[Optional[float]] = (None,),
        stds: Sequence[Optional[float]] = (None,),
        apply_to: Optional[Sequence[int]] = (0,),
        **super_kwargs,
    ):
        assert len(means) == len(stds)
        if apply_to is not None:
            assert len(apply_to) == len(means)

        super().__init__(apply_to=apply_to, **super_kwargs)

        self.eps = eps
        self.means = tuple(means)
        self.stds = tuple(stds)

    def apply(self, *arrays: PyBioArray) -> List[PyBioArray]:
        if isinstance(self.apply_to, ApplyToAll):
            assert len(self.means) == len(arrays)
            assert len(self.stds) == len(arrays)
            means = self.means
            stds = self.stds
        else:
            means = [
                self.means[self.apply_to.index(idx)] if idx in self.apply_to else None for idx in range(len(arrays))
            ]
            stds = [self.stds[self.apply_to.index(idx)] if idx in self.apply_to else None for idx in range(len(arrays))]

        means = [
            None if i not in self.apply_to else m if m is not None else a.mean()
            for i, (a, m) in enumerate(zip(arrays, means))
        ]
        stds = [
            None if i not in self.apply_to else s if s is not None else a.std()
            for i, (a, s) in enumerate(zip(arrays, stds))
        ]
        return [
            (a - m) / (s + self.eps) if i in self.apply_to else a
            for i, (a, m, s) in enumerate(zip(arrays, means, stds))
        ]

    def dynamic_output_shape(self, input_shape: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
        return input_shape

    def dynamic_input_shape(self, output_shape: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
        return output_shape

    def dynamic_outputs(self, inputs: Tuple[InputTensor]) -> Tuple[OutputTensor]:
        return tuple(
            OutputTensor(
                name=ipt.name,
                axes=ipt.axes,
                data_type=ipt.data_type,
                data_range=(float("-inf"), float("inf")),
                shape=ipt.shape,
                halo=(0,) * len(ipt.shape),
            )
            for ipt in inputs
        )

    def dynamic_inputs(self, outputs: Tuple[OutputTensor]) -> Tuple[InputTensor]:
        return tuple(
            InputTensor(
                name=out.name,
                axes=out.axes,
                data_type="numeric",
                data_range=(float("-inf"), float("inf")),
                shape=out.shape,
            )
            for out in outputs
        )


class NormalizeRange(PyBioTransformation):
    def __init__(
        self,
        *,
        apply_to: int,
        output_min: float,
        output_max: float,
        data_min: Optional[float] = None,
        data_max: Optional[float] = None,
        minimal_data_range=1.0e-6,
        **super_kwargs,
    ):
        if data_min is not None and data_max is not None:
            assert data_min < data_max, (data_min, data_max)
        assert isinstance(apply_to, int), type(apply_to)
        super().__init__(apply_to=(apply_to, ), **super_kwargs)

        self.output_min = output_min
        self.output_max = output_max
        self.data_min = data_min
        self.data_max = data_max
        self.minimal_data_range=minimal_data_range

    def apply_to_chosen(self, array: PyBioArray) -> PyBioArray:
        data_min = array.min() if self.data_min is None else self.data_min
        data_max = array.max() if self.data_max is None else self.data_max
        data_range = max(self.minimal_data_range, data_max - data_min)
        ret = array.astype("float32")
        ret -= data_min
        ret /= data_range
        output_range = self.output_max - self.output_min
        ret *= output_range
        ret += self.output_min
        return ret

    def dynamic_output_shape(self, input_shape: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
        return input_shape

    def dynamic_input_shape(self, output_shape: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
        return output_shape

    def dynamic_outputs(self, inputs: Tuple[InputTensor]) -> Tuple[OutputTensor]:
        return tuple(
            OutputTensor(
                name=ipt.name,
                axes=ipt.axes,
                data_type="float32",
                data_range=(self.output_min, self.output_max),
                shape=ipt.shape,
                halo=[0] * len(ipt.shape),
            )
            for ipt in inputs
        )

    def dynamic_inputs(self, outputs: Tuple[OutputTensor]) -> Tuple[InputTensor]:
        return tuple(
            InputTensor(
                name=out.name,
                axes=out.axes,
                data_type="numeric",
                data_range=(float("-inf") if self.data_min is None else self.data_min, float("inf") if self.data_max is None else self.data_max),
                shape=out.shape,
            )
            for out in outputs
        )
