from typing import List, Optional, Sequence, Tuple

from pybio.core.array import PyBioArray
from pybio.core.transformations import ApplyToAll, Transformation
from pybio.spec.node import InputArray, OutputArray


class NormalizeZeroMeanUnitVariance(Transformation):
    """ Sigmoid activation
    """

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

    def dynamic_outputs(self, inputs: Tuple[InputArray]) -> Tuple[OutputArray]:
        return tuple(
            OutputArray(
                name=ipt.name,
                axes=ipt.axes,
                data_type=ipt.data_type,
                data_range=(float("-inf"), float("inf")),
                shape=ipt.shape,
                halo=(0,) * len(ipt.shape),
            )
            for ipt in inputs
        )

    def dynamic_inputs(self, outputs: Tuple[OutputArray]) -> Tuple[InputArray]:
        return tuple(
            InputArray(
                name=out.name,
                axes=out.axes,
                data_type="numeric",
                data_range=(float("-inf"), float("inf")),
                shape=out.shape,
            )
            for out in outputs
        )
