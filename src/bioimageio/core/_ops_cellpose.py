from dataclasses import dataclass
from typing import Any, Collection

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Literal, cast

from bioimageio.core.sample import Sample
from bioimageio.spec.model.v0_5 import CellposeFlowDynamicsDescr

from ._op_base import SamplewiseOperator
from .axis import AxisId, PerAxis
from .common import MemberId
from .stat_measures import Measure
from .tensor import Tensor


@dataclass
class CellposeFlowDynamics(SamplewiseOperator):
    """Cellpose flow-dynamics postprocessing operator.

    Adds `output` member to the sample, containing instance labels (int32, 0 = background)
    decoded from the flow fields and cell probability output of a Cellpose model.

    """

    cellprob_threshold: float = 0.0
    flow_threshold: float = 0.4
    do_3D: bool = False
    labels_id: MemberId = MemberId("labels")
    output_dtype: Literal["uint16", "uint32"] = "uint16"

    @classmethod
    def from_proc_descr(
        cls, proc_descr: CellposeFlowDynamicsDescr, member_id: MemberId
    ) -> "CellposeFlowDynamics":
        kwargs = proc_descr.kwargs
        return cls(
            labels_id=member_id,
            cellprob_threshold=kwargs.cellprob_threshold,
            flow_threshold=kwargs.flow_threshold,
            do_3D=kwargs.do_3D,
            output_dtype=kwargs.output_dtype,
        )

    @property
    def required_measures(self) -> Collection[Measure]:
        return set()

    def get_output_shape(self, input_shape: PerAxis[int]) -> PerAxis[int]:
        output_shape = dict(input_shape)
        output_shape[AxisId("channel")] = 1
        return output_shape

    def __call__(self, sample: Sample) -> None:
        input_tensor = sample.members[self.labels_id]
        output_tensor = self._apply(input_tensor)
        sample.members[self.labels_id] = output_tensor

    def _apply(self, x: Tensor) -> Tensor:
        if x.dims[0] != AxisId("batch"):
            raise ValueError(
                "Expected first axis to be 'batch' for cellpose flow dynamics."
            )

        if x.dims[1] != AxisId("channel"):
            raise ValueError(
                "Expected first axis to be 'channel' with 3 channels for cellpose flow dynamics."
            )
        if x.shape[1] != 3:
            raise ValueError(
                "Expected 3 stacked tensors along first 'channel' axis: flow_y, flow_x, cellprob for cellpose flow dynamics."
            )

        masks = [self._apply_impl(xx) for xx in x]
        return Tensor.from_numpy(np.stack(masks, axis=0), dims=x.dims)

    def _apply_impl(self, x: Tensor) -> NDArray[Any]:
        """apply on a tensor without batch dimension"""
        *flows, cellprob = x.to_numpy()
        try:
            from cellpose import dynamics  # pyright: ignore[reportMissingTypeStubs]
        except ImportError as e:
            raise ImportError(
                "cellpose is required for cellpose_flow_dynamics. Install with: pip install cellpose"
            ) from e

        flows = np.stack(flows, axis=0)
        result = cast(
            NDArray[Any],
            dynamics.compute_masks(
                flows,
                cellprob,
                cellprob_threshold=self.cellprob_threshold,
                flow_threshold=self.flow_threshold,
                do_3D=self.do_3D,
            ),
        )
        # add singleton channel axis for output to keep dims consistent with postprocessing input
        result = result[None]
        return result.astype(np.dtype(self.output_dtype))
