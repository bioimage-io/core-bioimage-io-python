from dataclasses import dataclass
from typing import Any, Collection

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label as connected_components
from typing_extensions import Literal

from bioimageio.spec.model.v0_5 import MicroSamWatershedDescr

from ._op_base import SamplewiseOperator
from .axis import AxisId, PerAxis
from .common import MemberId
from .sample import Sample
from .stat_measures import Measure
from .tensor import Tensor


@dataclass
class MicroSamWatershed(SamplewiseOperator):
    """micro-sam instance segmentation postprocessing operator.

    Replaces the three dense maps predicted by a micro-sam instance segmentation
    (AIS) decoder -- foreground probability, center distance and inverted
    boundary distance -- with instance labels (0 = background).

    Seeds are connected components where both smoothed distance predictions are
    below their thresholds, intersected with the foreground mask; the seeded
    watershed then runs on the smoothed boundary distances restricted to that
    mask. This is a port of
    `micro_sam.instance_segmentation.InstanceSegmentationWithDecoder` /
    `torch_em.util.segmentation.watershed_from_center_and_boundary_distances`:

    - Anna Archit et al. [*Segment Anything for Microscopy*](https://www.nature.com/articles/s41592-024-02580-4).
      Nature Methods, 2025.
    """

    center_distance_threshold: float = 0.5
    boundary_distance_threshold: float = 0.5
    foreground_threshold: float = 0.5
    foreground_smoothing: float = 1.0
    """Sigma of the gaussian smoothing applied to the foreground prediction. Set to 0 to disable smoothing."""
    distance_smoothing: float = 1.6
    """Sigma of the gaussian smoothing applied to both distance predictions. Set to 0 to disable smoothing."""
    min_size: int = 0
    """Minimum size of objects to keep, in pixels. Set to 0 to disable filtering by size."""
    labels_id: MemberId = MemberId("labels")
    output_dtype: Literal["uint16", "uint32"] = "uint16"

    @classmethod
    def from_proc_descr(
        cls, proc_descr: MicroSamWatershedDescr, member_id: MemberId
    ) -> "MicroSamWatershed":
        kwargs = proc_descr.kwargs
        return cls(
            labels_id=member_id,
            center_distance_threshold=kwargs.center_distance_threshold,
            boundary_distance_threshold=kwargs.boundary_distance_threshold,
            foreground_threshold=kwargs.foreground_threshold,
            foreground_smoothing=kwargs.foreground_smoothing,
            distance_smoothing=kwargs.distance_smoothing,
            min_size=kwargs.min_size,
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
                "Expected first axis to be 'batch' for micro-sam watershed."
            )

        if x.dims[1] != AxisId("channel"):
            raise ValueError(
                "Expected second axis to be 'channel' with 3 channels for micro-sam watershed."
            )
        if x.shape[1] != 3:
            raise ValueError(
                "Expected 3 stacked tensors along the 'channel' axis: foreground,"
                + " center_distances, boundary_distances for micro-sam watershed."
            )

        masks = [self._apply_impl(xx) for xx in x]
        return Tensor.from_numpy(np.stack(masks, axis=0), dims=x.dims)

    def _apply_impl(self, x: Tensor) -> NDArray[Any]:
        """apply on a tensor without batch dimension"""
        try:
            from skimage.segmentation import (  # pyright: ignore[reportMissingTypeStubs]
                watershed,  # pyright: ignore[reportUnknownVariableType]
            )
        except ImportError as e:
            raise ImportError(
                "scikit-image is required for microsam_watershed. Install with: pip install bioimageio.core[microsam]"
            ) from e

        foreground, center_distances, boundary_distances = x.to_numpy()

        if self.foreground_smoothing > 0:
            foreground = gaussian_filter(foreground, self.foreground_smoothing)
        if self.distance_smoothing > 0:
            center_distances = gaussian_filter(center_distances, self.distance_smoothing)
            boundary_distances = gaussian_filter(
                boundary_distances, self.distance_smoothing
            )

        fg_mask = foreground > self.foreground_threshold
        marker_map = np.logical_and(
            center_distances < self.center_distance_threshold,
            boundary_distances < self.boundary_distance_threshold,
        )
        marker_map[~fg_mask] = False
        markers, _ = connected_components(marker_map)

        mask = np.asarray(watershed(boundary_distances, markers=markers, mask=fg_mask))

        if self.min_size > 0:
            ids, sizes = np.unique(mask, return_counts=True)
            too_small = ids[(sizes < self.min_size) & (ids != 0)]
            if too_small.size:
                mask[np.isin(mask, too_small)] = 0

        # add singleton channel axis for output to keep dims consistent with postprocessing input
        mask = mask[None]
        return mask.astype(np.dtype(self.output_dtype))
