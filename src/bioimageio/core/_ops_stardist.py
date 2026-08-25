from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    cast,
)

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self, TypeVar

from bioimageio.spec.model import v0_5

from ._op_base import SamplewiseOperator
from .axis import AxisId
from .common import MemberId
from .sample import Sample
from .stat_measures import (
    Measure,
)
from .tensor import Tensor

NdTuple = TypeVar("NdTuple", tuple[int, int], tuple[int, int, int])
NdBorder = TypeVar(
    "NdBorder",
    tuple[tuple[int, int], tuple[int, int]],
    tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
)


@dataclass
class _StardistPostprocessingBase(SamplewiseOperator, ABC, Generic[NdTuple, NdBorder]):
    prob_dist_input_id: MemberId
    instance_labels_output_id: MemberId

    grid: NdTuple
    """Grid size of network predictions."""

    prob_threshold: float
    """Object probability threshold for non-maximum suppression."""

    nms_threshold: float
    """The IoU threshold for non-maximum suppression."""

    b: int | NdBorder
    """Border region in which object probability is set to zero."""

    n_rays: int
    """Number of radial lines (rays) cast from the center of an object to its boundary."""

    @property
    def required_measures(self) -> Collection[Measure]:
        return set()

    def __call__(self, sample: Sample) -> None:
        prob_dist = sample.members[self.prob_dist_input_id]

        assert AxisId("channel") in prob_dist.dims, (
            "expected 'channel' axis in stardist probability/distance input"
        )
        allowed_spatial = tuple(
            map(AxisId, ("y", "x") if len(self.grid) == 2 else ("z", "y", "x"))
        )
        assert all(
            a in allowed_spatial or a in (AxisId("batch"), AxisId("channel"))
            for a in prob_dist.dims
        ), (
            f"expected prob_dist to have only 'batch', 'channel', and spatial axes {allowed_spatial}, but got {prob_dist.dims}"
        )

        spatial_shape = tuple(
            prob_dist.tagged_shape[a] * g for a, g in zip(allowed_spatial, self.grid)
        )
        if len(spatial_shape) != len(self.grid):
            raise ValueError(
                f"expected {len(self.grid)} spatial dimensions in prob_dist tensor, but got {len(spatial_shape)}"
            )
        else:
            spatial_shape = cast(NdTuple, spatial_shape)

        prob_dist = prob_dist.transpose(
            (AxisId("batch"), *allowed_spatial, AxisId("channel"))
        )
        labels: list[NDArray[Any]] = []
        for batch_idx in range(prob_dist.sizes[AxisId("batch")]):
            prob = prob_dist[
                {AxisId("batch"): batch_idx, AxisId("channel"): 0}
            ].to_numpy()
            dist = prob_dist[
                {AxisId("batch"): batch_idx, AxisId("channel"): slice(1, None)}
            ].to_numpy()

            labels_i = self._impl(prob, dist, spatial_shape)
            assert labels_i.shape == spatial_shape, (
                f"expected label image shape {spatial_shape}, but got {labels_i.shape}"
            )
            labels.append(labels_i)

        instance_labels = Tensor(
            np.stack(labels)[..., None],
            dims=(AxisId("batch"), *allowed_spatial, AxisId("channel")),
        )
        sample.members[self.instance_labels_output_id] = instance_labels

    @abstractmethod
    def _impl(
        self, prob: NDArray[Any], dist: NDArray[Any], spatial_shape: NdTuple
    ) -> NDArray[np.int32]:
        raise NotImplementedError


@dataclass
class StardistPostprocessing2D(
    _StardistPostprocessingBase[
        tuple[int, int], tuple[tuple[int, int], tuple[int, int]]
    ]
):
    def _impl(
        self, prob: NDArray[Any], dist: NDArray[Any], spatial_shape: tuple[int, int]
    ) -> NDArray[np.int32]:
        from stardist import (
            non_maximum_suppression,  # pyright: ignore[reportUnknownVariableType]
            polygons_to_label,  # pyright: ignore[reportUnknownVariableType]
        )

        points, probi, disti = non_maximum_suppression(  # pyright: ignore[reportUnknownVariableType]
            dist,
            prob,
            grid=self.grid,
            prob_thresh=self.prob_threshold,
            nms_thresh=self.nms_threshold,
            b=self.b,  # pyright: ignore[reportArgumentType]
        )

        return polygons_to_label(disti, points, prob=probi, shape=spatial_shape)

    @classmethod
    def from_proc_descr(
        cls, descr: v0_5.StardistPostprocessingDescr, member_id: MemberId
    ) -> Self:
        if not isinstance(descr.kwargs, v0_5.StardistPostprocessingKwargs2D):
            raise TypeError(
                f"expected v0_5.StardistPostprocessingKwargs2D for 2D stardist post-processing, but got {type(descr.kwargs)}"
            )

        kwargs = descr.kwargs
        return cls(
            prob_dist_input_id=member_id,
            instance_labels_output_id=member_id,
            grid=kwargs.grid,
            prob_threshold=kwargs.prob_threshold,
            nms_threshold=kwargs.nms_threshold,
            b=kwargs.b,
            n_rays=kwargs.n_rays,
        )


@dataclass
class StardistPostprocessing3D(
    _StardistPostprocessingBase[
        tuple[int, int, int], tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    ]
):
    anisotropy: tuple[float, float, float]
    """Anisotropy factors for 3D star-convex polyhedra, i.e. the physical pixel size along each spatial axis."""

    overlap_label: int | None = None
    """Optional label to apply to any area of overlapping predicted objects."""

    def _impl(
        self,
        prob: NDArray[Any],
        dist: NDArray[Any],
        spatial_shape: tuple[int, int, int],
    ) -> NDArray[np.int32]:
        from stardist import (
            Rays_GoldenSpiral,
            non_maximum_suppression_3d,  # pyright: ignore[reportUnknownVariableType]
            polyhedron_to_label,  # pyright: ignore[reportUnknownVariableType]
        )
        from stardist.matching import (
            relabel_sequential,  # pyright: ignore[reportUnknownVariableType]
        )

        rays = Rays_GoldenSpiral(self.n_rays, anisotropy=self.anisotropy)

        points, probi, disti = non_maximum_suppression_3d(  # pyright: ignore[reportUnknownVariableType]
            dist,
            prob,
            rays,
            grid=self.grid,
            prob_thresh=self.prob_threshold,
            nms_thresh=self.nms_threshold,
            b=self.b,  # pyright: ignore[reportArgumentType]
        )

        labels = polyhedron_to_label(  # pyright: ignore[reportUnknownVariableType]
            disti,
            points,
            rays=rays,
            prob=probi,
            shape=spatial_shape,
            overlap_label=self.overlap_label,
        )

        labels, _, _ = relabel_sequential(labels)
        assert isinstance(labels, np.ndarray) and labels.dtype == np.int32
        return labels

    @classmethod
    def from_proc_descr(
        cls, descr: v0_5.StardistPostprocessingDescr, member_id: MemberId
    ) -> Self:
        if not isinstance(descr.kwargs, v0_5.StardistPostprocessingKwargs3D):
            raise TypeError(
                f"expected v0_5.StardistPostprocessingKwargs3D for 3D stardist post-processing, but got {type(descr.kwargs)}"
            )

        kwargs = descr.kwargs
        return cls(
            prob_dist_input_id=member_id,
            instance_labels_output_id=member_id,
            grid=kwargs.grid,
            prob_threshold=kwargs.prob_threshold,
            nms_threshold=kwargs.nms_threshold,
            n_rays=kwargs.n_rays,
            anisotropy=kwargs.anisotropy,
            b=kwargs.b,
            overlap_label=kwargs.overlap_label,
        )
