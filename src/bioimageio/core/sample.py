from __future__ import annotations

import collections.abc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Literal,
)

import xarray as xr
from loguru import logger
from numpy.typing import NDArray
from typing_extensions import Self, assert_never

from ._utils.resolve_per_member_key import resolve_per_member_var
from .axis import AxisId, PerAxis
from .common import (
    MemberId,
    PadMode,
    PadWidthLike,
    PerMember,
    SampleId,
    SliceInfo,
)
from .stat_measures import Stat
from .tensor import Tensor


@dataclass
class Sample:
    """A dataset sample.

    A `Sample` has `members`, which allows to combine multiple tensors into a single
    sample.
    For example a `Sample` from a dataset with masked images may contain a
    `MemberId("raw")` and `MemberId("mask")` image.
    """

    id: SampleId
    """Identifies the `Sample` within the dataset -- typically a number or a string."""

    members: dict[MemberId, Tensor]
    """The sample's tensors"""

    stat: Stat
    """Sample and dataset statistics"""

    # unstacked_dim_order: PerMember[Sequence[AxisId] | None] | None = None
    # """The order of the fully unstacked tensor"""

    def __getitem__(
        self,
        key: PerMember[PerAxis[SliceInfo | slice | int] | Tensor | xr.DataArray]
        | PerAxis[SliceInfo | slice | int]
        | Tensor
        | xr.DataArray,
    ) -> Self:
        per_member_key = resolve_per_member_var(
            key,
            valid_per_member_types=(
                SliceInfo,
                slice,
                int,
                Tensor,
                xr.DataArray,
            ),
            valid_per_axis_types=(SliceInfo, slice, int),
            members=self.members,
        )

        return self.__class__(
            members={
                m: t[per_member_key[m]]
                for m, t in self.members.items()
                if m in per_member_key
            },
            stat=self.stat,
            id=self.id,
        )

    def __postinit__(self):
        # default tensor names to member ids
        for m, t in self.members.items():
            if t.name is None:
                t.name = str(m)

    @property
    def shape(self) -> PerMember[PerAxis[int]]:
        return {tid: t.sizes for tid, t in self.members.items()}

    def as_arrays(self) -> dict[MemberId, NDArray[Any]]:
        """Return sample as dictionary of arrays."""
        return {m: t.to_numpy() for m, t in self.members.items()}

    def pad(
        self,
        pad_width: PerMember[PerAxis[int | PadWidthLike]],
        mode: PerMember[PadMode] | PadMode,
    ) -> Self:
        """Convenience method to pad sample members."""
        default_mode = "symmetric"
        if isinstance(mode, collections.abc.Mapping):
            mode_per_member = mode
        else:
            mode_per_member: Mapping[MemberId, PadMode] = {}
            default_mode = mode

        return self.__class__(
            members={
                m: t.pad(
                    pad_width=pad_width.get(m, {}),
                    mode=mode_per_member.get(m, default_mode),
                )
                for m, t in self.members.items()
            },
            stat=self.stat,
            id=self.id,
        )

    def unstack_batch_multi_index(
        self,
        *,
        errors: Literal["raise", "warn", "ignore"] = "raise",
        unstacked_dim_order: PerMember[Sequence[AxisId]] | None = None,
    ) -> Self:
        """Unstack the batch multi-index of all sample members.

        Args:
            errors: Whether to raise an error if a member does not have a batch multi-index. Default is "raise".

        Returns:
            A new `Sample` with unstacked batch multi-index for all members.
        """
        if (
            len(
                no_batch := [
                    m for m, t in self.members.items() if AxisId("batch") not in t.dims
                ]
            )
            == len(self.members)
            and errors != "ignore"
        ):
            error_msg = f"No member has a batch dimension: {no_batch}"
            if errors == "raise":
                raise ValueError(error_msg)
            elif errors == "warn":
                logger.warning(error_msg)
            else:
                assert_never(errors)

        members = {
            m: t
            if AxisId("batch") not in t.dims
            else t.unstack_batch_multi_index(
                errors=errors,
                unstacked_dim_order=unstacked_dim_order[m]
                if unstacked_dim_order is not None
                else None,
            )
            for m, t in self.members.items()
        }
        if (
            len(
                batch_lengths := {
                    t.sizes.get(AxisId("batch"))
                    for t in members.values()
                    if AxisId("batch") in t.dims
                }
            )
            > 1
        ):
            raise ValueError(
                f"Different batch lengths after unstacking: {batch_lengths}"
            )

        stat: Stat = {  # TODO: update stat calculations to maintain any multi-index batch index.
            k: v.unstack_batch_multi_index(errors="ignore", unstacked_dim_order=None)
            if isinstance(v, Tensor)
            else float(v)
            for k, v in self.stat.items()
        }
        return self.__class__(
            members=members,
            stat=stat,
            id=self.id,
        )
