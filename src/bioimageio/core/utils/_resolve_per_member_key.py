from __future__ import annotations

from collections.abc import Collection
from typing import TypeVar

from typing_extensions import assert_never

from ..axis import AxisId, PerAxis
from ..common import MemberId, PerMember

T_pm = TypeVar("T_pm")
T_pa = TypeVar("T_pa")


def resolve_per_member_var(
    var: PerMember[T_pm | PerAxis[T_pa]] | PerAxis[T_pa],
    /,
    *,
    valid_member_index_types: tuple[type[T_pm], ...],
    valid_axis_index_types: tuple[type[T_pa], ...],
    members: Collection[MemberId],
):
    """Resolve a key that may be a per-member mapping or a per-axis mapping into a per-member mapping.

    If the key is a per-axis mapping, it will be expanded into a per-member mapping
    with the same per-axis mapping for each member in `members`.
    """

    per_member_key: PerMember[T_pm | PerAxis[T_pa]] = {}
    per_axis_key: PerAxis[T_pa] = {}
    for k, v in var.items():
        if isinstance(k, MemberId):
            if not isinstance(v, valid_member_index_types):
                raise TypeError(
                    f"Per-member indexing only supports value types {valid_member_index_types}, but got {v} of type {type(v)})"
                )
            per_member_key[k] = v
        elif isinstance(k, AxisId):
            if not isinstance(v, valid_axis_index_types):
                raise TypeError(
                    f"Per-axis indexing only supports value types {valid_axis_index_types}, but got {v} of type {type(v)})"
                )
            per_axis_key[k] = v
        else:
            assert_never(k)

    if per_member_key and per_axis_key:
        raise ValueError(
            f"Indexing a sample with both MemberId and AxisId keys is not supported, got keys: {list(per_member_key)} and {list(per_axis_key)}"
        )
    elif per_axis_key:
        per_member_key = {m: per_axis_key for m in members}

    return per_member_key
