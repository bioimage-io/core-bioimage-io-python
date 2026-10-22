from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any, TypeVar, cast

from ..axis import AxisId, PerAxis
from ..common import MemberId, PerMember
from .type_guards import is_mapping

T_pm = TypeVar("T_pm")
T_pa = TypeVar("T_pa")


def resolve_per_member_var(
    var: PerMember[T_pm | PerAxis[T_pa]] | T_pm | PerAxis[T_pa],
    /,
    *,
    valid_per_member_types: tuple[type[T_pm], ...],
    valid_per_axis_types: tuple[type[T_pa], ...],
    members: Collection[MemberId],
):
    """Resolve a key that may be a per-member mapping or a per-axis mapping into a per-member mapping.

    If the key is a per-axis mapping, it will be expanded into a per-member mapping
    with the same per-axis mapping for each member in `members`.
    """
    if not isinstance(var, Mapping):
        var = {m: cast(Any, var) for m in members}

    var = cast(Mapping[Any, Any], var)
    per_member_key: PerMember[T_pm | PerAxis[T_pa]] = {}
    per_axis_key: PerAxis[T_pa] = {}
    for k, v in var.items():
        if isinstance(k, MemberId):
            if valid_per_axis_types and is_mapping(v):
                if not all(isinstance(ak, AxisId) for ak in v):
                    raise TypeError(
                        f"Per-axis mappings only support keys of type AxisId, but got {list(v)}"
                    )
                if not all(isinstance(av, valid_per_axis_types) for av in v.values()):
                    raise TypeError(
                        f"Per-axis mappings only support value types {valid_per_axis_types}, but got {list(v.values())}"
                    )
            elif not isinstance(v, valid_per_member_types):
                raise TypeError(
                    f"Per-member keys only support value types {valid_per_member_types}"
                    + f", but got {v} of type {type(v)})"
                )
            per_member_key[k] = v
        elif isinstance(k, AxisId):
            if not isinstance(v, valid_per_axis_types):
                raise TypeError(
                    f"Per-axis keys only supports value types {valid_per_axis_types}, but got {v} of type {type(v)})"
                )
            per_axis_key[k] = v
        else:
            raise TypeError(
                "Only keys of type MemberId or AxisId are supported,"
                + f" but got {k} of type {type(k)}"
            )

    if per_member_key and per_axis_key:
        raise ValueError(
            "Mixing both MemberId and AxisId keys in the same mapping is not supported,"
            + f" got member keys: {list(per_member_key)} and axis keys: {list(per_axis_key)}"
        )
    elif per_axis_key:
        per_member_key = {m: per_axis_key for m in members}

    return per_member_key
