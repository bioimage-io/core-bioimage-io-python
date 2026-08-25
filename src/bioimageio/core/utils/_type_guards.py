"""use these type guards with caution!
They widen the type to T[Any], which is not always correct."""

from typing import Any

from typing_extensions import TypeGuard

from bioimageio.spec._internal import type_guards

is_dict = type_guards.is_dict
is_kwargs = type_guards.is_kwargs
is_list = type_guards.is_list
is_mapping = type_guards.is_mapping
is_ndarray = type_guards.is_ndarray
is_sequence = type_guards.is_sequence
is_tuple = type_guards.is_tuple


def is_list_of_string(v: Any) -> TypeGuard[list[str]]:
    """to avoid List[Unknown]"""
    return is_list(v) and all(isinstance(x, str) for x in v)
