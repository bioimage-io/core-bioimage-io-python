"""use these type guards with caution!
They widen the type to T[Any], which is not always correct."""

from __future__ import annotations

from typing import Any

from typing_extensions import TypeGuard

from bioimageio.spec._internal.type_guards import is_dict as is_dict
from bioimageio.spec._internal.type_guards import is_kwargs as is_kwargs
from bioimageio.spec._internal.type_guards import is_list as is_list
from bioimageio.spec._internal.type_guards import is_mapping as is_mapping
from bioimageio.spec._internal.type_guards import is_ndarray as is_ndarray
from bioimageio.spec._internal.type_guards import is_sequence as is_sequence
from bioimageio.spec._internal.type_guards import is_tuple as is_tuple


def is_list_of_string(v: Any) -> TypeGuard[list[str]]:
    """to avoid List[Unknown]"""
    return is_list(v) and all(isinstance(x, str) for x in v)
