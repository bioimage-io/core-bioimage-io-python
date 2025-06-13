"""use these type guards with caution!
They widen the type to T[Any], which is not always correct."""

from bioimageio.spec._internal import type_guards

is_list = type_guards.is_list
is_ndarray = type_guards.is_ndarray
is_tuple = type_guards.is_tuple
