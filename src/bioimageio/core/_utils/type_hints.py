from typing import Any, TypeVar, Union

from typing_extensions import TypeAliasType

_T = TypeVar("_T")
NestedList = TypeAliasType(
    "NestedList", Union[list["NestedList[_T]"], _T], type_params=(_T,)
)


def cast_and_assert(typ: type[_T], x: Any) -> _T:
    assert isinstance(x, typ), (x, typ)
    return x
