from typing import Any, TypeAlias, TypeVar, Union

_T = TypeVar("_T")
NestedList: TypeAlias = Union[list["NestedList[_T]"], _T]


def cast_and_assert(typ: type[_T], x: Any) -> _T:
    assert isinstance(x, typ), (x, typ)
    return x
