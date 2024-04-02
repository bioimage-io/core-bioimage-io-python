from __future__ import annotations

from typing import Literal, NamedTuple, Tuple, TypeVar, Union

from typing_extensions import Self, assert_never

DTypeStr = Literal[
    "bool",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]


_LeftRight_T = TypeVar("_LeftRight_T", bound="_LeftRight")
_LeftRightLike = Union[int, Tuple[int, int], _LeftRight_T]


class _LeftRight(NamedTuple):
    left: int
    right: int

    @classmethod
    def create(cls, like: _LeftRightLike[Self]) -> Self:
        if isinstance(like, cls):
            return like
        elif isinstance(like, tuple):
            return cls(*like)
        elif isinstance(like, int):
            return cls(like, like)
        else:
            assert_never(like)


_Where = Literal["left", "right", "left_and_right"]


class CropWidth(_LeftRight):
    pass


CropWidthLike = _LeftRightLike[CropWidth]
CropWhere = _Where


class Halo(_LeftRight):
    pass


HaloLike = _LeftRightLike[Halo]


class OverlapWidth(_LeftRight):
    pass


class PadWidth(_LeftRight):
    pass


PadWidthLike = _LeftRightLike[PadWidth]
PadMode = Literal["edge", "reflect", "symmetric"]
PadWhere = _Where


class SliceInfo(NamedTuple):
    start: int
    stop: int


BlockNumber = int
TotalNumberOfBlocks = int
