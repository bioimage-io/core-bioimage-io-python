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


LeftRight_T = TypeVar("LeftRight_T", bound="LeftRight")
LeftRightLike = Union[int, Tuple[int, int], LeftRight_T]


class LeftRight(NamedTuple):
    left: int
    right: int

    @classmethod
    def create(cls, like: LeftRightLike[Self]) -> Self:
        if isinstance(like, cls):
            return like
        elif isinstance(like, tuple):
            return cls(*like)
        elif isinstance(like, int):
            return cls(like, like)
        else:
            assert_never(like)


class Halo(LeftRight):
    pass


HaloLike = LeftRightLike[Halo]


class PadWidth(LeftRight):
    pass


PadWidthLike = LeftRightLike[PadWidth]
PadMode = Literal["edge", "reflect", "symmetric"]
PadWhere = Literal["before", "center", "after"]


class SliceInfo(NamedTuple):
    start: int
    stop: int


TileNumber = int
TotalNumberOfTiles = int
