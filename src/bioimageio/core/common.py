from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import (
    Literal,
    NamedTuple,
    TypeVar,
    Union,
)

from typing_extensions import Self, TypeAlias, assert_never

from bioimageio.spec._internal.utils import SLOTS
from bioimageio.spec.model import v0_5

SupportedWeightsFormat: TypeAlias = Literal[
    "keras_hdf5",
    "keras_v3",
    "onnx",
    "pytorch_state_dict",
    "tensorflow_saved_model_bundle",
    "torchscript",
]

QuantileMethod: TypeAlias = Literal[
    "inverted_cdf",
    # "averaged_inverted_cdf",
    # "closest_observation",
    # "interpolated_inverted_cdf",
    # "hazen",
    # "weibull",
    "linear",
    # "median_unbiased",
    # "normal_unbiased",
]
"""Methods to use when the desired quantile lies between two data points.
See https://numpy.org/devdocs/reference/generated/numpy.quantile.html#numpy-quantile for details.

Note:
    Only relevant for `SampleQuantile` measures, as `DatasetQuantile` measures computed by [bioimageio.core.stat_calculators.][] are approximations (and use the "linear" method for each sample quantiles)

!!! warning
    Limited choices to map more easily to bioimageio.spec descriptions.
    Current implementations:
    - [bioimageio.spec.model.v0_5.ClipKwargs][] implies "inverted_cdf" for sample quantiles and "linear" (numpy's default) for dataset quantiles.
    - [bioimageio.spec.model.v0_5.ScaleRangeKwargs][] implies "linear" (numpy's default)

"""

DTypeStr: TypeAlias = Literal[
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
_LeftRightLike = Union[int, tuple[int, int], _LeftRight_T]


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


_Where: TypeAlias = Literal["left", "right", "left_and_right"]


class CropWidth(_LeftRight):
    pass


CropWidthLike: TypeAlias = _LeftRightLike[CropWidth]
CropWhere: TypeAlias = _Where


class Halo(_LeftRight):
    pass


HaloLike: TypeAlias = _LeftRightLike[Halo]


class OverlapWidth(_LeftRight):
    pass


class PadWidth(_LeftRight):
    pass


PadWidthLike: TypeAlias = _LeftRightLike[PadWidth]
Padding: TypeAlias = v0_5.Padding
PadMode: TypeAlias = Union[Literal["constant", "edge", "reflect", "symmetric"], Padding]
PadWhere: TypeAlias = _Where


@dataclass(frozen=True, **SLOTS)
class SliceInfo:
    start: int
    stop: int


@dataclass(frozen=True, **SLOTS)
class SliceInfoWithOverlap(SliceInfo):
    """SliceInfo aware of the fact that it overlaps with neighboring blocks."""

    start_wo_overlap: int
    stop_wo_overlap: int


SampleId = Hashable
"""ID of a sample, see `bioimageio.core.sample.Sample`"""
MemberId = v0_5.TensorId
"""ID of a `Sample` member, see `bioimageio.core.sample.Sample`"""

BlocksizeParameter: TypeAlias = Union[
    v0_5.ParameterizedSize_N,
    Mapping[tuple[MemberId, v0_5.AxisId], v0_5.ParameterizedSize_N],
]
"""
Parameter to determine a concrete size for paramtrized axis sizes defined by
`bioimageio.spec.model.v0_5.ParameterizedSize`.
"""

_T = TypeVar("_T")
# PerMember = Union[
#     Mapping[MemberId, _T], Mapping[str, _T], Mapping[Union[MemberId, str], _T]
# ]
PerMember = Mapping[MemberId, _T]
BlockLoc = tuple[int, ...]
BlockLocShape = tuple[int, ...]


class DataDepSize(NamedTuple):
    min: int
    max: int | None


def resolve_pad_mode(mode: PadMode):
    constant_value = None
    if isinstance(mode, str):
        mode_name = mode
    elif isinstance(mode, v0_5.ConstantPadding):
        mode_name = mode.mode
        constant_value = mode.value
    elif isinstance(
        mode, (v0_5.EdgePadding, v0_5.ReflectPadding, v0_5.SymmetricPadding)
    ):
        mode_name = mode.mode
    else:
        assert_never(mode)

    return mode_name, constant_value
