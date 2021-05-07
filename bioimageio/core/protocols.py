from __future__ import annotations

from typing import Any, Callable, Sequence, Union

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class Scalar(Protocol):
    item: Callable[[], float]


@runtime_checkable
class Tensor(Protocol):
    shape: Sequence[int, ...]
    dtype: Any
    reshape: Callable[[Sequence[int, ...]], Tensor]

    mean: Callable[[], Scalar]
    std: Callable[[], Scalar]

    __add__: Callable[[Union[float, Tensor]], Tensor]
    __sub__: Callable[[Union[float, Tensor]], Tensor]
    __mul__: Callable[[Union[float, Tensor]], Tensor]
    __truediv__: Callable[[Union[float, Tensor]], Tensor]
