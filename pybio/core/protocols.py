from __future__ import annotations

from typing import Any, Callable, Sequence

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class Tensor(Protocol):
    shape: Sequence[int, ...]
    dtype: Any
    reshape: Callable[[Sequence[int, ...]], Tensor]
