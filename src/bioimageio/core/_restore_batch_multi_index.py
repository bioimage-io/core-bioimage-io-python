from __future__ import annotations

import pandas as pd

from .axis import AxisId
from .common import PerMember
from .tensor import Tensor


def restore_batch_multi_index(
    inputs: PerMember[Tensor | None], outputs: PerMember[Tensor | None]
) -> PerMember[Tensor | None]:
    """Restore the first batch multi-index found in the inputs to all outputs with batch dimension."""
    for tensor in inputs.values():
        if tensor is None:
            continue

        idx = tensor.data.indexes.get(AxisId("batch"))  # pyright: ignore[reportUnknownVariableType]
        if isinstance(idx, pd.MultiIndex):
            outputs = {
                k: v.assign_batch_multi_index(idx)
                if v is not None and AxisId("batch") in v.dims
                else v
                for k, v in outputs.items()
            }
            break

    return outputs
