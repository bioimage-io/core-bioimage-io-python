from collections import defaultdict
from typing import DefaultDict, Dict, Optional, Tuple

from bioimageio.spec.model.v0_5 import (
    InputAxis,
    ModelDescr,
    ParameterizedSize,
    SizeReference,
)
from torch.export import Dim
from typing_extensions import assert_never


def get_dynamic_shapes(model_descr: ModelDescr):
    dynamic_shapes: DefaultDict[str, Dict[int, Optional[Dim]]] = defaultdict(dict)
    potential_ref_axes: Dict[str, Tuple[InputAxis, int]] = {}
    # add dynamic dims from parameterized input sizes (and fixed sizes as None)
    for d in model_descr.inputs:
        for i, ax in enumerate(d.axes):
            dim_name = f"{d.id}_{ax.id}"
            if isinstance(ax.size, int):
                dim = None  # fixed size (could also be left out)
            elif ax.size is None:
                dim = Dim(dim_name, min=1)
            elif isinstance(ax.size, ParameterizedSize):
                dim = Dim(dim_name, min=ax.size.min)
            elif isinstance(ax.size, SizeReference):
                continue  # handled below
            else:
                assert_never(ax.size)

            dynamic_shapes[str(d.id)][i] = dim
            potential_ref_axes[dim_name] = (ax, i)

    # add dynamic dims from size references
    for d in model_descr.inputs:
        for i, ax in enumerate(d.axes):
            if not isinstance(ax.size, SizeReference):
                continue  # handled above

            dim_name_ref = f"{ax.size.tensor_id}_{ax.size.axis_id}"
            ax_ref, i_ref = potential_ref_axes[dim_name_ref]
            a = ax_ref.scale / ax.scale
            b = ax.size.offset
            dim_ref = dynamic_shapes[str(ax.size.tensor_id)][i_ref]
            dim = a * dim_ref + b if dim_ref is not None else None
            dynamic_shapes[str(d.id)][i] = dim

    return dict(dynamic_shapes)
