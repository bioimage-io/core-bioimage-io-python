from typing import List

from bioimageio.core.common import Tensor
from bioimageio.spec.model import AnyModelDescr, v0_4
from bioimageio.spec.utils import load_array

from .image_helper import interprete_array


def get_test_inputs(model: AnyModelDescr) -> List[Tensor]:
    axes = [d.axes for d in model.inputs]
    if isinstance(model, v0_4.ModelDescr):
        arrays = [load_array(tt) for tt in model.test_inputs]
    else:
        arrays = [load_array(d.test_tensor) for d in model.inputs]

    return [interprete_array(arr, ax) for arr, ax in zip(arrays, axes)]


def get_test_outputs(model: AnyModelDescr) -> List[Tensor]:
    axes = [d.axes for d in model.outputs]
    if isinstance(model, v0_4.ModelDescr):
        arrays = [load_array(tt) for tt in model.test_outputs]
    else:
        arrays = [load_array(d.test_tensor) for d in model.outputs]

    return [interprete_array(arr, ax) for arr, ax in zip(arrays, axes)]
