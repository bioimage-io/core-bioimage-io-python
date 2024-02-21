from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
import torch
from numpy.testing import assert_array_almost_equal
from typing_extensions import Any, assert_never

from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.model.v0_5 import Version

from .utils import load_model


# FIXME: remove Any
def _check_predictions(
    model: Any, scripted_model: Any, model_spec: "v0_4.ModelDescr | v0_5.ModelDescr", input_data: Sequence[torch.Tensor]
):
    def _check(input_: Sequence[torch.Tensor]) -> None:
        expected_tensors = model(*input_)
        if isinstance(expected_tensors, torch.Tensor):
            expected_tensors = [expected_tensors]
        expected_outputs: List[np.ndarray[Any, Any]] = [out.numpy() for out in expected_tensors]

        output_tensors = scripted_model(*input_)
        if isinstance(output_tensors, torch.Tensor):
            output_tensors = [output_tensors]
        outputs: List[np.ndarray[Any, Any]] = [out.numpy() for out in output_tensors]

        try:
            for exp, out in zip(expected_outputs, outputs):
                assert_array_almost_equal(exp, out, decimal=4)
        except AssertionError as e:
            raise ValueError(f"Results before and after weights conversion do not agree:\n {str(e)}")

    _check(input_data)

    if len(model_spec.inputs) > 1:
        return  # FIXME: why don't we check multiple inputs?

    input_descr = model_spec.inputs[0]
    if isinstance(input_descr, v0_4.InputTensorDescr):
        if not isinstance(input_descr.shape, v0_4.ParametrizedInputShape):
            return
        min_shape = input_descr.shape.min
        step = input_descr.shape.step
    else:
        min_shape: List[int] = []
        step: List[int] = []
        for axis in input_descr.axes:
            if isinstance(axis.size, v0_5.ParameterizedSize):
                min_shape.append(axis.size.min)
                step.append(axis.size.step)
            elif isinstance(axis.size, int):
                min_shape.append(axis.size)
                step.append(0)
            elif isinstance(axis.size, (v0_5.AxisId, v0_5.TensorAxisId, type(None))):
                raise NotImplementedError(f"Can't verify inputs that don't specify their shape fully: {axis}")
            elif isinstance(axis.size, v0_5.SizeReference):
                raise NotImplementedError(f"Can't handle axes like '{axis}' yet")
            else:
                assert_never(axis.size)

    half_step = [st // 2 for st in step]
    max_steps = 4

    # check that input and output agree for decreasing input sizes
    for step_factor in range(1, max_steps + 1):
        slice_ = tuple(slice(None) if st == 0 else slice(step_factor * st, -step_factor * st) for st in half_step)
        this_input = [inp[slice_] for inp in input_data]
        this_shape = this_input[0].shape
        if any(tsh < msh for tsh, msh in zip(this_shape, min_shape)):
            raise ValueError(f"Mismatched shapes: {this_shape}. Expected at least {min_shape}")
        _check(this_input)


def convert_weights_to_torchscript(
    model_descr: Union[v0_4.ModelDescr, v0_5.ModelDescr], output_path: Path, use_tracing: bool = True
) -> v0_5.TorchscriptWeightsDescr:
    """Convert model weights from format 'pytorch_state_dict' to 'torchscript'.

    Args:
        model_descr: location of the resource for the input bioimageio model
        output_path: where to save the torchscript weights
        use_tracing: whether to use tracing or scripting to export the torchscript format
    """

    state_dict_weights_descr = model_descr.weights.pytorch_state_dict
    if state_dict_weights_descr is None:
        raise ValueError("The provided model does not have weights in the pytorch state dict format")

    input_data = model_descr.get_input_test_arrays()

    with torch.no_grad():
        input_data = [torch.from_numpy(inp.astype("float32")) for inp in input_data]

        model = load_model(state_dict_weights_descr)

        # FIXME: remove Any
        if use_tracing:
            scripted_model: Any = torch.jit.trace(model, input_data)
        else:
            scripted_model: Any = torch.jit.script(model)

        _check_predictions(model=model, scripted_model=scripted_model, model_spec=model_descr, input_data=input_data)

    # save the torchscript model
    scripted_model.save(str(output_path))  # does not support Path, so need to cast to str

    return v0_5.TorchscriptWeightsDescr(
        source=output_path, pytorch_version=Version(torch.__version__), parent="pytorch_state_dict"
    )
