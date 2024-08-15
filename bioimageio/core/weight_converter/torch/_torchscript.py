# type: ignore  # TODO: type
from __future__ import annotations
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
from numpy.testing import assert_array_almost_equal
from torch.jit import ScriptModule
from typing_extensions import Any, assert_never

from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.model.v0_5 import Version

from ._utils import load_torch_model

try:
    import torch
except ImportError:
    torch = None


def _check_predictions(
    model: Any,
    scripted_model: Any,
    model_spec: v0_4.ModelDescr | v0_5.ModelDescr,
    input_data: Sequence[torch.Tensor],
):
    assert torch is not None

    def _check(input_: Sequence[torch.Tensor]) -> None:
        expected_tensors = model(*input_)
        if isinstance(expected_tensors, torch.Tensor):
            expected_tensors = [expected_tensors]
        expected_outputs: List[np.ndarray[Any, Any]] = [
            out.numpy() for out in expected_tensors
        ]

        output_tensors = scripted_model(*input_)
        if isinstance(output_tensors, torch.Tensor):
            output_tensors = [output_tensors]
        outputs: List[np.ndarray[Any, Any]] = [out.numpy() for out in output_tensors]

        try:
            for exp, out in zip(expected_outputs, outputs):
                assert_array_almost_equal(exp, out, decimal=4)
        except AssertionError as e:
            raise ValueError(
                f"Results before and after weights conversion do not agree:\n {str(e)}"
            )

    _check(input_data)

    if len(model_spec.inputs) > 1:
        return  # FIXME: why don't we check multiple inputs?

    input_descr = model_spec.inputs[0]
    if isinstance(input_descr, v0_4.InputTensorDescr):
        if not isinstance(input_descr.shape, v0_4.ParameterizedInputShape):
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
            elif axis.size is None:
                raise NotImplementedError(
                    f"Can't verify inputs that don't specify their shape fully: {axis}"
                )
            elif isinstance(axis.size, v0_5.SizeReference):
                raise NotImplementedError(f"Can't handle axes like '{axis}' yet")
            else:
                assert_never(axis.size)

    input_data = input_data[0]
    max_shape = input_data.shape
    max_steps = 4

    # check that input and output agree for decreasing input sizes
    for step_factor in range(1, max_steps + 1):
        slice_ = tuple(
            (
                slice(None)
                if step_dim == 0
                else slice(0, max_dim - step_factor * step_dim, 1)
            )
            for max_dim, step_dim in zip(max_shape, step)
        )
        sliced_input = input_data[slice_]
        if any(
            sliced_dim < min_dim
            for sliced_dim, min_dim in zip(sliced_input.shape, min_shape)
        ):
            return
        _check([sliced_input])


def convert_weights_to_torchscript(
    model_descr: Union[v0_4.ModelDescr, v0_5.ModelDescr],
    output_path: Path,
    use_tracing: bool = True,
) -> v0_5.TorchscriptWeightsDescr:
    """Convert model weights from format 'pytorch_state_dict' to 'torchscript'.

    Args:
        model_descr: location of the resource for the input bioimageio model
        output_path: where to save the torchscript weights
        use_tracing: whether to use tracing or scripting to export the torchscript format
    """
    state_dict_weights_descr = model_descr.weights.pytorch_state_dict
    if state_dict_weights_descr is None:
        raise ValueError(
            "The provided model does not have weights in the pytorch state dict format"
        )

    input_data = model_descr.get_input_test_arrays()

    with torch.no_grad():
        input_data = [torch.from_numpy(inp.astype("float32")) for inp in input_data]
        model = load_torch_model(state_dict_weights_descr)
        scripted_module: ScriptModule = (
            torch.jit.trace(model, input_data)
            if use_tracing
            else torch.jit.script(model)
        )
        _check_predictions(
            model=model,
            scripted_model=scripted_module,
            model_spec=model_descr,
            input_data=input_data,
        )

    scripted_module.save(str(output_path))

    return v0_5.TorchscriptWeightsDescr(
        source=output_path,
        pytorch_version=Version(torch.__version__),
        parent="pytorch_state_dict",
    )
