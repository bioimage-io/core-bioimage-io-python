from typing import List, Sequence
from typing_extensions import Any, assert_never
from pathlib import Path
from typing import Union

import numpy as np
import torch
from numpy.testing import assert_array_almost_equal

from bioimageio.spec import load_description
from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec import load_description
from bioimageio.spec.common import InvalidDescription
from bioimageio.spec.utils import download

from .utils import load_model

# FIXME: remove Any
def _check_predictions(model: Any, scripted_model: Any, model_spec: "v0_4.ModelDescr | v0_5.ModelDescr", input_data: Sequence[torch.Tensor]):
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
        return # FIXME: why don't we check multiple inputs?

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
            elif isinstance(axis.size, v0_5.SizeReference): # pyright: ignore [reportUnnecessaryIsInstance]
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
    model_spec: Union[str, Path, v0_4.ModelDescr, v0_5.ModelDescr], output_path: Path, use_tracing: bool = True
):
    """Convert model weights from format 'pytorch_state_dict' to 'torchscript'.

    Args:
        model_spec: location of the resource for the input bioimageio model
        output_path: where to save the torchscript weights
        use_tracing: whether to use tracing or scripting to export the torchscript format
    """
    if isinstance(model_spec, (str, Path)):
        loaded_spec = load_description(Path(model_spec))
        if isinstance(loaded_spec, InvalidDescription):
            raise ValueError(f"Bad resource description: {loaded_spec}")
        if not isinstance(loaded_spec, (v0_4.ModelDescr, v0_5.ModelDescr)):
            raise TypeError(f"Path {model_spec} is a {loaded_spec.__class__.__name__}, expected a v0_4.ModelDescr or v0_5.ModelDescr")
        model_spec = loaded_spec

    state_dict_weights_descr = model_spec.weights.pytorch_state_dict
    if state_dict_weights_descr is None:
        raise ValueError(f"The provided model does not have weights in the pytorch state dict format")

    with torch.no_grad():
        if isinstance(model_spec, v0_4.ModelDescr):
            downloaded_test_inputs = [download(inp) for inp in model_spec.test_inputs]
        else:
            downloaded_test_inputs = [inp.test_tensor.download() for inp in model_spec.inputs]

        input_data = [np.load(dl.path).astype("float32") for dl in downloaded_test_inputs]
        input_data = [torch.from_numpy(inp) for inp in input_data]

        model = load_model(state_dict_weights_descr)

        # FIXME: remove Any
        if use_tracing:
            scripted_model: Any = torch.jit.trace(model, input_data)
        else:
            scripted_model: Any = torch.jit.script(model)

        ret = _check_predictions(
            model=model,
            scripted_model=scripted_model,
            model_spec=model_spec,
            input_data=input_data
        )

    # save the torchscript model
    scripted_model.save(str(output_path))  # does not support Path, so need to cast to str
    return ret
