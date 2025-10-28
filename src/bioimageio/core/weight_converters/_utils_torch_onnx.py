"""helper to export both TorchScript or PytorchStateDict to ONNX"""

from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, DefaultDict, Dict, List, Literal, Tuple, Union

import torch
from bioimageio.spec.model.v0_5 import (
    FileDescr,
    InputAxis,
    ModelDescr,
    OnnxWeightsDescr,
    ParameterizedSize,
    SizeReference,
)
from loguru import logger
from typing_extensions import assert_never

from .. import __version__
from ..digest_spec import get_member_id, get_test_input_sample
from ..proc_setup import get_pre_and_postprocessing

if TYPE_CHECKING:
    import torch.jit
    from torch.export.dynamic_shapes import (
        _DimHint as DimHint,  # pyright: ignore[reportPrivateUsage]
    )


def _get_dynamic_axes_noop(model_descr: ModelDescr):
    """noop for dynamo=True which uses `get_dynamic_shapes` instead"""

    return None


def _get_dynamic_axes_impl(model_descr: ModelDescr):
    """dynamic axes for (old) onnx export with dynamo=False"""
    dynamic_axes: DefaultDict[str, Dict[int, str]] = defaultdict(dict)
    for d in chain(model_descr.inputs, model_descr.outputs):
        for i, ax in enumerate(d.axes):
            if not isinstance(ax.size, int):
                dynamic_axes[str(d.id)][i] = str(ax.id)

    return dynamic_axes


try:
    from torch.export import Dim

    STATIC_DIM = Dim.STATIC if hasattr(Dim, "STATIC") else None
    TensorDim = Union[Dim, "DimHint", None]

except Exception as e:
    use_dynamo = False
    logger.info(f"Not using torch dynamo for ONNX export due to:\n{e}")

    def _get_dynamic_shapes_noop(model_descr: ModelDescr):
        """noop for dynamo=False which uses `get_dynamic_axes` instead"""

        return None

    get_dynamic_shapes = _get_dynamic_shapes_noop
    get_dynamic_axes = _get_dynamic_axes_impl
else:
    use_dynamo = True
    logger.info("Using torch dynamo for ONNX export")

    def _get_dynamic_shapes_impl(model_descr: ModelDescr):
        """Get dynamic shapes for torch dynamo export"""
        # dynamic shapes as list to match the source code which may have
        # different arg names than the tensor ids in the model description

        dynamic_shapes: List[Dict[int, TensorDim]] = []
        potential_ref_axes: Dict[str, Tuple[InputAxis, int]] = {}
        # add dynamic dims from parameterized input sizes (and fixed sizes as None)
        for d in model_descr.inputs:
            dynamic_tensor_dims: Dict[int, TensorDim] = {}
            for i, ax in enumerate(d.axes):
                dim_name = f"{d.id}_{ax.id}"
                if isinstance(ax.size, int):
                    dim = STATIC_DIM  # fixed size
                elif ax.size is None:
                    dim = Dim(dim_name, min=1)
                elif isinstance(ax.size, ParameterizedSize):
                    dim = Dim(dim_name, min=ax.size.min)
                elif isinstance(ax.size, SizeReference):
                    continue  # handled below
                else:
                    assert_never(ax.size)

                dynamic_tensor_dims[i] = dim
                potential_ref_axes[dim_name] = (ax, i)

            dynamic_shapes.append(dynamic_tensor_dims)

        # add dynamic dims from size references
        for d, dynamic_tensor_dims in zip(model_descr.inputs, dynamic_shapes):
            for i, ax in enumerate(d.axes):
                if not isinstance(ax.size, SizeReference):
                    continue  # handled above

                dim_name_ref = f"{ax.size.tensor_id}_{ax.size.axis_id}"
                ax_ref, i_ref = potential_ref_axes[dim_name_ref]
                dim_ref = dynamic_tensor_dims[i_ref]
                if isinstance(dim_ref, Dim):
                    a = ax_ref.scale / ax.scale
                    b = ax.size.offset
                    dim = a * dim_ref + b
                else:
                    dim = STATIC_DIM

                dynamic_tensor_dims[i] = dim

        return dynamic_shapes

    get_dynamic_shapes = _get_dynamic_shapes_impl
    get_dynamic_axes = _get_dynamic_axes_noop


def export_to_onnx(
    model_descr: ModelDescr,
    model: Union[torch.nn.Module, "torch.jit.ScriptModule"],
    output_path: Path,
    verbose: bool,
    opset_version: int,
    parent: Literal["torchscript", "pytorch_state_dict"],
) -> OnnxWeightsDescr:
    sample = get_test_input_sample(model_descr)
    procs = get_pre_and_postprocessing(
        model_descr, dataset_for_initial_statistics=[sample]
    )
    procs.pre(sample)
    inputs_numpy = [
        sample.members[get_member_id(ipt)].data.data for ipt in model_descr.inputs
    ]
    inputs_torch = [torch.from_numpy(ipt) for ipt in inputs_numpy]

    save_weights_externally = use_dynamo
    with torch.no_grad():
        outputs_original_torch = model(*inputs_torch)
        if isinstance(outputs_original_torch, torch.Tensor):
            outputs_original_torch = [outputs_original_torch]

        _ = torch.onnx.export(
            model,
            tuple(inputs_torch),
            str(output_path),
            dynamo=use_dynamo,
            external_data=save_weights_externally,
            input_names=[str(d.id) for d in model_descr.inputs],
            output_names=[str(d.id) for d in model_descr.outputs],
            dynamic_axes=get_dynamic_axes(model_descr),
            dynamic_shapes=get_dynamic_shapes(model_descr),
            verbose=verbose,
            opset_version=opset_version,
        )

    if save_weights_externally:
        external_data_path = output_path.with_suffix(
            output_path.suffix + ".data"
        ).absolute()
        if not external_data_path.exists():
            raise FileNotFoundError(
                f"Expected external data file at {external_data_path} not found."
            )
        external_data_descr = FileDescr(source=external_data_path)
    else:
        external_data_descr = None

    return OnnxWeightsDescr(
        source=output_path.absolute(),
        external_data=external_data_descr,
        parent=parent,
        opset_version=opset_version,
        comment=f"Converted with bioimageio.core {__version__}, dynamo={use_dynamo}.",
    )
