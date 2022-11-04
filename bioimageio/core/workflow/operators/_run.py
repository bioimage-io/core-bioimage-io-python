from os import PathLike
from typing import Any, Dict, IO, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr
from marshmallow import missing

from bioimageio.core import load_resource_description
from bioimageio.core.prediction_pipeline import create_prediction_pipeline
from bioimageio.core.resource_io import nodes
from bioimageio.spec import load_raw_resource_description
from bioimageio.spec.model import raw_nodes
from bioimageio.spec.shared.raw_nodes import ResourceDescription as RawResourceDescription


def run_model_inference(
    rdf_source: Union[dict, PathLike, IO, str, bytes, raw_nodes.URI, RawResourceDescription],
    *tensors,
    enable_preprocessing: bool = True,
    enable_postprocessing: bool = True,
    devices: Optional[Sequence[str]] = None,
) -> List[xr.DataArray]:
    """run model inference

    Returns:
        list: model outputs
    """
    model = load_raw_resource_description(rdf_source, update_to_format="latest")
    assert isinstance(model, raw_nodes.Model)
    # remove pre-/postprocessing if specified
    if not enable_preprocessing:
        for ipt in model.inputs:
            if ipt.preprocessing:
                ipt.preprocessing = missing
    if not enable_postprocessing:
        for ipt in model.outputs:
            if ipt.postprocessing:
                ipt.postprocessing = missing

    with create_prediction_pipeline(model, devices=devices) as pred_pipeline:
        return pred_pipeline.forward(*tensors)


def run_workflow(
    rdf_source: Union[dict, PathLike, IO, str, bytes, raw_nodes.URI, RawResourceDescription],
    inputs: Sequence = tuple(),
    options: Dict[str, Any] = None,
) -> Sequence:
    return _run_workflow(rdf_source, test_steps=False, inputs=inputs, options=options)


def run_workflow_test(
    rdf_source: Union[dict, PathLike, IO, str, bytes, raw_nodes.URI, RawResourceDescription],
) -> Sequence:
    return _run_workflow(rdf_source, test_steps=True)


def _run_workflow(
    rdf_source: Union[dict, PathLike, IO, str, bytes, raw_nodes.URI, RawResourceDescription],
    *,
    test_steps: bool,
    inputs: Sequence = tuple(),
    options: Dict[str, Any] = None,
) -> Tuple:
    import bioimageio.core.workflow.operators as ops

    workflow = load_resource_description(rdf_source)
    assert isinstance(workflow, nodes.Workflow)
    wf_options = {opt.name: opt.default for opt in workflow.options_spec}
    if test_steps:
        assert not inputs
        assert not options
        wf_inputs = {}
        steps = workflow.test_steps
    else:
        if not len(workflow.inputs_spec) == len(inputs):
            raise ValueError(f"Expected {len(workflow.inputs_spec)} inputs, but got {len(inputs)}.")

        wf_inputs = {ipt_spec.name: ipt for ipt_spec, ipt in zip(workflow.inputs_spec, inputs)}
        for k, v in options.items():
            if k not in wf_options:
                raise ValueError(f"Got unknown option {k}, expected one of {set(wf_options)}.")

            wf_options[k] = v

        steps = workflow.steps

    named_outputs = {}  # for later referencing

    def map_ref(value):
        assert isinstance(workflow, nodes.Workflow)
        if isinstance(value, str) and value.startswith("${{") and value.endswith("}}"):
            ref = value[4:-2].strip()
            if ref.startswith("self.inputs."):
                ref = ref[len("self.inputs.") :]
                if ref not in wf_inputs:
                    raise ValueError(f"Invalid workflow input reference {value}.")

                return wf_inputs[ref]
            elif ref.startswith("self.options."):
                ref = ref[len("self.options.") :]
                if ref not in wf_options:
                    raise ValueError(f"Invalid workflow option reference {value}.")

                return wf_options[ref]
            elif ref == "self.rdf_source":
                assert workflow.rdf_source is not missing
                return str(workflow.rdf_source)
            elif ref in named_outputs:
                return named_outputs[ref]
            else:
                raise ValueError(f"Invalid reference {value}.")
        else:
            return value

    # implicit inputs to a step are the outputs of the previous step.
    # For the first step these are the workflow inputs.
    outputs = inputs
    for step in steps:
        if not hasattr(ops, step.op):
            raise NotImplementedError(f"{step.op} not implemented in {ops}")

        op = getattr(ops, step.op)
        if step.inputs is missing:
            inputs = outputs
        else:
            inputs = [map_ref(ipt) for ipt in step.inputs]

        options = {k: map_ref(v) for k, v in (step.options or {}).items()}
        outputs = op(*inputs, **options)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        if step.outputs:
            assert step.id is not missing
            if len(step.outputs) != len(outputs):
                raise ValueError(
                    f"Got {len(step.outputs)} step output name{'s' if len(step.outputs) > 1 else ''} ({step.id}.outputs), "
                    f"but op {step.op} returned {len(outputs)} outputs."
                )

            named_outputs.update({f"{step.id}.outputs.{out_name}": out for out_name, out in zip(step.outputs, outputs)})

    if len(workflow.outputs_spec) != len(outputs):
        raise ValueError(f"Expected {len(workflow.outputs_spec)} outputs from last step, but got {len(outputs)}.")

    def tensor_as_xr(tensor, axes: Sequence[nodes.Axis]):
        spec_axes = [a.name or a.type for a in axes]
        if isinstance(tensor, xr.DataArray):
            if list(tensor.dims) != spec_axes:
                raise ValueError(
                    f"Last workflow step returned xarray.DataArray with dims {tensor.dims}, but expected dims {spec_axes}."
                )

            return tensor
        else:
            return xr.DataArray(tensor, dims=spec_axes)

    return [
        tensor_as_xr(out, out_spec.axes) if out_spec.type == "tensor" else out
        for out_spec, out in zip(workflow.outputs_spec, outputs)
    ]
