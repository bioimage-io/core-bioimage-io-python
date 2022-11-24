import collections.abc
import inspect
import sys
import types
import typing
import warnings
from argparse import ArgumentParser
from dataclasses import asdict
from importlib import import_module
from pathlib import Path

import docstring_parser
import numpy as np
import xarray as xr
from marshmallow import missing
from marshmallow.utils import _Missing

import bioimageio.core.contrib
from bioimageio.spec import serialize_raw_resource_description_to_dict, load_raw_resource_description
import bioimageio.spec.workflow.schema as wf_schema
from bioimageio.spec.shared import field_validators, fields, yaml
from bioimageio.spec.workflow.raw_nodes import (
    ArbitraryAxes,
    Axis,
    DEFAULT_TYPE_NAME_MAP,
    Input,
    Option,
    Output,
    Workflow as WorkflowRawNode,
)

try:
    from typing import get_args, get_origin, Literal
except ImportError:
    from typing_extensions import get_args, get_origin, Literal  # type: ignore


TYPE_NAME_MAP = {**DEFAULT_TYPE_NAME_MAP, **{xr.DataArray: "tensor", np.ndarray: "tensor"}}
ARBITRARY_AXES = get_args(ArbitraryAxes)

# keep this axes_field in sync with wf_schema.Workflow.axes
axes_field = fields.Union(
    [
        fields.List(fields.Nested(wf_schema.Axis())),
        fields.String(
            validate=field_validators.OneOf(get_args(ArbitraryAxes)),
        ),
    ],
)


def get_type_name(annotation):

    orig = get_origin(annotation)
    if orig is list or orig is tuple or orig is collections.abc.Sequence:
        annotation = list
    elif orig is dict or orig is collections.OrderedDict:
        annotation = dict
    elif orig is typing.Union:
        args = get_args(annotation)
        args = [a for a in args if a is not type(None)]
        assert args
        annotation = get_type_name(args[0])  # use first type in union annotation
    elif orig is Literal:
        args = get_args(annotation)
        assert args
        annotation = get_type_name(type(args[0]))  # use type of first literal

    if isinstance(annotation, str):
        assert annotation in TYPE_NAME_MAP.values(), annotation
        return annotation
    else:
        return TYPE_NAME_MAP[annotation]


def parse_args():
    p = ArgumentParser(description="Generate workflow RDFs for one contrib submodule")
    p.add_argument("contrib_name", choices=[c for c in dir(bioimageio.core.contrib) if c.startswith("contrib_")])

    return p.parse_args()


class WorkflowSignature(inspect.Signature):
    pass


def extract_axes_from_param_descr(
    descr: str,
) -> typing.Tuple[str, typing.Union[_Missing, ArbitraryAxes, typing.List[Axis]]]:
    if "\n" in descr:
        descr, *axes_descr_lines = descr.split("\n")
        axes_descr = "\n".join(axes_descr_lines).strip()
        assert axes_descr.startswith("axes:")
        axes_descr = axes_descr[len("axes:") :].strip()
        try:
            axes_data = yaml.load(axes_descr)
            axes = axes_field._deserialize(axes_data)
        except Exception as e:
            raise ValueError("Invalid axes description") from e
    else:
        axes = missing

    return descr, axes


def extract_serialized_wf_kwargs(descr: str) -> typing.Tuple[str, typing.Dict[str, typing.Any]]:
    separator = ".. code-block:: yaml"
    if separator in descr:
        descr, kwarg_descr = descr.split(separator)
        # kwarg_descr =
        try:
            kwargs = yaml.load(kwarg_descr)
        except Exception as e:
            raise ValueError("Invalid additional fields") from e
    else:
        kwargs = {}

    return descr.strip(), kwargs


def main(contrib_name):
    dist = Path(__file__).parent / "../dist/workflows"
    dist.mkdir(exist_ok=True)

    local_contrib = import_module(f"bioimageio.core.contrib.{contrib_name}.local")
    for wf_id in dir(local_contrib):
        wf_func = getattr(local_contrib, wf_id)
        if not isinstance(wf_func, types.FunctionType):
            if not wf_id.startswith("_"):
                warnings.warn(f"ignoring non-function {wf_id}")

            continue

        doc = docstring_parser.parse(wf_func.__doc__)

        param_descriptions = {param.arg_name: param.description for param in doc.params}
        inputs = []
        options = []
        sig = WorkflowSignature.from_callable(wf_func)
        assert sig.return_annotation is not inspect.Signature.empty
        for name, param in sig.parameters.items():
            type_name = get_type_name(param.annotation)
            descr = param_descriptions[name]
            if type_name == "tensor":
                descr, axes = extract_axes_from_param_descr(descr)
                if axes is missing:
                    raise ValueError(
                        f"Missing axes description in description of parameter '{name}' of workflow '{wf_id}'. Change '{name}: <description>' to e.g. '{name}: axes: arbitrary. <description>' or  '{name}: axes: b,c,x,y. <description>."
                    )
            else:
                axes = missing

            if param.default is inspect.Parameter.empty:
                inputs.append(Input(name=name, description=descr, type=type_name, axes=axes))
            else:
                default_value: typing.Any = param.default
                if isinstance(default_value, tuple):
                    default_value = list(default_value)

                options.append(Option(name=name, description=descr, type=type_name, axes=axes, default=default_value))

        return_descriptions = {}
        for ret in doc.many_returns:
            # note: doctring_parser seems to be buggy and not recover the return name
            # extract return name from return description
            name, *remaining = ret.description.split(".")
            return_descriptions[name.strip()] = ".".join(remaining).strip()

        outputs = []
        ret_annotations = sig.return_annotation
        if isinstance(ret_annotations, typing.Tuple):
            ret_annotations = get_args(ret_annotations)
        else:
            ret_annotations = [ret_annotations]

        if len(doc.many_returns) != len(ret_annotations):
            raise ValueError("number of documented return values does not match return annotation")

        for ret_type, (name, descr) in zip(ret_annotations, return_descriptions.items()):
            type_name = get_type_name(ret_type)
            if type_name == "tensor":
                descr, axes = extract_axes_from_param_descr(descr)
            else:
                axes = missing

            assert descr
            outputs.append(Output(name=name, description=descr, type=type_name, axes=axes))

        assert doc.long_description is not None
        description, serialized_kwargs = extract_serialized_wf_kwargs(doc.long_description)
        wf = WorkflowRawNode(
            name=doc.short_description,
            description=description,
            inputs=inputs,
            options=options,
            outputs=outputs,
        )
        serialized = serialize_raw_resource_description_to_dict(wf)
        serialized.update(serialized_kwargs)
        with (dist / wf_id).with_suffix(".yaml").open("w", encoding="utf-8") as f:
            yaml.dump(serialized, f)
        wf = load_raw_resource_description(serialized)
        serialized = serialize_raw_resource_description_to_dict(wf)

        print(f"saved {wf_id}")
    print("done")


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args.contrib_name))
