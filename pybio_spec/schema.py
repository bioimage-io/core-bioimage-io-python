from dataclasses import asdict
from typing import Any, Dict, Union

import typing
from marshmallow import Schema, pprint, ValidationError, post_load, validates_schema, validates, RAISE
from marshmallow.error_store import ErrorStore

from pybio_spec.exceptions import PyBioValidationException
from pybio_spec import spec_types, fields
from pybio_spec.spec_types import MagicTensorsValue, MagicShapeValue


class PyBioSchema(Schema):
    @post_load
    def make_object(self, data, **kwargs):
        if not data:
            return None

        this_type = getattr(spec_types, self.__class__.__name__)
        if issubclass(this_type, spec_types.Reader):
            pass
        try:
            return this_type(**data)
        except TypeError as e:
            e.args += (f"when initializing {this_type} from {self}",)
            raise e


class CiteEntry(PyBioSchema):
    text = fields.Str(required=True)
    doi = fields.Str(missing=None)
    url = fields.Str(missing=None)

    @validates_schema
    def doi_or_url(self, data, **kwargs):
        if data["doi"] is None and data["url"] is None:
            raise ValidationError("doi or url needs to be specified in a citation")


class MinimalYAML(PyBioSchema):
    name = fields.Str(required=True)
    format_version = fields.Str(required=True)
    description = fields.Str(required=True)
    cite = fields.Nested(CiteEntry, many=True, required=True)
    authors = fields.List(fields.Str(required=True))
    documentation = fields.Path(required=True)
    tags = fields.List(fields.Str(required=True))

    language = fields.Str(required=True)
    framework = fields.Str(missing=None)
    source = fields.ImportableSource(required=True)
    required_kwargs = fields.List(fields.Str, missing=list)
    optional_kwargs = fields.Dict(fields.Str, missing=dict)

    test_input = fields.Path(missing=None)
    test_output = fields.Path(missing=None)
    thumbnail = fields.Path(missing=None)


class BaseSpec(PyBioSchema):
    spec: fields.SpecURI
    kwargs = fields.Dict(missing=dict)

    @validates_schema
    def check_kwargs(self, data, partial, many):
        spec = data["spec"]
        for k in spec.required_kwargs:
            if not k in data["kwargs"]:
                raise PyBioValidationException(f"Missing kwarg: {k}")

        for k in data["kwargs"]:
            if not (k in spec.required_kwargs or k in spec.optional_kwargs):
                raise PyBioValidationException(f"Unexpected kwarg: {k}")


class InputShape(PyBioSchema):
    min = fields.List(fields.Integer(), required=True)
    step = fields.List(fields.Integer(), required=True)

    @validates_schema
    def matching_lengths(self, data, **kwargs):
        min_ = data["min"]
        step = data["step"]
        if min_ is None or step is None:
            return

        if len(min_) != len(step):
            raise PyBioValidationException(
                f"'min' and 'step' have to have the same length! (min: {min_}, step: {step})"
            )


class OutputShape(PyBioSchema):
    reference_input = fields.Str(missing=None)
    scale = fields.List(fields.Float(), required=True)
    offset = fields.List(fields.Integer(), required=True)

    @validates_schema
    def matching_lengths(self, data, **kwargs):
        len_scale = data["scale"]
        len_offset = data["offset"]
        if len(len_scale) != len(len_offset):
            raise PyBioValidationException(f"scale {len_scale} has to have same length as offset {len_offset}!")


class Array(PyBioSchema):
    name = fields.Str(required=True)
    axes = fields.Axes(missing=None)
    data_type = fields.Str(required=True)
    data_range = fields.Tuple((fields.Float(allow_nan=True), fields.Float(allow_nan=True)))

    shape: fields.Nested

    @validates_schema
    def axes_and_shape(self, data, **kwargs):
        axes = data["axes"]
        shape = data["shape"]
        if not isinstance(shape, MagicShapeValue) and axes is None:
            raise PyBioValidationException("Axes may not be 'null', when shape is specified")


class InputArray(Array):
    shape = fields.Shape(InputShape, valid_magic_values=[MagicShapeValue.any], required=True)

    @validates_schema
    def zero_batch_step(self, data, **kwargs):
        axes = data["axes"]
        shape = data["shape"]
        if not isinstance(shape, MagicShapeValue):
            if axes is None:
                raise PyBioValidationException("Axes field required when shape is specified")

            assert isinstance(axes, str), type(axes)

            bidx = axes.find("b")
            if isinstance(shape, tuple):
                # exact shape (with batch size 1)
                if bidx != -1 and shape[bidx] != 1:
                    raise PyBioValidationException("Input shape has to be one in the batch dimension.")

            elif isinstance(shape, spec_types.InputShape):
                step = shape.step
                if bidx != -1 and shape.min[bidx] != 1:
                    raise PyBioValidationException("Input shape has to be one in the batch dimension.")

                if bidx != -1 and step[bidx] != 0:
                    raise PyBioValidationException(
                        "Input shape step has to be zero in the batch dimension (the batch dimension can always be "
                        "increased, but `step` should specify how to increase the minimal shape to find the largest "
                        "single batch shape)"
                    )
            else:
                raise PyBioValidationException(f"Unknown shape type {type(shape)}")


class OutputArray(Array):
    shape = fields.Shape(OutputShape, valid_magic_values=[MagicShapeValue.any], required=True)
    halo = fields.List(fields.Integer(), missing=None)

    @validates_schema
    def matching_halo_length(self, data, **kwargs):
        shape = data["shape"]
        halo = data["halo"]
        if halo is None:
            return
        elif isinstance(shape, tuple) or isinstance(shape, spec_types.OutputShape):
            if len(halo) != len(shape):
                raise PyBioValidationException(f"halo {halo} has to have same length as shape {shape}!")
        elif not isinstance(shape, MagicShapeValue):
            raise NotImplementedError


class Transformation(MinimalYAML):
    dependencies = fields.Dependencies(required=True)
    inputs = fields.Tensors(
        InputArray, valid_magic_values=[MagicTensorsValue.any, MagicTensorsValue.dynamic], required=True
    )
    outputs = fields.Tensors(
        OutputArray, valid_magic_values=[MagicTensorsValue.same, MagicTensorsValue.dynamic], required=True
    )


class TransformationSpec(BaseSpec):
    spec = fields.SpecURI(Transformation, required=True)


class Weights(PyBioSchema):
    source = fields.Str(required=True)
    hash = fields.Dict()


class Prediction(PyBioSchema):
    weights = fields.Nested(Weights)
    dependencies = fields.Dependencies(missing=None)
    preprocess = fields.Nested(TransformationSpec, many=True, missing=None)
    postprocess = fields.Nested(TransformationSpec, many=True, missing=None)


class Reader(MinimalYAML):
    dependencies = fields.Dependencies(missing=None)
    outputs = fields.Tensors(OutputArray, valid_magic_values=[], required=True)


class ReaderSpec(BaseSpec):
    spec = fields.SpecURI(Reader)


class Sampler(MinimalYAML):
    dependencies = fields.Dependencies(missing=None)
    outputs = fields.Tensors(OutputArray, valid_magic_values=[fields.MagicTensorsValue.any], missing=None)


class SamplerSpec(BaseSpec):
    spec = fields.SpecURI(Sampler)


class Optimizer(PyBioSchema):
    source = fields.ImportableSource(required=True)
    required_kwargs = fields.List(fields.Str, missing=list)
    optional_kwargs = fields.Dict(fields.Str, missing=dict)


class Setup(PyBioSchema):
    reader = fields.Nested(ReaderSpec, required=True)
    sampler = fields.Nested(SamplerSpec, required=True)
    preprocess = fields.Nested(TransformationSpec, many=True, allow_none=True)
    loss = fields.Nested(TransformationSpec, many=True, missing=list)
    optimizer = fields.Nested(Optimizer, missing=None)


class Training(PyBioSchema):
    setup = fields.Nested(Setup)
    source = fields.ImportableSource(required=True)
    required_kwargs = fields.List(fields.Str, missing=list)
    optional_kwargs = fields.Dict(fields.Str, missing=dict)
    dependencies = fields.Dependencies(required=True)
    description = fields.Str(missing=None)


class Model(MinimalYAML):
    prediction = fields.Nested(Prediction)
    inputs = fields.Tensors(InputArray, valid_magic_values=[MagicTensorsValue.any], many=True)
    outputs = fields.Tensors(OutputArray, valid_magic_values=[MagicTensorsValue.same], many=True)
    training = fields.Nested(Training, missing=None)

    @validates("outputs")
    def validate_reference_input_names(self, data, **kwargs):
        pass  # todo validate_reference_input_names

    @validates_schema
    def input_propagation_from_training_reader(self, data, **kwargs):
        spec: spec_types.Model = self.make_object(data, **kwargs)
        if spec.training is None:
            return

        reader_axes = spec.training.setup.reader.spec.outputs
        inputs = spec.inputs


class ModelSpec(BaseSpec):
    spec = fields.SpecURI(Model, required=True)


def load_model_spec(uri: str, kwargs: Dict[str, Any] = None) -> spec_types.ModelSpec:
    return ModelSpec().load({"spec": uri, "kwargs": kwargs or {}})


def load_spec(
    uri: str, kwargs: Dict[str, Any] = None
) -> Union[spec_types.ModelSpec, spec_types.TransformationSpec, spec_types.ReaderSpec, spec_types.SamplerSpec]:
    data = {"spec": uri, "kwargs": kwargs or {}}
    last_dot = uri.rfind(".")
    second_last_dot = uri[:last_dot].rfind(".")
    spec_suffix = uri[second_last_dot + 1 : last_dot]
    if spec_suffix == "model":
        return ModelSpec().load(data)
    elif spec_suffix == "transformation":
        return TransformationSpec().load(data)
    elif spec_suffix == "reader":
        return ReaderSpec().load(data)
    elif spec_suffix == "sampler":
        return SamplerSpec().load(data)
    else:
        raise ValueError(f"Invalid spec suffix: {spec_suffix}")


if __name__ == "__main__":
    try:
        spec = load_model_spec(
            "https://github.com/bioimage-io/example-unet-configurations/blob/marshmallow/models/unet-2d-nuclei-broad/UNet2DNucleiBroad.model.yaml"
        )
    except PyBioValidationException as e:
        pprint(e.normalized_messages(), width=280)
    else:
        pprint(asdict(spec), width=280)
