from dataclasses import asdict
from typing import Any, Dict, Union

from marshmallow import Schema as _Schema, pprint, ValidationError, post_load, validates_schema, validates

from pybio.spec import pybio_types, fields
from pybio.spec.pybio_types import MagicTensorsValue, MagicShapeValue


class Schema(_Schema):
    # class Meta:
    # unknown = EXCLUDE
    @post_load
    def make_object(self, data, **kwargs):
        if not data:
            return None

        this_type = getattr(pybio_types, self.__class__.__name__)
        try:
            return this_type(**data)
        except TypeError as e:
            e.args += (f"when initializing {this_type} from {self}",)
            raise e


class CiteEntry(Schema):
    text = fields.Str(required=True)
    doi = fields.Str(missing=None)
    url = fields.Str(missing=None)

    @validates_schema
    def doi_or_url(self, data, **kwargs):
        if data["doi"] is None and data["url"] is None:
            raise ValidationError("doi or url needs to be specified in a citation")


class MinimalYAML(Schema):
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


class BaseSpec(Schema):
    def __init__(self, *args, context: dict = None, **kwargs):
        # make shallow context copy to allow for different 'spec_path' contexts
        super().__init__(*args, context=None if context is None else dict(context), **kwargs)

    spec: fields.SpecURI
    kwargs = fields.Dict(missing=dict)

    @validates_schema
    def check_kwargs(self, data, partial, many):
        spec = data["spec"]
        for k in spec.required_kwargs:
            if not k in data["kwargs"]:
                raise ValidationError

        for k in data["kwargs"]:
            if not (k in spec.required_kwargs or k in spec.optional_kwargs):
                raise ValidationError


class InputShape(Schema):
    # exact = fields.List(fields.Integer(), missing=None)
    min = fields.List(fields.Integer(), required=True)
    step = fields.List(fields.Integer(), required=True)

    # @validates_schema
    # def exact_xor_min_and_step(self, data, **kwargs):
    #     # exact = data["exact"]
    #     min_ = data["min"]
    #     step = data["step"]
    #     msg = "Specify either 'exact' or 'min' and 'step'"
    #     if exact is None:
    #         if min_ is None or step is None:
    #             raise ValidationError("Missing arguments: " + msg)
    #     else:
    #         if min_ is not None or step is not None:
    #             raise ValidationError("Exclusive arguments: " + msg)

    @validates_schema
    def matching_lengths(self, data, **kwargs):
        min_ = data["min"]
        step = data["step"]
        if min_ is None or step is None:
            return

        if len(min_) != len(step):
            raise ValidationError(f"'min' and 'step' have to have the same length! (min: {min_}, step: {step})")


class OutputShape(Schema):
    # exact = fields.List(fields.Integer(), missing=None)
    reference_input = fields.Str(missing=None)
    scale = fields.List(fields.Float(), required=True)
    offset = fields.List(fields.Integer(), required=True)
    halo = fields.List(fields.Integer(), required=True)

    @validates_schema
    def matching_lengths(self, data, **kwargs):
        len_scale = len(data["scale"])
        len_offset = len(data["offset"])
        len_halo = len(data["halo"])
        if len_scale != len_offset or len_scale != len_halo:
            raise ValidationError(
                f"'scale', 'offset', and 'halo' have to have the same length! (scale: {len_scale}, offset:"
                f" {len_offset}, halo: {len_halo})"
            )


class Tensor(Schema):
    name = fields.Str(required=True)
    axes = fields.Axes(missing=None)
    data_type = fields.Str(required=True)
    data_range = fields.Tuple((fields.Float(allow_nan=True), fields.Float(allow_nan=True)))

    shape: fields.Nested


class InputTensor(Tensor):
    shape = fields.Shape(InputShape, valid_magic_values=[MagicShapeValue.any], required=True)


class OutputTensor(Tensor):
    shape = fields.Shape(OutputShape, valid_magic_values=[MagicShapeValue.any], required=True)


class Transformation(MinimalYAML):
    dependencies = fields.Dependencies(required=True)
    inputs = fields.Tensors(InputTensor, valid_magic_values=[MagicTensorsValue.any], required=True)
    outputs = fields.Tensors(OutputTensor, valid_magic_values=[MagicTensorsValue.same], required=True)


class TransformationSpec(BaseSpec):
    spec = fields.SpecURI(Transformation, required=True)


class Weights(Schema):
    source = fields.Str(required=True)
    hash = fields.Dict()


class Prediction(Schema):
    weights = fields.Nested(Weights)
    dependencies = fields.Dependencies(missing=None)
    preprocess = fields.Nested(TransformationSpec, many=True, allow_none=True)
    postprocess = fields.Nested(TransformationSpec, many=True, allow_none=True)


class Reader(MinimalYAML):
    dependencies = fields.Dependencies(missing=None)


class ReaderSpec(BaseSpec):
    spec = fields.SpecURI(Reader)


class Sampler(MinimalYAML):
    dependencies = fields.Dependencies(missing=None)
    outputs = fields.Tensors(OutputTensor, valid_magic_values=[fields.MagicTensorsValue.any])


class SamplerSpec(BaseSpec):
    spec = fields.SpecURI(Sampler)


class Optimizer(Schema):
    source = fields.ImportableSource(required=True)
    required_kwargs = fields.List(fields.Str, missing=list)
    optional_kwargs = fields.Dict(fields.Str, missing=dict)


class Setup(Schema):
    reader = fields.Nested(ReaderSpec, required=True)
    sampler = fields.Nested(SamplerSpec, required=True)
    preprocess = fields.Nested(TransformationSpec, many=True, allow_none=True)
    loss = fields.Nested(TransformationSpec, many=True, required=True)
    optimizer = fields.Nested(Optimizer, required=True)


class Training(Schema):
    setup = fields.Nested(Setup)
    source = fields.ImportableSource(required=True)
    required_kwargs = fields.List(fields.Str, missing=list)
    optional_kwargs = fields.Dict(fields.Str, missing=dict)
    dependencies = fields.Dependencies(required=True)
    description = fields.Str(missing=None)


class Model(MinimalYAML):
    prediction = fields.Nested(Prediction)
    inputs = fields.Nested(InputTensor, many=True)
    outputs = fields.Nested(OutputTensor, many=True)
    training = fields.Nested(Training, missing=None)

    @validates("outputs")
    def validate_reference_input_names(self, data, **kwargs):
        pass  # todo validate_reference_input_names


class ModelSpec(BaseSpec):
    spec = fields.SpecURI(Model, required=True)


def load_model_spec(uri: str, kwargs: Dict[str, Any] = None) -> pybio_types.ModelSpec:
    return ModelSpec().load({"spec": uri, "kwargs": kwargs or {}})


def load_spec(
    uri: str, kwargs: Dict[str, Any] = None
) -> Union[pybio_types.ModelSpec, pybio_types.TransformationSpec, pybio_types.ReaderSpec, pybio_types.SamplerSpec]:
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
            "/repos/example-unet-configurations/models/unet-2d-nuclei-broad/UNet2DNucleiBroad.model.yaml"
        )
    except ValidationError as e:
        pprint(e.normalized_messages(), width=120)
    else:
        pprint(asdict(spec))
