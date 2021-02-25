import typing
from dataclasses import asdict
from pathlib import Path
from pprint import pprint

from marshmallow import Schema, ValidationError, missing, post_load, validate, validates, validates_schema

from pybio.spec import fields, raw_nodes
from pybio.spec.exceptions import PyBioValidationException


class PyBioSchema(Schema):
    @post_load
    def make_object(self, data, **kwargs):
        if not data:
            return None

        this_type = getattr(raw_nodes, self.__class__.__name__)
        try:
            return this_type(**data)
        except TypeError as e:
            e.args += (f"when initializing {this_type} from {self}",)
            raise e


class CiteEntry(PyBioSchema):
    text = fields.String(required=True)
    doi = fields.String(missing=None)
    url = fields.String(missing=None)

    @validates_schema
    def doi_or_url(self, data, **kwargs):
        if data["doi"] is None and data["url"] is None:
            raise ValidationError("doi or url needs to be specified in a citation")


class Spec(PyBioSchema):
    format_version = fields.String(validate=validate.OneOf(raw_nodes.FormatVersion.__args__), required=True)
    name = fields.String(required=True)
    description = fields.String(required=True)

    authors = fields.List(fields.String, required=True)
    cite = fields.Nested(CiteEntry, many=True, required=True)

    git_repo = fields.String(validate=validate.URL(schemes=["http", "https"]))
    tags = fields.List(fields.String, required=True)
    license = fields.String(required=True)

    documentation = fields.URI(required=True)
    covers = fields.List(fields.URI, missing=list)
    attachments = fields.Dict(fields.String, missing=dict)

    config = fields.Dict(missing=dict)

    language = fields.String(validate=validate.OneOf(raw_nodes.Language.__args__), required=True)
    framework = fields.String(validate=validate.OneOf(raw_nodes.Framework.__args__), required=True)
    dependencies = fields.Dependencies(missing=None)
    timestamp = fields.DateTime(required=True)

    # root = fields.Path(required=True)


class SpecWithKwargs(PyBioSchema):
    spec: fields.SpecURI
    kwargs = fields.Dict(fields.String, missing=dict)


class InputShape(PyBioSchema):
    min = fields.List(fields.Integer, required=True)
    step = fields.List(fields.Integer, required=True)

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
    reference_input = fields.String(required=True)
    scale = fields.List(fields.Float, required=True)
    offset = fields.List(fields.Integer, required=True)

    @validates_schema
    def matching_lengths(self, data, **kwargs):
        scale = data["scale"]
        offset = data["offset"]
        if len(scale) != len(offset):
            raise PyBioValidationException(f"scale {scale} has to have same length as offset {offset}!")


class Tensor(PyBioSchema):
    name = fields.String(required=True, validate=validate.Predicate("isidentifier"))
    description = fields.String(required=False)
    axes = fields.Axes(required=True)  # todo check if null is ok (it shouldn't)
    data_type = fields.String(required=True)
    data_range = fields.Tuple((fields.Float(allow_nan=True), fields.Float(allow_nan=True)))
    shape: fields.Shape

    processing_name: str

    @validates_schema
    def validate_processing_kwargs(self, data, **kwargs):
        axes = data["axes"]
        processing_list = data.get(self.processing_name, [])
        for processing in processing_list:
            name = processing.name
            kwargs = processing.kwargs or {}
            kwarg_axes = kwargs.get("axes", "")
            if any(a not in axes for a in kwarg_axes):
                raise PyBioValidationException("`kwargs.axes` needs to be subset of axes")


class Processing(PyBioSchema):
    class Binarize(Schema):  # do not inherit from PyBioSchema, return only a validated dict, no specific node
        threshold = fields.Float(required=True)

    class Clip(Schema):
        min = fields.Float(required=True)
        max = fields.Float(required=True)

    class ScaleLinear(Schema):
        axes = fields.Axes(required=True, valid_axes="czyx")
        gain = fields.Array(fields.Float(), missing=fields.Float(missing=1.0))  # todo: check if gain match input axes
        offset = fields.Array(fields.Float(), missing=fields.Float(missing=0.0))  # todo: check if offset match input axes

        @validates_schema
        def either_gain_or_offset(self, data, **kwargs):
            if data["gain"] == 1.0 and data["offset"] == 0:
                raise PyBioValidationException("Specify gain!=1.0 or offset!=0.0")

    @validates_schema
    def kwargs_match_selected_preprocessing_name(self, data, **kwargs):
        schema_name = "".join(word.title() for word in data["name"].split("_"))

        try:
            schema_class = getattr(self, schema_name)
        except AttributeError as missing_schema_error:
            raise NotImplementedError(
                f"Schema {schema_name} for {data['name']} {self.__class__.__name__.lower()}"
            ) from missing_schema_error

        kwargs_validation_errors = schema_class().validate(data["kwargs"])
        if kwargs_validation_errors:
            raise PyBioValidationException(f"Invalid `kwargs` for '{data['name']}': {kwargs_validation_errors}")

    class Sigmoid(Schema):
        pass

    class ZeroMeanUnitVariance(Schema):
        mode = fields.ProcMode(required=True)
        axes = fields.Axes(required=True, valid_axes="czyx")
        mean = fields.Array(fields.Float(), missing=None)  # todo: check if means match input axes (for mode 'fixed')
        std = fields.Array(fields.Float(), missing=None)
        eps = fields.Float(missing=1e-6)

        @validates_schema
        def mean_and_std_match_mode(self, data, **kwargs):
            if data["mode"] == "fixed" and (data["mean"] is None or data["std"] is None):
                raise PyBioValidationException(
                    "`kwargs` for 'zero_mean_unit_variance' preprocessing with `mode` 'fixed' require additional `kwargs`: `mean` and `std`."
                )
            elif data["mode"] != "fixed" and (data.get("mean") is not None or data.get("std") is not None):
                raise PyBioValidationException(
                    "`kwargs`: `mean` and `std` for 'zero_mean_unit_variance' preprocessing are only valid for `mode` 'fixed'."
                )


class Preprocessing(Processing):
    name = fields.String(required=True, validate=validate.OneOf(raw_nodes.PreprocessingName.__args__))
    kwargs = fields.Dict(fields.String, missing=dict)

    class ScaleRange(Schema):
        mode = fields.ProcMode(required=True, valid_modes=("per_dataset", "per_sample"))
        axes = fields.Axes(required=True, valid_axes="czyx")
        min_percentile = fields.Float(
            required=True, validate=validate.Range(0, 100, min_inclusive=True, max_inclusive=True)
        )
        max_percentile = fields.Float(
            required=True, validate=validate.Range(1, 100, min_inclusive=False, max_inclusive=True)
        )  # as a precaution 'max_percentile' needs to be greater than 1

        @validates_schema
        def min_smaller_max(self, data, **kwargs):
            min_p = data["min_percentile"]
            max_p = data["max_percentile"]
            if min_p >= max_p:
                raise PyBioValidationException(f"min_percentile {min_p} >= max_percentile {max_p}")


class Postprocessing(Processing):
    name = fields.String(validate=validate.OneOf(raw_nodes.PostprocessingName.__args__), required=True)
    kwargs = fields.Dict(fields.String, missing=dict)

    class ScaleRange(Preprocessing.ScaleRange):
        reference_tensor: fields.String(required=True, validate=validate.Predicate("isidentifier"))

    class ScaleMeanVariance(Schema):
        mode = fields.ProcMode(required=True, valid_modes=("per_dataset", "per_sample"))
        reference_tensor: fields.String(required=True, validate=validate.Predicate("isidentifier"))


class InputTensor(Tensor):
    shape = fields.Shape(InputShape, required=True)
    preprocessing = fields.List(fields.Nested(Preprocessing), missing=list)
    processing_name = "preprocessing"

    @validates_schema
    def zero_batch_step_and_one_batch_size(self, data, **kwargs):
        axes = data["axes"]
        shape = data["shape"]

        bidx = axes.find("b")
        if bidx == -1:
            return

        if isinstance(shape, raw_nodes.InputShape):
            step = shape.step
            shape = shape.min

        elif isinstance(shape, list):
            step = [0] * len(shape)
        else:
            raise PyBioValidationException(f"Unknown shape type {type(shape)}")

        if step[bidx] != 0:
            raise PyBioValidationException(
                "Input shape step has to be zero in the batch dimension (the batch dimension can always be "
                "increased, but `step` should specify how to increase the minimal shape to find the largest "
                "single batch shape)"
            )

        if shape[bidx] != 1:
            raise PyBioValidationException("Input shape has to be 1 in the batch dimension b.")


class OutputTensor(Tensor):
    shape = fields.Shape(OutputShape, required=True)
    halo = fields.Halo()
    postprocessing = fields.List(fields.Nested(Postprocessing), missing=list)
    processing_name = "postprocessing"

    @validates_schema
    def matching_halo_length(self, data, **kwargs):
        shape = data["shape"]
        halo = data["halo"]
        if halo is None:
            return
        elif isinstance(shape, tuple) or isinstance(shape, raw_nodes.OutputShape):
            if len(halo) != len(shape):
                raise PyBioValidationException(f"halo {halo} has to have same length as shape {shape}!")
        else:
            raise NotImplementedError(type(shape))


class WithFileSource(PyBioSchema):
    source = fields.URI(required=True)
    sha256 = fields.String(validate=validate.Length(equal=64), missing=None)


class WeightsEntry(WithFileSource):
    authors = fields.List(fields.String, missing=list)  # todo: copy root authors if missing
    attachments = fields.Dict(missing=dict)
    parent = fields.String(missing=None)
    # ONNX Specific
    opset_version = fields.Number(missing=None)
    # tensorflow_saved_model_bundle specific
    tensorflow_version = fields.StrictVersion(missing=None)


class Model(Spec):

    source = fields.ImportableSource(missing=None)
    sha256 = fields.String(validate=validate.Length(equal=64), missing=None)
    kwargs = fields.Dict(fields.String, missing=dict)

    weights = fields.Dict(
        fields.String(validate=validate.OneOf(raw_nodes.WeightsFormat.__args__), required=True),
        fields.Nested(WeightsEntry),
        required=True,
    )

    inputs = fields.Nested(InputTensor, many=True)
    outputs = fields.Nested(OutputTensor, many=True)

    test_inputs = fields.List(fields.URI, required=True)
    test_outputs = fields.List(fields.URI, required=True)

    sample_inputs = fields.List(fields.URI, missing=[])
    sample_outputs = fields.List(fields.URI, missing=[])

    config = fields.Dict(missing=dict)

    @validates_schema
    def language_and_framework_match(self, data, **kwargs):
        field_names = ("language", "framework")
        valid_combinations = {
            ("python", "scikit-learn"): {"requires_source": False},
            ("python", "pytorch"): {"requires_source": True},
            ("python", "tensorflow"): {"requires_source": False},
            ("java", "tensorflow"): {"requires_source": False},
        }
        combination = tuple(data[name] for name in field_names)
        if combination not in valid_combinations:
            raise PyBioValidationException(f"invalid combination of {dict(zip(field_names, combination))}")

        if valid_combinations[combination]["requires_source"] and data.get("source") is None:
            raise PyBioValidationException(
                f"{dict(zip(field_names, combination))} require source code to be specified."
            )

    @validates_schema
    def source_specified_if_required(self, data, **kwargs):
        if data["source"] is not None:
            return

        weight_format_requires_source = {
            "pickle": True,
            "pytorch_state_dict": True,
            "pytorch_script": False,
            "keras_hdf5": False,
            "tensorflow_js": False,
            "tensorflow_saved_model_bundle": False,
            "onnx": False,
        }
        require_source = {wf for wf in data["weights"] if weight_format_requires_source[wf]}
        if require_source:
            raise PyBioValidationException(
                f"These specified weight formats require source code to be specified: {require_source}"
            )

    @validates_schema
    def validate_reference_tensor_names(self, data, **kwargs):
        valid_input_tensor_references = [ipt.name for ipt in data["inputs"]]
        for out in data["outputs"]:
            for kwargs in out.postprocessing:
                ref_tensor = kwargs.reference_tensor
                if not (ref_tensor is None or ref_tensor in valid_input_tensor_references):
                    raise PyBioValidationException(f"{ref_tensor} not found in inputs")

    @validates_schema
    def weights_entries_match_weights_formats(self, data, **kwargs):
        weights: typing.Dict[str, WeightsEntry] = data["weights"]
        for weights_format, weights_entry in weights.items():
            if "tensorflow" not in weights_format and weights_entry.tensorflow_version is not None:
                raise PyBioValidationException(
                    f"invalid 'tensorflow_version' entry for weights format {weights_format}"
                )

            if weights_format != "onnx" and weights_entry.opset_version is not None:
                raise PyBioValidationException(
                    f"invalid 'opset_version' entry for weights format {weights_format} (only valid for onnx)"
                )


class BioImageIoManifestModelEntry(Schema):
    id = fields.String(required=True)
    source = fields.String(validate=validate.URL(schemes=["http", "https"]))
    links = fields.List(fields.String, missing=list)
    download_url = fields.String(validate=validate.URL(schemes=["http", "https"]))


class Badge(Schema):
    label = fields.String(required=True)
    icon = fields.URI()
    url = fields.URI()


class BioImageIoManifestNotebookEntry(Schema):
    id = fields.String(required=True)
    name = fields.String(required=True)
    documentation = fields.String(required=True)
    description = fields.String(required=True)

    cite = fields.List(fields.Nested(CiteEntry), missing=list)
    authors = fields.List(fields.String, required=True)
    covers = fields.List(fields.URI, missing=list)

    badges = fields.List(fields.Nested(Badge), missing=list)
    tags = fields.List(fields.String, missing=list)
    source = fields.URI(required=True)
    links = fields.List(fields.String, missing=list)  # todo: make List[URI]?


class BioImageIoManifest(Schema):
    format_version = fields.String(validate=validate.OneOf(raw_nodes.FormatVersion.__args__), required=True)
    config = fields.Dict()

    application = fields.List(fields.Dict, missing=list)
    collection = fields.List(fields.Dict, missing=list)
    model = fields.List(fields.Nested(BioImageIoManifestModelEntry), missing=list)
    dataset = fields.List(fields.Dict, missing=list)
    notebook = fields.List(fields.Nested(BioImageIoManifestNotebookEntry), missing=list)


if __name__ == "__main__":
    from pybio.spec import load_model_config

    try:
        model = load_model_config(
            (Path(__file__) / "../../../specs/models/sklearnbased/RandomForestClassifier.model.yaml").as_uri()
        )
    except PyBioValidationException as e:
        pprint(e.normalized_messages(), width=280)
    else:
        pprint(asdict(model), width=280)
