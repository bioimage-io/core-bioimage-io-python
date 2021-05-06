import typing
from dataclasses import asdict
from pathlib import Path
from pprint import pprint

from marshmallow import Schema, ValidationError, missing, post_load, validate, validates, validates_schema

from pybio.spec import fields, raw_nodes
from pybio.spec.exceptions import PyBioValidationException


class PyBioSchema(Schema):
    bioimageio_description: str = ""

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


class RunMode(PyBioSchema):
    name = fields.String(
        required=True, bioimageio_description="The name of the `run_mode`"
    )  # todo: limit valid run mode names
    kwargs = fields.Kwargs()


class Spec(PyBioSchema):
    format_version = fields.String(
        validate=validate.OneOf(raw_nodes.FormatVersion.__args__),
        required=True,
        bioimageio_description_order=0,
        bioimageio_description=f"""Version of the BioImage.IO Model Description File Specification used.
This is mandatory, and important for the consumer software to verify before parsing the fields.
The recommended behavior for the implementation is to keep backward compatibility and throw an error if the model yaml
is in an unsupported format version. The current format version described here is
{raw_nodes.FormatVersion.__args__[-1]}""",
    )
    name = fields.String(required=True)
    description = fields.String(required=True, bioimageio_description="A string containing a brief description.")

    authors = fields.List(
        fields.String,
        required=True,
        bioimageio_description="""A list of author strings.
A string can be separated by `;` in order to identify multiple handles per author.
The authors are the creators of the specifications and the primary points of contact.""",
    )
    cite = fields.Nested(
        CiteEntry,
        many=True,
        required=True,
        bioimageio_description="""A citation entry or list of citation entries.
Each entry contains a mandatory `text` field and either one or both of `doi` and `url`.
E.g. the citation for the model architecture and/or the training data used.""",
    )

    git_repo = fields.String(
        validate=validate.URL(schemes=["http", "https"]),
        missing=None,
        bioimageio_description="""A url to the git repository, e.g. to Github or Gitlab.
If the model is contained in a subfolder of a git repository, then a url to the exact folder
(which contains the configuration yaml file) should be used.""",
    )
    tags = fields.List(fields.String, required=True, bioimageio_description="A list of tags.")
    license = fields.String(
        required=True,
        bioimageio_description="A string to a common license name (e.g. `MIT`, `APLv2`) or a relative path to the "
        "license file.",
    )

    documentation = fields.URI(
        required=True, bioimageio_description="Relative path to file with additional documentation in markdown."
    )
    covers = fields.List(
        fields.URI,
        missing=list,
        bioimageio_description="A list of cover images provided by either a relative path to the model folder, or a "
        "hyperlink starting with 'https'.Please use an image smaller than 500KB and an aspect ratio width to height "
        "of 2:1. The supported image formats are: 'jpg', 'png', 'gif'.",  # todo: validate image format
    )
    attachments = fields.Dict(
        fields.String,
        fields.Union([fields.URI(), fields.List(fields.URI)]),
        missing=dict,
        bioimageio_maybe_required=True,
        bioimageio_description="""Dictionary of text keys and URI (or a list of URI) values to additional, relevant
files. E.g. we can place a list of URIs under the `files` to list images and other files that are necessary for the
documentation or for the model to run, these files will be included when generating the model package.""",
    )

    run_mode = fields.Nested(
        RunMode,
        missing=None,
        bioimageio_description="Custom run mode for this model: for more complex prediction procedures like test time "
        "data augmentation that currently cannot be expressed in the specification. The different run modes should be "
        "listed in [supported_formats_and_operations.md#Run Modes]"
        "(https://github.com/bioimage-io/configuration/blob/master/supported_formats_and_operations.md#run-modes).",
    )
    config = fields.Dict(missing=dict)

    language = fields.String(
        validate=validate.OneOf(raw_nodes.Language.__args__),
        missing=None,
        bioimageio_maybe_required=True,
        bioimageio_description=f"Programming language of the source code. One of: "
        f"{', '.join(raw_nodes.Language.__args__)}. This field is only required if the field `source` is present.",
    )
    framework = fields.String(
        validate=validate.OneOf(raw_nodes.Framework.__args__),
        missing=None,
        bioimageio_description=f"The deep learning framework of the source code. One of: "
        f"{', '.join(raw_nodes.Framework.__args__)}. This field is only required if the field `source` is present.",
    )
    dependencies = fields.Dependencies(
        missing=None,
        bioimageio_description="Dependency manager and dependency file, specified as `<dependency manager>:<relative "
        "path to file>`. For example: 'conda:./environment.yaml', 'maven:./pom.xml', or 'pip:./requirements.txt'",
    )
    timestamp = fields.DateTime(
        required=True,
        bioimageio_description="Timestamp of the initial creation of this model in [ISO 8601]"
        "(#https://en.wikipedia.org/wiki/ISO_8601) format.",
    )


class SpecWithKwargs(PyBioSchema):
    spec: fields.SpecURI
    kwargs = fields.Kwargs()


class ImplicitInputShape(PyBioSchema):
    min = fields.List(
        fields.Integer, required=True, bioimageio_description="The minimum input shape with same length as `axes`"
    )
    step = fields.List(
        fields.Integer, required=True, bioimageio_description="The minimum shape change with same length as `axes`"
    )

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


class ImplicitOutputShape(PyBioSchema):
    reference_input = fields.String(required=True, bioimageio_description="Name of the reference input tensor.")
    scale = fields.List(
        fields.Float, required=True, bioimageio_description="'output_pix/input_pix' for each dimension."
    )
    offset = fields.List(fields.Integer, required=True, bioimageio_description="Position of origin wrt to input.")

    @validates_schema
    def matching_lengths(self, data, **kwargs):
        scale = data["scale"]
        offset = data["offset"]
        if len(scale) != len(offset):
            raise PyBioValidationException(f"scale {scale} has to have same length as offset {offset}!")


class Tensor(PyBioSchema):
    name = fields.String(
        required=True, validate=validate.Predicate("isidentifier"), bioimageio_description="Tensor name."
    )
    description = fields.String(missing=None)
    axes = fields.Axes(
        required=True,
        bioimageio_description="""Axes identifying characters from: bitczyx. Same length and order as the axes in `shape`.

    | character | description |
    | --- | --- |
    |  b  |  batch (groups multiple samples) |
    |  i  |  instance/index/element |
    |  t  |  time |
    |  c  |  channel |
    |  z  |  spatial dimension z |
    |  y  |  spatial dimension y |
    |  x  |  spatial dimension x |""",
    )
    data_type = fields.String(
        required=True,
        bioimageio_description="The data type of this tensor. For inputs, only `float32` is allowed and the consumer "
        "software needs to ensure that the correct data type is passed here. For outputs can be any of `float32, "
        "float64, (u)int8, (u)int16, (u)int32, (u)int64`. The data flow in bioimage.io models is explained "
        "[in this diagram.](https://docs.google.com/drawings/d/1FTw8-Rn6a6nXdkZ_SkMumtcjvur9mtIhRqLwnKqZNHM/edit).",
    )
    data_range = fields.Tuple(
        (fields.Float(allow_nan=True), fields.Float(allow_nan=True)),
        missing=(None, None),
        bioimageio_description="Tuple `(minimum, maximum)` specifying the allowed range of the data in this tensor. "
        "If not specified, the full data range that can be expressed in `data_type` is allowed.",
    )
    shape: fields.Union

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

    class Clip(PyBioSchema):
        min = fields.Float(required=True)
        max = fields.Float(required=True)

    class ScaleLinear(PyBioSchema):
        axes = fields.Axes(required=True, valid_axes="czyx")
        gain = fields.Array(fields.Float(), missing=fields.Float(missing=1.0))  # todo: check if gain match input axes
        offset = fields.Array(
            fields.Float(), missing=fields.Float(missing=0.0)
        )  # todo: check if offset match input axes

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

    class Sigmoid(PyBioSchema):
        pass

    class ZeroMeanUnitVariance(PyBioSchema):
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
    name = fields.String(
        required=True,
        validate=validate.OneOf(raw_nodes.PreprocessingName.__args__),
        bioimageio_description=f"Name of preprocessing. One of: {', '.join(raw_nodes.PreprocessingName.__args__)} "
        f"(see [supported_formats_and_operations.md#preprocessing](https://github.com/bioimage-io/configuration/"
        f"blob/master/supported_formats_and_operations.md#preprocessing) "
        f"for information on which transformations are supported by specific consumer software).",
    )
    kwargs = fields.Kwargs()

    class ScaleRange(PyBioSchema):
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
    name = fields.String(
        validate=validate.OneOf(raw_nodes.PostprocessingName.__args__),
        required=True,
        bioimageio_description=f"Name of postprocessing. One of: {', '.join(raw_nodes.PostprocessingName.__args__)} "
        f"(see [supported_formats_and_operations.md#postprocessing](https://github.com/bioimage-io/configuration/"
        f"blob/master/supported_formats_and_operations.md#postprocessing) "
        f"for information on which transformations are supported by specific consumer software).",
    )
    kwargs = fields.Kwargs()

    class ScaleRange(Preprocessing.ScaleRange):
        reference_tensor: fields.String(required=True, validate=validate.Predicate("isidentifier"))

    class ScaleMeanVariance(PyBioSchema):
        mode = fields.ProcMode(required=True, valid_modes=("per_dataset", "per_sample"))
        reference_tensor: fields.String(required=True, validate=validate.Predicate("isidentifier"))


class InputTensor(Tensor):
    shape = fields.InputShape(required=True, bioimageio_description="Specification of tensor shape.")
    preprocessing = fields.List(
        fields.Nested(Preprocessing),
        missing=list,
        bioimageio_description="Description of how this input should be preprocessed.",
    )
    processing_name = "preprocessing"

    @validates_schema
    def zero_batch_step_and_one_batch_size(self, data, **kwargs):
        axes = data["axes"]
        shape = data["shape"]

        bidx = axes.find("b")
        if bidx == -1:
            return

        if isinstance(shape, raw_nodes.ImplicitInputShape):
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
    shape = fields.OutputShape(required=True)
    halo = fields.List(
        fields.Integer,
        missing=None,
        bioimageio_description="The halo to crop from the output tensor (for example to crop away boundary effects or "
        "for tiling). The halo should be cropped from both sides, i.e. `shape_after_crop = shape - 2 * halo`. The "
        "`halo` is not cropped by the bioimage.io model, but is left to be cropped by the consumer software. Use "
        "`shape:offset` if the model output itself is cropped and input and output shapes not fixed.",
    )
    postprocessing = fields.List(
        fields.Nested(Postprocessing),
        missing=list,
        bioimageio_description="Description of how this output should be postprocessed.",
    )
    processing_name = "postprocessing"

    @validates_schema
    def matching_halo_length(self, data, **kwargs):
        shape = data["shape"]
        halo = data["halo"]
        if halo is None:
            return
        elif isinstance(shape, list) or isinstance(shape, raw_nodes.ImplicitOutputShape):
            if len(halo) != len(shape):
                raise PyBioValidationException(f"halo {halo} has to have same length as shape {shape}!")
        else:
            raise NotImplementedError(type(shape))

    @post_load
    def make_object(self, data, **kwargs):
        shape = data["shape"]
        halo = data["halo"]
        if halo is None:
            data["halo"] = [0] * len(shape)

        return super().make_object(data, **kwargs)


_common_sha256_hint = (
    "You can drag and drop your file to this [online tool]"
    "(http://emn178.github.io/online-tools/sha256_checksum.html) to generate it in your browser. "
    "Or you can generate the SHA256 code for your model and weights by using for example, `hashlib` in Python. "
    # "[here is a codesnippet](#code-snippet-to-compute-sha256-checksum)."  # todo: link to code snippet and don't multiply it
    + """
Code snippet to compute SHA256 checksum

```python
import hashlib

filename = "your filename here"
with open(filename, "rb") as f:
  bytes = f.read() # read entire file as bytes
  readable_hash = hashlib.sha256(bytes).hexdigest()
  print(readable_hash)
  ```

"""
)


class WithFileSource(PyBioSchema):
    source = fields.URI(required=True, bioimageio_description="Link to the source file. Preferably a url.")
    sha256 = fields.String(
        validate=validate.Length(equal=64),
        missing=None,
        bioimageio_description="SHA256 checksum of the source file specified. " + _common_sha256_hint,
    )


class WeightsEntry(WithFileSource):
    authors = fields.List(
        fields.String,
        missing=list,
        bioimageio_description="A list of authors. If this is the root weight (it does not have a `parent` field): the "
        "person(s) that have trained this model. If this is a child weight (it has a `parent` field): the person(s) "
        "who have converted the weights to this format.",
    )  # todo: copy root authors if missing
    attachments = fields.Dict(
        missing=dict,
        bioimageio_description="Dictionary of text keys and URI (or a list of URI) values to additional, relevant "
        "files that are specific to the current weight format. A list of URIs can be listed under the `files` key to "
        "included additional files for generating the model package.",
    )
    parent = fields.String(
        missing=None,
        bioimageio_description="The source weights used as input for converting the weights to this format. For "
        "example, if the weights were converted from the format `pytorch_state_dict` to `pytorch_script`, the parent "
        "is `pytorch_state_dict`. All weight entries except one (the initial set of weights resulting from training "
        "the model), need to have this field.",
    )
    # ONNX Specific
    opset_version = fields.Number(missing=None)
    # tensorflow_saved_model_bundle specific
    tensorflow_version = fields.StrictVersion(missing=None)


class ModelParent(PyBioSchema):
    uri = fields.URI(
        bioimageio_description="Url of another model available on bioimage.io or path to a local model in the "
        "bioimage.io specification. If it is a url, it needs to be a github url linking to the page containing the "
        "model (NOT the raw file)."
    )
    sha256 = fields.SHA256(bioimageio_description="Hash of the weights of the parent model.")


class Model(Spec):
    bioimageio_description = f"""# BioImage.IO Model Description File Specification {raw_nodes.FormatVersion.__args__[-1]}
A model entry in the bioimage.io model zoo is defined by a configuration file model.yaml.
The configuration file must contain the following fields; optional fields are indicated by _optional_.
_optional*_ with an asterisk indicates the field is optional depending on the value in another field.
"""
    name = fields.String(
        # validate=validate.Length(max=36),  # todo: enforce in future version
        required=True,
        bioimageio_description="Name of this model. It should be human-readable and only contain letters, numbers, "
        "`_`, `-` or spaces and not be longer than 36 characters.",
    )

    packaged_by = fields.List(
        fields.String,
        missing=list,
        bioimageio_description=f"The persons that have packaged and uploaded this model. Only needs to be specified if "
        f"different from `authors` in root or any {WeightsEntry.__name__}.",
    )

    parent = fields.Nested(
        ModelParent,
        missing=None,
        bioimageio_description="Parent model from which the trained weights of this model have been derived, e.g. by "
        "finetuning the weights of this model on a different dataset. For format changes of the same trained model "
        "checkpoint, see `weights`.",
    )

    source = fields.ImportableSource(
        missing=None,
        bioimageio_maybe_required=True,
        bioimageio_description="Language and framework specific implementation. As some weights contain the model "
        "architecture, the source is optional depending on the present weight formats. `source` can either point to a "
        "local implementation: `<relative path to file>:<identifier of implementation within the source file>` or the "
        "implementation in an available dependency: `<root-dependency>.<sub-dependency>.<identifier>`.\nFor example: "
        "`./my_function:MyImplementation` or `core_library.some_module.some_function`.",
    )
    sha256 = fields.String(
        validate=validate.Length(equal=64),
        missing=None,
        bioimageio_description="SHA256 checksum of the model source code file."
        + _common_sha256_hint
        + " This field is only required if the field source is present.",
    )
    kwargs = fields.Kwargs(
        bioimageio_description="Keyword arguments for the implementation specified by `source`. "
        "This field is only required if the field `source` is present."
    )

    weights = fields.Dict(
        fields.String(
            validate=validate.OneOf(raw_nodes.WeightsFormat.__args__),
            required=True,
            bioimageio_description=f"Format of this set of weights. Weight formats can define additional (optional or "
            f"required) fields. See [supported_formats_and_operations.md#Weight Format]"
            f"(https://github.com/bioimage-io/configuration/blob/master/supported_formats_and_operations.md#weight_format). "
            f"One of: {', '.join(raw_nodes.WeightsFormat.__args__)}",
        ),
        fields.Nested(WeightsEntry),
        required=True,
        bioimageio_description="The weights for this model. Weights can be given for different formats, but should "
        "otherwise be equivalent. The available weight formats determine which consumers can use this model.",
    )

    inputs = fields.Nested(
        InputTensor, many=True, bioimageio_description="Describes the input tensors expected by this model."
    )
    outputs = fields.Nested(
        OutputTensor, many=True, bioimageio_description="Describes the output tensors from this model."
    )

    test_inputs = fields.List(
        fields.URI,
        required=True,
        bioimageio_description="List of URIs to test inputs as described in inputs for a single test case. "
        "Supported file formats/extensions: '.npy'",
    )
    test_outputs = fields.List(fields.URI, required=True, bioimageio_description="Analog to to test_inputs.")

    sample_inputs = fields.List(
        fields.URI,
        missing=[],
        bioimageio_description="List of URIs to sample inputs to illustrate possible inputs for the model, for example "
        "stored as png or tif images.",
    )
    sample_outputs = fields.List(
        fields.URI,
        missing=[],
        bioimageio_description="List of URIs to sample outputs corresponding to the `sample_inputs`.",
    )

    config = fields.Dict(
        missing=dict,
        bioimageio_description="""
A custom configuration field that can contain any other keys which are not defined above. It can be very specifc to a framework or specific tool. To avoid conflicted definitions, it is recommended to wrap configuration into a sub-field named with the specific framework or tool name.

For example:
```yaml
config:
  # custom config for DeepImageJ, see https://github.com/bioimage-io/configuration/issues/23
  deepimagej:
    model_keys:
      # In principle the tag "SERVING" is used in almost every tf model
      model_tag: tf.saved_model.tag_constants.SERVING
      # Signature definition to call the model. Again "SERVING" is the most general
      signature_definition: tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    test_information:
      input_size: [2048x2048] # Size of the input images
      output_size: [1264x1264 ]# Size of all the outputs
      device: cpu # Device used. In principle either cpu or GPU
      memory_peak: 257.7 Mb # Maximum memory consumed by the model in the device
      runtime: 78.8s # Time it took to run the model
      pixel_size: [9.658E-4µmx9.658E-4µm] # Size of the pixels of the input
```
""",
    )

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
            for postpr in out.postprocessing:
                ref_tensor = postpr.kwargs.get("reference_tensor", None)
                if ref_tensor is not None and ref_tensor not in valid_input_tensor_references:
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


class BioImageIoManifestModelEntry(PyBioSchema):
    id = fields.String(required=True)
    source = fields.String(validate=validate.URL(schemes=["http", "https"]))
    links = fields.List(fields.String, missing=list)
    download_url = fields.String(validate=validate.URL(schemes=["http", "https"]))


class Badge(PyBioSchema):
    label = fields.String(required=True)
    icon = fields.URI()
    url = fields.URI()


class BioImageIoManifestNotebookEntry(PyBioSchema):
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


class BioImageIoManifest(PyBioSchema):
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
