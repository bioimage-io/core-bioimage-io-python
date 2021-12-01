import bioimageio.spec as spec
from bioimageio.core import load_raw_resource_description, load_resource_description
from bioimageio.core.resource_io import nodes
from bioimageio.core.resource_io.utils import resolve_source
from marshmallow import missing


def _test_build_spec(
    spec_path,
    out_path,
    weight_type,
    tensorflow_version=None,
    opset_version=None,
    use_implicit_output_shape=False,
    add_deepimagej_config=False,
    use_original_covers=False,
):
    from bioimageio.core.build_spec import build_model

    model_spec = load_raw_resource_description(spec_path)
    root = model_spec.root_path
    assert isinstance(model_spec, spec.model.raw_nodes.Model)
    weight_source = model_spec.weights[weight_type].source

    cite = {entry.text: entry.doi if entry.url is missing else entry.url for entry in model_spec.cite}

    if weight_type == "pytorch_state_dict":
        weight_spec = model_spec.weights["pytorch_state_dict"]
        model_kwargs = None if weight_spec.kwargs is missing else weight_spec.kwargs
        architecture = str(weight_spec.architecture)
        weight_type_ = None  # the weight type can be auto-detected
    elif weight_type == "torchscript":
        architecture = None
        model_kwargs = None
        weight_type_ = "torchscript"  # the weight type CANNOT be auto-detcted
    else:
        architecture = None
        model_kwargs = None
        weight_type_ = None  # the weight type can be auto-detected

    dep_file = None if model_spec.dependencies is missing else resolve_source(model_spec.dependencies.file, root)
    authors = [{"name": auth.name, "affiliation": auth.affiliation} for auth in model_spec.authors]

    input_axes = [input_.axes for input_ in model_spec.inputs]
    output_axes = [output.axes for output in model_spec.outputs]
    preprocessing = [
        None if input_.preprocessing == missing else {preproc.name: preproc.kwargs for preproc in input_.preprocessing}
        for input_ in model_spec.inputs
    ]
    postprocessing = [
        None if output.postprocessing == missing else {preproc.name: preproc.kwargs for preproc in output.preprocessing}
        for output in model_spec.outputs
    ]

    kwargs = dict(
        weight_uri=weight_source,
        test_inputs=resolve_source(model_spec.test_inputs, root),
        test_outputs=resolve_source(model_spec.test_outputs, root),
        name=model_spec.name,
        description=model_spec.description,
        authors=authors,
        tags=model_spec.tags,
        license=model_spec.license,
        documentation=model_spec.documentation,
        dependencies=dep_file,
        cite=cite,
        root=model_spec.root_path,
        weight_type=weight_type_,
        input_axes=input_axes,
        output_axes=output_axes,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
        output_path=out_path,
        add_deepimagej_config=add_deepimagej_config,
    )
    if architecture is not None:
        kwargs["architecture"] = architecture
    if model_kwargs is not None:
        kwargs["kwargs"] = model_kwargs
    if tensorflow_version is not None:
        kwargs["tensorflow_version"] = tensorflow_version
    if opset_version is not None:
        kwargs["opset_version"] = opset_version
    if use_implicit_output_shape:
        kwargs["input_name"] = ["input"]
        kwargs["output_reference"] = ["input"]
        kwargs["output_scale"] = [[1.0, 1.0, 1.0, 1.0]]
        kwargs["output_offset"] = [[0.0, 0.0, 0.0, 0.0]]
    if add_deepimagej_config:
        kwargs["pixel_sizes"] = [{"x": 5.0, "y": 5.0}]
    if use_original_covers:
        kwargs["covers"] = resolve_source(model_spec.covers, root)

    build_model(**kwargs)
    assert out_path.exists()
    loaded_model = load_resource_description(out_path)
    assert isinstance(loaded_model, nodes.Model)
    if add_deepimagej_config:
        loaded_config = loaded_model.config
        assert "deepimagej" in loaded_config

    attachments = loaded_model.attachments
    if attachments is not missing and attachments.files is not missing:
        for attached_file in attachments["files"]:
            assert attached_file.exists()


def test_build_spec_pytorch(any_torch_model, tmp_path):
    _test_build_spec(any_torch_model, tmp_path / "model.zip", "pytorch_state_dict")


def test_build_spec_implicit_output_shape(unet2d_nuclei_broad_model, tmp_path):
    _test_build_spec(
        unet2d_nuclei_broad_model, tmp_path / "model.zip", "pytorch_state_dict", use_implicit_output_shape=True
    )


def test_build_spec_torchscript(any_torchscript_model, tmp_path):
    _test_build_spec(any_torchscript_model, tmp_path / "model.zip", "torchscript")


def test_build_spec_onnx(any_onnx_model, tmp_path):
    _test_build_spec(any_onnx_model, tmp_path / "model.zip", "onnx", opset_version=12)


def test_build_spec_keras(any_keras_model, tmp_path):
    _test_build_spec(
        any_keras_model, tmp_path / "model.zip", "keras_hdf5", tensorflow_version="1.12"
    )  # todo: keras for tf 2??


def test_build_spec_tf(any_tensorflow_model, tmp_path):
    _test_build_spec(
        any_tensorflow_model, tmp_path / "model.zip", "tensorflow_saved_model_bundle", tensorflow_version="1.12"
    )  # check tf version


def test_build_spec_tfjs(any_tensorflow_js_model, tmp_path):
    _test_build_spec(any_tensorflow_js_model, tmp_path / "model.zip", "tensorflow_js", tensorflow_version="1.12")


def test_build_spec_deepimagej(unet2d_nuclei_broad_model, tmp_path):
    _test_build_spec(unet2d_nuclei_broad_model, tmp_path / "model.zip", "torchscript", add_deepimagej_config=True)


def test_build_spec_deepimagej_keras(unet2d_keras, tmp_path):
    _test_build_spec(
        unet2d_keras, tmp_path / "model.zip", "keras_hdf5", add_deepimagej_config=True, tensorflow_version="1.12"
    )


# test with original covers
def test_build_spec_with_original_covers(unet2d_nuclei_broad_model, tmp_path):
    _test_build_spec(unet2d_nuclei_broad_model, tmp_path / "model.zip", "torchscript", use_original_covers=True)
