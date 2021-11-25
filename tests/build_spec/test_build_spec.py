import bioimageio.spec as spec
from bioimageio.core import load_raw_resource_description, load_resource_description
from marshmallow import missing


def _test_build_spec(
    spec_path,
    out_path,
    weight_type,
    tensorflow_version=None,
    use_implicit_output_shape=False,
    add_deepimagej_config=False
):
    from bioimageio.core.build_spec import build_model

    model_spec = load_raw_resource_description(spec_path)
    assert isinstance(model_spec, spec.model.raw_nodes.Model)
    weight_source = model_spec.weights[weight_type].source.path

    cite = {entry.text: entry.doi if entry.url is missing else entry.url for entry in model_spec.cite}

    if weight_type == "pytorch_state_dict":
        source_path = model_spec.source.source_file.path
        class_name = model_spec.source.callable_name
        model_source = f"{source_path}:{class_name}"
        weight_type_ = None  # the weight type can be auto-detected
    elif weight_type == "pytorch_script":
        model_source = None
        weight_type_ = "pytorch_script"  # the weight type CANNOT be auto-detcted
    else:
        model_source = None
        weight_type_ = None  # the weight type can be auto-detected

    dep_file = None if model_spec.dependencies is missing else model_spec.dependencies.file.path
    authors = [{"name": auth.name, "affiliation": auth.affiliation} for auth in model_spec.authors]
    covers = [cover.path for cover in model_spec.covers]
    kwargs = dict(
        source=model_source,
        model_kwargs=model_spec.kwargs,
        weight_uri=weight_source,
        test_inputs=[inp.path for inp in model_spec.test_inputs],
        test_outputs=[outp.path for outp in model_spec.test_outputs],
        name=model_spec.name,
        description=model_spec.description,
        authors=authors,
        tags=model_spec.tags,
        license=model_spec.license,
        documentation=model_spec.documentation,
        covers=covers,
        dependencies=dep_file,
        cite=cite,
        root=model_spec.root_path,
        weight_type=weight_type_,
        output_path=out_path,
        add_deepimagej_config=add_deepimagej_config,
    )
    if tensorflow_version is not None:
        kwargs["tensorflow_version"] = tensorflow_version
    if use_implicit_output_shape:
        kwargs["input_name"] = ["input"]
        kwargs["output_reference"] = ["input"]
        kwargs["output_scale"] = [[1.0, 1.0, 1.0, 1.0]]
        kwargs["output_offset"] = [[0.0, 0.0, 0.0, 0.0]]

    build_model(**kwargs)
    assert out_path.exists()
    loaded_model = load_resource_description(out_path)
    if add_deepimagej_config:
        loaded_config = loaded_model.config
        assert "deepimagej" in loaded_config


def test_build_spec_pytorch(any_torch_model, tmp_path):
    _test_build_spec(any_torch_model, tmp_path / "model.zip", "pytorch_state_dict")


def test_build_spec_implicit_output_shape(unet2d_nuclei_broad_model, tmp_path):
    _test_build_spec(
        unet2d_nuclei_broad_model, tmp_path / "model.zip", "pytorch_state_dict", use_implicit_output_shape=True
    )


def test_build_spec_torchscript(any_torchscript_model, tmp_path):
    _test_build_spec(any_torchscript_model, tmp_path / "model.zip", "pytorch_script")


def test_build_spec_onnx(any_onnx_model, tmp_path):
    _test_build_spec(any_onnx_model, tmp_path / "model.zip", "onnx")


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
    _test_build_spec(unet2d_nuclei_broad_model, tmp_path / "model.zip", "pytorch_script", add_deepimagej_config=True)
