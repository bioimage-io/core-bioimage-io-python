import datetime
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from ruamel.yaml import YAML

import bioimageio.spec as spec
import bioimageio.spec.model as model_spec

try:
    from typing import get_args
except ImportError:
    from typing_extensions import get_args  # type: ignore

#
# utility functions to build the spec from python
#


def _get_local_path(uri, root=None):
    is_local_path = os.path.exists(uri)
    if not is_local_path and root is not None:
        uri2 = os.path.join(root, uri)
        if os.path.exists(uri2):
            uri = uri2
            is_local_path = True
    if not is_local_path:
        uri = spec.shared.fields.URI().deserialize(uri)
        uri = spec.shared.download_uri_to_local_path(uri).as_posix()
    return uri


def _get_hash(path):
    with open(path, "rb") as f:
        data = f.read()
        return hashlib.sha256(data).hexdigest()


def _infer_weight_type(path):
    ext = os.path.splitext(path)[-1]
    if ext in (".pt", ".torch"):
        return "pytorch_state_dict"
    elif ext == ".onnx":
        return "onnx"
    elif ext in (".hdf", ".hdf5", ".h5"):
        return "keras_hdf5"
    elif ext == ".zip":
        return "tensorflow_saved_model_bundle"
    elif ext == ".json":
        return "tensorflow_js"
    else:
        raise ValueError(f"Could not infer weight type from extension {ext} for weight file {path}")


def _get_weights(weight_uri, weight_type, source, root, **kwargs):
    weight_path = _get_local_path(weight_uri, root)
    if weight_type is None:
        weight_type = _infer_weight_type(weight_path)
    weight_hash = _get_hash(weight_path)

    # if we have a ":" (or deprecated "::") this is a python file with class specified,
    # so we can compute the hash for it
    if source is not None and ":" in source:
        source_path = _get_local_path(":".join(source.replace("::", ":").split(":")[:-1]), root)
        source_hash = _get_hash(source_path)
    else:
        source_hash = None

    if "weight_attachments" in kwargs:
        attachments = {"attachments": ["weight_attachments"]}
    else:
        attachments = {}

    weight_types = model_spec.raw_nodes.WeightsFormat
    if weight_type == "pytorch_state_dict":
        # pytorch-state-dict -> we need a source
        assert source is not None
        weights = model_spec.raw_nodes.PytorchStateDictWeightsEntry(
            source=weight_uri, sha256=weight_hash, **attachments
        )
        language = "python"
        framework = "pytorch"

    elif weight_type == "onnx":
        weights = model_spec.raw_nodes.OnnxWeightsEntry(
            source=weight_uri, sha256=weight_hash, opset_version=kwargs.get("opset_version", 12), **attachments
        )
        language = None
        framework = None

    elif weight_type == "pytorch_script":
        weights = model_spec.raw_nodes.PytorchScriptWeightsEntry(source=weight_uri, sha256=weight_hash, **attachments)
        if source is None:
            language = None
            framework = None
        else:
            language = "python"
            framework = "pytorch"

    elif weight_type == "keras_hdf5":
        weights = model_spec.raw_nodes.KerasHdf5WeightsEntry(
            source=weight_uri,
            sha256=weight_hash,
            tensorflow_version=kwargs.get("tensorflow_version", "1.15"),
            **attachments,
        )
        language = "python"
        framework = "tensorflow"

    elif weight_type == "tensorflow_saved_model_bundle":
        weights = model_spec.raw_nodes.TensorflowSavedModelBundleWeightsEntry(
            source=weight_uri,
            sha256=weight_hash,
            tensorflow_version=kwargs.get("tensorflow_version", "1.15"),
            **attachments,
        )
        language = "python"
        framework = "tensorflow"

    elif weight_type == "tensorflow_js":
        weights = model_spec.raw_nodes.TensorflowJsWeightsEntry(source=weight_uri, sha256=weight_hash, **attachments)
        language = None
        framework = None

    elif weight_type in weight_types:
        raise ValueError(f"Weight type {weight_type} is not supported yet in 'build_spec'")
    else:
        raise ValueError(f"Invalid weight type {weight_type}, expect one of {weight_types}")

    weights = {weight_type: weights}
    return weights, language, framework, source_hash


def _get_data_range(data_range, dtype):
    if data_range is None:
        if np.issubdtype(np.dtype(dtype), np.integer):
            min_, max_ = np.iinfo(dtype).min, np.iinfo(dtype).max
        # for floating point numbers we assume valid range from -inf to inf
        elif np.issubdtype(np.dtype(dtype), np.floating):
            min_, max_ = -np.inf, np.inf
        elif np.issubdtype(np.dtype(dtype), np.bool):
            min_, max_ = 0, 1
        else:
            raise RuntimeError(f"Cannot derived data range for dtype {dtype}")
        data_range = (min_, max_)
    assert isinstance(data_range, (tuple, list))
    assert len(data_range) == 2
    return data_range


def _get_axes(axes, ndim):
    if axes is None:
        assert ndim in (2, 4, 5)
        default_axes = {2: "bc", 4: "bcyx", 5: "bczyx"}
        axes = default_axes[ndim]
    return axes


def _get_input_tensor(test_in, name, step, min_shape, data_range, axes, preprocessing):
    shape = test_in.shape
    if step is None:
        assert min_shape is None
        shape_description = shape
    else:
        shape_description = {"min": shape if min_shape is None else min_shape, "step": step}

    axes = _get_axes(axes, test_in.ndim)
    data_range = _get_data_range(data_range, test_in.dtype)

    kwargs = {}
    if preprocessing is not None:
        kwargs["preprocessing"] = preprocessing

    inputs = model_spec.raw_nodes.InputTensor(
        name="input" if name is None else name,
        data_type=str(test_in.dtype),
        axes=axes,
        shape=shape_description,
        data_range=data_range,
        **kwargs,
    )
    return inputs


def _get_output_tensor(test_out, name, reference_input, scale, offset, axes, data_range, postprocessing, halo):
    shape = test_out.shape
    if reference_input is None:
        assert scale is None
        assert offset is None
        shape_description = shape
    else:
        assert scale is not None
        assert offset is not None
        shape_description = {"reference_input": reference_input, "scale": scale, "offset": offset}

    axes = _get_axes(axes, test_out.ndim)
    data_range = _get_data_range(data_range, test_out.dtype)

    kwargs = {}
    if postprocessing is not None:
        kwargs["postprocessing"] = postprocessing
    if halo is not None:
        kwargs["halo"] = halo

    outputs = model_spec.raw_nodes.OutputTensor(
        name="output" if name is None else name,
        data_type=str(test_out.dtype),
        axes=axes,
        data_range=data_range,
        shape=shape_description,
        **kwargs,
    )
    return outputs


def _build_authors(authors: List[Dict[str, str]]):
    return [model_spec.raw_nodes.Author(**a) for a in authors]


# TODO The citation entry should be improved so that we can properly derive doi vs. url
def _build_cite(cite: Dict[str, str]):
    citation_list = [model_spec.raw_nodes.CiteEntry(text=k, url=v) for k, v in cite.items()]
    return citation_list


# TODO we should make the name more specific: "build_model_spec"?
# TODO maybe "build_raw_model" as it return raw_nodes.Model
# NOTE does not support multiple input / output tensors yet
# to implement this we should wait for 0.4.0, see also
# https://github.com/bioimage-io/spec-bioimage-io/issues/70#issuecomment-825737433
def build_spec(
    weight_uri: str,
    test_inputs: List[str],
    test_outputs: List[str],
    # general required
    name: str,
    description: str,
    authors: List[Dict[str, str]],
    tags: List[str],
    license: str,
    documentation: Union[str, Path],
    covers: List[str],
    cite: Dict[str, str],
    root: Optional[str] = None,
    # model specific optional
    source: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Union[int, float, str]]] = None,
    weight_type: Optional[str] = None,
    sample_inputs: Optional[str] = None,
    sample_outputs: Optional[str] = None,
    # tensor specific
    input_name: Optional[str] = None,
    input_step: Optional[List[int]] = None,
    input_min_shape: Optional[List[int]] = None,
    input_axes: Optional[str] = None,
    input_data_range: Optional[List[Union[int, str]]] = None,
    output_name: Optional[str] = None,
    output_reference: Optional[str] = None,
    output_scale: Optional[List[int]] = None,
    output_offset: Optional[List[int]] = None,
    output_axes: Optional[str] = None,
    output_data_range: Optional[List[Union[int, str]]] = None,
    halo: Optional[List[int]] = None,
    preprocessing: Optional[List[Dict[str, Dict[str, Union[int, float, str]]]]] = None,
    postprocessing: Optional[List[Dict[str, Dict[str, Union[int, float, str]]]]] = None,
    # general optional
    git_repo: Optional[str] = None,
    attachments: Optional[Dict[str, Union[str, List[str]]]] = None,
    packaged_by: Optional[List[str]] = None,
    run_mode: Optional[str] = None,
    parent: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    dependencies: Optional[str] = None,
    links: Optional[List[str]] = None,
    **weight_kwargs,
):
    """Create a bioimageio.spec.model.raw_nodes.Model object that can be used to serialize a model.yaml
    in the bioimage.io format.

    Example usage:
    ```
    import bioimageio.spec as spec
    model_spec = spec.build_spec(
        weight_uri="test_weights.pt",
        test_inputs=["./test_inputs"],
        test_outputs=["./test_outputs"],
        name="My Model",
        description="My very fancy model.",
        authors=[{"name": "John Doe", "affiliation": "My Institute"}],
        tags=["segmentation", "light sheet data"],
        license="CC-BY",
        documentation="./documentation.md",
        covers=["./my_cover.png"],
        cite={"Architecture": "https://my_architecture.com"}
    )
    spec.serialize_pec(model_spec, "model.yaml")
    ```

    Args:
        weight_uri: the url or relative local file path to the weight file for this model.
        test_inputs: list of test input files stored in numpy format.
        test_outputs: list of test outputs corresponding to test_inputs, stored in numpy format.
        name: name of this model.
        description: short description of this model.
        authors: the authors of this model.
        tags: list of tags for this model.
        license: the license for this model.
        documentation: relative file path to markdown documentation for this model.
        covers: list of relative file paths for cover images.
        cite: citations for this model.
        root: root folder for arguments given as relative paths.
        source: the file with the source code for the model architecture and the corresponding class.
        model_kwargs: the keyword arguments for the model class.
        weight_type: the type of the weights.
        sample_inputs: list of sample inputs to demonstrate the model performance.
        sample_outputs: list of sample outputs corresponding to sample_inputs.
        input_name: name of the input tensor.
        input_step: minimal valid increase of the input tensor shape.
        input_min_shape: minimal input tensor shape.
        input_axes: axes names for the input tensor.
        input_data_range: valid data range for the input tensor.
        output_name: name of the output tensor.
        output_reference: name of the input reference tensor used to cimpute the output tensor shape.
        output_scale: multiplicative factor to compute the output tensor shape.
        output_offset: additive term to compute the output tensor shape.
        output_axes: axes names of the output tensor.
        output_data_range: valid data range for the output tensor.
        halo: halo to be cropped from the output tensor.
        preprocessing: list of preprocessing operations for the input.
        postprocessing: list of postprocessing operations for the output.
        git_repo: reference git repository for this model.
        attachments: list of additional files to package with the model.
        packaged_by: list of authors that have packaged this model.
        run_mode: custom run mode for this model.
        parent: id of the parent mode from which this model is derived.
        config: custom configuration for this model.
        dependencies: relative path to file with dependencies for this model.
        weight_kwargs: keyword arguments for this weight type, e.g. "tensorflow_version".
    """
    #
    # generate the model specific fields
    #

    # check the test inputs and auto-generate input/output description from test inputs/outputs
    for test_in, test_out in zip(test_inputs, test_outputs):
        test_in, test_out = _get_local_path(test_in, root), _get_local_path(test_out, root)
        test_in, test_out = np.load(test_in), np.load(test_out)
    inputs = _get_input_tensor(
        test_in, input_name, input_step, input_min_shape, input_axes, input_data_range, preprocessing
    )
    outputs = _get_output_tensor(
        test_out,
        output_name,
        output_reference,
        output_scale,
        output_offset,
        output_axes,
        output_data_range,
        postprocessing,
        halo,
    )

    (weights, language, framework, source_hash) = _get_weights(weight_uri, weight_type, source, root, **weight_kwargs)

    #
    # generate general fields
    #
    format_version = get_args(model_spec.raw_nodes.FormatVersion)[-1]
    timestamp = datetime.datetime.now()

    if source is not None:
        source = spec.shared.fields.ImportableSource().deserialize(source)

    # optional kwargs, don't pass them if none
    optional_kwargs = {
        "git_repo": git_repo,
        "attachments": attachments,
        "packaged_by": packaged_by,
        "parent": parent,
        "run_mode": run_mode,
        "config": config,
        "sample_inputs": sample_inputs,
        "sample_outputs": sample_outputs,
        "framework": framework,
        "language": language,
        "source": source,
        "sha256": source_hash,
        "kwargs": model_kwargs,
        "dependencies": dependencies,
        "links": links,
    }
    kwargs = {k: v for k, v in optional_kwargs.items() if v is not None}

    # build raw_nodes objects
    authors = _build_authors(authors)
    cite = _build_cite(cite)
    documentation = Path(documentation)
    covers = [spec.shared.fields.URI().deserialize(uri) for uri in covers]
    test_inputs = [spec.shared.fields.URI().deserialize(uri) for uri in test_inputs]
    test_outputs = [spec.shared.fields.URI().deserialize(uri) for uri in test_outputs]

    model = model_spec.raw_nodes.Model(
        format_version=format_version,
        name=name,
        description=description,
        authors=authors,
        cite=cite,
        tags=tags,
        license=license,
        documentation=documentation,
        covers=covers,
        timestamp=timestamp,
        weights=weights,
        inputs=[inputs],
        outputs=[outputs],
        test_inputs=test_inputs,
        test_outputs=test_outputs,
        **kwargs,
    )

    # serialize and deserialize the raw_nodes.Model to
    # check that all fields are well formed
    serialized = model_spec.schema.Model().dump(model)
    model = model_spec.schema.Model().load(serialized)

    return model


def add_weights(model, weight_uri: str, root: Optional[str] = None, weight_type: Optional[str] = None, **weight_kwargs):
    """Add weight entry to bioimage.io model."""
    new_weights = _get_weights(weight_uri, weight_type, None, root, **weight_kwargs)[0]
    model.weights.update(new_weights)

    serialized = model_spec.schema.Model().dump(model)
    model = model_spec.schema.Model().load(serialized)

    return model


def serialize_spec(model, out_path):  # TODO change name to include model (see build_model_spec)
    yaml = YAML(typ="safe")
    serialized = model_spec.schema.Model().dump(model)
    with open(out_path, "w") as f:
        yaml.dump(serialized, f)
