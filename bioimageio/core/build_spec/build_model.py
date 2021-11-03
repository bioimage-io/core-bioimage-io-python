import datetime
import hashlib
import os
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np

import bioimageio.spec as spec
import bioimageio.spec.model as model_spec
from bioimageio.core import export_resource_package, load_raw_resource_description
from bioimageio.core.resource_io.utils import resolve_uri

try:
    from typing import get_args
except ImportError:
    from typing_extensions import get_args  # type: ignore

#
# utility functions to build the spec from python
#


def _get_hash(path):
    with open(path, "rb") as f:
        data = f.read()
        return hashlib.sha256(data).hexdigest()


def _process_uri(uri: Union[str, Path], root: Path, download=False):
    if os.path.exists(uri):
        return Path(uri)
    elif (root / uri).exists():
        return root / uri
    elif isinstance(uri, str) and uri.startswith("http"):
        return resolve_uri(uri, root) if download else uri
    else:
        raise ValueError(f"Invalid uri: {uri}")


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
    weight_path = _process_uri(weight_uri, root, download=True)
    if weight_type is None:
        weight_type = _infer_weight_type(weight_path)
    weight_hash = _get_hash(weight_path)

    tmp_source = None
    # if we have a ":" (or deprecated "::") this is a python file with class specified,
    # so we can compute the hash for it
    if source is not None and ":" in source:
        source_file, source_class = source.replace("::", ":").split(":")

        # get the source path
        source_file = _process_uri(source_file, root, download=True)
        source_hash = _get_hash(source_file)

        # if not relative, create local copy (otherwise this will not work)
        if os.path.isabs(source_file):
            copyfile(source_file, "this_model_architecture.py")
            source = f"this_model_architecture.py:{source_class}"
            tmp_source = "this_model_architecture.py"
        else:
            source = f"{source_file}:{source_class}"
        source = spec.shared.fields.ImportableSource().deserialize(source)
    else:
        source_hash = None

    if "weight_attachments" in kwargs:
        attachments = {"attachments": ["weight_attachments"]}
    else:
        attachments = {}

    weight_types = model_spec.raw_nodes.WeightsFormat
    weight_source = _process_uri(weight_uri, root)
    if weight_type == "pytorch_state_dict":
        # pytorch-state-dict -> we need a source
        assert source is not None
        weights = model_spec.raw_nodes.PytorchStateDictWeightsEntry(
            source=weight_source, sha256=weight_hash, **attachments
        )
        language = "python"
        framework = "pytorch"

    elif weight_type == "onnx":
        weights = model_spec.raw_nodes.OnnxWeightsEntry(
            source=weight_source, sha256=weight_hash, opset_version=kwargs.get("opset_version", 12), **attachments
        )
        language = None
        framework = None

    elif weight_type == "pytorch_script":
        weights = model_spec.raw_nodes.PytorchScriptWeightsEntry(
            source=weight_source, sha256=weight_hash, **attachments
        )
        if source is None:
            language = None
            framework = None
        else:
            language = "python"
            framework = "pytorch"

    elif weight_type == "keras_hdf5":
        weights = model_spec.raw_nodes.KerasHdf5WeightsEntry(
            source=weight_source,
            sha256=weight_hash,
            tensorflow_version=kwargs.get("tensorflow_version", "1.15"),
            **attachments,
        )
        language = "python"
        framework = "tensorflow"

    elif weight_type == "tensorflow_saved_model_bundle":
        weights = model_spec.raw_nodes.TensorflowSavedModelBundleWeightsEntry(
            source=weight_source,
            sha256=weight_hash,
            tensorflow_version=kwargs.get("tensorflow_version", "1.15"),
            **attachments,
        )
        language = "python"
        framework = "tensorflow"

    elif weight_type == "tensorflow_js":
        weights = model_spec.raw_nodes.TensorflowJsWeightsEntry(
            source=weight_uri,
            sha256=weight_hash,
            tensorflow_version=kwargs.get("tensorflow_version", "1.15"),
            **attachments,
        )
        language = None
        framework = None

    elif weight_type in weight_types:
        raise ValueError(f"Weight type {weight_type} is not supported yet in 'build_spec'")
    else:
        raise ValueError(f"Invalid weight type {weight_type}, expect one of {weight_types}")

    weights = {weight_type: weights}
    return weights, language, framework, source, source_hash, tmp_source


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
    assert isinstance(data_range, (tuple, list)), type(data_range)
    assert len(data_range) == 2
    return data_range


def _get_axes(axes, ndim):
    if axes is None:
        assert ndim in (2, 4, 5)
        default_axes = {2: "bc", 4: "bcyx", 5: "bczyx"}
        axes = default_axes[ndim]
    return axes


def _get_input_tensor(path, name, step, min_shape, data_range, axes, preprocessing):
    test_in = np.load(path)
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
        kwargs["preprocessing"] = [{"name": k, "kwargs": v} for k, v in preprocessing.items()]

    inputs = model_spec.raw_nodes.InputTensor(
        name="input" if name is None else name,
        data_type=str(test_in.dtype),
        axes=axes,
        shape=shape_description,
        data_range=data_range,
        **kwargs,
    )
    return inputs


def _get_output_tensor(path, name, reference_tensor, scale, offset, axes, data_range, postprocessing, halo):
    test_out = np.load(path)
    shape = test_out.shape
    if reference_tensor is None:
        assert scale is None
        assert offset is None
        shape_description = shape
    else:
        assert scale is not None
        assert offset is not None
        shape_description = {"reference_tensor": reference_tensor, "scale": scale, "offset": offset}

    axes = _get_axes(axes, test_out.ndim)
    data_range = _get_data_range(data_range, test_out.dtype)

    kwargs = {}
    if postprocessing is not None:
        kwargs["postprocessing"] = [{"name": k, "kwargs": v} for k, v in postprocessing.items()]
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


def _get_dependencies(dependencies, root):
    if ":" in dependencies:
        manager, path = dependencies.split(":")
    else:
        manager = "conda"
        path = dependencies
    return model_spec.raw_nodes.Dependencies(
        manager=manager, file=_process_uri(path, root)
    )


def build_model(
    weight_uri: str,
    test_inputs: List[Union[str, Path]],
    test_outputs: List[Union[str, Path]],
    # general required
    name: str,
    description: str,
    authors: List[Dict[str, str]],
    tags: List[Union[str, Path]],
    license: str,
    documentation: Union[str, Path],
    covers: List[str],
    cite: Dict[str, str],
    output_path: Union[str, Path],
    # model specific optional
    source: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Union[int, float, str]]] = None,
    weight_type: Optional[str] = None,
    sample_inputs: Optional[str] = None,
    sample_outputs: Optional[str] = None,
    # tensor specific
    input_name: Optional[List[str]] = None,
    input_step: Optional[List[List[int]]] = None,
    input_min_shape: Optional[List[List[int]]] = None,
    input_axes: Optional[List[str]] = None,
    input_data_range: Optional[List[List[Union[int, str]]]] = None,
    output_name: Optional[List[str]] = None,
    output_reference: Optional[List[str]] = None,
    output_scale: Optional[List[List[int]]] = None,
    output_offset: Optional[List[List[int]]] = None,
    output_axes: Optional[List[str]] = None,
    output_data_range: Optional[List[List[Union[int, str]]]] = None,
    halo: Optional[List[List[int]]] = None,
    preprocessing: Optional[List[Dict[str, Dict[str, Union[int, float, str]]]]] = None,
    postprocessing: Optional[List[Dict[str, Dict[str, Union[int, float, str]]]]] = None,
    # general optional
    git_repo: Optional[str] = None,
    attachments: Optional[Dict[str, Union[str, List[str]]]] = None,
    packaged_by: Optional[List[str]] = None,
    run_mode: Optional[str] = None,
    parent: Optional[Tuple[str, str]] = None,
    config: Optional[Dict[str, Any]] = None,
    dependencies: Optional[str] = None,
    links: Optional[List[str]] = None,
    root: Optional[Union[Path, str]] = None,
    **weight_kwargs,
):
    """Create a zipped bioimage.io model.

    Example usage:
    ```
    from pathlib import Path
    import bioimageio.spec as spec
    import bioimageio.core.build_spec as build_spec
    model_spec = build_spec.build_model(
        weight_uri="test_weights.pt",
        test_inputs=["./test_inputs"],
        test_outputs=["./test_outputs"],
        name="my-model",
        description="My very fancy model.",
        authors=[{"name": "John Doe", "affiliation": "My Institute"}],
        tags=["segmentation", "light sheet data"],
        license="CC-BY",
        documentation="./documentation.md",
        covers=["./my_cover.png"],
        cite={"Architecture": "https://my_architecture.com"},
        output_path="my-model.zip"
    )
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
        output_path: where to save the zipped model package.
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
        parent: id of the parent model from which this model is derived and sha256 of the corresponding weight file.
        config: custom configuration for this model.
        dependencies: relative path to file with dependencies for this model.
        root: optional root path for relative paths. This can be helpful when building a spec from another model spec.
        weight_kwargs: keyword arguments for this weight type, e.g. "tensorflow_version".
    """
    if root is None:
        root = "."
    root = Path(root)

    #
    # generate the model specific fields
    #

    assert len(test_inputs)
    assert len(test_outputs)
    test_inputs = [_process_uri(uri, root) for uri in test_inputs]
    test_outputs = [_process_uri(uri, root) for uri in test_outputs]

    n_inputs = len(test_inputs)
    input_name = n_inputs * [None] if input_name is None else input_name
    input_step = n_inputs * [None] if input_step is None else input_step
    input_min_shape = n_inputs * [None] if input_min_shape is None else input_min_shape
    input_axes = n_inputs * [None] if input_axes is None else input_axes
    input_data_range = n_inputs * [None] if input_data_range is None else input_data_range
    preprocessing = n_inputs * [None] if preprocessing is None else preprocessing

    inputs = [
        _get_input_tensor(test_in, name, step, min_shape, data_range, axes, preproc)
        for test_in, name, step, min_shape, axes, data_range, preproc in zip(
            test_inputs, input_name, input_step, input_min_shape, input_axes, input_data_range, preprocessing
        )
    ]

    n_outputs = len(test_outputs)
    output_name = n_outputs * [None] if output_name is None else output_name
    output_reference = n_outputs * [None] if output_reference is None else output_reference
    output_scale = n_outputs * [None] if output_scale is None else output_scale
    output_offset = n_outputs * [None] if output_offset is None else output_offset
    output_axes = n_outputs * [None] if output_axes is None else output_axes
    output_data_range = n_outputs * [None] if output_data_range is None else output_data_range
    postprocessing = n_outputs * [None] if postprocessing is None else postprocessing
    halo = n_outputs * [None] if halo is None else halo

    outputs = [
        _get_output_tensor(test_out, name, reference, scale, offset, axes, data_range, postproc, hal)
        for test_out, name, reference, scale, offset, axes, data_range, postproc, hal in zip(
            test_outputs,
            output_name,
            output_reference,
            output_scale,
            output_offset,
            output_axes,
            output_data_range,
            postprocessing,
            halo,
        )
    ]

    #
    # generate general fields
    #
    format_version = get_args(model_spec.raw_nodes.FormatVersion)[-1]
    timestamp = datetime.datetime.now()

    authors = _build_authors(authors)
    cite = _build_cite(cite)
    documentation = _process_uri(documentation, root, download=True)
    covers = [_process_uri(uri, root, download=True) for uri in covers]

    # parse the weights
    weights, language, framework, source, source_hash, tmp_source = _get_weights(
        weight_uri, weight_type, source, root, **weight_kwargs
    )

    # optional kwargs, don't pass them if none
    optional_kwargs = {
        "git_repo": git_repo,
        "attachments": attachments,
        "packaged_by": packaged_by,
        "run_mode": run_mode,
        "config": config,
        "sample_inputs": sample_inputs,
        "sample_outputs": sample_outputs,
        "framework": framework,
        "language": language,
        "source": source,
        "sha256": source_hash,
        "kwargs": model_kwargs,
        "links": links,
    }
    kwargs = {k: v for k, v in optional_kwargs.items() if v is not None}
    if dependencies is not None:
        kwargs["dependencies"] = _get_dependencies(dependencies, root)
    if parent is not None:
        assert len(parent) == 2
        kwargs["parent"] = {"uri": parent[0], "sha256": parent[1]}

    try:
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
            inputs=inputs,
            outputs=outputs,
            test_inputs=test_inputs,
            test_outputs=test_outputs,
            **kwargs,
        )
        model_package = export_resource_package(model, output_path=output_path)
    except Exception as e:
        raise e
    finally:
        if tmp_source is not None:
            os.remove(tmp_source)

    model = load_raw_resource_description(model_package)
    return model


def add_weights(
    model,
    weight_uri: Union[str, Path],
    weight_type: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    **weight_kwargs,
):
    """Add weight entry to bioimage.io model."""
    # we need to patss the weight path as abs path to avoid confusion with different root directories
    new_weights = _get_weights(Path(weight_uri).absolute(), weight_type, source=None, root=Path("."), **weight_kwargs)[
        0
    ]
    model.weights.update(new_weights)
    if output_path is not None:
        model_package = export_resource_package(model, output_path=output_path)
        model = load_raw_resource_description(model_package)
    return model
