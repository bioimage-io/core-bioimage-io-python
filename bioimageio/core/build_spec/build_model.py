import datetime
import hashlib
import os
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests

import bioimageio.spec as spec
import bioimageio.spec.model as model_spec
from bioimageio.core import export_resource_package, load_raw_resource_description
from bioimageio.core.resource_io.nodes import URI
from bioimageio.core.resource_io.utils import resolve_local_source, resolve_source

try:
    from typing import get_args
except ImportError:
    from typing_extensions import get_args  # type: ignore

# need tifffile for writing the deepimagej config
# we probably always have this, but wrap into an ImportGuard just in case
try:
    import tifffile
except ImportError:
    tifffile = None

#
# utility functions to build the spec from python
#


def _get_hash(path):
    with open(path, "rb") as f:
        data = f.read()
        return hashlib.sha256(data).hexdigest()


def _infer_weight_type(path):
    ext = os.path.splitext(path)[-1]
    if ext in (".pt", ".pth", ".torch"):
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


def _get_weights(original_weight_source, weight_type, source, root, **kwargs):
    weight_path = resolve_source(original_weight_source, root)
    if weight_type is None:
        weight_type = _infer_weight_type(weight_path)
    weight_hash = _get_hash(weight_path)

    tmp_source = None
    # if we have a ":" (or deprecated "::") this is a python file with class specified,
    # so we can compute the hash for it
    if source is not None and ":" in source:
        source_file, source_class = source.replace("::", ":").split(":")

        # get the source path
        source_file = _ensure_local(source_file, root)
        source_hash = _get_hash(root / source_file)

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

    attachments = {"attachments": kwargs["weight_attachments"]} if "weight_attachments" in kwargs else {}
    weight_types = model_spec.raw_nodes.WeightsFormat
    weight_source = _ensure_local_or_url(original_weight_source, root)

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
            source=weight_source,
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
    if isinstance(dependencies, Path) or ":" not in dependencies:
        manager = "conda"
        path = dependencies
    else:
        manager, path = dependencies.split(":")

    return model_spec.raw_nodes.Dependencies(manager=manager, file=_ensure_local(path, root))


def _get_deepimagej_macro(name, kwargs, export_folder):

    # macros available in deepimagej
    macro_names = ("binarize", "scale_linear", "scale_range", "zero_mean_unit_variance")
    if name == "scale_linear":
        macro = "scale_linear.ijm"
        replace = {"gain": kwargs["gain"], "offset": kwargs["offset"]}

    elif name == "scale_range":
        macro = "per_sample_scale_range.ijm"
        replace = {"min_precentile": kwargs["min_percentile"], "max_percentile": kwargs["max_percentile"]}

    elif name == "zero_mean_unit_variance":
        mode = kwargs["mode"]
        if mode == "fixed":
            macro = "fixed_zero_mean_unit_variance.ijm"
            replace = {"paramMean": kwargs["mean"], "paramStd": kwargs["std"]}
        else:
            macro = "zero_mean_unit_variance.ijm"
            replace = {}

    elif name == "binarize":
        macro = "binarize.ijm"
        replace = {"optimalThreshold": kwargs["threshold"]}

    else:
        raise ValueError(f"Macro {name} is not available, must be one of {macro_names}.")

    url = f"https://raw.githubusercontent.com/deepimagej/imagej-macros/master/bioimage.io/{macro}"

    path = os.path.join(export_folder, macro)
    # use https://github.com/bioimage-io/core-bioimage-io-python/blob/main/bioimageio/core/resource_io/utils.py#L267
    # instead if the implementation is update s.t. an output path is accepted
    with requests.get(url, stream=True) as r:
        text = r.text
        if text.startswith("4"):
            raise RuntimeError(f"An error occured when downloading {url}: {r.text}")
        with open(path, "w") as f:
            f.write(r.text)

    # replace the kwargs in the macro file
    if replace:
        lines = []
        with open(path) as f:
            for line in f:
                kwarg = [kwarg for kwarg in replace if line.startswith(kwarg)]
                if kwarg:
                    assert len(kwarg) == 1
                    kwarg = kwarg[0]
                    # each kwarg should only be replaced ones
                    val = replace.pop(kwarg)
                    lines.append(f"{kwarg} = {val};\n")
                else:
                    lines.append(line)

        with open(path, "w") as f:
            for line in lines:
                f.write(line)

    return {"spec": "ij.IJ::runMacroFile", "kwargs": macro}


def _get_deepimagej_config(export_folder, sample_inputs, sample_outputs, pixel_sizes, preprocessing, postprocessing):
    assert len(sample_inputs) == len(sample_outputs) == 1, "deepimagej config only valid for single input/output"

    if any(preproc is not None for preproc in preprocessing):
        assert len(preprocessing) == 1
        preprocess_ij = [
            _get_deepimagej_macro(name, kwargs, export_folder) for name, kwargs in preprocessing[0].items()
        ]
        attachments = [preproc["kwargs"] for preproc in preprocess_ij]
    else:
        preprocess_ij = [{"spec": None}]
        attachments = None

    if any(postproc is not None for postproc in postprocessing):
        assert len(postprocessing) == 1
        postprocess_ij = [
            _get_deepimagej_macro(name, kwargs, export_folder) for name, kwargs in postprocessing[0].items()
        ]
        if attachments is None:
            attachments = [postproc["kwargs"] for postproc in postprocess_ij]
        else:
            attachments.extend([postproc["kwargs"] for postproc in postprocess_ij])
    else:
        postprocess_ij = [{"spec": None}]

    def get_size(path):
        assert tifffile is not None, "need tifffile for writing deepimagej config"
        with tifffile.TiffFile(export_folder / path) as f:
            shape = f.asarray().shape
        # add singleton z axis if we have 2d data
        if len(shape) == 3:
            shape = shape[:2] + (1,) + shape[-1:]
        assert len(shape) == 4
        return " x ".join(map(str, shape))

    # deepimagej always expexts a pixel size for the z axis
    pixel_sizes_ = [pix_size if "z" in pix_size else dict(z=1.0, **pix_size) for pix_size in pixel_sizes]

    test_info = {
        "inputs": [
            {"name": in_path, "size": get_size(in_path), "pixel_size": pix_size}
            for in_path, pix_size in zip(sample_inputs, pixel_sizes_)
        ],
        "outputs": [{"name": out_path, "type": "image", "size": get_size(out_path)} for out_path in sample_outputs],
        "memory_peak": None,
        "runtime": None,
    }

    config = {
        "prediction": {"preprocess": preprocess_ij, "postprocess": postprocess_ij},
        "test_information": test_info,
        # other stuff deepimagej needs
        "pyramidal_model": False,
        "allow_tiling": True,
        "model_keys": None,
    }
    return {"deepimagej": config}, attachments


def _write_sample_data(input_paths, output_paths, input_axes, output_axes, export_folder: Path):
    def write_im(path, im, axes):
        assert tifffile is not None, "need tifffile for writing deepimagej config"
        assert len(axes) == im.ndim
        assert im.ndim in (3, 4)

        # deepimagej expects xyzc axis order
        if im.ndim == 3:
            assert set(axes) == {"x", "y", "c"}
            axes_ij = "xyc"
        else:
            assert set(axes) == {"x", "y", "z", "c"}
            axes_ij = "xyzc"

        axis_permutation = tuple(axes_ij.index(ax) for ax in axes)
        im = im.transpose(axis_permutation)

        with tifffile.TiffWriter(path) as f:
            f.write(im)

    sample_in_paths = []
    for i, (in_path, axes) in enumerate(zip(input_paths, input_axes)):
        inp = np.load(export_folder / in_path)[0]
        sample_in_path = export_folder / f"sample_input_{i}.tif"
        write_im(sample_in_path, inp, axes)
        sample_in_paths.append(sample_in_path)

    sample_out_paths = []
    for i, (out_path, axes) in enumerate(zip(output_paths, output_axes)):
        outp = np.load(export_folder / out_path)[0]
        sample_out_path = export_folder / f"sample_output_{i}.tif"
        write_im(sample_out_path, outp, axes)
        sample_out_paths.append(sample_out_path)

    return [Path(p.name) for p in sample_in_paths], [Path(p.name) for p in sample_out_paths]


def _ensure_local(source: Union[Path, URI, str, list], root: Path) -> Union[Path, URI, list]:
    """ensure source is local relative path in root"""
    if isinstance(source, list):
        return [_ensure_local(s, root) for s in source]

    local_source = resolve_source(source, root)
    local_source = resolve_source(local_source, root, root / local_source.name)
    return local_source.relative_to(root)


def _ensure_local_or_url(source: Union[Path, URI, str, list], root: Path) -> Union[Path, URI, list]:
    """ensure source is remote URI or local relative path in root"""
    if isinstance(source, list):
        return [_ensure_local_or_url(s, root) for s in source]

    local_source = resolve_local_source(source, root)
    local_source = resolve_local_source(
        local_source, root, None if isinstance(local_source, URI) else root / local_source.name
    )
    return local_source.relative_to(root)


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
    sample_inputs: Optional[List[str]] = None,
    sample_outputs: Optional[List[str]] = None,
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
    pixel_sizes: Optional[List[Dict[str, float]]] = None,
    # general optional
    git_repo: Optional[str] = None,
    attachments: Optional[Dict[str, Union[str, List[str]]]] = None,
    packaged_by: Optional[List[str]] = None,
    run_mode: Optional[str] = None,
    parent: Optional[Tuple[str, str]] = None,
    config: Optional[Dict[str, Any]] = None,
    dependencies: Optional[Union[Path, str]] = None,
    links: Optional[List[str]] = None,
    root: Optional[Union[Path, str]] = None,
    add_deepimagej_config: bool = False,
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
        pixel_sizes: the pixel sizes for the input tensors, only for spatial axes.
            This information is currently only used by deepimagej, but will be added to the spec soon.
        git_repo: reference git repository for this model.
        attachments: list of additional files to package with the model.
        packaged_by: list of authors that have packaged this model.
        run_mode: custom run mode for this model.
        parent: id of the parent model from which this model is derived and sha256 of the corresponding weight file.
        config: custom configuration for this model.
        dependencies: relative path to file with dependencies for this model.
        root: optional root path for relative paths. This can be helpful when building a spec from another model spec.
        add_deepimagej_config: add the deepimagej config to the model.
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
    test_inputs = _ensure_local_or_url(test_inputs, root)
    test_outputs = _ensure_local_or_url(test_outputs, root)

    n_inputs = len(test_inputs)
    input_name = n_inputs * [None] if input_name is None else input_name
    input_step = n_inputs * [None] if input_step is None else input_step
    input_min_shape = n_inputs * [None] if input_min_shape is None else input_min_shape
    input_axes = n_inputs * [None] if input_axes is None else input_axes
    input_data_range = n_inputs * [None] if input_data_range is None else input_data_range
    preprocessing = n_inputs * [None] if preprocessing is None else preprocessing

    inputs = [
        _get_input_tensor(root / test_in, name, step, min_shape, data_range, axes, preproc)
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
        _get_output_tensor(root / test_out, name, reference, scale, offset, axes, data_range, postproc, hal)
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

    # validate the pixel sizes (currently only used by deepimagej)
    spatial_axes = [[ax for ax in inp.axes if ax in "xyz"] for inp in inputs]
    if pixel_sizes is None:
        pixel_sizes = [{ax: 1.0 for ax in axes} for axes in spatial_axes]
    else:
        assert len(pixel_sizes) == n_inputs
        for pix_size, axes in zip(pixel_sizes, spatial_axes):
            assert isinstance(pix_size, dict)
            assert set(pix_size.keys()) == set(axes)

    #
    # generate general fields
    #
    format_version = get_args(model_spec.raw_nodes.FormatVersion)[-1]
    timestamp = datetime.datetime.now()

    authors = _build_authors(authors)
    cite = _build_cite(cite)
    documentation = _ensure_local(documentation, root)
    covers = _ensure_local(covers, root)

    # parse the weights
    weights, language, framework, source, source_hash, tmp_source = _get_weights(
        weight_uri, weight_type, source, root, **weight_kwargs
    )

    # validate the sample inputs and outputs (if given)
    if sample_inputs is not None:
        assert sample_outputs is not None
        assert len(sample_inputs) == n_inputs
        assert len(sample_outputs) == n_outputs

    # add the deepimagej config if specified
    if add_deepimagej_config:
        if sample_inputs is None:
            input_axes_ij = [inp.axes[1:] for inp in inputs]
            output_axes_ij = [out.axes[1:] for out in outputs]
            sample_inputs, sample_outputs = _write_sample_data(
                test_inputs, test_outputs, input_axes_ij, output_axes_ij, root
            )
        # deepimagej expect tifs as sample data
        assert all(os.path.splitext(path)[1] in (".tif", ".tiff") for path in sample_inputs)
        assert all(os.path.splitext(path)[1] in (".tif", ".tiff") for path in sample_outputs)

        ij_config, ij_attachments = _get_deepimagej_config(
            root, sample_inputs, sample_outputs, pixel_sizes, preprocessing, postprocessing
        )

        if config is None:
            config = ij_config
        else:
            config.update(ij_config)

        if ij_attachments is not None:
            if attachments is None:
                attachments = {"files": ij_attachments}
            elif "files" not in attachments:
                attachments["files"] = ij_attachments
            else:
                attachments["files"].extend(ij_attachments)

        if links is None:
            links = ["deepimagej/deepimagej"]
        else:
            links.append("deepimagej/deepimagej")

    # make sure links are unique
    if links is not None:
        links = list(set(links))

    # make sure sample inputs / outputs are relative paths
    if sample_inputs is not None:
        sample_inputs = _ensure_local_or_url(sample_inputs, root)

    if sample_outputs is not None:
        sample_outputs = _ensure_local_or_url(sample_outputs, root)

    # optional kwargs, don't pass them if none
    optional_kwargs = {
        "attachments": attachments,
        "config": config,
        "git_repo": git_repo,
        "packaged_by": packaged_by,
        "run_mode": run_mode,
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
            authors=authors,
            cite=cite,
            covers=covers,
            description=description,
            documentation=documentation,
            format_version=format_version,
            inputs=inputs,
            license=license,
            name=name,
            outputs=outputs,
            root_path=root,
            tags=tags,
            test_inputs=test_inputs,
            test_outputs=test_outputs,
            timestamp=timestamp,
            weights=weights,
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
