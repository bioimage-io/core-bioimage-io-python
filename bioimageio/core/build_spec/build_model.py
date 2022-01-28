import datetime
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn

import imageio
import numpy as np
import requests

import bioimageio.spec as spec
import bioimageio.spec.model as model_spec
from bioimageio.core import export_resource_package, load_raw_resource_description
from bioimageio.core.resource_io.nodes import URI
from bioimageio.spec.shared.raw_nodes import ImportableModule, ImportableSourceFile
from bioimageio.spec.shared.utils import resolve_local_source, resolve_source

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


def _get_pytorch_state_dict_weight_kwargs(architecture, model_kwargs, root):
    assert architecture is not None
    tmp_archtecture = None
    weight_kwargs = {"kwargs": model_kwargs} if model_kwargs else {}
    if ":" in architecture:
        # note: path itself might include : for absolute paths in windows
        *arch_file_parts, callable_name = architecture.replace("::", ":").split(":")
        arch_file = _ensure_local(":".join(arch_file_parts), root)
        arch = ImportableSourceFile(callable_name, arch_file)
        arch_hash = _get_hash(root / arch.source_file)
        weight_kwargs["architecture_sha256"] = arch_hash
    else:
        arch = spec.shared.fields.ImportableSource().deserialize(architecture)
        assert isinstance(arch, ImportableModule)

    weight_kwargs["architecture"] = arch
    return weight_kwargs, tmp_archtecture


def _get_attachments(attachments, root):
    assert isinstance(attachments, dict)
    if "files" in attachments:
        afiles = attachments["files"]
        if isinstance(afiles, str):
            afiles = [afiles]

        if isinstance(afiles, list):
            afiles = _ensure_local_or_url(afiles, root)
        else:
            raise TypeError(attachments)

        attachments["files"] = afiles
    return attachments


def _get_weights(
    original_weight_source,
    weight_type,
    root,
    architecture=None,
    model_kwargs=None,
    tensorflow_version=None,
    opset_version=None,
    pytorch_version=None,
    dependencies=None,
    attachments=None,
):
    weight_path = resolve_source(original_weight_source, root)
    if weight_type is None:
        weight_type = _infer_weight_type(weight_path)
    weight_hash = _get_hash(weight_path)

    weight_types = model_spec.raw_nodes.WeightsFormat
    weight_source = _ensure_local_or_url(original_weight_source, root)

    weight_kwargs = {"source": weight_source, "sha256": weight_hash}
    if attachments is not None:
        weight_kwargs["attachments"] = _get_attachments(attachments, root)
    if dependencies is not None:
        weight_kwargs["dependencies"] = _get_dependencies(dependencies, root)

    tmp_archtecture = None
    if weight_type == "pytorch_state_dict":
        # pytorch-state-dict -> we need an architecture definition
        pytorch_weight_kwargs, tmp_file = _get_pytorch_state_dict_weight_kwargs(architecture, model_kwargs, root)
        weight_kwargs.update(**pytorch_weight_kwargs)
        if pytorch_version is not None:
            weight_kwargs["pytorch_version"] = pytorch_version
        elif dependencies is None:
            warn(
                "You are building a pytorch model but have neither passed dependencies nor the pytorch_version."
                "It may not be possible to create an environmnet where your model can be used."
            )
        weights = model_spec.raw_nodes.PytorchStateDictWeightsEntry(**weight_kwargs)

    elif weight_type == "onnx":
        if opset_version is not None:
            weight_kwargs["opset_version"] = opset_version
        elif dependencies is None:
            warn(
                "You are building an onnx model but have neither passed dependencies nor the opset_version."
                "It may not be possible to create an environmnet where your model can be used."
            )
        weights = model_spec.raw_nodes.OnnxWeightsEntry(**weight_kwargs)

    elif weight_type == "torchscript":
        if pytorch_version is not None:
            weight_kwargs["pytorch_version"] = pytorch_version
        elif dependencies is None:
            warn(
                "You are building a pytorch model but have neither passed dependencies nor the pytorch_version."
                "It may not be possible to create an environmnet where your model can be used."
            )
        weights = model_spec.raw_nodes.TorchscriptWeightsEntry(**weight_kwargs)

    elif weight_type == "keras_hdf5":
        if tensorflow_version is not None:
            weight_kwargs["tensorflow_version"] = tensorflow_version
        elif dependencies is None:
            warn(
                "You are building a keras model but have neither passed dependencies nor the tensorflow_version."
                "It may not be possible to create an environmnet where your model can be used."
            )
        weights = model_spec.raw_nodes.KerasHdf5WeightsEntry(**weight_kwargs)

    elif weight_type == "tensorflow_saved_model_bundle":
        if tensorflow_version is not None:
            weight_kwargs["tensorflow_version"] = tensorflow_version
        elif dependencies is None:
            warn(
                "You are building a tensorflow model but have neither passed dependencies nor the tensorflow_version."
                "It may not be possible to create an environmnet where your model can be used."
            )
        weights = model_spec.raw_nodes.TensorflowSavedModelBundleWeightsEntry(**weight_kwargs)

    elif weight_type == "tensorflow_js":
        if tensorflow_version is not None:
            weight_kwargs["tensorflow_version"] = tensorflow_version
        elif dependencies is None:
            warn(
                "You are building a tensorflow model but have neither passed dependencies nor the tensorflow_version."
                "It may not be possible to create an environmnet where your model can be used."
            )
        weights = model_spec.raw_nodes.TensorflowJsWeightsEntry(**weight_kwargs)

    elif weight_type in weight_types:
        raise ValueError(f"Weight type {weight_type} is not supported yet in 'build_spec'")
    else:
        raise ValueError(f"Invalid weight type {weight_type}, expect one of {weight_types}")

    return {weight_type: weights}, tmp_archtecture


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


def _get_input_tensor(path, name, step, min_shape, data_range, axes, preprocessing):
    test_in = np.load(path)
    shape = test_in.shape
    if step is None:
        assert min_shape is None
        shape_description = shape
    else:
        shape_description = {"min": shape if min_shape is None else min_shape, "step": step}

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


# TODO The citation entry should be improved so that we can properly derive doi vs. url
def _build_cite(cite: Dict[str, str]):
    citation_list = [spec.rdf.raw_nodes.CiteEntry(text=k, url=v) for k, v in cite.items()]
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
    return {"deepimagej": config}, [Path(a) for a in attachments]


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


# create better cover images for 3d data and non-image outputs
def _generate_covers(in_path, out_path, input_axes, output_axes, root):
    def normalize(data, axis, eps=1e-7):
        data = data.astype("float32")
        data -= data.min(axis=axis, keepdims=True)
        data /= data.max(axis=axis, keepdims=True) + eps
        return data

    def to_image(data, data_axes):
        assert data.ndim in (4, 5)

        # transpose the data to "bczyx" / "bcyx" order
        axes = "bczyx" if data.ndim == 5 else "bcyx"
        assert set(data_axes) == set(axes)
        if axes != data_axes:
            ax_permutation = tuple(data_axes.index(ax) for ax in axes)
            data = data.transpose(ax_permutation)

        # select single image with channels from the data
        if data.ndim == 5:
            z0 = data.shape[2] // 2
            data = data[0, :, z0]
        else:
            data = data[0, :]

        # normalize the data and map to 8 bit
        data = normalize(data, axis=(1, 2))
        data = (data * 255).astype("uint8")
        return data

    cover_path = os.path.join(root, "cover.png")
    input_, output = np.load(in_path), np.load(out_path)

    input_ = to_image(input_, input_axes)
    # this is not image data so we only save the input image
    if output.ndim < 4:
        imageio.imwrite(cover_path, input_.transpose((1, 2, 0)))
        return [_ensure_local(cover_path, root)]
    output = to_image(output, output_axes)

    chan_in = input_.shape[0]
    # make sure the input is rgb
    if chan_in == 1:  # single channel -> repeat it 3 times
        input_ = np.repeat(input_, 3, axis=0)
    elif chan_in != 3:  # != 3 channels -> take first channe and repeat it 3 times
        input_ = np.repeat(input_[0:1], 3, axis=0)

    im_shape = input_.shape[1:]
    # we just save the input image if the shapes don't agree
    if im_shape != output.shape[1:]:
        imageio.imwrite(cover_path, input_.transpose((1, 2, 0)))
        return [_ensure_local(cover_path, root)]

    def diagonal_split(im0, im1):
        assert im0.shape[0] == im1.shape[0] == 3
        n, m = im_shape
        out = np.ones((3, n, m), dtype="uint8")
        for c in range(3):
            outc = np.tril(im0[c])
            mask = outc == 0
            outc[mask] = np.triu(im1[c])[mask]
            out[c] = outc
        return out

    def grid_im(im0, im1):
        ims_per_row = 3
        n_chan = im1.shape[0]
        n_images = n_chan + 1
        n_rows = int(np.ceil(float(n_images) / ims_per_row))

        n, m = im_shape
        x, y = ims_per_row * n, n_rows * m
        out = np.zeros((3, y, x))
        images = [im0] + [np.repeat(im1[i : i + 1], 3, axis=0) for i in range(n_chan)]

        i, j = 0, 0
        for im in images:
            x0, x1 = i * n, (i + 1) * n
            y0, y1 = j * m, (j + 1) * m
            out[:, y0:y1, x0:x1] = im

            i += 1
            if i == ims_per_row:
                i = 0
                j += 1

        return out

    chan_out = output.shape[0]
    if chan_out == 1:  # single prediction channel: create diagonal split
        im = diagonal_split(input_, np.repeat(output, 3, axis=0))
    elif chan_out == 3:  # three prediction channel: create diagonal split with rgb
        im = diagonal_split(input_, output)
    else:  # otherwise create grid image
        im = grid_im(input_, output)

    # to channel last
    imageio.imwrite(cover_path, im.transpose((1, 2, 0)))
    return [_ensure_local(cover_path, root)]


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
    if not isinstance(local_source, URI):
        local_source = resolve_local_source(local_source, root, root / local_source.name)
    return local_source.relative_to(root)


def build_model(
    # model or tensor specific and required
    weight_uri: str,
    test_inputs: List[Union[str, Path]],
    test_outputs: List[Union[str, Path]],
    input_axes: List[str],
    output_axes: List[str],
    # general required
    name: str,
    description: str,
    authors: List[Dict[str, str]],
    tags: List[Union[str, Path]],
    documentation: Union[str, Path],
    cite: Dict[str, str],
    output_path: Union[str, Path],
    # model specific optional
    architecture: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Union[int, float, str]]] = None,
    weight_type: Optional[str] = None,
    sample_inputs: Optional[List[str]] = None,
    sample_outputs: Optional[List[str]] = None,
    # tensor specific
    input_names: Optional[List[str]] = None,
    input_step: Optional[List[List[int]]] = None,
    input_min_shape: Optional[List[List[int]]] = None,
    input_data_range: Optional[List[List[Union[int, str]]]] = None,
    output_names: Optional[List[str]] = None,
    output_reference: Optional[List[str]] = None,
    output_scale: Optional[List[List[int]]] = None,
    output_offset: Optional[List[List[int]]] = None,
    output_data_range: Optional[List[List[Union[int, str]]]] = None,
    halo: Optional[List[List[int]]] = None,
    preprocessing: Optional[List[Dict[str, Dict[str, Union[int, float, str]]]]] = None,
    postprocessing: Optional[List[Dict[str, Dict[str, Union[int, float, str]]]]] = None,
    pixel_sizes: Optional[List[Dict[str, float]]] = None,
    # general optional
    maintainers: Optional[List[Dict[str, str]]] = None,
    license: Optional[str] = None,
    covers: Optional[List[str]] = None,
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
    tensorflow_version: Optional[str] = None,
    opset_version: Optional[int] = None,
    pytorch_version: Optional[str] = None,
    weight_attachments: Optional[Dict[str, Union[str, List[str]]]] = None,
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
        input_axes=["bcyx"],
        output_axes=["bcyx"],
        name="my-model",
        description="My very fancy model.",
        authors=[{"name": "John Doe", "affiliation": "My Institute"}],
        tags=["segmentation", "light sheet data"],
        license="CC-BY-4.0",
        documentation="./documentation.md",
        cite={"Architecture": "https://my_architecture.com"},
        output_path="my-model.zip"
    )
    ```

    Args:
        weight_uri: the url or relative local file path to the weight file for this model.
        test_inputs: list of test input files stored in numpy format.
        test_outputs: list of test outputs corresponding to test_inputs, stored in numpy format.
        input_axes: axis names of the input tensors.
        output_axes: axiss names of the output tensors.
        name: name of this model.
        description: short description of this model.
        authors: the authors of this model.
        tags: list of tags for this model.
        documentation: relative file path to markdown documentation for this model.
        cite: citations for this model.
        output_path: where to save the zipped model package.
        architecture: the file with the source code for the model architecture and the corresponding class.
            Only required for models with pytorch_state_dict weight format.
        model_kwargs: the keyword arguments for the model class.
            Only required for models with pytorch_state_dict weight format.
        weight_type: the type of the weights.
        sample_inputs: list of sample inputs to demonstrate the model performance.
        sample_outputs: list of sample outputs corresponding to sample_inputs.
        input_names: names of the input tensors.
        input_step: minimal valid increase of the input tensor shape.
        input_min_shape: minimal input tensor shape.
        input_data_range: valid data range for the input tensor.
        output_names: names of the output tensors.
        output_reference: name of the input reference tensor used to cimpute the output tensor shape.
        output_scale: multiplicative factor to compute the output tensor shape.
        output_offset: additive term to compute the output tensor shape.
        output_data_range: valid data range for the output tensor.
        halo: halo to be cropped from the output tensor.
        preprocessing: list of preprocessing operations for the input.
        postprocessing: list of postprocessing operations for the output.
        pixel_sizes: the pixel sizes for the input tensors, only for spatial axes.
            This information is currently only used by deepimagej, but will be added to the spec soon.
        license: the license for this model. By default CC-BY-4.0 will be set as license.
        covers: list of file paths for cover images.
            By default a cover will be generated from the input and output data.
        git_repo: reference git repository for this model.
        attachments: list of additional files to package with the model.
        packaged_by: list of authors that have packaged this model.
        run_mode: custom run mode for this model.
        parent: id of the parent model from which this model is derived and sha256 of the corresponding weight file.
        config: custom configuration for this model.
        dependencies: relative path to file with dependencies for this model.
        root: optional root path for relative paths. This can be helpful when building a spec from another model spec.
        add_deepimagej_config: add the deepimagej config to the model.
        tensorflow_version: the tensorflow version for this model. Only for tensorflow or keras weights.
        opset_version: the opset version for this model. Only for onnx weights.
        pytorch_version: the pytorch version for this model. Only for pytoch_state_dict or torchscript weights.
        weight_attachments: extra weight specific attachments.
    """
    assert architecture is None or isinstance(architecture, str)
    if root is None:
        root = "."
    root = Path(root)

    if attachments is not None:
        attachments = _get_attachments(attachments, root)

    #
    # generate the model specific fields
    #

    assert len(test_inputs)
    assert len(test_outputs)
    test_inputs = _ensure_local_or_url(test_inputs, root)
    test_outputs = _ensure_local_or_url(test_outputs, root)

    n_inputs = len(test_inputs)
    if input_names is None:
        input_names = [f"input{i}" for i in range(n_inputs)]
    else:
        assert len(input_names) == len(test_inputs)

    input_step = n_inputs * [None] if input_step is None else input_step
    input_min_shape = n_inputs * [None] if input_min_shape is None else input_min_shape
    input_data_range = n_inputs * [None] if input_data_range is None else input_data_range
    preprocessing = n_inputs * [None] if preprocessing is None else preprocessing

    inputs = [
        _get_input_tensor(root / test_in, name, step, min_shape, data_range, axes, preproc)
        for test_in, name, step, min_shape, axes, data_range, preproc in zip(
            test_inputs, input_names, input_step, input_min_shape, input_axes, input_data_range, preprocessing
        )
    ]

    n_outputs = len(test_outputs)
    if output_names is None:
        output_names = [f"output{i}" for i in range(n_outputs)]
    else:
        assert len(output_names) == len(test_outputs)

    output_reference = n_outputs * [None] if output_reference is None else output_reference
    output_scale = n_outputs * [None] if output_scale is None else output_scale
    output_offset = n_outputs * [None] if output_offset is None else output_offset
    output_data_range = n_outputs * [None] if output_data_range is None else output_data_range
    postprocessing = n_outputs * [None] if postprocessing is None else postprocessing
    halo = n_outputs * [None] if halo is None else halo

    outputs = [
        _get_output_tensor(root / test_out, name, reference, scale, offset, axes, data_range, postproc, hal)
        for test_out, name, reference, scale, offset, axes, data_range, postproc, hal in zip(
            test_outputs,
            output_names,
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

    authors = [model_spec.raw_nodes.Author(**a) for a in authors]
    cite = _build_cite(cite)
    documentation = _ensure_local(documentation, root)
    if covers is None:
        covers = _generate_covers(root / test_inputs[0], root / test_outputs[0], input_axes[0], output_axes[0], root)
    else:
        covers = _ensure_local(covers, root)
    if license is None:
        license = "CC-BY-4.0"

    # parse the weights
    weights, tmp_archtecture = _get_weights(
        weight_uri,
        weight_type,
        root,
        architecture,
        model_kwargs,
        tensorflow_version=tensorflow_version,
        opset_version=opset_version,
        pytorch_version=pytorch_version,
        dependencies=dependencies,
        attachments=weight_attachments,
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
                attachments["files"] = list(set(attachments["files"]) | set(ij_attachments))

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
        "config": config,
        "git_repo": git_repo,
        "packaged_by": packaged_by,
        "run_mode": run_mode,
        "sample_inputs": sample_inputs,
        "sample_outputs": sample_outputs,
        "links": links,
    }
    kwargs = {k: v for k, v in optional_kwargs.items() if v is not None}

    if attachments is not None:
        kwargs["attachments"] = spec.rdf.raw_nodes.Attachments(**attachments)

    if maintainers is not None:
        kwargs["maintainers"] = [model_spec.raw_nodes.Maintainer(**m) for m in maintainers]
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
        if tmp_archtecture is not None:
            os.remove(tmp_archtecture)

    model = load_raw_resource_description(model_package)
    return model
