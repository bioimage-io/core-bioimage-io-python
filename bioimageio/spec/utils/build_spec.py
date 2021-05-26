import os
import datetime
import hashlib
from typing import Any, Dict, List, Optional, Union

import numpy as np
import bioimageio.spec as spec

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
        uri = spec.fields.URI().deserialize(uri)
        uri = spec.utils.transformers._download_uri_node_to_local_path(uri).as_posix()
    return uri


def _get_hash(path):
    with open(path, 'rb') as f:
        data = f.read()
        return hashlib.sha256(data).hexdigest()


def _infer_weight_type(path):
    ext = os.path.splitext(path)[-1]
    if ext in ('.pt', '.torch'):
        return 'pytorch_state_dict'
    elif ext in ('.pickle', '.pkl'):
        return 'pickle'
    elif ext == '.onnx':
        return 'onnx'
    else:
        raise ValueError(f"Could not infer weight type from extension {ext} for weight file {path}")


# TODO extend supported weight types
def _get_weights(weight_uri, weight_type, source, root, **kwargs):
    weight_path = _get_local_path(weight_uri, root)
    if weight_type is None:
        weight_type = _infer_weight_type(weight_path)
    weight_hash = _get_hash(weight_path)

    # if we have a "::" this is a python file with class specified,
    # so we can compute the hash for it
    if source is not None and "::" in source:
        source_path = _get_local_path(source.split("::")[0], root)
        source_hash = _get_hash(source_path)
    else:
        source_hash = None

    weight_types = spec.raw_nodes.WeightsFormat
    if weight_type == 'pytorch_state_dict':
        # pytorch-state-dict -> we need a source
        assert source is not None
        weights = spec.raw_nodes.WeightsEntry(
            source=weight_uri,
            sha256=weight_hash
        )
        weights = {'pytorch_state_dict': weights}
        language = 'python'
        framework = 'pytorch'

    elif weight_type == 'pickle':
        weights = spec.raw_nodes.WeightsEntry(
            source=weight_uri,
            sha256=weight_hash
        )
        weights = {'pickle': weights}
        language = 'python'
        framework = 'scikit-learn'

    elif weight_type == 'onnx':
        weights = spec.raw_nodes.WeightsEntry(
            source=weight_uri,
            sha256=weight_hash,
            opset_version=kwargs.get('opset_version', 12)
        )
        weights = {'onnx': weights}
        language = None
        framework = None

    elif weight_type == 'pytorch_script':
        weights = spec.raw_nodes.WeightsEntry(
            source=weight_uri,
            sha256=weight_hash
        )
        weights = {'pytorch_script': weights}
        if source is None:
            language = None
            framework = None
        else:
            language = 'python'
            framework = 'pytorch'

    elif weight_type in weight_types:
        raise ValueError(f"Weight type {weight_type} is not supported yet in 'build_spec'")
    else:
        raise ValueError(f"Invalid weight type {weight_type}, expect one of {weight_types}")

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
        default_axes = {2: 'bc', 4: 'bcyx', 5: 'bczyx'}
        axes = default_axes[ndim]
    return axes


def _get_input_tensor(test_in, name, step, min_shape, data_range, axes, preprocessing):
    shape = test_in.shape
    if step is None:
        assert min_shape is None
        shape_description = shape
    else:
        shape_description = {
            'min': shape if min_shape is None else min_shape,
            'step': step
        }

    axes = _get_axes(axes, test_in.ndim)
    data_range = _get_data_range(data_range, test_in.dtype)

    kwargs = {}
    if preprocessing is not None:
        kwargs['preprocessing'] = preprocessing

    inputs = spec.raw_nodes.InputTensor(
        name='input' if name is None else name,
        data_type=str(test_in.dtype),
        axes=axes,
        shape=shape_description,
        data_range=data_range,
        **kwargs
    )
    return inputs


def _get_output_tensor(test_out, name,
                       reference_input, scale, offset,
                       axes, data_range,
                       postprocessing, halo):
    shape = test_out.shape
    if reference_input is None:
        assert scale is None
        assert offset is None
        shape_description = shape
    else:
        assert scale is not None
        assert offset is not None
        shape_description = {
            'reference_input': reference_input,
            'scale': scale,
            'offset': offset
        }

    axes = _get_axes(axes, test_out.ndim)
    data_range = _get_data_range(data_range, test_out.dtype)

    kwargs = {}
    if postprocessing is not None:
        kwargs['postprocessing'] = postprocessing
    if halo is not None:
        kwargs['halo'] = halo

    outputs = spec.raw_nodes.OutputTensor(
        name='output' if name is None else name,
        data_type=str(test_out.dtype),
        axes=axes,
        data_range=data_range,
        shape=shape_description,
        **kwargs
    )
    return outputs


# TODO The citation entry should be improved so that we can properly derive doi vs. url
def _build_cite(cite):
    citation_list = [
        spec.raw_nodes.CiteEntry(text=k, url=v) for k, v in cite.items()
    ]
    return citation_list


# NOTE does not support multiple input / output tensors yet
# to implement this we should wait for 0.4.0, see also
# https://github.com/bioimage-io/spec-bioimage-io/issues/70#issuecomment-825737433
def build_spec(
    # model specific required
    model_kwargs: Dict[str, Union[int, float, str]],
    weight_uri: str,
    test_inputs: List[str],
    test_outputs: List[str],
    # general required
    name: str,
    description: str,
    authors: List[str],
    tags: List[str],
    license: str,
    documentation: str,
    covers: List[str],
    dependencies: str,
    cite: Dict[str, str],
    root: Optional[str] = None,
    # model specific optional
    source: Optional[str] = None,
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
    **weight_kwargs
):
    """
    """
    #
    # generate the model specific fields
    #

    # check the test inputs and auto-generate input/output description from test inputs/outputs
    for test_in, test_out in zip(test_inputs, test_outputs):
        test_in, test_out = _get_local_path(test_in, root), _get_local_path(test_out, root)
        test_in, test_out = np.load(test_in), np.load(test_out)
    inputs = _get_input_tensor(test_in, input_name, input_step, input_min_shape,
                               input_axes, input_data_range, preprocessing)
    outputs = _get_output_tensor(test_out, output_name,
                                 output_reference, output_scale, output_offset,
                                 output_axes, output_data_range,
                                 postprocessing, halo)

    (weights, language,
     framework, source_hash) = _get_weights(weight_uri, weight_type,
                                            source, root, **weight_kwargs)

    #
    # generate general fields
    #
    format_version = spec.__version__
    timestamp = datetime.datetime.now()

    if source is not None:
        source = spec.fields.ImportableSource().deserialize(source)

    # optional kwargs, don't pass them if none
    optional_kwargs = {'git_repo': git_repo, 'attachments': attachments,
                       'packaged_by': packaged_by, 'parent': parent,
                       'run_mode': run_mode, 'config': config,
                       'sample_inputs': sample_inputs,
                       'sample_outputs': sample_outputs,
                       'framework': framework, 'language': language}
    kwargs = {
        k: v for k, v in optional_kwargs.items() if v is not None
    }

    # build the citation object
    cite = _build_cite(cite)

    model = spec.raw_nodes.Model(
        source=source,
        sha256=source_hash,
        kwargs=model_kwargs,
        format_version=format_version,
        name=name,
        description=description,
        authors=authors,
        cite=cite,
        tags=tags,
        license=license,
        documentation=documentation,
        covers=covers,
        dependencies=dependencies,
        timestamp=timestamp,
        weights=weights,
        inputs=[inputs],
        outputs=[outputs],
        test_inputs=test_inputs,
        test_outputs=test_outputs,
        **kwargs
    )

    # serialize and deserialize the raw_nodes.Model to
    # check that all fields are well formed
    serialized = spec.schema.Model().dump(model)
    model = spec.schema.Model().load(serialized)

    return model
