import os
import datetime
import hashlib
from typing import Any, Dict, List, Optional, Union

import numpy as np
import bioimageio.spec as spec

#
# utility functions to build the spec from python
#


# TODO this should handle both local file paths and urls and download the url somwhere temp
def _ensure_uri(uri, root=None):
    if not os.path.exists(uri) and root is not None:
        uri = os.path.join(root, uri)
    assert os.path.exists(uri), uri
    return uri


def _get_hash(path):
    with open(path, 'rb') as f:
        data = f.read()
        return hashlib.sha256(data).hexdigest()


def _get_weights(weight_uri, weight_type, source, root):
    assert weight_type is not None, "Weight type detection not supported"

    # TODO try to auto-dectect the weight type
    # TODO add the other weight types and get this from somwhere central
    weight_types = (
        'pytorch_state_dict',
        'pickle'
    )
    weight_path = _ensure_uri(weight_uri, root)
    weight_hash = _get_hash(weight_path)

    if weight_type == 'pytorch_state_dict':
        weights = spec.raw_nodes.WeightsEntry(
            source=weight_uri,
            sha256=weight_hash
        )
        weights = {'pytorch_state_dict': weights}
        language = 'python'
        framework = 'pytorch'

        # pytorch-state-dict -> we need a source file
        # generate sha256 for the source file
        assert source is not None
        source_path = _ensure_uri(source.split("::")[0], root)
        source_hash = _get_hash(source_path)
    elif weight_type == 'pickle':
        weights = spec.raw_nodes.WeightsEntry(
            source=weight_uri,
            sha256=weight_hash
        )
        weights = {'pickle': weights}
        language = 'python'
        framework = 'scikit-learn'

        source_hash = None
    else:
        raise ValueError(f"Invalid weight type {weight_type}, expect one of {weight_types}")

    return weights, language, framework, source_hash


def _get_input_tensor(test_in, name, step, min_shape, preprocessing):
    shape = test_in.shape
    if step is None:
        assert min_shape is None
        shape_description = shape
    else:
        shape_description = {
            'min': shape if min_shape is None else min_shape,
            'step': step
        }

    kwargs = {}
    if preprocessing is not None:
        kwargs['preprocessing'] = preprocessing

    inputs = spec.raw_nodes.InputTensor(
        name='input' if name is None else name,
        data_type=str(test_in.dtype),
        axes='bczyx' if test_in.ndim == 5 else 'bcyx',
        shape=shape_description,
        **kwargs
    )
    return inputs


def _get_output_tensor(test_out, name,
                       reference_input, scale, offset,
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

    kwargs = {}
    if postprocessing is not None:
        kwargs['postprocessing'] = postprocessing
    if halo is not None:
        kwargs['halo'] = halo

    outputs = spec.raw_nodes.OutputTensor(
        name='output' if name is None else name,
        data_type=str(test_out.dtype),
        axes='bczyx' if test_out.ndim == 5 else 'bcyx',
        shape=shape_description,
        **kwargs
    )
    return outputs


# TODO can we pattern match to decide if we have a url or doi?
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
    source: str,
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
    weight_type: Optional[str] = None,
    sample_inputs: Optional[str] = None,
    sample_outputs: Optional[str] = None,
    # tensor specific
    input_name: Optional[str] = None,
    input_step: Optional[List[int]] = None,
    input_min_shape: Optional[List[int]] = None,
    output_name: Optional[str] = None,
    output_reference: Optional[str] = None,
    output_scale: Optional[List[int]] = None,
    output_offset: Optional[List[int]] = None,
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
):
    """
    """
    #
    # generate the model specific fields
    #

    # check the test inputs and auto-generate input/output description from test inputs/outputs
    for test_in, test_out in zip(test_inputs, test_outputs):
        test_in, test_out = _ensure_uri(test_in, root), _ensure_uri(test_out, root)
        test_in, test_out = np.load(test_in), np.load(test_out)
    inputs = _get_input_tensor(test_in, input_name, input_step, input_min_shape, preprocessing)
    outputs = _get_output_tensor(test_out, output_name,
                                 output_reference, output_scale, output_offset,
                                 postprocessing, halo)

    (weights, language,
     framework, source_hash) = _get_weights(weight_uri, weight_type,
                                            source, root)

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
                       'sample_outputs': sample_outputs}
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
        language=language,
        framework=framework,
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
