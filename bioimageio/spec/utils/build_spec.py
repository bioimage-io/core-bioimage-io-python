import os
import datetime
import hashlib

import numpy as np
import bioimageio.spec.raw_nodes as raw_nodes
import bioimageio.spec.fields as fields

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
        weights = raw_nodes.WeightsEntry(
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
        weights = raw_nodes.WeightsEntry(
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


# TODO enable different shape specification and description
def _get_input_tensor(test_in, preprocessing):
    kwargs = {} if preprocessing is None else {'preprocessing': preprocessing}
    inputs = raw_nodes.InputTensor(
        name='input',
        data_type=str(test_in.dtype),
        axes='bczyx' if test_in.ndim == 5 else 'bcyx',
        shape=test_in.shape,
        **kwargs
    )
    return inputs


# TODO enable different shape specification and description
def _get_output_tensor(test_out, postprocessing, halo):
    kwargs = {}
    if postprocessing is not None:
        kwargs['postprocessing'] = postprocessing
    if halo is not None:
        kwargs['halo'] = halo
    outputs = raw_nodes.OutputTensor(
        name='output',
        data_type=str(test_out.dtype),
        axes='bczyx' if test_out.ndim == 5 else 'bcyx',
        shape=test_out.shape,
        **kwargs
    )
    return outputs


# TODO type-annotations
def build_spec(
    # model specific required
    source,
    model_kwargs,
    weight_uri,
    test_inputs,
    test_outputs,
    # general required
    name,
    description,
    authors,
    tags,
    license,
    documentation,
    covers,
    dependencies,
    root=None,
    # model specific optional
    weight_type=None,
    sample_inputs=None,
    sample_outputs=None,
    # TODO optional arguments to over-ride the input / output tensor descriptions
    # tensor specific
    preprocessing=None,
    postprocessing=None,
    halo=None,
    # general optional
    cite=None,
    git_repo=None,
    attachments=None,
    packaged_by=None,
    run_mode=None,
    parent=None,
    config=None
):
    """
    """
    #
    # generate the model specific fields
    #

    # check the test inputs and auto-generate input/  output description from test inputs / outputs
    if isinstance(test_inputs, list):
        assert isinstance(test_outputs, list)
    elif isinstance(test_inputs, str):
        assert isinstance(test_outputs, str)
        test_inputs, test_outputs = [test_inputs], [test_outputs]
    else:
        raise ValueError("Invalid input tensor URI")

    for test_in, test_out in zip(test_inputs, test_outputs):
        test_in, test_out = _ensure_uri(test_in, root), _ensure_uri(test_out, root)
        test_in, test_out = np.load(test_in), np.load(test_out)

    inputs = _get_input_tensor(test_in, preprocessing)
    outputs = _get_output_tensor(test_out, postprocessing, halo)

    (weights, language,
     framework, source_hash) = _get_weights(weight_uri, weight_type,
                                            source, root)

    #
    # generate general fields
    #
    format_version = '0.3.1'  # TODO get this from somewhere central
    timestamp = datetime.datetime.now()

    if source is not None:
        source = fields.ImportableSource().deserialize(source)

    # optional kwargs, don't pass them if none
    optional_kwargs = {'git_repo': git_repo, 'attachments': attachments,
                       'packaged_by': packaged_by, 'parent': parent,
                       'run_mode': run_mode, 'config': config,
                       'sample_inputs': sample_inputs,
                       'sample_outputs': sample_outputs}
    kwargs = {
        k: v for k, v in optional_kwargs.items() if v is not None
    }

    model = raw_nodes.Model(
        source=source,
        sha256=source_hash,
        kwargs=model_kwargs,
        format_version=format_version,
        name=name,
        description=description,
        authors=authors,
        cite=[] if cite is None else cite,
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
    return model
