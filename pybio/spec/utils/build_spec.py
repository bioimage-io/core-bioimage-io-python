import os
import datetime
import hashlib
import numpy as np

import pybio.spec.raw_nodes as raw_nodes

#
# utility functions to build the spec from python
#


# TODO refactor common stuff here
def _build_spec():
    pass


# TODO this should handle both local file paths and urls and download the url somwhere temp
def _ensure_uri(uri):
    assert os.path.exists(uri)
    return uri


def _get_hash(path):
    with open(path, 'rb') as f:
        data = f.read()
        return hashlib.sha256(data).hexdigest()


# TODO optional
# TODO type-annotations
def build_spec_torch(
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
    # model specific optional
    sample_inputs=None,
    sample_outputs=None,
    # TODO optional arguments to over-ride the input / output tensor descriptions
    # general optional
    cite=None,
    git_repo=None,
    attachments=None,
    packaged_by=None,
    parent=None,
    run_mode=None,
    config=None
):
    """
    """
    #
    # generate the model specific fields
    #

    # generate sha256
    source_path = _ensure_uri(source.split("::")[0])
    source_hash = _get_hash(source_path)

    # check the test inputs and auto-generate input/  output description from test inputs / outputs
    if isinstance(test_inputs, list):
        assert isinstance(test_outputs, list)
    elif isinstance(test_inputs, str):
        assert isinstance(test_outputs, str)
        test_inputs, test_outputs = [test_inputs], [test_outputs]
    else:
        raise ValueError("Invalid input tensor URI")

    for test_in, test_out in zip(test_inputs, test_outputs):
        test_in, test_out = _ensure_uri(test_in), _ensure_uri(test_out)
        test_in, test_out = np.load(test_in), np.load(test_out)
        assert test_in.ndim in (4, 5)
        assert test_out.ndim in (4, 5)

    # TODO enable over-riding with optional arguments
    # TODO description, preprocessing from optional arguments
    inputs = raw_nodes.InputTensor(
        name='input',
        data_type=str(test_in.dtype),
        axes=['b', 'c', 'z', 'y', 'x'] if test_in.ndim == 5 else ['b', 'c', 'y', 'x'],
        shape=test_in.shape,
        preprocessing=None
    )

    # TODO enable over-riding with optional arguments
    # TODO description, halo, postprocessing from optional arguments
    outputs = raw_nodes.OutputTensor(
        name='output',
        data_type=str(test_out.dtype),
        axes=['b', 'c', 'z', 'y', 'x'] if test_out.ndim == 5 else ['b', 'c', 'y', 'x'],
        shape=test_out.shape,
        postprocessing=None,
        halo=None
    )

    weight_hash = _get_hash(weight_path)
    weights = raw_nodes.WeightsEntry(
        authors=None,
        attachments=None,
        parent=None,
        opset_version=None,
        tensorflow_version=None,
        source=weight_uri,
        sha256=weight_hash,
    )

    #
    # generate general fields
    #
    format_version = '0.3.1'  # TODO get this from somewhere central
    timestamp = datetime.datetime.now().isoformat()
    language = 'python'
    framework = 'pytorch'

    model = raw_nodes.Model(
        source=source,
        sha256=source_hash,
        kwargs=model_kwargs,
        format_version=format_version,
        name=name,
        description=description,
        authors=authors,
        cite=[] if cite is None else cite,
        git_repo=git_repo,
        tags=tags,
        license=license,
        documentation=documentation,
        covers=covers,
        attachments=attachments,
        language=language,
        framework=framework,
        dependencies=dependencies,
        timestamp=timestamp,
        run_mode=run_mode,
        config=config,
        weights={'pytorch_state_dict': weights},
        inputs=[inputs],
        outputs=[outputs],
        test_inputs=test_inputs,
        test_outputs=test_outputs,
        sample_inputs=None,
        sample_outputs=None
    )
    return model


# just a quick local test
if __name__ == '__main__':
    import torch
    import imageio
    from torch_em.model import UNet2d

    model_kwargs = {"in_channels": 1, "out_channels": 2, "initial_features": 8}
    weight_path = "/home/pape/Work/my_projects/torch-em/experiments/dsb/checkpoints/dsb-boundary-model/weights.pt"

    model = UNet2d(**model_kwargs)
    state = torch.load(weight_path)
    model.load_state_dict(state)

    test_inp = './test_input.npy'
    test_outp = './test_output.npy'

    im_path = '/home/pape/Work/data/data_science_bowl/dsb2018/test/images/0bda515e370294ed94efd36bd53782288acacb040c171df2ed97fd691fc9d8fe.tif'
    im = np.asarray(imageio.imread(im_path))
    inp = im[None, None].astype('float32')
    np.save(test_inp, inp)

    with torch.no_grad():
        outp = model(torch.from_numpy(inp))
    outp = outp.numpy()
    np.save(test_outp, outp)

    model_spec = build_spec_torch(
        source='/home/pape/Work/my_projects/torch-em/torch_em/model/unet.py::UNet2d',
        model_kwargs=model_kwargs,
        weight_uri=weight_path,
        test_inputs=test_inp,
        test_outputs=test_outp,
        name="Unet2dDSB",
        description="A unet trained on DSB",
        authors=["Constantin Pape"],
        tags=["nuclei"],
        license="MIT",
        documentation="not_documented.md",
        covers=["no covers"],
        dependencies="conda:./no_deps.yaml"
    )
    print(model_spec)
