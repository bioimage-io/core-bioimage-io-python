import os
from bioimageio.spec import schema
from bioimageio.spec.utils import yaml


def test_build_spec_pickle(rf_config_path):
    from bioimageio.spec.utils.build_spec import build_spec
    source = yaml.load(rf_config_path)

    root = rf_config_path.parents[0]

    weight_path = os.path.join(root, source['weights']['pickle']['source'])
    assert os.path.exists(weight_path), weight_path
    test_inputs = [os.path.join(root, pp) for pp in source['test_inputs']]
    test_outputs = [os.path.join(root, pp) for pp in source['test_outputs']]

    cite = {'source': 'https://citation.com'}
    attachments = {'files': './some_local_file',
                   'urls': ['https://attachment1.com', 'https://attachment2.com']}

    raw_model = build_spec(
        source=source['source'],
        model_kwargs=source['kwargs'],
        weight_uri=weight_path,
        test_inputs=test_inputs,
        test_outputs=test_outputs,
        name=source['name'],
        description=source['description'],
        authors=source['authors'],
        tags=source['tags'],
        license=source['license'],
        documentation=source['documentation'],
        covers=source['covers'],
        dependencies=source['dependencies'],
        weight_type='pickle',
        cite=cite,
        attachments=attachments,
        input_name='raw',
        input_min_shape=[1, 1],
        input_step=[0, 0],
        output_reference='raw',
        output_scale=[1, 1],
        output_offset=[0, 0]
    )
    serialized = schema.Model().dump(raw_model)

    # TODO test that dicts agree properly
    assert type(serialized) == type(source)
