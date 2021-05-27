from copy import deepcopy

from bioimageio.spec import load_spec, raw_nodes, schema
from bioimageio.spec.utils import yaml
from bioimageio.spec.utils.transformers import load_model_spec


def test_spec_roundtrip(rf_config_path):
    data = yaml.load(rf_config_path)

    raw_model, root = load_spec(rf_config_path)
    assert isinstance(raw_model, raw_nodes.Model)

    serialized = schema.Model().dump(raw_model)
    assert isinstance(serialized, dict)

    # yaml.dump(serialized, Path() / "serialized.yml")

    # manually remove all inserted defaults to test round trip at raw data level
    serialized_wo_defaults = deepcopy(serialized)
    serialized_wo_defaults["cite"][0].pop("doi")
    serialized_wo_defaults.pop("config")
    serialized_wo_defaults["inputs"][0].pop("preprocessing")
    serialized_wo_defaults["outputs"][0].pop("halo")
    serialized_wo_defaults["outputs"][0].pop("postprocessing")
    serialized_wo_defaults.pop("packaged_by")
    serialized_wo_defaults.pop("parent")
    serialized_wo_defaults.pop("run_mode")
    serialized_wo_defaults.pop("sample_inputs")
    serialized_wo_defaults.pop("sample_outputs")
    serialized_wo_defaults.pop("sha256")
    serialized_wo_defaults["weights"]["pickle"].pop("attachments")
    serialized_wo_defaults["weights"]["pickle"].pop("authors")
    serialized_wo_defaults["weights"]["pickle"].pop("opset_version")
    serialized_wo_defaults["weights"]["pickle"].pop("parent")
    serialized_wo_defaults["weights"]["pickle"].pop("tensorflow_version")

    assert serialized_wo_defaults == data

    assert not schema.Model().validate(serialized)
    assert not schema.Model().validate(serialized_wo_defaults)

    raw_model_from_serialized = load_model_spec(serialized, root_path=root)
    assert raw_model_from_serialized == raw_model

    raw_model_from_serialized_wo_defaults = load_model_spec(serialized, root_path=root)
    assert raw_model_from_serialized_wo_defaults == raw_model
