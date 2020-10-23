from pybio.runners.base import ModelInferenceRunner, SklearnModelInferenceRunner


def test_SklearnModelInferenceRunner(rf_config):
    runner = ModelInferenceRunner(rf_config)
    assert isinstance(runner, SklearnModelInferenceRunner)
