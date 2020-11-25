from pybio.runners.base import ModelInferenceRunner


class TestInferenceRunner:
    def test_rf_dummy(self, rf_config):
        runner = ModelInferenceRunner(rf_config)
        runner.run_on_test_inputs()
