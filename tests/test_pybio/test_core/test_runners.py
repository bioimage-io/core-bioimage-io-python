import numpy

from pybio.runners.base import ModelInferenceRunner


class TestInferenceRunner:
    def test_rf_dummy(self, rf_resolved_spec):
        runner = ModelInferenceRunner(rf_resolved_spec)
        expected = numpy.load(str(rf_resolved_spec.test_outputs[0]))
        actual = runner.run_on_test_inputs()
        assert numpy.allclose(expected, actual)
