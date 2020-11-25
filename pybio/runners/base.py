from __future__ import annotations

import pathlib
from typing import List, Sequence

import numpy

from pybio.core.array import PyBioTensor
from pybio.spec.exceptions import PyBioRunnerException
from pybio.spec.nodes import InputTensor, Language, Model
from pybio.spec.utils import get_instance


def read_input(path: pathlib.Path, input_spec: InputTensor):
    return numpy.load(str(path))  # todo: check input spec


def read_inputs(paths: Sequence[pathlib.Path], inputs_spec: Sequence[InputTensor]):
    return [read_input(p, inspec) for p, inspec in zip(paths, inputs_spec)]


class ModelInferenceRunner:
    weights_format_priority_order: Sequence[Language] = ("pickle",)

    def __init__(self, model_spec: Model):
        compatible_weights_formats = [wf for wf in model_spec.weights if wf in self.weights_format_priority_order]
        if not compatible_weights_formats:
            raise PyBioRunnerException("No compatible weights formats found.")

        self.weights_format = compatible_weights_formats[0]
        self.model_spec = model_spec

    def run_on_test_inputs(self):
        test_inputs = self.model_spec.weights[self.weights_format].test_inputs
        test_inputs = read_inputs(test_inputs, self.model_spec.inputs)
        return self(test_inputs)

    def __call__(self, inputs: List[PyBioTensor]) -> List:
        model = get_instance(self.model_spec)

        raise PyBioRunnerException from NotImplementedError("inference")
