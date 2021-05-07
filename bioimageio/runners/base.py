from __future__ import annotations

import collections
import pathlib
import pickle
from typing import List, Sequence

try:
    from typing import OrderedDict
except ImportError:
    from typing import MutableMapping as OrderedDict


import numpy

import bioimageio
from bioimageio.core.protocols import Tensor
from bioimageio.spec.exceptions import PyBioRunnerException
from bioimageio.spec.nodes import InputTensor, Language, Model
from bioimageio.spec.utils import get_instance


def read_tensor(path: pathlib.Path, input_spec: InputTensor) -> Tensor:
    ret = numpy.load(str(path))  # todo: check input spec
    assert isinstance(ret, Tensor)
    assert ret.shape
    return ret


class ModelInferenceRunner:
    weights_format_priority_order: Sequence[Language] = ("pickle",)

    def __init__(self, spec: Model):
        self.spec = spec
        compatible_weights_formats = [wf for wf in spec.weights if wf in self.weights_format_priority_order]
        if not compatible_weights_formats:
            raise PyBioRunnerException("No compatible weights formats found.")

        self.weights_format = compatible_weights_formats[0]
        with spec.weights[self.weights_format].source.open("rb") as f:
            self.model_instance = pickle.load(f)

    def run_on_test_inputs(self):
        return self(
            collections.OrderedDict(
                [(ipt.name, read_tensor(t_ipt, ipt)) for t_ipt, ipt in zip(self.spec.test_inputs, self.spec.inputs)]
            )
        )

    def __call__(self, tensors: OrderedDict[str, Tensor]) -> OrderedDict[str, Tensor]:
        return self.model_instance(*tensors.values())
