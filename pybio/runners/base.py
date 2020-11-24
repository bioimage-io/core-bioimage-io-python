from __future__ import annotations
from typing import List, Optional

from pybio.spec.nodes import Model


class ModelInferenceRunner:
    # def __new__(cls, spec: Model, weights_id: Optional[str] = None) -> SklearnModelInferenceRunner:
    #     if weights_id is None:  # use newest weights
    #         weights_id = sorted(spec.weights, key=lambda w: w.timestamp)[-1].id
    #
    #     assert spec.language == "python"
    #     if spec.framework == "scikit-learn":
    #         return SklearnModelInferenceRunner(spec=spec, weights_id=weights_id)

    def __init__(self, model: Model, weights_id: str = None):
        raise NotImplementedError

    def __call__(self, inputs: List) -> List:
        raise NotImplementedError


class SklearnModelInferenceRunner(ModelInferenceRunner):
    pass
