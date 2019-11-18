from typing import Any, Dict, List, Union

from pybio.spec.common import CommonSpec


class TransformationSpec(CommonSpec):
    def __init__(
        self, inputs: Union[str, List[Dict[str, Any]]], outputs: Union[str, List[Dict[str, Any]]], **super_kwargs
    ):
        self.type_validation("inputs", inputs, [str, list])
        self.type_validation("outputs", outputs, [str, list])
        super().__init__(**super_kwargs)
        self.inputs = inputs
        self.outputs = outputs
