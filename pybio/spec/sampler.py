from typing import Any, Dict, List, Union

from pybio.spec import CommonSpec


class SamplerSpec(CommonSpec):
    def __init__(self, outputs: Union[str, List[Dict[str, Any]]], **super_kwargs):
        self.type_validation("outputs", outputs, [str, list])
        super().__init__(**super_kwargs)
        self.outputs = outputs
