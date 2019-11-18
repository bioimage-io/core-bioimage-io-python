import logging
from pathlib import Path
from typing import Optional, Dict, Any, Type, List, Union

logger = logging.getLogger(__name__)
__version__ = "0.1.0"


class Spec:
    """Python Spec

    A spec[ification] object represents the content of the corresponding <name>.<type>.yaml specification file.
    There should not be framework specific python objects upon initialization, but there may be framework specific
    objects returned by an instance property method.
    """

    def __init__(self, name: str, _rel_path: Path = Path(".")):
        assert isinstance(name, str), type(name)
        assert isinstance(_rel_path, Path), type(_rel_path)

        self.name: str = name
        self._rel_path = _rel_path

    def type_validation(self, arg_name: str, arg: Any, type_or_types: Union[Type, List[Type]]):
        if isinstance(type_or_types, list):
            if not any(isinstance(arg, t) for t in type_or_types):
                raise ValueError(
                    f"Spec {self.name} expected '{arg_name}' to be of one of the following types: {type_or_types}, but got type {type(arg)}"
                )
        else:
            if not isinstance(arg, type_or_types):
                raise ValueError(
                    f"Spec {self.name} expected '{arg_name}' to be of type: {type_or_types}, but got type {type(arg)}"
                )


class CiteEntry(Spec):
    def __init__(self, text: str, doi: Optional[str] = None, url: Optional[str] = None):
        if doi is None and url is None:
            raise ValueError("Require `doi` or `url`")

        super().__init__(name=self.__class__.__name__)
        self.text = text
        # todo: check doi/url
        self.doi = doi
        self.url = url


class TensorSpec(Spec):
    def __init__(self, name: str, axes: str, data_type: str, data_range: List[float], description: str = ""):
        self.type_validation("data_range", data_range, list)
        self.type_validation("description", description, str)

        if len(data_range) != 2:
            raise ValueError(
                f"Expected `data_range` to be sequence of lenght 2, but got: {data_range} of length {len(data_range)}"
            )

        super().__init__(name=name)
        self.axes = axes  # todo: check axes
        self.data_type = data_type  # todo: check data type
        self.data_range = tuple(data_range)  # todo: check data range
        self.description = description


class InputTensorSpec(TensorSpec):
    def __init__(self, shape: Dict[str, Any], **tensor_spec_kwargs):
        super().__init__(**tensor_spec_kwargs)
        self.shape = shape  # todo: check input tensor shape


class OutputTensorSpec(TensorSpec):
    def __init__(self, shape: Dict[str, Any], **tensor_spec_kwargs):
        super().__init__(**tensor_spec_kwargs)
        self.shape = shape  # todo: check output tensor shape
