"""DEPRECATED"""

from typing import List

from .backends._model_adapter import DEFAULT_WEIGHT_FORMAT_PRIORITY_ORDER
from .backends._model_adapter import ModelAdapter as ModelAdapter
from .backends._model_adapter import create_model_adapter as create_model_adapter


def get_weight_formats() -> List[str]:
    """
    Return list of supported weight types
    """
    return list(DEFAULT_WEIGHT_FORMAT_PRIORITY_ORDER)
