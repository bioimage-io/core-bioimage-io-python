from typing import get_type_hints

from . import nodes
from .utils import load_spec, load_model_spec

__version__ = get_type_hints(nodes.Spec)["format_version"].__args__
