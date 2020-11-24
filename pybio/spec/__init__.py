from typing import get_type_hints

from pybio.spec.utils import load_model_spec, load_spec, maybe_convert
from . import nodes

__version__ = get_type_hints(nodes.Spec)["format_version"].__args__
