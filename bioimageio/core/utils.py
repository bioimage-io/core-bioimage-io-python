from marshmallow import missing

from bioimageio.spec.shared.io import Node
from bioimageio.spec.shared.nodes import ImportedSource

try:
    from typing import get_args
except ImportError:
    from typing_extensions import get_args  # type: ignore


def get_nn_instance(model_node: Node, **kwargs):
    assert isinstance(model_node.source, ImportedSource)

    joined_kwargs = {} if model_node.kwargs is missing else dict(model_node.kwargs)
    joined_kwargs.update(kwargs)
    return model_node.source(**joined_kwargs)
