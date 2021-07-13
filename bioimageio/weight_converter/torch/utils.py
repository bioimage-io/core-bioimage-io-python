from marshmallow import missing


# NOTE: copied from tiktorch; this should go into python-bioimageio and then we use it from there
def get_nn_instance(node, **kwargs):
    joined_kwargs = {} if node.kwargs is missing else dict(node.kwargs)  # type: ignore
    joined_kwargs.update(kwargs)
    return node.source(**joined_kwargs)
