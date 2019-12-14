from pybio.spec.spec_types import ModelSpec
from pybio.spec.utils import get_instance


def classic_fit(model_spec: ModelSpec, start: int = 0, batch_size: int = 1):
    """classic fit a la 'model.fit(X, y)'"""

    model = get_instance(model_spec)
    reader = get_instance(model_spec.spec.training.setup.reader)
    sampler = get_instance(model_spec.spec.training.setup.sampler, reader=reader)
    X, y = sampler[start, batch_size]
    model.fit([X], [y])
    return model
    # todo: save model weights
