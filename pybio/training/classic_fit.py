from pybio_spec.spec_types import ModelSpec


def classic_fit(model_spec: ModelSpec, start: int = 0, batch_size: int = 1):
    """classic fit a la 'model.fit(X, y)'"""

    model = model_spec.get_instance()
    reader = model_spec.spec.training.setup.reader.get_instance()
    sampler = model_spec.spec.training.setup.sampler.get_instance(reader=reader)
    X, y = sampler[start, batch_size]
    model.fit([X], [y])
    return model
    # todo: save model weights
