from typing import Optional

from pybio.spec.spec_types import ModelSpec


def classic_fit(model_spec: ModelSpec, sampler_start: Optional[int] = None, sampler_stop: Optional[int] = None):
    """classic fit a la 'model.fit(X, y)'"""

    model = model_spec.get_instance()
    reader = model_spec.spec.training.setup.reader.get_instance()
    sampler = model_spec.spec.training.setup.sampler.get_instance(data_source=reader)
    X, y = sampler[slice(sampler_start, sampler_stop)]
    model.fit([X], [y])
    return model
    # todo: save model weights
