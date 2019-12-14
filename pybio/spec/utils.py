import importlib
from typing import Any, Dict

from .spec_types import ModelSpec, Importable



def _resolve_import(importable: Importable):

    if isinstance(importable, Importable.Module):
        module = importlib.import_module(importable.module_name)
        return getattr(module, importable.callable_name)
    elif isinstance(importable, Importable.Path):
        raise NotImplementedError()

    raise NotImplementedError(f"Can't resolve import for type {type(importable)}")


def get_instance(spec, **kwargs) -> Any:
    joined_kwargs = dict(spec.spec.optional_kwargs)
    joined_kwargs.update(spec.kwargs)
    joined_kwargs.update(kwargs)
    cls = _resolve_import(spec.spec.source)
    return cls(**joined_kwargs)


def train(model: ModelSpec, kwargs: Dict[str, Any] = None) -> Any:
    if kwargs is None:
        kwargs = {}

    complete_kwargs = dict(model.spec.training.optional_kwargs)
    complete_kwargs.update(kwargs)

    mspec = "model_spec"
    if mspec not in complete_kwargs and mspec in model.spec.training.required_kwargs:
        complete_kwargs[mspec] = model

    train_cls = _resolve_import(model.spec.training.source)
    return train_cls(**complete_kwargs)
