import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from bioimageio.core.remote_backends import create_remote_model_adapter
from bioimageio.spec.model import v0_5


@pytest.mark.parametrize("explicit_type", [False, True])
def test_remote_backend_selection(monkeypatch: pytest.MonkeyPatch, explicit_type: bool):
    module = ModuleType("bioimageio.core.remote_backends.gradio.client")
    factory = Mock()
    module.__dict__["GradioModelAdapter"] = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    model = v0_5.ModelDescr.model_construct()
    assert (
        create_remote_model_adapter(
            model, server="local", server_type="gradio" if explicit_type else None
        )
        is factory.return_value
    )
    factory.assert_called_once_with(model_description=model, server="local")


def test_remote_backend_missing_extra(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(
        sys.modules, "bioimageio.core.remote_backends.gradio.client", None
    )
    with pytest.raises(
        ImportError, match=r"pip install bioimageio.core\[gradio-client\]"
    ) as caught:
        _ = create_remote_model_adapter(v0_5.ModelDescr.model_construct())
    assert isinstance(caught.value.__cause__, ImportError)
