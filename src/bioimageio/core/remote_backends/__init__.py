from typing import TYPE_CHECKING, Literal, Optional

from typing_extensions import assert_never

from bioimageio.spec.model import AnyModelDescr

if TYPE_CHECKING:
    from .gradio.client import GradioModelAdapter


def create_remote_model_adapter(
    model_description: AnyModelDescr,
    server: Optional[str] = None,
    backend: Literal["gradio"] = "gradio",
) -> "GradioModelAdapter":
    """Create a remote model adapter"""

    try:
        if backend == "gradio":
            from .gradio.client import GradioModelAdapter as RemoteModelAdapterImpl
        else:
            assert_never(backend)
    except ImportError as e:
        raise ImportError(
            f"Failed to import {backend.capitalize()}ModelAdapter. Make sure to install the '{backend}-client' extra,"
            + f" e.g. with `pip install bioimageio.core[{backend}-client]`."
        ) from e

    return RemoteModelAdapterImpl(model_description=model_description, server=server)
