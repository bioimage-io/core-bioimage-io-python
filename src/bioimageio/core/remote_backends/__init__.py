from typing import TYPE_CHECKING, Literal, Optional

from typing_extensions import assert_never

from bioimageio.spec.model import AnyModelDescr

if TYPE_CHECKING:
    from .gradio.client import GradioModelAdapter


def create_remote_model_adapter(
    model_description: AnyModelDescr,
    server: Optional[str] = None,
    server_type: Optional[Literal["gradio"]] = None,
) -> "GradioModelAdapter":
    """Create a remote model adapter

    Args:
        model_description: The model to run inference with.
        server: The URL or Hugging Face space name of a running bioimageio server instance
        server_type: The type of the remote server to connect to. Currently only "gradio" is supported.
    """

    if server_type is None:
        server_type = "gradio"

    try:
        if server_type == "gradio":
            from .gradio.client import GradioModelAdapter as RemoteModelAdapterImpl
        else:
            assert_never(server_type)
    except ImportError as e:
        raise ImportError(
            f"Failed to import {server_type.capitalize()}ModelAdapter. Make sure to install the '{server_type}-client' extra,"
            + f" e.g. with `pip install bioimageio.core[{server_type}-client]`."
        ) from e

    return RemoteModelAdapterImpl(model_description=model_description, server=server)
