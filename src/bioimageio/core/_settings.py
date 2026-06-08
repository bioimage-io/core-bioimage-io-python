import os
import platform
from typing import Literal, Optional

from loguru import logger
from pydantic import Field, field_validator
from typing_extensions import Annotated

from bioimageio.spec._internal._settings import Settings as SpecSettings


class Settings(SpecSettings):
    """environment variables for bioimageio.spec and bioimageio.core"""

    keras_backend: Annotated[
        Literal["torch", "tensorflow", "jax"], Field(alias="KERAS_BACKEND")
    ] = "torch"

    pytorch_enable_mps_fallback: Annotated[
        Optional[bool], Field(alias="PYTORCH_ENABLE_MPS_FALLBACK")
    ] = None

    @field_validator("pytorch_enable_mps_fallback", mode="after")
    @classmethod
    def _set_default_mps_fallback(cls, value: Optional[bool]):
        # pytorch versions up to the 2.6 don't support all operations (esp 3d) on MPS
        # this env variable allows falling back to CPU for those networks instead of failing
        # see for current status https://github.com/pytorch/pytorch/issues/141287
        if (
            value is None
            and platform.system().lower() == "darwin"
            and platform.machine().lower() == "arm64"
        ):
            logger.info("Set environment variable 'PYTORCH_ENABLE_MPS_FALLBACK=1'.")
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            return True

        return value

    collection_index_url: str = "https://bioimage-io.github.io/collection/index.json"
    """URL to the bioimageio collection index"""

    collection_config_url: str = (
        "https://bioimage-io.github.io/collection/bioimageio_collection_config.json"
    )
    """URL to the bioimageio collection config"""

    default_gradio_server: str = "bioimage-io/bioimage-io-gradio-server"
    """Default URL or Hugging Face space name to connect to with the remote gradio model adapter or remote gradio prediction pipeline."""

    gradio_server_model_cache_max_size: int = 10
    """Max number of models to cache in the gradio server for prediction pipelines using the gradio backend."""

    gradio_server_model_cache_max_memory: str = "40GB"
    """Max memory to use for model caching in the gradio server for prediction pipelines using the gradio backend."""


settings = Settings()
"""parsed environment variables for bioimageio.spec and bioimageio.core"""
