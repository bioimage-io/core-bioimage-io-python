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


settings = Settings()
"""parsed environment variables for bioimageio.spec and bioimageio.core"""
