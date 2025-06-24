from typing import Literal

from pydantic import Field
from typing_extensions import Annotated

from bioimageio.spec._internal._settings import Settings as SpecSettings


class Settings(SpecSettings):
    """environment variables for bioimageio.spec and bioimageio.core"""

    keras_backend: Annotated[
        Literal["torch", "tensorflow", "jax"], Field(alias="KERAS_BACKEND")
    ] = "torch"


settings = Settings()
"""parsed environment variables for bioimageio.spec and bioimageio.core"""
