import sys

from loguru import logger
from pydantic_settings import CliApp

logger.enable("bioimageio")

logger.remove()
_ = logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{elapsed:}</green> | "
    + "<level>{level: <8}</level> | "
    + "<cyan>{module}</cyan> - <level>{message}</level>",
)

from .cli import Bioimageio

_ = CliApp.run(Bioimageio)
