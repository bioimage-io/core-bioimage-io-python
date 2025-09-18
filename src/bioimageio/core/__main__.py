import sys

from loguru import logger

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


def main():
    cli = Bioimageio()  # pyright: ignore[reportCallIssue]
    cli.run()


if __name__ == "__main__":
    main()
