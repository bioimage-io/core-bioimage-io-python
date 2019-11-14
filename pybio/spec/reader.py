from pathlib import Path

from pybio.spec import Spec


class ReaderSpec(Spec):
    def __init__(self, file_path: Path):
        super().__init__(name=self.__class__.__name__, file_path=file_path)