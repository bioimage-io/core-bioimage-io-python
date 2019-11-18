import logging

from pathlib import Path
from pybio.spec import __version__
from pybio.spec.spec import CiteEntry
from pybio.spec.source import SpecWithSource
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class CommonSpec(SpecWithSource):
    def __init__(
        self,
        format_version: str,
        language: str,
        description: str,
        cite: List[Dict[str, Optional[str]]],
        authors: List[str],
        documentation: str,
        tags: List[str],
        framework: Optional[str] = None,
        dependencies: Optional[str] = None,
        thumbnail: Optional[Path] = None,
        test_input: Optional[str] = None,
        test_output: Optional[str] = None,
        **super_kwargs,
    ):

        assert language == "python", language
        if format_version != __version__:  # todo: activate downward compatibility
            raise ValueError(
                f"Format version mismatch: Running version {__version__}, got format version {format_version}"
            )

        assert framework is None or framework in ["pytorch"], framework
        assert isinstance(description, str), type(description)
        assert isinstance(cite, list), type(cite)
        assert all(isinstance(el, dict) for el in cite), [type(el) for el in cite]
        assert isinstance(authors, list), type(authors)
        assert all(isinstance(el, str) for el in authors), [type(el) for el in authors]
        assert isinstance(documentation, str)
        assert isinstance(tags, list), type(tags)
        assert all(isinstance(el, str) for el in tags), [type(el) for el in tags]
        assert dependencies is None or isinstance(dependencies, str), type(dependencies)  # todo: handle dependencies
        if thumbnail is not None:
            logger.warning("thumbnail not yet implemented")

        if test_input is not None:
            logger.warning("test_input not yet implemented")

        if test_output is not None:
            logger.warning(f"test_output not yet implemented")

        super().__init__(**super_kwargs)
        self.framework = framework
        self.description = description
        self.cite = [CiteEntry(*el) for el in cite]
