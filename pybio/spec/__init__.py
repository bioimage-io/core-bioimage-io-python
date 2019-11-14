import logging
import sys
import warnings
from importlib import import_module
from pathlib import Path
from typing import Optional, Dict, Any, Type, TypeVar, List
from urllib.parse import urlparse

import yaml


logger = logging.getLogger(__name__)
__version__ = "0.1.0"
SpecType = TypeVar("SpecType", bound="Spec")


class Spec:
    """Python Spec

    A spec[ification] object represents the content of the corresponding <name>.<type>.yaml specification file.
    There should not be framework specific python objects upon initialization, but there may be framework specific
    objects returned by an instance property method.
    """

    def __init__(self, name: str, file_path: Optional[Path] = None):
        assert isinstance(name, str), type(name)
        assert file_path is None or isinstance(file_path, Path)

        self.name: str = name
        self.file_path = file_path

    @classmethod
    def from_yaml(cls: Type[SpecType], spec: str, **spec_kwargs: Dict[str, Any]) -> SpecType:
        """Factory classmethod to generate a spec instance from a spec yaml file."""
        if spec_kwargs is None:
            spec_kwargs = {}

        file_path = Path(spec)
        try:
           spec_path_exists = file_path.exists()
        except OSError:
           spec_path_exists = False

        if not spec_path_exists:
            warnings.warn(f"Could not find spec file at {spec}. Trying to resolve as uri...")
            uri = urlparse(spec)
            assert uri.scheme == "file", (spec, uri.scheme)
            assert uri.netloc == ""
            file_path = Path(uri.path)
            assert file_path.exists(), file_path.absolute()

        with file_path.open() as f:
            kwargs = yaml.safe_load(f)

        return cls(file_path=file_path, **kwargs, **spec_kwargs)


class SpecWithSource(Spec):
    source_file_path: Optional[Path]
    source_module_name: Optional[str]

    def __init__(
        self,
        name: str,
        file_path: Path,
        source: str,
        hash: Optional[Dict[str, str]] = None,
        kwargs: Optional[Dict[str, Any]] = Any,
    ):
        # todo: handle zenodo sources
        assert hash is None or isinstance(hash, dict)
        super().__init__(name=name, file_path=file_path)
        if source.startswith("."):
            # source is a relative file path
            assert source.count(":") == 1, source
            source, self.object_name = source.split(":")
            self.source_file_path = file_path / source
            self.source_module_name = None
        else:
            # source is an importable python object
            assert "." in source, source
            last_dot_idx = source.rfind(".")
            self.source_module_name = source[:last_dot_idx]
            self.object_name = source[last_dot_idx + 1 :]
            self.source_file_path = None

        self._object = None
        self.hash = hash or {}
        self.kwargs = kwargs or {}

    @property
    def object_(self):
        # todo: check source hash
        if self._object is None:
            if self.source_module_name is None:
                # import from file  # todo: improve
                assert self.source_file_path is not None
                cur_sys_path = list(sys.path)
                logger.debug("temporarily adding %s to sys.path", self.source_file_path.parent.as_posix())
                sys.path.append(self.source_file_path.parent.as_posix())
                logger.debug("attempting to import %s", self.source_file_path.stem)
                dep = import_module(self.source_file_path.stem)
                sys.path = cur_sys_path
            else:
                # import from dependency  # todo: improve
                assert self.source_file_path is None
                dep = import_module(self.source_module_name)

            self._object = getattr(dep, self.object_name)

        return self._object


class CiteEntry(Spec):
    def __init__(self, text: str, doi: Optional[str] = None, url: Optional[str] = None):
        if doi is None and url is None:
            raise ValueError("Require `doi` or `url`")

        super().__init__(name=self.__class__.__name__)
        self.text = text
        # todo: check doi/url
        self.doi = doi
        self.url = url


class TensorSpec(Spec):
    def __init__(self, name: str, axes: str, data_type: str, data_range: List[float], description: str = ""):
        assert isinstance(data_range, list), type(data_range)
        assert isinstance(description, str), type(description)

        if len(data_range) != 2:
            raise ValueError(
                f"Expected `data_range` to be sequence of lenght 2, but got: {data_range} of length {len(data_range)}"
            )

        super().__init__(name=name)
        self.axes = axes  # todo: check axes
        self.data_type = data_type  # todo: check data type
        self.data_range = tuple(data_range)  # todo: check data range
        self.description = description


class InputTensorSpec(TensorSpec):
    def __init__(self, shape: Dict[str, Any], **tensor_spec_kwargs):
        super().__init__(**tensor_spec_kwargs)
        self.shape = shape  # todo: check input tensor shape


class OutputTensorSpec(TensorSpec):
    def __init__(self, shape: Dict[str, Any], **tensor_spec_kwargs):
        super().__init__(**tensor_spec_kwargs)
        self.shape = shape  # todo: check output tensor shape


class CommonSpec(SpecWithSource):
    def __init__(
        self,
        name: str,
        file_path: Path,
        format_version: str,
        language: str,
        framework: str,
        description: str,
        cite: List[Dict[str, Optional[str]]],
        authors: List[str],
        documentation: str,
        tags: List[str],
        source: str,
        hash: Optional[Dict[str, Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        thumbnail: Optional[Path] = None,
        test_input: Optional[str] = None,
        test_output: Optional[str] = None,
    ):

        assert language == "python", language
        if format_version != __version__:  # todo: activate downward compatibility
            raise ValueError(
                f"Format version mismatch: Running version {__version__}, got format version {format_version}"
            )

        assert framework in ["pytorch"], framework
        assert isinstance(description, str), type(description)
        assert isinstance(cite, list), type(cite)
        assert all(isinstance(el, dict) for el in cite), [type(el) for el in cite]
        assert isinstance(authors, list), type(authors)
        assert all(isinstance(el, str) for el in authors), [type(el) for el in authors]
        assert isinstance(documentation, str)
        assert isinstance(tags, list), type(tags)
        assert all(isinstance(el, str) for el in tags), [type(el) for el in tags]
        assert dependencies is None, "todo: add dependency management"
        if thumbnail is not None:
            logger.warning("thumbnail not yet implemented")

        if test_input is not None:
            logger.warning("test_input not yet implemented")

        if test_output is not None:
            logger.warning(f"test_output not yet implemented")

        super().__init__(name=name, file_path=file_path, source=source, hash=hash, kwargs=kwargs)
        self.framework = framework
        self.description = description
        self.cite = [CiteEntry(*el) for el in cite]


if __name__ == "__main__":
    s = Spec(Path("/"))
    print(s)
