from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Type, TypeVar, List
from urllib.parse import urlparse

import yaml

__version__ = "0.1.0"


SpecType = TypeVar("SpecType", bound="Spec")


@dataclass
class Spec:
    """Python Spec

    A spec[ification] should exactly match the content of the corresponding <name>.<type>.yaml specification.
    There should not be framework specific specification objects, but there may be framework specific config objects,
    which are constructed by parsing their spec counterpart.
    """

    file_path: Path
    name: Optional[str]

    @classmethod
    def interpret(cls, file_path: Path, kwargs: Dict[str, Any]) -> Tuple[Path, Dict[str, Any]]:
        """Interpretates loaded/given key/value pairs to translate primitive data types to more pythonic ones.
        Add interpret calls to a subclass to interpret specific keys (but don't forget to call super().interpret)"""

        if "name" not in kwargs:
            kwargs["name"] = None

        return file_path, kwargs

    @classmethod
    def from_yaml(cls: Type[SpecType], spec: str, spec_kwargs: Optional[Dict[str, Any]] = None) -> SpecType:
        """Factory classmethod to generate a spec instance from a spec yaml file."""
        if spec_kwargs is None:
            spec_kwargs = {}

        uri = urlparse(spec)
        assert uri.scheme == "file"
        assert uri.netloc == ""
        file_path = Path(uri.path)

        with file_path.open() as f:
            kwargs = yaml.safe_load(f)

        _, kwargs = cls.interpret(file_path, kwargs)
        return cls(file_path=file_path, **kwargs, **spec_kwargs)

    @classmethod
    def from_dict(cls: Type[SpecType], file_path: Path, kwargs: Optional[Dict[str, Any]] = None) -> SpecType:
        """Factory classmethod to generate a spec instance from primitive data types given in a dict."""
        if kwargs is None:
            kwargs = {}

        _, kwargs = cls.interpret(file_path, kwargs)
        return cls(file_path=file_path, **kwargs)


@dataclass
class Source(Spec):
    path: Path
    name: str

    @classmethod
    def interpret(cls, file_path, kwargs):
        source = kwargs.pop("source")
        assert source.count(":") == 1
        rel_path, name = source.split(":")
        kwargs["path"] = file_path / rel_path
        kwargs["name"] = name
        return super().interpret(file_path, kwargs)


@dataclass
class CiteEntry(Spec):
    text: str
    doi: Optional[str] = None
    url: Optional[str] = None

    def __post_init__(self):
        assert self.doi or self.url


@dataclass
class TensorSpec(Spec):
    axes: str  # todo make axes type
    data_type: str  # todo make dtype type?? float32
    data_range: Tuple[float, float]

    description: str = ""


@dataclass
class InputTensorSpec(TensorSpec):
    shape: Optional[Dict[str, Any]] = None


@dataclass
class OutputTensorSpec(TensorSpec):
    shape: Optional[Dict[str, Any]] = None


@dataclass
class StandardSpec(Spec):
    """Python Spec

    A spec[ification] should exactly match the content of the corresponding <name>.<type>.yaml specification.
    There should not be framework specific specification objects, but there may be framework specific config objects,
    which are constructed by parsing their spec counterpart.
    """

    format_version: str
    language: str
    framework: Optional[str]

    description: str
    format_version: str
    language: str
    framework: str
    cite: List[CiteEntry]
    authors: List[str]
    documentation: str
    tags: List[str]
    source: Source
    kwargs: Dict[str, Any]
    dependencies: str
    # optional
    thumbnail: Optional[Path]

    def __post_init__(self):
        assert self.language == "python", self.language
        assert self.format_version == __version__

    @classmethod
    def interpret(cls, file_path: Path, kwargs: Dict[str, Any]) -> Tuple[Path, Dict[str, Any]]:
        kwargs["source"] = Source.from_dict(file_path, kwargs)
        kwargs["cite"] = [CiteEntry.from_dict(file_path, c) for c in kwargs["cite"]]

        # cast primitive types to more pythonic types
        for name, cast, optional in [("thumbnail", Path, True)]:
            value = kwargs.get(name, None)
            if value is None:
                assert optional, name
            else:
                value = cast(value)

            kwargs[name] = value

        return super().interpret(file_path, kwargs)

    @classmethod
    def from_yaml(cls: Type[SpecType], spec: str, spec_kwargs: Optional[Dict[str, Any]] = None) -> SpecType:
        """Factory classmethod to generate a spec instance from primitive data types.
         If all spec fields are primitive, no  kwarg transformations are required and loading is equivalent to initializing.
        """
        if spec_kwargs is None:
            spec_kwargs = {}

        uri = urlparse(spec)
        assert uri.scheme == "file"
        assert uri.netloc == ""
        file_path = Path(uri.path)

        with file_path.open() as f:
            kwargs = yaml.safe_load(f)

        _, kwargs = cls.interpret(file_path, kwargs)
        return cls(file_path=file_path, **kwargs, **spec_kwargs)
