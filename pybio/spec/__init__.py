import logging
import re
import sys
from importlib import import_module
from pathlib import Path
from typing import Optional, Dict, Any, Type, TypeVar, List, Union
from urllib.parse import urlparse

import numpy
import requests
import yaml

logger = logging.getLogger(__name__)
__version__ = "0.1.0"

# todo: improve cached file handling
cache_path = Path(__file__).parent.parent.parent / "cache"


class Spec:
    """Python Spec

    A spec[ification] object represents the content of the corresponding <name>.<type>.yaml specification file.
    There should not be framework specific python objects upon initialization, but there may be framework specific
    objects returned by an instance property method.
    """

    def __init__(self, name: str, _rel_path: Path = Path(".")):
        assert isinstance(name, str), type(name)
        assert isinstance(_rel_path, Path), type(_rel_path)

        self.name: str = name
        self._rel_path = _rel_path

    def type_validation(self, arg_name: str, arg: Any, type_or_types: Union[Type, List[Type]]):
        if isinstance(type_or_types, list):
            if not any(isinstance(arg, t) for t in type_or_types):
                raise ValueError(
                    f"Spec {self.name} expected '{arg_name}' to be of one of the following types: {type_or_types}, but got type {type(arg)}"
                )
        else:
            if not isinstance(arg, type_or_types):
                raise ValueError(
                    f"Spec {self.name} expected '{arg_name}' to be of type: {type_or_types}, but got type {type(arg)}"
                )


SpecWithSourceType = TypeVar("SpecWithSourceType", bound="SpecWithSource")


class Source:
    local_file_path: Optional[Path] = None
    module_name: Optional[str] = None

    def __init__(self, source: str, hash_: Dict[str, str], rel_path: Path = Path(".")):
        assert isinstance(source, str), type(source)
        assert isinstance(hash_, dict), type(hash_)
        assert isinstance(rel_path, Path), type(rel_path)
        self.original_source = source
        self.hash_ = hash_
        uri = urlparse(source)
        self.uri = uri
        self.object_name = uri.fragment
        assert uri.params == "", uri.params
        assert uri.query == "", uri.query

        self._requires_download = False
        if uri.scheme == "" or uri.scheme == "file":
            assert uri.netloc == "", uri.netloc
            self.local_file_path = rel_path / uri.path if uri.path.startswith(".") else Path(uri.path)
        elif uri.scheme in ["doi", "https", "http"]:
            self._requires_download = True
            file_name = uri.path.strip("/")
            self.local_file_path = cache_path / file_name
        else:
            if self.object_name:
                # <module name>.<sub module name>#<object name>
                self.module_name = uri.scheme
            else:
                # <module name>.<sub module name>.<object name>
                last_dot_idx = uri.scheme.rfind(".")
                self.module_name = uri.scheme[:last_dot_idx]
                self.object_name = uri.scheme[last_dot_idx + 1 :]

                if not re.match("[a-zA-Z_][a-zA-Z0-9_]*", self.object_name):
                    raise ValueError(f"invalid object name {self.object_name} extracted from source {source}")

    def download(self):
        """download and cache source from doi or url"""
        if not self._requires_download:
            return

        assert cache_path in self.local_file_path.parents, "only download files to cache folder!"

        self._requires_download = False
        uri = self.uri
        if uri.scheme == "doi":
            assert uri.netloc == "", uri.netloc
            uri.scheme = "https"
            uri.netloc = "dx.doi.org"
        # elif uri.scheme == "github":
        #     assert uri.netloc == "", uri.netloc
        #     uri.scheme = "https"
        #     uri.netloc = "github.com"
        elif uri.scheme == "https" or uri.scheme == "http":
            pass
        else:
            raise NotImplementedError(f"Unknown uri scheme {uri.scheme}")

        if uri.netloc == "dx.doi.org":
            raise NotImplementedError("todo: resolve doi")

        if self.local_file_path.exists():
            # todo: check hash
            download = False
        else:
            download = True

        if download:
            # todo: improve file download
            url = f'http://{uri.netloc}/{uri.path.strip("/")}'
            r = requests.get(url)
            self.local_file_path.parent.mkdir(parents=True, exist_ok=True)
            with self.local_file_path.open("wb") as f:
                f.write(r.content)

    def get(self) -> Any:
        self.download()
        if self.local_file_path is None:
            # import from dependency  # todo: improve
            assert self.module_name is not None, "no local path, nor module name!?!"
            dep = import_module(self.module_name)
            return getattr(dep, self.object_name)
        elif self.object_name:
            # import <object_name> from local (python) file  # todo: improve
            cur_sys_path = list(sys.path)
            logger.debug("temporarily adding %s to sys.path", self.local_file_path.parent.as_posix())
            sys.path.append(self.local_file_path.parent.as_posix())
            logger.debug("attempting to import %s", self.local_file_path.stem)
            dep = import_module(self.local_file_path.stem)
            sys.path = cur_sys_path
            return getattr(dep, self.object_name)
        elif self.local_file_path.suffix in [".yml", ".yaml"]:
            with self.local_file_path.open() as f:
                return yaml.safe_load(f)
        elif self.local_file_path.suffix in [".npy", ".npz"]:
            return numpy.load(self.local_file_path)
        elif self.local_file_path.suffix in [".torch", ".pt", ".pth"]:
            import torch

            return torch.load(self.local_file_path)
        else:
            raise NotImplementedError(f"Unknown source type {self.original_source}")


class SpecWithSource(Spec):
    source_file_path: Optional[Path]
    source_module_name: Optional[str]

    def __init__(
        self,
        name: str,
        _rel_path: Path,
        source: Source,
        kwargs: Dict[str, Any],
        required_kwargs: Optional[List[str]] = None,
        default_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """note: a SpecWithSource should not be initialized directly but instead loaded with its factory classmethod 'load'"""
        assert isinstance(source, Source), type(source)
        self.type_validation("kwargs", kwargs, dict)
        required_kwargs = required_kwargs or {}
        default_kwargs = default_kwargs or {}

        for req in required_kwargs:
            if req not in kwargs:
                raise ValueError(f"Missing kwarg '{req}' for spec {name}")

        for kw in kwargs:
            if kw not in required_kwargs and kw not in default_kwargs:
                raise ValueError(
                    f"Unexpected kwarg '{kw}' for spec {name}. Required kwargs: {required_kwargs}. Optional (default) kwargs: {default_kwargs}"
                )

        super().__init__(name=name, _rel_path=_rel_path)

        self.source = source
        self.kwargs = kwargs
        self._object = None

    @classmethod
    def type_validation_for_load(
        cls, load_from: Union[str, Path], arg_name: str, arg: Any, type_or_types: Union[Type, List[Type]]
    ):
        msg = f"When loading {cls.__name__} from {load_from} expected '{arg_name}' "
        if isinstance(type_or_types, list):
            if not any(isinstance(arg, t) for t in type_or_types):
                raise ValueError(
                    msg + f"to be of one of the following types: {type_or_types}, but got type {type(arg)}"
                )
        else:
            if not isinstance(arg, type_or_types):
                raise ValueError(msg + f"to be of type {type_or_types}, but got type {type(arg)}")

    @classmethod
    def load(
        cls: Type[SpecWithSourceType],
        spec: Optional[str] = None,
        source: Optional[str] = None,
        hash: Optional[Dict[str, str]] = None,  # todo: make hash non-optional
        kwargs: Optional[Dict[str, Any]] = None,
        rel_path: Path = Path("."),
        **spec_kwargs: Dict[str, Any],
    ) -> SpecWithSourceType:
        if spec is not None:
            assert spec.endswith(".yaml") or spec.endswith(".yml"), spec
        assert spec is not None or source is not None, (spec, source)
        assert isinstance(rel_path, Path), type(rel_path)
        cls.type_validation_for_load(spec or rel_path, "spec", spec, [type(None), str])
        cls.type_validation_for_load(spec or rel_path, "source", source, [type(None), str])
        cls.type_validation_for_load(spec or rel_path, "hash", hash, [type(None), dict])
        cls.type_validation_for_load(spec or rel_path, "kwargs", kwargs, [type(None), dict])

        hash_ = hash or {}
        kwargs = kwargs or {}

        if source is None:
            assert spec is not None
            spec_source = Source(source=spec, hash_=hash_, rel_path=rel_path)
            loaded_kwargs = spec_source.get()
            if not isinstance(loaded_kwargs, dict):
                raise ValueError(f"Expected loaded spec to be of type dict. Loaded spec from {spec}")
            if "source" in loaded_kwargs:
                return cls.load(spec=spec, rel_path=rel_path, kwargs=kwargs, **spec_kwargs, **loaded_kwargs)
        else:
            loaded_kwargs = {"source": Source(source=source, hash_=hash_, rel_path=rel_path)}

        return cls(_rel_path=rel_path, kwargs=kwargs, **spec_kwargs, **loaded_kwargs)

    @property
    def object_(self):
        # todo: check source hash
        if self._object is None:
            self._object = self.source.get()

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
        self.type_validation("data_range", data_range, list)
        self.type_validation("description", description, str)

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


if __name__ == "__main__":
    s = Spec(Path("/"))
    print(s)
