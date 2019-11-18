import logging
import numpy
import re
import requests
import sys
import yaml

from importlib import import_module
from pathlib import Path
from pybio.spec.spec import Spec
from subprocess import call
from typing import Optional, TypeVar, Dict, Any, List, Union, Type
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

SpecWithSourceType = TypeVar("SpecWithSourceType", bound="SpecWithSource")


class Source:
    # todo: improve cache location
    cache_path = Path(__file__).parent.parent.parent / "cache"

    local_file_path: Optional[Path] = None
    module_name: Optional[str] = None
    object_name: str

    def __init__(self, source: str, hash: Dict[str, str], rel_path: Path = Path("."), require_git: bool = True):
        assert isinstance(source, str), type(source)
        assert isinstance(hash, dict), type(hash)
        assert isinstance(rel_path, Path), type(rel_path)
        assert isinstance(require_git, bool), type(require_git)
        self.original_source = source
        self.hash_ = hash
        self.require_git = require_git

        uri = urlparse(source)
        if uri.scheme == "doi" or uri.netloc == "dx.doi.org":
            uri = self.resolve_doi(uri)

        self.object_name = uri.fragment
        assert uri.params == "", uri.params
        assert uri.query == "", uri.query

        if uri.scheme == "file" or uri.scheme == "" and (uri.path.startswith(".") or uri.path.startswith("/")):
            assert uri.netloc == "", uri.netloc
            self.local_file_path = rel_path / uri.path if uri.path.startswith(".") else Path(uri.path)
        elif uri.scheme == "":
            if self.object_name:
                # <module name>.<sub module name>#<object name>
                self.module_name = uri.scheme
            else:
                # <module name>.<sub module name>.<object name>
                last_dot_idx = uri.path.rfind(".")
                self.module_name = uri.path[:last_dot_idx]
                self.object_name = uri.path[last_dot_idx + 1 :]

                if not re.match("[a-zA-Z_][a-zA-Z0-9_]*", self.object_name):
                    raise ValueError(f"Invalid object name {self.object_name} extracted from source {source}")

        elif uri.netloc == "github.com":
            orga, repo_name, blob, commit_id, *in_repo_path = uri.path.strip("/").split("/")
            in_repo_path = "/".join(in_repo_path)
            cached_repo_path = self.cache_path / orga / repo_name
            self.local_file_path = cached_repo_path / in_repo_path
            cached_repo_path = cached_repo_path.resolve().as_posix()

            self.download_commands = [
                ["git", "clone", f"{uri.scheme}://{uri.netloc}/{orga}/{repo_name}.git", cached_repo_path],
                # -C <working_dir> available in git 1.8.5+
                # https://github.com/git/git/blob/5fd09df3937f54c5cfda4f1087f5d99433cce527/Documentation/RelNotes/1.8.5.txt#L115-L116
                ["git", "-C", cached_repo_path, "checkout", "--force", commit_id]
            ]

        elif require_git:
            raise ValueError(f"Invalid source: {source}")
        elif uri.scheme in ["https", "http"]:
            file_name = uri.path.strip("/")
            self.local_file_path = self.cache_path / file_name
        else:
            raise ValueError(f"Unknown uri scheme {uri.scheme}")

        self.uri = uri

    @staticmethod
    def resolve_doi(uri):
        raise NotImplementedError

    def download(self) -> None:
        """download and cache source from doi or url"""
        if self.local_file_path is None:
            return

        if self.local_file_path.exists():
            # todo: check hash
            return  # or not

        assert self.cache_path in self.local_file_path.parents, "only download files to cache folder!"

        uri = self.uri
        if self.download_commands:
            for command in self.download_commands:
                call(command)

            assert self.local_file_path.exists(), self.local_file_path
            if self.object_name and self.object_name[0] == "L" and self.object_name[1:].isdigit():
                # get object name from line number
                line_number = int(self.object_name[1:])
                with self.local_file_path.open() as f:
                    for ln, line in enumerate(f):
                        if ln == line_number:
                            match = re.match("((class)|(def)) (?P<obj_name>\D\S*):", self.object_name)
                            if match:
                                self.object_name = match.group("obj_name")
                                logger.info(f"found object name '{self.object_name}' in {self.original_source}")

        else:
            url = f'https://{uri.netloc}/{uri.path.strip("/")}'
            r = requests.get(url)
            self.local_file_path.parent.mkdir(parents=True, exist_ok=True)
            with self.local_file_path.open("wb") as f:
                f.write(r.content)

            logger.info("downloaded %s", self.local_file_path)

        assert self.local_file_path.exists(), self.local_file_path
        # todo: check hash

    def get(self) -> Any:
        self.download()

        if self.local_file_path is None:
            if not self.module_name:
                raise ValueError(f"Invalid source: {self.original_source}")
            # import from dependency  # todo: improve
            dep = import_module(self.module_name)
            return getattr(dep, self.object_name)
        elif self.local_file_path.suffix in [".yml", ".yaml"]:
            assert self.object_name == "", self.object_name
            with self.local_file_path.open() as f:
                return yaml.safe_load(f)
        elif self.local_file_path.suffix in [".npy", ".npz"]:
            assert self.object_name == "", self.object_name
            return numpy.load(self.local_file_path)
        elif self.local_file_path.suffix in [".torch", ".pt", ".pth"]:
            import torch

            state_dict = torch.load(self.local_file_path)
            if self.object_name:
                return state_dict[self.object_name]
            else:
                return state_dict
        elif self.local_file_path.suffix == ".py" and self.object_name:
            # import <object_name> from local (python) file  # todo: improve
            cur_sys_path = list(sys.path)
            logger.debug("temporarily resetting sys.path to %s", self.local_file_path.parent.as_posix())
            sys.path = [self.local_file_path.parent.as_posix()]
            logger.debug("attempting to import %s", self.local_file_path.stem)
            dep = import_module(self.local_file_path.stem)
            sys.path = cur_sys_path
            return getattr(dep, self.object_name)
        else:
            raise NotImplementedError(f"Unknown source type {self.original_source}")


class SpecWithSource(Spec):
    def __init__(
        self,
        name: str,
        _rel_path: Path,
        source: str,
        kwargs: Dict[str, Any] = None,
        required_kwargs: Optional[List[str]] = None,
        optional_kwargs: Optional[Dict[str, Any]] = None,
        hash: Optional[Dict[str, str]] = None,  # todo: make hash mandatory
        _require_git: bool = True,
    ):
        self.type_validation("source", source, str)
        self.type_validation("kwargs", kwargs, [type(None), dict])
        self.type_validation("required_kwargs", required_kwargs, [type(None), list])
        self.type_validation("optional_kwargs", optional_kwargs, [type(None), dict])
        self.type_validation("hash", hash, [type(None), dict])  # todo: make hash mandatory

        hash = hash or {}
        kwargs = kwargs or {}
        required_kwargs = required_kwargs or []
        optional_kwargs = optional_kwargs or {}

        for req in required_kwargs:
            if req not in kwargs:
                raise ValueError(f"Missing kwarg '{req}' for spec {name}")

        for kw in kwargs:
            if kw not in required_kwargs and kw not in optional_kwargs:
                raise ValueError(
                    f"Unexpected kwarg '{kw}' for spec {name}. Required kwargs: {required_kwargs}. Optional kwargs: {optional_kwargs}"
                )

        kwargs = {**optional_kwargs, **kwargs}

        super().__init__(name=name, _rel_path=_rel_path)
        self.source = Source(source=source, hash=hash, rel_path=_rel_path, require_git=_require_git)
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
        spec: str,
        hash: Optional[Dict[str, str]] = None,  # todo: make hash non-optional
        kwargs: Optional[Dict[str, Any]] = None,
        _rel_path: Path = Path("."),
        **spec_kwargs: Dict[str, Any],
    ) -> SpecWithSourceType:

        assert spec.endswith(".yaml") or spec.endswith(
            ".yml"
        ), spec  # todo: account for uri fragment (#...) and git hash (if at end)
        assert isinstance(_rel_path, Path), type(_rel_path)
        cls.type_validation_for_load(spec or _rel_path, "spec", spec, [type(None), str])
        cls.type_validation_for_load(spec or _rel_path, "hash", hash, [type(None), dict])
        cls.type_validation_for_load(spec or _rel_path, "kwargs", kwargs, [type(None), dict])

        hash = hash or {}
        kwargs = kwargs or {}

        spec_source = Source(source=spec, hash=hash, rel_path=_rel_path, require_git=True)
        loaded_kwargs = spec_source.get()
        if not isinstance(loaded_kwargs, dict):
            raise ValueError(f"Expected loaded spec to be of type dict. Loaded spec from {spec}")

        if "kwargs" in loaded_kwargs:
            raise ValueError(
                f"'kwargs' specified in {spec}! Specificaiton files are not allowed to specify kwargs directly, "
                f"instead they may specify a list of 'required_kwargs' or a dictionary of 'optional_kwargs' (with "
                f"default values)"
            )

        return cls(_rel_path=_rel_path, kwargs=kwargs, **spec_kwargs, **loaded_kwargs)

    @property
    def object_(self):
        if self._object is None:
            self._object = self.source.get()

        return self._object
