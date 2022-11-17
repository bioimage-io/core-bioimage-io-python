import ast
import asyncio
import logging
import os
from functools import partial
from inspect import getmembers, isfunction
from pathlib import Path
from typing import List, Optional

from imjoy_rpc.hypha import connect_to_server

from bioimageio.core import contrib
from ._ast import get_ast_tree

logger = logging.getLogger(__name__)


async def start_contrib_service(contrib_name: str, server_url: Optional[str] = None):
    server = await connect_to_server({"server_url": server_url or get_contrib_server_url(contrib_name)})

    contrib_part = getattr(contrib, contrib_name)
    service_name = f"BioImageIO {' '.join(n.capitalize() for n in contrib_name.split('_'))} Module"
    service_config = dict(
        name=service_name,
        id=f"bioimageio-{contrib_name}",
        config={
            "visibility": "public",
            "run_in_executor": True,  # This will make sure all the sync functions run in a separate thread
        },
    )

    for func_name, func in getmembers(contrib_part, isfunction):
        assert func_name not in service_config
        service_config[func_name] = func

    await server.register_service(service_config)

    logger.info(f"{service_name} service registered at workspace: {server.config.workspace}")


class ImportCollector(ast.NodeVisitor):
    def __init__(self):
        self.imported: List[str] = []

    def visit_Import(self, node: ast.Import):
        raise ValueError("Found 'import' statement. Expected 'from .<local module> import <func>' only")

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if not node.level:
            raise ValueError(f"Unsupported absolute import from {node.module}")

        if "." in node.module:
            raise ValueError(f"Unsupported nested import from {node.module}")

        for alias_node in node.names:
            self.imported.append(alias_node.name)
            if alias_node.asname is not None:
                raise ValueError(
                    f"Please import contrib functions without 'as', i.e. use '{alias_node.name}' instead of '{alias_node.asname}'."
                )


SERVER_URL_ENV_NAME = "BIOIMAGEIO_CONTRIB_URL"
DEFAULT_SERVER_URL = "http://localhost:9000"


def get_contrib_specific_server_url_env_name(contrib_name):
    return f"BIOIMAGEIO_{contrib_name.capitalize()}_URL"


def get_contrib_server_url(contrib_name) -> str:
    return os.getenv(
        get_contrib_specific_server_url_env_name(contrib_name), os.getenv(SERVER_URL_ENV_NAME, DEFAULT_SERVER_URL)
    )


class RemoteContrib:
    def __init__(self, contrib_name: str, server_url: Optional[str] = None):
        self.server_url = server_url or get_contrib_server_url(contrib_name)
        self.contrib_name = contrib_name
        self.contrib = None
        local_src = Path(__file__).parent.parent / contrib_name / "local.py"
        tree = get_ast_tree(local_src)
        import_collector = ImportCollector()
        import_collector.visit(tree)
        self.__all__ = import_collector.imported
        self.service_funcs = {}
        for name in self.__all__:
            setattr(self, name, partial(self._service_call, _contrib_func_name=name))

    def __await__(self):
        yield from self._ainit().__await__()

    async def _ainit(self):
        try:
            server = await asyncio.create_task(connect_to_server({"server_url": self.server_url}))
        except Exception as e:
            raise Exception(
                f"Failed to connect to {self.server_url}. "
                f"Make sure {get_contrib_specific_server_url_env_name(self.contrib_name)} or {SERVER_URL_ENV_NAME} "
                f"is set or {self.server_url} is running."
            ) from e
        try:
            contrib_service = await server.get_service(f"bioimageio-{self.contrib_name}")
        except Exception as e:
            raise Exception(
                f"bioimageio-{self.contrib_name} service not found. Start with 'python -m bioimageio.core.contrib.{self.contrib_name}' in a suitable (conda) environment."
            ) from e
            # todo: start contrib service entry point, e.g. f"bioimageio start {contrib_name}"

        self.service_funcs = {name: getattr(contrib_service, name) for name in self.__all__}
        return self

    async def _service_call(self, *args, _contrib_func_name, **kwargs):
        await self
        return await self.service_funcs[_contrib_func_name](*args, **kwargs)
