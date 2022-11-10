import ast
import sys
from pathlib import Path
from typing import Any, Sequence, Set

import black


def get_tree(path: Path):
    src = path.read_text()
    return ast.parse(src)


def write_tree(tree, path):
    new_src = ast.unparse(ast.fix_missing_locations(tree))
    new_src = black.format_str(new_src, mode=black.FileMode(line_length=120))
    path.write_text(new_src)


# turn op function defs into async function defs
class OpTransformer(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AsyncFunctionDef:
        """any function in ops is an operator. make them async"""
        self.generic_visit(node)
        return ast.AsyncFunctionDef(
            name=node.name,
            args=node.args,
            body=node.body,
            decorator_list=node.decorator_list,
            returns=node.returns,
            type_comment=node.type_comment,
        )


class ImportTransformer(ast.NodeTransformer):
    def __init__(self, *, module_allow_list: Sequence[str] = tuple()):
        self.known_ops = set()
        self.module_allow_list = frozenset(
            {mn for mn in sys.stdlib_module_names if not mn.startswith("_")} | set(module_allow_list)
        )

    def visit_Import(self, node: ast.Import) -> ast.Import:
        for alias in node.names:
            if alias.name.split(".")[0] not in self.module_allow_list:
                raise ValueError(f"Invalid 'import {alias.name}'.")

        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        if node.module == "ops":
            for alias_node in node.names:
                op_name = alias_node.name
                self.known_ops.add(op_name)
                if alias_node.asname is not None:
                    raise ValueError(
                        f"Please import operator names without 'as', i.e. use '{op_name}' instead of '{alias_node.asname}'."
                    )
            node.module = "compiled_ops"
        elif node.module.split(".")[0] in self.module_allow_list:
            pass
        else:
            raise ValueError(f"Unsupported import from {node.module}")

        return node


class OpCallTransformer(ast.NodeTransformer):
    def __init__(self, known_ops: Set[str]):
        self.known_ops = known_ops

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AsyncFunctionDef:
        """any function in wfs is a workflow. make them async"""
        self.generic_visit(node)
        return ast.AsyncFunctionDef(
            name=node.name,
            args=node.args,
            body=node.body,
            decorator_list=node.decorator_list,
            returns=node.returns,
            type_comment=node.type_comment,
        )

    def visit_Call(self, node: ast.Call) -> Any:
        """await any operator call"""
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            if node.func.id in self.known_ops:
                return ast.Await(node)
            else:
                return node
        elif isinstance(node.func, ast.Attribute):
            return node  # e.g. method call
        else:
            raise NotImplementedError(node.func)


ops_path = Path("ops.py")
ops_tree = get_tree(ops_path)
ops_tree = OpTransformer().visit(ops_tree)
write_tree(ops_tree, ops_path.with_name("compiled_" + ops_path.name))

wfs_path = Path("wfs.py")
wfs_tree = get_tree(wfs_path)

allowed_module_names = []  # todo: add modules from appropriate env to allow-list
import_transformer = ImportTransformer(module_allow_list=allowed_module_names)
wfs_tree = import_transformer.visit(wfs_tree)
wfs_tree = OpCallTransformer(known_ops=import_transformer.known_ops).visit(wfs_tree)

write_tree(wfs_tree, wfs_path.with_name("compiled_" + wfs_path.name))
