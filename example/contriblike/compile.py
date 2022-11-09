from ast import (
    AsyncFunctionDef,
    Await,
    Call,
    FunctionDef,
    Import,
    ImportFrom,
    NodeTransformer,
    fix_missing_locations,
    parse,
    unparse,
)
from pathlib import Path
from typing import Any, Set

import black


def get_tree(path: Path):
    src = path.read_text()
    return parse(src)


def write_tree(tree, path):
    new_src = unparse(fix_missing_locations(tree))
    new_src = black.format_str(new_src, mode=black.FileMode(line_length=120))
    path.write_text(new_src)


# turn op function defs into async function defs
class OpTransformer(NodeTransformer):
    def visit_FunctionDef(self, node: FunctionDef) -> AsyncFunctionDef:
        """any function in ops is an operator. make them async"""
        self.generic_visit(node)
        return AsyncFunctionDef(
            name=node.name,
            args=node.args,
            body=node.body,
            decorator_list=node.decorator_list,
            returns=node.returns,
            type_comment=node.type_comment,
        )


class ImportTransformer(NodeTransformer):
    def __init__(self):
        self.known_ops = set()

    def visit_Import(self, node: Import) -> Import:
        raise ValueError("Invalid import")

    def visit_ImportFrom(self, node: ImportFrom) -> ImportFrom:
        if node.module == "ops":
            for alias_node in node.names:
                op_name = alias_node.name
                self.known_ops.add(op_name)
                if alias_node.asname is not None:
                    raise ValueError(
                        f"Please import operator names without 'as', i.e. use '{op_name}' instead of '{alias_node.asname}'."
                    )
            node.module = "compiled_ops"
        elif node.module in ["typing"]:  # module white list
            pass
        else:
            raise ValueError(f"Unsupported import from {node.module}")

        return node


class OpCallTransformer(NodeTransformer):
    def __init__(self, known_ops: Set[str]):
        self.known_ops = known_ops

    def visit_FunctionDef(self, node: FunctionDef) -> AsyncFunctionDef:
        """any function in wfs is a workflow. make them async"""
        self.generic_visit(node)
        return AsyncFunctionDef(
            name=node.name,
            args=node.args,
            body=node.body,
            decorator_list=node.decorator_list,
            returns=node.returns,
            type_comment=node.type_comment,
        )

    def visit_Call(self, node: Call) -> Any:
        """await any operator call"""
        self.generic_visit(node)
        if node.func.id in self.known_ops:
            return Await(node)
        else:
            return node


ops_path = Path("ops.py")
ops_tree = get_tree(ops_path)
ops_tree = OpTransformer().visit(ops_tree)
write_tree(ops_tree, ops_path.with_name("compiled_" + ops_path.name))

wfs_path = Path("wfs.py")
wfs_tree = get_tree(wfs_path)
import_transformer = ImportTransformer()
wfs_tree = import_transformer.visit(wfs_tree)
wfs_tree = OpCallTransformer(known_ops=import_transformer.known_ops).visit(wfs_tree)

write_tree(wfs_tree, wfs_path.with_name("compiled_" + wfs_path.name))
