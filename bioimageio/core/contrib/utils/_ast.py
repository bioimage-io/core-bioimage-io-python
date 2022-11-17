import ast
from pathlib import Path


def get_ast_tree(path: Path):
    src = path.read_text()
    return ast.parse(src)
