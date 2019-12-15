import pytest

from dataclasses import dataclass
from pybio.spec import utils, spec_types
from typing import Optional, Any

@dataclass
class MyNode(spec_types.Node):
    field_a: str
    field_b: int


def test_iter_fields():
    entry = MyNode("a", 42)
    assert [
        ("field_a", "a"),
        ("field_b", 42),
    ] == list(utils.iter_fields(entry))


class TestNodeVisitor:
    @dataclass
    class Tree(spec_types.Node):
        left: Any
        right: Any

    @dataclass
    class URL:
        url: str

    @pytest.fixture
    def tree(self):
        return self.Tree(
            self.Tree(self.Tree(None, None), self.URL("https://example.com")),
            self.Tree(None, self.Tree(None, self.Tree(None, None))),
        )

    def test_node(self, tree):
        visitor = utils.NodeVisitor()
        visitor.visit(tree)

    def test_node_transform(self, tree):
        @dataclass
        class Content:
            blob: str

        class MyTransformer(utils.NodeTransormer):
            def visit_URL(self, node):
                return self.Transform(Content(f"content of url {node.url}"))

        assert isinstance(tree.left.right, self.URL)
        transformer = MyTransformer()
        transformer.visit(tree)
        assert isinstance(tree.left.right, Content)
