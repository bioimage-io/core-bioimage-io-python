import pytest

from dataclasses import dataclass
from pybio.spec import utils, spec_types, fields, schema
from typing import Optional, Any
from marshmallow import post_load

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


@dataclass
class Content:
    data: str


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
        class MyTransformer(utils.NodeTransformer):
            def visit_URL(self, node):
                return self.Transform(Content(f"content of url {node.url}"))

        assert isinstance(tree.left.right, self.URL)
        transformer = MyTransformer()
        transformer.visit(tree)
        assert isinstance(tree.left.right, Content)


@dataclass
class MySpec(spec_types.Node):
    uri_a: spec_types.URI
    uri_b: spec_types.URI


class SubSpec(schema.Schema):
    axes = fields.Axes()


class Spec(schema.Schema):
    uri_a = fields.SpecURI(SubSpec)
    uri_b = fields.SpecURI(SubSpec)

    @post_load
    def convert(self, data, **kwargs):
        return MySpec(**data)


class TestTraversingSpecURI:

    def test_resolve_spec(self):
        tree = Spec().load({
            "uri_a": "https://example.com",
            "uri_b": "../file.yml",
        })

        class MyTransformer(utils.NodeTransformer):
            def visit_URI(self, node):
                res = {"axes": "xyc"}
                return self.Transform(node.loader.load(res))

        transformer = MyTransformer()
        transformer.visit(tree)
        assert {"axes": "xyc"} == tree.uri_a
        assert {"axes": "xyc"} == tree.uri_b
