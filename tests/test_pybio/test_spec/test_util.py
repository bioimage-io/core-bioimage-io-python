import pytest

from dataclasses import dataclass
from pybio.spec import utils, nodes, fields, schema
from typing import Any
from marshmallow import post_load


@dataclass
class MyNode(nodes.Node):
    field_a: str
    field_b: int


def test_iter_fields():
    entry = MyNode("a", 42)
    assert [("field_a", "a"), ("field_b", 42)] == list(utils.iter_fields(entry))


@dataclass
class Content:
    data: str


class TestNodeVisitor:
    @dataclass
    class Tree(nodes.Node):
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
            def transform_URL(self, node):
                return Content(f"content of url {node.url}")

        assert isinstance(tree.left.right, self.URL)
        transformer = MyTransformer()
        transformed_tree = transformer.transform(tree)
        assert isinstance(transformed_tree.left.right, Content)


@dataclass
class MySpec(nodes.Node):
    spec_uri_a: nodes.SpecURI
    spec_uri_b: nodes.SpecURI


class SubSpec(schema.Schema):
    axes = fields.Axes()


class Spec(schema.Schema):
    spec_uri_a = fields.SpecURI(SubSpec)
    spec_uri_b = fields.SpecURI(SubSpec)

    @post_load
    def convert(self, data, **kwargs):
        return MySpec(**data)


class TestTraversingSpecURI:
    def test_resolve_spec(self):
        tree = Spec().load({"spec_uri_a": "https://example.com", "spec_uri_b": "../file.yml"})

        class MyTransformer(utils.NodeTransformer):
            def transform_SpecURI(self, node):
                res = {"axes": "xyc"}
                return node.spec_schema.load(res)

        transformer = MyTransformer()
        transformed_tree = transformer.transform(tree)
        assert {"axes": "xyc"} == transformed_tree.spec_uri_a
        assert {"axes": "xyc"} == transformed_tree.spec_uri_b
