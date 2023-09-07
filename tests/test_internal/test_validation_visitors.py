from functools import singledispatchmethod

from bioimageio.spec._internal.base_nodes import Node
from bioimageio.spec.summary import ErrorOutcome

from bioimageio.core._internal.validation_visitors import Note, ValidationVisitor


def test_traversing_nodes():
    class MyVisitor(ValidationVisitor):
        @singledispatchmethod
        def visit(self, obj: type, note: Note = Note()):
            super().visit(obj, note)

        @visit.register
        def _visit_int(self, nr: int, note: Note = Note()):
            super().visit(nr, note)
            self.errors.append(ErrorOutcome(loc=note.loc, msg=f"nr: {nr}", type="got-int"))

    class NestedNode(Node, frozen=True):
        leaf: int

    class MyNode(Node, frozen=True):
        nested: NestedNode

    tree = {
        "a": MyNode(nested=NestedNode(leaf=1)),
        "b": [NestedNode(leaf=2), NestedNode(leaf=3)],
        "c": (NestedNode(leaf=4),),
        "d": {"deep": MyNode(nested=NestedNode(leaf=5))},
    }
    visitor = MyVisitor()
    visitor.visit(tree)
    assert len(visitor.errors) == [
        ErrorOutcome(loc=("a", "nested", "leaf"), msg="nr: 1", type="got-int"),
        ErrorOutcome(loc=("b", 0, "leaf"), msg="nr: 2", type="got-int"),
        ErrorOutcome(loc=("b", 1, "leaf"), msg="nr: 3", type="got-int"),
        ErrorOutcome(loc=("c", 0, "leaf"), msg="nr: 4", type="got-int"),
        ErrorOutcome(loc=("d", "deep", "nested", "leaf"), msg="nr: 5", type="got-int"),
    ]
