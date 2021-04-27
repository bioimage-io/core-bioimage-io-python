import collections
import dataclasses
import inspect
import typing
from pathlib import Path
from pprint import pprint

from marshmallow import Schema

import pybio.spec.schema
from pybio.spec.fields import DocumentedField, Nested, Union


@dataclasses.dataclass
class DocNode:
    description: str
    sub_docs: typing.OrderedDict[str, "DocNode"]
    options: typing.List["DocNode"]
    many: bool  # expecting a list of the described sub spec
    optional: bool

    def __post_init__(self):
        assert not (self.sub_docs and self.options)


def doc_from_schema(obj) -> typing.Union[typing.Dict[str, DocNode], DocNode]:
    description = ""
    options = []
    sub_docs = collections.OrderedDict()
    required = True
    if isinstance(obj, Nested):
        required = obj.required
        description += obj.bioimageio_description
        obj = obj.nested

    if inspect.isclass(obj) and issubclass(obj, pybio.spec.schema.PyBioSchema):
        obj = obj()

    description += obj.bioimageio_description
    print(type(obj), isinstance(obj, Schema))
    if isinstance(obj, pybio.spec.schema.PyBioSchema):

        def sort_key(name_and_nested_field):
            name, nested_field = name_and_nested_field
            if nested_field.bioimageio_description_order is None:
                manual_order = ""
            else:
                manual_order = f"{nested_field.bioimageio_description_order:09}"

            return f"{manual_order}{int(not nested_field.required)}{name}"

        sub_fields = sorted(obj.fields.items(), key=sort_key)
        sub_docs = collections.OrderedDict([(name, doc_from_schema(nested_field)) for name, nested_field in sub_fields])
    elif isinstance(obj, Union):
        required = obj.required
        options = [doc_from_schema(opt) for opt in obj._candidate_fields]
    else:
        required = obj.required
        assert isinstance(obj, DocumentedField), (type(obj), obj)

    return DocNode(
        description=description,
        sub_docs=sub_docs,
        options=options,
        many=isinstance(obj, Nested) and obj.many,
        optional=not required,
    )


def markdown_from_doc(doc: DocNode, indent: int = 0):
    optional = "[optional] " if doc.optional else ""
    list_descr = "[list] A list of: " if doc.many else ""

    if doc.sub_docs:
        sub_docs = [(name, sdn) for name, sdn in doc.sub_docs.items()]
        enumerate_symbol = "*"
    elif doc.options:
        sub_docs = [("", sdn) for sdn in doc.options]
        enumerate_symbol = "1."
    else:
        sub_docs = []
        enumerate_symbol = None

    sub_doc = ""
    for name, sdn in sub_docs:
        name = f"`{name}` " if name else ""
        sub_doc += f"{'  ' * indent}{enumerate_symbol} {name}{markdown_from_doc(sdn, indent+1)}"

    return f"{optional}{list_descr}{doc.description}\n{sub_doc}"


def get_model_spec_doc_as_markdown():
    doc = doc_from_schema(pybio.spec.schema.Model())
    return markdown_from_doc(doc)


def export_markdown_doc(path: Path):
    doc = get_model_spec_doc_as_markdown()
    path.write_text(doc)


if __name__ == "__main__":
    export_markdown_doc(Path(__file__).parent / "../generated/bioimageio_model_spec.md")
