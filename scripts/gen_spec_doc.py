from pathlib import Path
from pprint import pprint

import marshmallow
from marshmallow import Schema
from marshmallow.fields import Field, Nested

import pybio.spec.schema

INTRO = f"""
# BioImage.IO Model Description File Specification (a.k.a. model.yaml spec) {pybio.spec.__version__}

A model entry in the bioimage.io model zoo is defined by a configuration file model.yaml. The configuration file must contain the following fields; optional fields are followed by [optional]. If a field is followed by [optional]*, they are optional depending on another field.
    """


def doc_from_field(field):
    if hasattr(field, "get_bioimageio_doc"):
        text = field.get_bioimageio_doc()
    else:
        text = "documentation missing" # field.__doc__
        if text is None:
            text = "missing"

    if field.required:
        assert not text.startswith('*')  # asterisk indicates a field is optional only dependent on other fields' values
    else:
        text = f"[optional]{'' if text.startswith('*') else ' '}" + text

    return text


def doc_dict_from_schema(obj):
    if isinstance(obj, Schema):
        return {name: doc_dict_from_schema(nested) for name, nested in obj.fields.items()}
    elif isinstance(obj, Nested):
        return doc_dict_from_schema(obj.schema)
    elif isinstance(obj, Field):
        return doc_from_field(obj)
    else:
        raise NotImplementedError(obj)


def markdown_from_doc_dict(doc_dict, indent: int = 0, output: str = ""):
    for key in sorted(doc_dict):
        if key not in ["format_version", "language", "source"]:  # todo: remove: skip undocumented fields
            continue

        output += f"{'  ' * indent}* `{key}` {doc_dict[key]}\n"

    return output


def get_model_spec_doc_as_markdown():
    doc = doc_dict_from_schema(pybio.spec.schema.Model())
    return markdown_from_doc_dict(doc)


def main():
    doc = get_model_spec_doc_as_markdown()
    pprint(doc)
    Path("bioimageio_model_spec.md").write_text(doc)


if __name__ == "__main__":
    main()
