from typing import Any
from xml.etree import ElementTree

import pytest

from bioimageio.core._utils.xml_utils import etree_to_dict, xml_to_dict


@pytest.mark.parametrize(
    "xml,expected",
    [
        ("<root/>", {"root": None}),
        ("<root>  value  </root>", {"root": "value"}),
        ("<root>   </root>", {"root": ""}),
        ('<root unit="um"/>', {"root": {"@unit": "um"}}),
        ('<root unit="um"> 2 </root>', {"root": {"@unit": "um", "#text": "2"}}),
        ('<root unit="um"> </root>', {"root": {"@unit": "um"}}),
        (
            '<root unit="um"> caption <x>1</x><x>2</x><y/></root>',
            {"root": {"@unit": "um", "#text": "caption", "x": ["1", "2"], "y": None}},
        ),
        ("<root>\n <child>text</child>\n</root>", {"root": {"child": "text"}}),
    ],
)
def test_xml_conversion(xml: str, expected: dict[str, Any]):
    assert xml_to_dict(xml) == expected
    assert etree_to_dict(ElementTree.fromstring(xml)) == expected


def test_malformed_xml():
    with pytest.raises(ElementTree.ParseError):
        _ = xml_to_dict("<root>")
