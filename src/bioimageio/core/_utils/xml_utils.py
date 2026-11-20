from __future__ import annotations

from collections import defaultdict
from typing import Any
from xml.etree import ElementTree


def xml_to_dict(xml_string: str):
    return etree_to_dict(ElementTree.XML(xml_string))


def etree_to_dict(t: ElementTree.Element):
    # adapted from https://stackoverflow.com/a/10077069
    d: dict[str, Any] = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd: defaultdict[str, list[Any]] = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(("@" + k, v) for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]["#text"] = text
        else:
            d[t.tag] = text
    return d
