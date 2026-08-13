"""Conversion between A2A 1.0 `Part`s and LangChain message content.

A2A 1.0 flattened `Part`: the oneof is `text | raw | url | data`, with
`media_type` and `filename` as siblings, and `metadata` alongside them. There is
no `kind` discriminator and no nested `file` object — those are the 0.3 shapes,
and a part built in that shape will not parse here.
"""

from __future__ import annotations

import base64
from typing import Any

from google.protobuf import json_format, struct_pb2

from a2a.types.a2a_pb2 import Part

_MEDIA_BLOCK_TYPES = ("image", "audio", "video")


def value_to_python(value: struct_pb2.Value) -> Any:
    """A protobuf `Value` as plain Python."""
    return json_format.MessageToDict(value)


def python_to_value(payload: Any) -> struct_pb2.Value:
    """Plain Python as a protobuf `Value`.

    The payload must already be JSON-compatible; run it through
    `langgraph.a2a._encoding.encode` first if it came from user code.
    """
    value = struct_pb2.Value()
    json_format.ParseDict(payload, value)
    return value


def text_part(text: str) -> Part:
    """A text part."""
    return Part(text=text)


def data_part(payload: Any, metadata: dict[str, Any] | None = None) -> Part:
    """A `data` part, optionally carrying part-level metadata."""
    part = Part(data=python_to_value(payload))
    if metadata:
        json_format.ParseDict(metadata, part.metadata)
    return part


def part_metadata(part: Part) -> dict[str, Any]:
    """Part-level metadata as plain Python, `{}` when absent."""
    if not part.HasField("metadata"):
        return {}
    return json_format.MessageToDict(part.metadata)


def parts_text(parts: list[Part], delimiter: str = "") -> str:
    """The text parts of a message, joined."""
    return delimiter.join(p.text for p in parts if p.WhichOneof("content") == "text")


def parts_data(parts: list[Part]) -> list[Any]:
    """The payloads of the `data` parts of a message."""
    return [value_to_python(p.data) for p in parts if p.WhichOneof("content") == "data"]


def _block_type(media_type: str | None) -> str:
    if media_type:
        top = media_type.split("/", 1)[0].strip().lower()
        if top in _MEDIA_BLOCK_TYPES:
            return top
    return "file"


def part_to_content_block(part: Part) -> dict[str, Any] | None:
    """Map one A2A `Part` onto a LangChain content block.

    Returns `None` for a `data` part: structured input is not message content.
    Where it goes instead is the caller's decision — see `StateAdapter`.
    """
    which = part.WhichOneof("content")
    if which == "text":
        return {"type": "text", "text": part.text}
    if which == "data":
        return None
    if which in ("raw", "url"):
        block: dict[str, Any] = {"type": _block_type(part.media_type or None)}
        if which == "raw":
            block["base64"] = base64.b64encode(part.raw).decode()
        else:
            block["url"] = part.url
        if part.media_type:
            block["mime_type"] = part.media_type
        if part.filename:
            block["extras"] = {"filename": part.filename}
        return block
    raise ValueError("A2A Part carries none of text/raw/url/data")


def parts_to_content(parts: list[Part]) -> str | list[dict[str, Any]] | None:
    """The content of one LangChain message built from A2A parts.

    A single text part becomes a plain string, because that is what most graphs
    expect; anything richer becomes a content-block list. `data` parts are not
    message content and are skipped.
    """
    blocks = [
        block for part in parts if (block := part_to_content_block(part)) is not None
    ]
    if not blocks:
        return None
    if len(blocks) == 1 and blocks[0]["type"] == "text":
        return blocks[0]["text"]
    return blocks
