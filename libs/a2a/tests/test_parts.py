"""Part conversion, including the file shapes 0.3-era adapters get wrong."""

from __future__ import annotations

import base64

import pytest
from a2a.types.a2a_pb2 import Part
from a2a.utils.errors import InvalidParamsError
from google.protobuf import json_format, struct_pb2

from langgraph.a2a.parts import (
    part_to_content_block,
    parts_data,
    parts_text,
    parts_to_content,
)
from langgraph.a2a.state import StateAdapter


def test_a_single_text_part_becomes_a_plain_string() -> None:
    assert parts_to_content([Part(text="hello")]) == "hello"


def test_several_parts_become_content_blocks() -> None:
    content = parts_to_content(
        [Part(text="look"), Part(url="https://x/y.png", media_type="image/png")]
    )

    assert content == [
        {"type": "text", "text": "look"},
        {"type": "image", "url": "https://x/y.png", "mime_type": "image/png"},
    ]


def test_binary_file_parts_keep_their_bytes() -> None:
    """A file flattened to a text description loses the file."""
    block = part_to_content_block(
        Part(raw=b"\x89PNG\r\n", media_type="image/png", filename="a.png")
    )

    assert block == {
        "type": "image",
        "base64": base64.b64encode(b"\x89PNG\r\n").decode(),
        "mime_type": "image/png",
        "extras": {"filename": "a.png"},
    }


def test_an_unknown_media_type_is_a_file_block() -> None:
    block = part_to_content_block(Part(raw=b"x", media_type="application/pdf"))

    assert block["type"] == "file"


def test_an_empty_part_is_an_error_not_a_silent_drop() -> None:
    with pytest.raises(ValueError, match="none of text/raw/url/data"):
        part_to_content_block(Part())


def test_data_parts_are_not_message_content() -> None:
    value = struct_pb2.Value()
    json_format.ParseDict({"a": 1}, value)

    assert part_to_content_block(Part(data=value)) is None
    assert parts_to_content([Part(data=value)]) is None
    assert parts_data([Part(data=value)]) == [{"a": 1.0}]


def test_parts_text_joins_only_text() -> None:
    assert parts_text([Part(text="a"), Part(url="u"), Part(text="b")], " ") == "a b"


def test_the_adapter_always_builds_a_user_turn() -> None:
    state = StateAdapter().to_graph_input([Part(text="hi")])

    assert state == {"messages": [{"role": "user", "content": "hi"}]}


def test_the_adapter_refuses_data_parts_by_default() -> None:
    value = struct_pb2.Value()
    json_format.ParseDict({"messages": []}, value)

    with pytest.raises(InvalidParamsError, match="no structured input"):
        StateAdapter().to_graph_input([Part(data=value)])


def test_declared_data_parts_land_on_one_key() -> None:
    value = struct_pb2.Value()
    json_format.ParseDict({"messages": ["evil"], "sku": "x"}, value)

    state = StateAdapter(input_data_key="order").to_graph_input(
        [Part(text="hi"), Part(data=value)]
    )

    assert state == {
        "messages": [{"role": "user", "content": "hi"}],
        "order": {"messages": ["evil"], "sku": "x"},
    }
