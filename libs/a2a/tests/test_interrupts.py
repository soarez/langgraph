"""The pause encoding, unit level: the wire shape and the resume rules."""

from __future__ import annotations

import pytest
from google.protobuf import json_format

from langgraph.a2a._encoding import encode
from langgraph.a2a.interrupts import (
    INTERRUPT_ID_KEY,
    KIND_CREDENTIAL_REQUEST,
    KIND_INTERRUPT,
    KIND_INTERRUPT_RESPONSE,
    METADATA_KIND_KEY,
    VALUE_KEY,
    AmbiguousResume,
    CredentialRequest,
    PendingInterrupt,
    interrupt_part,
    interrupt_responses,
    prompt_text,
    resume_map,
)
from langgraph.a2a.parts import data_part, part_metadata, text_part, value_to_python


def response_part(interrupt_id: str, value: object) -> object:
    return data_part(
        {INTERRUPT_ID_KEY: interrupt_id, VALUE_KEY: value},
        metadata={METADATA_KIND_KEY: KIND_INTERRUPT_RESPONSE},
    )


def test_a_pause_carries_its_interrupt_id_under_our_own_key() -> None:
    """The keys are this project's. Borrowing ADK's would be a compatibility
    claim nobody has tested, on fields it defines a different schema for."""
    part = interrupt_part(PendingInterrupt(id="i1", value={"question": "Why?"}))

    assert part_metadata(part) == {METADATA_KIND_KEY: KIND_INTERRUPT}
    assert value_to_python(part.data) == {
        INTERRUPT_ID_KEY: "i1",
        "payload": {"question": "Why?"},
    }


def test_a_scalar_payload_needs_no_wrapper() -> None:
    part = interrupt_part(PendingInterrupt(id="i1", value="Your name?"))

    assert value_to_python(part.data)["payload"] == "Your name?"


def test_the_node_is_reported_when_known() -> None:
    part = interrupt_part(PendingInterrupt(id="i1", value="x", node="approve"))

    assert value_to_python(part.data)["node"] == "approve"


def test_a_credential_request_is_its_own_kind() -> None:
    part = interrupt_part(
        PendingInterrupt(id="i1", value=CredentialRequest(scheme="github"))
    )

    assert part_metadata(part)[METADATA_KIND_KEY] == KIND_CREDENTIAL_REQUEST
    assert value_to_python(part.data)["payload"]["scheme"] == "github"


def test_prompt_text_uses_only_human_readable_material() -> None:
    pending = [
        PendingInterrupt(id="a", value="plain question"),
        PendingInterrupt(id="b", value={"question": "keyed question"}),
        PendingInterrupt(id="c", value={"opaque": [1, 2, 3]}),
    ]

    assert prompt_text(pending) == "plain question\nkeyed question"


def test_answers_are_decoded_by_id() -> None:
    parts = [response_part("i1", "Ada"), response_part("i2", {"ok": True})]

    assert interrupt_responses(parts) == {"i1": "Ada", "i2": {"ok": True}}


def test_answers_for_interrupts_no_longer_pending_are_dropped() -> None:
    parts = [response_part("stale", "x")]

    assert resume_map(parts, "", [PendingInterrupt(id="live", value="?")]) == {}


def test_text_answers_a_single_pause() -> None:
    assert resume_map([], "Ada", [PendingInterrupt(id="i1", value="?")]) == {
        "i1": "Ada"
    }


def test_text_cannot_answer_two_pauses() -> None:
    pending = [PendingInterrupt(id="a", value="?"), PendingInterrupt(id="b", value="?")]

    with pytest.raises(AmbiguousResume):
        resume_map([], "Ada", pending)


def test_structured_answers_win_over_text() -> None:
    pending = [PendingInterrupt(id="a", value="?"), PendingInterrupt(id="b", value="?")]
    parts = [text_part("ignored"), response_part("a", 1), response_part("b", 2)]

    assert resume_map(parts, "ignored", pending) == {"a": 1, "b": 2}


def test_a_response_with_no_id_is_ignored() -> None:
    part = data_part(
        {VALUE_KEY: "x"}, metadata={METADATA_KIND_KEY: KIND_INTERRUPT_RESPONSE}
    )

    assert interrupt_responses([part]) == {}


def test_bytes_in_a_resume_value_are_decoded() -> None:
    part = response_part("i1", encode(b"raw"))

    assert interrupt_responses([part]) == {"i1": b"raw"}


def test_a_plain_data_part_is_not_a_resume() -> None:
    """Structured input and a resume are different things and must not collide."""
    part = data_part({INTERRUPT_ID_KEY: "i1", VALUE_KEY: "x"})

    assert interrupt_responses([part]) == {}


def test_metadata_is_a_struct_not_a_string() -> None:
    part = interrupt_part(PendingInterrupt(id="i1", value="x"))

    assert json_format.MessageToDict(part.metadata)[METADATA_KIND_KEY] == KIND_INTERRUPT
