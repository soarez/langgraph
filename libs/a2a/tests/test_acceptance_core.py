"""Core acceptance criteria, driven by the unmodified `a2a-sdk` client."""

from __future__ import annotations

from typing import Any

from a2a.client.card_resolver import A2ACardResolver
from a2a.types.a2a_pb2 import GetTaskRequest, TaskState
from a2a.utils.constants import PROTOCOL_VERSION_1_0
from google.protobuf import json_format

from langgraph.a2a import StateAdapter, last_ai_text
from tests.agent import LONG_ANSWER
from tests.conftest import (
    BASE_URL,
    answer,
    artifact_data,
    artifact_text,
    build_app,
    collect,
    find_pause,
    http_client,
    last_task,
    make_client,
    send,
    text_message,
)


async def test_card_resolves_and_advertises_declared_skills(alice: Any) -> None:
    card = await A2ACardResolver(alice, BASE_URL).get_agent_card()

    assert card.name == "toy-langgraph-agent"
    assert [skill.id for skill in card.skills] == ["echo", "greet"]
    assert "bearer" in card.security_schemes


async def test_card_declares_the_1_0_interface_with_enum_cased_binding(
    alice: Any,
) -> None:
    card = await A2ACardResolver(alice, BASE_URL).get_agent_card()

    primary = card.supported_interfaces[0]
    assert primary.protocol_version == PROTOCOL_VERSION_1_0
    # Lowercase here fails transport selection with an error naming neither the
    # card nor the field.
    assert primary.protocol_binding == "JSONRPC"


async def test_client_selects_the_strict_1_0_transport(client: Any) -> None:
    assert type(client._transport).__name__ == "JsonRpcTransport"


async def test_message_returns_a_task_that_completes(client: Any) -> None:
    task = await send(client, text_message("hello world"))

    assert task is not None
    assert task.status.state == TaskState.TASK_STATE_COMPLETED


async def test_result_arrives_as_an_artifact_with_structured_output(
    client: Any,
) -> None:
    task = await send(client, text_message("hello world"))

    assert artifact_text(task) == ["echo: hello world"]
    assert artifact_data(task) == [{"echoed": True}]


async def test_get_task_returns_the_artifacts(client: Any) -> None:
    task = await send(client, text_message("hello world"))

    fetched = await client._transport.get_task(GetTaskRequest(id=task.id))

    assert artifact_text(fetched) == ["echo: hello world"]


async def test_streaming_reports_working_then_appends_then_completes(
    streaming_client: Any,
) -> None:
    events = await collect(streaming_client, text_message("stream please"))

    states = [
        TaskState.Name(event.status_update.status.state)
        for kind, event in events
        if kind == "status_update"
    ]
    assert "TASK_STATE_WORKING" in states
    assert states[-1] == "TASK_STATE_COMPLETED"

    chunks = [
        event.artifact_update for kind, event in events if kind == "artifact_update"
    ]
    assert [c for c in chunks if c.append], (
        "no appended chunk: the stream was not incremental"
    )
    # Reconstructed the way A2A defines it: append extends the artifact, and an
    # event without append replaces it. A client applying those rules must end
    # up with what GetTask holds.
    assert _reconstruct(chunks) == [LONG_ANSWER, {"streamed": True}]


def _reconstruct(chunks: list[Any]) -> list[Any]:
    """Apply A2A artifact semantics to a stream, as a client would."""
    parts: list[Any] = []
    for chunk in chunks:
        if not chunk.append:
            parts = []
        for part in chunk.artifact.parts:
            if part.HasField("text"):
                parts.append(part.text)
            elif part.WhichOneof("content") == "data":
                parts.append(json_format.MessageToDict(part.data))
    # Adjacent text chunks are one value once joined, as any renderer would.
    joined: list[Any] = []
    for part in parts:
        if isinstance(part, str) and joined and isinstance(joined[-1], str):
            joined[-1] += part
        else:
            joined.append(part)
    return joined


async def test_a_streamed_answer_ends_up_identical_to_a_fetched_one(
    streaming_client: Any,
) -> None:
    """The transcript is the stream, so the bytes are what `GetTask` holds.

    Not the parts: a long answer is chunked as it arrives and the chunks stay
    where they fell. What must not differ is the text a client assembles from
    them.
    """
    task = await send(streaming_client, text_message("stream please"))

    fetched = await streaming_client._transport.get_task(GetTaskRequest(id=task.id))

    assert "".join(artifact_text(fetched)) == LONG_ANSWER
    assert artifact_data(fetched) == [{"streamed": True}]


async def test_streaming_coalesces_tokens_rather_than_one_part_per_token(
    streaming_client: Any,
) -> None:
    """A part per token is quadratic against the task store; buffering is the point."""
    events = await collect(streaming_client, text_message("stream please"))

    chunks = [e.artifact_update for k, e in events if k == "artifact_update"]
    parts = [p for chunk in chunks for p in chunk.artifact.parts]
    assert len(parts) < LONG_ANSWER.count(" ") / 4


async def test_node_exception_maps_to_failed(client: Any) -> None:
    task = await send(client, text_message("please fail"))

    assert task.status.state == TaskState.TASK_STATE_FAILED


async def test_completed_task_carries_the_turn_in_its_history(client: Any) -> None:
    task = await send(client, text_message("hello world"))

    roles = [message.role for message in task.history]
    assert roles, "the task recorded no history"


async def test_history_length_trims_the_task_and_not_the_conversation(
    client: Any,
) -> None:
    """`historyLength` is a question about the task record, not about memory.

    A task that paused and was answered has several messages of its own. Asking
    for one gets one — and the graph, asked afterwards what it was told, still
    has every turn, because trimming a response is not forgetting.
    """
    paused = await send(client, text_message("ask me"))
    pause = find_pause(paused, "What is your name?")
    task = await send(
        client,
        answer(
            pause["interrupt_id"],
            "Ada",
            task_id=paused.id,
            context_id=paused.context_id,
        ),
    )
    assert len(task.history) > 1, "not enough history to trim"

    trimmed = await client._transport.get_task(
        GetTaskRequest(id=task.id, history_length=1)
    )
    assert len(trimmed.history) == 1
    assert trimmed.artifacts == task.artifacts

    later = await send(
        client, text_message("what did i say", context_id=task.context_id)
    )
    assert "ask me" in artifact_text(later)[0]


async def test_a_graph_that_speaks_twice_produces_a_transcript_of_both(
    alice: Any,
) -> None:
    """With no output mapping the artifact is everything the model said.

    No run knows which of its turns is the last until it ends, so nothing
    assembled while tokens arrive can be a verdict on which one was the answer.
    A graph that speaks before it answers therefore publishes both — and
    publishes the same both whether or not anyone was watching, because
    streaming decides when the bytes arrive and never what they are.
    """
    streamed_app = build_app()
    plain_app = build_app(stream_tokens=False)

    async with http_client(streamed_app, token="alice") as http:
        client = await make_client(http, streaming=True)
        streamed = await send(client, text_message("two turns please"))
        streamed = await client._transport.get_task(GetTaskRequest(id=streamed.id))

    async with http_client(plain_app, token="alice") as http:
        client = await make_client(http)
        plain = await send(client, text_message("two turns please"))

    assert artifact_text(streamed) == artifact_text(plain)
    assert artifact_data(streamed) == artifact_data(plain)
    assert artifact_text(plain) == ["looking", "found it"]


async def test_an_output_mapping_replaces_the_transcript_and_stops_the_stream(
    alice: Any,
) -> None:
    """The other shape: a graph that can state its result supplies a mapping.

    The result is read from the final state once the run is over, so there is
    nothing to stream — and no flag can say otherwise, which is why supplying
    the mapping is itself the switch.
    """
    app = build_app(
        state=StateAdapter(output_text=lambda state: last_ai_text(state)),
        stream_tokens=True,
    )

    async with http_client(app, token="alice") as http:
        client = await make_client(http, streaming=True)
        events = await collect(client, text_message("two turns please"))
        task = await client._transport.get_task(GetTaskRequest(id=last_task(events).id))

    chunks = [e.artifact_update for kind, e in events if kind == "artifact_update"]
    assert not [c for c in chunks if c.append], (
        "an output mapping must not stream token chunks"
    )
    assert artifact_text(task) == ["found it"]
