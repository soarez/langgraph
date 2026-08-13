"""Continuity: the agent remembers the conversation it is in.

Every other criterion in this suite can pass while the agent forgets who it is
talking to. A2A makes a task terminal when it completes, so the second turn of a
conversation is a *new task carrying the same `contextId`* — and a server that
keys its thread on the task starts that turn on an empty checkpoint, answers
perfectly, and has no idea what was said a second ago.

That is the failure these checks exist for. It is invisible to a conformance
suite, invisible to a single-turn test, and the whole reason a LangGraph
checkpointer is worth having.
"""

from __future__ import annotations

from typing import Any

from a2a.types.a2a_pb2 import TaskState

from tests.conftest import (
    answer,
    artifact_text,
    find_pause,
    http_client,
    make_client,
    send,
    text_message,
)
from tests.test_durability import durable_app


async def test_a_second_turn_sees_the_first(client: Any) -> None:
    first = await send(client, text_message("remember: the sky is green"))

    second = await send(
        client, text_message("what did I say", context_id=first.context_id)
    )

    assert second.id != first.id, "a follow-up turn is a new task by definition"
    assert second.context_id == first.context_id
    assert "the sky is green" in artifact_text(second)[0]


async def test_a_turn_in_another_conversation_sees_nothing(client: Any) -> None:
    """Continuity is per conversation, not per caller."""
    await send(client, text_message("remember: the sky is green"))

    elsewhere = await send(client, text_message("what did I say"))

    assert "the sky is green" not in artifact_text(elsewhere)[0]


async def test_a_turn_after_an_interrupt_sees_both_question_and_answer(
    client: Any,
) -> None:
    paused = await send(client, text_message("ask me"))
    pause = find_pause(paused, "What is your name?")
    await send(
        client,
        answer(
            pause["interrupt_id"],
            "Ada",
            task_id=paused.id,
            context_id=paused.context_id,
        ),
    )

    later = await send(
        client, text_message("what did I say", context_id=paused.context_id)
    )

    # The greeting the resumed run produced is part of the conversation.
    assert "hello Ada" in artifact_text(later)[0]


async def test_the_conversation_survives_a_restart(tmp_path: Any) -> None:
    async with durable_app(tmp_path) as first_process:
        async with http_client(first_process, token="alice") as http:
            client = await make_client(http)
            first = await send(client, text_message("remember: the sky is green"))

    async with durable_app(tmp_path) as second_process:
        async with http_client(second_process, token="alice") as http:
            client = await make_client(http)
            second = await send(
                client, text_message("what did I say", context_id=first.context_id)
            )

    assert second.status.state == TaskState.TASK_STATE_COMPLETED
    assert "the sky is green" in artifact_text(second)[0]
