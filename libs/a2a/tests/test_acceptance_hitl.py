"""Human-in-the-loop acceptance criteria.

What separates this from every other A2A agent is that `input-required` is a
suspended graph. These checks are the difference: a pause that carries an id, a
resume that continues the same task, several pauses answered independently, and
the re-execution the suspension costs.
"""

from __future__ import annotations

from typing import Any

import pytest
from a2a.types.a2a_pb2 import TaskState

from tests import agent
from tests.conftest import (
    answer,
    artifact_data,
    artifact_text,
    find_pause,
    pauses,
    send,
    status_text,
    text_message,
)


async def test_interrupt_surfaces_as_input_required(client: Any) -> None:
    task = await send(client, text_message("ask me"))

    assert task.status.state == TaskState.TASK_STATE_INPUT_REQUIRED


async def test_pause_is_machine_readable_and_human_readable(client: Any) -> None:
    task = await send(client, text_message("ask me"))

    found = pauses(task)
    assert len(found) == 1
    assert found[0]["payload"] == {"question": "What is your name?"}
    assert found[0]["interrupt_id"]
    assert found[0]["node"] == "respond", "the pause should name the node it came from"
    # The same question, in text, for a caller that speaks no extension.
    assert "What is your name?" in status_text(task)


async def test_resume_continues_the_same_task_with_a_structured_value(
    client: Any,
) -> None:
    task = await send(client, text_message("ask me"))
    pause = find_pause(task, "What is your name?")

    resumed = await send(
        client,
        answer(
            pause["interrupt_id"], "Ada", task_id=task.id, context_id=task.context_id
        ),
    )

    assert resumed.id == task.id
    assert resumed.status.state == TaskState.TASK_STATE_COMPLETED
    assert artifact_text(resumed) == ["hello Ada"]


async def test_plain_text_answers_a_single_pause(client: Any) -> None:
    """A caller that knows nothing about the encoding is not locked out."""
    task = await send(client, text_message("ask me"))

    resumed = await send(
        client, text_message("Grace", task_id=task.id, context_id=task.context_id)
    )

    assert artifact_text(resumed) == ["hello Grace"]


async def test_pydantic_interrupt_payload_round_trips(client: Any) -> None:
    task = await send(client, text_message("pydantic please"))
    pause = find_pause(task, "Approve the transfer?")

    assert pause["payload"] == {
        "question": "Approve the transfer?",
        "amount": 42.5,
        "tags": ["risk"],
    }
    resumed = await send(
        client,
        answer(
            pause["interrupt_id"], True, task_id=task.id, context_id=task.context_id
        ),
    )
    assert artifact_text(resumed) == ["approved: True"]


async def test_parallel_interrupts_are_both_reported_and_individually_resumable(
    client: Any,
) -> None:
    task = await send(client, text_message("parallel please"))

    found = pauses(task)
    assert len(found) == 2, f"expected two pauses, got {found}"
    north = find_pause(task, "North?")
    south = find_pause(task, "South?")
    assert north["interrupt_id"] != south["interrupt_id"]

    both = answer(
        north["interrupt_id"], "up", task_id=task.id, context_id=task.context_id
    )
    both.message.parts.append(answer(south["interrupt_id"], "down").message.parts[0])
    resumed = await send(client, both)

    assert resumed.status.state == TaskState.TASK_STATE_COMPLETED
    assert artifact_data(resumed) == [{"north": "up", "south": "down"}]


async def test_text_cannot_answer_when_several_pauses_are_outstanding(
    client: Any,
) -> None:
    task = await send(client, text_message("parallel please"))
    outstanding = [pause["interrupt_id"] for pause in pauses(task)]
    assert len(outstanding) == 2

    with pytest.raises(Exception) as caught:
        await send(
            client, text_message("up", task_id=task.id, context_id=task.context_id)
        )

    # Naming them is what makes the refusal actionable: the caller answers one.
    assert "cannot say which one" in str(caught.value)
    assert all(interrupt_id in str(caught.value) for interrupt_id in outstanding)


async def test_credential_request_parks_the_task_at_auth_required(client: Any) -> None:
    task = await send(client, text_message("credential please"))

    assert task.status.state == TaskState.TASK_STATE_AUTH_REQUIRED
    pause = pauses(task)[0]
    # Our own marker, not a borrowed one: a credential request is a distinct
    # kind of pause, answered exactly like any other.
    assert pause["kind"] == "credential_request"
    assert pause["payload"]["scheme"] == "github"
    assert "repo scope" in status_text(task)

    resumed = await send(
        client,
        answer(
            pause["interrupt_id"],
            "tok-123",
            task_id=task.id,
            context_id=task.context_id,
        ),
    )
    assert artifact_text(resumed) == ["used token tok-123"]


async def test_a_pre_pause_side_effect_runs_again_on_resume(client: Any) -> None:
    """Resumption re-executes the paused node from its start. Documented, and true."""
    agent.SIDE_EFFECTS.clear()
    task = await send(client, text_message("twice please"))
    assert agent.SIDE_EFFECTS == ["ran"]
    pause = find_pause(task, "Confirm?")

    await send(
        client,
        answer(
            pause["interrupt_id"], "yes", task_id=task.id, context_id=task.context_id
        ),
    )

    assert agent.SIDE_EFFECTS == ["ran", "ran"]


async def test_a_resume_that_omits_the_context_id_still_finds_its_pause(
    client: Any,
) -> None:
    """`taskId` alone identifies the task, so it must be enough to answer it.

    A resume carrying no `contextId` gets a freshly generated one from the SDK.
    Deriving the thread from that would send the answer to an empty thread, and
    the task would fail as though its pause had been lost.
    """
    task = await send(client, text_message("ask me"))
    pause = find_pause(task, "What is your name?")

    resumed = await send(client, answer(pause["interrupt_id"], "Ada", task_id=task.id))

    assert resumed.status.state == TaskState.TASK_STATE_COMPLETED
    assert artifact_text(resumed) == ["hello Ada"]
    assert resumed.context_id == task.context_id


async def test_a_node_that_pauses_twice_can_be_answered_twice(client: Any) -> None:
    """A second pause in the same node keeps the first one's interrupt id.

    The graph task also carries a result from the answer already given, so a
    pending set read from tasks-without-results would call this waiting graph
    empty and fail a task that is perfectly resumable.
    """
    task = await send(client, text_message("wizard please"))
    first = find_pause(task, "Step one?")

    again = await send(
        client,
        answer(first["interrupt_id"], "a", task_id=task.id, context_id=task.context_id),
    )
    assert again.status.state == TaskState.TASK_STATE_INPUT_REQUIRED
    second = find_pause(again, "Step two?")

    done = await send(
        client,
        answer(
            second["interrupt_id"], "b", task_id=task.id, context_id=task.context_id
        ),
    )

    assert done.status.state == TaskState.TASK_STATE_COMPLETED
    assert artifact_text(done) == ["wizard a/b"]
