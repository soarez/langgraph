"""Concurrency and identity: who may advance which task, and when.

One conversation is one thread, and a thread is a serial lineage of state. That
makes exclusion a correctness property rather than a performance one, and it
makes *which* task a message advances a security property: the SDK's own
`contextId`/`taskId` agreement check never runs, so a caller can otherwise
present a victim's `taskId` alongside a context it legitimately owns.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest
from a2a.server.context import ServerCallContext
from a2a.types.a2a_pb2 import (
    GetTaskRequest,
    ListTasksRequest,
    Task,
    TaskState,
    TaskStatus,
)
from google.protobuf import json_format
from langgraph.store.memory import InMemoryStore

from langgraph.a2a.executor import CHECKPOINT_METADATA_KEY
from tests.conftest import (
    ConditionalStore,
    answer,
    artifact_text,
    build_app,
    find_pause,
    http_client,
    make_client,
    pauses,
    send,
    text_message,
)


async def test_a_turn_arriving_while_another_runs_waits_for_it(client: Any) -> None:
    """Running is not parked, and the difference is whether the wait is bounded.

    A run is bounded by its own timeout, so queueing behind one ends. Queueing
    behind a question does not, because nothing obliges a counterparty to answer
    — that case is refused instead.
    """
    first = await send(client, text_message("hello 0"))

    turns = await asyncio.gather(
        *(
            send(client, text_message(f"hello {n}", context_id=first.context_id))
            for n in range(1, 4)
        )
    )

    assert [t.status.state for t in turns] == [TaskState.TASK_STATE_COMPLETED] * 3
    assert sorted(artifact_text(t)[0] for t in turns) == [
        f"echo: hello {n}" for n in range(1, 4)
    ]


async def test_a_turn_waits_out_a_claim_it_cannot_see_the_run_of() -> None:
    """Beside a second replica the wait moves from the lock to the store.

    An in-process lock excludes nothing across processes, so a conversation
    another replica is working looks idle here. What says otherwise is the task
    record, and the arriving turn waits on that rather than running a second
    turn into the same thread.
    """
    task_store = ConditionalStore(InMemoryStore())
    app = build_app(task_store=task_store, multi_replica=True)
    claim = await _claim(task_store, "shared-context")

    async def release() -> None:
        await asyncio.sleep(0.1)
        claim.status.state = TaskState.TASK_STATE_COMPLETED
        await task_store.save(claim, _OWNER)

    async with http_client(app, token="alice") as http:
        client = await make_client(http)
        turn, _ = await asyncio.gather(
            send(client, text_message("hello world", context_id="shared-context")),
            release(),
        )

    assert turn.status.state == TaskState.TASK_STATE_COMPLETED
    assert artifact_text(turn) == ["echo: hello world"]


async def test_a_claim_nothing_ever_clears_is_refused_rather_than_swept() -> None:
    """No lease means no way to tell a live claim from an abandoned one.

    So the wait is bounded by the same timeout that bounds a run, and what the
    caller gets at the end of it is the refusal — naming the task to clear —
    rather than a turn that runs anyway or a sweep this package cannot justify.
    """
    task_store = ConditionalStore(InMemoryStore())
    app = build_app(task_store=task_store, multi_replica=True, run_timeout=0.2)
    claim = await _claim(task_store, "abandoned-context")

    async with http_client(app, token="alice") as http:
        client = await make_client(http)
        with pytest.raises(Exception, match=claim.id):
            await send(
                client, text_message("hello world", context_id="abandoned-context")
            )

    still_there = await task_store.get(claim.id, _OWNER)
    assert still_there.status.state == TaskState.TASK_STATE_WORKING


_OWNER = ServerCallContext(state={"a2a_subject": "alice"})
"""The subject the claim belongs to, so the arriving turn can read it at all."""


async def _claim(task_store: ConditionalStore, context_id: str) -> Any:
    """A task record as another replica's live run leaves it in the store."""
    task = Task(
        id=uuid.uuid4().hex,
        context_id=context_id,
        status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
    )
    await task_store.save(task, _OWNER)
    return task


async def test_a_victims_task_id_cannot_be_advanced_from_another_context(
    app: Any, alice: Any, bob: Any
) -> None:
    """Owning a context is not permission to move a task that lives elsewhere."""
    victim = await send(await make_client(alice), text_message("ask me"))

    attacker = await make_client(bob)
    mine = await send(attacker, text_message("hello world"))

    with pytest.raises(Exception) as caught:
        await send(
            attacker,
            text_message("Ada", task_id=victim.id, context_id=mine.context_id),
        )

    assert "not found" in str(caught.value).lower()

    # And the victim's task is untouched: still waiting, still answerable.
    fetched = await (await make_client(alice))._transport.get_task(
        GetTaskRequest(id=victim.id)
    )
    assert fetched.status.state == TaskState.TASK_STATE_INPUT_REQUIRED


async def test_a_message_to_a_terminal_task_is_refused(client: Any) -> None:
    """Two layers answer this, and the executor's is the one that always can.

    The SDK refuses when the active task starts, which is what fires here. It
    cannot catch a message that was already queued when the task closed, so the
    executor re-checks; both produce a refusal and neither advances the task.
    """
    done = await send(client, text_message("hello world"))
    assert done.status.state == TaskState.TASK_STATE_COMPLETED

    with pytest.raises(Exception) as caught:
        await send(
            client,
            text_message("more", task_id=done.id, context_id=done.context_id),
        )

    message = str(caught.value).lower()
    assert "completed" in message or "cannot take another message" in message

    fetched = await client._transport.get_task(GetTaskRequest(id=done.id))
    assert artifact_text(fetched) == ["echo: hello world"]


async def test_a_pause_records_the_checkpoint_it_paused_at(client: Any) -> None:
    """Resuming targets that checkpoint, not whatever the thread reached later."""
    task = await send(client, text_message("ask me"))

    fetched = await client._transport.get_task(GetTaskRequest(id=task.id))
    metadata = json_format.MessageToDict(fetched.metadata)
    assert metadata[CHECKPOINT_METADATA_KEY]["checkpoint_id"]
    # checkpoint_ns is carried deliberately: addressing a checkpoint without it
    # raises inside the checkpointer.
    assert "checkpoint_ns" in metadata[CHECKPOINT_METADATA_KEY]


async def test_answering_one_of_two_pauses_does_not_re_ask_it(client: Any) -> None:
    """`StateSnapshot.interrupts` keeps reporting an answered pause.

    The pending set is computed from the graph's tasks that have no result, so a
    partial answer leaves exactly the unanswered question outstanding.
    """
    task = await send(client, text_message("parallel please"))
    north = find_pause(task, "North?")
    south = find_pause(task, "South?")

    still_waiting = await send(
        client,
        answer(
            north["interrupt_id"], "up", task_id=task.id, context_id=task.context_id
        ),
    )

    assert still_waiting.status.state == TaskState.TASK_STATE_INPUT_REQUIRED
    outstanding = [p["payload"]["question"] for p in pauses(still_waiting)]
    assert outstanding == ["South?"], outstanding

    finished = await send(
        client,
        answer(
            south["interrupt_id"], "down", task_id=task.id, context_id=task.context_id
        ),
    )
    assert finished.status.state == TaskState.TASK_STATE_COMPLETED


async def test_a_resume_payload_of_hex_keys_is_not_read_as_interrupt_ids(
    client: Any,
) -> None:
    """LangGraph detects the resume-map form by key shape.

    A client payload whose keys happen to be 32-character hex must reach the
    graph as a value, not be reinterpreted as a map of interrupt ids.
    """
    task = await send(client, text_message("ask me"))
    pause = find_pause(task, "What is your name?")
    hex_keys = {"0123456789abcdef0123456789abcdef": "surprise"}

    resumed = await send(
        client,
        answer(
            pause["interrupt_id"],
            hex_keys,
            task_id=task.id,
            context_id=task.context_id,
        ),
    )

    assert resumed.status.state == TaskState.TASK_STATE_COMPLETED
    assert artifact_text(resumed) == [f"hello {hex_keys}"]


async def test_a_refused_turn_does_not_keep_the_claim_it_took() -> None:
    """The claim is a write, so a turn that never runs must not keep it.

    Stated as the guarantee rather than as its mechanism, because two things
    provide it: the executor puts the record back, and `a2a-sdk` 1.1.2 also
    fails the task when the executor raises. Either would do. Neither doing it
    leaves the refused task at `WORKING`, holding the conversation against
    everyone including itself, and no request fails to say so.
    """
    task_store = ConditionalStore(InMemoryStore())
    app = build_app(task_store=task_store, multi_replica=True)

    async with http_client(app, token="alice") as http:
        client = await make_client(http)
        paused = await send(client, text_message("ask me"))
        assert paused.status.state == TaskState.TASK_STATE_INPUT_REQUIRED

        with pytest.raises(Exception, match="waiting for an answer"):
            await send(
                client, text_message("hello world", context_id=paused.context_id)
            )

        working = await task_store.list(
            ListTasksRequest(
                context_id=paused.context_id, status=TaskState.TASK_STATE_WORKING
            ),
            _OWNER,
        )
        assert not working.tasks, "the refused turn is still holding the conversation"

        # And the conversation is still usable, which is the point of releasing.
        pause = find_pause(paused, "What is your name?")
        resumed = await send(
            client,
            answer(
                pause["interrupt_id"],
                "Ada",
                task_id=paused.id,
                context_id=paused.context_id,
            ),
        )
    assert artifact_text(resumed) == ["hello Ada"]
