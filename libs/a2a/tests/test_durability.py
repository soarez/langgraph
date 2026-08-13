"""Durability: what survives the process, and what the server refuses to pretend.

The claim under test is the one that separates a suspended graph from a turn
that ended: a task parked at `input-required` is still resumable after
everything holding it in memory is gone.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from a2a.server.tasks import (
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.types.a2a_pb2 import ListTasksRequest, TaskState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite import AsyncSqliteStore

from langgraph.a2a import A2AConfigurationError, BaseStoreTaskStore, subject_scope
from tests.agent import build_graph
from tests.conftest import (
    answer,
    artifact_text,
    build_app,
    find_pause,
    http_client,
    make_client,
    send,
    text_message,
)


@asynccontextmanager
async def durable_app(tmp_path: Any) -> AsyncIterator[Any]:
    """An app whose graph state and whose tasks are both on disk."""
    async with (
        AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.db")) as saver,
        AsyncSqliteStore.from_conn_string(str(tmp_path / "tasks.db")) as store,
    ):
        await saver.setup()
        await store.setup()
        yield build_app(
            graph=build_graph(checkpointer=saver),
            task_store=BaseStoreTaskStore(store),
        )


async def test_a_task_parked_at_input_required_survives_a_restart(
    tmp_path: Any,
) -> None:
    async with durable_app(tmp_path) as first:
        async with http_client(first, token="alice") as http:
            client = await make_client(http)
            paused = await send(client, text_message("ask me"))
            assert paused.status.state == TaskState.TASK_STATE_INPUT_REQUIRED
            pause = find_pause(paused, "What is your name?")

    # Everything above is gone: the saver, the store handle, the task registry,
    # the executor and the app. Only the two files remain.
    async with durable_app(tmp_path) as second:
        async with http_client(second, token="alice") as http:
            client = await make_client(http)
            resumed = await send(
                client,
                answer(
                    pause["interrupt_id"],
                    "Ada",
                    task_id=paused.id,
                    context_id=paused.context_id,
                ),
            )

    assert resumed.id == paused.id
    assert resumed.status.state == TaskState.TASK_STATE_COMPLETED
    assert artifact_text(resumed) == ["hello Ada"]


async def test_a_durable_checkpointer_with_in_memory_tasks_is_refused(
    tmp_path: Any,
) -> None:
    async with AsyncSqliteSaver.from_conn_string(
        str(tmp_path / "checkpoints.db")
    ) as saver:
        with pytest.raises(A2AConfigurationError, match="kept in memory"):
            build_app(
                graph=build_graph(checkpointer=saver),
                task_store=InMemoryTaskStore(),
            )
        with pytest.raises(A2AConfigurationError, match="kept in memory"):
            build_app(
                graph=build_graph(checkpointer=saver),
                task_store=BaseStoreTaskStore(InMemoryStore()),
            )


async def test_an_in_memory_pair_is_allowed(tmp_path: Any) -> None:
    """Both halves in memory is a coherent development setup, not a trap."""
    build_app(graph=build_graph(checkpointer=InMemorySaver()))


async def test_webhook_registrations_must_be_as_durable_as_the_tasks(
    tmp_path: Any,
) -> None:
    """A registration that evaporates leaves the task it belongs to unannounced.

    The task survives the restart, reaches a terminal state and fires nothing,
    because the config saying where to fire is gone. So the pairing is refused
    for the same reason durable checkpoints beside in-memory tasks are.
    """
    async with (
        AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.db")) as saver,
        AsyncSqliteStore.from_conn_string(str(tmp_path / "tasks.db")) as store,
    ):
        await saver.setup()
        await store.setup()
        with pytest.raises(A2AConfigurationError, match="kept in memory"):
            build_app(
                graph=build_graph(checkpointer=saver),
                task_store=BaseStoreTaskStore(store),
                push_config_store=InMemoryPushNotificationConfigStore(
                    owner_resolver=subject_scope
                ),
            )


async def test_task_store_round_trips_and_lists_by_context(tmp_path: Any) -> None:
    async with durable_app(tmp_path) as app:
        async with http_client(app, token="alice") as http:
            client = await make_client(http)
            first = await send(client, text_message("hello one"))
            await send(client, text_message("hello two", context_id=first.context_id))
            other_context = await send(client, text_message("hello three"))

            listed = await client._transport.list_tasks(
                ListTasksRequest(context_id=first.context_id, include_artifacts=True)
            )

    ids = {task.id for task in listed.tasks}
    assert first.id in ids
    assert other_context.id not in ids
    assert listed.total_size == 2
