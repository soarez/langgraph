"""Work that outlives the connection — and what it does not outlive.

A2A's design centre is a task the caller does not sit and wait for: submit and
poll, subscribe, or be called back. All three work here while the process that
accepted the request is alive. None of them survives that process, and this file
draws the line in both directions rather than claiming either half.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from a2a.client.card_resolver import A2ACardResolver
from a2a.server.context import ServerCallContext
from a2a.server.tasks import InMemoryPushNotificationConfigStore
from a2a.types.a2a_pb2 import (
    AuthenticationInfo,
    DeleteTaskPushNotificationConfigRequest,
    GetTaskRequest,
    ListTaskPushNotificationConfigsRequest,
    ListTasksRequest,
    SendMessageConfiguration,
    TaskPushNotificationConfig,
    TaskState,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from langgraph.a2a import A2AConfigurationError, BaseStoreTaskStore, subject_scope
from langgraph.a2a.server import ORPHANED_TASK_MESSAGE
from langgraph.a2a.webhooks import (
    NOTIFICATION_TOKEN_HEADER,
    AuthenticatingPushSender,
    WebhookRefused,
    check_webhook_url,
)
from tests.agent import build_graph
from tests.conftest import (
    BASE_URL,
    answer,
    artifact_text,
    build_app,
    find_pause,
    http_client,
    make_client,
    send,
    text_message,
)

# -- submit and poll ---------------------------------------------------------


async def test_a_caller_can_submit_and_poll_rather_than_wait(client: Any) -> None:
    """`return_immediately` hands back a task id; `GetTask` collects the answer.

    Handled by the SDK's request handler, which keeps consuming the queue in a
    background task after answering — so this works as shipped, on one process,
    and is worth asserting rather than assuming.
    """
    request = text_message("hello world")
    request.configuration.CopyFrom(SendMessageConfiguration(return_immediately=True))

    submitted = None
    async for event in client.send_message(request):
        if event.WhichOneof("payload") == "task":
            submitted = event.task
            break

    assert submitted is not None
    assert submitted.status.state == TaskState.TASK_STATE_SUBMITTED

    fetched = await poll_until_terminal(client, submitted.id)

    assert fetched.status.state == TaskState.TASK_STATE_COMPLETED
    assert artifact_text(fetched) == ["echo: hello world"]


async def poll_until_terminal(client: Any, task_id: str, *, tries: int = 50) -> Any:
    """Poll for a terminal state, bounded — a test that can hang is not a test.

    The sleep matters as well as the bound: the work is finishing in a
    background task on this same event loop, and a tight polling loop starves
    it.
    """
    for _ in range(tries):
        task = await client._transport.get_task(GetTaskRequest(id=task_id))
        if task.status.state in (
            TaskState.TASK_STATE_COMPLETED,
            TaskState.TASK_STATE_FAILED,
        ):
            return task
        await asyncio.sleep(0.02)
    raise AssertionError(f"task {task_id} never reached a terminal state")


# -- what the process boundary costs ------------------------------------------


async def test_a_task_left_running_by_a_dead_process_is_closed_at_start() -> None:
    """Nothing sweeps, so a poller would otherwise wait on `WORKING` for ever.

    The reconciliation is honest about what it can do: it cannot resume the run,
    only convert silence into an answer the caller can act on.
    """
    store = InMemoryStore()
    task_store = BaseStoreTaskStore(store)

    # A task the previous process was running when it stopped.
    async with http_client(
        build_app(
            graph=build_graph(checkpointer=InMemorySaver()), task_store=task_store
        ),
        token="alice",
    ) as http:
        client = await make_client(http)
        finished = await send_and_strand(client, task_store)

    # The next process starts and closes it out.
    restarted = build_app(
        graph=build_graph(checkpointer=InMemorySaver()), task_store=task_store
    )
    await restarted.state.a2a.astart()

    async with http_client(restarted, token="alice") as http:
        client = await make_client(http)
        fetched = await client._transport.get_task(GetTaskRequest(id=finished))

    assert fetched.status.state == TaskState.TASK_STATE_FAILED
    assert ORPHANED_TASK_MESSAGE in " ".join(
        part.text for part in fetched.status.message.parts if part.HasField("text")
    )


async def send_and_strand(client: Any, task_store: BaseStoreTaskStore) -> str:
    """Run a task, then put its record back to `WORKING` as a dead process would."""
    task = None
    async for event in client.send_message(text_message("hello world")):
        if event.WhichOneof("payload") == "task":
            task = event.task

    context = ServerCallContext(state={"a2a_subject": "alice"})
    page = await task_store.list(ListTasksRequest(), context)
    stranded = next(t for t in page.tasks if t.id == task.id)
    stranded.status.state = TaskState.TASK_STATE_WORKING
    await task_store.save(stranded, context)
    return task.id


async def test_a_paused_task_survives_the_restart_that_closes_a_running_one() -> None:
    """Reconciliation must not touch a pause.

    A parked task is not orphaned by a process dying — it is waiting on a
    checkpoint, and resuming it later is the single thing this package promises
    survives a restart. A sweep that closed it would destroy exactly that.
    """
    task_store = BaseStoreTaskStore(InMemoryStore())
    checkpointer = InMemorySaver()

    async with http_client(
        build_app(graph=build_graph(checkpointer=checkpointer), task_store=task_store),
        token="alice",
    ) as http:
        client = await make_client(http)
        paused = await send(client, text_message("ask me"))
        assert paused.status.state == TaskState.TASK_STATE_INPUT_REQUIRED
        pause = find_pause(paused, "What is your name?")

    restarted = build_app(
        graph=build_graph(checkpointer=checkpointer), task_store=task_store
    )
    await restarted.state.a2a.astart()

    async with http_client(restarted, token="alice") as http:
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

    assert resumed.status.state == TaskState.TASK_STATE_COMPLETED
    assert artifact_text(resumed) == ["hello Ada"]


async def test_reconciliation_is_skipped_when_another_replica_may_be_running() -> None:
    """The same scan would close a live task belonging to another process."""
    server = build_app(task_store=BaseStoreTaskStore(InMemoryStore())).state.a2a
    server.multi_replica = True

    await server.astart()  # a no-op, and deliberately so


# -- webhooks ------------------------------------------------------------------


def test_a_webhook_must_be_https_and_public() -> None:
    """A registered URL is a destination the server will make requests to."""
    check_webhook_url("https://example.com/hook")

    with pytest.raises(WebhookRefused, match="must be https"):
        check_webhook_url("http://example.com/hook")
    with pytest.raises(WebhookRefused, match="loopback"):
        check_webhook_url("https://127.0.0.1/hook")
    with pytest.raises(WebhookRefused, match="loopback"):
        check_webhook_url("https://localhost/hook")
    with pytest.raises(WebhookRefused, match="cannot resolve"):
        check_webhook_url("https://nx.invalid/hook")

    # A deployment whose receivers really are internal can say so, and that one
    # switch covers both halves: the address and the missing TLS.
    check_webhook_url("https://127.0.0.1/hook", allow_private=True)
    check_webhook_url("http://127.0.0.1:9000/hook", allow_private=True)


async def test_a_refused_webhook_is_rejected_when_it_is_registered(alice: Any) -> None:
    """At registration, so the caller learns now rather than never hearing back."""
    app = build_app(
        push_config_store=InMemoryPushNotificationConfigStore(
            owner_resolver=subject_scope
        )
    )

    async with http_client(app, token="alice") as http:
        client = await make_client(http)
        task = None
        async for event in client.send_message(text_message("hello world")):
            if event.WhichOneof("payload") == "task":
                task = event.task

        with pytest.raises(Exception, match="loopback|https"):
            await client._transport.create_task_push_notification_config(
                TaskPushNotificationConfig(
                    task_id=task.id, url="http://127.0.0.1:9/steal"
                )
            )


async def test_a_webhook_cannot_be_registered_or_read_on_someone_elses_task() -> None:
    """A config is bound to its task, so seeing the task is the whole permission.

    The check is the SDK's — every config method resolves the task with the
    caller's own context first — but it is only as good as the task store's
    partitioning, and it is someone else's code. Asserted here rather than
    assumed, because a change upstream would otherwise turn a caller's callback
    URL into a way to read another subject's task events.
    """
    app = build_app(
        push_config_store=InMemoryPushNotificationConfigStore(
            owner_resolver=subject_scope
        )
    )

    async with http_client(app, token="alice") as http:
        victim = await send(await make_client(http), text_message("hello world"))
        registered = await (
            await make_client(http)
        )._transport.create_task_push_notification_config(
            TaskPushNotificationConfig(
                task_id=victim.id, url="https://example.com/alice-hook"
            )
        )

    async with http_client(app, token="mallory") as http:
        attacker = await make_client(http)
        with pytest.raises(Exception, match="[Nn]ot found"):
            await attacker._transport.create_task_push_notification_config(
                TaskPushNotificationConfig(
                    task_id=victim.id, url="https://example.com/mallory-hook"
                )
            )
        with pytest.raises(Exception, match="[Nn]ot found"):
            await attacker._transport.list_task_push_notification_configs(
                ListTaskPushNotificationConfigsRequest(task_id=victim.id)
            )
        with pytest.raises(Exception, match="[Nn]ot found"):
            await attacker._transport.delete_task_push_notification_config(
                DeleteTaskPushNotificationConfigRequest(
                    task_id=victim.id, id=registered.id
                )
            )


def test_a_config_store_keyed_by_the_principal_is_refused() -> None:
    """Tasks are partitioned by subject; registrations have to be as well.

    The SDK's default keys them by the authenticated principal, which is the
    same key right up until a credential rotates — at which point the task is
    still the caller's and the webhook it registered belongs to nobody.
    """
    with pytest.raises(A2AConfigurationError, match="owner_resolver"):
        build_app(push_config_store=InMemoryPushNotificationConfigStore())


def test_the_sender_carries_the_credentials_the_caller_registered() -> None:
    """The SDK's sender ignores `authentication` entirely, so a receiver checking
    `Authorization` drops every delivery while the sender reports success."""
    sender = AuthenticatingPushSender(None, None)  # only the header logic is under test

    headers = sender._headers(
        TaskPushNotificationConfig(
            url="https://example.com/hook",
            token="tok",
            authentication=AuthenticationInfo(scheme="Bearer", credentials="secret"),
        )
    )

    assert headers["Authorization"] == "Bearer secret"
    assert headers[NOTIFICATION_TOKEN_HEADER] == "tok"


# -- the card says which of these are available --------------------------------


async def test_push_notifications_are_advertised_only_when_configured(
    alice: Any,
) -> None:
    without = build_app()
    with_store = build_app(
        push_config_store=InMemoryPushNotificationConfigStore(
            owner_resolver=subject_scope
        )
    )

    async with http_client(without, token="alice") as http:
        assert not (
            await A2ACardResolver(http, BASE_URL).get_agent_card()
        ).capabilities.push_notifications

    async with http_client(with_store, token="alice") as http:
        assert (
            await A2ACardResolver(http, BASE_URL).get_agent_card()
        ).capabilities.push_notifications
