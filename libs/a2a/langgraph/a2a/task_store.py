"""An A2A `TaskStore` backed by a LangGraph `BaseStore`.

A2A task state and graph state are two durable stores, and a server is only as
durable as the weaker of them. Pairing a Postgres checkpointer with the SDK's
in-memory task store produces a system that loses the task and keeps the
thread: after a restart the caller's answer to a question arrives as a fresh
turn, and the suspended graph waits forever. Putting both in the same backend
removes the divergence rather than managing it.

Not solved here: the read-modify-write on `save` has no compare-and-set,
because no `BaseStore` offers one. Two writers to one task can lose an update.
The SDK serialises requests per task within a process, so this is a
multi-replica concern.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from google.protobuf import json_format

from a2a.server.context import ServerCallContext
from a2a.server.owner_resolver import OwnerResolver
from a2a.server.tasks import TaskStore
from a2a.types.a2a_pb2 import (
    ListTasksRequest,
    ListTasksResponse,
    Message,
    Part,
    Role,
    Task,
    TaskState,
)
from a2a.utils.constants import DEFAULT_LIST_TASKS_PAGE_SIZE
from a2a.utils.errors import InvalidParamsError
from a2a.utils.task import decode_page_token, encode_page_token
from langgraph.a2a.authorization import subject_scope

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore

logger = logging.getLogger(__name__)


@runtime_checkable
class ConditionalTaskStore(Protocol):
    """A task store that can save a task only if nobody else changed it first.

    `a2a-sdk` 1.1.2's `TaskStore` is save / get / list / delete, and `save`
    takes no precondition or version; `BaseStore.aput` takes none either. So
    there is no primitive under this package on which mutual exclusion across
    replicas can be built, and a lock inside one process is not one — beside a
    second replica it produces no error, it produces a lost turn that both
    callers were told had succeeded.

    This is the narrow extension that closes it. A deployment is excluded across
    replicas only if its store implements this; one that does not is
    single-replica, and `create_a2a_app` says so at construction rather than
    leaving it to be discovered.

    **Two guarantees, and the second is what actually excludes.** A version read
    with `get_versioned` and passed back to `save_if_unchanged` stops two
    writers losing each other's update to *one* task. That is not enough on its
    own: two replicas opening two different tasks in one conversation write two
    different records and never collide. So a store that supports
    `multi_replica=True` must also refuse a conditional write that would leave
    **two `WORKING` tasks in one context**. A unique partial index on
    `(owner, context_id) WHERE state = WORKING` is the whole implementation, and
    it is what makes the claim below a claim rather than a hope.

    `WORKING` and not `SUBMITTED`, because the SDK accepts a task and writes it
    `SUBMITTED` through the plain `save` before any of this code runs: a
    constraint over that state would reject the acceptance itself. `WORKING`
    means a replica is executing the turn now, which is the thing there can only
    be one of.

    A replica takes a conversation with one such write and releases it by the
    ordinary state write that ends the turn: a task that completes, fails or
    parks at a question is no longer `WORKING`, so the next claim succeeds.

    Nothing shipped here implements it — neither the SDK's stores nor
    `BaseStoreTaskStore`, because `BaseStore` offers neither a version nor a
    constraint. It is declared so a deployment whose storage can (Postgres,
    SQLite, anything with an index) has something to implement against.
    """

    async def get_versioned(
        self, task_id: str, context: ServerCallContext
    ) -> tuple[Task | None, Any]:
        """Read a task and the version token that a later write can assert on.

        Returns:
            The task and its version, or `(None, version)` when no record
            exists — the version then stands for "absent", which is what a
            caller creating a task passes back.
        """
        ...

    async def save_if_unchanged(
        self, task: Task, version: Any, context: ServerCallContext
    ) -> bool:
        """Save `task` only if its stored version is still `version`.

        Returns:
            `True` when the write happened. `False` when it did not, which means
            either that the record moved under the caller or that the write
            would have left two `WORKING` tasks in one context. Both say the same
            thing to a caller — re-read and decide again — so they are not
            distinguished.
        """
        ...


@runtime_checkable
class ReconcilableTaskStore(Protocol):
    """A task store that can find and close what a dead process left behind.

    A task's progress is tied to the process running it: if that process dies,
    its record says `WORKING` and nothing in this package sweeps, because
    nothing here runs unprompted. A poller then waits for ever on a task no one
    is executing.

    Reconciliation is the cheap half of the fix — at start, whatever the store
    can enumerate that is non-terminal and has no producer in this process is
    failed with a stated reason. It cannot resurrect the run; it converts silence
    into an answer. Recovering the work itself needs a lease or a handoff, which
    is the deployment's substrate rather than this package's.

    Optional, because enumerating across owners is not something `TaskStore`
    expresses: `list` is scoped to the caller. A store that owns its backing
    storage can do it — `BaseStoreTaskStore` does.
    """

    async def fail_orphaned(self, reason: str) -> int:
        """Fail every task left *running*, returning how many were closed.

        A paused task is not orphaned — it is waiting on a checkpoint, and it
        resumes whenever the answer arrives. Only `SUBMITTED` and `WORKING`
        records belong to a process that is no longer there.
        """
        ...


class BaseStoreTaskStore(TaskStore):
    """Tasks in a `BaseStore`, namespaced by owner.

    Args:
        store: any `BaseStore` — `AsyncSqliteStore`, a Postgres store, or
            `InMemoryStore` if you want the SDK's default with LangGraph's
            namespacing and nothing more.
        owner_resolver: maps a call context onto the scope a task is stored
            under, which is what keeps one caller from reading another's task by
            guessing its id. The default is the resolved subject, falling back
            to the authenticated principal on RPCs that carry no message — see
            `authorization.subject_scope`, and note what that fallback means for
            a subject declared in message metadata.
        namespace_prefix: leading namespace elements, if this store is shared.
    """

    def __init__(
        self,
        store: BaseStore,
        *,
        owner_resolver: OwnerResolver = subject_scope,
        namespace_prefix: tuple[str, ...] = ("a2a", "tasks"),
    ) -> None:
        self._store = store
        self._owner_resolver = owner_resolver
        self._prefix = namespace_prefix

    def _namespace(self, context: ServerCallContext) -> tuple[str, ...]:
        return (*self._prefix, self._owner_resolver(context) or "")

    async def save(self, task: Task, context: ServerCallContext) -> None:
        await self._store.aput(
            self._namespace(context),
            task.id,
            {
                "task": json_format.MessageToDict(task),
                "context_id": task.context_id,
                "state": int(task.status.state),
                "timestamp": task.status.timestamp.ToJsonString()
                if task.status.HasField("timestamp")
                else "",
            },
        )

    async def fail_orphaned(self, reason: str) -> int:
        """Close what an earlier process left running. See `ReconcilableTaskStore`.

        Scans every owner partition, which `list` cannot do because it answers
        for one caller. Safe to run at start of a single-replica deployment; on
        more than one it would close another replica's live work, which is why
        `create_a2a_app` only calls it when the deployment has not declared
        itself multi-replica.
        """
        closed = 0
        for namespace in await self._store.alist_namespaces(prefix=self._prefix):
            for item in await self._store.asearch(namespace, limit=_SEARCH_LIMIT):
                task = _to_task(item.value)
                if task is None or task.status.state not in RUNNING_STATES:
                    # Only work that was *running* is orphaned by a process
                    # dying. A task at input-required is parked on a checkpoint
                    # and resumes whenever its answer arrives — closing those
                    # would destroy the one thing that survives a restart.
                    continue
                task.status.state = TaskState.TASK_STATE_FAILED
                task.status.message.CopyFrom(
                    Message(
                        message_id=uuid.uuid4().hex,
                        context_id=task.context_id,
                        task_id=task.id,
                        role=Role.ROLE_AGENT,
                        parts=[Part(text=reason)],
                    )
                )
                await self._store.aput(
                    namespace,
                    task.id,
                    {
                        "task": json_format.MessageToDict(task),
                        "context_id": task.context_id,
                        "state": int(task.status.state),
                        "timestamp": task.status.timestamp.ToJsonString()
                        if task.status.HasField("timestamp")
                        else "",
                    },
                )
                closed += 1
        if closed:
            logger.warning(
                "failed %d task(s) left non-terminal by an earlier run", closed
            )
        return closed

    async def get(self, task_id: str, context: ServerCallContext) -> Task | None:
        item = await self._store.aget(self._namespace(context), task_id)
        if item is None:
            return None
        return _to_task(item.value)

    async def delete(self, task_id: str, context: ServerCallContext) -> None:
        await self._store.adelete(self._namespace(context), task_id)

    async def list(
        self, params: ListTasksRequest, context: ServerCallContext
    ) -> ListTasksResponse:
        items = await self._store.asearch(self._namespace(context), limit=_SEARCH_LIMIT)
        tasks = [task for item in items if (task := _to_task(item.value)) is not None]

        if params.context_id:
            tasks = [t for t in tasks if t.context_id == params.context_id]
        if params.status:
            tasks = [t for t in tasks if t.status.state == params.status]
        if params.HasField("status_timestamp_after"):
            after = params.status_timestamp_after.ToJsonString()
            tasks = [t for t in tasks if _timestamp(t) >= after]

        tasks.sort(key=lambda t: (_timestamp(t), t.id), reverse=True)

        total = len(tasks)
        start = 0
        if params.page_token:
            start_id = decode_page_token(params.page_token)
            for index, task in enumerate(tasks):
                if task.id == start_id:
                    start = index
                    break
            else:
                raise InvalidParamsError(f"Invalid page token: {params.page_token}")

        page_size = params.page_size or DEFAULT_LIST_TASKS_PAGE_SIZE
        end = start + page_size
        return ListTasksResponse(
            tasks=tasks[start:end],
            next_page_token=encode_page_token(tasks[end].id) if end < total else None,
            total_size=total,
            page_size=page_size,
        )


RUNNING_STATES = (
    TaskState.TASK_STATE_SUBMITTED,
    TaskState.TASK_STATE_WORKING,
)
"""A task some process is working on now.

Two readers: reconciliation, because this is what a dead process orphans and a
pause is not — a pause is waiting, not abandoned; and a turn arriving beside a
replica whose run it cannot see, because this is what it waits out.
"""

_SEARCH_LIMIT = 1000
"""Tasks read from the store before filtering. Paging happens after."""


def _timestamp(task: Task) -> str:
    if task.HasField("status") and task.status.HasField("timestamp"):
        return task.status.timestamp.ToJsonString()
    return ""


def _to_task(value: dict) -> Task | None:
    payload = value.get("task")
    if not isinstance(payload, dict):
        return None
    task = Task()
    json_format.ParseDict(payload, task)
    return task
