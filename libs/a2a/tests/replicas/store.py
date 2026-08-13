"""A `ConditionalTaskStore` on SQLite, for the two-replica test.

The package ships no store that can exclude across processes, because
`BaseStore` offers neither a row version nor a constraint — the seam is a
protocol and filling it is the deployment's. This is what filling it looks like,
and it is here rather than in `langgraph/` for exactly that reason: a test needs
one real implementation to drive two processes against, and one file of SQL is
the smallest honest one.

Two columns do the work. `version` is bumped on every write and asserted by
`save_if_unchanged`, which stops two writers losing each other's update to one
task. The unique partial index does the part that actually excludes: at most one
`WORKING` task per `(owner, context_id)`, so a second replica cannot start a
turn in a conversation another replica is running.
"""

from __future__ import annotations

from typing import Any

import aiosqlite
from a2a.server.context import ServerCallContext
from a2a.server.owner_resolver import OwnerResolver
from a2a.server.tasks import TaskStore
from a2a.types.a2a_pb2 import (
    ListTasksRequest,
    ListTasksResponse,
    Task,
    TaskState,
)
from google.protobuf import json_format

from langgraph.a2a import subject_scope

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    owner      TEXT NOT NULL,
    id         TEXT NOT NULL,
    context_id TEXT NOT NULL,
    state      INTEGER NOT NULL,
    version    INTEGER NOT NULL,
    task       TEXT NOT NULL,
    PRIMARY KEY (owner, id)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_working_task_per_context
    ON tasks (owner, context_id) WHERE state = 2;
"""
"""`state = 2` is `TASK_STATE_WORKING`; SQLite cannot name the enum."""


class SqliteTaskStore(TaskStore):
    """Tasks in a SQLite file, shared by every replica pointed at it."""

    def __init__(
        self, path: str, *, owner_resolver: OwnerResolver = subject_scope
    ) -> None:
        self._path = path
        self._owner_resolver = owner_resolver

    async def setup(self) -> None:
        async with self._connect() as db:
            await db.executescript(SCHEMA)
            await db.commit()

    def _connect(self) -> Any:
        # WAL and a busy timeout, or two processes writing at once meet
        # "database is locked" rather than each other.
        connection = aiosqlite.connect(self._path, timeout=30, isolation_level=None)
        connection.daemon = True
        return _Prepared(connection)

    def _owner(self, context: ServerCallContext) -> str:
        return self._owner_resolver(context) or ""

    async def save(self, task: Task, context: ServerCallContext) -> None:
        async with self._connect() as db:
            await self._write(db, task, self._owner(context))

    async def _write(self, db: Any, task: Task, owner: str) -> None:
        await db.execute(
            """
            INSERT INTO tasks (owner, id, context_id, state, version, task)
            VALUES (?, ?, ?, ?, 1, ?)
            ON CONFLICT (owner, id) DO UPDATE SET
                context_id = excluded.context_id,
                state = excluded.state,
                version = tasks.version + 1,
                task = excluded.task
            """,
            (
                owner,
                task.id,
                task.context_id,
                int(task.status.state),
                json_format.MessageToJson(task),
            ),
        )

    async def get(self, task_id: str, context: ServerCallContext) -> Task | None:
        async with self._connect() as db:
            row = await (
                await db.execute(
                    "SELECT task FROM tasks WHERE owner = ? AND id = ?",
                    (self._owner(context), task_id),
                )
            ).fetchone()
        return _to_task(row[0]) if row else None

    async def get_versioned(
        self, task_id: str, context: ServerCallContext
    ) -> tuple[Task | None, Any]:
        async with self._connect() as db:
            row = await (
                await db.execute(
                    "SELECT task, version FROM tasks WHERE owner = ? AND id = ?",
                    (self._owner(context), task_id),
                )
            ).fetchone()
        return (_to_task(row[0]), row[1]) if row else (None, None)

    async def save_if_unchanged(
        self, task: Task, version: Any, context: ServerCallContext
    ) -> bool:
        owner = self._owner(context)
        async with self._connect() as db:
            await db.execute("BEGIN IMMEDIATE")
            try:
                row = await (
                    await db.execute(
                        "SELECT version FROM tasks WHERE owner = ? AND id = ?",
                        (owner, task.id),
                    )
                ).fetchone()
                if (row[0] if row else None) != version:
                    await db.execute("ROLLBACK")
                    return False
                await self._write(db, task, owner)
            except aiosqlite.IntegrityError:
                # The unique index: another replica is already working this
                # conversation. Indistinguishable, to a caller, from losing the
                # version race — both mean re-read and decide again.
                await db.execute("ROLLBACK")
                return False
            await db.execute("COMMIT")
        return True

    async def delete(self, task_id: str, context: ServerCallContext) -> None:
        async with self._connect() as db:
            await db.execute(
                "DELETE FROM tasks WHERE owner = ? AND id = ?",
                (self._owner(context), task_id),
            )

    async def list(
        self, params: ListTasksRequest, context: ServerCallContext
    ) -> ListTasksResponse:
        query = "SELECT task FROM tasks WHERE owner = ?"
        arguments: list[Any] = [self._owner(context)]
        if params.context_id:
            query += " AND context_id = ?"
            arguments.append(params.context_id)
        if params.status != TaskState.TASK_STATE_UNSPECIFIED:
            query += " AND state = ?"
            arguments.append(int(params.status))
        async with self._connect() as db:
            rows = await (await db.execute(query, arguments)).fetchall()
        tasks = [_to_task(row[0]) for row in rows]
        return ListTasksResponse(tasks=tasks, total_size=len(tasks))


class _Prepared:
    """`aiosqlite.connect` with the pragmas every connection needs."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    async def __aenter__(self) -> Any:
        db = await self._connection
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA busy_timeout=30000")
        return db

    async def __aexit__(self, *exc: Any) -> None:
        await self._connection.close()


def _to_task(payload: str) -> Task:
    task = Task()
    json_format.Parse(payload, task)
    return task
