"""One replica, as its own OS process.

    python -m tests.replicas.serve <port> <directory>

Two of these against one directory is the arrangement the exclusion claim is
about: one SQLite file for the checkpoints, one for the tasks, two processes
that share nothing else — no lock, no event loop, no memory. Whatever holds
here holds because the store made it hold.

The graph counts the turns already in the conversation before it speaks. Two
turns that overlapped would both count zero; two that were serialised count
zero and one, so the artifact text is the assertion.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated, Any

import aiosqlite
import uvicorn
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from langgraph.a2a import StateAdapter, create_a2a_app
from tests.replicas.store import SqliteTaskStore

TURN_SECONDS = 0.4
"""Long enough that two turns launched together would overlap if nothing stopped them."""


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]


async def respond(state: State) -> dict[str, Any]:
    earlier = [m for m in state.get("messages", []) if getattr(m, "type", None) == "ai"]
    await asyncio.sleep(TURN_SECONDS)
    return {"messages": [AIMessage(content=f"saw {len(earlier)} earlier turns")]}


def build() -> Any:
    builder = StateGraph(State)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    builder.add_edge("respond", END)
    return builder


async def _wal(path: Path) -> None:
    """Two processes on one SQLite file need WAL, or the second one meets a lock.

    Set on the file rather than the connection — it is a property of the
    database, so doing it once before anyone opens the checkpointer is enough.
    """
    connection = await aiosqlite.connect(str(path), timeout=30)
    try:
        await connection.execute("PRAGMA journal_mode=WAL")
        await connection.commit()
    finally:
        await connection.close()


async def serve(port: int, directory: Path) -> None:
    await _wal(directory / "checkpoints.db")
    async with AsyncSqliteSaver.from_conn_string(
        str(directory / "checkpoints.db")
    ) as saver:
        await saver.setup()
        tasks = SqliteTaskStore(str(directory / "tasks.db"))
        await tasks.setup()
        app = create_a2a_app(
            build().compile(checkpointer=saver),
            name="replica",
            description="One of two processes sharing one conversation.",
            version="0.1.0",
            url=f"http://127.0.0.1:{port}/a2a",
            single_tenant=True,
            task_store=tasks,
            multi_replica=True,
            state=StateAdapter(),
            run_timeout=30.0,
        )
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        await uvicorn.Server(config).serve()


if __name__ == "__main__":
    asyncio.run(serve(int(sys.argv[1]), Path(sys.argv[2])))
