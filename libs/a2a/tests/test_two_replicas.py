"""Exclusion across replicas, driven as two processes.

Every other concurrency check in this suite runs two coroutines on one event
loop, where an `asyncio.Lock` is a real lock and the check would pass on a
server that has no cross-process exclusion at all. This is the arrangement that
cannot fail that way: two OS processes, sharing a SQLite checkpointer and a
SQLite task store and nothing else, each told `multi_replica=True`.

The store is `tests/replicas/store.py` — a `ConditionalTaskStore` with a version
column and a unique partial index over `WORKING`. That is the seam the package
declares and does not fill, so filling it is part of what this test proves.
"""

from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from a2a.client import ClientConfig, ClientFactory
from a2a.client.card_resolver import A2ACardResolver
from a2a.types.a2a_pb2 import TaskState

from tests.conftest import artifact_text, send, text_message

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    """A port nothing is listening on, so a leftover server cannot answer for us."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


async def test_two_replicas_do_not_run_two_turns_of_one_conversation(
    tmp_path: Any,
) -> None:
    """Both turns land, and the second one saw the first.

    The graph counts the assistant turns already in the conversation before it
    speaks. Serialised, the answers are "saw 0" and "saw 1" in some order. Run
    at once, both would say "saw 0" — and one of the two writes would be lost.
    """
    async with replicas(tmp_path) as clients:
        first, second = await asyncio.gather(
            send(clients[0], text_message("go", context_id="shared")),
            send(clients[1], text_message("go", context_id="shared")),
        )

    assert first.status.state == TaskState.TASK_STATE_COMPLETED
    assert second.status.state == TaskState.TASK_STATE_COMPLETED
    assert first.id != second.id
    assert sorted(artifact_text(first) + artifact_text(second)) == [
        "saw 0 earlier turns",
        "saw 1 earlier turns",
    ]


class replicas:
    """Two server processes, up and answering, then gone."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._processes: list[subprocess.Popen[bytes]] = []
        self._clients: list[httpx.AsyncClient] = []
        self._ports = (free_port(), free_port())

    async def __aenter__(self) -> list[Any]:
        clients = []
        # Started one at a time: both run `setup()` against the same SQLite
        # files, and two processes creating the same schema at once meet a
        # write lock rather than each other. Nothing about the test needs them
        # to start together — it needs them to *serve* together.
        for port in self._ports:
            self._processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "tests.replicas.serve",
                        str(port),
                        str(self._directory),
                    ],
                    cwd=PACKAGE_ROOT,
                )
            )
            http = httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=30)
            self._clients.append(http)
            card = await _await_card(http, port)
            clients.append(
                ClientFactory(ClientConfig(httpx_client=http, streaming=False)).create(
                    card
                )
            )
        return clients

    async def __aexit__(self, *exc: Any) -> None:
        for http in self._clients:
            await http.aclose()
        for process in self._processes:
            process.terminate()
        for process in self._processes:
            process.wait(timeout=30)


async def _await_card(http: httpx.AsyncClient, port: int, *, tries: int = 100) -> Any:
    """Poll until the replica is serving, so the test times out here and not later."""
    for _ in range(tries):
        try:
            return await A2ACardResolver(
                http, f"http://127.0.0.1:{port}"
            ).get_agent_card()
        except Exception:
            await asyncio.sleep(0.1)
    pytest.fail(f"replica on {port} never served its card")
