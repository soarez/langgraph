"""The example, run.

An example nobody executes is a claim about the past. This drives
`examples/approval_agent.py` exactly as its `__main__` does, so the code a
reader copies is the code CI proved works this build.
"""

from __future__ import annotations

from typing import Any

import httpx
from a2a.client import ClientConfig, ClientFactory
from a2a.client.card_resolver import A2ACardResolver
from a2a.types.a2a_pb2 import TaskState

from examples.approval_agent import (
    BASE_URL,
    _answer,
    _pauses,
    _send,
    _text,
    build_app,
    converse,
)
from langgraph.a2a.interrupts import INTERRUPT_ID_KEY


def client_for(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url=BASE_URL,
        headers={"authorization": "Bearer alice"},
    )


async def test_the_example_runs() -> None:
    """The whole narrated exchange, start to finish."""
    async with client_for(build_app()) as http:
        await converse(http)


async def test_the_example_pauses_and_resumes_the_same_task() -> None:
    """What the example is for: the pause carries an id, and answering it
    continues the same task rather than starting another."""
    async with client_for(build_app()) as http:
        card = await A2ACardResolver(http, BASE_URL).get_agent_card()
        client = ClientFactory(ClientConfig(httpx_client=http, streaming=False)).create(
            card
        )

        asked = await _send(client, _text("42.00"))
        assert asked.status.state == TaskState.TASK_STATE_INPUT_REQUIRED
        question = _pauses(asked)[0]
        assert "Approve a refund of 42.00?" in str(question)

        done = await _send(
            client,
            _answer(
                question[INTERRUPT_ID_KEY],
                True,
                task_id=asked.id,
                context_id=asked.context_id,
            ),
        )

    assert done.id == asked.id
    assert done.status.state == TaskState.TASK_STATE_COMPLETED
    assert [
        part.text
        for artifact in done.artifacts
        for part in artifact.parts
        if part.HasField("text")
    ] == ["Refunded 42.00."]


async def test_declining_is_answered_too() -> None:
    """The resume value is the caller's, not a formality."""
    async with client_for(build_app()) as http:
        card = await A2ACardResolver(http, BASE_URL).get_agent_card()
        client = ClientFactory(ClientConfig(httpx_client=http, streaming=False)).create(
            card
        )

        asked = await _send(client, _text("99.00"))
        done = await _send(
            client,
            _answer(
                _pauses(asked)[0][INTERRUPT_ID_KEY],
                False,
                task_id=asked.id,
                context_id=asked.context_id,
            ),
        )

    assert [
        part.text
        for artifact in done.artifacts
        for part in artifact.parts
        if part.HasField("text")
    ] == ["Refund declined."]
