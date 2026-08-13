"""Fixtures: an authenticating server, and official-SDK clients that reach it.

Every end-to-end check here is driven by the unmodified `a2a-sdk` client
against an app built by `create_a2a_app`. Nothing reaches into the executor:
a criterion that can only be checked from inside is not a criterion a caller
can rely on.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from a2a.auth.user import User
from a2a.client import ClientConfig, ClientFactory
from a2a.client.card_resolver import A2ACardResolver
from a2a.extensions.common import HTTP_EXTENSION_HEADER, get_requested_extensions
from a2a.server.context import ServerCallContext
from a2a.server.routes import ServerCallContextBuilder
from a2a.types.a2a_pb2 import (
    AgentSkill,
    APIKeySecurityScheme,
    ListTasksRequest,
    Message,
    Part,
    Role,
    SecurityScheme,
    SendMessageRequest,
    TaskState,
)
from google.protobuf import json_format, struct_pb2
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.a2a import BaseStoreTaskStore, StateAdapter, create_a2a_app
from langgraph.a2a.interrupts import (
    INTERRUPT_ID_KEY,
    KIND_CREDENTIAL_REQUEST,
    KIND_INTERRUPT,
    KIND_INTERRUPT_RESPONSE,
    METADATA_KIND_KEY,
    VALUE_KEY,
)
from tests.agent import build_graph

BASE_URL = "http://a2a.test"
RPC_URL = f"{BASE_URL}/a2a"

SKILLS = [
    AgentSkill(
        id="echo",
        name="Echo",
        description="Repeats what it is told.",
        tags=["demo"],
        examples=["say hello"],
    ),
    AgentSkill(
        id="greet",
        name="Greet by name",
        description="Asks for a name, then greets. Exercises input-required.",
        tags=["demo", "hitl"],
    ),
]

SECURITY_SCHEMES = {
    "bearer": SecurityScheme(
        api_key_security_scheme=APIKeySecurityScheme(
            name="Authorization", location="header"
        )
    )
}


class _BearerUser(User):
    """The token is the user name. Enough to test ownership, and nothing more."""

    def __init__(self, token: str) -> None:
        self._token = token

    @property
    def is_authenticated(self) -> bool:
        return bool(self._token)

    @property
    def user_name(self) -> str:
        return self._token


class BearerContextBuilder(ServerCallContextBuilder):
    """Reads `Authorization: Bearer <name>` and calls that the caller."""

    def build(self, request: Any) -> ServerCallContext:
        header = request.headers.get("authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else ""
        return ServerCallContext(
            user=_BearerUser(token),
            state={"headers": dict(request.headers)},
            requested_extensions=get_requested_extensions(
                request.headers.getlist(HTTP_EXTENSION_HEADER)
            ),
        )


class ConditionalStore(BaseStoreTaskStore):
    """What a deployment brings when it runs more than one replica.

    `BaseStore` has neither a version nor a constraint, so the package ships no
    store that can do this and the seam is a protocol rather than an
    implementation. This is the smallest thing that satisfies it: a version
    counter, and the rule that only one task in a context may be `WORKING`. It
    is correct in one process, which is all this file needs — the version lives
    in a dict and the atomicity comes from an `asyncio.Lock`, where a real store
    would use a row version and a unique index.
    """

    def __init__(self, store: Any) -> None:
        super().__init__(store)
        self._lock = asyncio.Lock()
        self._versions: dict[str, int] = {}

    async def save(self, task: Any, context: ServerCallContext) -> None:
        self._versions[task.id] = self._versions.get(task.id, 0) + 1
        await super().save(task, context)

    async def get_versioned(
        self, task_id: str, context: ServerCallContext
    ) -> tuple[Any, Any]:
        return await self.get(task_id, context), self._versions.get(task_id)

    async def save_if_unchanged(
        self, task: Any, version: Any, context: ServerCallContext
    ) -> bool:
        async with self._lock:
            if self._versions.get(task.id) != version:
                return False
            if (
                task.status.state == TaskState.TASK_STATE_WORKING
                and await self._working(task, context)
            ):
                return False
            await self.save(task, context)
            return True

    async def _working(self, task: Any, context: ServerCallContext) -> bool:
        """Is some other task already working in this task's context?"""
        page = await self.list(
            ListTasksRequest(
                context_id=task.context_id, status=TaskState.TASK_STATE_WORKING
            ),
            context,
        )
        return any(other.id != task.id for other in page.tasks)


def build_app(**overrides: Any) -> Any:
    """The app under test, with `create_a2a_app`'s production guards intact."""
    kwargs: dict[str, Any] = {
        "name": "toy-langgraph-agent",
        "description": "A LangGraph graph served over A2A 1.0.",
        "version": "0.1.0",
        "url": RPC_URL,
        "skills": SKILLS,
        "security_schemes": SECURITY_SCHEMES,
        "context_builder": BearerContextBuilder(),
        "state": StateAdapter(
            input_data_key="order",
            output_data=lambda state: (state or {}).get("receipt"),
        ),
    }
    graph = overrides.pop("graph", None)
    kwargs.update(overrides)
    return create_a2a_app(graph or build_graph(checkpointer=InMemorySaver()), **kwargs)


def http_client(app: Any, *, token: str = "alice", **kwargs: Any) -> httpx.AsyncClient:
    """An httpx client wired straight to the ASGI app, carrying a bearer token."""
    headers = {"authorization": f"Bearer {token}", **kwargs.pop("headers", {})}
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url=BASE_URL,
        headers=headers,
        timeout=30,
        **kwargs,
    )


async def make_client(http: httpx.AsyncClient, *, streaming: bool = False) -> Any:
    """The official client, configured from the card the server actually serves."""
    card = await A2ACardResolver(http, BASE_URL).get_agent_card()
    return ClientFactory(ClientConfig(httpx_client=http, streaming=streaming)).create(
        card
    )


@pytest.fixture
async def app() -> Any:
    return build_app()


@pytest.fixture
async def alice(app: Any) -> AsyncIterator[httpx.AsyncClient]:
    async with http_client(app, token="alice") as client:
        yield client


@pytest.fixture
async def bob(app: Any) -> AsyncIterator[httpx.AsyncClient]:
    async with http_client(app, token="bob") as client:
        yield client


@pytest.fixture
async def client(alice: httpx.AsyncClient) -> Any:
    return await make_client(alice)


@pytest.fixture
async def streaming_client(alice: httpx.AsyncClient) -> Any:
    return await make_client(alice, streaming=True)


# -- message helpers ---------------------------------------------------------

_counter = iter(range(1, 1_000_000))


def text_message(text: str, **kwargs: Any) -> SendMessageRequest:
    message = Message(message_id=f"m-{next(_counter)}", role=Role.ROLE_USER, **kwargs)
    message.parts.append(Part(text=text))
    return SendMessageRequest(message=message)


def data_message(
    payload: dict, *, metadata: dict | None = None, text: str = "", **kwargs: Any
) -> SendMessageRequest:
    message = Message(message_id=f"d-{next(_counter)}", role=Role.ROLE_USER, **kwargs)
    if text:
        message.parts.append(Part(text=text))
    part = Part(data=_value(payload))
    if metadata:
        json_format.ParseDict(metadata, part.metadata)
    message.parts.append(part)
    return SendMessageRequest(message=message)


def answer(interrupt_id: str, value: Any, **kwargs: Any) -> SendMessageRequest:
    """The answer to one pause, in the encoding the pause was reported in."""
    return data_message(
        {INTERRUPT_ID_KEY: interrupt_id, VALUE_KEY: value},
        metadata={METADATA_KIND_KEY: KIND_INTERRUPT_RESPONSE},
        **kwargs,
    )


def with_subject(request: SendMessageRequest, subject: str) -> SendMessageRequest:
    """Declare, in request metadata, which end user the caller is acting for."""
    json_format.ParseDict({"end_user": subject}, request.metadata)
    return request


def _value(payload: Any) -> struct_pb2.Value:
    value = struct_pb2.Value()
    json_format.ParseDict(payload, value)
    return value


# -- response helpers --------------------------------------------------------


async def collect(client: Any, request: SendMessageRequest) -> list[tuple[str, Any]]:
    events = []
    async for event in client.send_message(request):
        events.append((event.WhichOneof("payload"), event))
    return events


def last_task(events: list[tuple[str, Any]]) -> Any:
    for kind, event in reversed(events):
        if kind == "task":
            return event.task
    return None


async def send(client: Any, request: SendMessageRequest) -> Any:
    return last_task(await collect(client, request))


def artifact_text(task: Any) -> list[str]:
    return [
        part.text
        for artifact in task.artifacts
        for part in artifact.parts
        if part.HasField("text")
    ]


def artifact_data(task: Any) -> list[Any]:
    return [
        json_format.MessageToDict(part.data)
        for artifact in task.artifacts
        for part in artifact.parts
        if part.WhichOneof("content") == "data"
    ]


def status_text(task: Any) -> str:
    return "".join(
        part.text for part in task.status.message.parts if part.HasField("text")
    )


def pauses(task: Any) -> list[dict[str, Any]]:
    """The pause payloads carried by a paused task's status message."""
    found = []
    for part in task.status.message.parts:
        if part.WhichOneof("content") != "data":
            continue
        metadata = (
            json_format.MessageToDict(part.metadata)
            if part.HasField("metadata")
            else {}
        )
        kind = metadata.get(METADATA_KIND_KEY)
        if kind in (KIND_INTERRUPT, KIND_CREDENTIAL_REQUEST):
            found.append({**json_format.MessageToDict(part.data), "kind": kind})
    return found


def find_pause(task: Any, question: str) -> dict[str, Any]:
    for pause in pauses(task):
        if question in str(pause.get("payload")):
            return pause
    raise AssertionError(f"no pause matching {question!r} in {pauses(task)}")


__all__: list[str] = [
    "BASE_URL",
    "RPC_URL",
    "SECURITY_SCHEMES",
    "SKILLS",
    "BearerContextBuilder",
    "ConditionalStore",
    "artifact_data",
    "artifact_text",
    "build_app",
    "collect",
    "data_message",
    "answer",
    "find_pause",
    "http_client",
    "last_task",
    "make_client",
    "pauses",
    "send",
    "status_text",
    "text_message",
    "with_subject",
]
