"""A LangGraph agent served over A2A, and a client that talks to it.

Both halves, because the interesting part of A2A is the exchange and no
one-sided snippet shows it. The graph pauses mid-run to ask for approval; the
client answers that question and gets the result on the same task.

```bash
uv run python -m examples.approval_agent          # runs the whole exchange
uv run python -m examples.approval_agent --serve  # just the server, on :8080
```

It runs with no credentials and no network: the model is a deterministic stand-in
so the example is executable, and `tests/test_example.py` runs it on every CI
build. An example that is not executed is a claim about the past.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Annotated, Any

import httpx
from a2a.auth.user import User
from a2a.client import ClientConfig, ClientFactory
from a2a.client.card_resolver import A2ACardResolver
from a2a.server.context import ServerCallContext
from a2a.server.routes import ServerCallContextBuilder
from a2a.types.a2a_pb2 import (
    AgentSkill,
    APIKeySecurityScheme,
    Message,
    Part,
    Role,
    SecurityScheme,
    SendMessageRequest,
    TaskState,
)
from google.protobuf import json_format, struct_pb2
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from typing_extensions import TypedDict

from langgraph.a2a import StateAdapter, create_a2a_app
from langgraph.a2a.interrupts import (
    INTERRUPT_ID_KEY,
    KIND_INTERRUPT,
    KIND_INTERRUPT_RESPONSE,
    METADATA_KIND_KEY,
    PAYLOAD_KEY,
    VALUE_KEY,
)

HOST, PORT = "127.0.0.1", 8080
BASE_URL = f"http://{HOST}:{PORT}"
RPC_URL = f"{BASE_URL}/a2a"

# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    receipt: dict[str, Any]


async def refund(state: State) -> dict[str, Any]:
    """Ask a human before doing something expensive, then do it.

    `interrupt()` suspends the graph here. The process can restart between the
    question and the answer; the run resumes at this checkpoint either way.

    Note what is *not* above this line: nothing irreversible. Resuming
    re-executes this node from its start, so anything between the node's entry
    and the `interrupt()` call happens again — issue the refund first and it is
    issued twice.
    """
    amount = state["messages"][-1].content
    approved = interrupt(
        {"question": f"Approve a refund of {amount}?", "amount": amount}
    )

    if not approved:
        return {"messages": [AIMessage(content="Refund declined.")]}
    return {
        "messages": [AIMessage(content=f"Refunded {amount}.")],
        "receipt": {"amount": amount, "approved": True},
    }


def build_graph() -> Any:
    builder = StateGraph(State)
    builder.add_node("refund", refund)
    builder.add_edge(START, "refund")
    builder.add_edge("refund", END)
    # A checkpointer is what makes the pause durable. Without one the agent
    # still serves, but the card stops advertising a resumable pause.
    return builder.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class TokenUser(User):
    def __init__(self, token: str) -> None:
        self._token = token

    @property
    def is_authenticated(self) -> bool:
        return bool(self._token)

    @property
    def user_name(self) -> str:
        return self._token


class BearerContextBuilder(ServerCallContextBuilder):
    """Turns a credential into a caller identity.

    `create_a2a_app` refuses to start without one of these or an explicit
    `single_tenant=True`, because `contextId` is chosen by the caller: with
    nobody authenticated, a second caller presenting the same id is
    indistinguishable from the first.
    """

    def build(self, request: Any) -> ServerCallContext:
        header = request.headers.get("authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else ""
        # `state["headers"]` is not decoration: the SDK reads the negotiated
        # protocol version and the requested extensions out of it.
        return ServerCallContext(
            user=TokenUser(token), state={"headers": dict(request.headers)}
        )


def build_app() -> Any:
    return create_a2a_app(
        build_graph(),
        name="refund-desk",
        description="Issues refunds, with human approval.",
        version="1.0.0",
        url=RPC_URL,
        skills=[
            AgentSkill(
                id="refund",
                name="Issue a refund",
                description="Refunds an order once a human approves the amount.",
                tags=["support"],
                examples=["refund 42.00"],
            )
        ],
        security_schemes={
            "bearer": SecurityScheme(
                api_key_security_scheme=APIKeySecurityScheme(
                    name="Authorization", location="header"
                )
            )
        },
        context_builder=BearerContextBuilder(),
        state=StateAdapter(output_data=lambda state: (state or {}).get("receipt")),
    )


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------


def _text(text: str, **kwargs: Any) -> SendMessageRequest:
    message = Message(message_id=f"m-{id(text)}", role=Role.ROLE_USER, **kwargs)
    message.parts.append(Part(text=text))
    return SendMessageRequest(message=message)


def _answer(interrupt_id: str, value: Any, **kwargs: Any) -> SendMessageRequest:
    """The answer to a pause, addressed to the question it answers.

    A client that does not implement the extension can send plain text instead:
    the server binds it to the single outstanding pause, and refuses it when
    there is more than one, rather than guessing.
    """
    payload = struct_pb2.Value()
    json_format.ParseDict({INTERRUPT_ID_KEY: interrupt_id, VALUE_KEY: value}, payload)
    part = Part(data=payload)
    json_format.ParseDict({METADATA_KIND_KEY: KIND_INTERRUPT_RESPONSE}, part.metadata)

    message = Message(message_id=f"a-{interrupt_id}", role=Role.ROLE_USER, **kwargs)
    message.parts.append(part)
    return SendMessageRequest(message=message)


def _pauses(task: Any) -> list[dict[str, Any]]:
    """The questions a paused task is asking, read off its status message."""
    found = []
    for part in task.status.message.parts:
        if part.WhichOneof("content") != "data":
            continue
        metadata = json_format.MessageToDict(part.metadata)
        if metadata.get(METADATA_KIND_KEY) == KIND_INTERRUPT:
            found.append(json_format.MessageToDict(part.data))
    return found


async def _send(client: Any, request: SendMessageRequest) -> Any:
    task = None
    async for event in client.send_message(request):
        if event.WhichOneof("payload") == "task":
            task = event.task
    return task


async def converse(http: httpx.AsyncClient) -> None:
    """Discover the agent, ask for a refund, approve it, read the result."""
    card = await A2ACardResolver(http, BASE_URL).get_agent_card()
    print(f"agent: {card.name} — {card.description}")
    print(f"skills: {[skill.id for skill in card.skills]}")
    for ext in card.capabilities.extensions:
        print(f"extension: {ext.uri}")

    client = ClientFactory(ClientConfig(httpx_client=http, streaming=False)).create(
        card
    )

    task = await _send(client, _text("42.00"))
    print(f"\ntask {task.id} -> {TaskState.Name(task.status.state)}")

    # The agent is asking something. The question is readable two ways: as text
    # for a human, and as a `data` part carrying the id to answer.
    prompt = "".join(p.text for p in task.status.message.parts if p.HasField("text"))
    question = _pauses(task)[0]
    print(f"it asks: {prompt}")
    print(f"  (interrupt {question[INTERRUPT_ID_KEY]}, {question[PAYLOAD_KEY]})")

    # Answering continues the *same* task. Days could pass here, and a restart
    # of the server would not lose the question.
    done = await _send(
        client,
        _answer(
            question[INTERRUPT_ID_KEY],
            True,
            task_id=task.id,
            context_id=task.context_id,
        ),
    )
    print(f"\ntask {done.id} -> {TaskState.Name(done.status.state)}")
    for artifact in done.artifacts:
        for part in artifact.parts:
            if part.HasField("text"):
                print(f"  {part.text}")
            elif part.WhichOneof("content") == "data":
                print(f"  data: {json_format.MessageToDict(part.data)}")

    # A follow-up is a new task carrying the same contextId, and it remembers.
    followup = await _send(
        client, _text("what did I just ask for?", context_id=task.context_id)
    )
    print(f"\nfollow-up task {followup.id} in the same conversation")


async def main() -> None:
    app = build_app()
    headers = {"authorization": "Bearer alice"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE_URL, headers=headers
    ) as http:
        await converse(http)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="serve, do not drive")
    if parser.parse_args().serve:
        import uvicorn

        uvicorn.run(build_app(), host=HOST, port=PORT, log_level="warning")
    else:
        asyncio.run(main())
