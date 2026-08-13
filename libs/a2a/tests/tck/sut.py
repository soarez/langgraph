"""The system under test for the official A2A TCK.

The TCK is not a black-box probe: it drives an agent through scenarios keyed by
a `messageId` prefix (`scenarios/*.feature` in `a2aproject/a2a-tck`), so a SUT
has to cooperate with that signal. Here the cooperation is a graph — the prefix
reaches it through `config_from_context`, which is the same mechanism a real
deployment uses to pass caller identity into a run. Everything under the graph
is the shipped server: the executor, the card, the task store, `create_a2a_app`
and its guards.

Run it and point the TCK at it:

```bash
python -m tests.tck.sut &
./run_tck.py --sut-host http://127.0.0.1:9999
```

`tests/tck/run.py` does both against a pinned checkout, and `tests/test_tck.py`
runs that from pytest when `A2A_TCK_PATH` is set.

**What the gate does not reach.** The TCK authenticates nothing, so this server
runs `single_tenant` with in-memory stores — which switches off the
authenticating context builder, the card's security schemes, subject
resolution, context ownership and the durable-pairing check, all at once. A
green conformance run says the protocol layer is right; it says nothing about
those. They are what the adversarial acceptance suite covers with the guards
switched on.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from a2a.server.tasks import InMemoryPushNotificationConfigStore
from a2a.types.a2a_pb2 import AgentSkill, Part
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from typing_extensions import TypedDict

from langgraph.a2a import StateAdapter, create_a2a_app, subject_scope
from langgraph.a2a.executor import TaskRejected
from langgraph.a2a.parts import data_part, text_part

HOST = "127.0.0.1"
PORT = 9999
RPC_PATH = "/"
URL = f"http://{HOST}:{PORT}{RPC_PATH}"
"""The TCK posts JSON-RPC at the host root, so that is where this SUT serves it."""

RESUBSCRIBE_SECONDS = 4.0
"""`2 x TCK_STREAMING_TIMEOUT` at the TCK's default of 2 seconds."""

MESSAGE_ID_KEY = "tck_message_id"


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    parts: Any


async def scenario(state: State, config: RunnableConfig) -> dict[str, Any]:
    """One node, branching on the prefix the TCK signalled with."""
    message_id = (config.get("configurable") or {}).get(MESSAGE_ID_KEY, "")

    if message_id.startswith("test-resubscribe-message-id"):
        await asyncio.sleep(RESUBSCRIBE_SECONDS)
        return {"messages": [AIMessage(content="done")]}

    while message_id.startswith("tck-input-required"):
        # One pause per turn: on a resume the first call returns the answer,
        # and the next iteration parks the task again — which is what the
        # scenario asks for, a task that stays at input_required until a
        # different prefix arrives.
        interrupt({"question": "More input required"})

    if message_id.startswith("tck-reject-task"):
        raise TaskRejected("rejected")

    if message_id.startswith(("tck-artifact-file-url", "tck-stream-artifact-file-url")):
        return _parts({"url": "https://example.com/output.txt"})

    if message_id.startswith(("tck-artifact-file", "tck-stream-artifact-file")):
        return _parts({"raw": "tck"})

    if message_id.startswith("tck-artifact-data"):
        return _parts({"data": {"key": "value", "count": 42}})

    if message_id.startswith("tck-artifact-text"):
        return _parts({"text": "Generated text content"})

    if message_id.startswith("tck-stream-artifact-chunked"):
        return _parts({"text": "chunk-1 "}, {"text": "chunk-2"})

    if message_id.startswith("tck-stream-artifact-text"):
        return _parts({"text": "Streamed text content"})

    if message_id.startswith("tck-stream-ordering"):
        return _parts({"text": "Ordered output"})

    if message_id.startswith("tck-stream-001"):
        return _parts({"text": "Stream hello from TCK"})

    if message_id.startswith("tck-stream-003"):
        return _parts({"text": "Stream task lifecycle"})

    return {"messages": [AIMessage(content="Hello from TCK")]}


def _parts(*described: dict[str, Any]) -> dict[str, Any]:
    """Describe the artifact in plain data.

    A protobuf `Part` is not msgpack-serialisable, so it cannot be written to
    graph state at all. `output_parts` builds the parts from a description that
    can.
    """
    return {"messages": [AIMessage(content="")], "parts": list(described)}


def build_parts(state: State) -> list[Part] | None:
    """Turn those descriptions into A2A parts, outside the checkpointer."""
    described = (state or {}).get("parts")
    if not described:
        return None
    parts: list[Part] = []
    for item in described:
        if "text" in item:
            parts.append(text_part(item["text"]))
        elif "data" in item:
            parts.append(data_part(item["data"]))
        elif "url" in item:
            parts.append(
                Part(url=item["url"], media_type="text/plain", filename="output.txt")
            )
        else:
            parts.append(
                Part(
                    raw=item["raw"].encode(),
                    media_type="text/plain",
                    filename="output.txt",
                )
            )
    return parts


def build_graph() -> Any:
    builder = StateGraph(State)
    builder.add_node("scenario", scenario)
    builder.add_edge(START, "scenario")
    builder.add_edge("scenario", END)
    return builder.compile(checkpointer=InMemorySaver())


def build() -> Any:
    return create_a2a_app(
        build_graph(),
        name="langgraph-a2a-tck-sut",
        description="A LangGraph graph served over A2A 1.0, under conformance test.",
        version="0.1.0",
        url=URL,
        skills=[
            AgentSkill(
                id="tck",
                name="TCK scenarios",
                description="Drives the conformance scenarios.",
                tags=["tck"],
            )
        ],
        single_tenant=True,
        rpc_path=RPC_PATH,
        push_config_store=InMemoryPushNotificationConfigStore(
            owner_resolver=subject_scope
        ),
        # The suite's webhook receiver runs on loopback, which the destination
        # policy refuses by default. A conformance run is exactly the deployment
        # whose receivers are genuinely its own.
        allow_private_webhooks=True,
        config_from_context=lambda context: {
            "configurable": {
                MESSAGE_ID_KEY: context.message.message_id if context.message else ""
            }
        },
        state=StateAdapter(output_parts=build_parts),
    )


app = build()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
