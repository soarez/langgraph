"""A toy graph exercising every mapping the package claims to implement.

No credentials and no network: token streaming comes from a fake chat model, so
the whole acceptance suite runs offline. The graph branches on keywords in the
incoming text, which keeps one graph able to stand in for a dozen agents.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from pydantic import BaseModel
from typing_extensions import TypedDict

from langgraph.a2a import CredentialRequest

LONG_ANSWER = " ".join(f"token{n:03d}" for n in range(200))
"""Long enough that a buffered stream has to emit more than one chunk."""

SIDE_EFFECTS: list[str] = []
"""Appended to before an `interrupt()`, so a test can watch a node re-execute."""

TOOL_RESULT = "account 8812 holds 40318 EUR"
"""A tool's output, distinctive enough that a test can look for it on the wire."""


class Approval(BaseModel):
    """A pydantic interrupt payload, to prove one survives the wire."""

    question: str
    amount: float
    tags: list[str] = []


def _merge(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """So two parallel branches can both write a receipt without conflicting."""
    return {**(left or {}), **(right or {})}


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    receipt: Annotated[dict[str, Any], _merge]
    order: Any


def _text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return content or ""


def _last(state: State) -> str:
    messages = state.get("messages") or []
    return _text(messages[-1]).lower() if messages else ""


def _route(state: State) -> str | list[str]:
    last = _last(state)
    if "parallel" in last:
        return ["ask_north", "ask_south"]
    return "respond"


async def respond(state: State) -> dict[str, Any]:
    last = _last(state)

    if "what did i say" in last:
        # Reads the conversation out of the thread, so a test can see whether
        # the thread actually spans the conversation.
        earlier = [_text(m) for m in (state.get("messages") or [])[:-1]]
        return {
            "messages": [AIMessage(content="earlier: " + " | ".join(earlier))],
            "receipt": {"turns": len(earlier)},
        }

    if "pydantic" in last:
        answer = interrupt(
            Approval(question="Approve the transfer?", amount=42.5, tags=["risk"])
        )
        return {
            "messages": [AIMessage(content=f"approved: {answer}")],
            "receipt": {"approved": answer},
        }

    if "credential" in last:
        token = interrupt(
            CredentialRequest(scheme="github", description="repo scope, read only")
        )
        return {"messages": [AIMessage(content=f"used token {token}")]}

    if "wizard" in last:
        # Pauses, is answered, pauses again inside one node. The re-raised
        # interrupt keeps its id and the graph task carries a result from the
        # answer already given, which is what makes the pending set delicate.
        first = interrupt({"question": "Step one?"})
        second = interrupt({"question": "Step two?"})
        return {"messages": [AIMessage(content=f"wizard {first}/{second}")]}

    if "twice" in last:
        SIDE_EFFECTS.append("ran")
        answer = interrupt({"question": "Confirm?"})
        return {"messages": [AIMessage(content=f"confirmed {answer}")]}

    if "ask" in last:
        answer = interrupt({"question": "What is your name?"})
        return {
            "messages": [AIMessage(content=f"hello {answer}")],
            "receipt": {"greeted": answer},
        }

    if "tool" in last:
        # A tool ran and its output is in the conversation. What a peer sees of
        # it is the deployment's decision, so the package publishes none of it
        # unless an adapter asks for it.
        call = AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "lookup", "args": {"q": "balance"}}],
        )
        result = ToolMessage(content=TOOL_RESULT, tool_call_id="c1", name="lookup")
        return {
            "messages": [call, result, AIMessage(content="your balance is fine")],
            "receipt": {"tool": "lookup"},
        }

    if "two turns" in last:
        # Speaks, then answers — the shape a tool-calling graph has. The stream
        # carries both turns; the final state's last assistant turn is only the
        # second, so this is where a streamed and a non-streamed run could
        # disagree about what the answer was.
        preamble = GenericFakeChatModel(messages=iter([AIMessage(content="looking")]))
        await preamble.ainvoke(state["messages"])
        answer = GenericFakeChatModel(messages=iter([AIMessage(content="found it")]))
        reply = await answer.ainvoke(state["messages"])
        return {"messages": [reply], "receipt": {"turns": 2}}

    if "stream" in last:
        model = GenericFakeChatModel(messages=iter([AIMessage(content=LONG_ANSWER)]))
        reply = await model.ainvoke(state["messages"])
        return {"messages": [reply], "receipt": {"streamed": True}}

    if "sleep" in last:
        await asyncio.sleep(30)
        return {"messages": [AIMessage(content="awake")]}

    if "fail" in last:
        raise RuntimeError("credit card 4111-1111-1111-1111 rejected by upstream")

    if "order" in last:
        return {
            "messages": [AIMessage(content=f"order: {state.get('order')!r}")],
            "receipt": {"order": state.get("order")},
        }

    return {
        "messages": [AIMessage(content=f"echo: {_text(state['messages'][-1])}")],
        "receipt": {"echoed": True},
    }


async def ask_north(_state: State) -> dict[str, Any]:
    answer = interrupt({"question": "North?"})
    return {
        "messages": [AIMessage(content=f"north={answer}")],
        "receipt": {"north": answer},
    }


async def ask_south(_state: State) -> dict[str, Any]:
    answer = interrupt({"question": "South?"})
    return {
        "messages": [AIMessage(content=f"south={answer}")],
        "receipt": {"south": answer},
    }


def build_graph(checkpointer: Any = None, store: Any = None) -> Any:
    builder = StateGraph(State)
    builder.add_node("respond", respond)
    builder.add_node("ask_north", ask_north)
    builder.add_node("ask_south", ask_south)
    builder.add_conditional_edges(START, _route, ["respond", "ask_north", "ask_south"])
    builder.add_edge("respond", END)
    builder.add_edge("ask_north", END)
    builder.add_edge("ask_south", END)
    return builder.compile(checkpointer=checkpointer, store=store)
