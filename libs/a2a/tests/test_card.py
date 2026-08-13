"""The agent card: 1.0 shape, 0.3 shape, and skills that are never invented."""

from __future__ import annotations

from typing import Any

import pytest
from a2a.client.card_resolver import A2ACardResolver
from a2a.types.a2a_pb2 import AgentSkill
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from langgraph.a2a import build_agent_card, compat_card, derive_skills
from langgraph.a2a.card import DERIVED_SKILL_DESCRIPTION
from langgraph.a2a.extension import URI as EXTENSION_URI
from tests.conftest import BASE_URL, http_client, make_client


@tool
def search(query: str) -> str:
    """Search the web."""
    return "ok"


@tool
def summarise(text: str) -> str:
    """Summarise a passage."""
    return "ok"


def _noop(_state: Any) -> dict[str, Any]:
    return {"messages": []}


def test_card_puts_connection_details_in_supported_interfaces() -> None:
    card = build_agent_card(
        name="a", description="b", version="1", url="https://x/a2a", skills=[]
    )

    assert [i.protocol_version for i in card.supported_interfaces] == ["1.0", "0.3"]
    assert {i.protocol_binding for i in card.supported_interfaces} == {"JSONRPC"}
    assert {i.url for i in card.supported_interfaces} == {"https://x/a2a"}


def test_a_card_without_0_3_serves_only_1_0() -> None:
    card = build_agent_card(
        name="a",
        description="b",
        version="1",
        url="https://x/a2a",
        skills=[],
        serve_v0_3=False,
    )

    assert [i.protocol_version for i in card.supported_interfaces] == ["1.0"]
    with pytest.raises(Exception, match="compatible protocol version"):
        compat_card(card)


def test_compat_card_is_the_0_3_shape() -> None:
    card = build_agent_card(
        name="a",
        description="b",
        version="1",
        url="https://x/a2a",
        skills=[AgentSkill(id="s", name="S", description="d")],
    )

    payload = compat_card(card)

    # 0.3 puts the URL and the protocol version at the top level; 1.0 does not.
    assert payload["url"] == "https://x/a2a"
    assert payload["protocolVersion"] == "0.3"
    assert payload["preferredTransport"] == "JSONRPC"
    assert [skill["id"] for skill in payload["skills"]] == ["s"]


def test_skills_are_derived_from_tools() -> None:
    graph = (
        StateGraph(MessagesState)
        .add_node("tools", ToolNode([search, summarise]))
        .add_edge(START, "tools")
        .compile()
    )

    skills = derive_skills(graph)

    assert [skill.id for skill in skills] == ["search", "summarise"]
    # Never the tool's own description: it was written for a model, and the
    # card is a public document.
    assert skills[0].description == DERIVED_SKILL_DESCRIPTION
    assert "Search the web." not in str(skills)
    assert list(skills[0].tags) == ["tool", "derived"]


def test_subgraph_skills_are_namespaced_by_their_node() -> None:
    child = (
        StateGraph(MessagesState)
        .add_node("tools", ToolNode([search]))
        .add_edge(START, "tools")
        .compile(name="child")
    )
    parent = (
        StateGraph(MessagesState)
        .add_node("research", child)
        .add_node("tools", ToolNode([summarise]))
        .add_edge(START, "research")
        .compile()
    )

    assert sorted(skill.id for skill in derive_skills(parent)) == [
        "research_search",
        "summarise",
    ]


def test_a_graph_with_no_structure_derives_no_skills() -> None:
    """A synthetic skill per agent tells a caller nothing it did not know."""
    graph = (
        StateGraph(MessagesState).add_node("n", _noop).add_edge(START, "n").compile()
    )

    assert derive_skills(graph) == []


async def test_declared_skills_win_over_derived_ones(app: Any) -> None:
    async with http_client(app, token="alice") as http:
        client = await make_client(http)

    assert [skill.id for skill in client._card.skills] == ["echo", "greet"]


async def test_the_card_advertises_the_durable_interrupt_extension(alice: Any) -> None:
    card = await A2ACardResolver(alice, BASE_URL).get_agent_card()

    extensions = {e.uri: e for e in card.capabilities.extensions}
    assert EXTENSION_URI in extensions
    assert extensions[EXTENSION_URI].required is False


async def test_both_card_shapes_are_served(alice: Any) -> None:
    """The 1.0 path carries both shapes; the pre-0.3 path carries only the old one.

    `agent_card_to_dict` merges the legacy top-level fields into the 1.0
    document, so one fetch satisfies both generations. The second path exists
    because a client old enough to need those fields may also be old enough to
    look for the card at `/.well-known/agent.json`, which is
    `PREV_AGENT_CARD_WELL_KNOWN_PATH` in `a2a-sdk` 0.3.
    """
    one_zero = (await alice.get("/.well-known/agent-card.json")).json()
    legacy = (await alice.get("/.well-known/agent.json")).json()

    assert "supportedInterfaces" in one_zero
    assert one_zero["url"].endswith("/a2a")
    assert one_zero["preferredTransport"] == "JSONRPC"

    assert "supportedInterfaces" not in legacy
    assert legacy["url"].endswith("/a2a")
    assert legacy["protocolVersion"] == "0.3"
