"""The card is honest: nothing advertised that the configuration cannot honour.

A card is read by a stranger deciding whether to talk to this agent at all, on
an unauthenticated URL, and nothing in the protocol checks it against the
server. Everything here is a way the card could lie while every other test
passes: promising a pause that survives a restart from a graph that cannot
suspend, listing skills the deployment never declared, or republishing a tool
docstring written for a model.
"""

from __future__ import annotations

from typing import Any

import pytest
from a2a.client.card_resolver import A2ACardResolver
from a2a.types.a2a_pb2 import AgentExtension, AgentSkill
from google.protobuf import json_format
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from langgraph.a2a import A2AConfigurationError, StateAdapter, create_a2a_app
from langgraph.a2a.card import DERIVED_SKILL_DESCRIPTION
from langgraph.a2a.executor import DEFAULT_PAUSE_DEADLINE
from langgraph.a2a.extension import CONVERSATION_URI, PAUSE_DEADLINE_PARAM
from langgraph.a2a.extension import URI as DURABLE_INTERRUPT_URI
from tests.agent import build_graph
from tests.conftest import (
    BASE_URL,
    RPC_URL,
    SECURITY_SCHEMES,
    BearerContextBuilder,
    build_app,
    http_client,
)

SECRET_DESCRIPTION = "Query the customer PII warehouse. Internal only; never mention."


@tool
def internal_db_query(sql: str) -> str:
    """Query the customer PII warehouse. Internal only; never mention."""
    return "rows"


def tool_graph(checkpointer: Any = None) -> Any:
    return (
        StateGraph(MessagesState)
        .add_node("tools", ToolNode([internal_db_query]))
        .add_edge(START, "tools")
        .compile(checkpointer=checkpointer)
    )


async def card_of(app: Any) -> Any:
    async with http_client(app, token="alice") as http:
        return await A2ACardResolver(http, BASE_URL).get_agent_card()


async def test_no_durable_interrupt_extension_without_a_checkpointer() -> None:
    """The one sentence a counterparty reads must not promise durability."""
    app = build_app(graph=build_graph(checkpointer=None))

    card = await card_of(app)

    uris = {e.uri for e in card.capabilities.extensions}
    assert DURABLE_INTERRUPT_URI not in uris
    # The conversation model is still true of a graph that cannot suspend.
    assert CONVERSATION_URI in uris


async def test_the_extension_is_advertised_when_the_graph_can_honour_it(
    app: Any,
) -> None:
    card = await card_of(app)

    extensions = {e.uri: e for e in card.capabilities.extensions}
    assert DURABLE_INTERRUPT_URI in extensions
    assert extensions[DURABLE_INTERRUPT_URI].required is False
    # And it states the cost, not only the guarantee.
    assert "re-executes" in extensions[DURABLE_INTERRUPT_URI].description


def test_requiring_the_extension_without_a_checkpointer_is_refused() -> None:
    with pytest.raises(A2AConfigurationError, match="no checkpointer"):
        build_app(graph=build_graph(checkpointer=None), durable_interrupt_required=True)


async def test_no_skill_is_published_that_was_not_declared() -> None:
    """Derivation is opt-in. A tool inventory is not a description of an agent."""
    app = create_a2a_app(
        tool_graph(InMemorySaver()),
        name="n",
        description="d",
        version="1",
        url=RPC_URL,
        security_schemes=SECURITY_SCHEMES,
        context_builder=BearerContextBuilder(),
    )

    card = await card_of(app)

    assert list(card.skills) == []


async def test_a_derived_skill_does_not_republish_the_tool_description() -> None:
    app = create_a2a_app(
        tool_graph(InMemorySaver()),
        name="n",
        description="d",
        version="1",
        url=RPC_URL,
        security_schemes=SECURITY_SCHEMES,
        context_builder=BearerContextBuilder(),
        derive_skills=True,
    )

    card = await card_of(app)

    assert [s.id for s in card.skills] == ["internal_db_query"]
    assert card.skills[0].description == DERIVED_SKILL_DESCRIPTION
    assert SECRET_DESCRIPTION not in str(card)


async def test_declared_skills_are_served_verbatim(app: Any) -> None:
    card = await card_of(app)

    assert [s.id for s in card.skills] == ["echo", "greet"]
    assert card.skills[0].description == "Repeats what it is told."


async def test_declared_input_modes_match_what_the_adapter_accepts() -> None:
    """`application/json` on the card means a `data` part will be accepted."""
    refusing = build_app(state=None)
    accepting = build_app()

    refusing_card = await card_of(refusing)
    accepting_card = await card_of(accepting)

    assert "application/json" not in refusing_card.default_input_modes
    assert "text/plain" in refusing_card.default_input_modes
    assert "application/json" in accepting_card.default_input_modes


def test_a_state_key_the_graph_does_not_have_is_refused() -> None:
    with pytest.raises(A2AConfigurationError, match="does not declare"):
        build_app(state=StateAdapter(message_key="chat"))


def test_a_declared_skill_needs_no_graph_structure(app: Any) -> None:
    assert [s.id for s in app.state.a2a.card.skills] == ["echo", "greet"]


def test_the_card_can_declare_skills_and_derive_them_together() -> None:
    declared = [AgentSkill(id="research", name="Research", description="Reads.")]
    server = build_app(
        graph=tool_graph(InMemorySaver()),
        skills=declared,
        derive_skills=True,
        state=StateAdapter(),
    ).state.a2a

    assert [s.id for s in server.card.skills] == ["research", "internal_db_query"]


async def test_the_deadline_is_published_as_a_number_not_prose(app: Any) -> None:
    """`AgentExtension.params` is a Struct and exists for this.

    A duration buried in a description is not something a caller can act on.
    """
    card = await card_of(app)

    conversation = next(
        e for e in card.capabilities.extensions if e.uri == CONVERSATION_URI
    )
    params = json_format.MessageToDict(conversation.params)
    assert params[PAUSE_DEADLINE_PARAM] == DEFAULT_PAUSE_DEADLINE


async def test_a_server_without_a_deadline_publishes_none() -> None:
    app = build_app(pause_deadline=None)

    card = await card_of(app)

    conversation = next(
        e for e in card.capabilities.extensions if e.uri == CONVERSATION_URI
    )
    assert not json_format.MessageToDict(conversation.params)


async def test_declared_schemes_are_published_as_required_not_merely_available(
    app: Any,
) -> None:
    """A scheme with no requirement reads as "authentication is optional here".

    The server refuses an unauthenticated caller, so a card that does not say
    so sends a conformant client to a rejection it could have avoided. Any one
    of the declared schemes satisfies the default; a deployment that means two
    together, or a scope list, states its own.
    """
    card = await card_of(app)

    required = [set(r.schemes.keys()) for r in card.security_requirements]
    assert required == [{name} for name in SECURITY_SCHEMES]


async def test_a_single_tenant_server_requires_nothing() -> None:
    """Nothing is declared, so nothing is required, and the card says as much."""
    app = build_app(single_tenant=True, security_schemes=None, context_builder=None)

    card = await card_of(app)

    assert not card.security_schemes
    assert not card.security_requirements


async def test_a_declared_extension_is_published_whole() -> None:
    """A deployment's own promise about its own agent, carried as given.

    `params` is where an extension puts anything a caller can act on — the
    package's own conversation model publishes the deadline there — so dropping
    it would advertise a card that looks right and says nothing.
    """
    declared = AgentExtension(
        uri="https://example.com/ext/receipts/v1",
        description="Every completed task is receipted to the payer.",
        required=True,
    )
    json_format.ParseDict({"currency": "EUR"}, declared.params)

    card = await card_of(build_app(extensions=[declared]))

    published = {e.uri: e for e in card.capabilities.extensions}
    assert DURABLE_INTERRUPT_URI in published, "the package's own are still there"
    mine = published["https://example.com/ext/receipts/v1"]
    assert mine.required is True
    assert json_format.MessageToDict(mine.params) == {"currency": "EUR"}
    assert mine.description == declared.description
