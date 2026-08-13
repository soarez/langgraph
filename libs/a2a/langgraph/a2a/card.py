"""The agent card: what this agent is, where to reach it, how to authenticate.

Two things here are easy to get wrong and cost a debugging cycle each:

- In 1.0 the connection details live in `supported_interfaces`. There is no
  top-level `url` and no top-level `protocol_version`; those are 0.3 shapes.
- `protocol_binding` is an enum value, so `JSONRPC` and not `jsonrpc`. A card
  with the wrong casing fails transport selection in the official client before
  any RPC happens, with an error that names neither the card nor the field.
"""

from __future__ import annotations

import logging
from typing import Any

from a2a.compat.v0_3.conversions import to_compat_agent_card
from a2a.types.a2a_pb2 import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentInterface,
    AgentSkill,
    SecurityRequirement,
    SecurityScheme,
    StringList,
)
from a2a.utils.constants import (
    PROTOCOL_VERSION_0_3,
    PROTOCOL_VERSION_1_0,
    TransportProtocol,
)
from langgraph.a2a.extension import CONVERSATION_URI as CONVERSATION_EXTENSION_URI
from langgraph.a2a.extension import URI as EXTENSION_URI
from langgraph.a2a.extension import (
    conversation_model_extension,
    durable_interrupt_extension,
)

logger = logging.getLogger(__name__)

DERIVED_SKILL_DESCRIPTION = (
    "Derived from the agent's graph structure. Declare skills explicitly to "
    "describe them."
)
"""What a derived skill says about itself, in place of a tool's own docstring."""

_MAX_SUBGRAPH_DEPTH = 3


def build_agent_card(
    *,
    name: str,
    description: str,
    version: str,
    url: str,
    skills: list[AgentSkill] | None = None,
    streaming: bool = True,
    push_notifications: bool = False,
    security_schemes: dict[str, SecurityScheme] | None = None,
    security_requirements: list[SecurityRequirement] | None = None,
    extensions: list[AgentExtension] | None = None,
    input_modes: list[str] | None = None,
    output_modes: list[str] | None = None,
    serve_v0_3: bool = True,
) -> AgentCard:
    """An A2A 1.0 agent card describing this server.

    `serve_v0_3` adds a second interface at the same URL declaring the 0.3
    protocol version, which is what lets a 0.3 counterparty select a transport.
    The 0.3-shaped rendering of the card is served alongside it — see
    `compat_card`.

    Declaring schemes without requirements would publish a card saying
    authentication is available and optional, on a server that refuses an
    unauthenticated caller. So the default requirement is any one of the
    declared schemes, and a deployment that means something else — two together,
    a scope list — passes `security_requirements` itself.
    """
    interfaces = [
        AgentInterface(
            url=url,
            protocol_binding=TransportProtocol.JSONRPC.value,
            protocol_version=PROTOCOL_VERSION_1_0,
        )
    ]
    if serve_v0_3:
        interfaces.append(
            AgentInterface(
                url=url,
                protocol_binding=TransportProtocol.JSONRPC.value,
                protocol_version=PROTOCOL_VERSION_0_3,
            )
        )

    card = AgentCard(
        name=name,
        description=description,
        version=version,
        supported_interfaces=interfaces,
        capabilities=AgentCapabilities(
            streaming=streaming,
            push_notifications=push_notifications,
            extensions=extensions or [],
        ),
        default_input_modes=input_modes or ["text/plain", "application/json"],
        default_output_modes=output_modes or ["text/plain", "application/json"],
        skills=skills or [],
        security_requirements=(
            security_requirements
            if security_requirements is not None
            else _any_of(security_schemes)
        ),
    )
    for scheme_name, scheme in (security_schemes or {}).items():
        card.security_schemes[scheme_name].CopyFrom(scheme)
    return card


def _any_of(
    security_schemes: dict[str, SecurityScheme] | None,
) -> list[SecurityRequirement]:
    """One requirement per scheme: satisfying any single one is enough."""
    requirements = []
    for scheme_name in security_schemes or {}:
        requirement = SecurityRequirement()
        requirement.schemes[scheme_name].CopyFrom(StringList())
        requirements.append(requirement)
    return requirements


def compat_card(card: AgentCard) -> dict[str, Any]:
    """The same card in the 0.3 shape, as JSON.

    Raises:
        VersionNotSupportedError: the card declares no 0.3 interface, so there
            is nothing a 0.3 client could select. Build it with
            `serve_v0_3=True`.
    """
    return to_compat_agent_card(card).model_dump(
        mode="json", by_alias=True, exclude_none=True
    )


def derive_skills(graph: Any) -> list[AgentSkill]:
    """Skills a compiled graph supplies directly: its tools and its subgraphs.

    Off by default where it is wired in, and worth understanding before turning
    it on. A skill is not an endpoint — A2A has no skill selector, so a skill is
    a description a caller reads when choosing an agent, never something it
    invokes. Deriving one per tool therefore publishes an inventory nobody can
    address, on an unauthenticated well-known URL.

    So nothing is invented and nothing is copied: the tool's name becomes the
    skill id, and its docstring stays where it is. A tool description written for
    the model — "Query the customer PII warehouse. Internal only." — is not
    something to paste into a public document.
    """
    return _derive(graph, prefix="", depth=0, seen=set())


def _derive(graph: Any, *, prefix: str, depth: int, seen: set[int]) -> list[AgentSkill]:
    if depth > _MAX_SUBGRAPH_DEPTH or id(graph) in seen:
        return []
    seen = seen | {id(graph)}

    skills: list[AgentSkill] = []
    for tool in _tools(graph):
        skills.append(
            AgentSkill(
                id=f"{prefix}{tool.name}",
                name=tool.name,
                # Not the tool's own description: see `derive_skills`.
                description=DERIVED_SKILL_DESCRIPTION,
                tags=["tool", "derived"],
            )
        )

    for node_name, subgraph in _subgraphs(graph):
        skills.extend(
            _derive(
                subgraph, prefix=f"{prefix}{node_name}_", depth=depth + 1, seen=seen
            )
        )

    deduped: dict[str, AgentSkill] = {}
    for skill in skills:
        deduped.setdefault(skill.id, skill)
    return list(deduped.values())


def _tools(graph: Any) -> list[Any]:
    """Tools reachable from a compiled graph, via any `ToolNode` it contains."""
    found: list[Any] = []
    for node in getattr(graph, "nodes", {}).values():
        by_name = getattr(getattr(node, "bound", None), "tools_by_name", None)
        if isinstance(by_name, dict):
            found.extend(by_name.values())
    return found


def _subgraphs(graph: Any) -> list[tuple[str, Any]]:
    get_subgraphs = getattr(graph, "get_subgraphs", None)
    if get_subgraphs is None:
        return []
    try:
        return list(get_subgraphs())
    except Exception:
        logger.debug("could not enumerate subgraphs of %r", graph, exc_info=True)
        return []


def card_extensions(
    *,
    durable_interrupt: bool = True,
    required: bool = False,
    durable_interrupt_uri: str = EXTENSION_URI,
    conversation_model: bool = True,
    conversation_model_uri: str = CONVERSATION_EXTENSION_URI,
    pause_deadline: float | None = None,
    declared: list[AgentExtension] | None = None,
) -> list[AgentExtension]:
    """Extensions to advertise.

    `durable_interrupt` is passed as `False` for a graph that cannot honour it —
    an agent with no checkpointer must not advertise a pause that survives a
    restart. That judgement is only available for the package's own two, which
    it publishes because the server performs them. A `declared` extension is
    somebody else's promise about their own agent: it is published exactly as
    given, `params` and all, and honouring it is theirs.
    """
    extensions: list[AgentExtension] = []
    if durable_interrupt:
        extensions.append(
            durable_interrupt_extension(required=required, uri=durable_interrupt_uri)
        )
    if conversation_model:
        extensions.append(
            conversation_model_extension(
                uri=conversation_model_uri, pause_deadline=pause_deadline
            )
        )
    extensions.extend(declared or [])
    return extensions
