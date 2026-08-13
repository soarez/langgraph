"""How a graph's state is read and written across the A2A boundary.

One object rather than a handful of loose keyword arguments, because the four
decisions are a single contract: where the conversation lives, where structured
input is allowed to land, and how the final state becomes an answer.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from a2a.types.a2a_pb2 import Part
from a2a.utils.errors import InvalidParamsError
from langgraph.a2a.parts import parts_data, parts_to_content, text_part

logger = logging.getLogger(__name__)


def last_ai_text(state: Any, message_key: str = "messages") -> str:
    """The last assistant turn in a graph's final state, as text.

    Offered for `StateAdapter(output_text=...)`, which is where a graph says
    that its last turn is its answer. Nothing uses it by default: with no
    mapping the artifact is the transcript, and picking the last turn is a
    claim only the graph's author can make.
    """
    messages = _get(state, message_key) or []
    for message in reversed(messages):
        content = getattr(message, "content", None)
        role = getattr(message, "type", None)
        if content is None and isinstance(message, dict):
            content, role = message.get("content"), message.get("role")
        if role in ("ai", "assistant") and content:
            if isinstance(content, str):
                return content
            return "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
    return ""


def _get(state: Any, key: str) -> Any:
    if isinstance(state, dict):
        return state.get(key)
    return getattr(state, key, None)


@dataclass
class StateAdapter:
    """Maps A2A messages onto graph state and graph state onto A2A results.

    Args:
        message_key: state key holding the conversation.
        input_data_key: the single state key inbound `data` parts may write.
            `None` — the default — refuses them. They can never merge into the
            top level of the state: a caller that could write any key could
            overwrite the conversation, including with a non-user role.
        output_text: final state to the text of the response artifact. Supplying
            it says this graph can state its own result, which turns token
            streaming off — see below.
        output_data: final state to an optional structured payload, appended as
            a `data` part when the run ends. Additive: it does not choose
            between the two shapes and does not turn streaming off.
        output_parts: final state to the artifact's parts directly. The escape
            hatch for an answer that is neither text nor JSON — a generated
            file, an image, a URL — which `output_text` cannot express. When it
            returns parts, they are the artifact and `output_text` is not
            consulted. Build the parts here rather than in a node: a protobuf
            `Part` is not msgpack-serialisable, so it cannot be written to graph
            state at all. Keep a description in state and construct parts from
            it.

    An artifact has one of two shapes and the adapter picks which. With **no**
    output mapping it is the transcript: tokens open it and every visible model
    token appends, which is all anything assembled during a run can be — a graph
    that calls tools speaks more than once and no run knows which turn is its
    last until it ends. With `output_text` or `output_parts` it is that
    mapping's result, read from the final state once the run is over, and
    nothing streams into it.
    """

    message_key: str = "messages"
    input_data_key: str | None = None
    output_text: Callable[[Any], str] | None = None
    output_data: Callable[[Any], Any] | None = None
    output_parts: Callable[[Any], list[Part] | None] | None = None

    def to_graph_input(self, parts: Sequence[Part]) -> dict[str, Any]:
        """Build the graph input for one inbound A2A message.

        The message is always constructed with a user role. There is no path by
        which a caller sets the role of the turn it is sending.
        """
        state: dict[str, Any] = {}
        content = parts_to_content(list(parts))
        if content is not None:
            state[self.message_key] = [{"role": "user", "content": content}]

        payloads = parts_data(list(parts))
        if not payloads:
            return state
        if self.input_data_key is None:
            raise InvalidParamsError(
                message=(
                    "This agent accepts no structured input. Declare "
                    "StateAdapter(input_data_key=...) to route `data` parts to a "
                    "state key."
                )
            )
        state[self.input_data_key] = payloads[0] if len(payloads) == 1 else payloads
        return state

    @property
    def maps_output(self) -> bool:
        """Whether this graph states its own result.

        The switch between the two shapes an artifact can have, and there is
        deliberately no second one: a result computed from the final state
        cannot be streamed while the run is still producing it, so supplying a
        mapping is what turns token streaming off.
        """
        return self.output_text is not None or self.output_parts is not None

    def to_parts(self, state: Any) -> list[Part]:
        """The mapping's result. Only called when there is a mapping."""
        if self.output_parts is not None:
            parts = self.output_parts(state)
            if parts:
                return list(parts)
        if self.output_text is not None:
            text = self.output_text(state)
            return [text_part(text)] if text else []
        return []

    def to_data(self, state: Any) -> Any:
        """The structured payload, or `None`. Additive in either shape."""
        return self.output_data(state) if self.output_data else None

    def input_modes(self) -> list[str]:
        """What this adapter will actually accept, for the card to declare.

        `application/json` on a card is a promise that a `data` part will be
        taken. When no state key is declared for structured input, that promise
        would be false — the part is refused — so the card does not make it.
        """
        modes = ["text/plain"]
        if self.input_data_key:
            modes.append("application/json")
        return modes

    def output_modes(self) -> list[str]:
        """What this adapter can produce."""
        modes = ["text/plain"]
        if self.output_data is not None or self.output_parts is not None:
            modes.append("application/json")
        return modes
