"""How a LangGraph `interrupt()` crosses the A2A boundary, and comes back.

A2A says what a pause *is* — the task goes `input-required` and the caller
answers on the same `taskId` — and deliberately says nothing about how the agent
holds its place. So there is no protocol-blessed way to say "these are the three
questions, answer them by id", and a graph that pauses in two parallel branches
needs exactly that.

The shape here is id correlation: the pause is a `data` part carrying the
interrupt id, and the answer is a `data` part naming the same id.
`Command(resume={id: value})` takes the decoded map unchanged.

**The keys are ours.** Google ADK solves the same problem the same way, and this
design is modelled on it, but its keys are prefixed into its own namespace and
its credential request carries a schema this package does not implement — so a
client that recognised them would mis-handle what we send. Interoperability with
a specific client is a claim to be tested with that client in the loop, not
asserted by borrowing its private field names.

| Key | Value |
|---|---|
| `metadata["langgraph_a2a_kind"]` | `interrupt` / `credential_request` on a pause, `interrupt_response` on an answer |
| pause `data` | `{"interrupt_id", "node", "payload"}` |
| answer `data` | `{"interrupt_id", "value"}` |

A caller that speaks none of this can still answer: plain text resumes the one
pending interrupt, and is refused when more than one is outstanding.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from langgraph.types import Interrupt

from a2a.types.a2a_pb2 import Part
from langgraph.a2a import _encoding
from langgraph.a2a.parts import data_part, part_metadata, value_to_python

logger = logging.getLogger(__name__)

METADATA_KIND_KEY = "langgraph_a2a_kind"
"""Part-level metadata key marking a part as a pause or an answer to one."""

KIND_INTERRUPT = "interrupt"
KIND_CREDENTIAL_REQUEST = "credential_request"
KIND_INTERRUPT_RESPONSE = "interrupt_response"

INTERRUPT_ID_KEY = "interrupt_id"
PAYLOAD_KEY = "payload"
VALUE_KEY = "value"
NODE_KEY = "node"


@dataclass
class CredentialRequest:
    """Pass this to `interrupt()` to park the task at `auth-required`.

    ```python
    token = interrupt(CredentialRequest(scheme="github", description="repo scope"))
    ```

    `scheme` names a security scheme declared on the agent card. The resume
    value is whatever the caller answers with — this package does not inspect or
    store it.
    """

    scheme: str
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)


PayloadEncoder = Callable[[Any], Any]
"""Maps an interrupt payload onto JSON-compatible data. See `_encoding.encode`."""


@dataclass(frozen=True)
class PendingInterrupt:
    """One interrupt awaiting an answer, and what we know of where it came from."""

    id: str
    value: Any
    node: str | None = None

    @property
    def is_credential_request(self) -> bool:
        return isinstance(self.value, CredentialRequest)

    @property
    def kind(self) -> str:
        return KIND_CREDENTIAL_REQUEST if self.is_credential_request else KIND_INTERRUPT


def to_pending(
    interrupts: Sequence[Interrupt], nodes: dict[str, str] | None = None
) -> list[PendingInterrupt]:
    """Adapt LangGraph `Interrupt`s, tagging each with its node where known."""
    nodes = nodes or {}
    return [
        PendingInterrupt(id=i.id, value=i.value, node=nodes.get(i.id))
        for i in interrupts
    ]


def interrupt_part(
    pending: PendingInterrupt, *, encoder: PayloadEncoder = _encoding.encode
) -> Part:
    """The `data` part that carries one pause across the wire."""
    payload: dict[str, Any] = {
        INTERRUPT_ID_KEY: pending.id,
        PAYLOAD_KEY: encoder(pending.value),
    }
    if pending.node:
        payload[NODE_KEY] = pending.node
    return data_part(payload, metadata={METADATA_KIND_KEY: pending.kind})


def prompt_text(pending: Sequence[PendingInterrupt]) -> str:
    """A human-readable rendering of the pauses, for a caller that has no other.

    Only genuinely human-readable material is used: a string payload, or the
    first of a few conventional keys. A payload with neither contributes nothing
    rather than a serialised blob.
    """
    lines: list[str] = []
    for item in pending:
        value = item.value
        if isinstance(value, CredentialRequest):
            lines.append(
                f"Credentials required for {value.scheme}"
                + (f": {value.description}" if value.description else "")
            )
            continue
        if isinstance(value, str):
            lines.append(value)
            continue
        if isinstance(value, dict):
            for key in ("question", "prompt", "message", "text", "description"):
                if isinstance(value.get(key), str):
                    lines.append(value[key])
                    break
    return "\n".join(lines)


def interrupt_responses(parts: Sequence[Part]) -> dict[str, Any]:
    """Decode `{interrupt id: resume value}` from a caller's answers."""
    resumes: dict[str, Any] = {}
    for part in parts:
        if part.WhichOneof("content") != "data":
            continue
        if part_metadata(part).get(METADATA_KIND_KEY) != KIND_INTERRUPT_RESPONSE:
            continue
        payload = value_to_python(part.data)
        if not isinstance(payload, dict):
            continue
        interrupt_id = payload.get(INTERRUPT_ID_KEY)
        if not isinstance(interrupt_id, str) or not interrupt_id:
            logger.warning("interrupt_response part with no interrupt id; ignoring")
            continue
        resumes[interrupt_id] = _encoding.decode(payload.get(VALUE_KEY))
    return resumes


class AmbiguousResume(ValueError):
    """Plain text cannot answer a pause when more than one is outstanding."""


def resume_map(
    parts: Sequence[Part], text: str, pending: Sequence[PendingInterrupt]
) -> dict[str, Any]:
    """Build the `Command(resume=...)` mapping for this answer.

    Structured answers win. Failing those, **the server binds plain text to the
    single outstanding pause** — the caller does not address it and is not asked
    to. With more than one outstanding, the text is refused with an error naming
    them, because guessing would answer the wrong question irreversibly.

    Answers naming an interrupt that is no longer pending are dropped: resuming
    an id the graph is not waiting on does nothing at all and would look like
    success.
    """
    pending_ids = [item.id for item in pending]
    resumes = {
        interrupt_id: value
        for interrupt_id, value in interrupt_responses(parts).items()
        if interrupt_id in pending_ids
    }
    if resumes:
        return resumes
    if not text:
        return {}
    if len(pending_ids) > 1:
        # The server binds a text answer to a pause; it never guesses which.
        raise AmbiguousResume(
            f"{len(pending_ids)} questions are awaiting an answer "
            f"({', '.join(pending_ids)}); a text reply cannot say which one it "
            "answers. Reply with a data part naming the interrupt id."
        )
    return {pending_ids[0]: text} if pending_ids else {}
