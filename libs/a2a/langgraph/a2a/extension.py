"""The extensions this server advertises.

Two things are true about this server that a caller cannot infer from the
protocol, and both are worth publishing rather than leaving to be discovered by
surprise:

- **The pause is a suspended graph.** `input-required` here means the run is
  parked at a checkpoint and survives a restart. It also means resuming
  re-executes the paused node from its start, so effects between the node's
  entry and the pause happen more than once — a caller deciding whether to
  answer an approval question ten minutes later deserves to know both halves.
- **A conversation is serial.** Tasks in one `contextId` run one at a time, and
  a new task arriving while the conversation is parked at `input-required` is
  refused rather than queued behind a question that may never be answered.

An extension URI is expected to resolve to a description of the extension. The
text those URIs must serve is in `extensions/` next to this package; publishing
it is a release step, and `extension_uri=` lets a deployment point at its own
copy in the meantime.
"""

from __future__ import annotations

from collections.abc import Iterable

from google.protobuf import json_format

from a2a.types.a2a_pb2 import AgentExtension
from a2a.utils.errors import ExtensionSupportRequiredError

URI = "https://langchain.dev/a2a/extensions/durable-interrupt/v1"
"""Declares `input-required` to be a checkpointed suspension, with its costs."""

CONVERSATION_URI = "https://langchain.dev/a2a/extensions/conversation-model/v1"
"""Declares how tasks in one `contextId` relate: serialised, and not queued."""

DURABLE_INTERRUPT_DESCRIPTION = (
    "input-required is a graph suspended at a checkpoint. The task survives a "
    "restart of the server and is resumed by answering it on the same taskId. "
    "Pauses carry an id and are answered by naming that id, so parallel pauses "
    "are individually answerable. Resuming re-executes the paused node from its "
    "start: any effect between that node's entry and its pause runs again, so "
    "the pre-pause body executes at least once per answer."
)

CONVERSATION_MODEL_DESCRIPTION = (
    "A contextId is one conversation on one thread of state, and one task runs "
    "in it at a time. A task sent while another is running waits for it and "
    "then runs, because a run is bounded by the agent's run timeout. A task "
    "sent while the conversation is parked at input-required is refused with "
    "UnsupportedOperationError naming the blocking task, which can be answered "
    "or cancelled: nothing obliges a counterparty to answer, so that wait would "
    "not be bounded by anything. Work that does not need this conversation's "
    "memory belongs in a conversation of its own, where it runs fully in "
    "parallel."
)

EXPIRY_DESCRIPTION = (
    "A question asked longer ago than pause_deadline_seconds is displaced when "
    "another task arrives at the conversation: that task is ended without an "
    "answer and the arriving one proceeds. Nothing sweeps — a pause nothing is "
    "waiting behind stays parked and answerable indefinitely."
)

PAUSE_DEADLINE_PARAM = "pause_deadline_seconds"
"""Where the deadline is published: a number in `params`, not prose.

`AgentExtension.params` is a `Struct` and exists for exactly this. A duration
buried in a description is not something a caller can act on.
"""


def durable_interrupt_extension(
    *, required: bool = False, uri: str = URI
) -> AgentExtension:
    """The pause contract, for `capabilities.extensions`.

    Advertised only when the graph can honour it — see `server.build_a2a_server`,
    which drops it for a graph compiled without a checkpointer rather than
    promising durability it does not have.
    """
    return AgentExtension(
        uri=uri, description=DURABLE_INTERRUPT_DESCRIPTION, required=required
    )


def conversation_model_extension(
    *, uri: str = CONVERSATION_URI, pause_deadline: float | None = None
) -> AgentExtension:
    """The conversation contract. Never `required`: it constrains us, not the caller.

    The pause deadline is stated here because A2A permits an expiration policy
    and asks that it be documented, and because a caller that walks away from a
    question deserves to know when the door closes.
    """
    description = CONVERSATION_MODEL_DESCRIPTION
    extension = AgentExtension(uri=uri, description=description, required=False)
    if pause_deadline is not None:
        extension.description += " " + EXPIRY_DESCRIPTION
        json_format.ParseDict({PAUSE_DEADLINE_PARAM: pause_deadline}, extension.params)
    return extension


def require(requested: Iterable[str], uri: str = URI) -> None:
    """Refuse a caller that has not declared support for the extension.

    Raises:
        ExtensionSupportRequiredError: the request carried no `A2A-Extensions`
            header naming it.
    """
    if uri not in set(requested):
        raise ExtensionSupportRequiredError(
            message=f"This agent requires the {uri} extension.",
            data={"required_extensions": [uri]},
        )
