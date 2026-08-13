"""Who a conversation belongs to, and which thread it runs on.

`contextId` is chosen by the caller. Nothing in A2A stops a second caller from
presenting one it did not create, so a context is bound on first use and a later
mismatch is reported as a missing task — the specification does not permit
distinguishing "not yours" from "not there".

**It binds to a subject, not to the caller's credential.** A2A's caller is
normally an agent presenting one service credential on behalf of many end users,
and 1.0 has no on-behalf-of field. Binding to the credential would collapse
every user behind a peer into a single owner, so the isolation would be nil in
exactly the deployment that wants it. The subject comes from a hook the
deployment supplies.

**The subject is not part of `thread_id`.** Credentials rotate and principals
get renamed; a conversation must not fork onto an empty thread because a key was
renewed. The binding lives in the ownership store, where it can be re-pointed.

**Deriving the thread id is hygiene, not access control.** It keeps a caller
from naming an internal key, colliding with something else in the store, or
learning the shape of the store's keys. It is deterministic, so anyone holding a
`contextId` and this function holds the thread id too. The authorization is the
ownership binding and nothing else — a hash in front of a key is not a lock.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.utils.errors import InvalidRequestError, TaskNotFoundError

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore

SINGLE_TENANT_OWNER = "__single_tenant__"
"""The subject every request runs as when the server is declared single-tenant."""

SUBJECT_STATE_KEY = "a2a_subject"
"""Where the resolved subject is recorded on the `ServerCallContext`.

RPCs other than `SendMessage` carry no message and therefore no message
metadata, so a resolver that reads metadata cannot answer for them. Recording
the subject here lets a task store scope by it on the path that has one.
"""

_STORE_NAMESPACE = ("a2a", "context_owners")


SubjectResolver = Callable[[ServerCallContext, RequestContext | None], str]
"""Resolves the subject a request acts for.

Called with the request context on `SendMessage`, and with `None` for it on
every other RPC.

**The subject cannot travel in message metadata.** `metadata` is a field on
`SendMessageRequest` and `CancelTaskRequest` and on none of `GetTaskRequest`,
`ListTasksRequest` or `SubscribeToTaskRequest`. A subject carried there is
present on write and absent on read, so a task store scoped by it does not fail
cleanly — it returns a task to the wrong reader or hides one from its owner,
depending on which key it fell back to. The channels that work are a security
scheme that carries the subject, or an extension header: both reach the
`ServerCallContextBuilder`, which puts the result where every RPC can see it.
"""


def principal_subject(
    call_context: ServerCallContext, _request: RequestContext | None = None
) -> str:
    """The authenticated principal is the subject.

    Correct when one credential really means one subject: a personal token, or
    a peer that fronts a single user. Wrong the moment a peer serves several.
    """
    user = call_context.user
    if not user.is_authenticated or not user.user_name:
        raise InvalidRequestError(
            message="Unauthenticated request. This agent requires an authenticated caller."
        )
    if call_context.tenant:
        return f"{call_context.tenant}\x1f{user.user_name}"
    return user.user_name


def state_subject(key: str, *, required: bool = True) -> SubjectResolver:
    """Take the subject from `ServerCallContext.state[key]`.

    The recommended production hook: the `ServerCallContextBuilder` reads a
    claim out of the verified credential and puts it there, so the subject is
    attested by the same thing that authenticated the call, and it is available
    on every RPC rather than only on `SendMessage`.
    """

    def resolve(
        call_context: ServerCallContext, _request: RequestContext | None = None
    ) -> str:
        value = call_context.state.get(key)
        if isinstance(value, str) and value:
            return value
        if required:
            raise InvalidRequestError(
                message=f"This agent requires a subject in the call context under {key!r}."
            )
        return principal_subject(call_context)

    return resolve


def single_tenant_subject(
    _call_context: ServerCallContext, _request: RequestContext | None = None
) -> str:
    """Every request is the same subject. Only for a server nobody else reaches."""
    return SINGLE_TENANT_OWNER


def subject_of(
    call_context: ServerCallContext,
    request: RequestContext | None = None,
    *,
    resolver: SubjectResolver | None = None,
) -> str:
    """The subject this request acts for, recorded on the call context."""
    resolve = resolver or principal_subject
    subject = resolve(call_context, request)
    call_context.state[SUBJECT_STATE_KEY] = subject
    return subject


def subject_scope(call_context: ServerCallContext) -> str:
    """Owner scope for the SDK's task stores: the resolved subject if there is one.

    Falls back to the principal, which is what an RPC that carries no message
    can offer when the subject is declared in metadata.
    """
    subject = call_context.state.get(SUBJECT_STATE_KEY)
    if isinstance(subject, str) and subject:
        return subject
    user = call_context.user
    if not user.is_authenticated or not user.user_name:
        return ""
    if call_context.tenant:
        return f"{call_context.tenant}\x1f{user.user_name}"
    return user.user_name


def thread_id(context_id: str) -> str:
    """The LangGraph `thread_id` for one conversation.

    A **pure function of the `contextId`** — one context, one thread, forever,
    and nothing else is an input. Not the subject: identities rotate, and a
    renewed key must not fork the conversation onto an empty thread. Not the
    task: A2A makes a task terminal when it completes, so the second turn is a
    new task carrying the same `contextId`, and keying on it would start every
    turn after the first with no memory. Not a timestamp or a nonce, for the
    same reason — anything that can change makes one conversation resolve to two
    threads.

    Deriving rather than using the id directly is hygiene, not access control:
    it keeps a caller-chosen string out of the store's key space. Ownership is
    what authorizes.
    """
    digest = hashlib.sha256(context_id.encode()).hexdigest()
    return f"a2a-{digest[:32]}"


def context_namespace(subject: str, context_id: str) -> tuple[str, ...]:
    """`BaseStore` namespace for memory a deployment wants outside the thread."""
    return ("a2a", "context", subject, context_id)


class ContextAuthorizer(ABC):
    """Binds a `contextId` to its subject and refuses everyone else."""

    @abstractmethod
    async def authorize(self, context_id: str, subject: str) -> None:
        """Bind `context_id` to `subject`, or raise if it belongs to another.

        Raises:
            TaskNotFoundError: the context is bound to a different subject.
        """


class InMemoryContextAuthorizer(ContextAuthorizer):
    """Bindings in a process-local dict.

    This is the default, and it makes the ownership guarantee **single-replica
    as shipped**: a second replica has an empty map, so it binds the same
    context to whoever reaches it first. Use `StoreContextAuthorizer` for
    anything with more than one process — which is also the deployment shape the
    rest of the package supports today.
    """

    def __init__(self) -> None:
        self._owners: dict[str, str] = {}

    async def authorize(self, context_id: str, subject: str) -> None:
        bound = self._owners.setdefault(context_id, subject)
        if bound != subject:
            raise TaskNotFoundError


class StoreContextAuthorizer(ContextAuthorizer):
    """Bindings in a LangGraph `BaseStore`, so they survive a restart.

    The read and the write are not atomic. Two first uses of one context id by
    different subjects, arriving concurrently at different replicas, can both
    succeed; the loser's later requests are refused. A store offering
    compare-and-set would close that window, and none does today.
    """

    def __init__(self, store: BaseStore) -> None:
        self._store = store

    async def authorize(self, context_id: str, subject: str) -> None:
        item = await self._store.aget(_STORE_NAMESPACE, context_id)
        if item is None:
            await self._store.aput(_STORE_NAMESPACE, context_id, {"subject": subject})
            return
        if item.value.get("subject") != subject:
            raise TaskNotFoundError
