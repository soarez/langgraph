"""Wiring a compiled graph into an application that speaks A2A 1.0 (and 0.3).

`create_a2a_app` builds a Starlette app; `add_a2a_routes` mounts the same
routes into an application you already have. Both refuse a configuration that
produces a server which looks correct and is not — no authentication and no
declaration, a durable checkpointer beside in-memory tasks, `multi_replica`
over a store that cannot exclude, webhook registrations less durable or
differently keyed than the tasks they belong to, a required extension the graph
cannot honour, or a state key the graph does not have.

Every one of those refusals is a developer's first contact with this package, so
each names the argument that resolves it and what accepting that argument means.
A refusal someone cannot act on immediately is one they work around, which loses
the guarantee it existed to protect.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from email.utils import formatdate
from typing import Any

from langchain_core.runnables import RunnableConfig
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import BaseRoute, Route

from a2a.server.agent_execution import RequestContext, SimpleRequestContextBuilder
from a2a.server.context import ServerCallContext
from a2a.server.owner_resolver import resolve_user_scope
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.request_handlers.response_helpers import agent_card_to_dict
from a2a.server.routes import (
    DefaultServerCallContextBuilder,
    ServerCallContextBuilder,
    create_jsonrpc_routes,
)
from a2a.server.tasks import (
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
    PushNotificationConfigStore,
    PushNotificationSender,
    TaskStore,
)
from a2a.types.a2a_pb2 import (
    AgentCard,
    AgentExtension,
    AgentSkill,
    SecurityRequirement,
    SecurityScheme,
    SendMessageRequest,
    Task,
)
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from langgraph.a2a import extension
from langgraph.a2a.authorization import (
    ContextAuthorizer,
    InMemoryContextAuthorizer,
    SubjectResolver,
    principal_subject,
    single_tenant_subject,
    subject_of,
    subject_scope,
)
from langgraph.a2a.card import build_agent_card, card_extensions, compat_card
from langgraph.a2a.card import derive_skills as derive_skills_from_graph
from langgraph.a2a.executor import (
    DEFAULT_PAUSE_DEADLINE,
    DEFAULT_RUN_TIMEOUT,
    REQUESTED_CONTEXT_ID_KEY,
    LangGraphAgentExecutor,
)
from langgraph.a2a.state import StateAdapter
from langgraph.a2a.task_store import ConditionalTaskStore, ReconcilableTaskStore
from langgraph.a2a.webhooks import (
    GuardedPushNotificationConfigStore,
    default_push_sender,
)

logger = logging.getLogger(__name__)

ORPHANED_TASK_MESSAGE = (
    "This task was running when the process executing it stopped. Its progress "
    "was tied to that process, so it cannot be resumed; send the request again."
)

RPC_PATH = "/a2a"
COMPAT_CARD_PATH = "/.well-known/agent.json"
"""Where the 0.3-shaped card is served, alongside the 1.0 card at its own path."""


class _RecordRequestedContextId(SimpleRequestContextBuilder):
    """Remembers the `contextId` the caller actually sent.

    By the time a request reaches the executor, `RequestContext.context_id` is
    either the caller's or one the SDK minted because the caller sent none, and
    the two are indistinguishable. The difference matters: a mismatch against
    the stored task is an attempt to point someone else's task at another
    conversation, while an absent one is just a client that only sent a
    `taskId`. This builder is handed the pre-generation value, so it records it.
    """

    async def build(
        self,
        context: ServerCallContext,
        params: SendMessageRequest | None = None,
        task_id: str | None = None,
        context_id: str | None = None,
        task: Task | None = None,
    ) -> RequestContext:
        context.state[REQUESTED_CONTEXT_ID_KEY] = context_id
        return await super().build(
            context=context,
            params=params,
            task_id=task_id,
            context_id=context_id,
            task=task,
        )


class _ResolveSubject(ServerCallContextBuilder):
    """Resolves the subject once, for every RPC, before anything else runs.

    It cannot be done in the executor. Only `SendMessage` reaches an executor,
    so a subject resolved there is present when a task is written and absent
    when one is read — and a task store scoped on it then hides a task from its
    own owner. Every RPC passes through a context builder, so that is where the
    subject belongs.

    A resolver that raises leaves the subject unset rather than failing the
    build: the send path raises the same error a moment later with the request
    in hand, and a read path scopes to the principal instead of dying.
    """

    def __init__(
        self, inner: ServerCallContextBuilder, resolver: SubjectResolver
    ) -> None:
        self._inner = inner
        self._resolver = resolver

    def build(self, request: Any) -> ServerCallContext:
        context = self._inner.build(request)

        # The SDK reads the negotiated protocol version out of
        # `state["headers"]` and assumes 0.3 when it is not there — so a custom
        # builder that does not copy the request headers gets every 1.0 request
        # refused, with an error that names neither headers nor the builder.
        # Cheap to prevent, expensive to debug.
        if not context.state.get("headers"):
            context.state["headers"] = dict(request.headers)

        try:
            subject_of(context, None, resolver=self._resolver)
        except Exception:
            logger.debug("could not resolve a subject for this call", exc_info=True)
        return context


class A2AConfigurationError(ValueError):
    """A server configuration that would be unsafe or silently lossy in production."""


@dataclass
class A2AServer:
    """The assembled pieces, for an application that wants to mount them itself."""

    card: AgentCard
    handler: Any
    executor: LangGraphAgentExecutor
    task_store: TaskStore | None = None
    multi_replica: bool = False
    card_routes: list[BaseRoute] = field(default_factory=list)
    rpc_routes: list[BaseRoute] = field(default_factory=list)

    @property
    def routes(self) -> list[BaseRoute]:
        return [*self.rpc_routes, *self.card_routes]

    async def astart(self) -> None:
        """Close out what an earlier process left running.

        A task's progress is tied to the process running it, so a `WORKING`
        record from a process that died belongs to nobody. A *paused* task is
        untouched: it is waiting on a checkpoint, not running, and resuming it
        is the one thing that does survive a restart.

        Nothing here sweeps on a timer — this runs once, at start, and only when
        the deployment has not declared itself multi-replica, because on more
        than one replica the same scan would close another replica's live work.

        `create_a2a_app` wires this into the lifespan; an application mounting
        the routes itself should call it on startup.
        """
        if self.multi_replica or not isinstance(self.task_store, ReconcilableTaskStore):
            return
        await self.task_store.fail_orphaned(ORPHANED_TASK_MESSAGE)

    async def aclose(self) -> None:
        """Drain in-flight tasks. `create_a2a_app` wires this into the lifespan;
        an application mounting the routes itself must call it on shutdown."""
        await self.handler.aclose()


def build_a2a_server(
    graph: Any,
    *,
    name: str,
    description: str,
    version: str,
    url: str,
    skills: list[AgentSkill] | None = None,
    derive_skills: bool = False,
    task_store: TaskStore | None = None,
    security_schemes: dict[str, SecurityScheme] | None = None,
    security_requirements: list[SecurityRequirement] | None = None,
    single_tenant: bool = False,
    context_builder: ServerCallContextBuilder | None = None,
    context_authorizer: ContextAuthorizer | None = None,
    subject_resolver: SubjectResolver | None = None,
    config_from_context: Callable[[RequestContext], RunnableConfig] | None = None,
    state: StateAdapter | None = None,
    rpc_path: str = RPC_PATH,
    enable_v0_3_compat: bool = True,
    streaming: bool = True,
    stream_tokens: bool = True,
    input_modes: list[str] | None = None,
    output_modes: list[str] | None = None,
    run_timeout: float | None = DEFAULT_RUN_TIMEOUT,
    pause_deadline: float | None = DEFAULT_PAUSE_DEADLINE,
    multi_replica: bool = False,
    push_config_store: PushNotificationConfigStore | None = None,
    push_sender: PushNotificationSender | None = None,
    allow_private_webhooks: bool = False,
    durable_interrupt_required: bool = False,
    durable_interrupt_extension: bool = True,
    extensions: list[AgentExtension] | None = None,
    extension_uri: str = extension.URI,
) -> A2AServer:
    """Validate the configuration and assemble card, handler and routes.

    Args:
        graph: a compiled graph. Compile it with a checkpointer to use `interrupt()`.
        name, description, version: the agent's identity on its card.
        url: the absolute URL callers reach `rpc_path` at. It goes on the card,
            so it must be the address a client can actually use.
        skills: declared skills. Merged ahead of derived ones.
        derive_skills: also derive skills from the graph's tools and subgraphs.
            Off by default: a skill is a description a caller reads when
            choosing an agent, not something it can invoke — A2A has no skill
            selector — so deriving one per tool publishes an inventory nobody
            can address, on an unauthenticated well-known URL.
        task_store: where A2A tasks live. In memory unless supplied, so tasks
            last as long as the process; see `BaseStoreTaskStore` for one that
            outlives it. A store supplied here must scope tasks by the
            resolved subject (`authorization.subject_scope`) or it will hand one
            end user's task to another behind the same credential; the default
            store built here already does.
        security_schemes, security_requirements: how a caller authenticates.
            Required unless `single_tenant=True`.
        single_tenant: every caller is the same subject. Say this only when the
            server is not reachable by anyone else.
        context_builder: builds the per-request `ServerCallContext`, and is
            where authentication becomes a `User`.
        context_authorizer: binds each `contextId` to its first subject. The
            default is process-local; see `InMemoryContextAuthorizer`.
        subject_resolver: who a request acts for, when the caller is an agent
            acting on behalf of its own users. Defaults to the principal.
        config_from_context: extra `RunnableConfig` per request.
        state: how A2A messages map onto graph state. See `StateAdapter`.
        input_modes, output_modes: what the card declares it accepts and
            produces. Derived from `state` when omitted, so the card cannot
            promise a content type the adapter refuses.
        run_timeout: seconds one run may take before the task is failed.
        pause_deadline: seconds a question may go unanswered before its task is
            ended and the conversation released. Stated on the card, because
            A2A asks that an expiration policy be documented.
        multi_replica: declare that more than one process serves this agent.
            Refused unless the task store implements `ConditionalTaskStore`,
            because without a conditional write two replicas cannot be kept from
            running two turns of one conversation at once — and the failure is
            not an error, it is a lost turn both callers were told had
            succeeded.
        allow_private_webhooks: permit push notification URLs pointing at
            loopback or private addresses. Off, because a registered webhook is
            a URL a caller chose and the server then makes requests to.
        durable_interrupt_required: refuse callers that have not declared
            support for the durable-interrupt extension. Off by default:
            requiring it refuses every conformant client that has not heard of
            a LangChain URI.
        durable_interrupt_extension: advertise the pause contract at all. Forced
            off for a graph with no checkpointer, which cannot honour it.
        extensions: extensions this deployment declares about its own agent,
            published on the card alongside the package's two and exactly as
            given — `params` included, since that is where an extension puts
            anything a caller can act on. The package withholds *its* own when
            the configuration cannot honour them; it cannot make that judgement
            about a graph's, so honouring a declared one is the deployment's.
        extension_uri: the URI this deployment publishes the contract at.

    Raises:
        A2AConfigurationError: an unauthenticated multi-tenant server, a durable
            checkpointer paired with an in-memory task store, a state key the
            graph does not have, or a required extension the graph cannot
            honour.
    """
    if not single_tenant and context_builder is None:
        raise A2AConfigurationError(
            "contextId is supplied by the caller, so serving without authentication "
            "lets a second caller resume the first's conversation. Two ways out: "
            "pass context_builder=... that authenticates the request, or declare "
            "single_tenant=True — which says every caller is the same subject and "
            "conversations are not owned, and is the right answer for local work."
        )
    if not single_tenant and not security_schemes:
        raise A2AConfigurationError(
            "This server authenticates its callers, and an agent that does must say "
            "so on its card or a conformant client arrives without credentials. Pass "
            "security_schemes={...}, which is published as the card's schemes and "
            "required by any one of them; or declare single_tenant=True, which serves "
            "an anonymous caller and advertises nothing."
        )

    # Scoped by the resolved subject, not by the principal. With a subject
    # resolver in play, a store keyed on the credential hands one end user's
    # task to another behind the same peer.
    store = (
        task_store
        if task_store is not None
        else InMemoryTaskStore(owner_resolver=subject_scope)
    )
    if multi_replica and not isinstance(store, ConditionalTaskStore):
        raise A2AConfigurationError(
            f"multi_replica=True with {type(store).__name__}, which cannot save "
            "conditionally, so nothing would stop two replicas running two turns of "
            "one conversation — and that failure has no symptom: both callers are "
            "told their turn succeeded and one of the turns is gone. Either pass a "
            "task_store implementing langgraph.a2a.ConditionalTaskStore (a version "
            "the write asserts on, and one WORKING task per context), or leave "
            "multi_replica=False — the default, which says this process is the only "
            "copy and lets an in-process lock be the whole answer."
        )
    if _is_durable_checkpointer(graph) and _is_in_memory(store):
        raise A2AConfigurationError(
            "This graph checkpoints durably while its tasks would be kept in memory, "
            "and the two halves of a pause live in different places: after a restart "
            "the graph is still suspended at its question and the task that asked it "
            "is gone, so the caller's answer arrives as a fresh turn. Match them. "
            "For a deployment, pass task_store=BaseStoreTaskStore(store) over the same "
            "backend as the checkpointer. To exercise interrupt() locally, compile "
            "with InMemorySaver instead and leave the task store alone — both halves "
            "in memory is a coherent pair, and this refusal is only about the mix."
        )
    if (
        push_config_store is not None
        and not _is_in_memory(store)
        and isinstance(push_config_store, InMemoryPushNotificationConfigStore)
    ):
        raise A2AConfigurationError(
            "Push notification configs would be kept in memory beside a durable task "
            "store. The tasks a caller registered a webhook for outlive the "
            "registration, so after a restart those tasks reach a terminal state and "
            "nobody is told. Pass push_config_store=... over the same backend as the "
            "tasks, or leave it unset — no config store means no webhooks, and the "
            "card says so rather than promising a delivery that will not happen."
        )
    adapter = state or StateAdapter()
    _check_state_keys(graph, adapter)

    # A graph that cannot suspend must not advertise a pause that survives a
    # restart: the extension is a promise, and this one it cannot keep.
    can_suspend = getattr(graph, "checkpointer", None) is not None
    if not can_suspend:
        if durable_interrupt_required:
            raise A2AConfigurationError(
                "durable_interrupt_required=True on a graph with no checkpointer: "
                "the card would demand that every caller support a pause this server "
                "cannot perform, and refuse the ones that do not. Compile the graph "
                "with a checkpointer, or leave durable_interrupt_required=False — the "
                "default, which serves the same graph without the pause and without "
                "advertising it."
            )
        if durable_interrupt_extension:
            logger.warning(
                "graph %r has no checkpointer: serving without the durable-interrupt "
                "extension, and interrupt() cannot be resumed",
                getattr(graph, "name", graph),
            )
        durable_interrupt_extension = False

    card = build_agent_card(
        name=name,
        description=description,
        version=version,
        url=url,
        skills=_skills(graph, skills, derive_skills),
        streaming=streaming,
        push_notifications=push_config_store is not None,
        security_schemes=security_schemes,
        security_requirements=security_requirements,
        input_modes=input_modes or adapter.input_modes(),
        output_modes=output_modes or adapter.output_modes(),
        extensions=card_extensions(
            durable_interrupt=durable_interrupt_extension,
            required=durable_interrupt_required,
            durable_interrupt_uri=extension_uri,
            pause_deadline=pause_deadline,
            declared=extensions,
        ),
        serve_v0_3=enable_v0_3_compat,
    )

    resolver = (
        subject_resolver
        if subject_resolver is not None
        else (single_tenant_subject if single_tenant else principal_subject)
    )
    executor = LangGraphAgentExecutor(
        graph,
        state=adapter,
        config_from_context=config_from_context,
        context_authorizer=(
            context_authorizer
            if context_authorizer is not None
            else InMemoryContextAuthorizer()
        ),
        subject_resolver=resolver,
        stream_tokens=stream_tokens,
        require_extension=durable_interrupt_required,
        extension_uri=extension_uri,
        run_timeout=run_timeout,
        pause_deadline=pause_deadline,
        task_store=store,
        multi_replica=multi_replica,
    )
    if push_config_store is not None:
        if getattr(push_config_store, "owner_resolver", None) is resolve_user_scope:
            raise A2AConfigurationError(
                f"{type(push_config_store).__name__} partitions registrations by the "
                "authenticated principal while tasks are partitioned by the resolved "
                "subject, so the two disagree about who owns a registration as soon "
                "as a credential is rotated or one credential fronts several users. "
                "Pass owner_resolver=langgraph.a2a.subject_scope."
            )
        # Registrations are checked before they are stored, so a caller learns
        # that a destination is refused at once rather than never hearing from
        # a webhook it believes is armed.
        push_config_store = GuardedPushNotificationConfigStore(
            push_config_store, allow_private=allow_private_webhooks
        )
        if push_sender is None:
            # A config store with nothing to send from accepts registrations and
            # delivers nothing, which is worse than declining them.
            push_sender = default_push_sender(push_config_store)

    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=store,
        agent_card=card,
        push_config_store=push_config_store,
        push_sender=push_sender,
        request_context_builder=_RecordRequestedContextId(task_store=store),
    )

    card_routes: list[BaseRoute] = [
        Route(
            AGENT_CARD_WELL_KNOWN_PATH,
            _card_endpoint(agent_card_to_dict(card)),
            methods=["GET"],
        )
    ]
    if enable_v0_3_compat:
        card_routes.append(
            Route(COMPAT_CARD_PATH, _card_endpoint(compat_card(card)), methods=["GET"])
        )
    rpc_routes: list[BaseRoute] = list(
        create_jsonrpc_routes(
            handler,
            rpc_path,
            context_builder=_ResolveSubject(
                context_builder or DefaultServerCallContextBuilder(), resolver
            ),
            enable_v0_3_compat=enable_v0_3_compat,
        )
    )
    return A2AServer(
        card=card,
        handler=handler,
        executor=executor,
        task_store=store,
        multi_replica=multi_replica,
        card_routes=card_routes,
        rpc_routes=rpc_routes,
    )


def create_a2a_app(graph: Any, **kwargs: Any) -> Starlette:
    """A Starlette application serving `graph` as an A2A agent.

    Takes the arguments of `build_a2a_server`.
    """
    server = build_a2a_server(graph, **kwargs)

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        await server.astart()
        try:
            yield
        finally:
            await server.aclose()

    app = Starlette(routes=server.routes, lifespan=lifespan)
    app.state.a2a = server
    return app


def add_a2a_routes(app: Any, graph: Any, **kwargs: Any) -> A2AServer:
    """Mount A2A routes onto an application you already have.

    Takes the arguments of `build_a2a_server` and returns what it built, so the
    served card can be asserted on or published elsewhere. FastAPI applications
    get their routes through the SDK's FastAPI helper, so the endpoints appear
    in the generated OpenAPI schema.

    Call `A2AServer.astart()` on startup and `A2AServer.aclose()` on shutdown,
    so tasks an earlier process left running are closed out and in-flight ones
    are drained rather than abandoned.
    """
    server = build_a2a_server(graph, **kwargs)

    if _is_fastapi(app):
        from a2a.server.routes import add_a2a_routes_to_fastapi

        add_a2a_routes_to_fastapi(
            app,
            agent_card_routes=server.card_routes,
            jsonrpc_routes=server.rpc_routes,
        )
    else:
        app.router.routes.extend(server.routes)

    return server


CARD_MAX_AGE = 300
"""Seconds a card may be cached. Discovery is a hot path and cards move rarely."""


def _card_endpoint(payload: dict[str, Any]) -> Callable[[Any], Any]:
    """Serve a fixed card body with validators, so callers can revalidate cheaply.

    A card is fetched before every conversation with an unfamiliar agent and
    changes when the agent is redeployed. Without `ETag` and `Cache-Control`
    every discovery is a full transfer.
    """
    body = json.dumps(payload, separators=(",", ":")).encode()
    etag = f'"{hashlib.sha256(body).hexdigest()[:32]}"'
    last_modified = formatdate(usegmt=True)
    headers = {
        "cache-control": f"public, max-age={CARD_MAX_AGE}",
        "etag": etag,
        "last-modified": last_modified,
    }

    async def endpoint(request: Any) -> Response:
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=headers)
        return Response(body, media_type="application/json", headers=headers)

    return endpoint


def _check_state_keys(graph: Any, adapter: StateAdapter) -> None:
    """Refuse a state key the graph does not have.

    Conformance is a property of the package, the graph and the configuration
    together. An adapter pointed at a key the graph never declares fails at the
    first request, as a protocol-visible error on a server that started
    cleanly — so it fails here instead, where the message can name the key.

    A graph whose schema cannot be introspected is served without the check
    rather than refused: an unreadable schema is not a wrong one.
    """
    try:
        schema = graph.get_input_jsonschema()
        properties = set(schema.get("properties") or {})
    except Exception:
        logger.debug("could not read the input schema of %r", graph, exc_info=True)
        return
    if not properties:
        return

    wanted = {"message_key": adapter.message_key}
    if adapter.input_data_key:
        wanted["input_data_key"] = adapter.input_data_key
    missing = {
        argument: key for argument, key in wanted.items() if key not in properties
    }
    if missing:
        raise A2AConfigurationError(
            "The state adapter names keys this graph does not declare: "
            + ", ".join(
                f"StateAdapter({argument}={key!r})" for argument, key in missing.items()
            )
            + f". The graph's input schema has {sorted(properties)}. Point the adapter "
            "at one of those, or add the key to the graph's state — unchecked, this "
            "surfaces as a protocol error on the first request to a server that "
            "started cleanly."
        )


def _skills(
    graph: Any, declared: list[AgentSkill] | None, derive: bool
) -> list[AgentSkill]:
    """Declared skills win; derived ones fill in. Neither invents one."""
    skills = list(declared or [])
    if not derive:
        return skills
    known = {skill.id for skill in skills}
    skills.extend(s for s in derive_skills_from_graph(graph) if s.id not in known)
    return skills


def _is_durable_checkpointer(graph: Any) -> bool:
    checkpointer = getattr(graph, "checkpointer", None)
    if not checkpointer or checkpointer is True:
        return False
    from langgraph.checkpoint.memory import InMemorySaver

    return not isinstance(checkpointer, InMemorySaver)


def _is_in_memory(task_store: TaskStore) -> bool:
    if isinstance(task_store, InMemoryTaskStore):
        return True
    from langgraph.a2a.task_store import BaseStoreTaskStore

    if isinstance(task_store, BaseStoreTaskStore):
        from langgraph.store.memory import InMemoryStore

        return isinstance(task_store._store, InMemoryStore)
    return False


def _is_fastapi(app: Any) -> bool:
    return any(base.__module__.startswith("fastapi") for base in type(app).__mro__[:-1])
