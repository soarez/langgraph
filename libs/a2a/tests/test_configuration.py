"""What the server refuses to start as, and what it passes into the graph."""

from __future__ import annotations

from typing import Any

import pytest
from a2a.server.context import ServerCallContext
from a2a.types.a2a_pb2 import GetTaskRequest, Part, TaskState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.store.memory import InMemoryStore
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from langgraph.a2a import (
    A2AConfigurationError,
    InMemoryContextAuthorizer,
    StateAdapter,
    TaskRejected,
    add_a2a_routes,
    build_a2a_server,
    context_namespace,
    create_a2a_app,
    thread_id,
)
from langgraph.a2a.authorization import (
    SINGLE_TENANT_OWNER,
    principal_subject,
    single_tenant_subject,
    state_subject,
)
from tests.agent import State, build_graph
from tests.conftest import (
    RPC_URL,
    SECURITY_SCHEMES,
    BearerContextBuilder,
    ConditionalStore,
    artifact_text,
    build_app,
    http_client,
    make_client,
    send,
    text_message,
)

BASE_KWARGS: dict[str, Any] = {
    "name": "n",
    "description": "d",
    "version": "1",
    "url": RPC_URL,
}


def graph() -> Any:
    return build_graph(checkpointer=InMemorySaver())


def test_serving_without_authentication_is_refused() -> None:
    with pytest.raises(A2AConfigurationError, match="without authentication"):
        create_a2a_app(graph(), **BASE_KWARGS)


def test_authenticating_without_declaring_a_scheme_is_refused() -> None:
    with pytest.raises(A2AConfigurationError, match="say so on its card"):
        create_a2a_app(graph(), context_builder=BearerContextBuilder(), **BASE_KWARGS)


def test_single_tenant_is_an_explicit_declaration() -> None:
    app = create_a2a_app(graph(), single_tenant=True, **BASE_KWARGS)

    assert app.state.a2a.card.name == "n"


async def test_a_single_tenant_server_serves_an_anonymous_caller() -> None:
    app = create_a2a_app(graph(), single_tenant=True, **BASE_KWARGS)

    async with http_client(app, token="") as http:
        client = await make_client(http)
        task = await send(client, text_message("hello world"))

    assert artifact_text(task) == ["echo: hello world"]


def test_the_principal_resolver_refuses_an_unauthenticated_caller() -> None:
    with pytest.raises(Exception, match="[Uu]nauthenticated"):
        principal_subject(ServerCallContext())

    assert single_tenant_subject(ServerCallContext()) == SINGLE_TENANT_OWNER


def test_a_thread_is_the_conversation_and_nothing_else() -> None:
    """Not the task — a follow-up turn is a new task and must see the first.

    Not the subject either — a rotated credential must not fork the
    conversation onto an empty thread.
    """
    assert thread_id("ctx") == thread_id("ctx")
    assert thread_id("ctx") != thread_id("other")


def test_shared_memory_is_namespaced_by_subject_and_context() -> None:
    assert context_namespace("alice", "ctx") == ("a2a", "context", "alice", "ctx")


async def test_the_context_authorizer_binds_on_first_use() -> None:
    authorizer = InMemoryContextAuthorizer()

    await authorizer.authorize("ctx", "alice")
    await authorizer.authorize("ctx", "alice")
    with pytest.raises(Exception, match="[Nn]ot found"):
        await authorizer.authorize("ctx", "bob")


async def test_config_from_context_reaches_the_graph_and_cannot_move_the_thread() -> (
    None
):
    seen: list[dict] = []

    def config_from_context(context: Any) -> dict:
        return {
            "configurable": {"thread_id": "hijacked", "tenant": context.tenant or "t"},
            "recursion_limit": 7,
        }

    app = create_a2a_app(
        graph(),
        security_schemes=SECURITY_SCHEMES,
        context_builder=BearerContextBuilder(),
        config_from_context=config_from_context,
        **BASE_KWARGS,
    )
    executor = app.state.a2a.executor
    original = executor.graph.astream

    def spy(input_: Any, config: Any = None, **kwargs: Any) -> Any:
        seen.append(config)
        return original(input_, config=config, **kwargs)

    executor.graph = type(
        "Spy",
        (),
        {"astream": staticmethod(spy), "aget_state": executor.graph.aget_state},
    )()

    async with http_client(app, token="alice") as http:
        client = await make_client(http)
        await send(client, text_message("hello world"))

    assert seen and seen[0]["recursion_limit"] == 7
    assert seen[0]["configurable"]["tenant"] == "t"
    assert seen[0]["configurable"]["thread_id"] != "hijacked"


async def test_routes_mount_onto_an_existing_starlette_app() -> None:
    host = Starlette(
        routes=[Route("/health", lambda _r: PlainTextResponse("ok"), methods=["GET"])]
    )
    server = add_a2a_routes(
        host,
        graph(),
        security_schemes=SECURITY_SCHEMES,
        context_builder=BearerContextBuilder(),
        **BASE_KWARGS,
    )

    try:
        async with http_client(host, token="alice") as http:
            assert (await http.get("/health")).text == "ok"
            client = await make_client(http)
            task = await send(client, text_message("hello world"))
            assert artifact_text(task) == ["echo: hello world"]
    finally:
        await server.aclose()


def test_build_a2a_server_returns_the_pieces_without_an_app() -> None:
    server = build_a2a_server(
        graph(),
        security_schemes=SECURITY_SCHEMES,
        context_builder=BearerContextBuilder(),
        **BASE_KWARGS,
    )

    assert [type(route).__name__ for route in server.routes]
    assert server.card.name == "n"


async def test_a_rejected_task_is_rejected_not_failed() -> None:
    """`FAILED` says the agent tried; `REJECTED` says it would not."""

    def refuse(_state: State) -> dict:
        raise TaskRejected("this agent does not do that")

    refusing = (
        StateGraph(State)
        .add_node("refuse", refuse)
        .add_edge(START, "refuse")
        .compile(checkpointer=InMemorySaver())
    )
    app = build_app(graph=refusing)

    async with http_client(app, token="alice") as http:
        client = await make_client(http)
        task = await send(client, text_message("anything"))

    assert task.status.state == TaskState.TASK_STATE_REJECTED
    # A refusal that will not say what was refused is not actionable, so unlike
    # a failure the message is passed through.
    assert "does not do that" in "".join(
        part.text for part in task.status.message.parts if part.HasField("text")
    )


async def test_output_parts_builds_the_artifact_directly() -> None:
    def emit(_state: State) -> dict:
        return {"messages": []}

    graph_ = (
        StateGraph(State)
        .add_node("emit", emit)
        .add_edge(START, "emit")
        .compile(checkpointer=InMemorySaver())
    )
    app = build_app(
        graph=graph_,
        state=StateAdapter(
            output_parts=lambda _state: [
                Part(raw=b"pdf-bytes", media_type="application/pdf", filename="r.pdf")
            ]
        ),
    )

    async with http_client(app, token="alice") as http:
        client = await make_client(http)
        task = await send(client, text_message("report please"))

    part = task.artifacts[0].parts[0]
    assert part.WhichOneof("content") == "raw"
    assert part.raw == b"pdf-bytes"
    assert part.filename == "r.pdf"


async def test_the_card_is_cacheable_and_revalidates(alice: Any) -> None:
    response = await alice.get("/.well-known/agent-card.json")

    assert "max-age" in response.headers["cache-control"]
    etag = response.headers["etag"]
    assert response.headers["last-modified"]

    revalidated = await alice.get(
        "/.well-known/agent-card.json", headers={"if-none-match": etag}
    )
    assert revalidated.status_code == 304


class ClaimContextBuilder(BearerContextBuilder):
    """A builder that puts an attested subject where every RPC can read it.

    Stands in for a security scheme carrying the subject, or an extension
    header. The point is the channel: it reaches the `ServerCallContext`, so
    `GetTask` — which has no `metadata` field at all — can scope by the same key
    the write path used.
    """

    def build(self, request: Any) -> Any:
        context = super().build(request)
        context.state["end_user"] = request.headers.get("x-end-user", "")
        return context


def subject_app(**overrides: Any) -> Any:
    return create_a2a_app(
        graph(),
        security_schemes=SECURITY_SCHEMES,
        context_builder=ClaimContextBuilder(),
        subject_resolver=state_subject("end_user"),
        **BASE_KWARGS,
        **overrides,
    )


async def test_a_subject_hook_isolates_users_behind_one_credential() -> None:
    """The case the principal alone cannot serve.

    A2A's caller is usually an agent holding one service credential for many end
    users, and 1.0 has no on-behalf-of field. Binding a context to the
    credential would give every user behind that peer the same conversation.
    """
    app = subject_app()

    async with http_client(
        app, token="peer-service", headers={"x-end-user": "ada"}
    ) as http:
        client = await make_client(http)
        first = await send(client, text_message("hello world"))

    # The same credential, a different end user, the first user's contextId.
    async with http_client(
        app, token="peer-service", headers={"x-end-user": "grace"}
    ) as http:
        client = await make_client(http)
        with pytest.raises(Exception, match="[Nn]ot found"):
            await send(client, text_message("hello world", context_id=first.context_id))


async def test_a_subject_written_on_send_is_readable_on_get_task() -> None:
    """The reason the subject cannot ride in message metadata.

    `metadata` is a field on `SendMessageRequest` and on none of `GetTaskRequest`,
    `ListTasksRequest` or `SubscribeToTaskRequest`. A store scoped by a key only
    half the calls supply either hides a task from its owner or hands it to
    someone else. Carried in the call context, both paths agree.
    """
    app = subject_app()

    async with http_client(app, token="peer", headers={"x-end-user": "ada"}) as http:
        client = await make_client(http)
        task = await send(client, text_message("hello world"))

        fetched = await client._transport.get_task(GetTaskRequest(id=task.id))
        assert artifact_text(fetched) == ["echo: hello world"]

    # And another subject behind the same credential cannot read it.
    async with http_client(app, token="peer", headers={"x-end-user": "grace"}) as http:
        client = await make_client(http)
        with pytest.raises(Exception, match="[Nn]ot found"):
            await client._transport.get_task(GetTaskRequest(id=task.id))


async def test_a_missing_subject_is_refused_not_guessed() -> None:
    app = subject_app()

    async with http_client(app, token="peer") as http:
        client = await make_client(http)
        with pytest.raises(Exception, match="subject"):
            await send(client, text_message("hello world"))


async def test_a_rotated_credential_keeps_the_conversation() -> None:
    """The subject is bound in the ownership store, never hashed into the thread.

    A renewed key or a renamed principal must not fork the conversation onto an
    empty thread — §6 of the proposal calls the thread key non-migratable.
    """
    app = subject_app()

    async with http_client(app, token="old-key", headers={"x-end-user": "ada"}) as http:
        client = await make_client(http)
        first = await send(client, text_message("remember: the sky is green"))

    async with http_client(
        app, token="rotated-key", headers={"x-end-user": "ada"}
    ) as http:
        client = await make_client(http)
        second = await send(
            client, text_message("what did I say", context_id=first.context_id)
        )

    assert "the sky is green" in artifact_text(second)[0]


def test_a_multi_replica_deployment_needs_a_conditional_store() -> None:
    """A lock in one process is not exclusion, and this says so at construction.

    Beside a second replica an `asyncio.Lock` produces no error — it produces a
    lost turn that both callers were told had succeeded.
    """
    with pytest.raises(A2AConfigurationError, match="conditionally"):
        build_app(multi_replica=True)


def test_a_single_replica_deployment_is_the_default() -> None:
    build_app()


def test_single_tenant_and_multi_replica_are_independent() -> None:
    """Two declared bits about two different facts, and all four pairings are real.

    They look alike — each is one boolean about something the process cannot
    observe — so it is worth pinning that neither implies the other. Auth
    terminated at a gateway in front of four replicas is both at once, and two
    replicas race on one conversation whether or not their callers are
    authenticated.
    """
    store = ConditionalStore(InMemoryStore())

    build_app(single_tenant=True, security_schemes=None, context_builder=None)
    build_app(task_store=store, multi_replica=True)
    build_app(
        single_tenant=True,
        security_schemes=None,
        context_builder=None,
        task_store=store,
        multi_replica=True,
    )
