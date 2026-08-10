"""Tests for `add_node(subgraphs=...)`, declared instead of analysed.

Whoever adds the node says which graphs it may call, which reaches graphs no
analysis of the node's code could find - e.g. one picked at runtime by name.
Declaring several keeps them all on the node, for drawing, but leaves no way to
resolve a namespace to one of them, which must not be papered over.
"""

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.graph import START, StateGraph
from langgraph.types import interrupt

pytestmark = pytest.mark.anyio


class State(TypedDict):
    value: str


def _subgraph(name: str = "inner", *, pause: bool = False) -> Any:
    builder = StateGraph(State)
    builder.add_node(
        name, lambda state: {"value": interrupt("pause")} if pause else state
    )
    builder.add_edge(START, name)
    return builder.compile()


def _dispatch(graphs: list[Any]) -> Any:
    """A node that reaches its graphs by name, not through a closure."""

    def dispatch(state: State) -> State:
        # always the last one, so declaration order cannot be mistaken for it
        return graphs[-1].invoke(state)

    return dispatch


def _parent(graphs: list[Any], checkpointer: Any = None) -> Any:
    builder = StateGraph(State)
    builder.add_node("dispatch", _dispatch(graphs), subgraphs=graphs)
    builder.add_edge(START, "dispatch")
    return builder.compile(checkpointer=checkpointer)


def test_declaration_reaches_graphs_the_analysis_cannot() -> None:
    one, two = _subgraph(), _subgraph()

    assert _parent([one, two]).nodes["dispatch"].subgraphs == [one, two]

    # the same node without the declaration: the graphs sit in a list, which
    # the analysis does not open
    builder = StateGraph(State)
    builder.add_node("dispatch", _dispatch([one, two]))
    builder.add_edge(START, "dispatch")

    assert builder.compile().nodes["dispatch"].subgraphs == []


def test_declaring_a_graph_that_disabled_checkpointing_records_nothing() -> None:
    """As the analysis does: it has no state, so its namespace would be a phantom."""
    builder = StateGraph(State)
    builder.add_node("inner", lambda state: state)
    builder.add_edge(START, "inner")
    stateless = builder.compile(checkpointer=False)

    assert _parent([stateless]).nodes["dispatch"].subgraphs == []


def test_undeclared_node_still_falls_back_to_analysis() -> None:
    sub = _subgraph()

    builder = StateGraph(State)
    builder.add_node("dispatch", lambda state: sub.invoke(state))
    builder.add_edge(START, "dispatch")

    assert builder.compile().nodes["dispatch"].subgraphs == [sub]


def test_declaration_reaches_every_node_spec_path() -> None:
    """`add_node` builds its spec three ways, by how the input schema is decided."""

    class Other(TypedDict):
        value: str

    def annotated(state: Other) -> Other:
        return state

    sub = _subgraph()
    for action, kwargs in (
        (lambda state: state, {}),
        (lambda state: state, {"input_schema": Other}),
        (annotated, {}),
    ):
        builder = StateGraph(State)
        builder.add_node("dispatch", action, subgraphs=[sub], **kwargs)
        builder.add_edge(START, "dispatch")

        assert builder.compile().nodes["dispatch"].subgraphs == [sub]


def test_declaration_survives_being_compiled_twice() -> None:
    """The spec outlives `add_node`, so a one-shot iterable must be snapshotted."""
    sub = _subgraph()

    builder = StateGraph(State)
    builder.add_node("dispatch", lambda state: state, subgraphs=(g for g in [sub]))
    builder.add_edge(START, "dispatch")

    assert builder.compile().nodes["dispatch"].subgraphs == [sub]
    assert builder.compile().nodes["dispatch"].subgraphs == [sub]


def test_declaring_something_that_is_not_a_graph_is_rejected() -> None:
    """Loudly, at compile - as `destinations` does for an unknown node name."""
    builder = StateGraph(State)
    builder.add_node("dispatch", lambda state: state, subgraphs=[object()])  # type: ignore[list-item]
    builder.add_edge(START, "dispatch")

    with pytest.raises(ValueError, match="not a compiled graph"):
        builder.compile()


def test_declaring_an_empty_list_is_not_the_same_as_declaring_nothing() -> None:
    """`[]` says "none", and so suppresses the analysis; `None` says nothing."""
    sub = _subgraph()

    builder = StateGraph(State)
    builder.add_node("dispatch", lambda state: sub.invoke(state), subgraphs=[])
    builder.add_edge(START, "dispatch")

    assert builder.compile().nodes["dispatch"].subgraphs == []


def test_declaration_survives_copy() -> None:
    one, two = _subgraph(), _subgraph()
    parent = _parent([one, two])

    assert parent.with_config(tags=["x"]).nodes["dispatch"].subgraphs == [one, two]
    assert parent.nodes["dispatch"].copy({}).subgraphs == [one, two]


def test_one_declared_graph_is_listed_and_resolvable() -> None:
    parent = _parent([_subgraph()], checkpointer=InMemorySaver())

    assert [name for name, _ in parent.get_subgraphs()] == ["dispatch"]
    assert "dispatch:inner" in parent.get_graph(xray=True).nodes

    state = parent.get_state(
        {"configurable": {"thread_id": "1", "checkpoint_ns": "dispatch:abc"}}
    )
    assert state.values == {}


def test_several_declared_graphs_refuse_to_resolve() -> None:
    parent = _parent([_subgraph(), _subgraph()], checkpointer=InMemorySaver())

    # drawing and listing still work, they do not address one graph in particular
    assert [name for name, _ in parent.get_subgraphs()] == ["dispatch"]
    assert "dispatch:inner" in parent.get_graph(xray=True).nodes

    with pytest.raises(ValueError, match="cannot resolve namespace dispatch"):
        parent.get_state(
            {"configurable": {"thread_id": "1", "checkpoint_ns": "dispatch:abc"}}
        )


async def test_several_declared_graphs_refuse_to_resolve_async() -> None:
    parent = _parent([_subgraph(), _subgraph()], checkpointer=InMemorySaver())

    with pytest.raises(ValueError, match="cannot resolve namespace dispatch"):
        await parent.aget_state(
            {"configurable": {"thread_id": "1", "checkpoint_ns": "dispatch:abc"}}
        )


def test_several_declared_graphs_refuse_a_write() -> None:
    """The write path is the one that must not reach the wrong graph."""
    parent = _parent([_subgraph(), _subgraph()], checkpointer=InMemorySaver())

    with pytest.raises(ValueError, match="cannot resolve namespace dispatch"):
        parent.update_state(
            {"configurable": {"thread_id": "1", "checkpoint_ns": "dispatch:abc"}},
            {"value": "x"},
        )


def test_refusal_does_not_fire_for_a_sibling_node() -> None:
    """Only the node a namespace addresses refuses - not one merely named like it.

    `dispatch2` does not resolve either, for an unrelated pre-existing reason:
    `dispatch` matches it by bare string prefix and, recursing, rebinds the
    namespace it is searching for, so `dispatch2` is never reached.
    """
    two, one = [_subgraph(), _subgraph()], [_subgraph()]
    builder = StateGraph(State)
    builder.add_node("dispatch", _dispatch(two), subgraphs=two)
    builder.add_node("dispatch2", _dispatch(one), subgraphs=one)
    builder.add_edge(START, "dispatch")
    parent = builder.compile(checkpointer=InMemorySaver())

    with pytest.raises(ValueError, match="Subgraph dispatch2 not found"):
        parent.get_state(
            {"configurable": {"thread_id": "1", "checkpoint_ns": "dispatch2:abc"}}
        )


def test_one_declared_graph_reports_a_paused_task() -> None:
    parent = _parent([_subgraph("a", pause=True)], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}
    parent.invoke({"value": "go"}, config)

    (task,) = parent.get_state(config, subgraphs=True).tasks
    assert task.state is not None and task.state.next == ("a",)


def test_several_declared_graphs_report_no_task_state() -> None:
    """Rather than the first declared graph's reading of another's checkpoint."""
    parent = _parent(
        [_subgraph("b", pause=True), _subgraph("a", pause=True)],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "1"}}
    parent.invoke({"value": "go"}, config)

    (task,) = parent.get_state(config, subgraphs=True).tasks
    assert task.state is None


async def test_several_declared_graphs_report_no_task_state_async() -> None:
    parent = _parent(
        [_subgraph("b", pause=True), _subgraph("a", pause=True)],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "1"}}
    await parent.ainvoke({"value": "go"}, config)

    (task,) = (await parent.aget_state(config, subgraphs=True)).tasks
    assert task.state is None
