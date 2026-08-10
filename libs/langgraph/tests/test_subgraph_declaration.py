"""Tests for nodes that declare their subgraphs instead of being analysed.

A node whose `bound` implements `DeclaresSubgraphs` is taken at its word, which
reaches graphs no analysis of its code could find - e.g. one picked at runtime
by name. Declaring several keeps them all on the node, for drawing, but leaves
no way to resolve a namespace to one of them, which must not be papered over.
"""

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph._internal._runnable import RunnableCallable
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


class Declaring(RunnableCallable):
    """A node that reaches its graphs by name, not through a closure."""

    def __init__(self, graphs: list[Any]) -> None:
        super().__init__(self._func, name="dispatch", trace=False)
        self.graphs = graphs

    def _func(self, state: State) -> State:
        # always the last one, so declaration order cannot be mistaken for it
        return self.graphs[-1].invoke(state)

    def __langgraph_subgraphs__(self) -> list[Any]:
        return list(self.graphs)


def _parent(graphs: list[Any], checkpointer: Any = None) -> Any:
    builder = StateGraph(State)
    builder.add_node("dispatch", Declaring(graphs))
    builder.add_edge(START, "dispatch")
    return builder.compile(checkpointer=checkpointer)


def test_declaration_is_used_instead_of_analysis() -> None:
    one, two = _subgraph(), _subgraph()

    assert _parent([one, two]).nodes["dispatch"].subgraphs == [one, two]


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

    `dispatch2` does not resolve either, for the unrelated pre-existing reason
    that namespaces are matched by bare string prefix.
    """
    builder = StateGraph(State)
    builder.add_node("dispatch", Declaring([_subgraph(), _subgraph()]))
    builder.add_node("dispatch2", Declaring([_subgraph()]))
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
