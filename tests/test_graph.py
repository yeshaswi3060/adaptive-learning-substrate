import pytest

from adaptive_learning_substrate.events import EventLog
from adaptive_learning_substrate.graph import Edge, FixedSparseEventGraph


def make_two_choice_graph() -> FixedSparseEventGraph:
    return FixedSparseEventGraph(
        input_nodes=("cue",),
        output_groups={"choice": ("left", "right")},
        edges=(
            Edge("cue-left", "cue", "left", 0.2),
            Edge("cue-right", "cue", "right", 0.1),
        ),
        trace_decay=0.5,
        event_log=EventLog(),
    )


def test_participation_omission_and_delayed_local_update() -> None:
    graph = make_two_choice_graph()
    graph.begin_episode("episode-1")
    result = graph.forward({"cue": 1.0})
    assert result.selections == {"choice": "left"}
    left, right = graph.edges
    assert left.participation_trace == pytest.approx(1.0)
    assert left.omission_trace == pytest.approx(__import__("math").tanh(0.2))
    assert right.omission_trace == pytest.approx(__import__("math").tanh(0.1))
    graph.elapse(2)
    assert left.participation_trace == pytest.approx(0.25)
    assert right.omission_trace == pytest.approx(0.25 * __import__("math").tanh(0.1))
    before = graph.edge_weights()
    updates = graph.apply_credit({"right": 1.0, "left": -1.0}, learning_rate=0.4)
    assert {event.edge_id for event in updates} == {"cue-left", "cue-right"}
    assert graph.edge_weights()["cue-left"] < before["cue-left"]
    assert graph.edge_weights()["cue-right"] > before["cue-right"]
    assert {record["trace_kind"] for record in graph.event_log.by_type("ForwardEvent")} == {"participation"}
    right_trace = graph.unit_events[-1].edge_traces[0]
    assert right_trace.message_value == 1.0
    assert right_trace.omission_effect == pytest.approx(__import__("math").tanh(0.1))
    assert right_trace.weight_secant == pytest.approx(__import__("math").tanh(0.1) / 0.1)


def test_fixed_topology_rejects_cycles_and_defers_growth() -> None:
    with pytest.raises(ValueError, match="acyclic"):
        FixedSparseEventGraph(
            input_nodes=(),
            hidden_nodes=("a", "b"),
            output_groups={},
            edges=(Edge("a-b", "a", "b", 0.1), Edge("b-a", "b", "a", 0.1)),
        )
    graph = make_two_choice_graph()
    with pytest.raises(NotImplementedError, match="deferred"):
        graph.grow_edge()
    with pytest.raises(NotImplementedError, match="deferred"):
        graph.prune_edge()


def test_episode_reset_clears_traces_but_preserves_weights() -> None:
    graph = make_two_choice_graph()
    graph.begin_episode("one")
    graph.forward({"cue": 1.0})
    weights = graph.edge_weights()
    graph.begin_episode("two")
    assert all(edge.participation_trace == edge.omission_trace == 0.0 for edge in graph.edges)
    assert graph.edge_weights() == weights


def test_credit_after_trace_horizon_cannot_change_weights() -> None:
    graph = FixedSparseEventGraph(
        input_nodes=("cue",),
        output_groups={"choice": ("left", "right")},
        edges=(
            Edge("cue-left", "cue", "left", 0.2),
            Edge("cue-right", "cue", "right", 0.1),
        ),
        trace_decay=0.9,
        trace_horizon=2,
    )
    graph.begin_episode("expired")
    graph.forward({"cue": 1.0})
    before = graph.edge_weights()
    graph.elapse(3)
    assert graph.apply_credit({"right": 1.0}, learning_rate=0.5) == ()
    assert graph.edge_weights() == before
