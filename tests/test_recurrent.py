from dataclasses import FrozenInstanceError

import pytest

from adaptive_learning_substrate.recurrent import (
    RecurrentEdge,
    RecurrentEventGraph,
    build_experiment_000_graph,
)


def _episode(graph, cue: int, noise: tuple[int, ...] = (0, 1, 1, 0, 1, 0, 0, 1)):
    graph.begin_episode(f"cue-{cue}")
    graph.step({"cue": 1.0 if cue else -1.0})
    for bit in noise:
        graph.step({"noise": 1.0 if bit else -1.0})
    return graph.query()


def test_seeded_mask_has_frozen_protocol_degrees_and_is_deterministic() -> None:
    first = build_experiment_000_graph(17)
    second = build_experiment_000_graph(17, mode="no_trace")
    third = build_experiment_000_graph(18)

    assert first.topology_signature() == second.topology_signature()
    assert first.weights == second.weights
    assert first.topology_hash() == second.topology_hash()
    assert first.topology_hash() != third.topology_hash()

    recurrent = [edge for edge in first.edges if edge.kind == "recurrent"]
    inputs = [edge for edge in first.edges if edge.kind == "input"]
    outputs = [edge for edge in first.edges if edge.kind == "output"]
    for hidden in first.hidden_nodes:
        assert sum(edge.destination == hidden for edge in recurrent) == 8
        assert any(edge.source == hidden for edge in recurrent + outputs)
    for channel in first.input_nodes:
        assert sum(edge.source == channel for edge in inputs) == 8
    assert len(outputs) == 16
    assert all(edge.delay_ticks == 1 and edge.source != edge.destination for edge in first.edges)

    with pytest.raises(FrozenInstanceError):
        first.edges[0].destination = "changed"  # type: ignore[misc]


def test_recurrent_cycles_produce_strictly_earlier_immutable_event_parents() -> None:
    graph = build_experiment_000_graph(3)
    result = _episode(graph, 1)
    events = {event.event_id: event for event in graph.unit_events}

    assert result.event_id in events
    assert events[result.event_id].forced_output
    assert result.activation == pytest.approx(__import__("math").tanh(events[result.event_id].preactivation))
    assert any(edge.kind == "recurrent" for edge in graph.edges)
    traced = 0
    for event in events.values():
        for trace in event.edge_traces:
            parent = events[trace.parent_event_id]
            assert parent.step + 1 == event.step
            assert trace.created_step == event.step
            traced += 1
    assert traced > 0

    sample = next(event for event in events.values() if event.edge_traces)
    with pytest.raises(FrozenInstanceError):
        sample.activation = 9.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        sample.edge_traces[0].omission_effect = 9.0  # type: ignore[misc]


def test_full_credit_routes_by_parent_ids_while_no_trace_stops_after_output() -> None:
    full = build_experiment_000_graph(9, learning_rate=0.08, trace_decay=1.0)
    ablated = build_experiment_000_graph(
        9, mode="no_trace", learning_rate=0.08, trace_decay=1.0
    )
    full_query = _episode(full, 0)
    ablated_query = _episode(ablated, 0)
    assert full_query == ablated_query

    target = 1 - full.predict
    initial = full.weights
    full_updates = full.apply_supervised_credit(target)
    ablated_updates = ablated.apply_supervised_credit(target)

    assert full_updates
    assert ablated_updates
    assert {full.edges_by_id[update.edge_id].kind for update in full_updates} >= {
        "output",
        "recurrent",
    }
    assert {ablated.edges_by_id[update.edge_id].kind for update in ablated_updates} == {
        "output"
    }
    assert len(full_updates) > len(ablated_updates)
    assert any(full.weights[edge_id] != weight for edge_id, weight in initial.items())
    assert full.audit["topology_unchanged"] is True
    assert ablated.audit["topology_unchanged"] is True
    assert full.ledger["credit_event_touches"] > ablated.ledger["credit_event_touches"]
    events = full.unit_events_by_id
    assert all(events[packet.target_event_id].step <= full_query.tick for packet in full.credit_packets)


def test_compute_ledger_thresholds_clipping_and_defensive_weights() -> None:
    graph = build_experiment_000_graph(
        5,
        learning_rate=50.0,
        max_update=1e-5,
        weight_clip=2.0,
        hop_limit=2,
    )
    query = _episode(graph, 1)
    copied = graph.weights
    copied[next(iter(copied))] = 999.0
    assert 999.0 not in graph.weights.values()

    graph.apply_supervised_credit(1 - query.prediction)
    ledger = graph.ledger
    assert ledger["ticks"] == 12
    assert ledger["forward_edge_touches"] > 0
    assert ledger["omission_evaluations"] > 0
    assert ledger["credit_edge_touches"] > 0
    assert ledger["weight_write_touches"] > 0
    assert ledger["clipped_updates"] > 0
    assert ledger["credit_stop_hop_limit"] > 0
    assert ledger["nonfinite_values"] == 0
    assert graph.audit["ledger"] == ledger


def test_replay_is_bitwise_deterministic_and_episode_reset_preserves_weights() -> None:
    left = build_experiment_000_graph(27, learning_rate=0.03)
    right = build_experiment_000_graph(27, learning_rate=0.03)
    q_left = _episode(left, 1)
    q_right = _episode(right, 1)
    assert q_left == q_right
    target = 1 - q_left.prediction
    assert left.apply_supervised_credit(target) == right.apply_supervised_credit(target)
    assert left.weights_hash() == right.weights_hash()
    learned = left.weights
    left.begin_episode()
    assert left.weights == learned
    assert left.unit_events == ()
    assert left.credit_packets == ()
    with pytest.raises(RuntimeError, match=r"query\(\)"):
        _ = left.predict


def test_credit_after_root_trace_expiry_is_a_logged_no_update() -> None:
    graph = build_experiment_000_graph(31, trace_horizon=2)
    _episode(graph, 0)
    before = graph.weights
    graph.step({})
    graph.step({})
    graph.step({})
    assert graph.apply_supervised_credit(1) == ()
    assert graph.weights == before
    assert graph.ledger["credit_stop_expired"] == 1
    assert graph.credit_packets[-1].target_event_id == graph.audit["queried_output_event_id"]


def test_reconvergent_packets_aggregate_before_credit_minimum() -> None:
    """Two sub-threshold routes must combine before their shared parent stops."""

    edges = (
        RecurrentEdge("cue-to-parent", "cue", "parent", "input"),
        RecurrentEdge("parent-to-left", "parent", "left", "recurrent"),
        RecurrentEdge("parent-to-right", "parent", "right", "recurrent"),
        RecurrentEdge("left-to-output", "left", "output", "output"),
        RecurrentEdge("right-to-output", "right", "output", "output"),
    )
    graph = RecurrentEventGraph(
        input_nodes=("cue", "noise", "query"),
        hidden_nodes=("parent", "left", "right"),
        output_node="output",
        edges=edges,
        initial_weights={edge.edge_id: 1.0 for edge in edges},
        learning_rate=0.1,
        trace_decay=1.0,
        route_gain=0.9,
        credit_minimum=0.43,
        emit_threshold=0.0,
    )
    graph.begin_episode("reconvergent")
    graph.step({"cue": 1.0})
    query = graph.query()
    assert query.tick == 3

    updates = graph.apply_supervised_credit(0)
    updated_ids = {update.edge_id for update in updates}

    # Each left/right packet routed toward ``parent`` has magnitude about
    # 0.405 (< 0.43), while their aggregate is about 0.81 (> 0.43).
    assert "cue-to-parent" in updated_ids
    parent_packets = [
        packet.signal
        for packet in graph.credit_packets
        if graph.unit_events_by_id[packet.target_event_id].node == "parent"
    ]
    assert len(parent_packets) == 2
    assert all(abs(signal) < graph.credit_minimum for signal in parent_packets)
    assert abs(sum(parent_packets)) > graph.credit_minimum
