"""Regression coverage for finite numeric state and atomic credit updates."""

import math
from dataclasses import replace

import pytest

from adaptive_learning_substrate.graph import Edge, FixedSparseEventGraph
from adaptive_learning_substrate.recurrent import RecurrentEdge, RecurrentEventGraph

NONFINITE_VALUES = (float("nan"), float("inf"), float("-inf"))


def _fixed_graph(**options: float) -> FixedSparseEventGraph:
    return FixedSparseEventGraph(
        input_nodes=("cue",),
        hidden_nodes=("hidden",),
        output_groups={"choice": ("output",)},
        edges=(
            Edge("cue-hidden", "cue", "hidden", 0.5),
            Edge("hidden-output", "hidden", "output", 0.5),
        ),
        **options,
    )


def _recurrent_graph(**options: float) -> RecurrentEventGraph:
    edges = (
        RecurrentEdge("cue-hidden", "cue", "hidden", "input"),
        RecurrentEdge("query-hidden", "query", "hidden", "input"),
        RecurrentEdge("hidden-output", "hidden", "output", "output"),
    )
    graph_options = {"trace_decay": 1.0, "emit_threshold": 0.0}
    graph_options.update(options)
    return RecurrentEventGraph(
        input_nodes=("cue", "noise", "query"),
        hidden_nodes=("hidden",),
        output_node="output",
        edges=edges,
        initial_weights={edge.edge_id: 1.0 for edge in edges},
        **graph_options,
    )


@pytest.mark.parametrize(
    "field",
    (
        "trace_decay",
        "activation_threshold",
        "weight_clip",
        "route_gain",
        "epsilon",
        "max_update",
        "trace_horizon",
    ),
)
@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_fixed_graph_rejects_every_nonfinite_scalar_config(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite scalar"):
        _fixed_graph(**{field: value})


@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_fixed_graph_rejects_nonfinite_initial_weights(value: float) -> None:
    with pytest.raises(ValueError, match="initial weight"):
        FixedSparseEventGraph(
            input_nodes=("cue",),
            output_groups={"choice": ("output",)},
            edges=(Edge("cue-output", "cue", "output", value),),
        )


def test_fixed_graph_normalizes_finite_initial_weight_scalars() -> None:
    graph = FixedSparseEventGraph(
        input_nodes=("cue",),
        output_groups={"choice": ("output",)},
        edges=(Edge("cue-output", "cue", "output", "0.5"),),  # type: ignore[arg-type]
    )

    assert graph.edges[0].weight == 0.5
    assert type(graph.edges[0].weight) is float


def test_fixed_graph_requires_positive_epsilon() -> None:
    with pytest.raises(ValueError, match="epsilon must be positive"):
        _fixed_graph(epsilon=0.0)


def test_fixed_forward_rejects_nonfinite_input_before_mutating_episode_state() -> None:
    graph = _fixed_graph()
    graph.begin_episode("finite-input")
    before_step = graph._step
    before_weights = graph.edge_weights()
    before_records = graph.event_log.records

    with pytest.raises(ValueError, match="input activities"):
        graph.forward({"cue": float("nan")})

    assert graph._step == before_step
    assert graph.edge_weights() == before_weights
    assert graph.event_log.records == before_records
    assert all(edge.last_trace_step == -1 for edge in graph.edges)


def test_fixed_forward_rejects_finite_overflow_without_mutating_episode_state() -> None:
    graph = FixedSparseEventGraph(
        input_nodes=("cue",),
        output_groups={"choice": ("output",)},
        edges=(Edge("cue-output", "cue", "output", 1e308),),
    )
    graph.begin_episode("finite-overflow")
    before_step = graph._step
    before_weights = graph.edge_weights()
    before_records = graph.event_log.records
    before_unit_events = graph.unit_events
    before_activations = dict(graph._last_activations)
    before_latest = dict(graph._latest_event_by_node)
    before_edge_state = tuple(
        (
            edge.participation_trace,
            edge.omission_trace,
            edge.weight_secant_trace,
            edge.source_secant_trace,
            edge.active_this_event,
            edge.last_event_id,
            edge.last_trace_step,
        )
        for edge in graph.edges
    )

    with pytest.raises(FloatingPointError, match="forward contribution"):
        graph.forward({"cue": 1e308})

    assert graph._step == before_step
    assert graph.edge_weights() == before_weights
    assert graph.event_log.records == before_records
    assert graph.unit_events == before_unit_events
    assert graph._last_activations == before_activations
    assert graph._latest_event_by_node == before_latest
    assert tuple(
        (
            edge.participation_trace,
            edge.omission_trace,
            edge.weight_secant_trace,
            edge.source_secant_trace,
            edge.active_this_event,
            edge.last_event_id,
            edge.last_trace_step,
        )
        for edge in graph.edges
    ) == before_edge_state


def test_fixed_ccf_stages_all_weights_and_events_until_routing_validates() -> None:
    graph = _fixed_graph()
    graph.begin_episode("atomic-ccf")
    graph.forward({"cue": 1.0})
    before_weights = graph.edge_weights()
    before_records = graph.event_log.records

    # A finite but tampered gain overflows only after the first candidate update
    # has been calculated. The candidate and its events must remain uncommitted.
    graph.route_gain = 1e308
    with pytest.raises(FloatingPointError, match="routed credit"):
        graph.apply_credit({"output": 1e308}, learning_rate=0.1)

    assert graph.edge_weights() == before_weights
    assert graph.event_log.records == before_records


def test_fixed_ccf_prevalidates_the_entire_credit_map() -> None:
    graph = _fixed_graph()
    graph.begin_episode("atomic-map")
    graph.forward({"cue": 1.0})
    before_weights = graph.edge_weights()
    before_records = graph.event_log.records

    with pytest.raises(ValueError, match="credit signals"):
        graph.apply_credit(
            {"output": 1.0, "hidden": float("nan")}, learning_rate=0.1
        )

    assert graph.edge_weights() == before_weights
    assert graph.event_log.records == before_records


def test_eligibility_update_is_atomic_when_a_later_trace_is_nonfinite() -> None:
    graph = FixedSparseEventGraph(
        input_nodes=("left", "right"),
        output_groups={"choice": ("output",)},
        edges=(
            Edge("left-output", "left", "output", 0.5),
            Edge("right-output", "right", "output", 0.5),
        ),
    )
    graph.begin_episode("atomic-eligibility")
    graph.forward({"left": 1.0, "right": 1.0})
    graph.edges[1].weight_secant_trace = float("nan")
    before_weights = graph.edge_weights()
    before_records = graph.event_log.records

    with pytest.raises(FloatingPointError, match="eligibility traces"):
        graph.apply_eligibility_credit({"output": 1.0}, learning_rate=0.1)

    assert graph.edge_weights() == before_weights
    assert graph.event_log.records == before_records


@pytest.mark.parametrize("learning_rate", (-0.1, float("nan"), float("inf")))
def test_eligibility_rejects_invalid_learning_rate_without_writes(
    learning_rate: float,
) -> None:
    graph = _fixed_graph()
    graph.begin_episode("eligibility-rate")
    graph.forward({"cue": 1.0})
    before_weights = graph.edge_weights()
    before_records = graph.event_log.records

    with pytest.raises(ValueError):
        graph.apply_eligibility_credit({"output": 1.0}, learning_rate=learning_rate)

    assert graph.edge_weights() == before_weights
    assert graph.event_log.records == before_records


def test_eligibility_prevalidates_unknown_nodes_without_partial_writes() -> None:
    graph = _fixed_graph()
    graph.begin_episode("eligibility-map")
    graph.forward({"cue": 1.0})
    before_weights = graph.edge_weights()
    before_records = graph.event_log.records

    with pytest.raises(KeyError, match="bad"):
        graph.apply_eligibility_credit(
            {"output": 1.0, "bad": 1.0}, learning_rate=0.1
        )

    assert graph.edge_weights() == before_weights
    assert graph.event_log.records == before_records


@pytest.mark.parametrize(
    "field",
    (
        "learning_rate",
        "trace_decay",
        "route_gain",
        "emit_threshold",
        "epsilon_weight",
        "epsilon_message",
        "epsilon_route",
        "credit_limit",
        "credit_minimum",
        "trace_horizon",
        "hop_limit",
        "max_update",
        "weight_clip",
    ),
)
@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_recurrent_graph_rejects_every_nonfinite_scalar_config(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError, match="finite scalar"):
        _recurrent_graph(**{field: value})


@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_recurrent_graph_rejects_nonfinite_initial_weights(value: float) -> None:
    edge = RecurrentEdge("cue-output", "cue", "output", "input")
    with pytest.raises(ValueError, match="initial weights"):
        RecurrentEventGraph(
            input_nodes=("cue",),
            hidden_nodes=(),
            output_node="output",
            edges=(edge,),
            initial_weights={edge.edge_id: value},
        )


@pytest.mark.parametrize("field", ("epsilon_weight", "epsilon_message", "epsilon_route"))
def test_recurrent_graph_requires_strictly_positive_epsilons(field: str) -> None:
    with pytest.raises(ValueError, match="epsilon values must be positive"):
        _recurrent_graph(**{field: 0.0})


def test_recurrent_graph_retains_exact_zero_threshold_guards() -> None:
    graph = _recurrent_graph(emit_threshold=0.0, credit_minimum=0.0)
    graph.begin_episode("zero-thresholds")
    step = graph.step({"cue": 0.0})
    assert step.inputs == (("cue", 0.0),)


def test_recurrent_credit_rejects_tampered_nan_rate_without_any_commit() -> None:
    graph = _recurrent_graph(learning_rate=0.1)
    graph.begin_episode("nan-rate")
    graph.step({"cue": 1.0})
    graph.query()
    graph.learning_rate = float("nan")
    before_weights = graph.weights
    before_records = graph.event_log.records
    before_packets = graph.credit_packets
    before_ledger = graph.ledger

    with pytest.raises(FloatingPointError, match="configuration"):
        graph.apply_supervised_credit(0)

    assert graph.weights == before_weights
    assert graph.event_log.records == before_records
    assert graph.credit_packets == before_packets
    assert graph.ledger == before_ledger


def test_recurrent_credit_stages_the_whole_transaction_before_commit() -> None:
    graph = _recurrent_graph(learning_rate=0.1)
    graph.begin_episode("causal-rollback")
    graph.step({"cue": 1.0})
    query = graph.query()
    root = graph._event_store[query.event_id]
    assert root.edge_traces

    # This finite corruption is discovered only after an update proposal has
    # been staged. It must roll back weights, logs, packets, ledger, and store.
    invalid_trace = replace(root.edge_traces[0], parent_event_id=root.event_id)
    graph._event_store[root.event_id] = replace(
        root, edge_traces=(invalid_trace, *root.edge_traces[1:])
    )
    before_weights = graph.weights
    before_records = graph.event_log.records
    before_packets = graph.credit_packets
    before_ledger = graph.ledger
    before_store = dict(graph._event_store)

    with pytest.raises(RuntimeError, match="non-earlier parent"):
        graph.apply_supervised_credit(0)

    assert graph.weights == before_weights
    assert graph.event_log.records == before_records
    assert graph.credit_packets == before_packets
    assert graph.ledger == before_ledger
    assert graph._event_store == before_store
    assert all(math.isfinite(weight) for weight in graph.weights.values())
