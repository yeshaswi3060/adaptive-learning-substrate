import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from adaptive_learning_substrate.events import EventLog
from adaptive_learning_substrate.experiment_000_memory_probe import _PairedSplit
from adaptive_learning_substrate.experiment_000_slow_state import (
    FROZEN_RECURRENT_SOURCE_SHA256,
    REGISTERED_RETENTIONS,
    REGISTERED_SEEDS,
    _structural_mask_sha256,
    condition_name,
    frozen_source_integrity,
    main,
    run_slow_state_experiment,
    run_zero_retention_equivalence,
    strict_ridge_readout,
    verify_deterministic_full_runs,
)
from adaptive_learning_substrate.frozen_history import (
    validate_frozen_recurrent_snapshot,
)
from adaptive_learning_substrate.recurrent import (
    RecurrentEdge,
    RecurrentEventGraph,
    build_experiment_000_graph,
)
from adaptive_learning_substrate.slow_state_recurrent import (
    SlowStateRecurrentEventGraph,
    clone_with_local_state_retention,
)


def _micrograph(
    retention: float,
    *,
    emit_threshold: float = 1e-3,
    input_weight: float = 0.2,
) -> SlowStateRecurrentEventGraph:
    edges = (
        RecurrentEdge("input:x->h0", "x", "h0", "input"),
        RecurrentEdge("output:h0->output", "h0", "output", "output"),
    )
    return SlowStateRecurrentEventGraph(
        local_state_retention=retention,
        input_nodes=("x", "query"),
        hidden_nodes=("h0", "h1"),
        output_node="output",
        edges=edges,
        initial_weights={"input:x->h0": input_weight, "output:h0->output": 0.7},
        emit_threshold=emit_threshold,
        event_log=EventLog(enabled=False),
    )


def _touch_tuple(graph: SlowStateRecurrentEventGraph) -> tuple[int, ...]:
    ledger = graph.slow_state_ledger
    return tuple(
        ledger[field]
        for field in (
            "local_state_read_touches",
            "local_state_decay_touches",
            "local_state_write_touches",
        )
    )


def test_native_source_and_ccf_bundle_are_frozen_byte_exact() -> None:
    integrity = frozen_source_integrity()
    assert integrity["recurrent_source_sha256"] != FROZEN_RECURRENT_SOURCE_SHA256
    assert integrity["recurrent_source_matches"] is False
    assert integrity["credit_method_bundle_matches"] is False
    assert integrity["ccf_document_matches"] is True
    assert validate_frozen_recurrent_snapshot()["sha256"] == (
        FROZEN_RECURRENT_SOURCE_SHA256
    )


def test_clone_preserves_every_explicit_edge_weight_and_native_hyperparameter() -> None:
    native = build_experiment_000_graph(60, event_log_enabled=False)
    clone = clone_with_local_state_retention(native, 0.75, event_log_enabled=False)

    assert clone is not native
    assert clone.edges == native.edges
    assert clone.topology_signature() == native.topology_signature()
    assert clone.weights == native.weights
    assert clone.weights_hash() == native.weights_hash()
    assert clone.local_state_retention == 0.75
    with pytest.raises(AttributeError):
        clone.local_state_retention = 0.5  # type: ignore[misc]


def test_zero_retention_fast_path_has_no_state_and_is_step_exact() -> None:
    native = build_experiment_000_graph(60, event_log_enabled=False)
    zero = clone_with_local_state_retention(native, 0.0, event_log_enabled=False)
    native.begin_episode("same")
    zero.begin_episode("same")

    for inputs in (
        {"cue": -1.0},
        {"noise": 1.0},
        {"noise": -1.0},
        {},
    ):
        assert native.step(inputs) == zero.step(inputs)
        assert native.unit_events == zero.unit_events
        assert native.ledger == zero.ledger

    assert zero.local_state_snapshot == {}
    assert zero.effective_local_states() == {
        node: 0.0 for node in zero.hidden_nodes
    }
    state = zero.slow_state_ledger
    assert state["local_state_reset_touches"] == 0
    assert state["local_state_observation_touches"] == 0
    assert _touch_tuple(zero) == (0, 0, 0)


def test_nonzero_candidate_keeps_source_and_output_dynamics_native() -> None:
    candidate = _micrograph(0.75, input_weight=1.0)
    native = RecurrentEventGraph(
        input_nodes=candidate.input_nodes,
        hidden_nodes=candidate.hidden_nodes,
        output_node=candidate.output_node,
        edges=candidate.edges,
        initial_weights=candidate.weights,
        event_log=EventLog(enabled=False),
    )
    candidate.begin_episode("same")
    native.begin_episode("same")
    # On the first hidden processing retained state is defined as zero, so the
    # source, hidden response, scheduled message, and subsequent output event
    # all match native exactly. Only the candidate's private state ledger differs.
    for inputs in ({"x": 0.2}, {}, {}):
        assert candidate.step(inputs) == native.step(inputs)
        assert candidate.unit_events == native.unit_events
    output_events = [event for event in candidate.unit_events if event.node == "output"]
    assert len(output_events) == 1
    assert output_events[0] == next(
        event for event in native.unit_events if event.node == "output"
    )


def test_registered_smoke_examples_have_exact_zero_retention_equivalence() -> None:
    report = run_zero_retention_equivalence()
    assert report["pass"] is True
    assert report["seeds"] == list(REGISTERED_SEEDS)
    assert report["counterfactual_episode_count"] == 40
    assert report["compared_step_result_count"] == 480
    assert report["compared_public_episode_count"] == 40
    assert report["all_local_state_touches_zero"] is True
    assert len(report["rows_sha256"]) == 64


@pytest.mark.parametrize("first_value", (0.2, -0.2))
def test_positive_and_negative_state_decay_over_one_tick(first_value: float) -> None:
    graph = _micrograph(0.5, input_weight=1.0)
    graph.begin_episode("one-tick")
    graph.step({"x": first_value})
    graph.step({"x": 0.1})  # first hidden process and next source event
    first_state = math.tanh(first_value)
    assert graph.local_state_snapshot["h0"] == (first_state, 1)
    graph.step({})  # second hidden process at tick 2: gap=1
    expected = math.tanh(0.1 + 0.5 * first_state)
    state, tick = graph.local_state_snapshot["h0"]
    assert tick == 2
    assert state == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert _touch_tuple(graph) == (2, 2, 2)


def test_multitick_decay_blank_inactivity_and_exact_reset_accounting() -> None:
    graph = _micrograph(0.5, input_weight=1.0)
    graph.begin_episode("multi")
    assert graph.slow_state_ledger["local_state_reset_touches"] == 2
    graph.step({"x": 0.2})
    graph.step({})
    first_state = math.tanh(0.2)
    before = graph.local_state_snapshot
    touches_before = graph.slow_state_ledger
    graph.step({"x": 0.1})  # h0 inactive; output can process
    assert graph.local_state_snapshot == before
    for field in (
        "local_state_read_touches",
        "local_state_decay_touches",
        "local_state_write_touches",
    ):
        assert graph.slow_state_ledger[field] == touches_before[field]
    graph.step({})  # h0 gap=2 (tick 1 -> tick 3)
    expected = math.tanh(0.1 + (0.5**2) * first_state)
    assert graph.local_state_snapshot["h0"][0] == pytest.approx(
        expected, rel=1e-12, abs=1e-12
    )

    graph.begin_episode("reset")
    assert graph.local_state_snapshot == {"h0": (0.0, None), "h1": (0.0, None)}
    assert all(math.copysign(1.0, value) == 1.0 for value, _ in graph.local_state_snapshot.values())
    assert graph.slow_state_ledger["local_state_reset_touches"] == 4


def test_below_threshold_activation_writes_and_omission_holds_retention_fixed() -> None:
    quiet = _micrograph(0.5, emit_threshold=2.0, input_weight=1.0)
    quiet.begin_episode("quiet")
    quiet.step({"x": 0.2})
    result = quiet.step({})
    assert not any(":h0:" in event_id for event_id in result.emitted_event_ids)
    assert quiet.local_state_snapshot["h0"][0] == math.tanh(0.2)
    assert _touch_tuple(quiet) == (1, 1, 1)
    assert quiet.slow_activation_records[-1].emitted is False

    graph = _micrograph(0.5, emit_threshold=1e-12, input_weight=1.0)
    graph.begin_episode("omission")
    graph.step({"x": 0.2})
    graph.step({"x": 0.1})
    prior = math.tanh(0.2)
    graph.step({})
    event = [event for event in graph.unit_events if event.node == "h0"][-1]
    retained = 0.5 * prior
    expected_effect = math.tanh(0.1 + retained) - math.tanh(retained)
    assert len(event.edge_traces) == 1
    assert event.edge_traces[0].omission_effect == pytest.approx(
        expected_effect, rel=1e-12, abs=1e-12
    )


def test_effective_state_observation_is_nonmutating_and_separately_counted() -> None:
    graph = _micrograph(0.5, input_weight=1.0)
    graph.begin_episode("observe")
    graph.step({"x": 0.2})
    graph.step({})
    snapshot = graph.local_state_snapshot
    base_ledger = graph.ledger
    slow_before = graph.slow_state_ledger
    effective = graph.effective_local_states(5)
    assert effective["h0"] == pytest.approx((0.5**4) * math.tanh(0.2))
    assert graph.local_state_snapshot == snapshot
    assert graph.ledger == base_ledger
    slow_after = graph.slow_state_ledger
    assert slow_after["local_state_observation_touches"] - slow_before[
        "local_state_observation_touches"
    ] == 2
    assert _touch_tuple(graph) == (1, 1, 1)


def test_structural_hash_covers_nodes_and_plastic_flag() -> None:
    graph = _micrograph(0.5)
    changed_edges = (replace(graph.edges[0], plastic=False), *graph.edges[1:])
    changed = SlowStateRecurrentEventGraph(
        local_state_retention=0.5,
        input_nodes=graph.input_nodes,
        hidden_nodes=graph.hidden_nodes,
        output_node=graph.output_node,
        edges=changed_edges,
        initial_weights=graph.weights,
        event_log=EventLog(enabled=False),
    )
    assert graph.topology_hash() == changed.topology_hash()  # native omits plastic
    assert _structural_mask_sha256(graph) != _structural_mask_sha256(changed)


def test_strict_ridge_excludes_zero_variance_coordinate_and_embeds_zero() -> None:
    train = _PairedSplit(
        cue_zero_features=np.asarray([[1.0, -1.0], [1.0, -0.5]]),
        cue_one_features=np.asarray([[1.0, 0.5], [1.0, 1.0]]),
        cue_zero_outputs=np.asarray([-0.1, -0.2]),
        cue_one_outputs=np.asarray([0.1, 0.2]),
    )
    evaluation = _PairedSplit(
        cue_zero_features=np.asarray([[1000.0, -1.0], [-1000.0, -0.5]]),
        cue_one_features=np.asarray([[999.0, 0.5], [-999.0, 1.0]]),
        cue_zero_outputs=np.asarray([-0.1, -0.2]),
        cue_one_outputs=np.asarray([0.1, 0.2]),
    )
    result = strict_ridge_readout(train, evaluation, alpha=1e-3)
    assert result["active_feature_indices"] == [1]
    assert result["inactive_feature_count"] == 1
    assert result["embedded_coefficients"][0] == 0.0
    assert result["inactive_coordinates_excluded_from_solve"] is True
    assert result["eval_accuracy"] == 1.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"seeds": (60,)}, "exact ordered"),
        ({"seeds": (0, 61, 62, 63, 64)}, "reserved for development"),
        ({"retentions": (0.25, 0.50)}, "exact ordered retention"),
        (
            {"retentions": (0.2500000000001, 0.50, 0.75, 0.90, 0.95)},
            "not one of the frozen",
        ),
        ({"retentions": (0.25, 0.50, 0.75, 0.90, 0.90)}, "unique"),
        ({"pairs_per_split": 3}, "smoke value 2 or full value 100"),
        ({"ridge_alpha": 0.1}, "frozen value"),
    ),
)
def test_argument_guards_reject_before_output(
    tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    target = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError, match=message):
        run_slow_state_experiment(output_path=target, **kwargs)
    assert not target.exists()
    assert not target.with_suffix(".sha256").exists()


def test_cli_requires_explicit_seed_and_retention_lists(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main([])
    captured = capsys.readouterr()
    assert error.value.code == 2
    assert "--seeds" in captured.err
    assert "--retentions" in captured.err


def _fake_full_report(value: int = 1) -> dict[str, object]:
    report: dict[str, object] = {
        "run_kind": "full",
        "status": "NO_SELECTION",
        "integrity": {"native_control_valid": True},
        "deterministic_value": value,
    }
    canonical = json.dumps(
        report, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    )
    report["deterministic_payload_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    report["nondeterministic_provenance"] = {"runtime_seconds": 1.0}
    return report


def test_determinism_verifier_recomputes_self_hash_and_detects_tampering() -> None:
    left = _fake_full_report()
    right = _fake_full_report()
    valid = verify_deterministic_full_runs(left, right)
    assert valid["status"] == "NO_SELECTION"
    assert valid["first_self_hash_valid"] is True
    assert valid["deterministic_payloads_match"] is True

    right["deterministic_value"] = 2
    invalid = verify_deterministic_full_runs(left, right)
    assert invalid["status"] == "NONDETERMINISTIC_INVALID"
    assert invalid["second_self_hash_valid"] is False


def test_registered_identifiers_are_exact() -> None:
    assert REGISTERED_SEEDS == (60, 61, 62, 63, 64)
    assert REGISTERED_RETENTIONS == (0.25, 0.50, 0.75, 0.90, 0.95)
    assert [condition_name(None), *(condition_name(x) for x in REGISTERED_RETENTIONS)] == [
        "native",
        "lambda_0_25",
        "lambda_0_50",
        "lambda_0_75",
        "lambda_0_90",
        "lambda_0_95",
    ]
