"""Tests for the preregistered readout-decoupled local-trace experiment."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from adaptive_learning_substrate.experiment000_data import (
    SeedRole,
    generate_stream_pair,
)
from adaptive_learning_substrate.experiment_000 import (
    FROZEN_GRAPH_OPTIONS,
    forward_episode,
)
from adaptive_learning_substrate.experiment_000_memory_probe import (
    extract_query_features,
)
from adaptive_learning_substrate.experiment_000_readout_trace import (
    FROZEN_CONFIG_SHA256,
    FROZEN_PROTOCOL_SHA256,
    REGISTERED_RETENTIONS,
    REGISTERED_SEEDS,
    _validate_arguments,
    condition_name,
    frozen_source_integrity,
    run_readout_trace_experiment,
    run_zero_retention_equivalence,
)
from adaptive_learning_substrate.frozen_history import (
    validate_frozen_recurrent_snapshot,
)
from adaptive_learning_substrate.readout_trace_recurrent import (
    clone_with_readout_trace,
)
from adaptive_learning_substrate.recurrent import build_experiment_000_graph


def _native(seed: int):
    return build_experiment_000_graph(
        seed, mode="full", event_log_enabled=False, **FROZEN_GRAPH_OPTIONS
    )


def _episode(seed: int, cue: int):
    pair = generate_stream_pair(
        seed, episode_count=2, noise_events=8, required_role=SeedRole.CUSTOM
    )
    base = pair.train.episodes[0]
    return replace(base, episode_id=f"{base.episode_id}-t{cue}", cue=cue, target=cue)


def test_frozen_protocol_and_config_hashes_match_and_live_repair_fails_closed():
    integrity = frozen_source_integrity()
    assert integrity["recurrent_source_matches"] is False
    assert integrity["credit_method_bundle_matches"] is False
    snapshot = validate_frozen_recurrent_snapshot()
    assert snapshot["sha256"] == integrity["expected_recurrent_source_sha256"]
    assert integrity["ccf_document_matches"]
    assert integrity["protocol_matches"], integrity["protocol_sha256"]
    assert integrity["config_matches"], integrity["config_sha256"]
    assert integrity["expected_protocol_sha256"] == FROZEN_PROTOCOL_SHA256
    assert integrity["expected_config_sha256"] == FROZEN_CONFIG_SHA256


@pytest.mark.parametrize("seed", REGISTERED_SEEDS)
def test_zero_retention_matches_native_bit_for_bit(seed: int):
    native = _native(seed)
    zero = clone_with_readout_trace(native, 0.0, event_log_enabled=False)
    for cue in (0, 1):
        episode = _episode(seed, cue)
        qn = forward_episode(native, episode)
        qz = forward_episode(zero, episode)
        assert (qn.activation, qn.prediction) == (qz.activation, qz.prediction)
        assert native.unit_events == zero.unit_events
        assert native.ledger == zero.ledger
        assert extract_query_features(native, qn) == extract_query_features(zero, qz)
    ledger = zero.trace_ledger
    for field in (
        "local_trace_read_touches",
        "local_trace_decay_touches",
        "local_trace_write_touches",
        "local_trace_reset_touches",
        "local_trace_observation_touches",
    ):
        assert ledger[field] == 0


def test_bounded_ema_stays_in_unit_interval_and_decays_exactly():
    native = _native(65)
    graph = clone_with_readout_trace(native, 0.9, event_log_enabled=False)
    forward_episode(graph, _episode(65, 1))
    records = graph.trace_activation_records
    assert records
    assert all(abs(r.trace_after) <= 1.0 for r in records)
    assert all(abs(r.activation) <= 1.0 for r in records)
    # Exact lazy decay: a read at a later tick equals rho**gap * stored value.
    effective = graph.effective_trace(graph._tick + 5)
    snapshot = graph.trace_snapshot
    for node, value in effective.items():
        stored, tick = snapshot[node]
        if tick is None:
            assert value == 0.0
        else:
            gap = (graph._tick + 5) - tick
            assert math.isclose(value, (0.9**gap) * stored, rel_tol=0, abs_tol=1e-15)


def test_effective_trace_observation_is_non_mutating():
    native = _native(66)
    graph = clone_with_readout_trace(native, 0.95, event_log_enabled=False)
    forward_episode(graph, _episode(66, 0))
    before = graph.trace_snapshot
    graph.effective_trace(graph._tick)
    graph.effective_trace(graph._tick + 10)
    assert graph.trace_snapshot == before
    assert graph.trace_ledger["local_trace_observation_touches"] == 2 * len(
        graph.hidden_nodes
    )


def test_recurrent_dynamics_are_native_for_every_retention():
    native = _native(67)
    reference = _native(67)
    for retention in REGISTERED_RETENTIONS:
        graph = clone_with_readout_trace(native, retention, event_log_enabled=False)
        ref = build_experiment_000_graph(
            67, mode="full", event_log_enabled=False, **FROZEN_GRAPH_OPTIONS
        )
        for cue in (0, 1):
            episode = _episode(67, cue)
            forward_episode(graph, episode)
            forward_episode(ref, episode)
        assert graph.unit_events == ref.unit_events
        assert graph.ledger == ref.ledger
    del reference


def test_argument_guards_reject_before_output(tmp_path):
    with pytest.raises(ValueError):
        _validate_arguments(
            seeds=(60, 61, 62, 63, 64),
            retentions=REGISTERED_RETENTIONS,
            pairs_per_split=2,
            ridge_alpha=1e-3,
        )
    with pytest.raises(ValueError):
        _validate_arguments(
            seeds=REGISTERED_SEEDS,
            retentions=(0.25, 0.5, 0.75, 0.9),
            pairs_per_split=2,
            ridge_alpha=1e-3,
        )
    with pytest.raises(ValueError):
        _validate_arguments(
            seeds=REGISTERED_SEEDS,
            retentions=REGISTERED_RETENTIONS,
            pairs_per_split=7,
            ridge_alpha=1e-3,
        )


def test_condition_names_are_stable():
    assert condition_name(None) == "native"
    assert condition_name(0.25) == "rho_0_25"
    assert condition_name(0.95) == "rho_0_95"


def test_zero_retention_equivalence_helper_passes():
    result = run_zero_retention_equivalence()
    assert result["pass"]
    assert result["compared_counterfactual_episodes"] == 40


def test_superseded_smoke_refuses_post_freeze_source_drift(tmp_path):
    with pytest.raises(RuntimeError, match="frozen source integrity failed"):
        run_readout_trace_experiment(
            pairs_per_split=2, output_path=tmp_path / "a.json"
        )


def test_superseded_smoke_does_not_write_before_source_rejection(tmp_path):
    target = tmp_path / "s.json"
    with pytest.raises(RuntimeError, match="frozen source integrity failed"):
        run_readout_trace_experiment(pairs_per_split=2, output_path=target)
    assert not target.exists()
