from __future__ import annotations

import copy
import json
import math
import os
import stat
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Self

import numpy as np
import pytest

from adaptive_learning_substrate import experiment_000_online_trace_learning as learning
from adaptive_learning_substrate.experiment000_data import generate_stream_pair
from adaptive_learning_substrate.readout_trace_recurrent import clone_with_readout_trace
from adaptive_learning_substrate.recurrent import (
    RecurrentEventGraph,
    build_experiment_000_graph,
)


@pytest.fixture(autouse=True)
def _scratch_external_source_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """All tests bind a fresh custom-path freeze; official evidence stays absent."""

    freeze_path = tmp_path / "online-trace-learning.freeze.json"
    monkeypatch.setattr(learning, "FREEZE_RECORD_PATH", freeze_path)
    learning.create_pre_official_source_freeze(
        freeze_path, created_utc="2026-08-31T00:00:00Z"
    )
    monkeypatch.setenv(learning.FREEZE_ANCHOR_ENV, learning._file_sha256(freeze_path))
    yield
    if freeze_path.exists():
        freeze_path.chmod(stat.S_IWRITE | stat.S_IREAD)


def _condition(values: list[float]) -> dict[str, object]:
    return {
        "per_seed_eval_accuracy": values,
        "mean_eval_accuracy": float(np.mean(values)),
        "minimum_eval_accuracy": float(np.min(values)),
        "maximum_eval_accuracy": float(np.max(values)),
    }


def _synthetic_aggregate(*, successful: bool = True) -> dict[str, object]:
    count = 10
    if successful:
        values = {
            "d8_clean": {
                "trace_head_v1": [0.80] * count,
                "native_head": [0.65] * count,
                "trace_independent_label": [0.50] * count,
                "frozen_trace_head": [0.50] * count,
                "rand": [0.50] * count,
                "ridge_trace": [0.82] * count,
                "ridge_native": [0.65] * count,
            },
            "d4_clean": {
                "trace_head_v1": [0.80] * count,
                "native_head": [0.60] * count,
                "trace_independent_label": [0.50] * count,
                "frozen_trace_head": [0.50] * count,
                "rand": [0.50] * count,
                "ridge_trace": [0.82] * count,
                "ridge_native": [0.65] * count,
            },
            "d16_clean": {
                "trace_head_v1": [0.70] * count,
                "native_head": [0.55] * count,
                "trace_independent_label": [0.50] * count,
                "frozen_trace_head": [0.50] * count,
                "rand": [0.50] * count,
                "ridge_trace": [0.72] * count,
                "ridge_native": [0.60] * count,
            },
            "d8_cue_removed": {
                "trace_head_v1": [0.50] * count,
                "native_head": [0.50] * count,
                "trace_independent_label": [0.50] * count,
                "frozen_trace_head": [0.50] * count,
                "rand": [0.50] * count,
                "ridge_trace": [0.50] * count,
                "ridge_native": [0.50] * count,
            },
        }
    else:
        values = {
            cell: {condition: [0.50] * count for condition in learning.ORDERED_CONDITIONS}
            for cell in ("d8_clean", "d4_clean", "d16_clean", "d8_cue_removed")
        }
    return {
        "unit": "seed",
        "seed_count": count,
        "seed_order": list(learning.REGISTERED_SEEDS),
        "cells": {
            cell: {
                "conditions": {
                    condition: _condition(condition_values)
                    for condition, condition_values in conditions.items()
                }
            }
            for cell, conditions in values.items()
        },
    }


def test_frozen_cells_conditions_and_readiness_config() -> None:
    payload = learning.validate_protocol_config()
    assert learning.REGISTERED_SEEDS == tuple(range(80, 90))
    assert [(cell.name, cell.distractor_count, cell.train_examples) for cell in learning.FULL_CELLS] == [
        ("d8_clean", 8, 512),
        ("d4_clean", 4, 256),
        ("d16_clean", 16, 256),
        ("d8_cue_removed", 8, 256),
    ]
    assert payload["conditions"]["ordered"] == list(learning.ORDERED_CONDITIONS)
    assert payload["protocol"]["readiness_score_before"] == 21
    assert payload["protocol"]["readiness_score_after_pass"] == 25
    assert payload["protocol"]["readiness_score_after_fail"] == 21
    assert payload["compute_matching"]["scope"] == "total_mechanism"
    assert (
        payload["compute_matching"][
            "native_dummy_state_timestamp_history_replay_required"
        ]
        is True
    )
    assert (
        payload["compute_matching"][
            "native_dummy_query_feature_dot_tanh_replay_required"
        ]
        is True
    )
    assert payload["verification"]["registered_stream_reconstruction_required"] is True
    assert payload["freeze"]["recorded_before_official_metrics"] is True


@pytest.mark.parametrize("target", (-1.0, 1.0))
def test_online_update_is_exact_for_both_target_signs(target: float) -> None:
    theta = np.linspace(-0.25, 0.25, learning.FEATURE_DIMENSION)
    features = np.linspace(1.0, 2.5, learning.FEATURE_DIMENSION)
    updated, delta, activation, normalizer = learning.normalized_online_update(
        theta, features, target
    )
    expected_activation = math.tanh(math.fsum(theta * features))
    expected_normalizer = max(1.0, math.fsum(features * features))
    expected_delta = np.clip(
        0.5
        * (target - expected_activation)
        * (1.0 - expected_activation**2)
        * features
        / expected_normalizer,
        -0.05,
        0.05,
    )
    assert activation == expected_activation
    assert normalizer == expected_normalizer
    assert np.array_equal(delta, expected_delta)
    assert np.array_equal(updated, np.clip(theta + expected_delta, -3.0, 3.0))


def test_head_enforces_prediction_reveal_update_order_and_eval_is_read_only() -> None:
    head = learning.OnlineTanhHead()
    features = np.ones(learning.FEATURE_DIMENSION)
    with pytest.raises(RuntimeError, match="pending"):
        head.reveal_target_and_update(1.0)
    pending = head.predict_for_training(features)
    assert pending.activation == 0.0
    assert pending.prediction == 1.0
    with pytest.raises(RuntimeError, match="pending"):
        head.predict_for_training(features)
    evidence = head.reveal_target_and_update(-1.0)
    assert evidence["maximum_absolute_delta"] <= 0.05
    before = head.theta
    updates = head.ledger["update_count"]
    head.predict_for_evaluation(features)
    assert np.array_equal(head.theta, before)
    assert head.ledger["update_count"] == updates
    assert head.ledger["parameter_write_touches"] == learning.FEATURE_DIMENSION


def test_feature_and_target_guards_reject_wrong_shape_nan_and_nonbipolar() -> None:
    with pytest.raises(ValueError, match="16-dimensional"):
        learning.normalized_online_update(np.zeros(15), np.zeros(16), 1)
    features = np.zeros(16)
    features[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        learning.normalized_online_update(np.zeros(16), features, 1)
    with pytest.raises(ValueError, match="bipolar"):
        learning.normalized_online_update(np.zeros(16), np.zeros(16), 0.25)


def test_independent_labels_and_random_predictions_are_balanced_or_namespaced() -> None:
    left, left_manifest = learning.balanced_independent_labels(
        master_seed=900, cell_name="scratch_d8", count=10
    )
    right, right_manifest = learning.balanced_independent_labels(
        master_seed=900, cell_name="scratch_d8", count=10
    )
    assert np.array_equal(left, right)
    assert np.count_nonzero(left == -1.0) == 5
    assert np.count_nonzero(left == 1.0) == 5
    assert left_manifest == right_manifest
    random_train, random_manifest = learning.deterministic_random_predictions(
        master_seed=900, cell_name="scratch_d8", split="train", count=10
    )
    assert random_manifest["namespace"] != left_manifest["namespace"]
    assert random_manifest["seed"] != left_manifest["seed"]
    random_again, manifest_again = learning.deterministic_random_predictions(
        master_seed=900, cell_name="scratch_d8", split="train", count=10
    )
    assert np.array_equal(random_train, random_again)
    assert random_manifest == manifest_again


def test_ridge_fit_is_train_only_and_inactive_coordinates_remain_zero() -> None:
    features = np.zeros((6, learning.FEATURE_DIMENSION), dtype=np.float64)
    features[:, 0] = (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)
    targets = np.asarray((-1.0, -1.0, -1.0, 1.0, 1.0, 1.0))
    model = learning.fit_train_only_ridge(features, targets)
    diagnostic = model.diagnostic()
    assert diagnostic["fit_split"] == "train_only"
    assert diagnostic["evaluation_fit_count"] == 0
    assert diagnostic["active_feature_count"] == 1
    assert np.all(model.coefficients[1:] == 0.0)
    before = model.coefficients.copy()
    predictions = model.predictions(np.full((3, 16), 1000.0))
    assert predictions.shape == (3,)
    assert np.array_equal(model.coefficients, before)


def test_exact_sign_flip_enumerates_all_1024_and_holm_is_monotone() -> None:
    result = learning.exact_sign_flip_test([0.1] * 10)
    assert result["assignments"] == 1024
    assert result["extreme_assignments"] == 1
    assert result["p_value"] == 1 / 1024
    adjusted = learning.holm_adjust({"b": 0.02, "a": 0.01, "c": 0.04})
    assert adjusted == {"b": 0.04, "a": 0.03, "c": 0.04}


def test_bootstrap_is_fixed_and_uses_requested_replicate_count() -> None:
    first = learning.bootstrap_seed_mean_ci(
        [0.1, 0.2, 0.3], namespace="scratch-test", replicates=200
    )
    second = learning.bootstrap_seed_mean_ci(
        [0.1, 0.2, 0.3], namespace="scratch-test", replicates=200
    )
    assert first == second
    assert first["replicates"] == 200
    assert first["lower"] > 0.0


def test_registered_statistics_and_all_pass_gate_from_seed_level_inputs() -> None:
    aggregate = _synthetic_aggregate(successful=True)
    statistics = learning.compute_registered_statistics(aggregate)
    assert statistics["exact_sign_flip_assignments_per_contrast"] == 1024
    assert statistics["bootstrap_replicates_per_contrast"] == 20_000
    gates = learning.evaluate_registered_gates(
        aggregate, statistics, integrity_pass=True, eligible=True
    )
    assert gates["all_pass"] is True
    assert gates["failed_gate_ids"] == []
    assert gates["provisional_terminal_status"] == "LEARNING_RESULT_PASS"


def test_failed_gates_have_explicit_ids_and_smoke_is_nonselecting() -> None:
    passing_statistics = learning.compute_registered_statistics(
        _synthetic_aggregate(successful=True)
    )
    failed = learning.evaluate_registered_gates(
        _synthetic_aggregate(successful=False),
        passing_statistics,
        integrity_pass=False,
        eligible=True,
    )
    assert failed["all_pass"] is False
    assert "P01_D8_TRACE_MEAN_ACCURACY" in failed["failed_gate_ids"]
    assert "I01_ALL_INTEGRITY_GATES" in failed["failed_gate_ids"]
    assert failed["provisional_terminal_status"].startswith("LEARNING_RESULT_FAIL:")
    smoke = learning.evaluate_registered_gates(
        _synthetic_aggregate(), None, integrity_pass=True, eligible=False
    )
    assert smoke == {
        "eligible": False,
        "criteria": {},
        "all_pass": False,
        "failed_gate_ids": [],
        "provisional_terminal_status": "SMOKE_NONSELECTING",
    }


def test_reconstructed_integrity_false_cannot_be_promoted_to_i01_pass() -> None:
    integrity = {
        "confirmatory_seed_count": 0,
        "all_cell_integrity_pass": True,
        "global_episode_ids_disjoint": False,
    }
    assert learning._report_integrity_passes(integrity) is False
    aggregate = _synthetic_aggregate(successful=True)
    statistics = learning.compute_registered_statistics(aggregate)
    gates = learning.evaluate_registered_gates(
        aggregate,
        statistics,
        integrity_pass=learning._report_integrity_passes(integrity),
        eligible=True,
    )
    assert gates["criteria"]["I01_ALL_INTEGRITY_GATES"]["pass"] is False
    assert "I01_ALL_INTEGRITY_GATES" in gates["failed_gate_ids"]


def test_official_guards_require_exact_seed_types_order_cells_worker_and_path() -> None:
    with pytest.raises(TypeError, match="exact"):
        learning._validate_official_arguments(
            seeds=(*learning.REGISTERED_SEEDS[:-1], True),
            cells=learning.FULL_CELLS,
            ridge_alpha=0.001,
            workers=5,
            output_path=learning.PRIMARY_REPORT_PATH,
        )
    with pytest.raises(ValueError, match="exactly"):
        learning._validate_official_arguments(
            seeds=tuple(reversed(learning.REGISTERED_SEEDS)),
            cells=learning.FULL_CELLS,
            ridge_alpha=0.001,
            workers=5,
            output_path=learning.PRIMARY_REPORT_PATH,
        )
    with pytest.raises(ValueError, match="five"):
        learning._validate_official_arguments(
            seeds=learning.REGISTERED_SEEDS,
            cells=learning.FULL_CELLS,
            ridge_alpha=0.001,
            workers=4,
            output_path=learning.PRIMARY_REPORT_PATH,
        )
    resolved, cells, target = learning._validate_official_arguments(
        seeds=learning.REGISTERED_SEEDS,
        cells=learning.FULL_CELLS,
        ridge_alpha=0.001,
        workers=5,
        output_path=learning.PRIMARY_REPORT_PATH,
    )
    assert resolved == learning.REGISTERED_SEEDS
    assert cells == learning.FULL_CELLS
    assert target.resolve() == learning._project_path(learning.PRIMARY_REPORT_PATH).resolve()


def test_scratch_guards_reject_official_seed_and_canonical_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="official seeds"):
        learning._validate_scratch_arguments(
            seeds=(*learning.SCRATCH_SMOKE_SEEDS[:-1], 80),
            cells=learning.SMOKE_CELLS,
            trace_retention=0.5,
            ridge_alpha=0.001,
            output_path=tmp_path / "scratch.json",
        )
    with pytest.raises(ValueError, match="official artifact"):
        learning._validate_scratch_arguments(
            seeds=learning.SCRATCH_SMOKE_SEEDS,
            cells=learning.SMOKE_CELLS,
            trace_retention=0.5,
            ridge_alpha=0.001,
            output_path=learning.ARTIFACT_DIRECTORY / "scratch.json",
        )


def test_upstream_prerequisites_are_checked_in_a3_then_alignment_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptive_learning_substrate import (
        experiment_000_readout_trace_a3 as a3_module,
    )
    from adaptive_learning_substrate import (
        experiment_000_trace_head_alignment as alignment,
    )

    calls: list[str] = []
    a3 = {
        "status": "SELECTED:rho_0_90",
        "selected_condition": "rho_0_90",
        "retention_coefficient": 0.9,
    }
    aligned = {
        "status": "TRACE_HEAD_ALIGNMENT_PASS",
        "selected_condition": "rho_0_90",
        "trace_retention": 0.9,
    }

    def load_a3() -> dict[str, object]:
        calls.append("a3")
        return a3

    def load_alignment() -> dict[str, object]:
        calls.append("alignment")
        return aligned

    monkeypatch.setattr(a3_module, "load_verified_a3_selection", load_a3)
    monkeypatch.setattr(
        alignment, "load_verified_trace_head_alignment", load_alignment
    )
    result = learning.load_verified_learning_prerequisites()
    assert calls == ["a3", "alignment"]
    assert result["selected_condition"] == "rho_0_90"
    assert result["trace_retention"] == 0.9
    aligned["trace_retention"] = 0.75
    with pytest.raises(RuntimeError, match="does not bind"):
        learning.load_verified_learning_prerequisites()


def test_official_run_stops_at_missing_prerequisite_before_workers_or_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False
    target = learning._project_path(learning.PRIMARY_REPORT_PATH)
    before = target.read_bytes() if target.exists() else None

    def missing() -> dict[str, object]:
        raise RuntimeError("missing verified A3")

    def forbidden(**_: object) -> tuple[list[object], list[object]]:
        nonlocal called
        called = True
        raise AssertionError("official workers must not launch")

    monkeypatch.setattr(learning, "load_verified_learning_prerequisites", missing)
    monkeypatch.setattr(learning, "_ordered_seed_results", forbidden)
    with pytest.raises(RuntimeError, match="missing verified A3"):
        learning._run_online_trace_learning(
            seeds=learning.REGISTERED_SEEDS,
            cells=learning.FULL_CELLS,
            ridge_alpha=0.001,
            workers=5,
            output_path=learning.PRIMARY_REPORT_PATH,
            parent_blas=[{"num_threads": 1}],
        )
    assert called is False
    after = target.read_bytes() if target.exists() else None
    assert after == before


def test_official_report_transaction_cleans_json_and_sidecar_on_reopen_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "primary.json"
    freeze = {"freeze_file_sha256": "1" * 64}
    prerequisites = {
        "selected_condition": "scratch",
        "trace_retention": 0.75,
        "binding_sha256": "2" * 64,
    }
    source = {"files": [], "bundle_sha256": "3" * 64}
    monkeypatch.setattr(learning, "PRIMARY_REPORT_PATH", target)
    monkeypatch.setattr(learning, "RERUN_REPORT_PATH", tmp_path / "rerun.json")
    monkeypatch.setattr(learning, "validate_protocol_config", dict)
    monkeypatch.setattr(
        learning, "validate_pre_official_source_freeze", lambda: freeze
    )
    monkeypatch.setattr(
        learning, "load_verified_learning_prerequisites", lambda: prerequisites
    )
    monkeypatch.setattr(learning, "_source_manifest", lambda: source)
    monkeypatch.setattr(
        learning, "_ordered_seed_results", lambda **_: ([], [])
    )

    report: dict[str, object] = {"nondeterministic_provenance": {}}
    report["deterministic_payload_sha256"] = (
        learning._recompute_report_payload_sha256(report)
    )
    monkeypatch.setattr(
        learning,
        "_assemble_full_report",
        lambda **_: copy.deepcopy(report),
    )
    validation_calls = 0

    def fail_only_after_persistence(*_: object, **__: object) -> None:
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 2:
            raise RuntimeError("postwrite reconstruction failed")

    monkeypatch.setattr(learning, "validate_full_report", fail_only_after_persistence)
    with pytest.raises(RuntimeError, match="postwrite reconstruction failed"):
        learning._run_online_trace_learning(
            seeds=learning.REGISTERED_SEEDS,
            cells=learning.FULL_CELLS,
            ridge_alpha=learning.DEFAULT_RIDGE_ALPHA,
            workers=learning.ORDERED_CPU_WORKERS,
            output_path=target,
            parent_blas=[{"num_threads": 1}],
        )
    assert validation_calls == 2
    assert not target.exists()
    assert not target.with_suffix(".sha256").exists()


def test_one_real_scratch_observation_is_64_to_16_and_target_independent() -> None:
    episode = generate_stream_pair(
        900, episode_count=2, noise_events=8, required_role="custom"
    ).train.episodes[0]

    def run(target: int, *, cue_removed: bool) -> learning.EpisodeObservation:
        native = build_experiment_000_graph(
            900, event_log_enabled=False, **learning.FROZEN_GRAPH_OPTIONS
        )
        trace = clone_with_readout_trace(native, 0.75, event_log_enabled=False)
        changed = replace(episode, target=target)
        return learning.observe_episode(
            native,
            trace,
            changed,
            cue_removed=cue_removed,
            feature_edge_ids=learning._output_feature_edge_ids(native),
        )

    left = run(episode.target, cue_removed=False)
    right = run(1 - episode.target, cue_removed=False)
    assert left.trace_snapshot.shape == (64,)
    assert left.trace_features.shape == (16,)
    assert left.native_features.shape == (16,)
    assert np.array_equal(left.trace_snapshot, right.trace_snapshot)
    assert np.array_equal(left.trace_features, right.trace_features)
    removed = run(episode.target, cue_removed=True)
    assert removed.query_tick == 11
    assert math.isfinite(removed.trace_weighted_output)


def test_scratch_cell_has_total_compute_parity_and_independent_replay() -> None:
    cell = learning.SMOKE_CELLS[0]
    result = learning._run_cell(
        seed=900,
        cell=cell,
        trace_retention=0.75,
        ridge_alpha=0.001,
    )
    assert result["integrity"]["native_trace_compute_budget_matched"] is True
    assert result["integrity"]["native_dummy_trace_work_exact"] is True
    assert result["integrity"]["native_dummy_trace_state_exact"] is True
    assert result["integrity"]["native_dummy_trace_history_exact"] is True
    assert (
        result["integrity"]["native_dummy_trace_effective_snapshot_exact"] is True
    )
    assert result["integrity"]["native_dummy_trace_weighted_output_exact"] is True
    assert (
        result["graph"]["native_dummy_trace_ledger"]
        == result["graph"]["trace_ledger"]
    )
    for native_key, candidate_key in (
        ("native_dummy_trace_state_sha256", "candidate_trace_state_sha256"),
        ("native_dummy_trace_history_sha256", "candidate_trace_history_sha256"),
        (
            "native_dummy_trace_effective_snapshot_sha256",
            "candidate_trace_effective_snapshot_sha256",
        ),
        (
            "native_dummy_trace_weighted_outputs_sha256",
            "candidate_trace_weighted_outputs_sha256",
        ),
    ):
        assert result["graph"][native_key] == result["graph"][candidate_key]
    assert len(result["per_example_evidence"]["train"]) == 2
    learning._independently_reconstruct_cell(
        result,
        seed=900,
        spec=cell,
        trace_retention=0.75,
        ridge_alpha=0.001,
    )

    forged = copy.deepcopy(result)
    example = forged["per_example_evidence"]["train"][0]
    example["predictions"]["trace_head_v1"] *= -1.0
    example["evidence_sha256"] = learning._record_self_hash(
        example, "evidence_sha256"
    )
    evidence = forged["per_example_evidence"]
    evidence["evidence_sha256"] = learning._record_self_hash(
        evidence, "evidence_sha256"
    )
    with pytest.raises(RuntimeError, match="independent learning reconstruction"):
        learning._independently_reconstruct_cell(
            forged,
            seed=900,
            spec=cell,
            trace_retention=0.75,
            ridge_alpha=0.001,
        )

    missing_integrity_gate = copy.deepcopy(result)
    missing_integrity_gate["integrity"].pop("native_dummy_trace_history_exact")
    with pytest.raises(RuntimeError, match="independent learning reconstruction"):
        learning._independently_reconstruct_cell(
            missing_integrity_gate,
            seed=900,
            spec=cell,
            trace_retention=0.75,
            ridge_alpha=0.001,
        )


@pytest.mark.parametrize(
    ("old", "new"),
    (
        ("d8_trace_mean_accuracy_min = 0.75", "d8_trace_mean_accuracy_min = 0.99"),
        ("bootstrap_confidence = 0.95", "bootstrap_confidence = 0.60"),
        ('multiple_testing = "Holm"', 'multiple_testing = "none"'),
        ("separate_process_rerun_required = true", "separate_process_rerun_required = false"),
        (
            'determinism = "artifacts/experiment_000/online_trace_learning/DETERMINISM_VERIFICATION.json"',
            'determinism = "artifacts/experiment_000/online_trace_learning/changed.json"',
        ),
    ),
)
def test_exact_config_freeze_rejects_any_scientific_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    old: str,
    new: str,
) -> None:
    source = learning._project_path(learning.CONFIG_PATH).read_text(encoding="utf-8")
    assert old in source
    changed = tmp_path / "changed.toml"
    changed.write_text(source.replace(old, new, 1), encoding="utf-8")
    original_project_path = learning._project_path

    def routed(path: str | Path) -> Path:
        if Path(path) == learning.CONFIG_PATH:
            return changed
        return original_project_path(path)

    monkeypatch.setattr(learning, "_project_path", routed)
    with pytest.raises(RuntimeError, match="source freeze"):
        learning.validate_protocol_config()


def test_source_freeze_rejects_imported_scientific_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = Path("src/adaptive_learning_substrate/recurrent.py")
    source = learning._project_path(relative).read_bytes()
    changed = tmp_path / "recurrent.py"
    changed.write_bytes(source + b"\n# drift\n")
    original_project_path = learning._project_path

    def routed(path: str | Path) -> Path:
        if Path(path) == relative:
            return changed
        return original_project_path(path)

    monkeypatch.setattr(learning, "_project_path", routed)
    with pytest.raises(RuntimeError, match="source freeze"):
        learning.validate_pre_official_source_freeze()


def test_external_freeze_is_read_only_self_hashed_and_covers_import_closure() -> None:
    freeze_path = learning._project_path(learning.FREEZE_RECORD_PATH)
    record = learning.validate_pre_official_source_freeze()
    closure = set(record["scientific_import_closure"])
    assert freeze_path.is_file()
    assert learning._freeze_file_is_read_only(freeze_path)
    frozen_payload = dict(record)
    frozen_payload.pop("freeze_file_sha256")
    frozen_payload.pop("external_anchor_environment")
    assert record["record_sha256"] == learning._record_self_hash(
        frozen_payload, "record_sha256"
    )
    assert {
        "src/adaptive_learning_substrate/events.py",
        "src/adaptive_learning_substrate/experiment_000_readout_trace_a3.py",
        "src/adaptive_learning_substrate/experiment_000_slow_state.py",
        "src/adaptive_learning_substrate/experiment_000_trace_head_alignment.py",
        "src/adaptive_learning_substrate/slow_state_recurrent.py",
    }.issubset(closure)
    frozen_paths = {
        row["path"] for row in record["source_manifest"]["files"]
    }
    assert {
        "docs/EXPERIMENT_000_READOUT_TRACE_A3_PROTOCOL.md",
        "configs/experiment_000_readout_trace_a3.toml",
        "tests/test_experiment_000_readout_trace_a3.py",
        "docs/EXPERIMENT_000_TRACE_HEAD_ALIGNMENT_PROTOCOL.md",
        "configs/experiment_000_trace_head_alignment.toml",
        "tests/test_experiment_000_trace_head_alignment.py",
    }.issubset(frozen_paths)
    assert record["source_manifest"] == learning._source_manifest()


def test_external_freeze_rejects_mutability_and_rehashed_tampering() -> None:
    freeze_path = learning._project_path(learning.FREEZE_RECORD_PATH)
    freeze_path.chmod(stat.S_IWRITE | stat.S_IREAD)
    with pytest.raises(RuntimeError, match="absent or mutable"):
        learning.validate_pre_official_source_freeze()
    record = json.loads(freeze_path.read_text(encoding="utf-8"))
    record["created_utc"] = "2026-09-01T00:00:00Z"
    record["record_sha256"] = learning._record_self_hash(record, "record_sha256")
    freeze_path.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    freeze_path.chmod(stat.S_IREAD)
    # Re-self-hashing cannot move the independently supplied file-hash anchor.
    with pytest.raises(RuntimeError, match="external anchor changed"):
        learning.validate_pre_official_source_freeze()


@pytest.mark.parametrize("artifact_name", ("primary", "rerun", "terminal"))
def test_canonical_freeze_rejects_orphaned_official_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
) -> None:
    canonical_freeze = tmp_path / "canonical.freeze.json"
    paths = {
        "primary": tmp_path / "primary.json",
        "rerun": tmp_path / "rerun.json",
        "terminal": tmp_path / "terminal.json",
    }
    monkeypatch.setattr(learning, "CANONICAL_FREEZE_RECORD_PATH", canonical_freeze)
    monkeypatch.setattr(learning, "PRIMARY_REPORT_PATH", paths["primary"])
    monkeypatch.setattr(learning, "RERUN_REPORT_PATH", paths["rerun"])
    monkeypatch.setattr(
        learning, "DETERMINISM_VERIFICATION_PATH", paths["terminal"]
    )
    paths[artifact_name].with_suffix(".sha256").write_text(
        "orphaned official sidecar\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="official evidence exists"):
        learning.create_pre_official_source_freeze(
            canonical_freeze, created_utc="2026-08-31T00:00:00Z"
        )
    assert not canonical_freeze.exists()


def test_structural_hash_changes_when_only_plastic_flag_changes() -> None:
    native = build_experiment_000_graph(
        900, event_log_enabled=False, **learning.FROZEN_GRAPH_OPTIONS
    )
    changed_edges = (replace(native.edges[0], plastic=not native.edges[0].plastic), *native.edges[1:])
    changed = RecurrentEventGraph(
        input_nodes=native.input_nodes,
        hidden_nodes=native.hidden_nodes,
        output_node=native.output_node,
        edges=changed_edges,
        initial_weights=native.weights,
    )
    assert learning._structural_mask_sha256(native) != learning._structural_mask_sha256(changed)


def test_five_worker_order_contract_via_stub_without_spawning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeExecutor:
        def __init__(self, *, max_workers: int, mp_context: object) -> None:
            assert max_workers == 5
            assert mp_context is not None

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def map(
            self,
            function: object,
            jobs: list[
                tuple[
                    int,
                    tuple[learning.CellSpec, ...],
                    float,
                    float,
                    dict[str, object],
                ]
            ],
            *,
            chunksize: int,
        ) -> list[dict[str, object]]:
            del function
            assert chunksize == 1
            rows: list[dict[str, object]] = []
            for index, job in enumerate(jobs):
                coordinator = job[4]
                identity = {
                    "seed": job[0],
                    "process_id": 100 + index % 5,
                    "parent_process_id": coordinator["process_id"],
                    "process_instance_token": f"{100 + index % 5:064x}",
                    "process_started_utc": f"2026-01-0{1 + index % 5}T00:00:00Z",
                    "job_started_utc": f"2026-02-01T00:00:{index:02d}Z",
                    "coordinator_binding_sha256": coordinator["binding_sha256"],
                }
                rows.append(
                    {
                        "deterministic": {"seed": job[0]},
                        "runtime": {
                            **identity,
                            "worker_binding_sha256": learning._execution_binding(
                                identity
                            ),
                            "runtime_seconds": 0.0,
                            "blas": [{"num_threads": 1}],
                        },
                    }
                )
            return rows

    monkeypatch.setattr(learning, "ProcessPoolExecutor", FakeExecutor)
    deterministic, runtime = learning._ordered_seed_results(
        seeds=learning.REGISTERED_SEEDS,
        cells=learning.FULL_CELLS,
        trace_retention=0.9,
        ridge_alpha=0.001,
        workers=5,
    )
    assert [row["seed"] for row in deterministic] == list(learning.REGISTERED_SEEDS)
    assert len({row["process_id"] for row in runtime}) == 5


def test_scratch_smoke_runs_frozen_tiny_matrix_persists_and_never_selects(
    tmp_path: Path,
) -> None:
    output = tmp_path / "scratch-smoke.json"
    report = learning.run_scratch_smoke(
        seeds=learning.SCRATCH_SMOKE_SEEDS,
        cells=learning.SMOKE_CELLS,
        trace_retention=0.75,
        output_path=output,
    )
    assert report["status"] == "SMOKE_NONSELECTING"
    assert report["official_seeds_executed"] is False
    assert report["gates"]["eligible"] is False
    assert all(row["all_cell_integrity_pass"] is True for row in report["seed_results"])
    for seed_result in report["seed_results"]:
        assert all(seed_result["integrity"].values())
        roots = [
            row["cell_data_root"]["derived_seed"]
            for row in seed_result["cell_results"]
        ]
        assert len(roots) == len(set(roots)) == 4
    sidecar = output.with_suffix(".sha256")
    assert output.is_file() and sidecar.is_file()
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert sidecar_payload["report_file"] == output.name
    reopened, hashes = learning._read_report_with_sidecar(output)
    assert reopened == report
    assert hashes["report_file_sha256"] == sidecar_payload["report_sha256"]


def test_sidecar_tamper_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "scratch-sidecar.json"
    report = {"status": "SMOKE_NONSELECTING", "value": 1}
    learning._write_report_and_sidecar(output, report)
    output.write_text('{"status":"tampered"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="sidecar"):
        learning._read_report_with_sidecar(output)


def test_sidecar_loader_hashes_and_parses_one_byte_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "snapshot.json"
    original = {"status": "BOUND", "value": 1}
    learning._write_report_and_sidecar(output, original)
    path_hasher = learning._file_sha256

    def obsolete_path_hash_then_replace(path: str | Path) -> str:
        digest = path_hasher(path)
        if Path(path).resolve() == output.resolve():
            output.write_text('{"status":"REPLACED","value":2}\n', encoding="utf-8")
        return digest

    # A path-hash-then-reopen implementation would trigger this injected race
    # and return unbound replacement JSON.  The loader must not call it: both
    # hash and JSON come from its single report-byte snapshot.
    monkeypatch.setattr(learning, "_file_sha256", obsolete_path_hash_then_replace)
    reopened, hashes = learning._read_report_with_sidecar(output)
    assert reopened == original
    assert output.read_text(encoding="utf-8") != '{"status":"REPLACED","value":2}\n'
    assert hashes["report_file_sha256"] == path_hasher(output)


def _synthetic_provenance(
    *,
    role: str,
    identity_offset: int,
    primary_binding: dict[str, str] | None,
) -> dict[str, object]:
    return {
        "artifact_role": role,
        "process_id": 10_000 + identity_offset,
        "process_instance_token": f"{1 + identity_offset:064x}",
        "process_started_utc": f"2026-03-{1 + identity_offset:02d}T00:00:00Z",
        "run_instance_token": f"{101 + identity_offset:064x}",
        "run_started_utc": f"2026-04-{1 + identity_offset:02d}T00:00:00Z",
        "rerun_primary_artifact_binding": primary_binding,
        "worker_runtime": [
            {
                "process_id": 20_000 + identity_offset * 100 + worker,
                "process_instance_token": (
                    f"{1001 + identity_offset * 100 + worker:064x}"
                ),
                "process_started_utc": (
                    f"2026-05-{1 + identity_offset:02d}T00:00:{worker:02d}Z"
                ),
                "job_started_utc": (
                    f"2026-06-{1 + identity_offset:02d}T00:00:{worker:02d}Z"
                ),
            }
            for worker in range(5)
        ],
    }


def test_copied_primary_provenance_cannot_claim_a_fresh_rerun() -> None:
    binding = {
        "primary_report_file_sha256": "a" * 64,
        "primary_sidecar_file_sha256": "b" * 64,
    }
    primary = _synthetic_provenance(
        role="primary", identity_offset=0, primary_binding=None
    )
    copied = copy.deepcopy(primary)
    copied.update(
        {
            "artifact_role": "rerun",
            "process_id": 99_999,
            "process_instance_token": "c" * 64,
            "process_started_utc": "2026-07-01T00:00:00Z",
            "run_instance_token": "d" * 64,
            "run_started_utc": "2026-07-01T00:00:01Z",
            "rerun_primary_artifact_binding": binding,
        }
    )
    assert not learning._separate_execution_evidence_is_fresh(
        primary, copied, expected_primary_binding=binding
    )
    fresh = _synthetic_provenance(
        role="rerun", identity_offset=1, primary_binding=binding
    )
    # Even fully rewritten, self-consistent JSON is insufficient without a
    # verifier-observed fresh subprocess reconstruction.
    assert not learning._separate_execution_evidence_is_fresh(
        primary, fresh, expected_primary_binding=binding
    )
    challenge = {"verified": True}
    challenge["orchestration_payload_sha256"] = learning._record_self_hash(
        challenge, "orchestration_payload_sha256"
    )
    assert learning._separate_execution_evidence_is_fresh(
        primary,
        fresh,
        expected_primary_binding=binding,
        orchestrated_reconstruction=challenge,
    )


def test_orchestrated_challenges_bind_fresh_nonces_and_observed_process_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = (tmp_path / "first.json").resolve()
    second = (tmp_path / "second.json").resolve()
    bindings = {
        str(first): {
            "report_file_sha256": "1" * 64,
            "sidecar_file_sha256": "2" * 64,
        },
        str(second): {
            "report_file_sha256": "3" * 64,
            "sidecar_file_sha256": "4" * 64,
        },
    }
    payload_hashes = {str(first): "5" * 64, str(second): "5" * 64}
    reconstruction_hashes = {str(first): "6" * 64, str(second): "6" * 64}

    class FakeProcess:
        next_pid = 70_000

        def __init__(self, command: tuple[str, ...], **_: object) -> None:
            self.pid = FakeProcess.next_pid
            FakeProcess.next_pid += 1
            self.returncode = 0
            report = str(Path(command[command.index("--report") + 1]).resolve())
            nonce = command[command.index("--nonce") + 1]
            row: dict[str, object] = {
                "schema_version": learning.CHALLENGE_SCHEMA_VERSION,
                "nonce": nonce,
                "process_id": self.pid,
                "process_instance_token": f"{self.pid:064x}",
                "process_started_utc": (
                    f"2026-08-01T00:00:{self.pid - 70_000:02d}Z"
                ),
                "report_path": report,
                **bindings[report],
                "deterministic_payload_sha256": payload_hashes[report],
                "reconstructed_evidence_sha256": reconstruction_hashes[report],
            }
            row["challenge_payload_sha256"] = learning._record_self_hash(
                row, "challenge_payload_sha256"
            )
            self.stdout = json.dumps(row)

        def communicate(self) -> tuple[str, str]:
            return self.stdout, ""

        def poll(self) -> int:
            return 0

        def terminate(self) -> None:
            raise AssertionError("completed fake process must not be terminated")

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(learning.subprocess, "Popen", FakeProcess)
    result = learning._run_orchestrated_reconstruction_challenges(
        first,
        second,
        first_files=bindings[str(first)],
        second_files=bindings[str(second)],
        deterministic_payload_sha256es=("5" * 64, "5" * 64),
    )
    assert result["verified"] is True
    assert len({row["nonce"] for row in result["challenges"]}) == 2
    assert len({row["process_id"] for row in result["challenges"]}) == 2

    # Each child is checked against its own bound report hash.  A scientifically
    # valid but non-deterministic pair becomes a failed D02 result rather than a
    # verifier exception which would prevent a terminal FAIL artifact.
    payload_hashes[str(second)] = "7" * 64
    reconstruction_hashes[str(second)] = "8" * 64
    mismatch = learning._run_orchestrated_reconstruction_challenges(
        first,
        second,
        first_files=bindings[str(first)],
        second_files=bindings[str(second)],
        deterministic_payload_sha256es=("5" * 64, "7" * 64),
    )
    assert mismatch["verified"] is False
    assert len(mismatch["challenges"]) == 2


def _write_synthetic_deterministic_pair(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, object]]:
    prerequisites: dict[str, object] = {"binding_sha256": "e" * 64}
    primary_path = tmp_path / "primary.json"
    rerun_path = tmp_path / "rerun.json"
    common: dict[str, object] = {
        "provisional_terminal_status": "LEARNING_RESULT_FAIL:TEST_GATE",
        "prerequisites": prerequisites,
        "payload": {"same": True},
    }
    primary = {
        **copy.deepcopy(common),
        "nondeterministic_provenance": _synthetic_provenance(
            role="primary", identity_offset=0, primary_binding=None
        ),
    }
    primary["deterministic_payload_sha256"] = (
        learning._recompute_report_payload_sha256(primary)
    )
    learning._write_report_and_sidecar(primary_path, primary)
    primary_binding = {
        "primary_report_file_sha256": learning._file_sha256(primary_path),
        "primary_sidecar_file_sha256": learning._file_sha256(
            primary_path.with_suffix(".sha256")
        ),
    }
    rerun = {
        **copy.deepcopy(common),
        "nondeterministic_provenance": _synthetic_provenance(
            role="rerun", identity_offset=1, primary_binding=primary_binding
        ),
    }
    rerun["deterministic_payload_sha256"] = learning._recompute_report_payload_sha256(
        rerun
    )
    learning._write_report_and_sidecar(rerun_path, rerun)
    return primary_path, rerun_path, prerequisites


def test_terminal_verification_has_sidecar_and_postwrite_input_recheck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary, rerun, prerequisites = _write_synthetic_deterministic_pair(tmp_path)
    terminal = tmp_path / "verification.json"
    source = {"bundle_sha256": "f" * 64, "files": []}
    freeze = {"record_sha256": "0" * 64}
    monkeypatch.setattr(learning, "PRIMARY_REPORT_PATH", primary)
    monkeypatch.setattr(learning, "RERUN_REPORT_PATH", rerun)
    monkeypatch.setattr(learning, "DETERMINISM_VERIFICATION_PATH", terminal)
    monkeypatch.setattr(learning, "validate_protocol_config", dict)
    monkeypatch.setattr(
        learning, "validate_pre_official_source_freeze", lambda: freeze
    )
    monkeypatch.setattr(
        learning, "load_verified_learning_prerequisites", lambda: prerequisites
    )
    monkeypatch.setattr(learning, "_source_manifest", lambda: source)
    monkeypatch.setattr(learning, "validate_full_report", lambda *args, **kwargs: None)

    def reconstructed(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        evidence: dict[str, object] = {"verified": True}
        evidence["orchestration_payload_sha256"] = learning._record_self_hash(
            evidence, "orchestration_payload_sha256"
        )
        return evidence

    monkeypatch.setattr(
        learning, "_run_orchestrated_reconstruction_challenges", reconstructed
    )
    result = learning.verify_deterministic_full_runs(
        primary, rerun, output_path=terminal
    )
    assert result["status"].startswith("LEARNING_RESULT_FAIL:")
    assert terminal.is_file() and terminal.with_suffix(".sha256").is_file()
    reopened, _ = learning._read_report_with_sidecar(terminal)
    assert reopened == result

    second_terminal = tmp_path / "verification-toctou.json"
    monkeypatch.setattr(learning, "DETERMINISM_VERIFICATION_PATH", second_terminal)
    original_write = learning._write_report_and_sidecar

    def mutate_input_after_terminal_write(
        target: str | Path, record: dict[str, object]
    ) -> tuple[Path, Path]:
        written = original_write(target, record)
        if Path(target).resolve() == second_terminal.resolve():
            primary.write_text('{"changed":true}\n', encoding="utf-8")
        return written

    monkeypatch.setattr(
        learning, "_write_report_and_sidecar", mutate_input_after_terminal_write
    )
    with pytest.raises(RuntimeError, match="sidecar"):
        learning.verify_deterministic_full_runs(
            primary, rerun, output_path=second_terminal
        )
    assert not second_terminal.exists()
    assert not second_terminal.with_suffix(".sha256").exists()


def test_full_shaped_report_uses_real_five_processes_and_reconstructs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch_seeds = tuple(range(900, 910))
    monkeypatch.setattr(learning, "REGISTERED_SEEDS", scratch_seeds)
    monkeypatch.setattr(learning, "FULL_CELLS", learning.SMOKE_CELLS)
    coordinator: dict[str, object] = {
        "process_id": os.getpid(),
        "process_instance_token": learning._PROCESS_INSTANCE_TOKEN,
        "process_started_utc": learning._PROCESS_STARTED_UTC,
        "run_instance_token": "1" * 64,
        "run_started_utc": "2026-08-01T00:00:00Z",
    }
    coordinator["binding_sha256"] = learning._execution_binding(coordinator)
    seed_results, worker_runtime = learning._ordered_seed_results(
        seeds=scratch_seeds,
        cells=learning.SMOKE_CELLS,
        trace_retention=0.75,
        ridge_alpha=0.001,
        workers=5,
        coordinator=coordinator,
    )
    assert len({row["process_id"] for row in worker_runtime}) == 5
    assert len({row["process_instance_token"] for row in worker_runtime}) == 5
    prerequisites = {
        "selected_condition": "scratch_fixed_retention",
        "trace_retention": 0.75,
        "binding_sha256": "2" * 64,
    }
    source = learning._source_manifest()
    freeze = learning.validate_pre_official_source_freeze()
    report = learning._assemble_full_report(
        seed_results=seed_results,
        worker_runtime=worker_runtime,
        prerequisites=prerequisites,
        source_manifest=source,
        source_freeze_record=freeze,
        workers=5,
        run_started_utc="2026-08-01T00:00:00Z",
        runtime_seconds=1.0,
        artifact_role="primary",
        primary_binding=None,
        coordinator=coordinator,
    )
    report["nondeterministic_provenance"]["parent_blas"] = [
        {"num_threads": 1}
    ]
    provenance = report["nondeterministic_provenance"]
    provenance["provenance_sha256"] = learning._record_self_hash(
        provenance, "provenance_sha256"
    )
    report["deterministic_payload_sha256"] = (
        learning._recompute_report_payload_sha256(report)
    )
    learning.validate_full_report(
        report,
        expected_prerequisites=prerequisites,
        current_source_manifest=source,
    )

    forged = copy.deepcopy(report)
    metric = forged["seed_results"][0]["cell_results"][0]["condition_results"][
        "trace_head_v1"
    ]["eval"]
    metric["correct"] = (metric["correct"] + 1) % (metric["examples"] + 1)
    metric["accuracy"] = metric["correct"] / metric["examples"]
    metric["predictions_sha256"] = "3" * 64
    forged["deterministic_payload_sha256"] = (
        learning._recompute_report_payload_sha256(forged)
    )
    with pytest.raises(RuntimeError, match="independent learning reconstruction"):
        learning.validate_full_report(
            forged,
            expected_prerequisites=prerequisites,
            current_source_manifest=source,
        )
