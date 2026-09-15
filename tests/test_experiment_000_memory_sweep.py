import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from adaptive_learning_substrate.experiment_000 import FROZEN_GRAPH_OPTIONS
from adaptive_learning_substrate.experiment_000_memory_probe import (
    recurrent_spectral_radius,
)
from adaptive_learning_substrate.experiment_000_memory_sweep import (
    REGISTERED_SEEDS,
    REGISTERED_TARGET_RADII,
    SPECTRAL_RADIUS_ABS_TOLERANCE,
    clone_with_recurrent_spectral_radius,
    main,
    run_memory_strength_sweep,
)
from adaptive_learning_substrate.recurrent import build_experiment_000_graph

_HYPERPARAMETERS = (
    "mode",
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
)


def _float64_bytes(value: float) -> bytes:
    return np.asarray([value], dtype="<f8").tobytes()


def _deterministic_hash(report: dict[str, object]) -> str:
    payload = dict(report)
    payload.pop("nondeterministic_provenance")
    payload.pop("deterministic_payload_sha256")
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_public_clone_scales_exactly_recurrent_edges_only() -> None:
    native = build_experiment_000_graph(
        50, mode="full", event_log_enabled=False, **FROZEN_GRAPH_OPTIONS
    )
    native_radius = recurrent_spectral_radius(native)
    target = 0.80
    factor = target / native_radius

    clone = clone_with_recurrent_spectral_radius(native, target)

    assert clone is not native
    assert clone.event_log is not native.event_log
    assert clone.edges == native.edges
    assert clone.topology_hash() == native.topology_hash()
    assert clone.input_nodes == native.input_nodes
    assert clone.hidden_nodes == native.hidden_nodes
    assert clone.output_node == native.output_node
    for field in _HYPERPARAMETERS:
        assert getattr(clone, field) == getattr(native, field)
    native_weights = native.weights
    clone_weights = clone.weights
    for edge in native.edges:
        if edge.kind == "recurrent":
            assert clone_weights[edge.edge_id] == float(
                native_weights[edge.edge_id] * factor
            )
            assert math.copysign(1.0, clone_weights[edge.edge_id]) == math.copysign(
                1.0, native_weights[edge.edge_id]
            )
        else:
            assert _float64_bytes(clone_weights[edge.edge_id]) == _float64_bytes(
                native_weights[edge.edge_id]
            )
    assert recurrent_spectral_radius(clone) == pytest.approx(
        target, abs=SPECTRAL_RADIUS_ABS_TOLERANCE
    )
    assert max(abs(value) for value in clone_weights.values()) <= clone.weight_clip
    assert native.weights_hash() != clone.weights_hash()


@pytest.mark.parametrize(
    ("seeds", "message"),
    (
        ((0, 51, 52, 53, 54), "reserved for development"),
        ((1000, 51, 52, 53, 54), "reserved for confirmatory"),
        ((55, 51, 52, 53, 54), "not one of the frozen memory-sweep seeds"),
        ((50,), "exact ordered seed list"),
    ),
)
def test_seed_guard_rejects_before_writing(
    tmp_path: Path, seeds: tuple[int, ...], message: str
) -> None:
    target = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match=message):
        run_memory_strength_sweep(
            seeds=seeds, pairs_per_split=2, output_path=target
        )

    assert not target.exists()
    assert not target.with_suffix(".sha256").exists()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"target_radii": (0.60, 0.80, 0.95, 0.70)}, "not one of"),
        ({"target_radii": (0.60,)}, "exact ordered target list"),
        ({"pairs_per_split": 4}, "smoke value 2 or full value 100"),
        ({"ridge_alpha": 1.0}, "frozen value"),
    ),
)
def test_protocol_argument_guard_rejects_before_writing(
    tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    target = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match=message):
        run_memory_strength_sweep(output_path=target, **kwargs)

    assert not target.exists()


def test_smoke_sweep_is_deterministic_no_learning_and_records_a1_assays() -> None:
    left = run_memory_strength_sweep(pairs_per_split=2)
    right = run_memory_strength_sweep(pairs_per_split=2)

    assert left["deterministic_payload_sha256"] == right[
        "deterministic_payload_sha256"
    ]
    assert left["deterministic_payload_sha256"] == _deterministic_hash(left)
    assert left["schema_version"] == "experiment-000-memory-strength-sweep-v1-a1"
    assert left["protocol_version"] == "memory-sweep-v1-a1"
    assert left["seeds"] == list(REGISTERED_SEEDS)
    assert left["target_spectral_radii"] == list(REGISTERED_TARGET_RADII)
    assert left["conditions"] == [
        "native",
        "rho_0_60",
        "rho_0_80",
        "rho_0_95",
        "rho_1_05",
    ]
    assert left["confirmatory_seed_count"] == 0
    assert left["status"] == "NO_SELECTION"
    assert left["selection"]["smoke_selection_forbidden"] is True
    assert left["selection"]["passing_conditions"] == []
    assert all(not row["gate"]["pass"] for row in left["aggregate_conditions"])
    assert all(
        left["integrity"][key]
        for key in (
            "all_seeds_custom_and_registered",
            "all_conditions_registered",
            "all_topologies_match_native_and_remain_unchanged",
            "all_input_output_weights_match_native",
            "all_condition_weights_unchanged",
            "all_credit_and_update_counts_zero",
            "all_finite",
            "all_spectral_radii_within_tolerance",
            "all_scaled_weights_within_clip",
        )
    )

    for seed_result in left["seed_results"]:
        assert seed_result["seed_role"] == "custom"
        assert seed_result["data_manifest"]["pair_sha256"]
        conditions = seed_result["condition_results"]
        native = conditions[0]
        native_io_hash = native["graph"]["initial_input_output_weights_sha256"]
        native_topology_hash = native["graph"]["topology_sha256"]
        for condition in conditions:
            graph = condition["graph"]
            invariants = condition["invariants"]
            assert graph["topology_sha256"] == native_topology_hash
            assert graph["initial_input_output_weights_sha256"] == native_io_hash
            assert graph["initial_weights_sha256"] == graph[
                "after_train_weights_sha256"
            ] == graph["after_eval_weights_sha256"] == graph["final_weights_sha256"]
            assert all(invariants.values())
            assert condition["ledger"] == {
                "credit_event_touches": 0,
                "credit_edge_touches": 0,
                "weight_write_touches": 0,
                "nonfinite_values": 0,
            }
            assert condition["graph"]["feature_dimension"] == 16
            assert len(condition["graph"]["feature_edge_ids"]) == 16
            assert len(condition["pair_rows"]["train"]) == 2
            assert len(condition["pair_rows"]["eval"]) == 2
            first_pair = condition["pair_rows"]["eval"][0]
            for cue in (0, 1):
                assert math.isfinite(first_pair[f"cue_{cue}_output_activation"])
                assert first_pair[f"cue_{cue}_emitted_unit_events"] > 0
                assert 0.0 <= first_pair[f"cue_{cue}_hidden_emission_density"] <= 1.0
                assert first_pair[f"cue_{cue}_forward_edge_touches"] > 0
            stability = condition["blank_tail_stability"]
            assert [row["cue_input"] for row in stability["cue_polarities"]] == [
                -1.0,
                1.0,
            ]
            for polarity in stability["cue_polarities"]:
                assert len(polarity["hidden_emissions_per_blank_tick"]) == 256
                assert len(polarity["hidden_counts_sha256"]) == 64
                assert 0.0 <= polarity["hidden_emission_density"] <= 1.0
            assert graph["recurrent_matrix"]["operator_2_norm"] > 0.0
            assert graph["recurrent_matrix"]["nonnormality_ratio"] == graph[
                "recurrent_matrix"
            ]["operator_2_norm_to_spectral_radius"]
            assert (
                graph["recurrent_matrix"][
                    "operator_2_norm_to_spectral_radius"
                ]
                > 1.0
            )
            if condition["condition"] != "native":
                assert graph["initial_recurrent_weights_sha256"] != native[
                    "graph"
                ]["initial_recurrent_weights_sha256"]
                for split in ("train", "eval"):
                    activity = condition["paired_vs_native"]["activity_by_split"][
                        split
                    ]
                    assert activity["total_emitted_event_ratio"] > 0.0
                    assert activity["maximum_matched_episode_event_count_ratio"] > 0.0
                    assert len(
                        activity["matched_episode_event_count_ratios"]
                    ) == 4


def test_cli_writes_report_and_matching_full_file_hash_sidecar(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "memory-sweep.json"

    status = main(
        [
            "--pairs-per-split",
            "2",
            "--output",
            str(target),
        ]
    )
    capsys.readouterr()

    assert status == 0
    report = json.loads(target.read_text(encoding="utf-8"))
    sidecar_path = target.with_suffix(".sha256")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert report["status"] == "NO_SELECTION"
    assert report["deterministic_payload_sha256"] == _deterministic_hash(report)
    assert sidecar == {
        "algorithm": "sha256",
        "report_file": target.name,
        "report_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
