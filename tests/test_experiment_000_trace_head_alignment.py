from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np
import pytest

from adaptive_learning_substrate import experiment_000_trace_head_alignment as alignment


def _a3_selection_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    selected = "rho_0_75"
    selection: dict[str, Any] = {
        "schema_version": "experiment-000-readout-trace-a3-selection-binding-v1",
        "status": f"SELECTED:{selected}",
        "selected_condition": selected,
        "retention_coefficient": 0.75,
        "protocol_version": alignment.A3_REQUIRED_PROTOCOL_VERSION,
        "source_manifest_sha256": "1" * 64,
        "freeze_record_sha256": "2" * 64,
        "primary": {
            "path": str(alignment.A3_PRIMARY_REPORT_PATH).replace("\\", "/"),
            "report_file_sha256": "3" * 64,
            "sidecar_file_sha256": "4" * 64,
            "deterministic_payload_sha256": "5" * 64,
        },
        "rerun": {
            "path": str(alignment.A3_RERUN_REPORT_PATH).replace("\\", "/"),
            "report_file_sha256": "6" * 64,
            "sidecar_file_sha256": "7" * 64,
            "deterministic_payload_sha256": "5" * 64,
        },
        "determinism": {
            "path": str(alignment.A3_DETERMINISM_VERIFICATION_PATH).replace(
                "\\", "/"
            ),
            "file_sha256": "8" * 64,
            "verification_payload_sha256": "9" * 64,
        },
    }
    phases = [
        {"name": name, "artifacts": [{"synthetic": True}]}
        for name in alignment.A3_REQUIRED_PHASE_ORDER
    ]
    phase_record: dict[str, Any] = {
        "schema_version": "experiment-000-readout-trace-a3-phases-v1",
        "protocol_version": alignment.A3_REQUIRED_PROTOCOL_VERSION,
        "phase_order": list(alignment.A3_REQUIRED_PHASE_ORDER),
        "phases": phases,
    }
    phase_record["phase_sequence_payload_sha256"] = alignment._record_self_hash(
        phase_record, "phase_sequence_payload_sha256"
    )
    phase_info = {"record": phase_record, "file_sha256": "a" * 64}
    return selection, phase_info


def _alignment_process_pair(
    *, status: str = "TRACE_HEAD_ALIGNMENT_PASS"
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], dict[str, str]]:
    primary_files = {
        "report_file_sha256": "a" * 64,
        "sidecar_file_sha256": "b" * 64,
    }
    rerun_files = {
        "report_file_sha256": "c" * 64,
        "sidecar_file_sha256": "d" * 64,
    }
    primary: dict[str, Any] = {
        "status": status,
        "scientific_payload": [1.0, 2.0],
        "nondeterministic_provenance": {
            "artifact_role": "primary",
            "process_id": 2001,
            "process_instance_token": "e" * 64,
            "run_started_utc": "2026-09-03T02:03:04.000001Z",
            "primary_prerequisite": None,
        },
    }
    primary["deterministic_payload_sha256"] = (
        alignment._recompute_report_payload_sha256(primary)
    )
    rerun = copy.deepcopy(primary)
    rerun["nondeterministic_provenance"] = {
        "artifact_role": "rerun",
        "process_id": 2002,
        "process_instance_token": "f" * 64,
        "run_started_utc": "2026-09-03T02:03:05.000002Z",
        "primary_prerequisite": dict(primary_files),
    }
    return primary, rerun, primary_files, rerun_files


def _scratch_full_report(
    *, seeds: tuple[int, ...], episodes_per_seed: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    selection, phase_info = _a3_selection_fixture()
    binding = alignment._validated_a3_selection_binding(selection, phase_info)
    manifest = alignment._source_manifest()
    core = alignment._run_alignment_core(
        seeds=seeds,
        episodes_per_seed=episodes_per_seed,
        retention=0.75,
    )
    report: dict[str, Any] = {
        "schema_version": alignment.SCHEMA_VERSION,
        "protocol_name": alignment.PROTOCOL_NAME,
        "protocol_version": alignment.PROTOCOL_VERSION,
        "run_kind": "full",
        "claim_scope": "standalone_head_alignment_only_not_recurrent_temporal_credit",
        "seeds": list(seeds),
        "episodes_per_seed": episodes_per_seed,
        "noise_events": alignment.NOISE_EVENTS,
        "feature_dimension": alignment.FEATURE_DIMENSION,
        "target_encoding": {"cue_0": -1.0, "cue_1": 1.0},
        "head_contract": alignment._head_contract(),
        "finite_difference_contract": alignment._finite_difference_contract(),
        "a3_selection_binding": binding,
        "source_manifest_start": manifest,
        "source_manifest_end": manifest,
        **core,
        "status": (
            "TRACE_HEAD_ALIGNMENT_PASS"
            if core["all_gates_pass"]
            else "TRACE_HEAD_ALIGNMENT_FAIL"
        ),
    }
    report["deterministic_payload_sha256"] = (
        alignment._recompute_report_payload_sha256(report)
    )
    report["nondeterministic_provenance"] = {
        "excluded_from_deterministic_payload_sha256": True,
        "artifact_role": "primary",
        "process_id": 3001,
        "process_instance_token": "a" * 64,
        "run_started_utc": "2026-09-03T03:04:05.000001Z",
        "runtime_seconds": 0.25,
        "primary_prerequisite": None,
    }
    return report, binding, manifest


def test_protocol_config_is_internally_consistent() -> None:
    config = alignment.validate_protocol_config()
    prerequisite = config["a3_prerequisite"]
    assert prerequisite["required_protocol_version"] == "readout-trace-v1a3"
    assert prerequisite["resource_invalid_protocol_versions"] == [
        "readout-trace-v1a2"
    ]
    assert prerequisite["required_registered_seeds"] == [105, 106, 107, 108, 109]
    assert prerequisite["phase_sequence_path"] == str(
        alignment.A3_PHASE_SEQUENCE_PATH
    ).replace("\\", "/")
    assert prerequisite["primary_path"] == str(alignment.A3_PRIMARY_REPORT_PATH).replace(
        "\\", "/"
    )
    assert prerequisite["rerun_path"] == str(alignment.A3_RERUN_REPORT_PATH).replace(
        "\\", "/"
    )
    assert config["execution"]["official_seeds"] == [75, 76, 77, 78, 79]
    assert config["execution"]["episodes_per_seed"] == 32
    assert config["finite_difference"]["coordinates_per_case"] == 16


@pytest.mark.parametrize("target", [-1.0, 1.0])
def test_analytic_direction_matches_all_central_differences(target: float) -> None:
    theta = np.linspace(-0.2, 0.2, alignment.FEATURE_DIMENSION, dtype=np.float64)
    features = np.linspace(-0.7, 0.9, alignment.FEATURE_DIMENSION, dtype=np.float64)
    analytic = alignment.analytic_descent_direction(theta, features, target)
    numerical = alignment.central_difference_descent_direction(theta, features, target)
    absolute = np.abs(analytic - numerical)
    relative = absolute / np.maximum(
        np.maximum(np.abs(analytic), np.abs(numerical)),
        np.finfo(np.float64).tiny,
    )
    assert np.all(
        np.logical_or(
            absolute <= alignment.ABSOLUTE_TOLERANCE,
            relative <= alignment.RELATIVE_TOLERANCE,
        )
    )
    dot = float(np.dot(analytic, numerical))
    cosine = dot / float(np.linalg.norm(analytic) * np.linalg.norm(numerical))
    assert dot > 0.0
    assert cosine >= alignment.DIRECTION_COSINE_MINIMUM


def test_normalized_head_update_applies_both_clip_stages() -> None:
    theta = np.full(alignment.FEATURE_DIMENSION, 2.99, dtype=np.float64)
    features = np.full(alignment.FEATURE_DIMENSION, 0.25, dtype=np.float64)
    direction = np.full(alignment.FEATURE_DIMENSION, 100.0, dtype=np.float64)
    updated, delta, normalizer = alignment.normalized_head_update(
        theta, direction, features
    )
    assert normalizer == 1.0
    assert np.array_equal(delta, np.full(alignment.FEATURE_DIMENSION, 0.05))
    assert np.array_equal(updated, np.full(alignment.FEATURE_DIMENSION, 3.0))


def test_scratch_alignment_is_deterministic_and_never_mutates_graph() -> None:
    first = alignment.run_scratch_alignment(
        seeds=(9075,), episodes_per_seed=4, retention=0.75
    )
    second = alignment.run_scratch_alignment(
        seeds=(9075,), episodes_per_seed=4, retention=0.75
    )
    assert first == second
    assert first["scratch_nonselecting"] is True
    assert first["total_case_count"] == 4
    assert first["total_coordinate_check_count"] == 64
    assert first["all_gates_pass"] is True
    seed = first["seed_results"][0]
    assert (seed["cue_zeros"], seed["cue_ones"]) == (2, 2)
    assert seed["topology_unchanged"] is True
    assert seed["all_graph_weights_unchanged"] is True
    assert seed["recurrent_weights_unchanged"] is True
    assert all(len(row["coordinate_pass"]) == 16 for row in seed["cases"])


def test_scratch_execution_rejects_official_seeds() -> None:
    with pytest.raises(ValueError, match="rejects official seeds"):
        alignment.run_scratch_alignment(seeds=(75,), episodes_per_seed=2)


def test_a3_loader_binding_accepts_only_selected_complete_phase_chain() -> None:
    selection, phase_info = _a3_selection_fixture()
    binding = alignment._validated_a3_selection_binding(selection, phase_info)
    assert binding["a3_protocol_version"] == "readout-trace-v1a3"
    assert binding["a3_registered_seeds"] == [105, 106, 107, 108, 109]
    assert binding["selected_condition"] == "rho_0_75"
    assert binding["trace_retention"] == 0.75
    assert binding["a3_phase_sequence_file_sha256"] == "a" * 64

    unselected = copy.deepcopy(selection)
    unselected["status"] = "NO_SELECTION"
    with pytest.raises(RuntimeError, match="selection schema"):
        alignment._validated_a3_selection_binding(unselected, phase_info)

    incomplete_phase = copy.deepcopy(phase_info)
    incomplete_phase["record"]["phases"].pop()
    incomplete_phase["record"]["phase_sequence_payload_sha256"] = (
        alignment._record_self_hash(
            incomplete_phase["record"], "phase_sequence_payload_sha256"
        )
    )
    with pytest.raises(RuntimeError, match="complete phase sequence"):
        alignment._validated_a3_selection_binding(selection, incomplete_phase)


def test_a3_loader_wrapper_is_the_only_selection_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptive_learning_substrate import experiment_000_readout_trace_a3 as a3

    selection, phase_info = _a3_selection_fixture()
    selection_calls = 0
    phase_calls = 0

    def selected() -> dict[str, Any]:
        nonlocal selection_calls
        selection_calls += 1
        return copy.deepcopy(selection)

    def phases(names: Any) -> dict[str, Any]:
        nonlocal phase_calls
        phase_calls += 1
        assert tuple(names) == alignment.A3_REQUIRED_PHASE_ORDER
        return copy.deepcopy(phase_info)

    monkeypatch.setattr(a3, "load_verified_a3_selection", selected)
    monkeypatch.setattr(a3, "_validate_phase_sequence", phases)
    binding = alignment.load_verified_a3_selection()
    assert binding["selected_condition"] == "rho_0_75"
    assert selection_calls == 2
    assert phase_calls == 2


def test_resource_invalid_v1a2_is_explicitly_rejected() -> None:
    selection, phase_info = _a3_selection_fixture()
    selection["protocol_version"] = "readout-trace-v1a2"
    with pytest.raises(RuntimeError, match="resource-invalid predecessor"):
        alignment._validated_a3_selection_binding(selection, phase_info)

def test_report_payload_hash_ignores_only_nondeterministic_fields() -> None:
    report = {
        "status": "TRACE_HEAD_ALIGNMENT_PASS",
        "value": [1, 2, 3],
        "deterministic_payload_sha256": "placeholder",
        "nondeterministic_provenance": {"process_id": 1},
    }
    expected = alignment._sha256_json(
        {"status": "TRACE_HEAD_ALIGNMENT_PASS", "value": [1, 2, 3]}
    )
    assert alignment._recompute_report_payload_sha256(report) == expected
    changed = copy.deepcopy(report)
    changed["nondeterministic_provenance"]["process_id"] = 2
    assert alignment._recompute_report_payload_sha256(changed) == expected
    changed["value"] = [1, 2, 4]
    assert alignment._recompute_report_payload_sha256(changed) != expected


def test_report_validator_replays_exact_stream_features_and_graph_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch_seeds = (9075, 9076)
    monkeypatch.setattr(alignment, "OFFICIAL_SEEDS", scratch_seeds)
    monkeypatch.setattr(alignment, "EPISODES_PER_SEED", 4)
    report, binding, manifest = _scratch_full_report(
        seeds=scratch_seeds, episodes_per_seed=4
    )
    alignment._validate_alignment_report(
        report,
        expected_a3_binding=binding,
        current_manifest=manifest,
    )

    forged = copy.deepcopy(report)
    forged_row = copy.deepcopy(forged["seed_results"][0])
    forged_row["seed"] = scratch_seeds[1]
    forged_row["seed_role"] = "custom"
    forged_row["feature_edge_ids"] = ["forged_edge"] * alignment.FEATURE_DIMENSION
    forged_row["topology_sha256_before"] = "f" * 64
    forged_row["topology_sha256_after"] = "f" * 64
    forged_row["seed_payload_sha256"] = alignment._record_self_hash(
        forged_row, "seed_payload_sha256"
    )
    forged["seed_results"][1] = forged_row
    forged["deterministic_payload_sha256"] = (
        alignment._recompute_report_payload_sha256(forged)
    )
    with pytest.raises(RuntimeError, match="does not replay exactly"):
        alignment._validate_alignment_report(
            forged,
            expected_a3_binding=binding,
            current_manifest=manifest,
        )


def test_alignment_determinism_rejects_same_pid_forgery() -> None:
    primary, rerun, primary_files, rerun_files = _alignment_process_pair()
    a3_binding = {"selection": "rho_0_75"}
    source = {"bundle_sha256": "1" * 64}
    valid = alignment._expected_alignment_determinism_record(
        primary,
        rerun,
        primary_files=primary_files,
        rerun_files=rerun_files,
        a3_binding=a3_binding,
        source_manifest=source,
    )
    alignment._validate_alignment_determinism_record(
        valid,
        primary,
        rerun,
        primary_files=primary_files,
        rerun_files=rerun_files,
        a3_binding=a3_binding,
        source_manifest=source,
        require_pass=True,
    )

    rerun["nondeterministic_provenance"]["process_id"] = primary[
        "nondeterministic_provenance"
    ]["process_id"]
    forged = copy.deepcopy(valid)
    forged["second_process_id"] = forged["first_process_id"]
    forged["verification_payload_sha256"] = alignment._record_self_hash(
        forged, "verification_payload_sha256"
    )
    with pytest.raises(RuntimeError, match="inconsistent"):
        alignment._validate_alignment_determinism_record(
            forged,
            primary,
            rerun,
            primary_files=primary_files,
            rerun_files=rerun_files,
            a3_binding=a3_binding,
            source_manifest=source,
            require_pass=True,
        )


def test_alignment_determinism_cannot_promote_matching_failures_to_pass() -> None:
    primary, rerun, primary_files, rerun_files = _alignment_process_pair(
        status="TRACE_HEAD_ALIGNMENT_FAIL"
    )
    a3_binding = {"selection": "rho_0_75"}
    source = {"bundle_sha256": "2" * 64}
    failed = alignment._expected_alignment_determinism_record(
        primary,
        rerun,
        primary_files=primary_files,
        rerun_files=rerun_files,
        a3_binding=a3_binding,
        source_manifest=source,
    )
    assert failed["status"] == "TRACE_HEAD_ALIGNMENT_FAIL"
    alignment._validate_alignment_determinism_record(
        failed,
        primary,
        rerun,
        primary_files=primary_files,
        rerun_files=rerun_files,
        a3_binding=a3_binding,
        source_manifest=source,
        require_pass=False,
    )
    with pytest.raises(RuntimeError, match="verified pass"):
        alignment._validate_alignment_determinism_record(
            failed,
            primary,
            rerun,
            primary_files=primary_files,
            rerun_files=rerun_files,
            a3_binding=a3_binding,
            source_manifest=source,
            require_pass=True,
        )

    forged = copy.deepcopy(failed)
    forged["status"] = "TRACE_HEAD_ALIGNMENT_PASS"
    forged["verification_payload_sha256"] = alignment._record_self_hash(
        forged, "verification_payload_sha256"
    )
    with pytest.raises(RuntimeError, match="inconsistent"):
        alignment._validate_alignment_determinism_record(
            forged,
            primary,
            rerun,
            primary_files=primary_files,
            rerun_files=rerun_files,
            a3_binding=a3_binding,
            source_manifest=source,
            require_pass=True,
        )


def test_persisted_sidecar_detects_report_tampering(tmp_path) -> None:
    report_path = tmp_path / "scratch.json"
    report = {"schema": "scratch", "status": "nonselecting"}
    written, sidecar = alignment._write_report_and_sidecar(report_path, report)
    reopened, hashes = alignment._read_report_with_sidecar(written)
    assert reopened == report
    assert hashes["report_file_sha256"] == alignment._file_sha256(written)
    assert hashes["sidecar_file_sha256"] == alignment._file_sha256(sidecar)
    written.write_text(json.dumps({"tampered": True}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="sidecar is invalid"):
        alignment._read_report_with_sidecar(written)


def test_atomic_json_write_never_replaces_an_existing_artifact(tmp_path) -> None:
    target = tmp_path / "immutable.json"
    alignment._atomic_write_json(target, {"version": 1})
    before = target.read_bytes()
    with pytest.raises(RuntimeError, match="immutable artifact already exists"):
        alignment._atomic_write_json(target, {"version": 2})
    assert target.read_bytes() == before


def test_official_runner_stops_at_missing_a3_before_any_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reached_core = False

    def absent_selection() -> dict[str, object]:
        raise RuntimeError("synthetic A3 prerequisite absent")

    def forbidden_core(**_: object) -> dict[str, object]:
        nonlocal reached_core
        reached_core = True
        raise AssertionError("official graph execution must not start")

    monkeypatch.setattr(alignment, "load_verified_a3_selection", absent_selection)
    monkeypatch.setattr(alignment, "_run_alignment_core", forbidden_core)
    with pytest.raises(RuntimeError, match="synthetic A3 prerequisite absent"):
        alignment.run_trace_head_alignment(
            seeds=alignment.OFFICIAL_SEEDS,
            episodes_per_seed=alignment.EPISODES_PER_SEED,
            output_path=alignment.PRIMARY_REPORT_PATH,
        )
    assert reached_core is False


def test_source_manifest_covers_protocol_config_runner_and_tests() -> None:
    manifest = alignment._source_manifest()
    paths = {row["path"] for row in manifest["files"]}
    assert str(alignment.PROTOCOL_PATH).replace("\\", "/") in paths
    assert str(alignment.CONFIG_PATH).replace("\\", "/") in paths
    assert "src/adaptive_learning_substrate/experiment_000_trace_head_alignment.py" in paths
    assert "tests/test_experiment_000_trace_head_alignment.py" in paths
    assert "src/adaptive_learning_substrate/experiment_000_readout_trace_a3.py" in paths
    assert "src/adaptive_learning_substrate/experiment_000_readout_trace_a2.py" not in paths
