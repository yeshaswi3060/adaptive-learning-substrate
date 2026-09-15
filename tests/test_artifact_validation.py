from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from adaptive_learning_substrate import artifact_validation as validation


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _self_hash(record: dict[str, object], field: str) -> str:
    payload = dict(record)
    payload.pop(field, None)
    return _canonical_hash(payload)


def _write_json(path: Path, value: object) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    return payload


def _write_sidecar(report: Path) -> tuple[bytes, str]:
    report_hash = hashlib.sha256(report.read_bytes()).hexdigest()
    sidecar = report.with_suffix(".sha256")
    payload = _write_json(
        sidecar,
        {
            "algorithm": "sha256",
            "report_file": report.name,
            "report_sha256": report_hash,
        },
    )
    return payload, report_hash


def _write_a3_report(
    path: Path,
    *,
    terminal_status: str,
    process_id: int,
    process_token: str,
    started_utc: str,
) -> dict[str, object]:
    selected_condition = (
        None
        if terminal_status == "NO_SELECTION"
        else terminal_status.removeprefix("SELECTED:")
    )
    deterministic: dict[str, object] = {
        "schema_version": "fixture-a3-report-v1",
        "status": "PENDING_DETERMINISM_VERIFICATION",
        "selection": {
            "provisional_terminal_status": terminal_status,
            "selected_condition": selected_condition,
        },
    }
    deterministic_hash = _canonical_hash(deterministic)
    provenance = {
        "process_id": process_id,
        "process_instance_token": process_token,
        "run_started_utc": started_utc,
    }
    report = {
        **deterministic,
        "deterministic_payload_sha256": deterministic_hash,
        "nondeterministic_provenance": provenance,
    }
    report_bytes = _write_json(path, report)
    sidecar_bytes, report_hash = _write_sidecar(path)
    return {
        "report_bytes": report_bytes,
        "report_hash": report_hash,
        "sidecar_bytes": sidecar_bytes,
        "sidecar_hash": hashlib.sha256(sidecar_bytes).hexdigest(),
        "deterministic_hash": deterministic_hash,
        "provenance_hash": _canonical_hash(provenance),
        "process_id": process_id,
        "process_token": process_token,
        "started_utc": started_utc,
    }


def _phase_artifact(path: Path, root: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_a3_chain(root: Path, *, terminal_status: str) -> None:
    primary_path = root / validation.A3_PRIMARY
    rerun_path = root / validation.A3_RERUN
    primary = _write_a3_report(
        primary_path,
        terminal_status=terminal_status,
        process_id=101,
        process_token="1" * 64,
        started_utc="2026-09-03T01:00:00Z",
    )
    rerun = _write_a3_report(
        rerun_path,
        terminal_status=terminal_status,
        process_id=202,
        process_token="2" * 64,
        started_utc="2026-09-03T02:00:00Z",
    )

    freeze: dict[str, object] = {
        "schema_version": "fixture-a3-freeze-v1",
        "source_manifest_sha256": "c" * 64,
    }
    freeze["freeze_payload_sha256"] = _self_hash(freeze, "freeze_payload_sha256")
    freeze_path = root / validation.A3_FREEZE
    freeze_bytes = _write_json(freeze_path, freeze)

    terminal: dict[str, object] = {
        "schema_version": "experiment-000-readout-trace-a3-determinism-v1",
        "status": terminal_status,
        "first_deterministic_payload_sha256": primary["deterministic_hash"],
        "second_deterministic_payload_sha256": rerun["deterministic_hash"],
        "first_recomputed_payload_sha256": primary["deterministic_hash"],
        "second_recomputed_payload_sha256": rerun["deterministic_hash"],
        "first_report_file_sha256": primary["report_hash"],
        "second_report_file_sha256": rerun["report_hash"],
        "first_sidecar_file_sha256": primary["sidecar_hash"],
        "second_sidecar_file_sha256": rerun["sidecar_hash"],
        "first_nondeterministic_provenance_sha256": primary["provenance_hash"],
        "second_nondeterministic_provenance_sha256": rerun["provenance_hash"],
        "first_process_id": primary["process_id"],
        "second_process_id": rerun["process_id"],
        "first_process_instance_token": primary["process_token"],
        "second_process_instance_token": rerun["process_token"],
        "first_run_started_utc": primary["started_utc"],
        "second_run_started_utc": rerun["started_utc"],
        "freeze_record_sha256": hashlib.sha256(freeze_bytes).hexdigest(),
        "source_manifest_sha256": freeze["source_manifest_sha256"],
        **{
            key: True
            for key in (
                "canonical_deterministic_payloads_equal",
                "deterministic_payloads_match",
                "first_self_hash_valid",
                "fresh_processes_verified",
                "primary_prerequisite_verified",
                "second_self_hash_valid",
                "selection_objects_match",
                "selection_statuses_match",
                "valid_terminal_status",
            )
        },
    }
    terminal["verification_payload_sha256"] = _self_hash(
        terminal, "verification_payload_sha256"
    )
    terminal_path = root / validation.A3_TERMINAL
    _write_json(terminal_path, terminal)

    phases = {
        name: {"name": name, "artifacts": []} for name in validation.A3_PHASE_ORDER
    }
    phases["freeze"]["artifacts"] = [_phase_artifact(freeze_path, root)]
    phases["full"]["artifacts"] = [
        _phase_artifact(primary_path, root),
        _phase_artifact(primary_path.with_suffix(".sha256"), root),
    ]
    phases["rerun"]["artifacts"] = [
        _phase_artifact(rerun_path, root),
        _phase_artifact(rerun_path.with_suffix(".sha256"), root),
    ]
    phases["determinism_verification"]["artifacts"] = [
        _phase_artifact(terminal_path, root)
    ]
    phase: dict[str, object] = {
        "schema_version": "experiment-000-readout-trace-a3-phases-v1",
        "protocol_version": "readout-trace-v1a3",
        "phase_order": list(validation.A3_PHASE_ORDER),
        "phases": [phases[name] for name in validation.A3_PHASE_ORDER],
    }
    phase["phase_sequence_payload_sha256"] = _self_hash(
        phase, "phase_sequence_payload_sha256"
    )
    _write_json(root / validation.A3_PHASE_SEQUENCE, phase)


def _copy_project_fixture(
    tmp_path: Path, *, terminal_status: str = "NO_SELECTION"
) -> Path:
    root = tmp_path / "project"
    (root / "artifacts/stage1/ccf").mkdir(parents=True)
    (root / "artifacts/bound").mkdir(parents=True)
    (root / "runs/latest").mkdir(parents=True)
    (root / "docs").mkdir(parents=True)
    stage1 = {
        "confirmatory": False,
        "implemented_claim": "deterministic fixed-DAG software plumbing only",
        "composition": {"method": "external_two_pass_primitive_adapter"},
    }
    _write_json(root / validation.CANONICAL_STAGE1 / "report.json", stage1)
    for name in validation.LATEST_FILES:
        canonical = root / validation.CANONICAL_STAGE1 / name
        canonical.parent.mkdir(parents=True, exist_ok=True)
        if not canonical.exists():
            canonical.write_text(name, encoding="utf-8")
        shutil.copyfile(canonical, root / validation.LATEST_RUN / name)

    report = root / "artifacts/bound/report.json"
    report.write_text('{"ok":true}\n', encoding="utf-8")
    _write_sidecar(report)
    _write_a3_chain(root, terminal_status=terminal_status)

    frozen_input = root / "frozen_input.txt"
    frozen_input.write_text("frozen\n", encoding="utf-8")
    frozen_payload = frozen_input.read_bytes()
    lwoh_freeze: dict[str, object] = {
        "schema_version": "experiment-000-lwoh-l1-a3-preterminal-freeze-v2",
        "status": "FROZEN_BLINDED_PRE_TERMINAL",
        "v2_source_manifest": {
            "bundle_sha256": "d" * 64,
            "files": {
                "frozen_input.txt": {
                    "bytes": len(frozen_payload),
                    "sha256": hashlib.sha256(frozen_payload).hexdigest(),
                }
            },
        },
    }
    lwoh_freeze["freeze_payload_sha256"] = _self_hash(
        lwoh_freeze, "freeze_payload_sha256"
    )
    lwoh_path = root / validation.LWOH_SOURCE_FREEZE
    _write_json(lwoh_path, lwoh_freeze)
    _write_sidecar(lwoh_path)

    source_snapshot = (
        Path(validation.__file__).resolve().parent
        / validation.FROZEN_RECURRENT_SOURCE_RESOURCE
    )
    target_snapshot = (
        root
        / "src"
        / "adaptive_learning_substrate"
        / validation.FROZEN_RECURRENT_SOURCE_RESOURCE
    )
    target_snapshot.parent.mkdir(parents=True)
    shutil.copyfile(source_snapshot, target_snapshot)

    validation.write_status(project_root=root)
    return root


def _write_v3_chain(root: Path, *, status: str) -> Path:
    v3_root = root / validation.LWOH_V3_ROOT
    v3_root.mkdir(parents=True, exist_ok=True)
    lineage = root / validation.LWOH_V3_LINEAGE
    lineage_bytes = _write_json(lineage, {"status": "PRE_METRIC_LINEAGE"})
    required_count = 3 if status == "LWOH_ADMISSION_FAIL" else 5
    stages: list[tuple[Path, bytes]] = []
    for index, (name, schema, _, self_field) in enumerate(
        validation.LWOH_V3_STAGE_CONTRACT[:required_count]
    ):
        if index == 2:
            stage_status = (
                "LWOH_ADMISSION_FAIL"
                if status == "LWOH_ADMISSION_FAIL"
                else "LWOH_ADMISSION_PASS"
            )
        elif index == 4:
            stage_status = status
        else:
            stage_status = next(
                iter(validation.LWOH_V3_STAGE_CONTRACT[index][2])
            )
        stage: dict[str, object] = {
            "schema_version": schema,
            "status": stage_status,
        }
        if index in (0, 3):
            stage["source_manifest_sha256"] = "e" * 64
        stage[self_field] = _self_hash(stage, self_field)
        stage_path = v3_root / name
        stage_bytes = _write_json(stage_path, stage)
        _write_sidecar(stage_path)
        stages.append((stage_path, stage_bytes))

    if status == "LWOH_ADMISSION_FAIL":
        gates = (True, False, False, False, False, False)
    elif status == "LWOH_LEARNING_FAIL":
        gates = (True, True, True, False, True, True)
    else:
        gates = (True, True, True, True, True, True)
    readiness: dict[str, object] = {
        "schema_version": validation.LWOH_V3_READINESS_SCHEMA,
        "status": status,
        "readiness_before": 21,
        "readiness_after": 25 if status == "LWOH_LEARNING_PASS" else 21,
        "partial_score": False,
        "admission_seeds": list(range(90, 95)),
        "learning_seeds": list(range(95, 105)),
        "delays": [4, 8, 16],
        "baselines": list(validation.LWOH_V3_BASELINES),
        "deterministic_admission": gates[0],
        "deterministic_learning": gates[1],
        "all_admission_gates_pass": gates[2],
        "all_learning_gates_pass": gates[3],
        "all_robustness_gates_pass": gates[4],
        "all_integrity_gates_pass": gates[5],
        "lineage_handoff_path": validation.LWOH_V3_LINEAGE.as_posix(),
        "lineage_handoff_sha256": hashlib.sha256(lineage_bytes).hexdigest(),
        "chain_artifacts": {
            path.relative_to(root).as_posix(): hashlib.sha256(payload).hexdigest()
            for path, payload in stages
        },
    }
    readiness["verification_payload_sha256"] = _self_hash(
        readiness, "verification_payload_sha256"
    )
    readiness_path = root / validation.LWOH_V3_READINESS_PATH
    _write_json(readiness_path, readiness)
    _write_sidecar(readiness_path)
    return readiness_path


def _deep_v3_result(status: str) -> dict[str, object]:
    values: dict[str, object] = {
        field: True for field in validation.LWOH_V3_DEEP_BOOLEAN_FIELDS
    }
    if status == "LWOH_ADMISSION_FAIL":
        for field in (
            "raw_learning_primary_valid",
            "raw_learning_rerun_valid",
            "learning_gates_recomputed",
            "statistical_gates_recomputed",
            "robustness_gates_recomputed",
        ):
            values[field] = False
    return {
        "status": status,
        "readiness_before": 21,
        "readiness_after": 25 if status == "LWOH_LEARNING_PASS" else 21,
        **values,
    }


def _mock_v3_deep_verifier(
    monkeypatch: pytest.MonkeyPatch, *, status: str
) -> None:
    class Module:
        @staticmethod
        def validate_official_v3_chain(*, project_root: Path) -> dict[str, object]:
            assert project_root.is_absolute()
            return _deep_v3_result(status)

    monkeypatch.setattr(validation.importlib, "import_module", lambda _: Module)


def test_project_validator_accepts_bound_fixture(tmp_path: Path) -> None:
    root = _copy_project_fixture(tmp_path)
    result = validation.validate_project(project_root=root)
    assert result["status"] == "PASS"
    assert result["frozen_source_snapshots"][0]["sha256"] == (
        "b56216abd19f5d3aaf935b6e58831c54ccf078ce4e4c7ef80e006b04ca6231b3"
    )
    assert len(result["sidecars"]) == 4
    assert len(result["latest_run"]) == 3


def test_v3_absence_is_valid_and_keeps_readiness_at_21(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_project_fixture(tmp_path)
    monkeypatch.setattr(
        validation.importlib,
        "import_module",
        lambda _: (_ for _ in ()).throw(AssertionError("unexpected V3 import")),
    )

    result = validation.validate_lwoh_v3_state(project_root=root)

    assert result["state"] == "ABSENT"
    assert result["readiness_after"] == 21


def test_v3_partial_chain_fails_closed(tmp_path: Path) -> None:
    root = _copy_project_fixture(tmp_path)
    partial = root / validation.LWOH_V3_ROOT / "PRE_METRIC_FREEZE.json"
    _write_json(partial, {"schema_version": "partial"})

    with pytest.raises(validation.EvidenceValidationError, match="absent"):
        validation.validate_lwoh_v3_state(project_root=root)


def test_v3_malformed_readiness_fails_closed(tmp_path: Path) -> None:
    root = _copy_project_fixture(tmp_path)
    readiness_path = _write_v3_chain(root, status="LWOH_ADMISSION_FAIL")
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["schema_version"] = "wrong-schema"
    readiness["verification_payload_sha256"] = _self_hash(
        readiness, "verification_payload_sha256"
    )
    _write_json(readiness_path, readiness)
    _write_sidecar(readiness_path)

    with pytest.raises(validation.EvidenceValidationError, match="stage contract"):
        validation.validate_lwoh_v3_state(project_root=root)


def test_v3_chain_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    root = _copy_project_fixture(tmp_path)
    readiness_path = _write_v3_chain(root, status="LWOH_LEARNING_PASS")
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    first_path = next(iter(readiness["chain_artifacts"]))
    readiness["chain_artifacts"][first_path] = "0" * 64
    readiness["verification_payload_sha256"] = _self_hash(
        readiness, "verification_payload_sha256"
    )
    _write_json(readiness_path, readiness)
    _write_sidecar(readiness_path)

    with pytest.raises(validation.EvidenceValidationError, match="chain hash"):
        validation.validate_lwoh_v3_state(project_root=root)


def test_v3_six_json_metadata_alone_cannot_produce_readiness_25(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_project_fixture(tmp_path)
    _write_v3_chain(root, status="LWOH_LEARNING_PASS")

    class ModuleWithoutDeepVerifier:
        pass

    monkeypatch.setattr(
        validation.importlib, "import_module", lambda _: ModuleWithoutDeepVerifier
    )
    with pytest.raises(validation.EvidenceValidationError, match="deep verifier"):
        validation.validate_lwoh_v3_state(project_root=root)


def test_v3_source_manifest_mismatch_rejects_even_with_valid_deep_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_project_fixture(tmp_path)
    readiness_path = _write_v3_chain(root, status="LWOH_LEARNING_PASS")
    implementation_path = (
        root / validation.LWOH_V3_ROOT / "IMPLEMENTATION_SOURCE_FREEZE.json"
    )
    implementation = json.loads(implementation_path.read_text(encoding="utf-8"))
    implementation["source_manifest_sha256"] = "f" * 64
    implementation["freeze_payload_sha256"] = _self_hash(
        implementation, "freeze_payload_sha256"
    )
    implementation_bytes = _write_json(implementation_path, implementation)
    _write_sidecar(implementation_path)
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    relative = implementation_path.relative_to(root).as_posix()
    readiness["chain_artifacts"][relative] = hashlib.sha256(
        implementation_bytes
    ).hexdigest()
    readiness["verification_payload_sha256"] = _self_hash(
        readiness, "verification_payload_sha256"
    )
    _write_json(readiness_path, readiness)
    _write_sidecar(readiness_path)
    _mock_v3_deep_verifier(monkeypatch, status="LWOH_LEARNING_PASS")

    with pytest.raises(validation.EvidenceValidationError, match="source manifests"):
        validation.validate_lwoh_v3_state(project_root=root)


def test_v3_deep_false_gate_rejects_metadata_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_project_fixture(tmp_path)
    _write_v3_chain(root, status="LWOH_LEARNING_PASS")
    result = _deep_v3_result("LWOH_LEARNING_PASS")
    result["condition_ids_exact"] = False

    class Module:
        @staticmethod
        def validate_official_v3_chain(*, project_root: Path) -> dict[str, object]:
            assert project_root.is_absolute()
            return result

    monkeypatch.setattr(validation.importlib, "import_module", lambda _: Module)
    with pytest.raises(validation.EvidenceValidationError, match="gate is false"):
        validation.validate_lwoh_v3_state(project_root=root)


@pytest.mark.parametrize(
    "status", ("LWOH_ADMISSION_FAIL", "LWOH_LEARNING_FAIL")
)
def test_v3_well_formed_failure_keeps_readiness_at_21(
    tmp_path: Path, status: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_project_fixture(tmp_path)
    _write_v3_chain(root, status=status)
    _mock_v3_deep_verifier(monkeypatch, status=status)

    result = validation.validate_lwoh_v3_state(project_root=root)

    assert result["state"] == "TERMINAL"
    assert result["status"] == status
    assert result["readiness_after"] == 21


def test_v3_complete_exact_pass_is_the_only_route_to_readiness_25(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_project_fixture(tmp_path)
    _write_v3_chain(root, status="LWOH_LEARNING_PASS")
    _mock_v3_deep_verifier(monkeypatch, status="LWOH_LEARNING_PASS")

    result = validation.validate_lwoh_v3_state(project_root=root)
    status = validation.canonical_status(project_root=root)

    assert result["status"] == "LWOH_LEARNING_PASS"
    assert result["readiness_after"] == 25
    assert status["research_readiness_percent"] == 25


def test_a3_validator_accepts_registered_selected_status(tmp_path: Path) -> None:
    root = _copy_project_fixture(tmp_path, terminal_status="SELECTED:rho_0_95")

    result = validation.validate_a3_terminal(project_root=root)

    assert result["status"] == "SELECTED:rho_0_95"


@pytest.mark.parametrize("tamper", ("report", "filename", "hash", "latest", "status"))
def test_project_validator_rejects_integrity_tampering(
    tmp_path: Path, tamper: str
) -> None:
    root = _copy_project_fixture(tmp_path)
    sidecar = root / "artifacts/bound/report.sha256"
    binding = json.loads(sidecar.read_text(encoding="utf-8"))
    if tamper == "report":
        (root / "artifacts/bound/report.json").write_text("tampered", encoding="utf-8")
    elif tamper == "filename":
        binding["report_file"] = "../report.json"
        _write_json(sidecar, binding)
    elif tamper == "hash":
        binding["report_sha256"] = "0" * 64
        _write_json(sidecar, binding)
    elif tamper == "latest":
        (root / validation.LATEST_RUN / "report.json").write_text(
            "stale", encoding="utf-8"
        )
    else:
        (root / validation.CANONICAL_STATUS).write_text("stale", encoding="utf-8")
    with pytest.raises(validation.EvidenceValidationError):
        validation.validate_project(project_root=root)


def test_a3_terminal_rejects_plausible_but_wrong_self_hash(tmp_path: Path) -> None:
    root = _copy_project_fixture(tmp_path)
    path = root / validation.A3_TERMINAL
    terminal = json.loads(path.read_text(encoding="utf-8"))
    terminal["verification_payload_sha256"] = "0" * 64
    _write_json(path, terminal)

    with pytest.raises(validation.EvidenceValidationError, match="self-hash"):
        validation.validate_a3_terminal(project_root=root)


def test_a3_terminal_rejects_report_rewrite_even_with_matching_sidecar(
    tmp_path: Path,
) -> None:
    root = _copy_project_fixture(tmp_path)
    report = root / validation.A3_PRIMARY
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["nondeterministic_provenance"]["process_id"] = 999
    _write_json(report, payload)
    _write_sidecar(report)

    with pytest.raises(validation.EvidenceValidationError, match="terminal/report"):
        validation.validate_a3_terminal(project_root=root)


def test_frozen_snapshot_validation_uses_requested_project_root(tmp_path: Path) -> None:
    root = _copy_project_fixture(tmp_path)
    snapshot = (
        root
        / "src"
        / "adaptive_learning_substrate"
        / validation.FROZEN_RECURRENT_SOURCE_RESOURCE
    )
    snapshot.write_bytes(b"tampered")

    with pytest.raises(validation.EvidenceValidationError, match="snapshot hash mismatch"):
        validation.validate_project(project_root=root)


def test_source_freeze_drift_is_reported_without_advancing_readiness(
    tmp_path: Path,
) -> None:
    root = _copy_project_fixture(tmp_path)
    (root / "frozen_input.txt").write_text("changed\n", encoding="utf-8")

    status = validation.canonical_status(project_root=root)

    assert status["research_readiness_percent"] == 21
    assert status["official_experiments_enabled"] is False
    assert status["lwoh_l1_v2_source_freeze"]["status"] == "DRIFTED"
    assert status["lwoh_l1_v2_source_freeze"]["changed_files"] == [
        "frozen_input.txt"
    ]


def test_sidecar_validation_detects_semantically_equal_byte_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_project_fixture(tmp_path)
    sidecar = root / "artifacts/bound/report.sha256"
    original_stable_bytes = validation._stable_bytes
    reads = 0

    def replacing_stable_bytes(path: Path) -> bytes:
        nonlocal reads
        if path == sidecar:
            reads += 1
            if reads == 2:
                value = json.loads(sidecar.read_text(encoding="utf-8"))
                sidecar.write_text(json.dumps(value, indent=2), encoding="utf-8")
        return original_stable_bytes(path)

    monkeypatch.setattr(validation, "_stable_bytes", replacing_stable_bytes)
    with pytest.raises(validation.EvidenceValidationError, match="changed during"):
        validation.validate_sidecars(project_root=root)


def test_sidecar_validation_ignores_transaction_baselines_and_rollback_copies(
    tmp_path: Path,
) -> None:
    root = _copy_project_fixture(tmp_path)
    for transaction_namespace in ("BASELINE", "ROLLBACK_TEST_COPY"):
        archived = (
            root
            / "artifacts"
            / "evidence_integrity_transaction"
            / transaction_namespace
            / "archived.sha256"
        )
        archived.parent.mkdir(parents=True, exist_ok=True)
        _write_json(
            archived,
            {
                "algorithm": "sha256",
                "report_file": "historical-only.json",
                "report_sha256": "0" * 64,
            },
        )

    rows = validation.validate_sidecars(project_root=root)

    assert {row["sidecar"] for row in rows} == {
        "artifacts/bound/report.sha256",
        ("artifacts/experiment_000/lwoh_l1_v2/"
         "PRE_TERMINAL_CONTINGENCY_FREEZE.sha256"),
        validation.A3_PRIMARY.with_suffix(".sha256").as_posix(),
        validation.A3_RERUN.with_suffix(".sha256").as_posix(),
    }
