"""Fast non-scientific tests for the blinded LWOH-L1 A3 contingency.

These tests use only temporary synthetic bindings.  They never execute the A3
terminal verifier, replay an official report, or generate an LWOH-L1 sample.
"""

from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

import pytest

from adaptive_learning_substrate import experiment_000_lwoh_l1 as lwoh

ROOT = Path(__file__).resolve().parents[1]
BASE_PROTOCOL = ROOT / "docs/EXPERIMENT_000_LWOH_L1_PROTOCOL.md"
BASE_CONFIG = ROOT / "configs/experiment_000_lwoh_l1.toml"
AMENDMENT = ROOT / "docs/EXPERIMENT_000_LWOH_L1_A3_AMENDMENT_V2.md"
OVERLAY_CONFIG = ROOT / "configs/experiment_000_lwoh_l1_v2.toml"

BASE_PROTOCOL_SHA256 = (
    "42e237add4eac9ca24a6d5781efefce6367f8c9f2c4c4065b324a1703706dc7e"
)
BASE_CONFIG_SHA256 = (
    "7ed14e7dc85f7178e6dce0fbdf1e1d05d93c41129eba8b3c3f491dbfeffc86b8"
)
AMENDMENT_SHA256 = (
    "d46929ed8ce01e361580df1f119976797aabd915062815be70e7a4bde77b9c2b"
)
OVERLAY_CONFIG_SHA256 = (
    "c5f18680ff37f4a03bfb88ee5d8320783c85b07ab88d389b68543434d98e95fb"
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _synthetic_external_validation_evidence() -> tuple[dict[str, Any], ...]:
    """Return exact-schema passing evidence without running any subprocess."""

    outputs = {
        "py_compile": "",
        "ruff check": "All checks passed!\n",
        "pytest": "PYTEST_PASS_COUNT=45\n",
    }
    return tuple(
        {
            "command": command,
            "input": "",
            "literal_output": next(
                output for marker, output in outputs.items() if marker in command
            ),
            "exit_status": 0,
        }
        for command in lwoh._EXTERNAL_VALIDATION_ARGV
    )


def test_v1_inputs_and_v2_overlay_are_exactly_hash_bound() -> None:
    overlay = _load_toml(OVERLAY_CONFIG)

    assert _file_sha256(BASE_PROTOCOL) == BASE_PROTOCOL_SHA256
    assert _file_sha256(BASE_CONFIG) == BASE_CONFIG_SHA256
    assert _file_sha256(AMENDMENT) == AMENDMENT_SHA256
    assert _file_sha256(OVERLAY_CONFIG) == OVERLAY_CONFIG_SHA256
    assert overlay["base_v1"]["protocol_sha256"] == BASE_PROTOCOL_SHA256
    assert overlay["base_v1"]["config_sha256"] == BASE_CONFIG_SHA256
    assert overlay["amendment_sha256"] == AMENDMENT_SHA256


def test_overlay_allowlist_and_inherited_scientific_hashes_are_complete() -> None:
    base = _load_toml(BASE_CONFIG)
    overlay = _load_toml(OVERLAY_CONFIG)
    policy = overlay["overlay_policy"]

    assert set(overlay) == set(policy["allowed_top_level_keys"])
    assert not set(policy["forbidden_override_sections"]).intersection(overlay)
    inherited = policy["inherited_scientific_sections"]
    assert set(inherited) == set(policy["forbidden_override_sections"]) | {
        "seed_partition"
    }
    assert set(inherited) == set(overlay["base_v1"]["scientific_section_sha256"])
    for section in inherited:
        assert _canonical_sha256(base[section]) == overlay["base_v1"][
            "scientific_section_sha256"
        ][section]
    scientific_payload = {section: base[section] for section in inherited}
    assert _canonical_sha256(scientific_payload) == overlay["base_v1"][
        "scientific_payload_sha256"
    ]
    assert _canonical_sha256(base) == overlay["base_v1"][
        "canonical_toml_object_sha256"
    ]


def test_two_stage_source_boundary_keeps_controller_immutable_and_route_usable() -> None:
    overlay = _load_toml(OVERLAY_CONFIG)
    future = overlay["future_source_freeze"]
    post = overlay["post_activation_implementation"]

    assert tuple(future["required_paths"]) == lwoh.V2_SOURCE_PATHS
    assert future["scope"] == "immutable_preterminal_controller_bundle"
    assert future["post_activation_execution_paths_excluded"] is True
    assert future["future_path_creation_does_not_invalidate_preterminal_freeze"] is True
    assert future["require_fresh_process_validation"] is True
    assert lwoh.FUTURE_EXECUTION_RUNNER_PATH.as_posix() not in lwoh.V2_SOURCE_PATHS
    assert lwoh.FUTURE_EXECUTION_TEST_PATH.as_posix() not in lwoh.V2_SOURCE_PATHS

    assert post == {
        "activation_status_required": "ACTIVATED_A3_NO_SELECTION",
        "activation_provenance_path": lwoh.ACTIVATION_PROVENANCE_PATH.as_posix(),
        "activation_provenance_sidecar_path": (
            lwoh.ACTIVATION_PROVENANCE_SIDECAR_PATH.as_posix()
        ),
        "pre_terminal_controller_path": lwoh.RUNNER_PATH.as_posix(),
        "pre_terminal_controller_test_path": lwoh.TEST_PATH.as_posix(),
        "execution_runner_path": lwoh.FUTURE_EXECUTION_RUNNER_PATH.as_posix(),
        "execution_test_path": lwoh.FUTURE_EXECUTION_TEST_PATH.as_posix(),
        "implementation_freeze_path": lwoh.IMPLEMENTATION_SOURCE_FREEZE_PATH.as_posix(),
        "implementation_freeze_sidecar_path": (
            lwoh.IMPLEMENTATION_SOURCE_FREEZE_SIDECAR_PATH.as_posix()
        ),
        "creation_mode": "new_files_only_after_activation",
        "require_activation_provenance_before_source_creation": True,
        "require_preterminal_freeze_validation": True,
        "require_minimal_a3_activation_provenance": True,
        "required_before_any_lwoh_metric": True,
        "require_union_with_preterminal_source_manifest": True,
        "require_exact_v1_scientific_section_hashes": True,
        "require_all_transitive_runtime_dependencies": True,
        "require_fresh_process_validation": True,
        "pre_terminal_controller_may_change_after_freeze": False,
        "scientific_design_change_allowed": False,
    }
    boundary = overlay["implementation_boundary"]
    assert boundary["pre_terminal_controller_present_in_this_amendment"] is True
    assert boundary["pre_terminal_controller_tests_present_in_this_amendment"] is True
    assert boundary["post_activation_execution_runner_present_in_this_amendment"] is False
    assert boundary["post_activation_execution_tests_present_in_this_amendment"] is False
    assert boundary["official_lwoh_execution_authorized"] is False
    assert boundary["future_implementation_may_change_scientific_design"] is False

    effective = lwoh.load_effective_config()
    assert effective["post_activation_implementation"] == post
    assert all(
        "adaptive_learning_substrate.experiment_000_lwoh_l1_execution_v2"
        in command
        for command in effective["commands"].values()
    )
    assert all(
        "artifacts/experiment_000/lwoh_l1/" not in command
        and "artifacts/experiment_000/lwoh_l1_scratch/" not in command
        for command in effective["commands"].values()
    )


def test_effective_config_rejects_unknown_and_scientific_overlay_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    effective = lwoh.load_effective_config()
    assert isinstance(effective, dict)

    unknown = copy.deepcopy(_load_toml(OVERLAY_CONFIG))
    unknown["unregistered_override"] = {"enabled": True}
    with pytest.raises((ValueError, RuntimeError), match="(?i)(unknown|allowlist|overlay)"):
        lwoh._validate_overlay(unknown, _load_toml(BASE_CONFIG))

    scientific = copy.deepcopy(_load_toml(OVERLAY_CONFIG))
    scientific["graph"] = {"hidden_count": 65}
    with pytest.raises((ValueError, RuntimeError), match="(?i)(scientific|forbidden|overlay)"):
        lwoh._validate_overlay(scientific, _load_toml(BASE_CONFIG))

    weakened_boundary = copy.deepcopy(_load_toml(OVERLAY_CONFIG))
    weakened_boundary["post_activation_implementation"][
        "pre_terminal_controller_may_change_after_freeze"
    ] = True
    with pytest.raises(RuntimeError, match="post-activation implementation boundary"):
        lwoh._validate_overlay(weakened_boundary, _load_toml(BASE_CONFIG))


def test_seed_partitions_are_inherited_and_disjoint() -> None:
    effective = lwoh.load_effective_config()
    lwoh._validate_seed_partitions(effective)

    base = _load_toml(BASE_CONFIG)["seed_partition"]
    overlay = _load_toml(OVERLAY_CONFIG)["seed_inheritance"]
    admission = set(overlay["admission"])
    learning = set(overlay["learning"])
    scratch = set(overlay["scratch"])
    a3 = set(overlay["additional_forbidden_a3"])
    inherited_forbidden = {
        seed
        for key, seeds in base.items()
        if key.startswith("forbidden_")
        for seed in seeds
    }

    assert overlay["admission"] == base["admission"] == [90, 91, 92, 93, 94]
    assert overlay["learning"] == base["learning"] == list(range(95, 105))
    assert overlay["scratch"] == base["scratch"] == [9090, 9091]
    assert a3 == set(range(105, 110))
    assigned = admission | learning | scratch
    assert len(assigned) == len(admission) + len(learning) + len(scratch)
    assert assigned.isdisjoint(inherited_forbidden)
    assert assigned.isdisjoint(a3)
    assert assigned.isdisjoint(set(base["forbidden_confirmatory"]))


@pytest.mark.parametrize(
    ("terminal", "evidence_valid", "expected_status", "expected_action"),
    (
        (
            "NO_SELECTION",
            True,
            "ACTIVATED_A3_NO_SELECTION",
            "ACTIVATE_LWOH_L1_V2",
        ),
        (
            "SELECTED:rho_0_25",
            True,
            "CANCELLED_A3_SELECTED",
            "CANCEL_LWOH_L1_V2",
        ),
        (
            "SELECTED:rho_0_95",
            True,
            "CANCELLED_A3_SELECTED",
            "CANCEL_LWOH_L1_V2",
        ),
        (
            "PENDING_DETERMINISM_VERIFICATION",
            True,
            "INVALID_A3_DO_NOT_ACTIVATE_LWOH",
            "REPAIR_A3_DO_NOT_ACTIVATE_LWOH",
        ),
        (
            "NONDETERMINISTIC_INVALID",
            True,
            "INVALID_A3_DO_NOT_ACTIVATE_LWOH",
            "REPAIR_A3_DO_NOT_ACTIVATE_LWOH",
        ),
        (
            "SELECTED:unregistered",
            True,
            "INVALID_A3_DO_NOT_ACTIVATE_LWOH",
            "REPAIR_A3_DO_NOT_ACTIVATE_LWOH",
        ),
        (
            "NO_SELECTION",
            False,
            "INVALID_A3_DO_NOT_ACTIVATE_LWOH",
            "REPAIR_A3_DO_NOT_ACTIVATE_LWOH",
        ),
        (
            None,
            False,
            "INVALID_A3_DO_NOT_ACTIVATE_LWOH",
            "REPAIR_A3_DO_NOT_ACTIVATE_LWOH",
        ),
    ),
)
def test_terminal_classifier_is_exact_and_fail_closed(
    terminal: object,
    evidence_valid: bool,
    expected_status: str,
    expected_action: str,
) -> None:
    result = lwoh.classify_a3_terminal(terminal, evidence_valid=evidence_valid)
    assert result == {"status": expected_status, "action": expected_action}


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _rewrite_freeze_with_valid_sidecar(path: Path, record: dict[str, Any]) -> None:
    payload = lwoh._canonical_json_bytes(record)
    path.write_bytes(payload)
    sidecar = {
        "algorithm": "sha256",
        "report_file": path.name,
        "report_sha256": hashlib.sha256(payload).hexdigest(),
    }
    path.with_suffix(".sha256").write_bytes(lwoh._canonical_json_bytes(sidecar))


def _install_synthetic_freeze_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, Any]:
    """Install only metadata stubs; official A3 report bytes remain opaque."""

    for relative, payload in (
        (lwoh.BASE_PROTOCOL_PATH, b"base protocol\n"),
        (lwoh.BASE_CONFIG_PATH, b"base config\n"),
        (lwoh.AMENDMENT_PROTOCOL_PATH, b"amendment\n"),
        (lwoh.AMENDMENT_CONFIG_PATH, b"overlay\n"),
    ):
        _write(tmp_path / relative, payload)

    # Deliberately non-JSON task reports prove the freeze path treats them as
    # opaque bytes.  Their sidecars are opaque at this phase too.
    _write(tmp_path / lwoh.A3_PRIMARY_REPORT_PATH, b"not-json primary metrics")
    _write(tmp_path / lwoh.A3_PRIMARY_SIDECAR_PATH, b"not-json primary sidecar")
    _write(tmp_path / lwoh.A3_RERUN_REPORT_PATH, b"not-json rerun metrics")
    _write(tmp_path / lwoh.A3_RERUN_SIDECAR_PATH, b"not-json rerun sidecar")

    effective = {"_semantic_delta": {"allowlist_pass": True}}
    seed_validation = {
        "admission": list(lwoh.LWOH_ADMISSION_SEEDS),
        "learning": list(lwoh.LWOH_LEARNING_SEEDS),
        "scratch": list(lwoh.LWOH_SCRATCH_SEEDS),
        "a3_forbidden": list(lwoh.A3_REGISTERED_SEEDS),
        "all_partitions_disjoint": True,
    }
    manifest = {
        "files": {"synthetic": {"bytes": 1, "sha256": "1" * 64}},
        "bundle_sha256": "2" * 64,
    }
    a3_source = {
        "freeze_record": {
            "path": lwoh.A3_FREEZE_RECORD_PATH.as_posix(),
            "bytes": 1,
            "sha256": "3" * 64,
        },
        "source_manifest_sha256": lwoh.A3_SOURCE_MANIFEST_SHA256,
        "source_file_count": 14,
        "pre_freeze_verification": {
            "path": lwoh.A3_PRE_FREEZE_VERIFICATION_PATH.as_posix(),
            "bytes": 1,
            "sha256": "4" * 64,
        },
    }
    phase = {
        "identity": {
            "path": lwoh.A3_PHASE_SEQUENCE_PATH.as_posix(),
            "bytes": 1,
            "sha256": "5" * 64,
        },
        "phase_names": ["tests", "freeze"],
        "phases": [
            {"name": "tests", "artifacts": [{"synthetic": "tests"}]},
            {"name": "freeze", "artifacts": [{"synthetic": "freeze"}]},
        ],
        "payload_sha256": "6" * 64,
    }

    monkeypatch.setattr(lwoh, "_require_hash", lambda *_args, **_kwargs: "f" * 64)
    monkeypatch.setattr(
        lwoh, "load_effective_config", lambda **_kwargs: copy.deepcopy(effective)
    )
    monkeypatch.setattr(
        lwoh,
        "_validate_seed_partitions",
        lambda _effective: copy.deepcopy(seed_validation),
    )
    monkeypatch.setattr(
        lwoh, "_source_manifest", lambda *_args, **_kwargs: copy.deepcopy(manifest)
    )
    monkeypatch.setattr(
        lwoh,
        "_validate_a3_source_freeze",
        lambda _root: copy.deepcopy(a3_source),
    )
    monkeypatch.setattr(
        lwoh,
        "_validate_a3_phase_sequence",
        lambda _root, **_kwargs: copy.deepcopy(phase),
    )
    monkeypatch.setattr(lwoh, "_utc_now", lambda: "2026-09-03T04:00:00.000001Z")
    monkeypatch.setattr(
        lwoh, "_local_now", lambda: "2026-09-03T09:30:00.000001+05:30"
    )
    return {
        "effective": effective,
        "seed_validation": seed_validation,
        "manifest": manifest,
        "a3_source": a3_source,
        "phase": phase,
        "freeze": tmp_path / lwoh.CONTINGENCY_FREEZE_PATH,
    }


def test_freeze_refuses_existing_a3_terminal_before_any_other_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    terminal = tmp_path / lwoh.A3_TERMINAL_PATH
    _write(terminal, b"terminal already exists")

    def forbidden_read(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("freeze inspected another input after seeing the terminal")

    monkeypatch.setattr(lwoh, "_read_json", forbidden_read)
    monkeypatch.setattr(lwoh, "_read_toml", forbidden_read)
    with pytest.raises(RuntimeError, match="terminal artifact already exists"):
        lwoh.create_contingency_freeze(project_root=tmp_path)
    assert not (tmp_path / lwoh.CONTINGENCY_FREEZE_PATH).exists()
    assert not (tmp_path / lwoh.CONTINGENCY_FREEZE_SIDECAR_PATH).exists()


def test_freeze_hashes_but_never_parses_a3_full_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _install_synthetic_freeze_inputs(monkeypatch, tmp_path)
    real_read_json = lwoh._read_json
    parsed_paths: list[Path] = []

    def audited_read_json(path: str | Path, *, purpose: str = "metadata") -> dict[str, Any]:
        parsed_paths.append(Path(path).resolve())
        return real_read_json(path, purpose=purpose)

    monkeypatch.setattr(lwoh, "_read_json", audited_read_json)
    result = lwoh.create_contingency_freeze(
        project_root=tmp_path,
        validation_evidence=_synthetic_external_validation_evidence(),
    )

    forbidden = {
        (tmp_path / lwoh.A3_PRIMARY_REPORT_PATH).resolve(),
        (tmp_path / lwoh.A3_RERUN_REPORT_PATH).resolve(),
        (tmp_path / lwoh.A3_PRIMARY_SIDECAR_PATH).resolve(),
        (tmp_path / lwoh.A3_RERUN_SIDECAR_PATH).resolve(),
    }
    assert forbidden.isdisjoint(parsed_paths)
    assert result["status"] == "CREATED_PENDING_FRESH_PROCESS_VERIFICATION"
    assert result["fresh_process_verified"] is False
    assert result["external_validation_replayed"] is False
    record = json.loads(fixture["freeze"].read_text(encoding="ascii"))
    assert record["blinding"]["a3_task_reports_parsed"] is False
    assert record["blinding"]["a3_scientific_fields_observed"] is False
    for path in forbidden:
        relative = path.relative_to(tmp_path).as_posix()
        assert record["a3_opaque_prerequisites"][relative]["present"] is True
        assert record["a3_opaque_prerequisites"][relative]["sha256"] == _file_sha256(
            path
        )


def test_freeze_requires_fresh_process_and_allows_postterminal_source_additions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _install_synthetic_freeze_inputs(monkeypatch, tmp_path)
    created = lwoh.create_contingency_freeze(
        project_root=tmp_path,
        validation_evidence=_synthetic_external_validation_evidence(),
    )
    assert created["status"] == "CREATED_PENDING_FRESH_PROCESS_VERIFICATION"

    record = json.loads(fixture["freeze"].read_text(encoding="ascii"))
    assert record["creator_process"] == {
        "process_id": lwoh.os.getpid(),
        "process_instance_token": lwoh._PROCESS_INSTANCE_TOKEN,
        "created_utc": record["created_utc"],
    }
    with pytest.raises(RuntimeError, match="separate fresh process"):
        lwoh.verify_contingency_freeze(fixture["freeze"], project_root=tmp_path)

    creator_pid = record["creator_process"]["process_id"]
    monkeypatch.setattr(lwoh.os, "getpid", lambda: creator_pid + 1)
    monkeypatch.setattr(lwoh, "_PROCESS_INSTANCE_TOKEN", "f" * 64)
    replay_calls: list[tuple[Path, list[dict[str, Any]]]] = []

    def replay(
        root: Path, rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        replay_calls.append((root, rows))
        return [{"command": "synthetic fresh replay", "exit_status": 0}]

    monkeypatch.setattr(lwoh, "_replay_external_validation_evidence", replay)
    verified = lwoh.verify_contingency_freeze(
        fixture["freeze"], project_root=tmp_path
    )
    assert verified["status"] == "PASS"
    assert verified["fresh_process_verified"] is True
    assert verified["external_validation_replayed"] is True
    assert replay_calls and replay_calls[0][0] == tmp_path

    _write(tmp_path / lwoh.FUTURE_EXECUTION_RUNNER_PATH, b"postterminal runner\n")
    _write(tmp_path / lwoh.FUTURE_EXECUTION_TEST_PATH, b"postterminal tests\n")
    with pytest.raises(RuntimeError, match="post-activation execution sources exist"):
        lwoh.verify_contingency_freeze(
            fixture["freeze"], project_root=tmp_path, require_terminal_absent=True
        )

    _write(tmp_path / lwoh.A3_TERMINAL_PATH, b"synthetic terminal\n")
    postterminal = lwoh.verify_contingency_freeze(
        fixture["freeze"], project_root=tmp_path, require_terminal_absent=False
    )
    assert postterminal["status"] == "PASS"
    assert postterminal["a3_terminal_present"] is True


def test_freeze_creation_cleans_published_pair_on_terminal_race(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _install_synthetic_freeze_inputs(monkeypatch, tmp_path)

    def racing_verifier(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        _write(tmp_path / lwoh.A3_TERMINAL_PATH, b"raced terminal\n")
        raise RuntimeError("synthetic terminal race")

    monkeypatch.setattr(lwoh, "verify_contingency_freeze", racing_verifier)
    with pytest.raises(RuntimeError, match="synthetic terminal race"):
        lwoh.create_contingency_freeze(
            project_root=tmp_path,
            validation_evidence=_synthetic_external_validation_evidence(),
        )

    assert not fixture["freeze"].exists()
    assert not fixture["freeze"].with_suffix(".sha256").exists()


@pytest.mark.parametrize(
    "tamper", ("self_hash", "binding", "sidecar", "bound_file", "opaque_artifact")
)
def test_freeze_verifier_rejects_self_hash_binding_and_file_tampering(
    tamper: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _install_synthetic_freeze_inputs(monkeypatch, tmp_path)
    lwoh.create_contingency_freeze(
        project_root=tmp_path,
        validation_evidence=_synthetic_external_validation_evidence(),
    )
    freeze = fixture["freeze"]

    if tamper == "sidecar":
        sidecar = json.loads(freeze.with_suffix(".sha256").read_text(encoding="ascii"))
        sidecar["report_sha256"] = "0" * 64
        freeze.with_suffix(".sha256").write_bytes(lwoh._canonical_json_bytes(sidecar))
        match = "sidecar"
    elif tamper == "bound_file":
        (tmp_path / lwoh.BASE_PROTOCOL_PATH).write_bytes(b"changed base protocol\n")
        match = "binding"
    elif tamper == "opaque_artifact":
        (tmp_path / lwoh.A3_PRIMARY_REPORT_PATH).write_bytes(b"changed opaque report")
        match = "opaque A3 prerequisite changed"
    else:
        record = json.loads(freeze.read_text(encoding="ascii"))
        if tamper == "self_hash":
            record["status"] = "tampered"
            match = "self-hash"
        else:
            record["base_v1"]["scientific_design_inherited_unchanged"] = False
            record["freeze_payload_sha256"] = lwoh._record_self_hash(
                record, "freeze_payload_sha256"
            )
            match = "binding"
        _rewrite_freeze_with_valid_sidecar(freeze, record)

    with pytest.raises(RuntimeError, match=f"(?i){match}"):
        lwoh.verify_contingency_freeze(
            freeze,
            project_root=tmp_path,
            require_fresh_process=False,
            replay_external_validation=False,
        )


def test_freeze_and_verifier_reject_wrong_paths_and_bad_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _install_synthetic_freeze_inputs(monkeypatch, tmp_path)
    wrong = tmp_path / "wrong.json"
    with pytest.raises(ValueError, match="canonical path"):
        lwoh.create_contingency_freeze(wrong, project_root=tmp_path)
    with pytest.raises(ValueError, match="canonical path"):
        lwoh.verify_contingency_freeze(wrong, project_root=tmp_path)
    with pytest.raises(ValueError, match="exact command/input/output/exit"):
        lwoh.create_contingency_freeze(
            project_root=tmp_path,
            validation_evidence=({"exit_status": 0},),
        )

    arbitrary = list(_synthetic_external_validation_evidence())
    arbitrary[0] = {
        "command": "python -c print('claimed pass')",
        "input": "",
        "literal_output": "claimed pass\n",
        "exit_status": 0,
    }
    with pytest.raises(ValueError, match="command is not registered"):
        lwoh.create_contingency_freeze(
            project_root=tmp_path,
            validation_evidence=arbitrary,
        )

    scientific = list(_synthetic_external_validation_evidence())
    pytest_index = next(
        index for index, row in enumerate(scientific) if "pytest" in row["command"]
    )
    scientific[pytest_index] = {
        **scientific[pytest_index],
        "literal_output": "selected_condition=rho_0_75\nPYTEST_PASS_COUNT=44\n",
    }
    with pytest.raises(ValueError, match="forbidden A3 scientific field"):
        lwoh.create_contingency_freeze(
            project_root=tmp_path,
            validation_evidence=scientific,
        )
    assert not fixture["freeze"].exists()


def test_source_manifest_contains_only_registered_v2_sources() -> None:
    manifest = lwoh.frozen_source_manifest()
    assert set(manifest["files"]) == set(lwoh.V2_SOURCE_PATHS)
    assert len(manifest["files"]) == len(lwoh.V2_SOURCE_PATHS)
    assert manifest["bundle_sha256"] == lwoh._sha256_json(manifest["files"])
    for relative, identity in manifest["files"].items():
        path = ROOT / relative
        assert identity == {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def test_post_activation_sources_do_not_invalidate_preterminal_manifest(
    tmp_path: Path,
) -> None:
    for relative in lwoh.V2_SOURCE_PATHS:
        _write(tmp_path / relative, f"preterminal:{relative}\n".encode("ascii"))

    frozen = lwoh._source_manifest(tmp_path)
    _write(tmp_path / lwoh.FUTURE_EXECUTION_RUNNER_PATH, b"execution runner\n")
    _write(tmp_path / lwoh.FUTURE_EXECUTION_TEST_PATH, b"execution tests\n")

    assert lwoh._source_manifest(tmp_path) == frozen
    implementation_paths = (
        *lwoh.V2_SOURCE_PATHS,
        lwoh.FUTURE_EXECUTION_RUNNER_PATH.as_posix(),
        lwoh.FUTURE_EXECUTION_TEST_PATH.as_posix(),
    )
    implementation = lwoh._source_manifest(tmp_path, implementation_paths)
    assert set(implementation["files"]) == set(implementation_paths)
    assert implementation["bundle_sha256"] != frozen["bundle_sha256"]

    bound_controller = tmp_path / lwoh.RUNNER_PATH
    bound_controller.write_bytes(bound_controller.read_bytes() + b"drift\n")
    assert lwoh._source_manifest(tmp_path)["bundle_sha256"] != frozen["bundle_sha256"]


def _install_synthetic_terminal_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    terminal_status: str,
    failure_site: str | None = None,
) -> dict[str, Any]:
    from adaptive_learning_substrate import experiment_000_readout_trace_a3 as a3

    terminal = {
        "status": terminal_status,
        "valid_terminal_status": terminal_status == "NO_SELECTION"
        or terminal_status.startswith("SELECTED:"),
        "first_deterministic_payload_sha256": "a" * 64,
        "verification_payload_sha256": "b" * 64,
    }
    terminal_path = tmp_path / lwoh.A3_TERMINAL_PATH
    _write(terminal_path, lwoh._canonical_json_bytes(terminal))

    monkeypatch.setattr(
        lwoh,
        "verify_contingency_freeze",
        lambda **_kwargs: {
            "freeze_file_sha256": "c" * 64,
            "status": "PASS",
            "fresh_process_verified": True,
            "external_validation_replayed": True,
        },
    )
    monkeypatch.setattr(lwoh, "_require_a3_module_root", lambda *_args: None)
    original_import = lwoh.importlib.import_module

    def audited_import(name: str) -> Any:
        assert name == "adaptive_learning_substrate.experiment_000_readout_trace_a3"
        return original_import(name)

    monkeypatch.setattr(lwoh.importlib, "import_module", audited_import)

    manifests = iter(
        (
            {"bundle_sha256": lwoh.A3_SOURCE_MANIFEST_SHA256},
            {
                "bundle_sha256": (
                    "0" * 64
                    if failure_site == "source_changed"
                    else lwoh.A3_SOURCE_MANIFEST_SHA256
                )
            },
        )
    )

    def source_provenance() -> dict[str, Any]:
        if failure_site == "source_initial":
            return {"bundle_sha256": "0" * 64}
        try:
            return next(manifests)
        except StopIteration:
            return {"bundle_sha256": lwoh.A3_SOURCE_MANIFEST_SHA256}

    freeze_info = {"file_sha256": "d" * 64}

    def validate_freeze(**_kwargs: Any) -> dict[str, Any]:
        if failure_site == "freeze":
            raise RuntimeError("synthetic freeze mismatch")
        return copy.deepcopy(freeze_info)

    def validate_phase(_names: Any) -> dict[str, Any]:
        if failure_site == "phase":
            raise RuntimeError("synthetic phase mismatch")
        return {"file_sha256": "3" * 64}

    primary = {
        "status": "PENDING_DETERMINISM_VERIFICATION",
        "selection": {"provisional_terminal_status": terminal_status},
        "seed_results": ["scientific sentinel must not escape"],
        "nondeterministic_provenance": {"rerun_prerequisite_artifacts": None},
    }
    rerun = copy.deepcopy(primary)
    primary_files = {
        "report_file_sha256": "e" * 64,
        "sidecar_file_sha256": "f" * 64,
    }
    rerun_files = {
        "report_file_sha256": "1" * 64,
        "sidecar_file_sha256": "2" * 64,
    }
    rerun["nondeterministic_provenance"]["rerun_prerequisite_artifacts"] = {
        "primary_report_sha256": primary_files["report_file_sha256"],
        "primary_sidecar_sha256": primary_files["sidecar_file_sha256"],
    }

    def load_full(_value: Any, *, expected_path: Any) -> tuple[dict[str, Any], dict[str, str]]:
        if failure_site == "sidecar":
            raise RuntimeError("synthetic sidecar mismatch")
        if expected_path == a3.PRIMARY_REPORT_PATH:
            return copy.deepcopy(primary), copy.deepcopy(primary_files)
        return copy.deepcopy(rerun), copy.deepcopy(rerun_files)

    def validate_report(*_args: Any, **_kwargs: Any) -> None:
        if failure_site == "replay":
            raise RuntimeError("synthetic replay mismatch")

    validator_calls: list[dict[str, Any]] = []
    smoke_calls: list[dict[str, Any]] = []

    def validate_terminal(*_args: Any, **kwargs: Any) -> None:
        validator_calls.append(kwargs)
        if failure_site == "terminal":
            raise RuntimeError("synthetic terminal mismatch")

    def validate_smoke(**kwargs: Any) -> dict[str, Any]:
        smoke_calls.append(kwargs)
        if failure_site == "smoke":
            raise RuntimeError("synthetic smoke mismatch")
        return {"status": "PASS"}

    def forbidden_verifier(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("contingency loader invoked the A3 terminal verifier")

    monkeypatch.setattr(a3, "_source_provenance", source_provenance)
    monkeypatch.setattr(a3, "_validate_freeze_record", validate_freeze)
    monkeypatch.setattr(a3, "_validate_phase_sequence", validate_phase)
    monkeypatch.setattr(a3, "_load_full_input", load_full)
    monkeypatch.setattr(a3, "_validate_report_mapping", validate_report)
    monkeypatch.setattr(a3, "_validate_smoke_verification", validate_smoke)
    monkeypatch.setattr(a3, "_validate_determinism_verification", validate_terminal)
    monkeypatch.setattr(a3, "_file_sha256", lambda _path: "3" * 64)
    monkeypatch.setattr(a3, "_single_thread_blas_state", dict)
    monkeypatch.setattr(a3, "verify_deterministic_full_runs", forbidden_verifier)

    committed_hashes = {
        lwoh.A3_PRIMARY_REPORT_PATH.as_posix(): primary_files["report_file_sha256"],
        lwoh.A3_PRIMARY_SIDECAR_PATH.as_posix(): primary_files["sidecar_file_sha256"],
        lwoh.A3_RERUN_REPORT_PATH.as_posix(): rerun_files["report_file_sha256"],
        lwoh.A3_RERUN_SIDECAR_PATH.as_posix(): rerun_files["sidecar_file_sha256"],
        lwoh.A3_TERMINAL_PATH.as_posix(): _file_sha256(terminal_path),
    }

    def opaque_identity(_root: Path, path: str | Path) -> dict[str, Any]:
        relative = Path(path).as_posix()
        return {
            "path": relative,
            "present": True,
            "bytes": 1,
            "mtime_ns": 1,
            "sha256": committed_hashes[relative],
        }

    monkeypatch.setattr(lwoh, "_opaque_identity", opaque_identity)
    return {
        "validator_calls": validator_calls,
        "smoke_calls": smoke_calls,
        "terminal": terminal,
    }


def test_a3_validator_module_must_come_from_same_project_root(tmp_path: Path) -> None:
    from adaptive_learning_substrate import experiment_000_readout_trace_a3 as a3

    lwoh._require_a3_module_root(ROOT, a3)
    with pytest.raises(RuntimeError, match="project roots differ"):
        lwoh._require_a3_module_root(tmp_path, a3)


def test_loader_returns_only_terminal_and_provenance_hashes_without_rerunning_a3(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _install_synthetic_terminal_validation(
        monkeypatch, tmp_path, terminal_status="NO_SELECTION"
    )
    binding = lwoh.load_verified_a3_no_selection(project_root=tmp_path)

    assert binding["status"] == "ACTIVATED_A3_NO_SELECTION"
    assert binding["action"] == "ACTIVATE_LWOH_L1_V2"
    assert binding["terminal_status"] == "NO_SELECTION"
    assert set(binding) == {
        "status",
        "action",
        "terminal_status",
        "a3_source_manifest_sha256",
        "a3_freeze_record_sha256",
        "a3_phase_sequence_sha256",
        "primary_report_sha256",
        "primary_sidecar_sha256",
        "rerun_report_sha256",
        "rerun_sidecar_sha256",
        "deterministic_payload_sha256",
        "terminal_file_sha256",
        "terminal_payload_sha256",
        "contingency_freeze_sha256",
    }
    assert fixture["validator_calls"]
    assert len(fixture["validator_calls"]) == 2
    assert all(
        call["require_selected"] is False for call in fixture["validator_calls"]
    )
    assert len(fixture["smoke_calls"]) == 2
    assert all(
        call["smoke_path"].as_posix()
        == lwoh.A3_SMOKE_REPORT_PATH.as_posix()
        and call["verification_path"].as_posix()
        == lwoh.A3_SMOKE_VERIFICATION_PATH.as_posix()
        for call in fixture["smoke_calls"]
    )
    assert not {
        "selection",
        "selected_condition",
        "passing_conditions",
        "seed_results",
        "aggregate_conditions",
        "gate_values",
        "effect_sizes",
    }.intersection(binding)


def test_terminal_loader_remains_reusable_after_postactivation_sources_are_added(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_synthetic_terminal_validation(
        monkeypatch, tmp_path, terminal_status="NO_SELECTION"
    )
    committed = lwoh.commit_a3_no_selection_activation(project_root=tmp_path)
    assert committed["status"] == "PASS"
    assert committed["activation_provenance"]["status"] == (
        "ACTIVATED_A3_NO_SELECTION"
    )
    assert (tmp_path / lwoh.ACTIVATION_PROVENANCE_PATH).is_file()
    assert (tmp_path / lwoh.ACTIVATION_PROVENANCE_SIDECAR_PATH).is_file()

    _write(tmp_path / lwoh.FUTURE_EXECUTION_RUNNER_PATH, b"postactivation runner\n")
    _write(tmp_path / lwoh.FUTURE_EXECUTION_TEST_PATH, b"postactivation tests\n")

    first = lwoh.load_verified_a3_no_selection(project_root=tmp_path)
    second = lwoh.load_verified_a3_no_selection(project_root=tmp_path)
    assert first == second
    assert first["status"] == "ACTIVATED_A3_NO_SELECTION"
    assert first["terminal_status"] == "NO_SELECTION"


def test_first_activation_rejects_early_future_sources_and_wrong_receipt_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_synthetic_terminal_validation(
        monkeypatch, tmp_path, terminal_status="NO_SELECTION"
    )
    with pytest.raises(ValueError, match="canonical path"):
        lwoh.commit_a3_no_selection_activation(
            tmp_path / "wrong-activation.json", project_root=tmp_path
        )

    _write(tmp_path / lwoh.FUTURE_EXECUTION_RUNNER_PATH, b"too early\n")
    with pytest.raises(RuntimeError, match="post-activation execution sources exist"):
        lwoh.commit_a3_no_selection_activation(project_root=tmp_path)
    assert not (tmp_path / lwoh.ACTIVATION_PROVENANCE_PATH).exists()
    assert not (tmp_path / lwoh.ACTIVATION_PROVENANCE_SIDECAR_PATH).exists()


def test_activation_receipt_tamper_and_fresh_process_revalidation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_synthetic_terminal_validation(
        monkeypatch, tmp_path, terminal_status="NO_SELECTION"
    )
    lwoh.commit_a3_no_selection_activation(project_root=tmp_path)
    receipt = tmp_path / lwoh.ACTIVATION_PROVENANCE_PATH

    creator = json.loads(receipt.read_text(encoding="ascii"))["creator_process"]
    monkeypatch.setattr(lwoh.os, "getpid", lambda: creator["process_id"] + 1)
    monkeypatch.setattr(lwoh, "_PROCESS_INSTANCE_TOKEN", "e" * 64)
    fresh = lwoh.verify_a3_activation_provenance(project_root=tmp_path)
    assert fresh["status"] == "PASS"
    assert fresh["activation_provenance"]["terminal_status"] == "NO_SELECTION"

    record = json.loads(receipt.read_text(encoding="ascii"))
    record["activation_provenance"]["terminal_status"] = "SELECTED"
    record["activation_payload_sha256"] = lwoh._record_self_hash(
        record, "activation_payload_sha256"
    )
    _rewrite_freeze_with_valid_sidecar(receipt, record)
    with pytest.raises(RuntimeError, match="activation binding|exact NO_SELECTION"):
        lwoh.verify_a3_activation_provenance(project_root=tmp_path)


def test_activation_receipt_replacement_during_a3_replay_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_synthetic_terminal_validation(
        monkeypatch, tmp_path, terminal_status="NO_SELECTION"
    )
    lwoh.commit_a3_no_selection_activation(project_root=tmp_path)
    receipt = tmp_path / lwoh.ACTIVATION_PROVENANCE_PATH
    record = json.loads(receipt.read_text(encoding="ascii"))
    frozen_binding = copy.deepcopy(record["activation_provenance"])

    def replace_receipt_during_replay(_root: Path) -> dict[str, Any]:
        replacement = copy.deepcopy(record)
        replacement["local_timezone"] = "replaced-during-A3-replay"
        replacement["activation_payload_sha256"] = lwoh._record_self_hash(
            replacement, "activation_payload_sha256"
        )
        _rewrite_freeze_with_valid_sidecar(receipt, replacement)
        return copy.deepcopy(frozen_binding)

    monkeypatch.setattr(
        lwoh, "_validated_a3_terminal_binding", replace_receipt_during_replay
    )
    with pytest.raises(RuntimeError, match="changed during validation"):
        lwoh.verify_a3_activation_provenance(project_root=tmp_path)


def test_selected_a3_terminal_cancels_but_never_loads_lwoh_activation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_synthetic_terminal_validation(
        monkeypatch, tmp_path, terminal_status="SELECTED:rho_0_75"
    )
    result = lwoh.evaluate_a3_contingency(project_root=tmp_path)
    assert result["status"] == "CANCELLED_A3_SELECTED"
    assert result["action"] == "CANCEL_LWOH_L1_V2"
    serialized = json.dumps(result, sort_keys=True)
    assert "rho_0_75" not in serialized
    assert "terminal_status" not in result
    assert not {
        "selection",
        "selected_condition",
        "passing_conditions",
        "condition_rows",
        "seed_results",
        "gate_values",
        "effect_sizes",
    }.intersection(result)
    with pytest.raises(RuntimeError, match="activation binding|does not activate"):
        lwoh.load_verified_a3_no_selection(project_root=tmp_path)
    with pytest.raises(RuntimeError, match="activation binding|exact NO_SELECTION"):
        lwoh.commit_a3_no_selection_activation(project_root=tmp_path)
    assert not (tmp_path / lwoh.ACTIVATION_PROVENANCE_PATH).exists()


@pytest.mark.parametrize(
    ("terminal_status", "failure_site"),
    (
        ("PENDING_DETERMINISM_VERIFICATION", None),
        ("NONDETERMINISTIC_INVALID", None),
        ("INVALID_PROCEDURE_NONSELECTING", None),
        ("NO_SELECTION", "source_initial"),
        ("NO_SELECTION", "source_changed"),
        ("NO_SELECTION", "freeze"),
        ("NO_SELECTION", "phase"),
        ("NO_SELECTION", "sidecar"),
        ("NO_SELECTION", "replay"),
        ("NO_SELECTION", "smoke"),
        ("NO_SELECTION", "terminal"),
    ),
)
def test_invalid_provisional_nondeterministic_or_mismatched_a3_rejects_activation(
    terminal_status: str,
    failure_site: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_synthetic_terminal_validation(
        monkeypatch,
        tmp_path,
        terminal_status=terminal_status,
        failure_site=failure_site,
    )
    result = lwoh.evaluate_a3_contingency(project_root=tmp_path)
    assert result["status"] == "INVALID_A3_DO_NOT_ACTIVATE_LWOH"
    assert result["action"] == "REPAIR_A3_DO_NOT_ACTIVATE_LWOH"
    assert "validation_error_type" in result


def test_missing_a3_terminal_rejects_activation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        lwoh,
        "verify_contingency_freeze",
        lambda **_kwargs: {
            "freeze_file_sha256": "c" * 64,
            "status": "PASS",
            "fresh_process_verified": True,
            "external_validation_replayed": True,
        },
    )
    result = lwoh.evaluate_a3_contingency(project_root=tmp_path)
    assert result == {
        "status": "INVALID_A3_DO_NOT_ACTIVATE_LWOH",
        "action": "REPAIR_A3_DO_NOT_ACTIVATE_LWOH",
        "validation_error_type": "RuntimeError",
    }


def test_cli_rejects_missing_unknown_and_scientific_arguments() -> None:
    for argv in ((), ("unknown",), ("freeze", "--seeds", "90")):
        with pytest.raises(SystemExit) as raised:
            lwoh.main(argv)
        assert raised.value.code == 2
