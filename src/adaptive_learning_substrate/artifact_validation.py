"""Canonical validation for persisted project evidence and convenience outputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .frozen_history import (
    FROZEN_RECURRENT_SOURCE_RESOURCE,
    FROZEN_RECURRENT_SOURCE_SHA256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path("artifacts")
CANONICAL_STAGE1 = Path("artifacts/stage1/ccf")
LATEST_RUN = Path("runs/latest")
A3_TERMINAL = Path(
    "artifacts/experiment_000/readout_trace_a3/DETERMINISM_VERIFICATION.json"
)
A3_PRIMARY = Path(
    "artifacts/experiment_000/readout_trace_a3/custom_seeds_105_109_pairs_100.json"
)
A3_RERUN = Path(
    "artifacts/experiment_000/readout_trace_a3/custom_seeds_105_109_pairs_100_rerun.json"
)
A3_FREEZE = Path("artifacts/experiment_000/readout_trace_a3/FREEZE_RECORD.json")
A3_PHASE_SEQUENCE = Path(
    "artifacts/experiment_000/readout_trace_a3/PHASE_SEQUENCE.json"
)
LWOH_SOURCE_FREEZE = Path(
    "artifacts/experiment_000/lwoh_l1_v2/PRE_TERMINAL_CONTINGENCY_FREEZE.json"
)
LWOH_V3_ROOT = Path("artifacts/experiment_000/lwoh_l1_v3")
LWOH_V3_LINEAGE = Path(
    "artifacts/evidence_integrity_repair_2026-09-04/PRE_METRIC_LINEAGE.json"
)
CANONICAL_STATUS = Path("docs/CANONICAL_STATUS.md")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
LATEST_FILES = ("config.snapshot.json", "events.jsonl", "report.json")
A3_PHASE_ORDER = (
    "tests",
    "freeze",
    "persisted_smoke_and_sidecar",
    "machine_smoke_verification",
    "full",
    "rerun",
    "determinism_verification",
)
A3_SELECTED_STATUSES = {
    "SELECTED:rho_0_25",
    "SELECTED:rho_0_50",
    "SELECTED:rho_0_75",
    "SELECTED:rho_0_90",
    "SELECTED:rho_0_95",
}
LWOH_V3_STAGE_CONTRACT: tuple[
    tuple[str, str, frozenset[str], str], ...
] = (
    (
        "PRE_METRIC_FREEZE.json",
        "experiment-000-lwoh-l1-v3-pre-metric-freeze-v1",
        frozenset({"FROZEN_PRE_METRIC"}),
        "freeze_payload_sha256",
    ),
    (
        "ACTIVATION_VERIFICATION.json",
        "experiment-000-lwoh-l1-v3-activation-verification-v1",
        frozenset({"ACTIVATED_A3_NO_SELECTION_V3"}),
        "verification_payload_sha256",
    ),
    (
        "ADMISSION_DETERMINISM_VERIFICATION.json",
        "experiment-000-lwoh-l1-v3-admission-determinism-verification-v1",
        frozenset({"LWOH_ADMISSION_PASS", "LWOH_ADMISSION_FAIL"}),
        "verification_payload_sha256",
    ),
    (
        "IMPLEMENTATION_SOURCE_FREEZE.json",
        "experiment-000-lwoh-l1-v3-implementation-source-freeze-v1",
        frozenset({"FROZEN_FOR_LEARNING"}),
        "freeze_payload_sha256",
    ),
    (
        "LEARNING_DETERMINISM_VERIFICATION.json",
        "experiment-000-lwoh-l1-v3-learning-determinism-verification-v1",
        frozenset({"LWOH_LEARNING_PASS", "LWOH_LEARNING_FAIL"}),
        "verification_payload_sha256",
    ),
)
LWOH_V3_READINESS_PATH = LWOH_V3_ROOT / "READINESS_VERIFICATION.json"
LWOH_V3_READINESS_SCHEMA = "experiment-000-lwoh-l1-v3-readiness-v1"
LWOH_V3_READINESS_KEYS = {
    "schema_version",
    "status",
    "readiness_before",
    "readiness_after",
    "partial_score",
    "admission_seeds",
    "learning_seeds",
    "delays",
    "baselines",
    "deterministic_admission",
    "deterministic_learning",
    "all_admission_gates_pass",
    "all_learning_gates_pass",
    "all_robustness_gates_pass",
    "all_integrity_gates_pass",
    "lineage_handoff_path",
    "lineage_handoff_sha256",
    "chain_artifacts",
    "verification_payload_sha256",
}
LWOH_V3_BASELINES = (
    "random",
    "frozen_head",
    "latest_state",
    "independent_label",
    "ridge",
    "visible_cue",
)
LWOH_V3_DEEP_BOOLEAN_FIELDS = (
    "raw_admission_primary_valid",
    "raw_admission_rerun_valid",
    "raw_learning_primary_valid",
    "raw_learning_rerun_valid",
    "all_sidecars_valid",
    "assigned_seeds_exact",
    "per_seed_rows_complete",
    "admission_gates_recomputed",
    "learning_gates_recomputed",
    "statistical_gates_recomputed",
    "robustness_gates_recomputed",
    "controls_recomputed",
    "distinct_processes_verified",
    "phase_ledger_valid",
    "source_manifest_valid",
    "source_manifest_stable",
    "deterministic_payloads_equal",
    "absence_branches_valid",
    "condition_ids_exact",
    "official_and_forbidden_seeds_exact",
    "replay_verified",
    "process_freshness_verified",
)
LWOH_V3_DEEP_RESULT_KEYS = {
    "status",
    "readiness_before",
    "readiness_after",
    *LWOH_V3_DEEP_BOOLEAN_FIELDS,
}
LWOH_V3_ADMISSION_DEEP_REQUIRED = {
    "raw_admission_primary_valid",
    "raw_admission_rerun_valid",
    "all_sidecars_valid",
    "assigned_seeds_exact",
    "per_seed_rows_complete",
    "admission_gates_recomputed",
    "controls_recomputed",
    "distinct_processes_verified",
    "phase_ledger_valid",
    "source_manifest_valid",
    "source_manifest_stable",
    "deterministic_payloads_equal",
    "absence_branches_valid",
    "condition_ids_exact",
    "official_and_forbidden_seeds_exact",
    "replay_verified",
    "process_freshness_verified",
}


class EvidenceValidationError(RuntimeError):
    """Raised when a persisted evidence binding is incomplete or inconsistent."""


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _record_self_hash(record: Mapping[str, Any], field: str) -> str:
    payload = dict(record)
    payload.pop(field, None)
    return _sha256_json(payload)


def _stable_bytes(path: Path) -> bytes:
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(payload) != after.st_size:
        raise EvidenceValidationError(f"artifact changed while it was read: {path}")
    return payload


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _json_object_bytes(payload: bytes, *, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8-sig"), parse_constant=_reject_json_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise EvidenceValidationError(f"invalid JSON evidence artifact: {path}") from exc
    if not isinstance(value, dict):
        raise EvidenceValidationError(f"JSON evidence artifact is not an object: {path}")
    return value


def _json_object(path: Path) -> dict[str, Any]:
    try:
        payload = _stable_bytes(path)
    except OSError as exc:
        raise EvidenceValidationError(f"invalid JSON evidence artifact: {path}") from exc
    return _json_object_bytes(payload, path=path)


def validate_sidecars(
    *, project_root: str | Path = PROJECT_ROOT, artifact_root: str | Path = ARTIFACT_ROOT
) -> list[dict[str, Any]]:
    """Validate every JSON ``*.sha256`` file against the adjacent named report."""

    root = Path(project_root).resolve()
    directory = _resolve(root, artifact_root)
    if not directory.is_dir():
        raise EvidenceValidationError(f"artifact root is absent: {directory}")
    rows: list[dict[str, Any]] = []
    targets: set[Path] = set()
    for sidecar in sorted(directory.rglob("*.sha256")):
        relative_sidecar = sidecar.relative_to(directory)
        # Transaction baselines and exercised rollback copies preserve a
        # deliberately non-live evidence view.  Their sidecars are package
        # internals, not members of the active evidence graph.
        if any(
            part == "BASELINE" or part.startswith("ROLLBACK_TEST_COPY")
            for part in relative_sidecar.parts
        ):
            continue
        try:
            sidecar_bytes = _stable_bytes(sidecar)
        except OSError as exc:
            raise EvidenceValidationError(
                f"invalid JSON evidence artifact: {sidecar}"
            ) from exc
        binding = _json_object_bytes(sidecar_bytes, path=sidecar)
        if set(binding) != {"algorithm", "report_file", "report_sha256"}:
            raise EvidenceValidationError(f"sidecar schema differs: {sidecar}")
        report_file = binding.get("report_file")
        expected = binding.get("report_sha256")
        if (
            binding.get("algorithm") != "sha256"
            or not isinstance(report_file, str)
            or not report_file
            or Path(report_file).name != report_file
            or not isinstance(expected, str)
            or SHA256_RE.fullmatch(expected) is None
        ):
            raise EvidenceValidationError(f"sidecar binding is invalid: {sidecar}")
        report = sidecar.with_name(report_file)
        if report in targets:
            raise EvidenceValidationError(f"multiple sidecars bind the same report: {report}")
        targets.add(report)
        if not report.is_file():
            raise EvidenceValidationError(f"sidecar target is absent: {report}")
        report_bytes = _stable_bytes(report)
        observed = _sha256_bytes(report_bytes)
        if observed != expected:
            raise EvidenceValidationError(
                f"sidecar hash mismatch: {sidecar}; expected={expected}; observed={observed}"
            )
        # Reopen the exact bytes after hashing the target.  Comparing parsed
        # dictionaries would miss a concurrent replacement that preserved JSON
        # semantics while changing the evidence file itself.
        if _stable_bytes(sidecar) != sidecar_bytes:
            raise EvidenceValidationError(f"sidecar changed during validation: {sidecar}")
        rows.append(
            {
                "sidecar": sidecar.relative_to(root).as_posix(),
                "report": report.relative_to(root).as_posix(),
                "bytes": len(report_bytes),
                "sha256": observed,
            }
        )
    if not rows:
        raise EvidenceValidationError("no evidence sidecars were discovered")
    return rows


def validate_latest_run(*, project_root: str | Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    """Require ``runs/latest`` to be byte-identical to the canonical Stage-1 run."""

    root = Path(project_root).resolve()
    rows: list[dict[str, Any]] = []
    for name in LATEST_FILES:
        canonical = _resolve(root, CANONICAL_STAGE1 / name)
        latest = _resolve(root, LATEST_RUN / name)
        if not canonical.is_file() or not latest.is_file():
            raise EvidenceValidationError(f"canonical/latest run file is absent: {name}")
        canonical_bytes = _stable_bytes(canonical)
        latest_bytes = _stable_bytes(latest)
        if latest_bytes != canonical_bytes:
            raise EvidenceValidationError(f"runs/latest is stale: {name}")
        rows.append(
            {
                "path": latest.relative_to(root).as_posix(),
                "bytes": len(latest_bytes),
                "sha256": _sha256_bytes(latest_bytes),
            }
        )
    return rows


def validate_frozen_source_snapshots(
    *, project_root: str | Path = PROJECT_ROOT
) -> list[dict[str, Any]]:
    """Validate packaged historical source bytes from the requested project root."""

    root = Path(project_root).resolve()
    path = (
        root
        / "src"
        / "adaptive_learning_substrate"
        / Path(FROZEN_RECURRENT_SOURCE_RESOURCE)
    )
    if not path.is_file():
        raise EvidenceValidationError(f"frozen source snapshot is absent: {path}")
    payload = _stable_bytes(path)
    observed = _sha256_bytes(payload)
    if observed != FROZEN_RECURRENT_SOURCE_SHA256:
        raise EvidenceValidationError(
            f"frozen source snapshot hash mismatch: {path}; "
            f"expected={FROZEN_RECURRENT_SOURCE_SHA256}; observed={observed}"
        )
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": len(payload),
            "sha256": observed,
        }
    ]


def validate_lwoh_source_freeze_state(
    *, project_root: str | Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Report whether current files still match the prospective LWOH V2 freeze."""

    root = Path(project_root).resolve()
    path = _resolve(root, LWOH_SOURCE_FREEZE)
    freeze_bytes = _stable_bytes(path)
    freeze = _json_object_bytes(freeze_bytes, path=path)
    if (
        freeze.get("schema_version")
        != "experiment-000-lwoh-l1-a3-preterminal-freeze-v2"
        or freeze.get("status") != "FROZEN_BLINDED_PRE_TERMINAL"
        or freeze.get("freeze_payload_sha256")
        != _record_self_hash(freeze, "freeze_payload_sha256")
    ):
        raise EvidenceValidationError("LWOH V2 source-freeze record is invalid")
    manifest = freeze.get("v2_source_manifest")
    if not isinstance(manifest, Mapping):
        raise EvidenceValidationError("LWOH V2 source manifest is absent")
    frozen_bundle = manifest.get("bundle_sha256")
    files = manifest.get("files")
    if (
        not isinstance(frozen_bundle, str)
        or SHA256_RE.fullmatch(frozen_bundle) is None
        or not isinstance(files, Mapping)
        or not files
    ):
        raise EvidenceValidationError("LWOH V2 source manifest is malformed")

    changed_files: list[str] = []
    observed_files: list[dict[str, Any]] = []
    for relative, raw_binding in sorted(files.items()):
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(raw_binding, Mapping)
        ):
            raise EvidenceValidationError("LWOH V2 source-file binding is malformed")
        expected_hash = raw_binding.get("sha256")
        expected_bytes = raw_binding.get("bytes")
        if (
            not isinstance(expected_hash, str)
            or SHA256_RE.fullmatch(expected_hash) is None
            or isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
        ):
            raise EvidenceValidationError(
                f"LWOH V2 source-file binding is invalid: {relative}"
            )
        source_path = _resolve(root, relative)
        if source_path.is_file():
            payload = _stable_bytes(source_path)
            observed_hash: str | None = _sha256_bytes(payload)
            observed_bytes: int | None = len(payload)
        else:
            observed_hash = None
            observed_bytes = None
        matches = observed_hash == expected_hash and observed_bytes == expected_bytes
        if not matches:
            changed_files.append(relative)
        observed_files.append(
            {
                "path": relative,
                "matches_freeze": matches,
                "expected_sha256": expected_hash,
                "observed_sha256": observed_hash,
            }
        )
    return {
        "path": path.relative_to(root).as_posix(),
        "file_sha256": _sha256_bytes(freeze_bytes),
        "frozen_bundle_sha256": frozen_bundle,
        "status": "CURRENT" if not changed_files else "DRIFTED",
        "changed_file_count": len(changed_files),
        "changed_files": changed_files,
        "files": observed_files,
    }


def _validated_canonical_record_with_sidecar(
    *,
    root: Path,
    relative_path: Path,
    schema: str,
    statuses: frozenset[str],
    self_hash_field: str,
) -> dict[str, Any]:
    path = _resolve(root, relative_path)
    sidecar = path.with_suffix(".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise EvidenceValidationError(
            f"LWOH V3 stage or sidecar is absent: {relative_path.as_posix()}"
        )
    record_bytes = _stable_bytes(path)
    record = _json_object_bytes(record_bytes, path=path)
    if record_bytes != _canonical_json_bytes(record) + b"\n":
        raise EvidenceValidationError(
            f"LWOH V3 stage JSON is not canonical: {relative_path.as_posix()}"
        )
    if (
        record.get("schema_version") != schema
        or record.get("status") not in statuses
        or record.get(self_hash_field) != _record_self_hash(record, self_hash_field)
    ):
        raise EvidenceValidationError(
            f"LWOH V3 stage contract is invalid: {relative_path.as_posix()}"
        )

    sidecar_bytes = _stable_bytes(sidecar)
    sidecar_record = _json_object_bytes(sidecar_bytes, path=sidecar)
    if (
        sidecar_bytes != _canonical_json_bytes(sidecar_record) + b"\n"
        or set(sidecar_record) != {"algorithm", "report_file", "report_sha256"}
        or sidecar_record.get("algorithm") != "sha256"
        or sidecar_record.get("report_file") != path.name
        or sidecar_record.get("report_sha256") != _sha256_bytes(record_bytes)
        or _stable_bytes(sidecar) != sidecar_bytes
    ):
        raise EvidenceValidationError(
            f"LWOH V3 stage sidecar is invalid: {relative_path.as_posix()}"
        )
    return {
        "path": relative_path.as_posix(),
        "file_sha256": _sha256_bytes(record_bytes),
        "sidecar_sha256": _sha256_bytes(sidecar_bytes),
        "record": record,
    }


def _validated_lwoh_v3_deep_result(
    *, root: Path, status: str, expected_readiness: int
) -> dict[str, Any]:
    try:
        module = importlib.import_module(
            "adaptive_learning_substrate.experiment_000_lwoh_l1_execution_v3"
        )
        validator = module.validate_official_v3_chain
        result = validator(project_root=root)
    except Exception as exc:
        raise EvidenceValidationError("LWOH V3 deep verifier failed") from exc
    if not isinstance(result, Mapping) or set(result) != LWOH_V3_DEEP_RESULT_KEYS:
        raise EvidenceValidationError("LWOH V3 deep verifier result keys differ")
    if (
        result.get("status") != status
        or result.get("readiness_before") != 21
        or result.get("readiness_after") != expected_readiness
        or any(type(result.get(field)) is not bool for field in LWOH_V3_DEEP_BOOLEAN_FIELDS)
    ):
        raise EvidenceValidationError("LWOH V3 deep verifier status/readiness differs")
    required_true = (
        LWOH_V3_ADMISSION_DEEP_REQUIRED
        if status == "LWOH_ADMISSION_FAIL"
        else set(LWOH_V3_DEEP_BOOLEAN_FIELDS)
    )
    if any(result.get(field) is not True for field in required_true):
        raise EvidenceValidationError("LWOH V3 deep verification gate is false")
    return dict(result)


def validate_lwoh_v3_state(
    *, project_root: str | Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Fail closed on partial V3 evidence and admit readiness 25 only on full PASS."""

    root = Path(project_root).resolve()
    v3_root = _resolve(root, LWOH_V3_ROOT)
    if not v3_root.exists():
        return {
            "state": "ABSENT",
            "status": None,
            "readiness_before": 21,
            "readiness_after": 21,
            "path": LWOH_V3_ROOT.as_posix(),
            "chain_artifacts": {},
        }
    if not v3_root.is_dir():
        raise EvidenceValidationError("LWOH V3 artifact root is not a directory")

    readiness = _validated_canonical_record_with_sidecar(
        root=root,
        relative_path=LWOH_V3_READINESS_PATH,
        schema=LWOH_V3_READINESS_SCHEMA,
        statuses=frozenset(
            {"LWOH_ADMISSION_FAIL", "LWOH_LEARNING_FAIL", "LWOH_LEARNING_PASS"}
        ),
        self_hash_field="verification_payload_sha256",
    )
    record = readiness["record"]
    if set(record) != LWOH_V3_READINESS_KEYS:
        raise EvidenceValidationError("LWOH V3 readiness schema keys differ")
    status = record["status"]
    expected_after = 25 if status == "LWOH_LEARNING_PASS" else 21
    if (
        record.get("readiness_before") != 21
        or record.get("readiness_after") != expected_after
        or record.get("partial_score") is not False
        or record.get("admission_seeds") != list(range(90, 95))
        or record.get("learning_seeds") != list(range(95, 105))
        or record.get("delays") != [4, 8, 16]
        or record.get("baselines") != list(LWOH_V3_BASELINES)
        or record.get("lineage_handoff_path") != LWOH_V3_LINEAGE.as_posix()
    ):
        raise EvidenceValidationError("LWOH V3 readiness constants differ")

    lineage_path = _resolve(root, LWOH_V3_LINEAGE)
    if not lineage_path.is_file():
        raise EvidenceValidationError("LWOH V3 lineage handoff is absent")
    lineage_bytes = _stable_bytes(lineage_path)
    if record.get("lineage_handoff_sha256") != _sha256_bytes(lineage_bytes):
        raise EvidenceValidationError("LWOH V3 lineage handoff hash differs")

    required_stage_count = 3 if status == "LWOH_ADMISSION_FAIL" else 5
    required_contracts = LWOH_V3_STAGE_CONTRACT[:required_stage_count]
    forbidden_contracts = LWOH_V3_STAGE_CONTRACT[required_stage_count:]
    for name, _, _, _ in forbidden_contracts:
        path = _resolve(root, LWOH_V3_ROOT / name)
        if path.exists() or path.with_suffix(".sha256").exists():
            raise EvidenceValidationError(
                f"LWOH V3 post-admission artifact exists after admission failure: {name}"
            )

    stages: list[dict[str, Any]] = []
    for name, schema, statuses, self_field in required_contracts:
        stages.append(
            _validated_canonical_record_with_sidecar(
                root=root,
                relative_path=LWOH_V3_ROOT / name,
                schema=schema,
                statuses=statuses,
                self_hash_field=self_field,
            )
        )
    admission_status = stages[2]["record"]["status"]
    if (
        status == "LWOH_ADMISSION_FAIL"
        and admission_status != "LWOH_ADMISSION_FAIL"
    ) or (
        status != "LWOH_ADMISSION_FAIL"
        and admission_status != "LWOH_ADMISSION_PASS"
    ):
        raise EvidenceValidationError("LWOH V3 admission/readiness status differs")
    if required_stage_count == 5 and stages[4]["record"]["status"] != status:
        raise EvidenceValidationError("LWOH V3 learning/readiness status differs")
    if required_stage_count == 5:
        pre_metric_manifest = stages[0]["record"].get("source_manifest_sha256")
        implementation_manifest = stages[3]["record"].get(
            "source_manifest_sha256"
        )
        if (
            not isinstance(pre_metric_manifest, str)
            or SHA256_RE.fullmatch(pre_metric_manifest) is None
            or implementation_manifest != pre_metric_manifest
        ):
            raise EvidenceValidationError("LWOH V3 source manifests differ")

    expected_chain = {stage["path"]: stage["file_sha256"] for stage in stages}
    if record.get("chain_artifacts") != expected_chain:
        raise EvidenceValidationError("LWOH V3 readiness chain hash mapping differs")

    boolean_fields = (
        "deterministic_admission",
        "deterministic_learning",
        "all_admission_gates_pass",
        "all_learning_gates_pass",
        "all_robustness_gates_pass",
        "all_integrity_gates_pass",
    )
    if any(type(record.get(field)) is not bool for field in boolean_fields):
        raise EvidenceValidationError("LWOH V3 readiness gate values must be booleans")
    observed = tuple(record[field] for field in boolean_fields)
    if status == "LWOH_ADMISSION_FAIL":
        expected = (True, False, False, False, False, False)
        if observed != expected:
            raise EvidenceValidationError("LWOH V3 admission-fail gates differ")
    elif status == "LWOH_LEARNING_FAIL":
        if observed[:3] != (True, True, True) or all(observed[3:]):
            raise EvidenceValidationError("LWOH V3 learning-fail gates differ")
    elif observed != (True, True, True, True, True, True):
        raise EvidenceValidationError("LWOH V3 learning-pass gates differ")

    deep_result = _validated_lwoh_v3_deep_result(
        root=root, status=status, expected_readiness=expected_after
    )

    return {
        "state": "TERMINAL",
        "status": status,
        "readiness_before": 21,
        "readiness_after": expected_after,
        "path": LWOH_V3_ROOT.as_posix(),
        "readiness_file_sha256": readiness["file_sha256"],
        "lineage_handoff_sha256": _sha256_bytes(lineage_bytes),
        "chain_artifacts": expected_chain,
        "deep_verification": deep_result,
    }


def _validated_a3_report_binding(
    *,
    root: Path,
    report_path: Path,
    terminal: Mapping[str, Any],
    prefix: str,
) -> dict[str, Any]:
    target = _resolve(root, report_path)
    sidecar = target.with_suffix(".sha256")
    sidecar_bytes = _stable_bytes(sidecar)
    binding = _json_object_bytes(sidecar_bytes, path=sidecar)
    if set(binding) != {"algorithm", "report_file", "report_sha256"}:
        raise EvidenceValidationError(f"A3 report sidecar schema differs: {sidecar}")
    report_bytes = _stable_bytes(target)
    report_file_hash = _sha256_bytes(report_bytes)
    if (
        binding.get("algorithm") != "sha256"
        or binding.get("report_file") != target.name
        or binding.get("report_sha256") != report_file_hash
        or _stable_bytes(sidecar) != sidecar_bytes
    ):
        raise EvidenceValidationError(f"A3 report sidecar binding is invalid: {sidecar}")
    report = _json_object_bytes(report_bytes, path=target)
    stored_payload_hash = report.get("deterministic_payload_sha256")
    deterministic_payload = dict(report)
    provenance = deterministic_payload.pop("nondeterministic_provenance", None)
    deterministic_payload.pop("deterministic_payload_sha256", None)
    recomputed_payload_hash = _sha256_json(deterministic_payload)
    if (
        not isinstance(stored_payload_hash, str)
        or stored_payload_hash != recomputed_payload_hash
        or not isinstance(provenance, Mapping)
    ):
        raise EvidenceValidationError(f"A3 report self-hash is invalid: {target}")
    selection = report.get("selection")
    if not isinstance(selection, Mapping):
        raise EvidenceValidationError(f"A3 report selection is invalid: {target}")
    provisional_status = selection.get("provisional_terminal_status")
    if provisional_status not in {"NO_SELECTION", *A3_SELECTED_STATUSES}:
        raise EvidenceValidationError(f"A3 report terminal status is invalid: {target}")
    selected_condition = selection.get("selected_condition")
    if (
        provisional_status == "NO_SELECTION"
        and selected_condition is not None
    ) or (
        provisional_status in A3_SELECTED_STATUSES
        and selected_condition != str(provisional_status).removeprefix("SELECTED:")
    ):
        raise EvidenceValidationError(f"A3 report selection binding is invalid: {target}")

    expected_terminal_fields = {
        f"{prefix}_deterministic_payload_sha256": stored_payload_hash,
        f"{prefix}_recomputed_payload_sha256": recomputed_payload_hash,
        f"{prefix}_report_file_sha256": report_file_hash,
        f"{prefix}_sidecar_file_sha256": _sha256_bytes(sidecar_bytes),
        f"{prefix}_nondeterministic_provenance_sha256": _sha256_json(provenance),
        f"{prefix}_process_id": provenance.get("process_id"),
        f"{prefix}_process_instance_token": provenance.get("process_instance_token"),
        f"{prefix}_run_started_utc": provenance.get("run_started_utc"),
    }
    if any(terminal.get(field) != value for field, value in expected_terminal_fields.items()):
        raise EvidenceValidationError(f"A3 terminal/report binding differs: {target}")
    return {
        "path": target.relative_to(root).as_posix(),
        "bytes": len(report_bytes),
        "report_file_sha256": report_file_hash,
        "sidecar_path": sidecar.relative_to(root).as_posix(),
        "sidecar_bytes": len(sidecar_bytes),
        "sidecar_file_sha256": _sha256_bytes(sidecar_bytes),
        "deterministic_payload_sha256": stored_payload_hash,
        "selection_sha256": _sha256_json(selection),
        "provisional_terminal_status": provisional_status,
        "selected_condition": selected_condition,
        "process_id": provenance.get("process_id"),
        "process_instance_token": provenance.get("process_instance_token"),
    }


def _require_phase_artifact(
    *,
    phases: Mapping[str, Any],
    phase_name: str,
    path: str,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    phase = phases.get(phase_name)
    artifacts = phase.get("artifacts") if isinstance(phase, Mapping) else None
    if not isinstance(artifacts, list):
        raise EvidenceValidationError(f"A3 phase is malformed: {phase_name}")
    matches = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping) and artifact.get("path") == path
    ]
    if len(matches) != 1 or matches[0].get("bytes") != expected_bytes or matches[0].get(
        "sha256"
    ) != expected_sha256:
        raise EvidenceValidationError(
            f"A3 phase artifact binding differs: {phase_name}/{path}"
        )


def validate_a3_terminal(*, project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    """Validate A3 terminal self-hashes and every persisted report binding."""

    root = Path(project_root).resolve()
    path = _resolve(root, A3_TERMINAL)
    terminal_bytes = _stable_bytes(path)
    terminal = _json_object_bytes(terminal_bytes, path=path)
    required_true = (
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
    if (
        terminal.get("schema_version")
        != "experiment-000-readout-trace-a3-determinism-v1"
        or terminal.get("verification_payload_sha256")
        != _record_self_hash(terminal, "verification_payload_sha256")
    ):
        raise EvidenceValidationError("A3 terminal self-hash is invalid")
    status = terminal.get("status")
    if status not in {"NO_SELECTION", *A3_SELECTED_STATUSES}:
        raise EvidenceValidationError("A3 terminal status is not registered")
    if any(terminal.get(name) is not True for name in required_true):
        raise EvidenceValidationError("A3 terminal invariant is false or absent")

    first = _validated_a3_report_binding(
        root=root, report_path=A3_PRIMARY, terminal=terminal, prefix="first"
    )
    second = _validated_a3_report_binding(
        root=root, report_path=A3_RERUN, terminal=terminal, prefix="second"
    )
    if (
        first["deterministic_payload_sha256"]
        != second["deterministic_payload_sha256"]
        or first["selection_sha256"] != second["selection_sha256"]
        or first["provisional_terminal_status"]
        != second["provisional_terminal_status"]
        or first["provisional_terminal_status"] != status
        or first["process_id"] == second["process_id"]
        or first["process_instance_token"] == second["process_instance_token"]
    ):
        raise EvidenceValidationError("A3 deterministic rerun binding is invalid")

    freeze_path = _resolve(root, A3_FREEZE)
    freeze_bytes = _stable_bytes(freeze_path)
    freeze = _json_object_bytes(freeze_bytes, path=freeze_path)
    if (
        freeze.get("freeze_payload_sha256")
        != _record_self_hash(freeze, "freeze_payload_sha256")
        or terminal.get("freeze_record_sha256") != _sha256_bytes(freeze_bytes)
        or terminal.get("source_manifest_sha256")
        != freeze.get("source_manifest_sha256")
    ):
        raise EvidenceValidationError("A3 source-freeze binding is invalid")

    phase_path = _resolve(root, A3_PHASE_SEQUENCE)
    phase_bytes = _stable_bytes(phase_path)
    phase_record = _json_object_bytes(phase_bytes, path=phase_path)
    phase_rows = phase_record.get("phases")
    if (
        phase_record.get("schema_version")
        != "experiment-000-readout-trace-a3-phases-v1"
        or phase_record.get("phase_order") != list(A3_PHASE_ORDER)
        or phase_record.get("phase_sequence_payload_sha256")
        != _record_self_hash(phase_record, "phase_sequence_payload_sha256")
        or not isinstance(phase_rows, list)
        or [row.get("name") for row in phase_rows if isinstance(row, Mapping)]
        != list(A3_PHASE_ORDER)
    ):
        raise EvidenceValidationError("A3 phase-sequence binding is invalid")
    phases = {
        row["name"]: row for row in phase_rows if isinstance(row, Mapping)
    }
    _require_phase_artifact(
        phases=phases,
        phase_name="freeze",
        path=A3_FREEZE.as_posix(),
        expected_bytes=len(freeze_bytes),
        expected_sha256=_sha256_bytes(freeze_bytes),
    )
    for phase_name, report in (("full", first), ("rerun", second)):
        _require_phase_artifact(
            phases=phases,
            phase_name=phase_name,
            path=report["path"],
            expected_bytes=report["bytes"],
            expected_sha256=report["report_file_sha256"],
        )
        _require_phase_artifact(
            phases=phases,
            phase_name=phase_name,
            path=report["sidecar_path"],
            expected_bytes=report["sidecar_bytes"],
            expected_sha256=report["sidecar_file_sha256"],
        )
    _require_phase_artifact(
        phases=phases,
        phase_name="determinism_verification",
        path=A3_TERMINAL.as_posix(),
        expected_bytes=len(terminal_bytes),
        expected_sha256=_sha256_bytes(terminal_bytes),
    )
    if _stable_bytes(path) != terminal_bytes:
        raise EvidenceValidationError("A3 terminal changed during validation")
    return {
        "path": path.relative_to(root).as_posix(),
        "file_sha256": _sha256_bytes(terminal_bytes),
        "status": status,
        "deterministic_payload_sha256": first[
            "deterministic_payload_sha256"
        ],
        "verification_payload_sha256": terminal["verification_payload_sha256"],
        "freeze_record_sha256": _sha256_bytes(freeze_bytes),
        "phase_sequence_sha256": _sha256_bytes(phase_bytes),
    }


def canonical_status(*, project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    """Build the deterministic status object from canonical artifacts only."""

    root = Path(project_root).resolve()
    stage1_path = _resolve(root, CANONICAL_STAGE1 / "report.json")
    stage1 = _json_object(stage1_path)
    if (
        stage1.get("confirmatory") is not False
        or stage1.get("implemented_claim")
        != "deterministic fixed-DAG software plumbing only"
        or stage1.get("composition", {}).get("method")
        != "external_two_pass_primitive_adapter"
    ):
        raise EvidenceValidationError("canonical Stage-1 claim boundary is invalid")
    source_freeze = validate_lwoh_source_freeze_state(project_root=root)
    v3 = validate_lwoh_v3_state(project_root=root)
    blocked_reason = (
        "LWOH-L1 V3 passed; later CCF and continual-learning gates remain closed"
        if v3["status"] == "LWOH_LEARNING_PASS"
        else "evidence-integrity repair requires a fresh prospective source freeze"
    )
    return {
        "schema_version": "adaptive-learning-substrate-status-v1",
        "official_experiments_enabled": False,
        "official_experiments_blocked_reason": blocked_reason,
        "research_readiness_percent": v3["readiness_after"],
        "stage1": {
            "path": stage1_path.relative_to(root).as_posix(),
            "file_sha256": _sha256_bytes(_stable_bytes(stage1_path)),
            "confirmatory": False,
            "implemented_claim": stage1["implemented_claim"],
            "composition_method": stage1["composition"]["method"],
        },
        "a3": validate_a3_terminal(project_root=root),
        "lwoh_l1_v2_source_freeze": source_freeze,
        "lwoh_l1_v3": v3,
    }


def render_status_markdown(status: dict[str, Any]) -> str:
    """Render the canonical status object without clocks or machine-local paths."""

    return "\n".join(
        (
            "# Canonical project status",
            "",
            "> Generated from persisted terminal artifacts by ",
            "> `python -m adaptive_learning_substrate.artifact_validation write-status`.",
            "",
            "## Experiment gate",
            "",
            "- Official experiments enabled: **no**",
            f"- Reason: {status['official_experiments_blocked_reason']}.",
            f"- Evidence-based research readiness: **{status['research_readiness_percent']}/100**.",
            "",
            "## Stage-1 scaffold",
            "",
            f"- Claim: {status['stage1']['implemented_claim']}.",
            f"- Confirmatory: `{str(status['stage1']['confirmatory']).lower()}`.",
            f"- Composition method: `{status['stage1']['composition_method']}`.",
            f"- Report SHA-256: `{status['stage1']['file_sha256']}`.",
            "",
            "## Readout-trace A3",
            "",
            f"- Terminal result: **`{status['a3']['status']}`**.",
            f"- Deterministic payload SHA-256: `{status['a3']['deterministic_payload_sha256']}`.",
            f"- Verification payload SHA-256: `{status['a3']['verification_payload_sha256']}`.",
            f"- Terminal file SHA-256: `{status['a3']['file_sha256']}`.",
            "",
            "## LWOH-L1 V2 prospective freeze",
            "",
            f"- Current source state: **`{status['lwoh_l1_v2_source_freeze']['status']}`**.",
            f"- Changed frozen files: `{status['lwoh_l1_v2_source_freeze']['changed_file_count']}`.",
            *(
                f"- Drifted path: `{path}`."
                for path in status["lwoh_l1_v2_source_freeze"]["changed_files"]
            ),
            "",
            "## LWOH-L1 V3 readiness chain",
            "",
            f"- Evidence state: **`{status['lwoh_l1_v3']['state']}`**.",
            f"- Terminal status: `{status['lwoh_l1_v3']['status']}`.",
            f"- Readiness after validated chain: **{status['lwoh_l1_v3']['readiness_after']}/100**.",
            "",
            "## Advancement rule",
            "",
            "The A3 result is a reproducible negative development result. No memory",
            "condition advances, confirmatory seeds remain closed, and the next official",
            "protocol must receive a new prospective freeze after integrity repairs.",
            "",
        )
    )


def write_status(*, project_root: str | Path = PROJECT_ROOT) -> Path:
    root = Path(project_root).resolve()
    target = _resolve(root, CANONICAL_STATUS)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = render_status_markdown(canonical_status(project_root=root)).encode("utf-8")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    if _stable_bytes(target) != encoded:
        raise EvidenceValidationError("canonical status changed during publication")
    return target


def sync_latest(*, project_root: str | Path = PROJECT_ROOT) -> list[Path]:
    root = Path(project_root).resolve()
    latest_root = _resolve(root, LATEST_RUN)
    latest_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in LATEST_FILES:
        source = _resolve(root, CANONICAL_STAGE1 / name)
        target = latest_root / name
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        written.append(target)
    validate_latest_run(project_root=root)
    return written


def validate_project(*, project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).resolve()
    status_snapshot = canonical_status(project_root=root)
    expected_status = render_status_markdown(status_snapshot).encode("utf-8")
    status_path = _resolve(root, CANONICAL_STATUS)
    if _stable_bytes(status_path) != expected_status:
        raise EvidenceValidationError("canonical status page is stale")
    frozen_source_snapshots = validate_frozen_source_snapshots(project_root=root)
    sidecars = validate_sidecars(project_root=root)
    latest_run = validate_latest_run(project_root=root)
    if canonical_status(project_root=root) != status_snapshot:
        raise EvidenceValidationError("canonical evidence changed during validation")
    return {
        "status": "PASS",
        "frozen_source_snapshots": frozen_source_snapshots,
        "sidecars": sidecars,
        "latest_run": latest_run,
        "canonical_status": {
            "path": status_path.relative_to(root).as_posix(),
            "sha256": _sha256_bytes(expected_status),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("verify", "write-status", "sync-latest")
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "write-status":
        result: Any = {"written": str(write_status(project_root=args.project_root))}
    elif args.command == "sync-latest":
        result = {
            "written": [str(path) for path in sync_latest(project_root=args.project_root)]
        }
    else:
        result = validate_project(project_root=args.project_root)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
