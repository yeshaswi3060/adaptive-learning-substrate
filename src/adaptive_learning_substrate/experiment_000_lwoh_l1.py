"""Blinded A3-compatible LWOH-L1 V2 contingency controls.

This module deliberately implements only the pre-terminal amendment freeze and
the read-only A3 terminal trigger.  It does not generate LWOH task data.  The
scientific design remains the byte-bound V1 design; V2 is a narrow trigger,
seed-exclusion, namespace, and source-binding overlay.

The freeze path never parses an A3 task report.  Reports and sidecars are
observed only as opaque byte strings until the A3 terminal artifact exists.
The activation path does not invoke the A3 terminal verifier.  Instead it
re-opens the already committed evidence and calls the frozen A3 validation
primitives with ``require_selected=False`` so that a fully replay-validated
``NO_SELECTION`` can activate this contingency.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import platform
import re
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "experiment-000-lwoh-l1-a3-preterminal-freeze-v2"
PROTOCOL_VERSION = "experiment-000-lwoh-l1-v2"

BASE_PROTOCOL_PATH = Path("docs/EXPERIMENT_000_LWOH_L1_PROTOCOL.md")
BASE_CONFIG_PATH = Path("configs/experiment_000_lwoh_l1.toml")
AMENDMENT_PROTOCOL_PATH = Path("docs/EXPERIMENT_000_LWOH_L1_A3_AMENDMENT_V2.md")
AMENDMENT_CONFIG_PATH = Path("configs/experiment_000_lwoh_l1_v2.toml")
RUNNER_PATH = Path("src/adaptive_learning_substrate/experiment_000_lwoh_l1.py")
TEST_PATH = Path("tests/test_experiment_000_lwoh_l1.py")

ARTIFACT_DIRECTORY = Path("artifacts/experiment_000/lwoh_l1_v2")
CONTINGENCY_FREEZE_PATH = ARTIFACT_DIRECTORY / "PRE_TERMINAL_CONTINGENCY_FREEZE.json"
CONTINGENCY_FREEZE_SIDECAR_PATH = CONTINGENCY_FREEZE_PATH.with_suffix(".sha256")
PHASE_LEDGER_PATH = ARTIFACT_DIRECTORY / "PHASE_LEDGER.json"
ACTIVATION_DIRECTORY = ARTIFACT_DIRECTORY / "activation"
ACTIVATION_PROVENANCE_PATH = (
    ACTIVATION_DIRECTORY / "A3_NO_SELECTION_ACTIVATION.json"
)
ACTIVATION_PROVENANCE_SIDECAR_PATH = ACTIVATION_PROVENANCE_PATH.with_suffix(
    ".sha256"
)
FUTURE_EXECUTION_RUNNER_PATH = Path(
    "src/adaptive_learning_substrate/experiment_000_lwoh_l1_execution_v2.py"
)
FUTURE_EXECUTION_TEST_PATH = Path("tests/test_experiment_000_lwoh_l1_execution_v2.py")
IMPLEMENTATION_SOURCE_FREEZE_PATH = (
    ARTIFACT_DIRECTORY / "IMPLEMENTATION_SOURCE_FREEZE.json"
)
IMPLEMENTATION_SOURCE_FREEZE_SIDECAR_PATH = (
    IMPLEMENTATION_SOURCE_FREEZE_PATH.with_suffix(".sha256")
)

A3_ARTIFACT_DIRECTORY = Path("artifacts/experiment_000/readout_trace_a3")
A3_PRE_FREEZE_VERIFICATION_PATH = (
    A3_ARTIFACT_DIRECTORY / "PRE_FREEZE_VERIFICATION.json"
)
A3_FREEZE_RECORD_PATH = A3_ARTIFACT_DIRECTORY / "FREEZE_RECORD.json"
A3_PHASE_SEQUENCE_PATH = A3_ARTIFACT_DIRECTORY / "PHASE_SEQUENCE.json"
A3_SMOKE_REPORT_PATH = A3_ARTIFACT_DIRECTORY / "smoke_pairs_2.json"
A3_SMOKE_SIDECAR_PATH = A3_ARTIFACT_DIRECTORY / "smoke_pairs_2.sha256"
A3_SMOKE_VERIFICATION_PATH = A3_ARTIFACT_DIRECTORY / "SMOKE_VERIFICATION.json"
A3_PRIMARY_REPORT_PATH = (
    A3_ARTIFACT_DIRECTORY / "custom_seeds_105_109_pairs_100.json"
)
A3_PRIMARY_SIDECAR_PATH = (
    A3_ARTIFACT_DIRECTORY / "custom_seeds_105_109_pairs_100.sha256"
)
A3_RERUN_REPORT_PATH = (
    A3_ARTIFACT_DIRECTORY / "custom_seeds_105_109_pairs_100_rerun.json"
)
A3_RERUN_SIDECAR_PATH = (
    A3_ARTIFACT_DIRECTORY / "custom_seeds_105_109_pairs_100_rerun.sha256"
)
A3_TERMINAL_PATH = A3_ARTIFACT_DIRECTORY / "DETERMINISM_VERIFICATION.json"
A3_DETERMINISM_VERIFICATION_PATH = A3_TERMINAL_PATH

V1_PROTOCOL_SHA256 = "42e237add4eac9ca24a6d5781efefce6367f8c9f2c4c4065b324a1703706dc7e"
V1_CONFIG_SHA256 = "7ed14e7dc85f7178e6dce0fbdf1e1d05d93c41129eba8b3c3f491dbfeffc86b8"
AMENDMENT_PROTOCOL_SHA256 = (
    "d46929ed8ce01e361580df1f119976797aabd915062815be70e7a4bde77b9c2b"
)
AMENDMENT_CONFIG_SHA256 = (
    "c5f18680ff37f4a03bfb88ee5d8320783c85b07ab88d389b68543434d98e95fb"
)
V1_CANONICAL_OBJECT_SHA256 = (
    "a1704b1197c3858372b60e067237989460c0cdb43d3e59edc64fac594ab3fcbb"
)
V1_SCIENTIFIC_PAYLOAD_SHA256 = (
    "9910bdeab37ff5b633f62dcbb1cd8517a7c5a456107e4b6857fcd7eb9f31f66b"
)
AMENDMENT_CONFIG_OBJECT_SHA256 = (
    "68855fdee910c28128e5fe4c765e71f40740a2093c66642e59ff079599e804dd"
)
A3_FREEZE_RECORD_SHA256 = (
    "50c8a2af24394a15ec8f93abb2f7c851d60ca385fd65764e2d22bda2757bc164"
)
A3_SOURCE_MANIFEST_SHA256 = (
    "492e3165eed4cb3e5f425f74b615f5a71ca3d281a1107bb6fb8c701d204ae4db"
)
A3_PROTOCOL_SHA256 = (
    "cb023552d50d568c3fe1e44c40038b064c275b513785c26d8693efa21c666963"
)
A3_CONFIG_SHA256 = (
    "f1ca8047c7f5d559a4b6ac3888278d867826227917436af6f5873b6f4cfe0282"
)
A3_RUNNER_SHA256 = (
    "99403c6706c5e7cdb56f518a696ba9e9cf5101f048e810cee99b07eb828cfdc5"
)

LWOH_ADMISSION_SEEDS = (90, 91, 92, 93, 94)
LWOH_LEARNING_SEEDS = (95, 96, 97, 98, 99, 100, 101, 102, 103, 104)
LWOH_SCRATCH_SEEDS = (9090, 9091)
A3_REGISTERED_SEEDS = (105, 106, 107, 108, 109)
REGISTERED_A3_SELECTED_CONDITIONS = frozenset(
    {"rho_0_25", "rho_0_50", "rho_0_75", "rho_0_90", "rho_0_95"}
)
ACTIVATION_BINDING_KEYS = frozenset(
    {
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
)

FORBIDDEN_SEED_SETS: dict[str, tuple[int, ...]] = {
    "development": (0, 1, 2, 3, 4),
    "alignment": (42, 43, 44, 45, 46),
    "memory_sweep": (50, 51, 52, 53, 54),
    "slow_state": (60, 61, 62, 63, 64),
    "invalid_a1": (65, 66, 67, 68, 69),
    "a2": (70, 71, 72, 73, 74),
    "a2_selected_branch": tuple(range(75, 90)),
    "a3": A3_REGISTERED_SEEDS,
    "confirmatory": tuple(range(1000, 1020)),
}

# The overlay may describe only the trigger, bindings, inherited seeds,
# namespace, and implementation/freeze boundary.  Scientific sections are
# forbidden even if a value happens to equal V1 today: accepting them would
# create an unnecessary future override surface.
OVERLAY_ALLOWED_KEYS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "status",
        "claim_level",
        "amendment_path",
        "amendment_sha256",
        "base_v1",
        "overlay_policy",
        "trigger",
        "blinding",
        "a3_prerequisites",
        "seed_inheritance",
        "artifact_namespace",
        "future_source_freeze",
        "post_activation_implementation",
        "implementation_boundary",
        "inherited_decision_assertion",
    }
)
OVERLAY_ALLOWED_PATHS = frozenset(
    {
        "protocol_version",
        "protocol_path",
        "amendment_path",
        "amendment_sha256",
        "effective_protocol_binding",
        "status",
        "trigger",
        "seed_partition.forbidden_a3",
        "persistence",
        "source_freeze",
        "post_activation_implementation",
        "commands",
        "_v2_overlay",
        "_semantic_delta",
    }
)
SCIENTIFIC_V1_SECTIONS = (
    "seed_partition",
    "graph",
    "memory",
    "admission",
    "learner",
    "head_alignment_tests",
    "conditions",
    "data",
    "scratch",
    "learning_gates",
    "control_integrity",
    "compute",
    "integrity",
    "decision",
)

A3_SOURCE_PATHS = (
    "configs/experiment_000_readout_trace_a3.toml",
    "docs/CCF_V0.md",
    "docs/EXPERIMENT_000_READOUT_TRACE_A3_PROTOCOL.md",
    "pyproject.toml",
    "src/adaptive_learning_substrate/__init__.py",
    "src/adaptive_learning_substrate/events.py",
    "src/adaptive_learning_substrate/experiment000_data.py",
    "src/adaptive_learning_substrate/experiment_000.py",
    "src/adaptive_learning_substrate/experiment_000_memory_probe.py",
    "src/adaptive_learning_substrate/experiment_000_readout_trace_a3.py",
    "src/adaptive_learning_substrate/experiment_000_slow_state.py",
    "src/adaptive_learning_substrate/readout_trace_recurrent.py",
    "src/adaptive_learning_substrate/recurrent.py",
    "tests/test_experiment_000_readout_trace_a3.py",
)

V2_SOURCE_PATHS = (
    str(BASE_PROTOCOL_PATH).replace("\\", "/"),
    str(BASE_CONFIG_PATH).replace("\\", "/"),
    str(AMENDMENT_PROTOCOL_PATH).replace("\\", "/"),
    str(AMENDMENT_CONFIG_PATH).replace("\\", "/"),
    str(RUNNER_PATH).replace("\\", "/"),
    str(TEST_PATH).replace("\\", "/"),
    "src/adaptive_learning_substrate/recurrent.py",
    "src/adaptive_learning_substrate/experiment000_data.py",
    "pyproject.toml",
)
PRETERMINAL_SOURCE_PATHS = V2_SOURCE_PATHS

A3_OPAQUE_PREREQUISITE_PATHS = (
    A3_PRE_FREEZE_VERIFICATION_PATH,
    A3_FREEZE_RECORD_PATH,
    A3_PHASE_SEQUENCE_PATH,
    A3_SMOKE_REPORT_PATH,
    A3_SMOKE_SIDECAR_PATH,
    A3_SMOKE_VERIFICATION_PATH,
    A3_PRIMARY_REPORT_PATH,
    A3_PRIMARY_SIDECAR_PATH,
    A3_RERUN_REPORT_PATH,
    A3_RERUN_SIDECAR_PATH,
    A3_TERMINAL_PATH,
)

_PROCESS_INSTANCE_TOKEN = hashlib.sha256(os.urandom(32)).hexdigest()
_VALIDATION_COMMANDS = frozenset(
    {
        "internal:validate_v1_v2_hashes_and_allowlisted_semantic_delta",
        "internal:validate_a3_frozen_source_and_preterminal_phase_prefix",
        "python -m py_compile src/adaptive_learning_substrate/experiment_000_lwoh_l1.py tests/test_experiment_000_lwoh_l1.py",
        "python -m ruff check --no-cache src/adaptive_learning_substrate/experiment_000_lwoh_l1.py tests/test_experiment_000_lwoh_l1.py",
        "python -m pytest -q -o pythonpath=src tests/test_experiment_000_lwoh_l1.py",
    }
)
_EXTERNAL_VALIDATION_ARGV: dict[str, tuple[str, ...]] = {
    "python -m py_compile src/adaptive_learning_substrate/experiment_000_lwoh_l1.py tests/test_experiment_000_lwoh_l1.py": (
        sys.executable,
        "-m",
        "py_compile",
        "src/adaptive_learning_substrate/experiment_000_lwoh_l1.py",
        "tests/test_experiment_000_lwoh_l1.py",
    ),
    "python -m ruff check --no-cache src/adaptive_learning_substrate/experiment_000_lwoh_l1.py tests/test_experiment_000_lwoh_l1.py": (
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--no-cache",
        "src/adaptive_learning_substrate/experiment_000_lwoh_l1.py",
        "tests/test_experiment_000_lwoh_l1.py",
    ),
    "python -m pytest -q -o pythonpath=src tests/test_experiment_000_lwoh_l1.py": (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-o",
        "pythonpath=src",
        "tests/test_experiment_000_lwoh_l1.py",
    ),
}
_FORBIDDEN_EVIDENCE_TERMS = (
    "status",
    "selection",
    "selected",
    "no_selection",
    "selected_condition",
    "passing_conditions",
    "condition_rows",
    "seed_results",
    "aggregate",
    "aggregates",
    "aggregate_conditions",
    "gate_values",
    "effect_sizes",
    "near_miss",
    "ridge_eval_accuracy",
    "cue_delta",
    "retention_gate",
    "provisional_terminal_status",
)

_A3_PHASE_ORDER = (
    "tests",
    "freeze",
    "persisted_smoke_and_sidecar",
    "machine_smoke_verification",
    "full",
    "rerun",
    "determinism_verification",
)
_A3_PHASE_ARTIFACTS: dict[str, tuple[Path, ...]] = {
    "tests": (A3_PRE_FREEZE_VERIFICATION_PATH,),
    "freeze": (A3_FREEZE_RECORD_PATH,),
    "persisted_smoke_and_sidecar": (
        A3_SMOKE_REPORT_PATH,
        A3_SMOKE_SIDECAR_PATH,
    ),
    "machine_smoke_verification": (A3_SMOKE_VERIFICATION_PATH,),
    "full": (A3_PRIMARY_REPORT_PATH, A3_PRIMARY_SIDECAR_PATH),
    "rerun": (A3_RERUN_REPORT_PATH, A3_RERUN_SIDECAR_PATH),
    "determinism_verification": (A3_TERMINAL_PATH,),
}


def _project_root(project_root: str | Path | None = None) -> Path:
    if project_root is None:
        return Path(__file__).resolve().parents[2]
    return Path(project_root).resolve()


def _require_a3_module_root(root: Path, a3: Any) -> None:
    module_file = getattr(a3, "__file__", None)
    if not isinstance(module_file, str):
        raise TypeError("frozen A3 validator module has no source identity")
    module_root = Path(module_file).resolve().parents[2]
    if module_root != root.resolve():
        raise RuntimeError("A3 validator module and V2 project roots differ")


def _require_future_execution_absent(root: Path, *, context: str) -> None:
    present = [
        path.as_posix()
        for path in (FUTURE_EXECUTION_RUNNER_PATH, FUTURE_EXECUTION_TEST_PATH)
        if _resolve(root, path).exists()
    ]
    if present:
        raise RuntimeError(f"post-activation execution sources exist {context}: {present}")


def _require_activation_provenance_absent(root: Path, *, context: str) -> None:
    if _resolve(root, ACTIVATION_DIRECTORY).exists():
        raise RuntimeError(f"A3 activation-provenance namespace exists {context}")


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside the project root: {path}") from exc


def _stable_file_measurement(path: str | Path) -> tuple[int, int, str]:
    """Hash one stable open handle and reject concurrent replacement/write."""

    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(stream.fileno())
    signature_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    signature_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if signature_before != signature_after:
        raise RuntimeError(f"file changed while hashing: {target}")
    path_after = target.stat()
    if (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
    ) != signature_after:
        raise RuntimeError(f"file was replaced while hashing: {target}")
    return after.st_size, after.st_mtime_ns, digest.hexdigest()


def _stable_file_bytes(path: str | Path) -> bytes:
    target = Path(path)
    with target.open("rb") as stream:
        before = os.fstat(stream.fileno())
        payload = stream.read()
        after = os.fstat(stream.fileno())
    signature_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    signature_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if signature_before != signature_after or len(payload) != after.st_size:
        raise RuntimeError(f"file changed while reading: {target}")
    path_after = target.stat()
    if (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
    ) != signature_after:
        raise RuntimeError(f"file was replaced while reading: {target}")
    return payload


def _file_sha256(path: str | Path) -> str:
    return _stable_file_measurement(path)[2]


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _record_self_hash(record: Mapping[str, Any], field: str) -> str:
    payload = dict(record)
    payload.pop(field, None)
    return _sha256_json(payload)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("ascii")


def _read_json(path: str | Path, *, purpose: str = "metadata") -> dict[str, Any]:
    """Read one JSON metadata object.

    Freeze creation never calls this helper for either full A3 report.  Keeping
    this boundary explicit makes the pre-terminal blinding property directly
    testable.
    """

    value = json.loads(_stable_file_bytes(path).decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{purpose} root must be a JSON object")
    return value


def _read_toml(path: str | Path) -> dict[str, Any]:
    value = tomllib.loads(_stable_file_bytes(path).decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("configuration root must be a TOML table")
    return value


def _identity(root: Path, path: str | Path) -> dict[str, Any]:
    target = _resolve(root, path)
    if not target.is_file():
        raise FileNotFoundError(f"required file is absent: {_relative(root, target)}")
    size, _, digest = _stable_file_measurement(target)
    return {
        "path": _relative(root, target),
        "bytes": size,
        "sha256": digest,
    }


def _opaque_identity(root: Path, path: str | Path) -> dict[str, Any]:
    """Observe existence and whole-file identity without parsing file bytes."""

    target = _resolve(root, path)
    if not target.is_file():
        return {"path": _relative(root, target), "present": False}
    size, mtime_ns, digest = _stable_file_measurement(target)
    return {
        "path": _relative(root, target),
        "present": True,
        "bytes": size,
        "mtime_ns": mtime_ns,
        "sha256": digest,
    }


def _require_hash(root: Path, path: Path, expected: str, label: str) -> str:
    target = _resolve(root, path)
    if not target.is_file():
        raise FileNotFoundError(f"{label} is absent: {path.as_posix()}")
    actual = _file_sha256(target)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 differs from its immutable binding")
    return actual


def _validate_overlay(
    overlay: Mapping[str, Any], base: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove that the exact V2 overlay changes no V1 scientific value."""

    present = frozenset(overlay)
    unexpected = sorted(present - OVERLAY_ALLOWED_KEYS)
    if unexpected:
        raise RuntimeError(f"V2 overlay contains unrecognized top-level keys: {unexpected}")
    missing = sorted(
        {
            "schema_version",
            "protocol_version",
            "status",
            "claim_level",
            "amendment_path",
            "amendment_sha256",
            "base_v1",
            "overlay_policy",
            "trigger",
            "blinding",
            "a3_prerequisites",
            "seed_inheritance",
            "artifact_namespace",
            "future_source_freeze",
            "post_activation_implementation",
            "implementation_boundary",
            "inherited_decision_assertion",
        }
        - present
    )
    if missing:
        raise RuntimeError(f"V2 overlay is missing required top-level keys: {missing}")
    if overlay.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("V2 overlay protocol version is invalid")
    if overlay.get("schema_version") != "experiment-000-lwoh-l1-a3-amendment-overlay-v2":
        raise RuntimeError("V2 overlay schema version is invalid")
    if (
        overlay.get("status")
        != "prospective_blinded_pre_terminal_amendment_unfrozen"
        or overlay.get("claim_level") != "custom_seed_development_only"
        or overlay.get("amendment_path") != AMENDMENT_PROTOCOL_PATH.as_posix()
        or overlay.get("amendment_sha256") != AMENDMENT_PROTOCOL_SHA256
    ):
        raise RuntimeError("V2 overlay identity is invalid")
    for forbidden in SCIENTIFIC_V1_SECTIONS:
        if forbidden in overlay:
            raise RuntimeError(f"V2 overlay attempts a scientific override: {forbidden}")

    base_binding = overlay.get("base_v1")
    if not isinstance(base_binding, Mapping):
        raise TypeError("V2 overlay base binding is absent")
    if (
        base_binding.get("protocol_version") != "experiment-000-lwoh-l1-v1"
        or base_binding.get("protocol_path") != BASE_PROTOCOL_PATH.as_posix()
        or base_binding.get("protocol_sha256") != V1_PROTOCOL_SHA256
        or base_binding.get("config_path") != BASE_CONFIG_PATH.as_posix()
        or base_binding.get("config_sha256") != V1_CONFIG_SHA256
        or base_binding.get("canonical_toml_object_sha256")
        != V1_CANONICAL_OBJECT_SHA256
        or base_binding.get("scientific_payload_sha256")
        != V1_SCIENTIFIC_PAYLOAD_SHA256
        or _sha256_json(base) != V1_CANONICAL_OBJECT_SHA256
    ):
        raise RuntimeError("V2 overlay immutable V1 binding is invalid")
    section_hashes = base_binding.get("scientific_section_sha256")
    if not isinstance(section_hashes, Mapping) or set(section_hashes) != set(
        SCIENTIFIC_V1_SECTIONS
    ):
        raise RuntimeError("V2 overlay scientific-section hash set is invalid")
    for section in SCIENTIFIC_V1_SECTIONS:
        if section not in base or section_hashes.get(section) != _sha256_json(
            base[section]
        ):
            raise RuntimeError(f"V1 scientific section binding differs: {section}")
    scientific_payload = {section: base[section] for section in SCIENTIFIC_V1_SECTIONS}
    if _sha256_json(scientific_payload) != V1_SCIENTIFIC_PAYLOAD_SHA256:
        raise RuntimeError("V1 combined scientific payload hash differs")

    policy = overlay.get("overlay_policy")
    if not isinstance(policy, Mapping):
        raise TypeError("V2 overlay policy is absent")
    if set(policy.get("allowed_top_level_keys", ())) != set(OVERLAY_ALLOWED_KEYS):
        raise RuntimeError("V2 overlay allowlist differs from its registered set")
    if tuple(policy.get("inherited_scientific_sections", ())) != SCIENTIFIC_V1_SECTIONS:
        raise RuntimeError("V2 inherited scientific-section list differs")
    if set(policy.get("forbidden_override_sections", ())) != set(
        SCIENTIFIC_V1_SECTIONS[1:]
    ):
        raise RuntimeError("V2 forbidden scientific override set differs")
    if any(
        policy.get(key) is not False
        for key in (
            "scientific_value_change_allowed",
            "threshold_change_allowed",
            "sample_size_change_allowed",
            "compute_change_allowed",
            "assigned_seed_change_allowed",
        )
    ):
        raise RuntimeError("V2 overlay permits a scientific change")

    trigger = overlay.get("trigger")
    if not isinstance(trigger, Mapping) or (
        trigger.get("predecessor_protocol_version") != "readout-trace-v1a3"
        or trigger.get("activate_only_on_a3_terminal") != "NO_SELECTION"
        or trigger.get("activated_status") != "ACTIVATED_A3_NO_SELECTION"
        or trigger.get("cancel_status") != "CANCELLED_A3_SELECTED"
        or trigger.get("invalid_or_incomplete_a3_status")
        != "INVALID_A3_DO_NOT_ACTIVATE_LWOH"
        or trigger.get("read_only_validation_mode")
        != "a3_full_replay_require_selected_false"
        or trigger.get("selected_only_a3_loader_is_activation_path") is not False
        or trigger.get("rerun_a3_terminal_verifier_from_loader") is not False
        or trigger.get("loader_may_return_a3_scientific_fields") is not False
    ):
        raise RuntimeError("V2 A3 trigger policy differs")

    prerequisites = overlay.get("a3_prerequisites")
    if not isinstance(prerequisites, Mapping) or (
        prerequisites.get("freeze_record_path") != A3_FREEZE_RECORD_PATH.as_posix()
        or prerequisites.get("freeze_record_sha256") != A3_FREEZE_RECORD_SHA256
        or prerequisites.get("source_manifest_sha256")
        != A3_SOURCE_MANIFEST_SHA256
        or prerequisites.get("protocol_sha256") != A3_PROTOCOL_SHA256
        or prerequisites.get("config_sha256") != A3_CONFIG_SHA256
        or prerequisites.get("runner_sha256") != A3_RUNNER_SHA256
        or tuple(prerequisites.get("frozen_source_paths", ())) != A3_SOURCE_PATHS
    ):
        raise RuntimeError("V2 A3 prerequisite binding differs")

    seed_inheritance = overlay.get("seed_inheritance")
    if not isinstance(seed_inheritance, Mapping) or (
        tuple(seed_inheritance.get("admission", ())) != LWOH_ADMISSION_SEEDS
        or tuple(seed_inheritance.get("learning", ())) != LWOH_LEARNING_SEEDS
        or tuple(seed_inheritance.get("scratch", ())) != LWOH_SCRATCH_SEEDS
        or tuple(seed_inheritance.get("additional_forbidden_a3", ()))
        != A3_REGISTERED_SEEDS
    ):
        raise RuntimeError("V2 inherited seed binding differs")

    namespace = overlay.get("artifact_namespace")
    if not isinstance(namespace, Mapping) or (
        namespace.get("canonical_directory") != ARTIFACT_DIRECTORY.as_posix()
        or namespace.get("pre_terminal_freeze_record")
        != CONTINGENCY_FREEZE_PATH.as_posix()
        or namespace.get("pre_terminal_freeze_sidecar")
        != CONTINGENCY_FREEZE_SIDECAR_PATH.as_posix()
        or namespace.get("phase_ledger") != PHASE_LEDGER_PATH.as_posix()
        or namespace.get("activation_provenance")
        != ACTIVATION_PROVENANCE_PATH.as_posix()
        or namespace.get("activation_provenance_sidecar")
        != ACTIVATION_PROVENANCE_SIDECAR_PATH.as_posix()
        or namespace.get("reuse_v1_artifact_paths") is not False
    ):
        raise RuntimeError("V2 artifact namespace differs")

    future = overlay.get("future_source_freeze")
    if not isinstance(future, Mapping) or (
        tuple(future.get("required_paths", ())) != V2_SOURCE_PATHS
        or future.get("required_before_any_lwoh_metric") is not True
        or future.get("bind_complete_a3_source_manifest") is not True
        or future.get("scope") != "immutable_preterminal_controller_bundle"
        or future.get("post_activation_execution_paths_excluded") is not True
        or future.get("future_path_creation_does_not_invalidate_preterminal_freeze")
        is not True
        or future.get("canonical_json")
        != "sorted_keys_compact_separators_ascii_allow_nan_false"
        or future.get("hash_algorithm") != "sha256"
        or future.get("require_record_self_hash") is not True
        or future.get("require_binding_sidecar") is not True
        or future.get("require_fresh_process_validation") is not True
        or future.get("source_drift_terminal_prefix")
        != "INVALID_PROCEDURE_SOURCE_DRIFT:"
    ):
        raise RuntimeError("V2 future source-freeze path set differs")

    post_activation = overlay.get("post_activation_implementation")
    expected_post_activation = {
        "activation_status_required": "ACTIVATED_A3_NO_SELECTION",
        "activation_provenance_path": ACTIVATION_PROVENANCE_PATH.as_posix(),
        "activation_provenance_sidecar_path": (
            ACTIVATION_PROVENANCE_SIDECAR_PATH.as_posix()
        ),
        "pre_terminal_controller_path": RUNNER_PATH.as_posix(),
        "pre_terminal_controller_test_path": TEST_PATH.as_posix(),
        "execution_runner_path": FUTURE_EXECUTION_RUNNER_PATH.as_posix(),
        "execution_test_path": FUTURE_EXECUTION_TEST_PATH.as_posix(),
        "implementation_freeze_path": IMPLEMENTATION_SOURCE_FREEZE_PATH.as_posix(),
        "implementation_freeze_sidecar_path": (
            IMPLEMENTATION_SOURCE_FREEZE_SIDECAR_PATH.as_posix()
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
    if not isinstance(post_activation, Mapping) or dict(post_activation) != (
        expected_post_activation
    ):
        raise RuntimeError("V2 post-activation implementation boundary differs")

    boundary = overlay.get("implementation_boundary")
    expected_boundary = {
        "pre_terminal_controller_present_in_this_amendment": True,
        "pre_terminal_controller_tests_present_in_this_amendment": True,
        "pre_terminal_freeze_artifact_present_in_this_amendment": False,
        "post_activation_execution_runner_present_in_this_amendment": False,
        "post_activation_execution_tests_present_in_this_amendment": False,
        "official_lwoh_execution_authorized": False,
        "runner_and_tests_must_be_frozen_before_any_lwoh_metric": True,
        "future_implementation_may_change_scientific_design": False,
        "future_tests_use_synthetic_or_scratch_data_before_activation": True,
        "pyproject_edit_required": False,
        "package_init_edit_required": False,
    }
    if not isinstance(boundary, Mapping) or dict(boundary) != expected_boundary:
        raise RuntimeError("V2 implementation boundary differs")

    decision = overlay.get("inherited_decision_assertion")
    if not isinstance(decision, Mapping) or decision.get("must_equal_base_v1") is not True:
        raise RuntimeError("V2 decision-inheritance assertion is absent")
    for key, value in decision.items():
        if key == "must_equal_base_v1":
            continue
        if base.get("decision", {}).get(key) != value:
            raise RuntimeError(f"V2 inherited decision differs: {key}")
    if _sha256_json(overlay) != AMENDMENT_CONFIG_OBJECT_SHA256:
        raise RuntimeError("V2 overlay canonical object differs from its binding")
    return {
        "allowed_top_level_keys": sorted(OVERLAY_ALLOWED_KEYS),
        "present_top_level_keys": sorted(present),
        "unexpected_top_level_keys": [],
        "scientific_override_keys": [],
        "scientific_design_inherited_unchanged": True,
        "v1_canonical_object_sha256": V1_CANONICAL_OBJECT_SHA256,
        "v1_scientific_payload_sha256": V1_SCIENTIFIC_PAYLOAD_SHA256,
    }


def _validate_overlay_shape(overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper used only when the immutable base is available."""

    root = _project_root()
    base = _read_toml(_resolve(root, BASE_CONFIG_PATH))
    return _validate_overlay(overlay, base)


def _nested_get(mapping: Mapping[str, Any], path: str) -> Any:
    value: Any = mapping
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise KeyError(path)
        value = value[component]
    return value


def _validate_seed_partitions(effective_config: Mapping[str, Any]) -> dict[str, Any]:
    partition = effective_config.get("seed_partition")
    if not isinstance(partition, Mapping):
        raise TypeError("effective configuration has no seed partition")
    expected = {
        "admission": LWOH_ADMISSION_SEEDS,
        "learning": LWOH_LEARNING_SEEDS,
        "scratch": LWOH_SCRATCH_SEEDS,
        "forbidden_a3": A3_REGISTERED_SEEDS,
    }
    for key, values in expected.items():
        observed = partition.get(key)
        if not isinstance(observed, list) or tuple(observed) != values:
            raise RuntimeError(f"effective V2 seed partition differs at {key}")
        if any(type(seed) is not int for seed in observed):
            raise TypeError(f"seed partition {key} must contain exact integers")
    official = set(LWOH_ADMISSION_SEEDS) | set(LWOH_LEARNING_SEEDS)
    scratch = set(LWOH_SCRATCH_SEEDS)
    forbidden = set().union(*FORBIDDEN_SEED_SETS.values())
    if official & scratch or official & forbidden or scratch & forbidden:
        raise RuntimeError("V2 seed partitions overlap")
    return {
        "admission": list(LWOH_ADMISSION_SEEDS),
        "learning": list(LWOH_LEARNING_SEEDS),
        "scratch": list(LWOH_SCRATCH_SEEDS),
        "a3_forbidden": list(A3_REGISTERED_SEEDS),
        "all_partitions_disjoint": True,
    }


def load_effective_config(
    *, project_root: str | Path | None = None
) -> dict[str, Any]:
    """Load exact-hash V1 and apply the narrow non-scientific V2 overlay."""

    root = _project_root(project_root)
    _require_hash(root, BASE_PROTOCOL_PATH, V1_PROTOCOL_SHA256, "V1 protocol")
    _require_hash(root, BASE_CONFIG_PATH, V1_CONFIG_SHA256, "V1 configuration")
    _require_hash(
        root,
        AMENDMENT_PROTOCOL_PATH,
        AMENDMENT_PROTOCOL_SHA256,
        "V2 amendment protocol",
    )
    _require_hash(
        root,
        AMENDMENT_CONFIG_PATH,
        AMENDMENT_CONFIG_SHA256,
        "V2 amendment configuration",
    )
    base = _read_toml(_resolve(root, BASE_CONFIG_PATH))
    overlay = _read_toml(_resolve(root, AMENDMENT_CONFIG_PATH))
    semantic_delta = _validate_overlay(overlay, base)

    effective = copy.deepcopy(base)
    effective["protocol_version"] = PROTOCOL_VERSION
    effective["protocol_path"] = BASE_PROTOCOL_PATH.as_posix()
    effective["protocol_sha256"] = V1_PROTOCOL_SHA256
    effective["amendment_path"] = AMENDMENT_PROTOCOL_PATH.as_posix()
    effective["amendment_sha256"] = AMENDMENT_PROTOCOL_SHA256
    effective["effective_protocol_binding"] = {
        "base_protocol_sha256": V1_PROTOCOL_SHA256,
        "base_config_sha256": V1_CONFIG_SHA256,
        "amendment_protocol_sha256": AMENDMENT_PROTOCOL_SHA256,
        "amendment_config_sha256": AMENDMENT_CONFIG_SHA256,
    }
    effective["status"] = "prospective_blinded_pre_terminal_contingency"

    trigger = overlay.get("trigger")
    if not isinstance(trigger, Mapping):
        raise TypeError("V2 overlay trigger table is absent")
    effective["trigger"] = copy.deepcopy(dict(trigger))

    seed_partition = effective.get("seed_partition")
    if not isinstance(seed_partition, dict):
        raise TypeError("V1 seed partition is absent")
    seed_partition["forbidden_a3"] = list(A3_REGISTERED_SEEDS)

    namespace = overlay.get("artifact_namespace")
    if not isinstance(namespace, Mapping):
        raise TypeError("V2 artifact namespace table is absent")
    effective["persistence"] = copy.deepcopy(dict(namespace))

    source_freeze = overlay.get("future_source_freeze")
    if not isinstance(source_freeze, Mapping):
        raise TypeError("V2 source-freeze table is absent")
    effective["source_freeze"] = copy.deepcopy(dict(source_freeze))
    post_activation = overlay.get("post_activation_implementation")
    if not isinstance(post_activation, Mapping):
        raise TypeError("V2 post-activation implementation table is absent")
    effective["post_activation_implementation"] = copy.deepcopy(
        dict(post_activation)
    )
    commands = effective.get("commands")
    if not isinstance(commands, Mapping):
        raise TypeError("V1 command table is absent")
    effective["commands"] = {
        name: str(command)
        .replace(
            "adaptive_learning_substrate.experiment_000_lwoh_l1",
            "adaptive_learning_substrate.experiment_000_lwoh_l1_execution_v2",
        )
        .replace(
            "artifacts/experiment_000/lwoh_l1_scratch/",
            "artifacts/experiment_000/lwoh_l1_v2/scratch/",
        )
        .replace(
            "artifacts/experiment_000/lwoh_l1/",
            "artifacts/experiment_000/lwoh_l1_v2/",
        )
        for name, command in commands.items()
    }
    if any(
        "adaptive_learning_substrate.experiment_000_lwoh_l1 " in command
        or "artifacts/experiment_000/lwoh_l1/" in command
        or "artifacts/experiment_000/lwoh_l1_scratch/" in command
        for command in effective["commands"].values()
    ):
        raise RuntimeError("V2 effective command retained a V1 execution path")
    effective["_v2_overlay"] = copy.deepcopy(overlay)

    # Compare every inherited scientific value with a second pristine load.
    pristine = _read_toml(_resolve(root, BASE_CONFIG_PATH))
    for section in SCIENTIFIC_V1_SECTIONS:
        if section == "seed_partition":
            inherited_partition = copy.deepcopy(effective[section])
            if inherited_partition.pop("forbidden_a3", None) != list(
                A3_REGISTERED_SEEDS
            ) or inherited_partition != pristine[section]:
                raise RuntimeError("V2 changed an inherited seed partition")
            continue
        if effective.get(section) != pristine.get(section):
            raise RuntimeError(f"V2 changed inherited scientific section: {section}")
    _validate_seed_partitions(effective)
    effective["_semantic_delta"] = semantic_delta
    return effective


def _source_manifest(
    root: Path, paths: Sequence[str] = V2_SOURCE_PATHS
) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    if len(paths) != len(set(paths)):
        raise RuntimeError("V2 source manifest contains duplicate paths")
    for relative in sorted(paths):
        identity = _identity(root, relative)
        files[relative] = {
            "bytes": identity["bytes"],
            "sha256": identity["sha256"],
        }
    return {"files": files, "bundle_sha256": _sha256_json(files)}


def frozen_source_manifest(
    *, project_root: str | Path | None = None
) -> dict[str, Any]:
    return _source_manifest(_project_root(project_root))


def _validate_a3_source_freeze(root: Path) -> dict[str, Any]:
    freeze_path = _resolve(root, A3_FREEZE_RECORD_PATH)
    _require_hash(root, A3_FREEZE_RECORD_PATH, A3_FREEZE_RECORD_SHA256, "A3 freeze")
    record = _read_json(freeze_path, purpose="A3 freeze metadata")
    if (
        record.get("schema_version")
        != "experiment-000-readout-trace-a3-freeze-v1"
        or record.get("protocol_version") != "readout-trace-v1a3"
        or record.get("protocol_sha256") != A3_PROTOCOL_SHA256
        or record.get("config_sha256") != A3_CONFIG_SHA256
        or record.get("source_manifest_sha256") != A3_SOURCE_MANIFEST_SHA256
        or record.get("freeze_payload_sha256")
        != _record_self_hash(record, "freeze_payload_sha256")
    ):
        raise RuntimeError("A3 freeze metadata differs from its registered binding")
    manifest = record.get("source_manifest")
    if not isinstance(manifest, Mapping):
        raise TypeError("A3 freeze has no source manifest")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(A3_SOURCE_PATHS):
        raise RuntimeError("A3 freeze source path set is invalid")
    if (
        manifest.get("bundle_sha256") != A3_SOURCE_MANIFEST_SHA256
        or _sha256_json(files) != A3_SOURCE_MANIFEST_SHA256
    ):
        raise RuntimeError("A3 source bundle hash is invalid")
    for relative in A3_SOURCE_PATHS:
        identity = files.get(relative)
        if not isinstance(identity, Mapping):
            raise TypeError(f"A3 source identity is invalid: {relative}")
        current = _identity(root, relative)
        if (
            identity.get("bytes") != current["bytes"]
            or identity.get("sha256") != current["sha256"]
        ):
            raise RuntimeError(f"A3 frozen source drift: {relative}")
    if files[
        "src/adaptive_learning_substrate/experiment_000_readout_trace_a3.py"
    ].get("sha256") != A3_RUNNER_SHA256:
        raise RuntimeError("A3 runner hash differs from its registered binding")
    pre_freeze = record.get("pre_freeze_verification")
    if not isinstance(pre_freeze, Mapping):
        raise TypeError("A3 freeze lacks pre-freeze verification binding")
    pre_identity = _identity(root, A3_PRE_FREEZE_VERIFICATION_PATH)
    if pre_freeze.get("file_sha256") != pre_identity["sha256"]:
        raise RuntimeError("A3 pre-freeze verification hash differs")
    pre_record = _read_json(
        _resolve(root, A3_PRE_FREEZE_VERIFICATION_PATH),
        purpose="A3 pre-freeze metadata",
    )
    if (
        pre_record.get("status") != "PASS"
        or pre_record.get("verification_payload_sha256")
        != _record_self_hash(pre_record, "verification_payload_sha256")
        or pre_freeze.get("verification_payload_sha256")
        != pre_record.get("verification_payload_sha256")
    ):
        raise RuntimeError("A3 pre-freeze verification is invalid")
    return {
        "freeze_record": _identity(root, A3_FREEZE_RECORD_PATH),
        "source_manifest_sha256": A3_SOURCE_MANIFEST_SHA256,
        "source_file_count": len(A3_SOURCE_PATHS),
        "pre_freeze_verification": pre_identity,
    }


def _validate_a3_phase_sequence(
    root: Path, *, terminal_must_be_absent: bool
) -> dict[str, Any]:
    path = _resolve(root, A3_PHASE_SEQUENCE_PATH)
    record = _read_json(path, purpose="A3 non-scientific phase metadata")
    if (
        record.get("schema_version") != "experiment-000-readout-trace-a3-phases-v1"
        or record.get("protocol_version") != "readout-trace-v1a3"
        or record.get("phase_order") != list(_A3_PHASE_ORDER)
        or record.get("phase_sequence_payload_sha256")
        != _record_self_hash(record, "phase_sequence_payload_sha256")
    ):
        raise RuntimeError("A3 phase-sequence metadata is invalid")
    phases = record.get("phases")
    if not isinstance(phases, list):
        raise TypeError("A3 phase sequence has no phase list")
    names = [phase.get("name") if isinstance(phase, Mapping) else None for phase in phases]
    if names != list(_A3_PHASE_ORDER[: len(names)]):
        raise RuntimeError("A3 phase sequence is not an exact registered prefix")
    if terminal_must_be_absent and "determinism_verification" in names:
        raise RuntimeError("A3 terminal phase already exists")
    normalized_phases: list[dict[str, Any]] = []
    for phase in phases:
        assert isinstance(phase, Mapping)
        name = str(phase["name"])
        expected_paths = [value.as_posix() for value in _A3_PHASE_ARTIFACTS[name]]
        artifacts = phase.get("artifacts")
        if not isinstance(artifacts, list):
            raise TypeError(f"A3 phase has no artifact list: {name}")
        if [item.get("path") for item in artifacts if isinstance(item, Mapping)] != expected_paths:
            raise RuntimeError(f"A3 phase artifact paths differ: {name}")
        normalized: list[dict[str, Any]] = []
        for item, relative in zip(artifacts, expected_paths, strict=True):
            if not isinstance(item, Mapping):
                raise TypeError(f"A3 phase artifact identity is invalid: {name}")
            current = _identity(root, relative)
            if (
                item.get("bytes") != current["bytes"]
                or item.get("sha256") != current["sha256"]
            ):
                raise RuntimeError(f"A3 phase artifact drift: {relative}")
            normalized.append(current)
        normalized_phases.append({"name": name, "artifacts": normalized})
    return {
        "identity": _identity(root, A3_PHASE_SEQUENCE_PATH),
        "phase_names": names,
        "phases": normalized_phases,
        "payload_sha256": record["phase_sequence_payload_sha256"],
    }


def _validate_phase_extension(
    frozen_prefix: Sequence[Mapping[str, Any]], current: Mapping[str, Any]
) -> None:
    current_phases = current.get("phases")
    if not isinstance(current_phases, list) or len(current_phases) < len(frozen_prefix):
        raise RuntimeError("current A3 phase sequence is shorter than frozen prefix")
    for expected, observed in zip(frozen_prefix, current_phases, strict=False):
        if expected != observed:
            raise RuntimeError("current A3 phase sequence does not preserve frozen prefix")


def _opaque_a3_prerequisites(root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.as_posix(): _opaque_identity(root, path)
        for path in A3_OPAQUE_PREREQUISITE_PATHS
    }


def _validate_opaque_a3_extension(
    root: Path, frozen_opaque: Mapping[str, Any]
) -> None:
    expected_paths = {path.as_posix() for path in A3_OPAQUE_PREREQUISITE_PATHS}
    if set(frozen_opaque) != expected_paths:
        raise RuntimeError("V2 opaque A3 prerequisite path set differs")
    mutable_paths = {
        A3_PHASE_SEQUENCE_PATH.as_posix(),
        A3_RERUN_REPORT_PATH.as_posix(),
        A3_RERUN_SIDECAR_PATH.as_posix(),
        A3_TERMINAL_PATH.as_posix(),
    }
    for relative, frozen in frozen_opaque.items():
        if not isinstance(relative, str) or not isinstance(frozen, Mapping):
            raise TypeError("V2 opaque prerequisite binding is malformed")
        current = _opaque_identity(root, relative)
        mutable = relative in mutable_paths
        if not mutable and current != frozen:
            raise RuntimeError(f"opaque A3 prerequisite changed: {relative}")
        if (
            mutable
            and frozen.get("present") is True
            and relative != A3_PHASE_SEQUENCE_PATH.as_posix()
            and current != frozen
        ):
            raise RuntimeError(f"opaque A3 prerequisite changed: {relative}")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _local_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def _is_canonical_utc(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.utcoffset() == UTC.utcoffset(parsed) and value == parsed.isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_freeze_and_sidecar(
    target: Path, record: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    record_bytes = _canonical_json_bytes(record)
    record_sha256 = hashlib.sha256(record_bytes).hexdigest()
    sidecar_path = target.with_suffix(".sha256")
    sidecar = {
        "algorithm": "sha256",
        "report_file": target.name,
        "report_sha256": record_sha256,
    }
    sidecar_bytes = _canonical_json_bytes(sidecar)
    nonce = hashlib.sha256(os.urandom(32)).hexdigest()
    final_directory = target.parent
    final_directory.parent.mkdir(parents=True, exist_ok=True)
    stage_directory = final_directory.with_name(
        f".{final_directory.name}.{nonce}.pending"
    )
    stage_directory.mkdir()
    record_stage = stage_directory / target.name
    sidecar_stage = stage_directory / sidecar_path.name
    try:
        _exclusive_write(record_stage, record_bytes)
        _exclusive_write(sidecar_stage, sidecar_bytes)
        if final_directory.exists():
            raise FileExistsError("immutable artifact namespace already exists")
        # One same-volume directory rename publishes the fully written record
        # and sidecar together.  A crash can expose either neither file or both,
        # never a partially written/single-file freeze pair.
        stage_directory.rename(final_directory)
    finally:
        record_stage.unlink(missing_ok=True)
        sidecar_stage.unlink(missing_ok=True)
        try:
            stage_directory.rmdir()
        except FileNotFoundError:
            pass
    return dict(record), sidecar


def _validated_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    """Accept only non-scientific, registered validation command evidence."""

    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if set(row) != {"command", "input", "literal_output", "exit_status"}:
            raise ValueError(
                "validation evidence must have exact command/input/output/exit fields"
            )
        command = row["command"]
        input_value = row["input"]
        output = row["literal_output"]
        exit_status = row["exit_status"]
        if not isinstance(command, str):
            raise TypeError("validation evidence command must be text")
        if command not in _VALIDATION_COMMANDS:
            raise ValueError("validation evidence command is not registered")
        if command in seen:
            raise ValueError("validation evidence contains a duplicate command")
        seen.add(command)
        if not isinstance(input_value, str) or not isinstance(output, str):
            raise TypeError("validation evidence input and output must be text")
        if command in _EXTERNAL_VALIDATION_ARGV and input_value != "":
            raise ValueError("registered validation commands accept no standard input")
        if type(exit_status) is not int or exit_status != 0:
            raise RuntimeError("only observed passing validation evidence may be frozen")
        if len(input_value) > 4096 or len(output) > 16384:
            raise ValueError("validation evidence exceeds its non-scientific size bound")
        normalized = re.sub(r"[^a-z0-9_]+", "_", f"{input_value}\n{output}".lower())
        if any(term in normalized for term in _FORBIDDEN_EVIDENCE_TERMS):
            raise ValueError("validation evidence contains a forbidden A3 scientific field")
        normalized_output = output.replace("\r\n", "\n")
        if "py_compile" in command and normalized_output != "":
            raise ValueError("py_compile validation evidence must have empty output")
        if "ruff check" in command and normalized_output != "All checks passed!\n":
            raise ValueError("ruff validation evidence has an unexpected literal output")
        if "pytest" in command and _frozen_pytest_pass_count(output) <= 0:
            raise ValueError("pytest validation evidence has no passing tests")
        validated.append(
            {
                "command": command,
                "input": input_value,
                "literal_output": output,
                "exit_status": exit_status,
            }
        )
    if require_complete and seen != set(_VALIDATION_COMMANDS):
        raise ValueError("validation evidence does not contain every registered command")
    return validated


def _observed_pytest_pass_count(output: str) -> int:
    normalized = output.replace("\r\n", "\n").strip()
    match = re.fullmatch(
        r"(?P<dots>\.+)[ \t]+\[100%\]\n(?P<count>\d+) passed in "
        r"\d+(?:\.\d+)?s",
        normalized,
    )
    if match is None:
        raise RuntimeError("pytest validation output is not a canonical quiet pass")
    count = int(match.group("count"))
    if len(match.group("dots")) != count:
        raise RuntimeError("pytest progress count differs from its passing summary")
    return count


def _frozen_pytest_pass_count(output: str) -> int:
    match = re.fullmatch(r"PYTEST_PASS_COUNT=([1-9][0-9]*)\n", output)
    if match is None:
        raise RuntimeError("frozen pytest evidence is not a sanitized pass count")
    return int(match.group(1))


def _run_external_validation_command(
    root: Path, command: str
) -> subprocess.CompletedProcess[str]:
    argv = _EXTERNAL_VALIDATION_ARGV[command]
    try:
        completed = subprocess.run(
            argv,
            cwd=root,
            input="",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=600,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"registered validation command timed out: {command}"
        ) from exc
    if completed.returncode != 0:
        raise RuntimeError(f"registered validation command failed: {command}")
    return completed


def _capture_external_validation_evidence(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for command in _EXTERNAL_VALIDATION_ARGV:
        output = _run_external_validation_command(root, command).stdout
        if "pytest" in command:
            output = f"PYTEST_PASS_COUNT={_observed_pytest_pass_count(output)}\n"
        rows.append(
            {
                "command": command,
                "input": "",
                "literal_output": output,
                "exit_status": 0,
            }
        )
    _validated_evidence(rows, require_complete=False)
    return rows


def _replay_external_validation_evidence(
    root: Path, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Rerun every registered external check without shell interpretation."""

    by_command = {str(row["command"]): row for row in rows}
    replayed: list[dict[str, Any]] = []
    for command in _EXTERNAL_VALIDATION_ARGV:
        frozen = by_command.get(command)
        if frozen is None:
            raise RuntimeError(f"registered validation evidence is absent: {command}")
        completed = _run_external_validation_command(root, command)
        output = completed.stdout
        frozen_output = str(frozen["literal_output"])
        if "pytest" in command:
            if _observed_pytest_pass_count(output) != _frozen_pytest_pass_count(
                frozen_output
            ):
                raise RuntimeError("pytest passing-test count differs from frozen evidence")
        elif output.replace("\r\n", "\n") != frozen_output.replace("\r\n", "\n"):
            raise RuntimeError(f"validation output differs from frozen evidence: {command}")
        replayed.append(
            {
                "command": command,
                "exit_status": completed.returncode,
                "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                "pytest_pass_count": (
                    _observed_pytest_pass_count(output) if "pytest" in command else None
                ),
            }
        )
    return replayed


def _internal_validation_evidence(a3_phase: Mapping[str, Any]) -> list[dict[str, Any]]:
    identity = a3_phase.get("identity")
    phase_names = a3_phase.get("phase_names")
    if not isinstance(identity, Mapping) or not isinstance(phase_names, list):
        raise TypeError("A3 phase evidence input is invalid")
    phase_sha256 = identity.get("sha256")
    if not isinstance(phase_sha256, str):
        raise TypeError("A3 phase evidence hash is invalid")
    return [
        {
            "command": "internal:validate_v1_v2_hashes_and_allowlisted_semantic_delta",
            "input": (
                f"v1_protocol={V1_PROTOCOL_SHA256};v1_config={V1_CONFIG_SHA256};"
                f"amendment={AMENDMENT_PROTOCOL_SHA256};overlay={AMENDMENT_CONFIG_SHA256}"
            ),
            "literal_output": (
                "SCIENTIFIC_DESIGN_INHERITED_UNCHANGED=True\n"
                "SEED_PARTITIONS_DISJOINT=True"
            ),
            "exit_status": 0,
        },
        {
            "command": "internal:validate_a3_frozen_source_and_preterminal_phase_prefix",
            "input": (
                f"a3_freeze={A3_FREEZE_RECORD_SHA256};"
                f"a3_source_bundle={A3_SOURCE_MANIFEST_SHA256};"
                f"phase_sequence={phase_sha256}"
            ),
            "literal_output": (
                f"A3_FROZEN_SOURCE_FILE_COUNT={len(A3_SOURCE_PATHS)}\n"
                f"A3_PHASE_PREFIX={','.join(str(name) for name in phase_names)}\n"
                "A3_TERMINAL_ABSENT=True\nA3_REPORTS_PARSED=False"
            ),
            "exit_status": 0,
        },
    ]


def create_contingency_freeze(
    output_path: str | Path = CONTINGENCY_FREEZE_PATH,
    *,
    project_root: str | Path | None = None,
    validation_evidence: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create the write-once blinded V2 freeze while A3 is pre-terminal."""

    root = _project_root(project_root)
    target = _resolve(root, output_path)
    expected_target = _resolve(root, CONTINGENCY_FREEZE_PATH)
    if target != expected_target:
        raise ValueError("V2 contingency freeze requires its canonical path")
    terminal = _resolve(root, A3_TERMINAL_PATH)
    if terminal.exists():
        raise RuntimeError("A3 terminal artifact already exists; V2 freeze is too late")
    if target.exists() or target.with_suffix(".sha256").exists():
        raise FileExistsError("V2 contingency freeze is immutable and already exists")
    future_paths = (FUTURE_EXECUTION_RUNNER_PATH, FUTURE_EXECUTION_TEST_PATH)
    _require_future_execution_absent(root, context="before the A3 terminal decision")
    _require_activation_provenance_absent(
        root, context="before the pre-terminal freeze"
    )

    _require_hash(root, BASE_PROTOCOL_PATH, V1_PROTOCOL_SHA256, "V1 protocol")
    _require_hash(root, BASE_CONFIG_PATH, V1_CONFIG_SHA256, "V1 configuration")
    effective = load_effective_config(project_root=root)
    semantic_delta = dict(effective["_semantic_delta"])
    seed_validation = _validate_seed_partitions(effective)
    a3_source = _validate_a3_source_freeze(root)
    a3_phase = _validate_a3_phase_sequence(root, terminal_must_be_absent=True)

    # Opaque collection is intentionally separate from `_read_json`; neither
    # report nor report sidecar is deserialized here.
    opaque_before = _opaque_a3_prerequisites(root)
    if opaque_before[A3_TERMINAL_PATH.as_posix()]["present"]:
        raise RuntimeError("A3 terminal artifact appeared during V2 freeze")
    manifest = _source_manifest(root)
    environment = {
        "implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    evidence = _internal_validation_evidence(a3_phase)
    external_evidence = (
        _capture_external_validation_evidence(root)
        if validation_evidence is None
        else [dict(row) for row in validation_evidence]
    )
    evidence.extend(external_evidence)
    evidence = _validated_evidence(evidence)

    created_utc = _utc_now()
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "status": "FROZEN_BLINDED_PRE_TERMINAL",
        "created_utc": created_utc,
        "created_local": _local_now(),
        "local_timezone": str(datetime.now().astimezone().tzinfo),
        "environment": environment,
        "creator_process": {
            "process_id": os.getpid(),
            "process_instance_token": _PROCESS_INSTANCE_TOKEN,
            "created_utc": created_utc,
        },
        "base_v1": {
            "protocol": _identity(root, BASE_PROTOCOL_PATH),
            "configuration": _identity(root, BASE_CONFIG_PATH),
            "scientific_design_inherited_unchanged": True,
        },
        "amendment_v2": {
            "protocol": _identity(root, AMENDMENT_PROTOCOL_PATH),
            "configuration": _identity(root, AMENDMENT_CONFIG_PATH),
            "semantic_delta": semantic_delta,
            "seed_validation": seed_validation,
        },
        "v2_source_manifest": manifest,
        "a3_frozen_source": a3_source,
        "a3_phase_prefix": a3_phase,
        "a3_opaque_prerequisites": opaque_before,
        "a3_terminal_absence": {
            "path": A3_TERMINAL_PATH.as_posix(),
            "absent_at_start": True,
            "absent_before_commit": True,
        },
        "post_activation_source_absence": {
            path.as_posix(): {
                "path": path.as_posix(),
                "absent_at_start": True,
                "absent_before_commit": True,
            }
            for path in future_paths
        },
        "activation_provenance_absence": {
            "path": ACTIVATION_PROVENANCE_PATH.as_posix(),
            "sidecar_path": ACTIVATION_PROVENANCE_SIDECAR_PATH.as_posix(),
            "absent_at_start": True,
            "absent_before_commit": True,
        },
        "blinding": {
            "a3_task_reports_parsed": False,
            "a3_scientific_fields_observed": False,
            "allowed_observations": [
                "path_existence",
                "byte_length",
                "mtime_ns",
                "whole_file_sha256",
                "a3_source_freeze_integrity",
                "non_scientific_phase_artifact_bindings",
            ],
        },
        "validation_evidence": evidence,
        "artifact_paths": {
            "freeze": CONTINGENCY_FREEZE_PATH.as_posix(),
            "sidecar": CONTINGENCY_FREEZE_SIDECAR_PATH.as_posix(),
            "phase_ledger": PHASE_LEDGER_PATH.as_posix(),
            "activation_provenance": ACTIVATION_PROVENANCE_PATH.as_posix(),
            "activation_provenance_sidecar": (
                ACTIVATION_PROVENANCE_SIDECAR_PATH.as_posix()
            ),
            "future_execution_runner": FUTURE_EXECUTION_RUNNER_PATH.as_posix(),
            "future_execution_test": FUTURE_EXECUTION_TEST_PATH.as_posix(),
            "implementation_source_freeze": (
                IMPLEMENTATION_SOURCE_FREEZE_PATH.as_posix()
            ),
            "implementation_source_freeze_sidecar": (
                IMPLEMENTATION_SOURCE_FREEZE_SIDECAR_PATH.as_posix()
            ),
        },
    }
    record["freeze_payload_sha256"] = _record_self_hash(
        record, "freeze_payload_sha256"
    )

    # Re-open all mutable prerequisites immediately before the exclusive write.
    if terminal.exists():
        raise RuntimeError("A3 terminal artifact appeared before V2 freeze commit")
    if _source_manifest(root) != manifest:
        raise RuntimeError("V2 source manifest changed during freeze creation")
    if _validate_a3_source_freeze(root) != a3_source:
        raise RuntimeError("A3 frozen sources changed during V2 freeze creation")
    current_phase = _validate_a3_phase_sequence(root, terminal_must_be_absent=True)
    if current_phase != a3_phase:
        raise RuntimeError("A3 phase sequence changed during V2 freeze creation")
    if _opaque_a3_prerequisites(root) != opaque_before:
        raise RuntimeError("A3 opaque prerequisites changed during V2 freeze creation")
    if any(_resolve(root, path).exists() for path in future_paths):
        raise RuntimeError("post-activation execution source raced the V2 freeze commit")
    _require_activation_provenance_absent(
        root, context="before the pre-terminal freeze commit"
    )

    _write_freeze_and_sidecar(target, record)
    try:
        if terminal.exists():
            raise RuntimeError("A3 terminal artifact raced the V2 freeze commit")
        if any(_resolve(root, path).exists() for path in future_paths):
            raise RuntimeError(
                "post-activation execution source raced the V2 freeze commit"
            )
        _require_activation_provenance_absent(
            root, context="during pre-terminal freeze creation"
        )
        verification = verify_contingency_freeze(
            target,
            project_root=root,
            require_terminal_absent=True,
            require_fresh_process=False,
            replay_external_validation=False,
        )
        if terminal.exists() or any(
            _resolve(root, path).exists() for path in future_paths
        ):
            raise RuntimeError(
                "pre-terminal prerequisite changed during V2 creation verification"
            )
        _require_activation_provenance_absent(
            root, context="during pre-terminal freeze creation verification"
        )
        return verification
    except BaseException:
        # This process published the pair immediately above.  A failed creation
        # check must never strand an invalid object that looks immutable.
        target.unlink(missing_ok=True)
        target.with_suffix(".sha256").unlink(missing_ok=True)
        try:
            target.parent.rmdir()
        except OSError:
            pass
        raise


def _validate_sidecar(target: Path, sidecar_path: Path) -> dict[str, Any]:
    if not target.is_file() or not sidecar_path.is_file():
        raise RuntimeError("V2 freeze or binding sidecar is absent")
    sidecar = _read_json(sidecar_path, purpose="V2 freeze sidecar")
    actual = _file_sha256(target)
    expected_sidecar = {
        "algorithm": "sha256",
        "report_file": target.name,
        "report_sha256": actual,
    }
    if sidecar != expected_sidecar or _stable_file_bytes(
        sidecar_path
    ) != _canonical_json_bytes(expected_sidecar):
        raise RuntimeError("V2 freeze sidecar is invalid")
    return {"file_sha256": actual, "sidecar_sha256": _file_sha256(sidecar_path)}


def verify_contingency_freeze(
    path: str | Path = CONTINGENCY_FREEZE_PATH,
    *,
    project_root: str | Path | None = None,
    require_terminal_absent: bool = True,
    require_fresh_process: bool = True,
    replay_external_validation: bool = True,
) -> dict[str, Any]:
    """Independently re-open and validate the V2 amendment freeze."""

    root = _project_root(project_root)
    target = _resolve(root, path)
    if target != _resolve(root, CONTINGENCY_FREEZE_PATH):
        raise ValueError("V2 contingency verification requires its canonical path")
    identities = _validate_sidecar(target, target.with_suffix(".sha256"))
    record = _read_json(target, purpose="V2 contingency freeze")
    if _stable_file_bytes(target) != _canonical_json_bytes(record):
        raise RuntimeError("V2 contingency freeze is not canonical JSON")
    expected_record_keys = {
        "schema_version",
        "protocol_version",
        "status",
        "created_utc",
        "created_local",
        "local_timezone",
        "environment",
        "creator_process",
        "base_v1",
        "amendment_v2",
        "v2_source_manifest",
        "a3_frozen_source",
        "a3_phase_prefix",
        "a3_opaque_prerequisites",
        "a3_terminal_absence",
        "post_activation_source_absence",
        "activation_provenance_absence",
        "blinding",
        "validation_evidence",
        "artifact_paths",
        "freeze_payload_sha256",
    }
    if (
        set(record) != expected_record_keys
        or record.get("schema_version") != SCHEMA_VERSION
        or record.get("protocol_version") != PROTOCOL_VERSION
        or record.get("status") != "FROZEN_BLINDED_PRE_TERMINAL"
        or record.get("freeze_payload_sha256")
        != _record_self_hash(record, "freeze_payload_sha256")
    ):
        raise RuntimeError("V2 contingency freeze metadata or self-hash is invalid")
    try:
        created_local = datetime.fromisoformat(str(record.get("created_local")))
    except ValueError as exc:
        raise RuntimeError("V2 freeze local timestamp is invalid") from exc
    creator = record.get("creator_process")
    if not isinstance(creator, Mapping):
        raise TypeError("V2 freeze creator-process binding is absent")
    creator_token = creator.get("process_instance_token")
    creator_pid = creator.get("process_id")
    try:
        created_utc = datetime.fromisoformat(str(record.get("created_utc")))
    except ValueError as exc:
        raise RuntimeError("V2 freeze UTC timestamp is invalid") from exc
    if (
        not _is_canonical_utc(record.get("created_utc"))
        or created_local.utcoffset() is None
        or not isinstance(record.get("local_timezone"), str)
        or not record["local_timezone"]
        or abs((created_local.astimezone(UTC) - created_utc).total_seconds()) >= 1
        or set(creator) != {"process_id", "process_instance_token", "created_utc"}
        or creator.get("created_utc") != record.get("created_utc")
        or type(creator_pid) is not int
        or creator_pid <= 0
        or not isinstance(creator_token, str)
        or re.fullmatch(r"[0-9a-f]{64}", creator_token) is None
    ):
        raise RuntimeError("V2 freeze timestamp or creator-process binding is invalid")
    fresh_process = (
        creator_pid != os.getpid() and creator_token != _PROCESS_INSTANCE_TOKEN
    )
    if require_fresh_process and not fresh_process:
        raise RuntimeError("V2 freeze requires verification in a separate fresh process")
    expected_environment = {
        "implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if record.get("environment") != expected_environment:
        raise RuntimeError("V2 freeze environment binding differs")
    if record.get("artifact_paths") != {
        "freeze": CONTINGENCY_FREEZE_PATH.as_posix(),
        "sidecar": CONTINGENCY_FREEZE_SIDECAR_PATH.as_posix(),
        "phase_ledger": PHASE_LEDGER_PATH.as_posix(),
        "activation_provenance": ACTIVATION_PROVENANCE_PATH.as_posix(),
        "activation_provenance_sidecar": (
            ACTIVATION_PROVENANCE_SIDECAR_PATH.as_posix()
        ),
        "future_execution_runner": FUTURE_EXECUTION_RUNNER_PATH.as_posix(),
        "future_execution_test": FUTURE_EXECUTION_TEST_PATH.as_posix(),
        "implementation_source_freeze": IMPLEMENTATION_SOURCE_FREEZE_PATH.as_posix(),
        "implementation_source_freeze_sidecar": (
            IMPLEMENTATION_SOURCE_FREEZE_SIDECAR_PATH.as_posix()
        ),
    }:
        raise RuntimeError("V2 freeze artifact path binding differs")
    expected_future_absence = {
        path.as_posix(): {
            "path": path.as_posix(),
            "absent_at_start": True,
            "absent_before_commit": True,
        }
        for path in (FUTURE_EXECUTION_RUNNER_PATH, FUTURE_EXECUTION_TEST_PATH)
    }
    if record.get("post_activation_source_absence") != expected_future_absence:
        raise RuntimeError("V2 freeze future-source absence binding differs")
    if record.get("activation_provenance_absence") != {
        "path": ACTIVATION_PROVENANCE_PATH.as_posix(),
        "sidecar_path": ACTIVATION_PROVENANCE_SIDECAR_PATH.as_posix(),
        "absent_at_start": True,
        "absent_before_commit": True,
    }:
        raise RuntimeError("V2 freeze activation-provenance absence binding differs")
    absence = record.get("a3_terminal_absence")
    if absence != {
        "path": A3_TERMINAL_PATH.as_posix(),
        "absent_at_start": True,
        "absent_before_commit": True,
    }:
        raise RuntimeError("V2 freeze lacks exact A3 pre-terminal evidence")
    blinding = record.get("blinding")
    if blinding != {
        "a3_task_reports_parsed": False,
        "a3_scientific_fields_observed": False,
        "allowed_observations": [
            "path_existence",
            "byte_length",
            "mtime_ns",
            "whole_file_sha256",
            "a3_source_freeze_integrity",
            "non_scientific_phase_artifact_bindings",
        ],
    }:
        raise RuntimeError("V2 freeze does not attest the blinded boundary")
    validation_evidence = record.get("validation_evidence")
    if not isinstance(validation_evidence, list) or _validated_evidence(
        validation_evidence, require_complete=True
    ) != validation_evidence:
        raise RuntimeError("V2 freeze validation evidence is invalid")
    replayed_validation: list[dict[str, Any]] = []

    effective = load_effective_config(project_root=root)
    semantic_delta = effective["_semantic_delta"]
    seed_validation = _validate_seed_partitions(effective)
    amendment = record.get("amendment_v2")
    base = record.get("base_v1")
    if not isinstance(amendment, Mapping) or not isinstance(base, Mapping):
        raise TypeError("V2 base/amendment bindings are absent")
    expected_base = {
        "protocol": _identity(root, BASE_PROTOCOL_PATH),
        "configuration": _identity(root, BASE_CONFIG_PATH),
        "scientific_design_inherited_unchanged": True,
    }
    expected_amendment = {
        "protocol": _identity(root, AMENDMENT_PROTOCOL_PATH),
        "configuration": _identity(root, AMENDMENT_CONFIG_PATH),
        "semantic_delta": semantic_delta,
        "seed_validation": seed_validation,
    }
    if dict(base) != expected_base or dict(amendment) != expected_amendment:
        raise RuntimeError("V2 base or amendment binding differs")
    manifest = _source_manifest(root)
    if record.get("v2_source_manifest") != manifest:
        raise RuntimeError("V2 frozen source manifest differs from current files")
    a3_source = _validate_a3_source_freeze(root)
    if record.get("a3_frozen_source") != a3_source:
        raise RuntimeError("V2 freeze A3 source binding differs")

    terminal_exists = _resolve(root, A3_TERMINAL_PATH).is_file()
    if require_terminal_absent and terminal_exists:
        raise RuntimeError("A3 terminal artifact exists during pre-terminal verification")
    if require_terminal_absent:
        _require_future_execution_absent(
            root, context="during pre-terminal freeze verification"
        )
        _require_activation_provenance_absent(
            root, context="during pre-terminal freeze verification"
        )
    current_phase = _validate_a3_phase_sequence(
        root, terminal_must_be_absent=require_terminal_absent
    )
    frozen_phase = record.get("a3_phase_prefix")
    if not isinstance(frozen_phase, Mapping):
        raise TypeError("V2 freeze lacks an A3 phase prefix")
    if set(frozen_phase) != {"identity", "phase_names", "phases", "payload_sha256"}:
        raise RuntimeError("V2 frozen A3 phase-prefix schema differs")
    frozen_phases = frozen_phase.get("phases")
    if not isinstance(frozen_phases, list):
        raise TypeError("V2 frozen A3 phase prefix is invalid")
    if frozen_phase.get("phase_names") != [
        phase.get("name") if isinstance(phase, Mapping) else None
        for phase in frozen_phases
    ]:
        raise RuntimeError("V2 frozen A3 phase names differ from its phase prefix")
    if validation_evidence[:2] != _internal_validation_evidence(frozen_phase):
        raise RuntimeError("V2 internal validation evidence differs")
    _validate_phase_extension(frozen_phases, current_phase)

    opaque = record.get("a3_opaque_prerequisites")
    if not isinstance(opaque, Mapping):
        raise TypeError("V2 freeze lacks opaque A3 prerequisite identities")
    # The phase sequence is an append-only mutable ledger.  Rerun and terminal
    # artifacts may legitimately appear after a pre-terminal freeze.  Every
    # other file that existed at freeze must remain exact.
    _validate_opaque_a3_extension(root, opaque)

    # Execute project code only after all immutable file, source-manifest, A3,
    # and blinded-record bindings above have been revalidated.
    if replay_external_validation:
        replayed_validation = _replay_external_validation_evidence(
            root, validation_evidence
        )
        if _source_manifest(root) != manifest:
            raise RuntimeError("V2 sources changed during validation replay")
        if _validate_a3_source_freeze(root) != a3_source:
            raise RuntimeError("A3 frozen sources changed during validation replay")
        replay_phase = _validate_a3_phase_sequence(
            root, terminal_must_be_absent=require_terminal_absent
        )
        _validate_phase_extension(frozen_phases, replay_phase)
        _validate_opaque_a3_extension(root, opaque)
        if require_terminal_absent and _resolve(root, A3_TERMINAL_PATH).exists():
            raise RuntimeError("A3 terminal artifact appeared during freeze verification")
        if require_terminal_absent:
            _require_future_execution_absent(
                root, context="during validation replay"
            )
            _require_activation_provenance_absent(
                root, context="during validation replay"
            )
        current_phase = replay_phase

    terminal_exists = _resolve(root, A3_TERMINAL_PATH).is_file()
    if require_terminal_absent and terminal_exists:
        raise RuntimeError("A3 terminal artifact appeared during freeze verification")

    return {
        "schema_version": "experiment-000-lwoh-l1-v2-freeze-verification-v1",
        "status": (
            "PASS"
            if fresh_process and replay_external_validation
            else "CREATED_PENDING_FRESH_PROCESS_VERIFICATION"
        ),
        "verified_utc": _utc_now(),
        "verification_process_id": os.getpid(),
        "verification_process_instance_token": _PROCESS_INSTANCE_TOKEN,
        "fresh_process_verified": fresh_process,
        "external_validation_replayed": replay_external_validation,
        "external_validation_replay": replayed_validation,
        "freeze_file_sha256": identities["file_sha256"],
        "freeze_sidecar_sha256": identities["sidecar_sha256"],
        "freeze_payload_sha256": record["freeze_payload_sha256"],
        "v2_source_manifest_sha256": manifest["bundle_sha256"],
        "a3_freeze_record_sha256": a3_source["freeze_record"]["sha256"],
        "a3_source_manifest_sha256": a3_source["source_manifest_sha256"],
        "a3_phase_sequence_sha256": current_phase["identity"]["sha256"],
        "a3_terminal_present": terminal_exists,
        "scientific_design_inherited_unchanged": True,
        "a3_reports_parsed": False,
    }


def classify_a3_terminal(
    status: object, *, evidence_valid: bool
) -> dict[str, str]:
    """Classify a validated terminal status without observing any metric."""

    if not evidence_valid or not isinstance(status, str):
        return {
            "status": "INVALID_A3_DO_NOT_ACTIVATE_LWOH",
            "action": "REPAIR_A3_DO_NOT_ACTIVATE_LWOH",
        }
    if status == "NO_SELECTION":
        return {
            "status": "ACTIVATED_A3_NO_SELECTION",
            "action": "ACTIVATE_LWOH_L1_V2",
        }
    if status.startswith("SELECTED:") and status.removeprefix(
        "SELECTED:"
    ) in REGISTERED_A3_SELECTED_CONDITIONS:
        return {
            "status": "CANCELLED_A3_SELECTED",
            "action": "CANCEL_LWOH_L1_V2",
        }
    return {
        "status": "INVALID_A3_DO_NOT_ACTIVATE_LWOH",
        "action": "REPAIR_A3_DO_NOT_ACTIVATE_LWOH",
    }


def _validated_a3_terminal_binding(root: Path) -> dict[str, Any]:
    """Replay-validate A3 without invoking its terminal verifier."""

    freeze_verification = verify_contingency_freeze(
        project_root=root, require_terminal_absent=False
    )
    if (
        freeze_verification.get("status") != "PASS"
        or freeze_verification.get("fresh_process_verified") is not True
        or freeze_verification.get("external_validation_replayed") is not True
    ):
        raise RuntimeError("V2 pre-terminal freeze lacks independent verification")
    if not _resolve(root, A3_TERMINAL_PATH).is_file():
        raise RuntimeError("A3 terminal artifact is absent")

    a3 = importlib.import_module(
        "adaptive_learning_substrate.experiment_000_readout_trace_a3"
    )
    _require_a3_module_root(root, a3)
    # These are the frozen validator primitives used by the A3 verifier.  The
    # verifier entry point itself is intentionally never called here.
    required_primitives = (
        "_source_provenance",
        "_validate_freeze_record",
        "_validate_phase_sequence",
        "_load_full_input",
        "_validate_report_mapping",
        "_validate_smoke_verification",
        "_validate_determinism_verification",
        "_file_sha256",
        "_record_self_hash",
        "_single_thread_blas_state",
        "threadpool_limits",
        "DEFAULT_SMOKE_REPORT_PATH",
        "SMOKE_VERIFICATION_PATH",
        "PRIMARY_REPORT_PATH",
        "RERUN_REPORT_PATH",
        "_PHASE_ORDER",
    )
    if any(not hasattr(a3, name) for name in required_primitives):
        raise RuntimeError("frozen A3 validator primitive is absent")

    with a3.threadpool_limits(limits=1, user_api="blas"):
        a3._single_thread_blas_state()
        return _replay_validated_a3_terminal_binding(
            root, a3=a3, freeze_verification=freeze_verification
        )


def _replay_validated_a3_terminal_binding(
    root: Path, *, a3: Any, freeze_verification: Mapping[str, Any]
) -> dict[str, Any]:
    """Execute the frozen A3 replay checks under one-thread BLAS limits."""

    phase_info = a3._validate_phase_sequence(a3._PHASE_ORDER)
    primary, primary_files = a3._load_full_input(
        a3.PRIMARY_REPORT_PATH, expected_path=a3.PRIMARY_REPORT_PATH
    )
    rerun, rerun_files = a3._load_full_input(
        a3.RERUN_REPORT_PATH, expected_path=a3.RERUN_REPORT_PATH
    )
    # Reject malformed report payloads before relying on external prerequisites,
    # matching the frozen verifier's validation order.
    a3._validate_report_mapping(primary, expected_kind="full")
    a3._validate_report_mapping(rerun, expected_kind="full")

    current_manifest = a3._source_provenance()
    if current_manifest.get("bundle_sha256") != A3_SOURCE_MANIFEST_SHA256:
        raise RuntimeError("current A3 source bundle differs from its freeze")
    freeze_info = a3._validate_freeze_record(current_manifest=current_manifest)
    a3._validate_report_mapping(
        primary,
        expected_kind="full",
        freeze_info=freeze_info,
        current_manifest=current_manifest,
    )
    a3._validate_report_mapping(
        rerun,
        expected_kind="full",
        freeze_info=freeze_info,
        current_manifest=current_manifest,
    )
    expected_rerun_binding = {
        "primary_report_sha256": primary_files["report_file_sha256"],
        "primary_sidecar_sha256": primary_files["sidecar_file_sha256"],
    }
    if (
        primary.get("nondeterministic_provenance", {}).get(
            "rerun_prerequisite_artifacts"
        )
        is not None
        or rerun.get("nondeterministic_provenance", {}).get(
            "rerun_prerequisite_artifacts"
        )
        != expected_rerun_binding
    ):
        raise RuntimeError("A3 primary/rerun artifact provenance is invalid")
    a3._validate_smoke_verification(
        smoke_path=a3.DEFAULT_SMOKE_REPORT_PATH,
        verification_path=a3.SMOKE_VERIFICATION_PATH,
        freeze_info=freeze_info,
        current_manifest=current_manifest,
    )
    terminal_path = _resolve(root, A3_TERMINAL_PATH)
    terminal = _read_json(terminal_path, purpose="A3 terminal verification")
    terminal_file_sha256 = _file_sha256(terminal_path)
    a3._validate_determinism_verification(
        terminal,
        primary,
        rerun,
        left_files=primary_files,
        right_files=rerun_files,
        freeze_info=freeze_info,
        current_manifest=current_manifest,
        require_selected=False,
    )

    final_manifest = a3._source_provenance()
    if final_manifest != current_manifest:
        raise RuntimeError("A3 sources changed during contingency validation")
    final_freeze = a3._validate_freeze_record(current_manifest=final_manifest)
    if final_freeze != freeze_info:
        raise RuntimeError("A3 freeze changed during contingency validation")
    final_phase = a3._validate_phase_sequence(a3._PHASE_ORDER)
    if final_phase != phase_info:
        raise RuntimeError("A3 phase sequence changed during contingency validation")
    final_primary, final_primary_files = a3._load_full_input(
        a3.PRIMARY_REPORT_PATH, expected_path=a3.PRIMARY_REPORT_PATH
    )
    final_rerun, final_rerun_files = a3._load_full_input(
        a3.RERUN_REPORT_PATH, expected_path=a3.RERUN_REPORT_PATH
    )
    if (
        final_primary != primary
        or final_rerun != rerun
        or final_primary_files != primary_files
        or final_rerun_files != rerun_files
    ):
        raise RuntimeError("A3 full report artifacts changed during validation")
    a3._validate_report_mapping(
        final_primary,
        expected_kind="full",
        freeze_info=final_freeze,
        current_manifest=final_manifest,
    )
    a3._validate_report_mapping(
        final_rerun,
        expected_kind="full",
        freeze_info=final_freeze,
        current_manifest=final_manifest,
    )
    a3._validate_smoke_verification(
        smoke_path=a3.DEFAULT_SMOKE_REPORT_PATH,
        verification_path=a3.SMOKE_VERIFICATION_PATH,
        freeze_info=final_freeze,
        current_manifest=final_manifest,
    )
    final_terminal = _read_json(terminal_path, purpose="A3 terminal verification")
    if final_terminal != terminal or _file_sha256(terminal_path) != terminal_file_sha256:
        raise RuntimeError("A3 terminal artifact changed during contingency validation")
    a3._validate_determinism_verification(
        final_terminal,
        final_primary,
        final_rerun,
        left_files=final_primary_files,
        right_files=final_rerun_files,
        freeze_info=final_freeze,
        current_manifest=final_manifest,
        require_selected=False,
    )
    committed_manifest = a3._source_provenance()
    if committed_manifest != final_manifest:
        raise RuntimeError("A3 sources changed at contingency validation commit")
    committed_freeze = a3._validate_freeze_record(
        current_manifest=committed_manifest
    )
    if committed_freeze != final_freeze:
        raise RuntimeError("A3 freeze changed at contingency validation commit")
    committed_phase = a3._validate_phase_sequence(a3._PHASE_ORDER)
    if committed_phase != final_phase:
        raise RuntimeError("A3 phase sequence changed at contingency validation commit")
    committed_files = {
        "primary": _opaque_identity(root, A3_PRIMARY_REPORT_PATH),
        "primary_sidecar": _opaque_identity(root, A3_PRIMARY_SIDECAR_PATH),
        "rerun": _opaque_identity(root, A3_RERUN_REPORT_PATH),
        "rerun_sidecar": _opaque_identity(root, A3_RERUN_SIDECAR_PATH),
        "terminal": _opaque_identity(root, A3_TERMINAL_PATH),
    }
    expected_hashes = {
        "primary": final_primary_files["report_file_sha256"],
        "primary_sidecar": final_primary_files["sidecar_file_sha256"],
        "rerun": final_rerun_files["report_file_sha256"],
        "rerun_sidecar": final_rerun_files["sidecar_file_sha256"],
        "terminal": terminal_file_sha256,
    }
    if any(
        committed_files[name].get("sha256") != expected
        for name, expected in expected_hashes.items()
    ):
        raise RuntimeError("A3 artifact identity changed at contingency validation commit")

    terminal_status = terminal.get("status")
    classification = classify_a3_terminal(terminal_status, evidence_valid=True)
    if classification["status"] == "INVALID_A3_DO_NOT_ACTIVATE_LWOH":
        raise RuntimeError("A3 terminal status is not registered")
    provenance = {
        **classification,
        "a3_source_manifest_sha256": committed_manifest["bundle_sha256"],
        "a3_freeze_record_sha256": committed_freeze["file_sha256"],
        "a3_phase_sequence_sha256": committed_phase["file_sha256"],
        "primary_report_sha256": committed_files["primary"]["sha256"],
        "primary_sidecar_sha256": committed_files["primary_sidecar"]["sha256"],
        "rerun_report_sha256": committed_files["rerun"]["sha256"],
        "rerun_sidecar_sha256": committed_files["rerun_sidecar"]["sha256"],
        "deterministic_payload_sha256": final_terminal.get(
            "first_deterministic_payload_sha256"
        ),
        "terminal_file_sha256": committed_files["terminal"]["sha256"],
        "terminal_payload_sha256": final_terminal.get("verification_payload_sha256"),
        "contingency_freeze_sha256": freeze_verification["freeze_file_sha256"],
    }
    if classification["status"] == "ACTIVATED_A3_NO_SELECTION":
        provenance["terminal_status"] = "NO_SELECTION"
    return provenance


def evaluate_a3_contingency(
    *, project_root: str | Path | None = None
) -> dict[str, Any]:
    """Return the exact V2 branch outcome and only non-scientific bindings."""

    root = _project_root(project_root)
    try:
        return _validated_a3_terminal_binding(root)
    except (
        ArithmeticError,
        AttributeError,
        FileNotFoundError,
        ImportError,
        IndexError,
        KeyError,
        LookupError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        classification = classify_a3_terminal(None, evidence_valid=False)
        return {
            **classification,
            "validation_error_type": type(exc).__name__,
        }


def _validated_activation_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != ACTIVATION_BINDING_KEYS:
        raise RuntimeError("A3 activation binding exposed an unregistered field")
    if (
        value.get("status") != "ACTIVATED_A3_NO_SELECTION"
        or value.get("action") != "ACTIVATE_LWOH_L1_V2"
        or value.get("terminal_status") != "NO_SELECTION"
    ):
        raise RuntimeError("A3 activation binding is not an exact NO_SELECTION")
    for key in ACTIVATION_BINDING_KEYS - {"status", "action", "terminal_status"}:
        field = value.get(key)
        if not isinstance(field, str) or re.fullmatch(r"[0-9a-f]{64}", field) is None:
            raise RuntimeError(f"A3 activation provenance hash is invalid: {key}")
    return dict(value)


def _activation_source_absence() -> dict[str, dict[str, Any]]:
    return {
        path.as_posix(): {"path": path.as_posix(), "absent_at_activation": True}
        for path in (FUTURE_EXECUTION_RUNNER_PATH, FUTURE_EXECUTION_TEST_PATH)
    }


def verify_a3_activation_provenance(
    path: str | Path = ACTIVATION_PROVENANCE_PATH,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Revalidate a committed minimal activation pair and the current A3 chain."""

    root = _project_root(project_root)
    target = _resolve(root, path)
    if target != _resolve(root, ACTIVATION_PROVENANCE_PATH):
        raise ValueError("A3 activation provenance requires its canonical path")
    identities = _validate_sidecar(target, target.with_suffix(".sha256"))
    record = _read_json(target, purpose="A3 activation provenance")
    if _stable_file_bytes(target) != _canonical_json_bytes(record):
        raise RuntimeError("A3 activation provenance is not canonical JSON")
    if set(record) != {
        "schema_version",
        "protocol_version",
        "status",
        "action",
        "activated_utc",
        "activated_local",
        "local_timezone",
        "environment",
        "creator_process",
        "activation_provenance",
        "execution_sources_absent",
        "activation_payload_sha256",
    }:
        raise RuntimeError("A3 activation provenance schema differs")
    if (
        record.get("schema_version")
        != "experiment-000-lwoh-l1-v2-a3-activation-provenance-v1"
        or record.get("protocol_version") != PROTOCOL_VERSION
        or record.get("status") != "ACTIVATED_A3_NO_SELECTION"
        or record.get("action") != "ACTIVATE_LWOH_L1_V2"
        or record.get("activation_payload_sha256")
        != _record_self_hash(record, "activation_payload_sha256")
        or record.get("execution_sources_absent") != _activation_source_absence()
    ):
        raise RuntimeError("A3 activation provenance metadata or self-hash is invalid")
    try:
        activated_utc = datetime.fromisoformat(str(record.get("activated_utc")))
        activated_local = datetime.fromisoformat(str(record.get("activated_local")))
    except ValueError as exc:
        raise RuntimeError("A3 activation provenance timestamp is invalid") from exc
    creator = record.get("creator_process")
    if not isinstance(creator, Mapping):
        raise TypeError("A3 activation creator-process binding is absent")
    if (
        not _is_canonical_utc(record.get("activated_utc"))
        or activated_local.utcoffset() is None
        or abs((activated_local.astimezone(UTC) - activated_utc).total_seconds()) >= 1
        or not isinstance(record.get("local_timezone"), str)
        or not record["local_timezone"]
        or set(creator) != {"process_id", "process_instance_token", "created_utc"}
        or creator.get("created_utc") != record.get("activated_utc")
        or type(creator.get("process_id")) is not int
        or creator["process_id"] <= 0
        or not isinstance(creator.get("process_instance_token"), str)
        or re.fullmatch(r"[0-9a-f]{64}", creator["process_instance_token"])
        is None
        or record.get("environment")
        != {
            "implementation": platform.python_implementation(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
    ):
        raise RuntimeError("A3 activation timestamp, process, or environment differs")
    frozen_binding = record.get("activation_provenance")
    if not isinstance(frozen_binding, Mapping):
        raise TypeError("A3 activation provenance binding is absent")
    frozen_binding = _validated_activation_binding(frozen_binding)
    current_binding = _validated_activation_binding(
        _validated_a3_terminal_binding(root)
    )
    if current_binding != frozen_binding:
        raise RuntimeError("A3 activation provenance differs from current validated evidence")
    final_identities = _validate_sidecar(target, target.with_suffix(".sha256"))
    final_record = _read_json(target, purpose="A3 activation provenance")
    if (
        final_identities != identities
        or final_record != record
        or _stable_file_bytes(target) != _canonical_json_bytes(final_record)
    ):
        raise RuntimeError("A3 activation provenance changed during validation")
    return {
        "schema_version": "experiment-000-lwoh-l1-v2-activation-verification-v1",
        "status": "PASS",
        "verified_utc": _utc_now(),
        "activation_file_sha256": final_identities["file_sha256"],
        "activation_sidecar_sha256": final_identities["sidecar_sha256"],
        "activation_payload_sha256": record["activation_payload_sha256"],
        "activation_provenance": current_binding,
    }


def commit_a3_no_selection_activation(
    output_path: str | Path = ACTIVATION_PROVENANCE_PATH,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Atomically commit the first sanitized activation before future sources exist."""

    root = _project_root(project_root)
    target = _resolve(root, output_path)
    if target != _resolve(root, ACTIVATION_PROVENANCE_PATH):
        raise ValueError("A3 activation provenance requires its canonical path")
    _require_activation_provenance_absent(root, context="before first activation")
    _require_future_execution_absent(root, context="before first activation")
    binding = _validated_activation_binding(_validated_a3_terminal_binding(root))
    _require_future_execution_absent(root, context="during first activation")
    activated_utc = _utc_now()
    record: dict[str, Any] = {
        "schema_version": "experiment-000-lwoh-l1-v2-a3-activation-provenance-v1",
        "protocol_version": PROTOCOL_VERSION,
        "status": "ACTIVATED_A3_NO_SELECTION",
        "action": "ACTIVATE_LWOH_L1_V2",
        "activated_utc": activated_utc,
        "activated_local": _local_now(),
        "local_timezone": str(datetime.now().astimezone().tzinfo),
        "environment": {
            "implementation": platform.python_implementation(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "creator_process": {
            "process_id": os.getpid(),
            "process_instance_token": _PROCESS_INSTANCE_TOKEN,
            "created_utc": activated_utc,
        },
        "activation_provenance": binding,
        "execution_sources_absent": _activation_source_absence(),
    }
    record["activation_payload_sha256"] = _record_self_hash(
        record, "activation_payload_sha256"
    )
    _require_future_execution_absent(root, context="before activation commit")
    _write_freeze_and_sidecar(target, record)
    try:
        _require_future_execution_absent(root, context="during activation commit")
        verification = verify_a3_activation_provenance(target, project_root=root)
        _require_future_execution_absent(
            root, context="during activation commit verification"
        )
        return verification
    except BaseException:
        target.unlink(missing_ok=True)
        target.with_suffix(".sha256").unlink(missing_ok=True)
        try:
            target.parent.rmdir()
        except OSError:
            pass
        raise


def load_verified_a3_no_selection(
    *, project_root: str | Path | None = None
) -> dict[str, Any]:
    """Load only a fully replay-validated A3 ``NO_SELECTION`` trigger."""

    root = _project_root(project_root)
    activation_directory = _resolve(root, ACTIVATION_DIRECTORY)
    if activation_directory.exists():
        return dict(
            verify_a3_activation_provenance(project_root=root)[
                "activation_provenance"
            ]
        )
    _require_future_execution_absent(root, context="before first activation")
    binding = _validated_activation_binding(_validated_a3_terminal_binding(root))
    _require_future_execution_absent(root, context="during first activation")
    return binding


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LWOH-L1 V2 blinded contingency freeze and verifier"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="create the pre-terminal freeze")
    freeze.add_argument("--output", type=Path, default=CONTINGENCY_FREEZE_PATH)
    verify = subparsers.add_parser(
        "verify-freeze", help="independently verify the pre-terminal freeze"
    )
    verify.add_argument("--input", type=Path, default=CONTINGENCY_FREEZE_PATH)
    activate = subparsers.add_parser(
        "commit-activation",
        help="commit sanitized NO_SELECTION activation provenance",
    )
    activate.add_argument("--output", type=Path, default=ACTIVATION_PROVENANCE_PATH)
    verify_activation = subparsers.add_parser(
        "verify-activation", help="revalidate committed activation provenance"
    )
    verify_activation.add_argument(
        "--input", type=Path, default=ACTIVATION_PROVENANCE_PATH
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "freeze":
        result = create_contingency_freeze(args.output)
    elif args.command == "verify-freeze":
        result = verify_contingency_freeze(args.input, require_terminal_absent=True)
    elif args.command == "commit-activation":
        result = commit_a3_no_selection_activation(args.output)
    elif args.command == "verify-activation":
        result = verify_a3_activation_provenance(args.input)
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "A3_CONFIG_SHA256",
    "A3_DETERMINISM_VERIFICATION_PATH",
    "A3_FREEZE_RECORD_PATH",
    "A3_FREEZE_RECORD_SHA256",
    "A3_PHASE_SEQUENCE_PATH",
    "A3_PRIMARY_REPORT_PATH",
    "A3_PRIMARY_SIDECAR_PATH",
    "A3_PROTOCOL_SHA256",
    "A3_REGISTERED_SEEDS",
    "A3_RERUN_REPORT_PATH",
    "A3_RERUN_SIDECAR_PATH",
    "A3_RUNNER_SHA256",
    "A3_SOURCE_MANIFEST_SHA256",
    "A3_TERMINAL_PATH",
    "ACTIVATION_BINDING_KEYS",
    "ACTIVATION_DIRECTORY",
    "ACTIVATION_PROVENANCE_PATH",
    "ACTIVATION_PROVENANCE_SIDECAR_PATH",
    "AMENDMENT_CONFIG_OBJECT_SHA256",
    "AMENDMENT_CONFIG_PATH",
    "AMENDMENT_CONFIG_SHA256",
    "AMENDMENT_PROTOCOL_PATH",
    "AMENDMENT_PROTOCOL_SHA256",
    "ARTIFACT_DIRECTORY",
    "BASE_CONFIG_PATH",
    "BASE_PROTOCOL_PATH",
    "CONTINGENCY_FREEZE_PATH",
    "CONTINGENCY_FREEZE_SIDECAR_PATH",
    "FORBIDDEN_SEED_SETS",
    "FUTURE_EXECUTION_RUNNER_PATH",
    "FUTURE_EXECUTION_TEST_PATH",
    "IMPLEMENTATION_SOURCE_FREEZE_PATH",
    "IMPLEMENTATION_SOURCE_FREEZE_SIDECAR_PATH",
    "LWOH_ADMISSION_SEEDS",
    "LWOH_LEARNING_SEEDS",
    "LWOH_SCRATCH_SEEDS",
    "OVERLAY_ALLOWED_KEYS",
    "OVERLAY_ALLOWED_PATHS",
    "PHASE_LEDGER_PATH",
    "PRETERMINAL_SOURCE_PATHS",
    "PROTOCOL_VERSION",
    "RUNNER_PATH",
    "SCHEMA_VERSION",
    "TEST_PATH",
    "V1_CONFIG_SHA256",
    "V1_PROTOCOL_SHA256",
    "V2_SOURCE_PATHS",
    "classify_a3_terminal",
    "commit_a3_no_selection_activation",
    "create_contingency_freeze",
    "evaluate_a3_contingency",
    "frozen_source_manifest",
    "load_effective_config",
    "load_verified_a3_no_selection",
    "main",
    "verify_a3_activation_provenance",
    "verify_contingency_freeze",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
