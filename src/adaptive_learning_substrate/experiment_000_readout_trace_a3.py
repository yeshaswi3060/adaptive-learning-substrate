"""A3 architecture-only readout-decoupled local-trace experiment.

Implements ``readout-trace-v1a3``
(``docs/EXPERIMENT_000_READOUT_TRACE_A3_PROTOCOL.md``).
The native recurrent forward/emission path is byte-identical to ``recurrent.py``;
each hidden unit keeps one bounded passive EMA of its own activations that is
read only at ``QUERY`` through the 16 frozen output edges. CCF is never called.
Only fresh custom seeds 105--109 and the frozen ``0.25..0.95`` grid are accepted.

The stable probe, ridge, distribution, matrix, hashing, gate-threshold, and
determinism helpers are imported from :mod:`experiment_000_slow_state` and
:mod:`experiment_000_memory_probe` so the numerical contract matches the
slow-state run one-for-one.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing
import os
import platform
import struct
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

for _thread_env in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = "1"

import numpy as np
import threadpoolctl
from threadpoolctl import threadpool_info, threadpool_limits

from .experiment000_data import (
    Experiment000Episode,
    Experiment000Stream,
    SeedRole,
    generate_stream_pair,
    require_seed_role,
)
from .experiment_000 import FROZEN_GRAPH_OPTIONS, forward_episode
from .experiment_000_memory_probe import (
    _output_feature_edge_ids,
    _PairedSplit,
    extract_query_features,
    paired_retention_metrics,
)
from .experiment_000_slow_state import (
    GATE_THRESHOLDS,
    MATRIX_ABS_TOLERANCE,
    MATRIX_REL_TOLERANCE,
    SATURATION_THRESHOLDS,
    _array_bundle_sha256,
    _constructor_hyperparameters,
    _credit_method_bundle_sha256,
    _distribution,
    _edge_weight_sha256,
    _file_sha256,
    _paired_difference,
    _recurrent_matrix_metrics,
    _safe_ratio,
    _sha256_json,
    _structural_mask_sha256,
    _within_matrix_tolerance,
    strict_ridge_readout,
)
from .readout_trace_recurrent import (
    ReadoutTraceRecurrentEventGraph,
    clone_with_readout_trace,
)
from .recurrent import QueryResult, RecurrentEventGraph, build_experiment_000_graph

SCHEMA_VERSION = "experiment-000-readout-decoupled-local-trace-v1a3"
PROTOCOL_NAME = "experiment_000_readout_decoupled_local_trace_a3"
PROTOCOL_VERSION = "readout-trace-v1a3"
REGISTERED_SEEDS: tuple[int, ...] = (105, 106, 107, 108, 109)
REGISTERED_RETENTIONS: tuple[float, ...] = (0.25, 0.50, 0.75, 0.90, 0.95)
NATIVE_CONDITION = "native"
SMOKE_PAIRS_PER_SPLIT = 2
FULL_PAIRS_PER_SPLIT = 100
DEFAULT_PAIRS_PER_SPLIT = SMOKE_PAIRS_PER_SPLIT
DEFAULT_RIDGE_ALPHA = 1e-3
ARTIFACT_DIRECTORY = Path("artifacts/experiment_000/readout_trace_a3")
DEFAULT_OUTPUT_PATH = ARTIFACT_DIRECTORY / "report.json"
DEFAULT_SMOKE_REPORT_PATH = ARTIFACT_DIRECTORY / "smoke_pairs_2.json"
FREEZE_RECORD_PATH = ARTIFACT_DIRECTORY / "FREEZE_RECORD.json"
PRE_FREEZE_VERIFICATION_PATH = ARTIFACT_DIRECTORY / "PRE_FREEZE_VERIFICATION.json"
PHASE_SEQUENCE_PATH = ARTIFACT_DIRECTORY / "PHASE_SEQUENCE.json"
SMOKE_VERIFICATION_PATH = ARTIFACT_DIRECTORY / "SMOKE_VERIFICATION.json"
PRIMARY_REPORT_PATH = ARTIFACT_DIRECTORY / "custom_seeds_105_109_pairs_100.json"
RERUN_REPORT_PATH = (
    ARTIFACT_DIRECTORY / "custom_seeds_105_109_pairs_100_rerun.json"
)
DETERMINISM_VERIFICATION_PATH = ARTIFACT_DIRECTORY / "DETERMINISM_VERIFICATION.json"
ORDERED_CPU_WORKERS = 5
BLANK_TAIL_TICKS = 256
BLANK_TAIL_FINAL_WINDOW_TICKS = 64
NOISE_EVENTS = 8
COMPACT_EVIDENCE_VERSION = "readout-trace-compact-evidence-v1"
MAX_SMOKE_REPORT_BYTES = 8 * 1024 * 1024
MAX_FULL_REPORT_BYTES = 64 * 1024 * 1024

_PHASE_ORDER = (
    "tests",
    "freeze",
    "persisted_smoke_and_sidecar",
    "machine_smoke_verification",
    "full",
    "rerun",
    "determinism_verification",
)

_PHASE_ARTIFACT_PATHS: dict[str, tuple[Path, ...]] = {
    "tests": (PRE_FREEZE_VERIFICATION_PATH,),
    "freeze": (FREEZE_RECORD_PATH,),
    "persisted_smoke_and_sidecar": (
        DEFAULT_SMOKE_REPORT_PATH,
        DEFAULT_SMOKE_REPORT_PATH.with_suffix(".sha256"),
    ),
    "machine_smoke_verification": (SMOKE_VERIFICATION_PATH,),
    "full": (PRIMARY_REPORT_PATH, PRIMARY_REPORT_PATH.with_suffix(".sha256")),
    "rerun": (RERUN_REPORT_PATH, RERUN_REPORT_PATH.with_suffix(".sha256")),
    "determinism_verification": (DETERMINISM_VERIFICATION_PATH,),
}

_PROCESS_INSTANCE_TOKEN = hashlib.sha256(os.urandom(32)).hexdigest()


def _single_thread_blas_state() -> list[dict[str, Any]]:
    """Return stable BLAS identity and fail unless every loaded pool is single-threaded."""

    state = sorted(
        (
            {
                "user_api": str(info.get("user_api")),
                "internal_api": str(info.get("internal_api")),
                "prefix": str(info.get("prefix")),
                "version": str(info.get("version")),
                "num_threads": int(info.get("num_threads", -1)),
            }
            for info in threadpool_info()
            if info.get("user_api") == "blas"
        ),
        key=lambda row: (row["internal_api"], row["prefix"], row["version"]),
    )
    if not state or any(row["num_threads"] != 1 for row in state):
        raise RuntimeError(f"A3 requires loaded BLAS pools to use one thread: {state}")
    return state

FROZEN_RECURRENT_SOURCE_SHA256 = (
    "b56216abd19f5d3aaf935b6e58831c54ccf078ce4e4c7ef80e006b04ca6231b3"
)
FROZEN_CREDIT_METHOD_BUNDLE_SHA256 = (
    "38378b7dc7d0b69644c9bd790a1f09d2ebfb9ec642da5136a2475f89f260607d"
)
FROZEN_CCF_DOCUMENT_SHA256 = (
    "24907da979acef2ce95b6bb50ca52461d7b9d8b3bb6154fe871ae9000d7b7e9b"
)
FROZEN_PROTOCOL_SHA256 = (
    "cb023552d50d568c3fe1e44c40038b064c275b513785c26d8693efa21c666963"
)
FROZEN_CONFIG_SHA256 = (
    "f1ca8047c7f5d559a4b6ac3888278d867826227917436af6f5873b6f4cfe0282"
)

FORBIDDEN_SEEDS: dict[str, tuple[int, ...]] = {
    "all_prior_0_104": tuple(range(105)),
    "confirmatory_1000_1019": tuple(range(1000, 1020)),
}

_TRACE_TOUCH_FIELDS = (
    "local_trace_read_touches",
    "local_trace_decay_touches",
    "local_trace_write_touches",
    "local_trace_reset_touches",
    "local_trace_observation_touches",
)

_CONDITION_INVARIANT_KEYS = frozenset(
    {
        "native_control_is_literal_recurrent_event_graph",
        "topology_unchanged",
        "topology_matches_native",
        "structural_mask_unchanged",
        "structural_mask_matches_native",
        "weights_unchanged_during_probe",
        "framed_weights_unchanged_during_probe",
        "weights_match_native",
        "recurrent_weights_match_native",
        "input_weights_match_native",
        "output_weights_match_native",
        "spectral_radius_matches_native",
        "operator_norm_matches_native",
        "retention_immutable",
        "trace_bound_ok",
        "trace_resets_valid",
        "trace_reset_count_matches_episodes",
        "reset_touches_exact",
        "candidate_trace_touches_consistent",
        "native_zero_trace_touches",
        "feature_edges_match_native",
        "all_feature_reconstructions_pass",
        "all_output_reconstructions_pass",
        "all_frozen_snapshots_equal_native",
        "credit_event_touches_zero",
        "credit_edge_touches_zero",
        "weight_write_touches_zero",
        "credit_packets_zero",
        "nonfinite_values_zero",
        "all_weights_finite",
        "all_weights_within_clip",
        "recurrent_activity_equals_native",
    }
)


# ----------------------------------------------------------------------
# Argument and provenance guards


def condition_name(retention: float | None) -> str:
    if retention is None:
        return NATIVE_CONDITION
    return f"rho_{retention:.2f}".replace(".", "_")


def _validate_arguments(
    *,
    seeds: Sequence[int],
    retentions: Sequence[float],
    pairs_per_split: int,
    ridge_alpha: float,
    workers: int,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if any(isinstance(value, bool) or type(value) is not int for value in seeds):
        raise TypeError("seeds must be exact built-in integers")
    resolved_seeds = tuple(seeds)
    if resolved_seeds != REGISTERED_SEEDS:
        raise ValueError(
            "the readout-trace experiment requires the exact ordered seed list "
            f"{REGISTERED_SEEDS}"
        )
    for seed in resolved_seeds:
        require_seed_role(seed, SeedRole.CUSTOM)
        for name, block in FORBIDDEN_SEEDS.items():
            if seed in block:
                raise ValueError(f"seed {seed} is reserved ({name})")

    for raw in retentions:
        if isinstance(raw, bool):
            raise TypeError("retentions must be finite registered numbers")
    resolved_retentions = tuple(float(value) for value in retentions)
    if not all(math.isfinite(value) for value in resolved_retentions):
        raise ValueError("retentions must be finite")
    if resolved_retentions != REGISTERED_RETENTIONS:
        raise ValueError(
            "the readout-trace experiment requires the exact ordered retention "
            f"list {REGISTERED_RETENTIONS}"
        )

    if (
        isinstance(pairs_per_split, bool)
        or not isinstance(pairs_per_split, int)
        or pairs_per_split not in {SMOKE_PAIRS_PER_SPLIT, FULL_PAIRS_PER_SPLIT}
    ):
        raise ValueError(
            "pairs_per_split must be the frozen smoke value 2 or full value 100"
        )
    if not math.isfinite(ridge_alpha) or ridge_alpha != DEFAULT_RIDGE_ALPHA:
        raise ValueError(f"ridge_alpha must equal the frozen value {DEFAULT_RIDGE_ALPHA}")
    if isinstance(workers, bool) or type(workers) is not int or workers != 5:
        raise ValueError("official A3 execution requires exactly five CPU workers")
    return resolved_seeds, resolved_retentions


def frozen_source_integrity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    recurrent_path = root / "src/adaptive_learning_substrate/recurrent.py"
    ccf_path = root / "docs/CCF_V0.md"
    protocol_path = root / "docs/EXPERIMENT_000_READOUT_TRACE_A3_PROTOCOL.md"
    config_path = root / "configs/experiment_000_readout_trace_a3.toml"
    recurrent_hash = _file_sha256(recurrent_path)
    credit_hash = _credit_method_bundle_sha256(recurrent_path)
    ccf_hash = _file_sha256(ccf_path)
    protocol_hash = _file_sha256(protocol_path)
    config_hash = _file_sha256(config_path)
    return {
        "recurrent_source_sha256": recurrent_hash,
        "expected_recurrent_source_sha256": FROZEN_RECURRENT_SOURCE_SHA256,
        "recurrent_source_matches": recurrent_hash == FROZEN_RECURRENT_SOURCE_SHA256,
        "credit_method_bundle_sha256": credit_hash,
        "expected_credit_method_bundle_sha256": FROZEN_CREDIT_METHOD_BUNDLE_SHA256,
        "credit_method_bundle_matches": credit_hash
        == FROZEN_CREDIT_METHOD_BUNDLE_SHA256,
        "ccf_document_sha256": ccf_hash,
        "expected_ccf_document_sha256": FROZEN_CCF_DOCUMENT_SHA256,
        "ccf_document_matches": ccf_hash == FROZEN_CCF_DOCUMENT_SHA256,
        "protocol_sha256": protocol_hash,
        "expected_protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "protocol_matches": protocol_hash == FROZEN_PROTOCOL_SHA256,
        "config_sha256": config_hash,
        "expected_config_sha256": FROZEN_CONFIG_SHA256,
        "config_matches": config_hash == FROZEN_CONFIG_SHA256,
    }


def _source_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "pyproject.toml",
        "src/adaptive_learning_substrate/__init__.py",
        "docs/EXPERIMENT_000_READOUT_TRACE_A3_PROTOCOL.md",
        "configs/experiment_000_readout_trace_a3.toml",
        "src/adaptive_learning_substrate/experiment_000_readout_trace_a3.py",
        "src/adaptive_learning_substrate/readout_trace_recurrent.py",
        "src/adaptive_learning_substrate/experiment_000_slow_state.py",
        "src/adaptive_learning_substrate/experiment_000_memory_probe.py",
        "src/adaptive_learning_substrate/experiment_000.py",
        "src/adaptive_learning_substrate/experiment000_data.py",
        "src/adaptive_learning_substrate/recurrent.py",
        "src/adaptive_learning_substrate/events.py",
        "tests/test_experiment_000_readout_trace_a3.py",
        "docs/CCF_V0.md",
    )
    files: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"readout-trace provenance file missing: {relative}")
        files[relative] = {"sha256": _file_sha256(path), "bytes": path.stat().st_size}
    return {"files": files, "bundle_sha256": _sha256_json(files)}


def _project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path(__file__).resolve().parents[2] / value


def _record_self_hash(record: Mapping[str, Any], field: str) -> str:
    payload = dict(record)
    payload.pop(field, None)
    return _sha256_json(payload)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = _project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target


def _serialized_json_size(payload: Mapping[str, Any]) -> int:
    """Return the exact UTF-8 byte count produced by :func:`_write_json`."""

    return len(
        (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    )


def _write_report_and_sidecar(
    path: str | Path, payload: Mapping[str, Any]
) -> tuple[Path, Path]:
    target = _write_json(path, payload)
    report_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    sidecar = target.with_suffix(".sha256")
    _write_json(
        sidecar,
        {
            "algorithm": "sha256",
            "report_file": target.name,
            "report_sha256": report_hash,
        },
    )
    return target, sidecar


def _artifact_identity(path: str | Path) -> dict[str, Any]:
    target = _project_path(path)
    if not target.is_file():
        raise RuntimeError(f"required phase artifact is absent: {target.name}")
    return {
        "path": str(target.relative_to(Path(__file__).resolve().parents[2])).replace(
            "\\", "/"
        ),
        "sha256": _file_sha256(target),
        "bytes": target.stat().st_size,
    }


def _validate_pre_freeze_verification(
    current_manifest: Mapping[str, Any],
    path: str | Path = PRE_FREEZE_VERIFICATION_PATH,
) -> dict[str, Any]:
    target = _project_path(path)
    if not target.is_file():
        raise RuntimeError("A3 pre-freeze test/lint verification is absent")
    record = json.loads(target.read_text(encoding="utf-8"))
    checks = record.get("checks")
    if (
        record.get("schema_version")
        != "experiment-000-readout-trace-a3-pre-freeze-v1"
        or record.get("status") != "PASS"
        or record.get("all_exit_status_zero") is not True
        or record.get("tested_source_manifest_sha256")
        != current_manifest.get("bundle_sha256")
        or not isinstance(checks, list)
        or [row.get("name") for row in checks]
        != ["focused_analytic_tests", "ruff", "complete_pytest"]
        or any(
            not isinstance(row.get("command"), str)
            or not isinstance(row.get("input"), str)
            or not isinstance(row.get("literal_output"), str)
            or row.get("exit_status") != 0
            for row in checks
        )
        or record.get("verification_payload_sha256")
        != _record_self_hash(record, "verification_payload_sha256")
    ):
        raise RuntimeError("A3 pre-freeze test/lint verification is invalid")
    return {
        "record": record,
        "file_sha256": _file_sha256(target),
        "path": str(target.relative_to(Path(__file__).resolve().parents[2])).replace(
            "\\", "/"
        ),
    }


def create_pre_freeze_verification(
    checks: Sequence[Mapping[str, Any]],
    output_path: str | Path = PRE_FREEZE_VERIFICATION_PATH,
) -> dict[str, Any]:
    """Persist the exact successful test/lint evidence consumed by the freeze."""

    target = _project_path(output_path)
    if target.resolve() != _project_path(PRE_FREEZE_VERIFICATION_PATH).resolve():
        raise ValueError("A3 pre-freeze verification must use its registered path")
    if _project_path(FREEZE_RECORD_PATH).exists() or _project_path(
        PHASE_SEQUENCE_PATH
    ).exists():
        raise RuntimeError(
            "A3 pre-freeze verification is immutable after freeze/phase creation"
        )
    normalized = [dict(row) for row in checks]
    if [row.get("name") for row in normalized] != [
        "focused_analytic_tests",
        "ruff",
        "complete_pytest",
    ] or any(
        not isinstance(row.get("command"), str)
        or not isinstance(row.get("input"), str)
        or not isinstance(row.get("literal_output"), str)
        or row.get("exit_status") != 0
        for row in normalized
    ):
        raise ValueError("A3 pre-freeze checks are incomplete or unsuccessful")
    manifest = _source_provenance()
    record: dict[str, Any] = {
        "schema_version": "experiment-000-readout-trace-a3-pre-freeze-v1",
        "status": "PASS",
        "all_exit_status_zero": True,
        "tested_source_manifest_sha256": manifest["bundle_sha256"],
        "checks": normalized,
    }
    record["verification_payload_sha256"] = _record_self_hash(
        record, "verification_payload_sha256"
    )
    _write_json(target, record)
    return record


def _validate_phase_sequence(expected_names: Sequence[str]) -> dict[str, Any]:
    target = _project_path(PHASE_SEQUENCE_PATH)
    if not target.is_file():
        raise RuntimeError("A3 phase-sequence record is absent")
    record = json.loads(target.read_text(encoding="utf-8"))
    phases = record.get("phases")
    if (
        record.get("schema_version") != "experiment-000-readout-trace-a3-phases-v1"
        or record.get("protocol_version") != PROTOCOL_VERSION
        or record.get("phase_order") != list(_PHASE_ORDER)
        or not isinstance(phases, list)
        or [phase.get("name") for phase in phases] != list(expected_names)
        or tuple(expected_names) != _PHASE_ORDER[: len(expected_names)]
        or record.get("phase_sequence_payload_sha256")
        != _record_self_hash(record, "phase_sequence_payload_sha256")
    ):
        raise RuntimeError("A3 phase sequence is invalid or out of order")
    root = Path(__file__).resolve().parents[2]
    for phase in phases:
        artifacts = phase.get("artifacts")
        expected_paths = [
            str(path).replace("\\", "/")
            for path in _PHASE_ARTIFACT_PATHS[str(phase["name"])]
        ]
        if (
            not isinstance(artifacts, list)
            or not artifacts
            or [identity.get("path") for identity in artifacts] != expected_paths
        ):
            raise RuntimeError("A3 phase has no bound artifacts")
        for identity in artifacts:
            artifact = root / str(identity.get("path"))
            if (
                not artifact.is_file()
                or identity.get("sha256") != _file_sha256(artifact)
                or identity.get("bytes") != artifact.stat().st_size
            ):
                raise RuntimeError("A3 phase artifact identity no longer matches")
    return {"record": record, "file_sha256": _file_sha256(target)}


def _validate_phase_prefix(required_names: Sequence[str]) -> dict[str, Any]:
    """Validate an append-only phase ledger that contains at least this prefix."""

    target = _project_path(PHASE_SEQUENCE_PATH)
    if not target.is_file():
        raise RuntimeError("A3 phase-sequence record is absent")
    record = json.loads(target.read_text(encoding="utf-8"))
    phases = record.get("phases")
    if not isinstance(phases, list):
        raise TypeError("A3 phase sequence has no phase list")
    names = [phase.get("name") for phase in phases]
    if (
        len(names) < len(required_names)
        or names != list(_PHASE_ORDER[: len(names)])
        or names[: len(required_names)] != list(required_names)
    ):
        raise RuntimeError("A3 required phase prefix is absent or out of order")
    return _validate_phase_sequence(names)


def _record_phase(name: str, artifacts: Sequence[str | Path]) -> dict[str, Any]:
    if name not in _PHASE_ORDER:
        raise ValueError(f"unregistered A3 phase: {name}")
    index = _PHASE_ORDER.index(name)
    if index == 0:
        if _project_path(PHASE_SEQUENCE_PATH).exists():
            raise RuntimeError("A3 phase sequence already exists before freeze")
        phases: list[dict[str, Any]] = []
    else:
        phases = copy.deepcopy(
            _validate_phase_sequence(_PHASE_ORDER[:index])["record"]["phases"]
        )
    identities = [_artifact_identity(path) for path in artifacts]
    expected_paths = [
        str(path).replace("\\", "/") for path in _PHASE_ARTIFACT_PATHS[name]
    ]
    if [identity["path"] for identity in identities] != expected_paths:
        raise RuntimeError("A3 phase did not receive its registered artifact set")
    phases.append({"name": name, "artifacts": identities})
    record: dict[str, Any] = {
        "schema_version": "experiment-000-readout-trace-a3-phases-v1",
        "protocol_version": PROTOCOL_VERSION,
        "phase_order": list(_PHASE_ORDER),
        "phases": phases,
    }
    record["phase_sequence_payload_sha256"] = _record_self_hash(
        record, "phase_sequence_payload_sha256"
    )
    _write_json(PHASE_SEQUENCE_PATH, record)
    return record


def create_freeze_record(
    output_path: str | Path = FREEZE_RECORD_PATH,
) -> dict[str, Any]:
    """Create the A3 source/environment freeze after the fast tests pass."""

    target = _project_path(output_path)
    if target.resolve() != _project_path(FREEZE_RECORD_PATH).resolve():
        raise ValueError("A3 freeze must use the registered freeze-record path")
    if target.exists() or _project_path(PHASE_SEQUENCE_PATH).exists():
        raise RuntimeError("A3 freeze/phase artifacts already exist and are immutable")
    with threadpool_limits(limits=1, user_api="blas"):
        blas_state = _single_thread_blas_state()
    integrity = frozen_source_integrity()
    required = (
        "recurrent_source_matches",
        "credit_method_bundle_matches",
        "ccf_document_matches",
        "protocol_matches",
        "config_matches",
    )
    if not all(integrity[name] for name in required):
        raise RuntimeError("A3 frozen binding files do not match registered hashes")
    manifest = _source_provenance()
    pre_freeze = _validate_pre_freeze_verification(manifest)
    utc_now = datetime.now(UTC)
    local_now = utc_now.astimezone()
    record: dict[str, Any] = {
        "schema_version": "experiment-000-readout-trace-a3-freeze-v1",
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "created_utc": utc_now.isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        ),
        "created_local": local_now.isoformat(timespec="microseconds"),
        "local_timezone": str(local_now.tzinfo),
        "execution_enabled": True,
        "workers": ORDERED_CPU_WORKERS,
        "backend": {"engine": "numpy_cpu", "float_type": "float64", "gpu": False},
        "blas_runtime": blas_state,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "environment": {
            "implementation": platform.python_implementation(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "threadpoolctl": threadpoolctl.__version__,
            "platform": platform.platform(),
        },
        "source_manifest": manifest,
        "source_manifest_sha256": manifest["bundle_sha256"],
        "pre_freeze_verification": {
            "path": pre_freeze["path"],
            "file_sha256": pre_freeze["file_sha256"],
            "verification_payload_sha256": pre_freeze["record"][
                "verification_payload_sha256"
            ],
        },
        "prior_v1a": {
            "classification": "INVALID_PROCEDURE_NONSELECTING",
            "seeds": [65, 66, 67, 68, 69],
            "observed_deterministic_payload_sha256": (
                "ab5d009ea3afdb8dced17919b255da01e7f8c309af9b2e371eb3b2963c5053cb"
            ),
            "observed_report_file_sha256": (
                "33bdf52cfdac5f30f797a5a75b034a8734f60729e42336fc91eff44ebe3b2d5a"
            ),
            "persisted_rerun_present": False,
            "persisted_determinism_record_present": False,
        },
        "predecessor_v1a2": {
            "classification": "PRESERVED_UNINSPECTED_RESOURCE_SUPERSEDED",
            "seeds": [70, 71, 72, 73, 74],
            "scientific_metrics_imported": False,
            "scientific_metrics_inspected_before_a3_freeze": False,
            "protocol_sha256": (
                "9623c23efb586dc89351e2c19887cc25886db42f42a600247feb0c38dcb11484"
            ),
            "config_sha256": (
                "45e2f1450d57a62a9a62a8dd6ce8770da8751de4533f8338cd11acd95cbb3327"
            ),
            "runner_sha256": (
                "22d7b9083ed057641bce96870a75a4274c99fc6e6b9ce0a7f5aa971e9b5b22b6"
            ),
        },
        "phase_order": list(_PHASE_ORDER),
        "artifact_paths": {
            "pre_freeze_verification": str(PRE_FREEZE_VERIFICATION_PATH).replace(
                "\\", "/"
            ),
            "freeze": str(FREEZE_RECORD_PATH).replace("\\", "/"),
            "phase_sequence": str(PHASE_SEQUENCE_PATH).replace("\\", "/"),
            "smoke": str(DEFAULT_SMOKE_REPORT_PATH).replace("\\", "/"),
            "smoke_verification": str(SMOKE_VERIFICATION_PATH).replace("\\", "/"),
            "primary": str(PRIMARY_REPORT_PATH).replace("\\", "/"),
            "rerun": str(RERUN_REPORT_PATH).replace("\\", "/"),
            "determinism": str(DETERMINISM_VERIFICATION_PATH).replace("\\", "/"),
        },
    }
    record["freeze_payload_sha256"] = _record_self_hash(
        record, "freeze_payload_sha256"
    )
    _write_json(target, record)
    _record_phase("tests", (PRE_FREEZE_VERIFICATION_PATH,))
    _record_phase("freeze", (FREEZE_RECORD_PATH,))
    if _source_provenance() != manifest:
        raise RuntimeError("A3 source manifest changed during freeze persistence")
    _validate_freeze_record(current_manifest=manifest)
    return record


def _validate_freeze_record(
    path: str | Path = FREEZE_RECORD_PATH,
    *,
    current_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target = _project_path(path)
    if not target.is_file():
        raise RuntimeError("A3 freeze record is absent")
    record = json.loads(target.read_text(encoding="utf-8"))
    expected_hash = record.get("freeze_payload_sha256")
    if not isinstance(expected_hash, str) or expected_hash != _record_self_hash(
        record, "freeze_payload_sha256"
    ):
        raise RuntimeError("A3 freeze record self-hash is invalid")
    current_environment = {
        "implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "threadpoolctl": threadpoolctl.__version__,
        "platform": platform.platform(),
    }
    expected_artifact_paths = {
        "pre_freeze_verification": str(PRE_FREEZE_VERIFICATION_PATH).replace(
            "\\", "/"
        ),
        "freeze": str(FREEZE_RECORD_PATH).replace("\\", "/"),
        "phase_sequence": str(PHASE_SEQUENCE_PATH).replace("\\", "/"),
        "smoke": str(DEFAULT_SMOKE_REPORT_PATH).replace("\\", "/"),
        "smoke_verification": str(SMOKE_VERIFICATION_PATH).replace("\\", "/"),
        "primary": str(PRIMARY_REPORT_PATH).replace("\\", "/"),
        "rerun": str(RERUN_REPORT_PATH).replace("\\", "/"),
        "determinism": str(DETERMINISM_VERIFICATION_PATH).replace("\\", "/"),
    }
    with threadpool_limits(limits=1, user_api="blas"):
        current_blas_state = _single_thread_blas_state()
    if (
        record.get("schema_version")
        != "experiment-000-readout-trace-a3-freeze-v1"
        or record.get("protocol_version") != PROTOCOL_VERSION
        or record.get("protocol_sha256") != FROZEN_PROTOCOL_SHA256
        or record.get("config_sha256") != FROZEN_CONFIG_SHA256
        or record.get("execution_enabled") is not True
        or record.get("workers") != ORDERED_CPU_WORKERS
        or record.get("backend")
        != {"engine": "numpy_cpu", "float_type": "float64", "gpu": False}
        or record.get("blas_runtime") != current_blas_state
        or record.get("thread_environment")
        != {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
        or record.get("environment") != current_environment
        or record.get("phase_order") != list(_PHASE_ORDER)
        or record.get("artifact_paths") != expected_artifact_paths
        or record.get("prior_v1a", {}).get("classification")
        != "INVALID_PROCEDURE_NONSELECTING"
        or record.get("prior_v1a", {}).get("observed_report_file_sha256")
        != "33bdf52cfdac5f30f797a5a75b034a8734f60729e42336fc91eff44ebe3b2d5a"
        or record.get("prior_v1a", {}).get("observed_deterministic_payload_sha256")
        != "ab5d009ea3afdb8dced17919b255da01e7f8c309af9b2e371eb3b2963c5053cb"
        or record.get("predecessor_v1a2")
        != {
            "classification": "PRESERVED_UNINSPECTED_RESOURCE_SUPERSEDED",
            "seeds": [70, 71, 72, 73, 74],
            "scientific_metrics_imported": False,
            "scientific_metrics_inspected_before_a3_freeze": False,
            "protocol_sha256": "9623c23efb586dc89351e2c19887cc25886db42f42a600247feb0c38dcb11484",
            "config_sha256": "45e2f1450d57a62a9a62a8dd6ce8770da8751de4533f8338cd11acd95cbb3327",
            "runner_sha256": "22d7b9083ed057641bce96870a75a4274c99fc6e6b9ce0a7f5aa971e9b5b22b6",
        }
    ):
        raise RuntimeError("A3 freeze record metadata is invalid")
    try:
        parsed_utc = datetime.fromisoformat(str(record["created_utc"]))
        parsed_local = datetime.fromisoformat(str(record["created_local"]))
    except (KeyError, ValueError) as exc:
        raise RuntimeError("A3 freeze timestamps are invalid") from exc
    if parsed_utc.utcoffset() != UTC.utcoffset(parsed_utc):
        raise RuntimeError("A3 freeze UTC timestamp is not UTC")
    if parsed_local.utcoffset() is None or not record.get("local_timezone"):
        raise RuntimeError("A3 freeze local timezone is absent")
    manifest = dict(current_manifest or _source_provenance())
    if (
        record.get("source_manifest") != manifest
        or record.get("source_manifest_sha256") != manifest.get("bundle_sha256")
    ):
        raise RuntimeError("current A3 source manifest differs from the freeze")
    pre_freeze = _validate_pre_freeze_verification(manifest)
    recorded_pre_freeze = record.get("pre_freeze_verification", {})
    if (
        recorded_pre_freeze.get("file_sha256") != pre_freeze["file_sha256"]
        or recorded_pre_freeze.get("verification_payload_sha256")
        != pre_freeze["record"]["verification_payload_sha256"]
    ):
        raise RuntimeError("A3 freeze does not bind the passing pre-freeze checks")
    _validate_phase_prefix(_PHASE_ORDER[:2])
    return {
        "record": record,
        "file_sha256": _file_sha256(target),
        "path": str(FREEZE_RECORD_PATH).replace("\\", "/"),
    }


def _read_report_with_valid_sidecar(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    target = _project_path(path)
    sidecar_path = target.with_suffix(".sha256")
    if not target.is_file() or not sidecar_path.is_file():
        raise RuntimeError(f"report or SHA-256 sidecar is absent: {target.name}")
    report_bytes = target.read_bytes()
    report_hash = hashlib.sha256(report_bytes).hexdigest()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if (
        sidecar.get("algorithm") != "sha256"
        or sidecar.get("report_file") != target.name
        or sidecar.get("report_sha256") != report_hash
    ):
        raise RuntimeError(f"report sidecar is invalid: {target.name}")
    report = json.loads(report_bytes)
    if not isinstance(report, dict):
        raise TypeError("report root must be an object")
    return report, {
        "report_file_sha256": report_hash,
        "sidecar_file_sha256": _file_sha256(sidecar_path),
    }


def _recompute_report_payload_hash(report: Mapping[str, Any]) -> str:
    return _sha256_json(_deterministic_payload_object(report))


def _deterministic_payload_object(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical report object covered by deterministic comparison."""

    payload = copy.deepcopy(dict(report))
    payload.pop("nondeterministic_provenance", None)
    payload.pop("deterministic_payload_sha256", None)
    return payload


# ----------------------------------------------------------------------
# Forward collection


def _counterfactual_episode(
    base: Experiment000Episode, *, cue: int, condition: str
) -> Experiment000Episode:
    del condition  # Conditions must share the same administrative event identities.
    return replace(
        base,
        episode_id=f"{base.episode_id}-readout-trace-cue-{cue}",
        cue=cue,
        target=cue,
    )


def _unit_events_sha256(graph: RecurrentEventGraph) -> str:
    return _sha256_json([asdict(event) for event in graph.unit_events])


def _activation_dynamics_sha256(
    records: Sequence[Any],
) -> str:
    return _sha256_json(
        [
            {
                "node": record.node,
                "tick": record.tick,
                "activation": record.activation,
                "emitted": record.emitted,
            }
            for record in records
        ]
    )


def _legacy_ledger_delta(
    after: Mapping[str, int], before: Mapping[str, int]
) -> dict[str, int]:
    gauges = {"peak_retained_event_records", "peak_pending_messages"}
    return {
        key: int(after[key] - before[key])
        for key in sorted(set(after).intersection(before).difference(gauges))
    }


def _zero_trace_ledger() -> dict[str, int]:
    return {
        "hidden_activation_evaluations": 0,
        **{field: 0 for field in _TRACE_TOUCH_FIELDS},
    }


def _trace_ledger(graph: RecurrentEventGraph) -> dict[str, int]:
    if isinstance(graph, ReadoutTraceRecurrentEventGraph):
        return graph.trace_ledger
    return _zero_trace_ledger()


def _candidate_trace_readout(
    graph: ReadoutTraceRecurrentEventGraph,
    query_tick: int,
    feature_edge_ids: tuple[str, ...],
) -> tuple[list[tuple[str, float]], np.ndarray, float, float]:
    """Read one 64-D snapshot and derive the selecting 16-D view from it."""

    effective = graph.effective_trace(query_tick)
    source_by_edge = {
        edge.edge_id: edge.source
        for edge in graph.edges
        if edge.kind == "output" and edge.destination == graph.output_node
    }
    if tuple(sorted(source_by_edge)) != feature_edge_ids:
        raise RuntimeError("candidate output-feature edge identity changed")
    features = [
        (edge_id, float(effective[source_by_edge[edge_id]]))
        for edge_id in feature_edge_ids
    ]
    full_trace = np.asarray(
        [effective[node] for node in graph.hidden_nodes], dtype=np.float64
    )
    weights = graph.weights
    preactivation = math.fsum(
        weights[edge_id] * value for edge_id, value in features
    )
    return features, full_trace, preactivation, math.tanh(preactivation)


def _normalized_recurrent_dynamics(
    graph: RecurrentEventGraph,
    query: QueryResult,
    activation_records: Sequence[Any],
    ledger_before: Mapping[str, int],
    ledger_after: Mapping[str, int],
) -> dict[str, Any]:
    """Serialize every native recurrent effect while excluding episode-name text."""

    events = graph.unit_events
    local_id = {event.event_id: index for index, event in enumerate(events)}

    def event_index(event_id: str) -> int:
        if event_id not in local_id:
            raise RuntimeError("recurrent dynamics references an unknown local event")
        return local_id[event_id]

    event_rows: list[dict[str, Any]] = []
    bit_values: list[float] = []
    for index, event in enumerate(events):
        traces: list[dict[str, Any]] = []
        bit_values.extend((float(event.preactivation), float(event.activation)))
        for trace in event.edge_traces:
            values = (
                float(trace.message_value),
                float(trace.omission_effect),
                float(trace.weight_secant),
                float(trace.source_secant),
            )
            bit_values.extend(values)
            traces.append(
                {
                    "edge_id": trace.edge_id,
                    "parent_event_index": event_index(trace.parent_event_id),
                    "message_value": values[0],
                    "omission_effect": values[1],
                    "weight_secant": values[2],
                    "source_secant": values[3],
                    "created_step": trace.created_step,
                }
            )
        event_rows.append(
            {
                "event_index": index,
                "node": event.node,
                "step": event.step,
                "preactivation": float(event.preactivation),
                "activation": float(event.activation),
                "forced_output": event.forced_output,
                "edge_traces": traces,
            }
        )
    pending_rows: list[dict[str, Any]] = []
    for arrival_tick in sorted(graph._pending):
        for message in graph._pending[arrival_tick]:
            bit_values.append(float(message.value))
            pending_rows.append(
                {
                    "arrival_tick": arrival_tick,
                    "edge_id": message.edge_id,
                    "parent_event_index": event_index(message.parent_event_id),
                    "value": float(message.value),
                }
            )
    activation_rows = [
        {
            "node": record.node,
            "tick": record.tick,
            "activation": float(record.activation),
            "emitted": bool(record.emitted),
        }
        for record in activation_records
    ]
    bit_values.extend(row["activation"] for row in activation_rows)
    final_activations = [
        [node, float(graph._activations[node])]
        for node in (*graph.input_nodes, *graph.hidden_nodes, graph.output_node)
    ]
    final_last_emitted = [
        [node, float(graph._last_emitted[node])]
        for node in (*graph.input_nodes, *graph.hidden_nodes, graph.output_node)
    ]
    bit_values.extend(value for _, value in final_activations)
    bit_values.extend(value for _, value in final_last_emitted)
    return {
        "tick": query.tick,
        "query_event_index": event_index(query.event_id),
        "query_activation": float(query.activation),
        "query_prediction": query.prediction,
        "events": event_rows,
        "hidden_activation_evaluations": activation_rows,
        "pending_messages": pending_rows,
        "final_activations": final_activations,
        "final_last_emitted": final_last_emitted,
        "event_sequence": [
            [node, int(graph._event_sequence[node])]
            for node in (*graph.input_nodes, *graph.hidden_nodes, graph.output_node)
        ],
        "ledger_delta": {
            key: int(ledger_after[key] - ledger_before[key]) for key in sorted(ledger_after)
        },
        "float64_bundle_sha256": _array_bundle_sha256(bit_values),
    }


def _validate_recurrent_dynamics_payload(
    payload: Mapping[str, Any], graph: RecurrentEventGraph
) -> None:
    """Validate the typed, internally linked recurrent episode transcript."""

    top_level_keys = {
        "tick",
        "query_event_index",
        "query_activation",
        "query_prediction",
        "events",
        "hidden_activation_evaluations",
        "pending_messages",
        "final_activations",
        "final_last_emitted",
        "event_sequence",
        "ledger_delta",
        "float64_bundle_sha256",
    }
    event_keys = {
        "event_index",
        "node",
        "step",
        "preactivation",
        "activation",
        "forced_output",
        "edge_traces",
    }
    trace_keys = {
        "edge_id",
        "parent_event_index",
        "message_value",
        "omission_effect",
        "weight_secant",
        "source_secant",
        "created_step",
    }

    def require_object(value: Any, keys: set[str], name: str) -> dict[str, Any]:
        if type(value) is not dict or set(value) != keys:
            raise RuntimeError(f"recurrent dynamics {name} schema is invalid")
        return value

    def require_list(value: Any, name: str) -> list[Any]:
        if type(value) is not list:
            raise RuntimeError(f"recurrent dynamics {name} must be a list")
        return value

    def require_int(value: Any, name: str, *, minimum: int = 0) -> int:
        if type(value) is not int or value < minimum:
            raise RuntimeError(f"recurrent dynamics {name} must be an integer >= {minimum}")
        return value

    def require_float(value: Any, name: str) -> float:
        if type(value) is not float or not math.isfinite(value):
            raise RuntimeError(f"recurrent dynamics {name} must be a finite float")
        return value

    def same_float(left: float, right: float) -> bool:
        return struct.pack("<d", left) == struct.pack("<d", right)

    data = require_object(payload, top_level_keys, "payload")
    tick = require_int(data["tick"], "tick")
    query_index = require_int(data["query_event_index"], "query_event_index")
    query_activation = require_float(data["query_activation"], "query_activation")
    query_prediction = require_int(data["query_prediction"], "query_prediction")
    if query_prediction not in (0, 1):
        raise RuntimeError("recurrent dynamics query_prediction must be 0 or 1")

    nodes = (*graph.input_nodes, *graph.hidden_nodes, graph.output_node)
    node_set = set(nodes)
    input_set = set(graph.input_nodes)
    hidden_set = set(graph.hidden_nodes)
    edges = graph.edges_by_id
    weights = graph.weights
    outgoing = {node: [] for node in nodes}
    for edge in graph.edges:
        outgoing[edge.source].append(edge)
    events = require_list(data["events"], "events")
    if not events:
        raise RuntimeError("recurrent dynamics events must be nonempty")

    event_values: list[float] = []
    event_by_node_tick: dict[tuple[int, str], dict[str, Any]] = {}
    previous_order: tuple[int, int] | None = None
    process_order = {
        node: index
        for index, node in enumerate(
            (*graph.hidden_nodes, graph.output_node, *graph.input_nodes)
        )
    }
    for index, raw_event in enumerate(events):
        event = require_object(raw_event, event_keys, f"events[{index}]")
        if require_int(event["event_index"], f"events[{index}].event_index") != index:
            raise RuntimeError("recurrent dynamics event_index sequence is invalid")
        node = event["node"]
        if type(node) is not str or node not in node_set:
            raise RuntimeError(f"recurrent dynamics events[{index}].node is invalid")
        step = require_int(event["step"], f"events[{index}].step")
        if step > tick:
            raise RuntimeError(f"recurrent dynamics events[{index}].step exceeds tick")
        order = (step, process_order[node])
        if previous_order is not None and order <= previous_order:
            raise RuntimeError("recurrent dynamics event ordering is invalid")
        previous_order = order
        key = (step, node)
        if key in event_by_node_tick:
            raise RuntimeError("recurrent dynamics duplicates a node event within one tick")
        event_by_node_tick[key] = event

        preactivation = require_float(
            event["preactivation"], f"events[{index}].preactivation"
        )
        activation = require_float(event["activation"], f"events[{index}].activation")
        event_values.extend((preactivation, activation))
        if type(event["forced_output"]) is not bool:
            raise RuntimeError(
                f"recurrent dynamics events[{index}].forced_output must be a boolean"
            )
        traces = require_list(event["edge_traces"], f"events[{index}].edge_traces")
        if node in input_set:
            if traces or event["forced_output"] or not same_float(preactivation, activation):
                raise RuntimeError("recurrent dynamics source event structure is invalid")
        elif not same_float(activation, math.tanh(preactivation)):
            raise RuntimeError("recurrent dynamics event activation is inconsistent")

        seen_trace_edges: set[str] = set()
        for trace_index, raw_trace in enumerate(traces):
            trace = require_object(
                raw_trace,
                trace_keys,
                f"events[{index}].edge_traces[{trace_index}]",
            )
            edge_id = trace["edge_id"]
            if type(edge_id) is not str or edge_id not in edges or edge_id in seen_trace_edges:
                raise RuntimeError("recurrent dynamics edge trace identity is invalid")
            seen_trace_edges.add(edge_id)
            edge = edges[edge_id]
            parent_index = require_int(
                trace["parent_event_index"],
                f"events[{index}].edge_traces[{trace_index}].parent_event_index",
            )
            if parent_index >= index:
                raise RuntimeError("recurrent dynamics edge trace parent is not ancestral")
            parent = events[parent_index]
            created_step = require_int(
                trace["created_step"],
                f"events[{index}].edge_traces[{trace_index}].created_step",
            )
            if (
                edge.source != parent["node"]
                or edge.destination != node
                or step != parent["step"] + edge.delay_ticks
                or created_step != step
            ):
                raise RuntimeError("recurrent dynamics edge trace linkage is invalid")
            trace_values = [
                require_float(
                    trace[name], f"events[{index}].edge_traces[{trace_index}].{name}"
                )
                for name in (
                    "message_value",
                    "omission_effect",
                    "weight_secant",
                    "source_secant",
                )
            ]
            if not same_float(trace_values[0], parent["activation"]):
                raise RuntimeError("recurrent dynamics edge trace message is inconsistent")
            event_values.extend(trace_values)
        if node not in input_set and not same_float(
            preactivation,
            math.fsum(weights[trace["edge_id"]] * trace["message_value"] for trace in traces),
        ):
            raise RuntimeError("recurrent dynamics event preactivation is inconsistent")

    if query_index >= len(events):
        raise RuntimeError("recurrent dynamics query_event_index is out of range")
    query_event = events[query_index]
    if (
        query_index != len(events) - 1
        or query_event["node"] != graph.output_node
        or query_event["step"] != tick
        or query_event["forced_output"] is not True
        or sum(event["forced_output"] is True for event in events) != 1
        or not same_float(query_activation, query_event["activation"])
    ):
        raise RuntimeError("recurrent dynamics query event is inconsistent")
    expected_prediction = 1 if query_activation >= 0.0 else 0
    if query_prediction != expected_prediction:
        raise RuntimeError("recurrent dynamics query_prediction is inconsistent")

    scheduled: list[tuple[int, str, str, int, float]] = []
    for parent_index, event in enumerate(events):
        for edge in outgoing[event["node"]]:
            scheduled.append(
                (
                    event["step"] + edge.delay_ticks,
                    edge.destination,
                    edge.edge_id,
                    parent_index,
                    event["activation"],
                )
            )
    arrivals: dict[tuple[int, str], list[tuple[int, str, str, int, float]]] = {}
    for message in scheduled:
        if message[0] <= tick:
            arrivals.setdefault((message[0], message[1]), []).append(message)
    for event in events:
        if event["node"] in input_set:
            continue
        messages = sorted(
            arrivals.get((event["step"], event["node"]), ()), key=lambda item: item[2]
        )
        traces = event["edge_traces"]
        if len(traces) != len(messages) or any(
            trace["edge_id"] != message[2]
            or trace["parent_event_index"] != message[3]
            or not same_float(trace["message_value"], message[4])
            for trace, message in zip(traces, messages, strict=True)
        ):
            raise RuntimeError("recurrent dynamics edge trace set is inconsistent")

    evaluation_keys = set(arrivals)
    evaluation_keys.add((tick, graph.output_node))
    internal_order = {
        node: index
        for index, node in enumerate((*graph.hidden_nodes, graph.output_node))
    }
    expected_hidden_keys = {
        evaluation_key for evaluation_key in evaluation_keys if evaluation_key[1] in hidden_set
    }

    activation_rows = require_list(
        data["hidden_activation_evaluations"], "hidden_activation_evaluations"
    )
    activation_values: list[float] = []
    activation_row_keys = {"node", "tick", "activation", "emitted"}
    observed_hidden_rows: dict[tuple[int, str], tuple[float, bool]] = {}
    previous_hidden_order: tuple[int, int] | None = None
    for index, raw_row in enumerate(activation_rows):
        row = require_object(
            raw_row, activation_row_keys, f"hidden_activation_evaluations[{index}]"
        )
        node = row["node"]
        if type(node) is not str or node not in hidden_set:
            raise RuntimeError("recurrent dynamics hidden activation node is invalid")
        row_tick = require_int(
            row["tick"], f"hidden_activation_evaluations[{index}].tick"
        )
        if row_tick > tick:
            raise RuntimeError("recurrent dynamics hidden activation tick exceeds query tick")
        activation = require_float(
            row["activation"], f"hidden_activation_evaluations[{index}].activation"
        )
        emitted = row["emitted"]
        if type(emitted) is not bool:
            raise RuntimeError("recurrent dynamics hidden activation emitted must be a boolean")
        row_key = (row_tick, node)
        row_order = (row_tick, internal_order[node])
        if row_key in observed_hidden_rows or (
            previous_hidden_order is not None and row_order <= previous_hidden_order
        ):
            raise RuntimeError("recurrent dynamics hidden activation ordering is invalid")
        previous_hidden_order = row_order
        observed_hidden_rows[row_key] = (activation, emitted)
        activation_values.append(activation)
    if set(observed_hidden_rows) != expected_hidden_keys:
        raise RuntimeError("recurrent dynamics hidden activation evaluations are inconsistent")
    for key, (activation, emitted) in observed_hidden_rows.items():
        event = event_by_node_tick.get(key)
        if emitted is not (event is not None) or (
            event is not None and not same_float(activation, event["activation"])
        ):
            raise RuntimeError("recurrent dynamics hidden activation emission is inconsistent")

    pending_rows = require_list(data["pending_messages"], "pending_messages")
    pending_values: list[float] = []
    pending_keys = {"arrival_tick", "edge_id", "parent_event_index", "value"}
    observed_pending: list[tuple[int, str, int, bytes]] = []
    for index, raw_message in enumerate(pending_rows):
        message = require_object(raw_message, pending_keys, f"pending_messages[{index}]")
        arrival_tick = require_int(
            message["arrival_tick"], f"pending_messages[{index}].arrival_tick"
        )
        edge_id = message["edge_id"]
        parent_index = require_int(
            message["parent_event_index"],
            f"pending_messages[{index}].parent_event_index",
        )
        value = require_float(message["value"], f"pending_messages[{index}].value")
        if (
            arrival_tick <= tick
            or type(edge_id) is not str
            or edge_id not in edges
            or parent_index >= len(events)
            or edges[edge_id].source != events[parent_index]["node"]
            or arrival_tick != events[parent_index]["step"] + edges[edge_id].delay_ticks
            or not same_float(value, events[parent_index]["activation"])
        ):
            raise RuntimeError("recurrent dynamics pending message linkage is invalid")
        pending_values.append(value)
        observed_pending.append(
            (arrival_tick, edge_id, parent_index, struct.pack("<d", value))
        )
    expected_pending = [
        (
            arrival_tick,
            edge_id,
            parent_index,
            struct.pack("<d", value),
        )
        for arrival_tick, _, edge_id, parent_index, value in scheduled
        if arrival_tick > tick
    ]
    if sorted(observed_pending) != sorted(expected_pending):
        raise RuntimeError("recurrent dynamics pending message set is inconsistent")

    final_values: list[float] = []

    def validate_node_float_rows(raw_rows: Any, name: str) -> dict[str, float]:
        rows = require_list(raw_rows, name)
        if len(rows) != len(nodes):
            raise RuntimeError(f"recurrent dynamics {name} length is invalid")
        result: dict[str, float] = {}
        for index, (row, expected_node) in enumerate(zip(rows, nodes, strict=True)):
            if type(row) is not list or len(row) != 2 or row[0] != expected_node:
                raise RuntimeError(f"recurrent dynamics {name}[{index}] is invalid")
            result[expected_node] = require_float(row[1], f"{name}[{index}].value")
            final_values.append(result[expected_node])
        return result

    final_activations = validate_node_float_rows(
        data["final_activations"], "final_activations"
    )
    final_last_emitted = validate_node_float_rows(
        data["final_last_emitted"], "final_last_emitted"
    )
    expected_last_emitted = {node: 0.0 for node in nodes}
    for event in events:
        expected_last_emitted[event["node"]] = event["activation"]
    expected_final_activations = {
        node: expected_last_emitted[node] for node in graph.input_nodes
    }
    expected_final_activations.update(
        {
            node: next(
                (
                    activation
                    for (row_tick, row_node), (activation, _) in reversed(
                        list(observed_hidden_rows.items())
                    )
                    if row_node == node and row_tick <= tick
                ),
                0.0,
            )
            for node in graph.hidden_nodes
        }
    )
    expected_final_activations[graph.output_node] = query_activation
    if any(
        not same_float(final_last_emitted[node], expected_last_emitted[node])
        or not same_float(final_activations[node], expected_final_activations[node])
        for node in nodes
    ):
        raise RuntimeError("recurrent dynamics final node state is inconsistent")

    sequence_rows = require_list(data["event_sequence"], "event_sequence")
    if len(sequence_rows) != len(nodes):
        raise RuntimeError("recurrent dynamics event_sequence length is invalid")
    event_counts = {node: 0 for node in nodes}
    for event in events:
        event_counts[event["node"]] += 1
    for index, (row, expected_node) in enumerate(zip(sequence_rows, nodes, strict=True)):
        if (
            type(row) is not list
            or len(row) != 2
            or row[0] != expected_node
            or require_int(row[1], f"event_sequence[{index}].value")
            != event_counts[expected_node]
        ):
            raise RuntimeError("recurrent dynamics event_sequence is inconsistent")

    ledger = require_object(data["ledger_delta"], set(graph.ledger), "ledger_delta")
    for name, value in ledger.items():
        require_int(value, f"ledger_delta.{name}")
    arrived_count = sum(len(messages) for messages in arrivals.values())
    trace_count = sum(len(event["edge_traces"]) for event in events)
    expected_ledger_values = {
        "episodes": 1,
        "ticks": tick + 1,
        "source_events": sum(event["node"] in input_set for event in events),
        "arrived_messages": arrived_count,
        "forward_edge_touches": arrived_count,
        "activation_evaluations": len(evaluation_keys),
        "omission_evaluations": trace_count,
        "emitted_unit_events": len(events),
        "expired_event_records": sum(
            tick - event["step"] > graph.trace_horizon for event in events
        ),
        "nonfinite_values": 0,
    }
    zero_ledger_fields = {
        "credit_event_touches",
        "credit_edge_touches",
        "weight_write_touches",
        "clipped_updates",
        "credit_stop_expired",
        "credit_stop_hop_limit",
        "credit_stop_small",
        "credit_stop_source",
        "credit_stop_zero_omission",
        "credit_stop_missing_parent",
    }
    if any(ledger[name] != value for name, value in expected_ledger_values.items()) or any(
        ledger[name] != 0 for name in zero_ledger_fields
    ):
        raise RuntimeError("recurrent dynamics ledger_delta is inconsistent")

    stored = data["float64_bundle_sha256"]
    values = event_values + pending_values + activation_values + final_values
    if type(stored) is not str or stored != _array_bundle_sha256(values):
        raise RuntimeError("recurrent dynamics float64 bundle hash is invalid")


def _collect_split(
    graph: RecurrentEventGraph,
    stream: Experiment000Stream,
    *,
    condition: str,
    is_native: bool,
    feature_edge_ids: tuple[str, ...],
    native_diagnostic_graph: ReadoutTraceRecurrentEventGraph | None = None,
    expected_rows: Sequence[Mapping[str, Any]] | None = None,
    retain_rows: bool = True,
) -> dict[str, Any]:
    cue_zero_features: list[np.ndarray] = []
    cue_one_features: list[np.ndarray] = []
    cue_zero_outputs: list[float] = []
    cue_one_outputs: list[float] = []
    cue_zero_trace64: list[np.ndarray] = []
    cue_one_trace64: list[np.ndarray] = []
    event_counts: list[int] = []
    hidden_event_counts: list[int] = []
    hidden_activation_counts: list[int] = []
    saturation_counts = {threshold: 0 for threshold in SATURATION_THRESHOLDS}
    maximum_absolute_preactivation = 0.0
    pair_rows: list[dict[str, Any]] = []
    ledger_before = graph.ledger
    trace_before = _trace_ledger(graph)

    if expected_rows is not None and len(expected_rows) != len(stream.episodes):
        raise RuntimeError("compact replay row count differs from regenerated stream")

    for pair_position, base in enumerate(stream.episodes):
        sample_identity = {
            "master_seed": int(base.master_seed),
            "split": base.split,
            "pair_position": pair_position,
            "pair_index": int(base.index),
            "base_episode_id": base.episode_id,
            "noise_stream_id": base.noise_stream_id,
            "base_episode_sha256": _sha256_json(base.canonical_record()),
        }
        row: dict[str, Any] = {
            "pair_index": int(base.index),
            "base_episode_id": base.episode_id,
            "noise_stream_id": base.noise_stream_id,
            "sample_identity": sample_identity,
            "sample_identity_sha256": _sha256_json(sample_identity),
        }
        for cue in (0, 1):
            episode = _counterfactual_episode(base, cue=cue, condition=condition)
            episode_ledger_before = graph.ledger
            episode_trace_before = _trace_ledger(graph)
            query = forward_episode(graph, episode)
            episode_ledger_after = graph.ledger

            if is_native:
                extracted = list(extract_query_features(graph, query))
                output = float(query.activation)
                trace64: np.ndarray | None = None
                if native_diagnostic_graph is None:
                    raise RuntimeError("native condition lacks its rho=0 observer")
                observer_query = forward_episode(native_diagnostic_graph, episode)
                if (
                    query != observer_query
                    or graph.unit_events != native_diagnostic_graph.unit_events
                    or graph.ledger != native_diagnostic_graph.ledger
                    or tuple(extracted)
                    != extract_query_features(native_diagnostic_graph, observer_query)
                ):
                    raise RuntimeError("native diagnostic observer changed native behavior")
                records = native_diagnostic_graph.trace_activation_records
                observed_trace_ledger = native_diagnostic_graph.trace_ledger
                if any(observed_trace_ledger[field] for field in _TRACE_TOUCH_FIELDS):
                    raise RuntimeError("rho=0 native observer touched the passive trace")
            else:
                if not isinstance(graph, ReadoutTraceRecurrentEventGraph):
                    raise TypeError("candidate condition requires the trace subclass")
                extracted, trace64, candidate_preactivation, output = _candidate_trace_readout(
                    graph, query.tick, feature_edge_ids
                )
                records = graph.trace_activation_records
            if tuple(edge_id for edge_id, _ in extracted) != feature_edge_ids:
                raise RuntimeError("query feature order changed during readout-trace run")
            feature = np.asarray([value for _, value in extracted], dtype=np.float64)
            if not np.all(np.isfinite(feature)) or not math.isfinite(output):
                raise FloatingPointError("non-finite readout-trace forward value")
            if trace64 is not None and (
                trace64.shape != (len(graph.hidden_nodes),)
                or not np.all(np.isfinite(trace64))
            ):
                raise FloatingPointError("64-D trace diagnostic is incomplete or non-finite")
            if any(
                not math.isfinite(float(event.preactivation))
                or not math.isfinite(float(event.activation))
                or abs(float(event.activation)) > 1.0 + 1e-15
                for event in graph.unit_events
            ):
                raise FloatingPointError("event left the finite tanh range")

            weighted_sum = math.fsum(
                graph.weights[edge_id] * value for edge_id, value in extracted
            )
            if is_native:
                output_event = graph.unit_events_by_id[query.event_id]
                readout_preactivation = float(output_event.preactivation)
            else:
                readout_preactivation = float(candidate_preactivation)
            reconstruction_error = abs(weighted_sum - readout_preactivation)
            reconstruction_tolerance = MATRIX_ABS_TOLERANCE + MATRIX_REL_TOLERANCE * max(
                abs(weighted_sum), abs(readout_preactivation)
            )
            if reconstruction_error > reconstruction_tolerance:
                raise RuntimeError("16-D output feature reconstruction failed")
            expected_output = math.tanh(weighted_sum)
            output_reconstruction_error = abs(float(output) - expected_output)
            output_reconstruction_tolerance = (
                MATRIX_ABS_TOLERANCE
                + MATRIX_REL_TOLERANCE * max(abs(float(output)), abs(expected_output))
            )
            if output_reconstruction_error > output_reconstruction_tolerance:
                raise RuntimeError("candidate/native output is not tanh of the 16-D sum")

            hidden_activation_count = len(records)
            if hidden_activation_count <= 0:
                raise RuntimeError("episode produced no hidden activation evaluations")
            if any(
                not all(
                    math.isfinite(v)
                    for v in (r.activation, r.retained_trace, r.trace_after)
                )
                or abs(r.activation) > 1.0
                or abs(r.trace_after) > 1.0 + 1e-12
                for r in records
            ):
                raise FloatingPointError("hidden activation / trace diagnostic invalid")
            episode_max_preactivation = max(
                abs(float(event.preactivation))
                for event in graph.unit_events
                if event.node in graph.hidden_nodes
            )
            maximum_absolute_preactivation = max(
                maximum_absolute_preactivation, episode_max_preactivation
            )
            for threshold in SATURATION_THRESHOLDS:
                saturation_counts[threshold] += sum(
                    abs(r.activation) >= threshold for r in records
                )

            event_count = len(graph.unit_events)
            hidden_event_count = sum(
                event.node in graph.hidden_nodes for event in graph.unit_events
            )
            event_counts.append(event_count)
            hidden_event_counts.append(hidden_event_count)
            hidden_activation_counts.append(hidden_activation_count)

            trace_after = _trace_ledger(graph)
            # A3 hashes the complete native transcript while it is the sole live
            # episode transcript.  The object is never attached to a retained row.
            dynamics_commitment = _sha256_json(_normalized_recurrent_dynamics(
                graph,
                query,
                records,
                episode_ledger_before,
                episode_ledger_after,
            ))
            row[f"cue_{cue}_output_activation"] = output
            row[f"cue_{cue}_recurrent_dynamics_sha256"] = dynamics_commitment
            row[f"cue_{cue}_native_query_activation"] = float(query.activation)
            row[f"cue_{cue}_native_query_preactivation"] = float(
                graph.unit_events_by_id[query.event_id].preactivation
            )
            row[f"cue_{cue}_feature_vector"] = feature.tolist()
            row[f"cue_{cue}_feature_vector_sha256"] = _array_bundle_sha256(feature)
            row[f"cue_{cue}_trace64_vector"] = (
                None if trace64 is None else trace64.tolist()
            )
            row[f"cue_{cue}_trace64_vector_sha256"] = (
                None if trace64 is None else _array_bundle_sha256(trace64)
            )
            row[f"cue_{cue}_unit_events_sha256"] = _unit_events_sha256(graph)
            row[f"cue_{cue}_activation_dynamics_sha256"] = (
                _activation_dynamics_sha256(records)
            )
            row[f"cue_{cue}_legacy_ledger_delta"] = _legacy_ledger_delta(
                episode_ledger_after, episode_ledger_before
            )
            row[f"cue_{cue}_legacy_ledger_delta_sha256"] = _sha256_json(
                row[f"cue_{cue}_legacy_ledger_delta"]
            )
            row[f"cue_{cue}_emitted_unit_events"] = event_count
            row[f"cue_{cue}_hidden_unit_events"] = hidden_event_count
            row[f"cue_{cue}_hidden_emission_density"] = float(
                hidden_event_count / (len(graph.hidden_nodes) * (stream.noise_events + 3))
            )
            row[f"cue_{cue}_forward_edge_touches"] = int(
                episode_ledger_after["forward_edge_touches"]
                - episode_ledger_before["forward_edge_touches"]
            )
            row[f"cue_{cue}_activation_evaluations"] = int(
                episode_ledger_after["activation_evaluations"]
                - episode_ledger_before["activation_evaluations"]
            )
            row[f"cue_{cue}_hidden_activation_evaluations"] = hidden_activation_count
            for field in _TRACE_TOUCH_FIELDS:
                row[f"cue_{cue}_{field}"] = int(
                    trace_after[field] - episode_trace_before[field]
                )
            row[f"cue_{cue}_maximum_absolute_hidden_preactivation"] = float(
                episode_max_preactivation
            )
            row[f"cue_{cue}_readout_preactivation"] = readout_preactivation
            row[f"cue_{cue}_weighted_feature_sum"] = weighted_sum
            row[f"cue_{cue}_feature_reconstruction_error"] = reconstruction_error
            row[f"cue_{cue}_feature_reconstruction_tolerance"] = (
                reconstruction_tolerance
            )
            row[f"cue_{cue}_feature_reconstruction_pass"] = True
            row[f"cue_{cue}_output_reconstruction_error"] = output_reconstruction_error
            row[f"cue_{cue}_output_reconstruction_tolerance"] = (
                output_reconstruction_tolerance
            )
            row[f"cue_{cue}_output_reconstruction_pass"] = True
            row[f"cue_{cue}_hidden_activation_saturation_counts"] = {
                f"abs_ge_{threshold}": sum(
                    abs(r.activation) >= threshold for r in records
                )
                for threshold in SATURATION_THRESHOLDS
            }

            if cue == 0:
                cue_zero_features.append(feature)
                cue_zero_outputs.append(output)
                if trace64 is not None:
                    cue_zero_trace64.append(trace64)
            else:
                cue_one_features.append(feature)
                cue_one_outputs.append(output)
                if trace64 is not None:
                    cue_one_trace64.append(trace64)
        if expected_rows is not None and row != dict(expected_rows[pair_position]):
            raise RuntimeError(
                "compact recurrent replay differs at "
                f"seed={stream.master_seed} split={stream.split} row={pair_position}"
            )
        if retain_rows:
            pair_rows.append(row)

    ledger_after = graph.ledger
    trace_ledger_after = _trace_ledger(graph)
    forward_episodes = 2 * len(stream.episodes)
    total_events = int(sum(event_counts))
    hidden_opportunities = int(
        forward_episodes * len(graph.hidden_nodes) * (stream.noise_events + 3)
    )
    emitted_delta = int(
        ledger_after["emitted_unit_events"] - ledger_before["emitted_unit_events"]
    )
    if emitted_delta != total_events:
        raise RuntimeError("per-episode activity does not match the graph ledger")
    if ledger_after["weight_write_touches"] != ledger_before["weight_write_touches"]:
        raise RuntimeError("no-learning readout-trace run wrote graph weights")
    if ledger_after["credit_event_touches"] != ledger_before["credit_event_touches"]:
        raise RuntimeError("no-learning readout-trace run processed credit")

    total_hidden_activations = int(sum(hidden_activation_counts))
    paired = _PairedSplit(
        cue_zero_features=np.stack(cue_zero_features),
        cue_one_features=np.stack(cue_one_features),
        cue_zero_outputs=np.asarray(cue_zero_outputs, dtype=np.float64),
        cue_one_outputs=np.asarray(cue_one_outputs, dtype=np.float64),
    )
    trace64_paired = (
        None
        if is_native
        else _PairedSplit(
            cue_zero_features=np.stack(cue_zero_trace64),
            cue_one_features=np.stack(cue_one_trace64),
            cue_zero_outputs=np.asarray(cue_zero_outputs, dtype=np.float64),
            cue_one_outputs=np.asarray(cue_one_outputs, dtype=np.float64),
        )
    )
    activity = {
        "counterfactual_forward_episodes": forward_episodes,
        "total_emitted_unit_events": total_events,
        "mean_emitted_unit_events_per_forward_episode": float(
            total_events / forward_episodes
        ),
        "minimum_emitted_unit_events_per_forward_episode": min(event_counts),
        "maximum_emitted_unit_events_per_forward_episode": max(event_counts),
        "event_counts_sha256": _array_bundle_sha256(event_counts),
        "hidden_event_counts_sha256": _array_bundle_sha256(hidden_event_counts),
        "pair_rows_sha256": _sha256_json(
            pair_rows if retain_rows else list(expected_rows or ())
        ),
        "forward_edge_touches": int(
            ledger_after["forward_edge_touches"] - ledger_before["forward_edge_touches"]
        ),
        "total_hidden_unit_events": int(sum(hidden_event_counts)),
        "hidden_emission_opportunities": hidden_opportunities,
        "hidden_emission_density": float(sum(hidden_event_counts) / hidden_opportunities),
        "activation_evaluations": int(
            ledger_after["activation_evaluations"]
            - ledger_before["activation_evaluations"]
        ),
        "hidden_activation_evaluations": total_hidden_activations,
        "hidden_activation_counts_sha256": _array_bundle_sha256(hidden_activation_counts),
        "maximum_absolute_hidden_preactivation": float(maximum_absolute_preactivation),
        "hidden_activation_saturation": {
            f"abs_ge_{threshold}": {
                "count": int(saturation_counts[threshold]),
                "fraction": float(saturation_counts[threshold] / total_hidden_activations),
            }
            for threshold in SATURATION_THRESHOLDS
        },
        "local_trace_touches": {
            field: int(trace_ledger_after[field] - trace_before[field])
            for field in _TRACE_TOUCH_FIELDS
        },
        "hidden_activation_evaluations_delta": int(
            trace_ledger_after["hidden_activation_evaluations"]
            - trace_before["hidden_activation_evaluations"]
        ),
        "nonfinite_values": int(
            ledger_after["nonfinite_values"] - ledger_before["nonfinite_values"]
        ),
        "credit_event_touches": int(
            ledger_after["credit_event_touches"] - ledger_before["credit_event_touches"]
        ),
        "weight_write_touches": int(
            ledger_after["weight_write_touches"] - ledger_before["weight_write_touches"]
        ),
    }
    return {
        "paired": paired,
        "trace64_paired": trace64_paired,
        "activity": activity,
        "pair_rows": pair_rows,
        "all_feature_reconstructions_pass": True,
        "all_output_reconstructions_pass": True,
    }


def _combine_activity(train: Mapping[str, Any], evaluation: Mapping[str, Any]) -> dict[str, Any]:
    episodes = int(train["counterfactual_forward_episodes"]) + int(
        evaluation["counterfactual_forward_episodes"]
    )
    total = int(train["total_emitted_unit_events"]) + int(
        evaluation["total_emitted_unit_events"]
    )
    total_hidden = int(train["total_hidden_unit_events"]) + int(
        evaluation["total_hidden_unit_events"]
    )
    hidden_opportunities = int(train["hidden_emission_opportunities"]) + int(
        evaluation["hidden_emission_opportunities"]
    )
    hidden_activations = int(train["hidden_activation_evaluations"]) + int(
        evaluation["hidden_activation_evaluations"]
    )
    return {
        "counterfactual_forward_episodes": episodes,
        "total_emitted_unit_events": total,
        "mean_emitted_unit_events_per_forward_episode": float(total / episodes),
        "minimum_emitted_unit_events_per_forward_episode": min(
            int(train["minimum_emitted_unit_events_per_forward_episode"]),
            int(evaluation["minimum_emitted_unit_events_per_forward_episode"]),
        ),
        "maximum_emitted_unit_events_per_forward_episode": max(
            int(train["maximum_emitted_unit_events_per_forward_episode"]),
            int(evaluation["maximum_emitted_unit_events_per_forward_episode"]),
        ),
        "forward_edge_touches": int(train["forward_edge_touches"])
        + int(evaluation["forward_edge_touches"]),
        "activation_evaluations": int(train["activation_evaluations"])
        + int(evaluation["activation_evaluations"]),
        "hidden_activation_evaluations": hidden_activations,
        "total_hidden_unit_events": total_hidden,
        "hidden_emission_opportunities": hidden_opportunities,
        "hidden_emission_density": float(total_hidden / hidden_opportunities),
        "maximum_absolute_hidden_preactivation": max(
            float(train["maximum_absolute_hidden_preactivation"]),
            float(evaluation["maximum_absolute_hidden_preactivation"]),
        ),
        "hidden_activation_saturation": {
            f"abs_ge_{threshold}": {
                "count": int(
                    train["hidden_activation_saturation"][f"abs_ge_{threshold}"]["count"]
                )
                + int(
                    evaluation["hidden_activation_saturation"][f"abs_ge_{threshold}"][
                        "count"
                    ]
                ),
                "fraction": float(
                    (
                        int(
                            train["hidden_activation_saturation"][
                                f"abs_ge_{threshold}"
                            ]["count"]
                        )
                        + int(
                            evaluation["hidden_activation_saturation"][
                                f"abs_ge_{threshold}"
                            ]["count"]
                        )
                    )
                    / hidden_activations
                ),
            }
            for threshold in SATURATION_THRESHOLDS
        },
        "local_trace_touches": {
            field: int(train["local_trace_touches"][field])
            + int(evaluation["local_trace_touches"][field])
            for field in _TRACE_TOUCH_FIELDS
        },
    }


# ----------------------------------------------------------------------
# Blank-tail / wake assay


def _blank_tail_stability(
    graph: RecurrentEventGraph,
    *,
    condition: str,
    is_native: bool,
    feature_edge_ids: tuple[str, ...],
) -> dict[str, Any]:
    ledger_before = graph.ledger
    weights_before = graph.weights_hash()
    structural_before = _structural_mask_sha256(graph)
    hidden_nodes = set(graph.hidden_nodes)
    polarity_rows: list[dict[str, Any]] = []
    for cue_input in (-1.0, 1.0):
        polarity_ledger_before = graph.ledger
        polarity_trace_before = _trace_ledger(graph)
        cue_label = "negative" if cue_input < 0.0 else "positive"
        graph.begin_episode(f"readout-trace-blank-tail-{cue_label}")
        native_step_rows = [asdict(graph.step({"cue": cue_input}))]
        hidden_counts: list[int] = []
        for _ in range(BLANK_TAIL_TICKS):
            result = graph.step({})
            native_step_rows.append(asdict(result))
            emitted_nodes = tuple(
                event_id.rsplit(":", 2)[1] for event_id in result.emitted_event_ids
            )
            hidden_counts.append(sum(node in hidden_nodes for node in emitted_nodes))

        if is_native:
            effective = {node: 0.0 for node in graph.hidden_nodes}
            effective_maximum = 0.0
        else:
            if not isinstance(graph, ReadoutTraceRecurrentEventGraph):
                raise TypeError("candidate tail requires the trace subclass")
            snapshot_before = graph.trace_snapshot
            effective = graph.effective_trace(BLANK_TAIL_TICKS)
            effective_maximum = max(abs(value) for value in effective.values())
            snapshot_after = graph.trace_snapshot
            if snapshot_before != snapshot_after:
                raise RuntimeError("effective-trace observation mutated local trace")

        # Advance to the forced-output tick and read the condition's own readout.
        native_step_rows.append(asdict(graph._advance({"query": 1.0})))
        native_step_rows.append(asdict(graph._advance({})))
        native_step_rows.append(asdict(graph._advance({}, force_output=True)))
        latest = graph.audit["latest_output_event_id"]
        if latest is None:
            raise RuntimeError("blank-tail wake query produced no forced output")
        wake_event = graph.unit_events_by_id[latest]
        if is_native:
            from .recurrent import QueryResult

            wake_result = QueryResult(
                tick=wake_event.step,
                event_id=wake_event.event_id,
                activation=wake_event.activation,
                prediction=1 if wake_event.activation >= 0.0 else 0,
            )
            wake_features = [v for _, v in extract_query_features(graph, wake_result)]
            wake_output = float(wake_event.activation)
        else:
            if not isinstance(graph, ReadoutTraceRecurrentEventGraph):
                raise TypeError("candidate wake requires the trace subclass")
            extracted, _, _, wake_output = _candidate_trace_readout(
                graph, wake_event.step, feature_edge_ids
            )
            wake_features = [v for _, v in extracted]

        polarity_ledger_after = graph.ledger
        last_nonzero = next(
            (
                index
                for index in range(BLANK_TAIL_TICKS, 0, -1)
                if hidden_counts[index - 1] > 0
            ),
            None,
        )
        final_window = hidden_counts[-BLANK_TAIL_FINAL_WINDOW_TICKS:]
        row = {
            "cue_input": cue_input,
            "hidden_emissions_per_blank_tick": hidden_counts,
            "hidden_counts_sha256": _array_bundle_sha256(hidden_counts),
            "total_hidden_emissions": int(sum(hidden_counts)),
            "hidden_emission_density": float(
                sum(hidden_counts) / (BLANK_TAIL_TICKS * len(graph.hidden_nodes))
            ),
            "last_nonzero_hidden_blank_tick": last_nonzero,
            "final_window_hidden_emissions": int(sum(final_window)),
            "final_window_quiescent": sum(final_window) == 0,
            "effective_trace_maximum_absolute": float(effective_maximum),
            "effective_trace_gate_pass": effective_maximum
            <= GATE_THRESHOLDS["effective_state_maximum_absolute"],
            "wake_feature": wake_features,
            "wake_feature_sha256": _array_bundle_sha256(wake_features),
            "wake_output_activation": float(wake_output),
            "forward_edge_touches": int(
                polarity_ledger_after["forward_edge_touches"]
                - polarity_ledger_before["forward_edge_touches"]
            ),
            "trace_touch_deltas": {
                field: int(_trace_ledger(graph)[field] - polarity_trace_before[field])
                for field in _TRACE_TOUCH_FIELDS
            },
            "native_recurrent_step_rows_sha256": _sha256_json(native_step_rows),
            "native_unit_events_sha256": _unit_events_sha256(graph),
            "native_legacy_ledger_delta": _legacy_ledger_delta(
                polarity_ledger_after, polarity_ledger_before
            ),
            "weights_unchanged": weights_before == graph.weights_hash(),
            "structural_unchanged": structural_before == _structural_mask_sha256(graph),
        }
        row["native_legacy_ledger_delta_sha256"] = _sha256_json(
            row["native_legacy_ledger_delta"]
        )
        row["native_recurrent_dynamics_sha256"] = _sha256_json(
            {
                "step_rows_sha256": row["native_recurrent_step_rows_sha256"],
                "unit_events_sha256": row["native_unit_events_sha256"],
                "legacy_ledger_delta": row["native_legacy_ledger_delta"],
            }
        )
        polarity_rows.append(row)
        if not row["weights_unchanged"] or not row["structural_unchanged"]:
            raise RuntimeError("a blank-tail polarity changed frozen graph state")

    negative_feature = np.asarray(polarity_rows[0]["wake_feature"], dtype=np.float64)
    positive_feature = np.asarray(polarity_rows[1]["wake_feature"], dtype=np.float64)
    wake_feature_half_l2 = float(np.linalg.norm(positive_feature - negative_feature) / 2.0)
    wake_output_half_abs = float(
        abs(
            polarity_rows[1]["wake_output_activation"]
            - polarity_rows[0]["wake_output_activation"]
        )
        / 2.0
    )
    ledger_after = graph.ledger
    if ledger_before["weight_write_touches"] != ledger_after["weight_write_touches"]:
        raise RuntimeError("blank-tail assay wrote graph weights")
    if ledger_before["credit_event_touches"] != ledger_after["credit_event_touches"]:
        raise RuntimeError("blank-tail assay processed credit")
    total_hidden = sum(row["total_hidden_emissions"] for row in polarity_rows)
    all_gates = (
        all(
            row["final_window_quiescent"] and row["effective_trace_gate_pass"]
            for row in polarity_rows
        )
        and wake_feature_half_l2
        <= GATE_THRESHOLDS["wake_feature_half_difference_l2_maximum"]
        and wake_output_half_abs
        <= GATE_THRESHOLDS["wake_output_half_difference_absolute_maximum"]
    )
    return {
        "cue_polarities": polarity_rows,
        "polarity_bundle_sha256": _sha256_json(polarity_rows),
        "all_polarities_final_window_quiescent": all(
            row["final_window_quiescent"] for row in polarity_rows
        ),
        "all_polarities_effective_trace_pass": all(
            row["effective_trace_gate_pass"] for row in polarity_rows
        ),
        "wake_feature_half_difference_l2": wake_feature_half_l2,
        "wake_output_half_difference_absolute": wake_output_half_abs,
        "all_tail_state_wake_gates_pass": all_gates,
        "total_hidden_emissions": int(total_hidden),
        "hidden_emission_density": float(
            total_hidden / (2 * BLANK_TAIL_TICKS * len(graph.hidden_nodes))
        ),
    }


# ----------------------------------------------------------------------
# Condition, seed, aggregate


def _frozen_graph_snapshot(graph: RecurrentEventGraph) -> dict[str, Any]:
    """Return every frozen structural/numeric field at one probe boundary."""

    return {
        "topology_sha256": graph.topology_hash(),
        "structural_mask_sha256": _structural_mask_sha256(graph),
        "complete_weight_sha256": graph.weights_hash(),
        "framed_complete_weight_sha256": _edge_weight_sha256(
            graph, kinds=("input", "recurrent", "output")
        ),
        "per_kind_weight_sha256": {
            kind: _edge_weight_sha256(graph, kinds=(kind,))
            for kind in ("input", "recurrent", "output")
        },
        "recurrent_matrix": _recurrent_matrix_metrics(graph),
    }


def _run_condition(
    *,
    native_graph: RecurrentEventGraph,
    retention: float | None,
    stream_pair: Any,
    ridge_alpha: float,
) -> dict[str, Any]:
    condition = condition_name(retention)
    is_native = retention is None
    graph: RecurrentEventGraph = (
        native_graph
        if is_native
        else clone_with_readout_trace(
            native_graph, float(retention), event_log_enabled=False
        )
    )
    native_diagnostic_graph = (
        clone_with_readout_trace(native_graph, 0.0, event_log_enabled=False)
        if is_native
        else None
    )
    native_snapshot = _frozen_graph_snapshot(native_graph)
    snapshots = {"before_train": _frozen_graph_snapshot(graph)}
    native_matrix = native_snapshot["recurrent_matrix"]
    condition_matrix = snapshots["before_train"]["recurrent_matrix"]
    feature_edge_ids = _output_feature_edge_ids(graph)
    native_feature_edge_ids = _output_feature_edge_ids(native_graph)

    topology_before = graph.topology_hash()
    structural_before = _structural_mask_sha256(graph)
    weights_before = graph.weights_hash()
    framed_before = _edge_weight_sha256(
        graph, kinds=("input", "recurrent", "output")
    )

    train = _collect_split(
        graph, stream_pair.train, condition=condition, is_native=is_native,
        feature_edge_ids=feature_edge_ids,
        native_diagnostic_graph=native_diagnostic_graph,
    )
    snapshots["after_train"] = _frozen_graph_snapshot(graph)
    weights_mid = graph.weights_hash()
    evaluation = _collect_split(
        graph, stream_pair.eval, condition=condition, is_native=is_native,
        feature_edge_ids=feature_edge_ids,
        native_diagnostic_graph=native_diagnostic_graph,
    )
    snapshots["after_eval"] = _frozen_graph_snapshot(graph)
    weights_after_eval = graph.weights_hash()
    blank_tail = _blank_tail_stability(
        graph, condition=condition, is_native=is_native,
        feature_edge_ids=feature_edge_ids,
    )
    snapshots["after_blank_tail"] = _frozen_graph_snapshot(graph)
    weights_after = graph.weights_hash()
    framed_after = _edge_weight_sha256(
        graph, kinds=("input", "recurrent", "output")
    )
    topology_after = graph.topology_hash()
    structural_after = _structural_mask_sha256(graph)

    train_metrics = paired_retention_metrics(train["paired"])
    eval_metrics = paired_retention_metrics(evaluation["paired"])
    ridge = strict_ridge_readout(
        train["paired"], evaluation["paired"], alpha=ridge_alpha
    )
    if is_native:
        trace64_diagnostic: dict[str, Any] = {
            "applicable": False,
            "selecting": False,
            "reason": "native control has no passive trace",
            "feature_dimension": 64,
            "hidden_node_order": list(graph.hidden_nodes),
        }
    else:
        train_trace64 = train["trace64_paired"]
        eval_trace64 = evaluation["trace64_paired"]
        if train_trace64 is None or eval_trace64 is None:
            raise RuntimeError("candidate omitted its 64-D diagnostic trace")
        trace64_diagnostic = {
            "applicable": True,
            "selecting": False,
            "feature_dimension": 64,
            "hidden_node_order": list(graph.hidden_nodes),
            "train": paired_retention_metrics(train_trace64),
            "eval": paired_retention_metrics(eval_trace64),
            "ridge_readout": strict_ridge_readout(
                train_trace64, eval_trace64, alpha=ridge_alpha
            ),
        }
    combined_activity = _combine_activity(train["activity"], evaluation["activity"])

    ledger = graph.ledger
    trace_ledger = _trace_ledger(graph)
    episodes = ledger["episodes"]
    expected_reset = 0 if is_native else episodes * len(graph.hidden_nodes)
    # 2 splits * 2 cues * pairs episodes + 2 blank-tail polarity episodes.
    trace_touch_equal = (
        trace_ledger["local_trace_read_touches"]
        == trace_ledger["local_trace_decay_touches"]
        == trace_ledger["local_trace_write_touches"]
    )
    native_weights = native_graph.weights
    condition_weights = graph.weights
    weights_match_native_bits = all(
        np.asarray([condition_weights[e.edge_id]], dtype="<f8").tobytes()
        == np.asarray([native_weights[e.edge_id]], dtype="<f8").tobytes()
        for e in graph.edges
    )

    invariant = {
        "native_control_is_literal_recurrent_event_graph": (
            not is_native or type(graph) is RecurrentEventGraph
        ),
        "topology_unchanged": topology_before == topology_after,
        "topology_matches_native": topology_before == native_graph.topology_hash(),
        "structural_mask_unchanged": structural_before == structural_after,
        "structural_mask_matches_native": structural_before
        == _structural_mask_sha256(native_graph),
        "weights_unchanged_during_probe": weights_before
        == weights_mid
        == weights_after_eval
        == weights_after,
        "framed_weights_unchanged_during_probe": framed_before == framed_after,
        "weights_match_native": weights_before == native_graph.weights_hash()
        and weights_match_native_bits,
        "recurrent_weights_match_native": _edge_weight_sha256(graph, kinds=("recurrent",))
        == _edge_weight_sha256(native_graph, kinds=("recurrent",)),
        "input_weights_match_native": _edge_weight_sha256(graph, kinds=("input",))
        == _edge_weight_sha256(native_graph, kinds=("input",)),
        "output_weights_match_native": _edge_weight_sha256(graph, kinds=("output",))
        == _edge_weight_sha256(native_graph, kinds=("output",)),
        "spectral_radius_matches_native": _within_matrix_tolerance(
            condition_matrix["spectral_radius"], native_matrix["spectral_radius"]
        ),
        "operator_norm_matches_native": _within_matrix_tolerance(
            condition_matrix["operator_2_norm"], native_matrix["operator_2_norm"]
        ),
        "retention_immutable": is_native
        or (
            isinstance(graph, ReadoutTraceRecurrentEventGraph)
            and graph.trace_retention == retention
        ),
        "trace_bound_ok": is_native
        or (
            isinstance(graph, ReadoutTraceRecurrentEventGraph)
            and graph.trace_bound_ok
        ),
        "trace_resets_valid": is_native
        or (
            isinstance(graph, ReadoutTraceRecurrentEventGraph)
            and graph.trace_resets_valid
        ),
        "trace_reset_count_matches_episodes": (
            isinstance(graph, ReadoutTraceRecurrentEventGraph)
            and graph.trace_reset_count == episodes
            if not is_native
            else True
        ),
        "reset_touches_exact": trace_ledger["local_trace_reset_touches"] == expected_reset,
        "candidate_trace_touches_consistent": (
            trace_touch_equal
            and (
                is_native
                or trace_ledger["local_trace_read_touches"]
                == trace_ledger["hidden_activation_evaluations"]
            )
        ),
        "native_zero_trace_touches": (
            not is_native
            or all(trace_ledger[field] == 0 for field in _TRACE_TOUCH_FIELDS)
        ),
        "feature_edges_match_native": feature_edge_ids == native_feature_edge_ids,
        "all_feature_reconstructions_pass": bool(
            train["all_feature_reconstructions_pass"]
            and evaluation["all_feature_reconstructions_pass"]
        ),
        "all_output_reconstructions_pass": bool(
            train["all_output_reconstructions_pass"]
            and evaluation["all_output_reconstructions_pass"]
        ),
        "all_frozen_snapshots_equal_native": all(
            snapshot == native_snapshot for snapshot in snapshots.values()
        ),
        "credit_event_touches_zero": ledger["credit_event_touches"] == 0,
        "credit_edge_touches_zero": ledger["credit_edge_touches"] == 0,
        "weight_write_touches_zero": ledger["weight_write_touches"] == 0,
        "credit_packets_zero": len(graph.credit_packets) == 0,
        "nonfinite_values_zero": ledger["nonfinite_values"] == 0,
        "all_weights_finite": all(math.isfinite(v) for v in condition_weights.values()),
        "all_weights_within_clip": all(
            abs(v) <= graph.weight_clip for v in condition_weights.values()
        ),
        "recurrent_activity_equals_native": True,  # filled by _attach_native_comparison
    }
    if not all(v for k, v in invariant.items() if k != "recurrent_activity_equals_native"):
        failed = sorted(k for k, v in invariant.items() if not v)
        raise RuntimeError(f"readout-trace condition invariant failed: {failed}")

    return {
        "condition": condition,
        "retention_coefficient": 0.0 if retention is None else retention,
        "target_retention": retention,
        "retention_half_life_ticks": (
            None if retention is None else float(math.log(0.5) / math.log(retention))
        ),
        "retention_after_ten_ticks": 0.0 if retention is None else float(retention**10),
        "data_pair_sha256": stream_pair.manifest.pair_sha256,
        "graph": {
            "implementation_class": type(graph).__name__,
            "native_control_is_literal_recurrent_event_graph": (
                is_native and type(graph) is RecurrentEventGraph
            ),
            "hidden_units": len(graph.hidden_nodes),
            "feature_dimension": len(feature_edge_ids),
            "feature_edge_ids": list(feature_edge_ids),
            "topology_sha256": topology_before,
            "structural_mask_sha256": structural_before,
            "initial_weights_sha256": weights_before,
            "final_weights_sha256": weights_after,
            "recurrent_matrix": condition_matrix,
            "hyperparameters": _constructor_hyperparameters(graph),
            "native_reference_snapshot": native_snapshot,
            "frozen_state_snapshots": snapshots,
        },
        "train": train_metrics,
        "eval": eval_metrics,
        "ridge_readout": ridge,
        "trace64_diagnostic_only": trace64_diagnostic,
        "activity": {
            "train": train["activity"],
            "eval": evaluation["activity"],
            "combined": combined_activity,
        },
        "blank_tail_stability": blank_tail,
        "pair_rows": {
            "train": train["pair_rows"],
            "eval": evaluation["pair_rows"],
            "bundle_sha256": _sha256_json([train["pair_rows"], evaluation["pair_rows"]]),
        },
        "ledger": {
            "credit_event_touches": ledger["credit_event_touches"],
            "weight_write_touches": ledger["weight_write_touches"],
            "nonfinite_values": ledger["nonfinite_values"],
            "trace": trace_ledger,
        },
        "invariants": invariant,
    }


def _seed_gate(candidate: Mapping[str, Any], *, eligible: bool) -> dict[str, Any]:
    paired = candidate["paired_vs_native"]
    split_activity_pass = all(
        paired["activity_by_split"][split]["total_emitted_event_ratio"] is not None
        and paired["activity_by_split"][split]["total_emitted_event_ratio"]
        <= GATE_THRESHOLDS["event_activity_total_ratio_maximum"]
        and paired["activity_by_split"][split]["maximum_matched_episode_event_count_ratio"]
        <= GATE_THRESHOLDS["matched_episode_event_count_ratio_maximum"]
        and paired["activity_by_split"][split]["forward_edge_touch_ratio"] is not None
        and paired["activity_by_split"][split]["forward_edge_touch_ratio"]
        <= GATE_THRESHOLDS["forward_edge_touch_total_ratio_maximum"]
        and paired["activity_by_split"][split][
            "maximum_matched_episode_forward_edge_touch_ratio"
        ]
        <= GATE_THRESHOLDS["matched_episode_forward_edge_touch_ratio_maximum"]
        for split in ("train", "eval")
    )
    criteria = {
        "eval_cue_to_noise_ratio_at_least_0_50": (
            candidate["eval"]["cue_to_noise_feature_ratio"] is not None
            and candidate["eval"]["cue_to_noise_feature_ratio"]
            >= GATE_THRESHOLDS["eval_cue_to_noise_feature_ratio_minimum"]
        ),
        "paired_native_cue_ratio_multiple_at_least_3": (
            paired["eval_cue_to_noise_feature_ratio_multiple"] is not None
            and paired["eval_cue_to_noise_feature_ratio_multiple"]
            >= GATE_THRESHOLDS["paired_native_cue_ratio_multiple_minimum"]
        ),
        "paired_native_absolute_cue_delta_multiple_at_least_3": (
            paired["eval_absolute_cue_feature_delta_rms_multiple"] is not None
            and paired["eval_absolute_cue_feature_delta_rms_multiple"]
            >= GATE_THRESHOLDS["paired_native_absolute_cue_delta_multiple_minimum"]
        ),
        "ridge_eval_accuracy_at_least_0_70": (
            candidate["ridge_readout"]["eval_accuracy"]
            >= GATE_THRESHOLDS["ridge_eval_accuracy_minimum"]
        ),
        "eval_output_cue_delta_to_sd_at_least_0_50": (
            candidate["eval"]["output_cue_delta_to_sd"] is not None
            and candidate["eval"]["output_cue_delta_to_sd"]
            >= GATE_THRESHOLDS["eval_output_cue_delta_to_sd_minimum"]
        ),
        "every_split_activity_guard": split_activity_pass,
        "tail_effective_state_and_wake_gates": candidate["blank_tail_stability"][
            "all_tail_state_wake_gates_pass"
        ],
        "finite": candidate["invariants"]["nonfinite_values_zero"],
        "all_implementation_and_graph_invariants": all(
            v for k, v in candidate["invariants"].items()
        ),
    }
    return {
        "criteria": criteria,
        "pass": bool(eligible and all(criteria.values())),
        "eligible": eligible,
    }


def _attach_native_comparison(
    condition: dict[str, Any], native: Mapping[str, Any], *, gate_eligible: bool
) -> None:
    candidate_ratio = condition["eval"]["cue_to_noise_feature_ratio"]
    native_ratio = native["eval"]["cue_to_noise_feature_ratio"]
    candidate_output = condition["eval"]["output_cue_delta_to_sd"]
    native_output = native["eval"]["output_cue_delta_to_sd"]
    candidate_events = condition["activity"]["combined"][
        "mean_emitted_unit_events_per_forward_episode"
    ]
    native_events = native["activity"]["combined"][
        "mean_emitted_unit_events_per_forward_episode"
    ]
    split_activity: dict[str, Any] = {}
    matched_episode_rows: dict[str, list[dict[str, Any]]] = {}
    recurrent_activity_equal = True
    for split in ("train", "eval"):
        candidate_split = condition["activity"][split]
        native_split = native["activity"][split]
        candidate_pair_rows = condition["pair_rows"][split]
        native_pair_rows = native["pair_rows"][split]
        matched_event_ratios: list[float] = []
        matched_forward_ratios: list[float] = []
        matched_rows: list[dict[str, Any]] = []
        for candidate_pair, native_pair in zip(
            candidate_pair_rows, native_pair_rows, strict=True
        ):
            if (
                candidate_pair["base_episode_id"] != native_pair["base_episode_id"]
                or candidate_pair["noise_stream_id"] != native_pair["noise_stream_id"]
            ):
                raise RuntimeError("candidate/native activity rows are not paired")
            for cue in (0, 1):
                dynamics_equal = (
                    candidate_pair[f"cue_{cue}_emitted_unit_events"]
                    == native_pair[f"cue_{cue}_emitted_unit_events"]
                    and candidate_pair[f"cue_{cue}_forward_edge_touches"]
                    == native_pair[f"cue_{cue}_forward_edge_touches"]
                    and candidate_pair[f"cue_{cue}_unit_events_sha256"]
                    == native_pair[f"cue_{cue}_unit_events_sha256"]
                    and candidate_pair[f"cue_{cue}_activation_dynamics_sha256"]
                    == native_pair[f"cue_{cue}_activation_dynamics_sha256"]
                    and candidate_pair[f"cue_{cue}_recurrent_dynamics_sha256"]
                    == native_pair[f"cue_{cue}_recurrent_dynamics_sha256"]
                    and candidate_pair[f"cue_{cue}_legacy_ledger_delta"]
                    == native_pair[f"cue_{cue}_legacy_ledger_delta"]
                    and np.asarray(
                        [candidate_pair[f"cue_{cue}_native_query_activation"]],
                        dtype="<f8",
                    ).tobytes()
                    == np.asarray(
                        [native_pair[f"cue_{cue}_native_query_activation"]],
                        dtype="<f8",
                    ).tobytes()
                )
                if not dynamics_equal:
                    recurrent_activity_equal = False
                matched_event_ratios.append(
                    _safe_ratio(
                        float(candidate_pair[f"cue_{cue}_emitted_unit_events"]),
                        float(native_pair[f"cue_{cue}_emitted_unit_events"]),
                    )
                )
                matched_forward_ratios.append(
                    _safe_ratio(
                        float(candidate_pair[f"cue_{cue}_forward_edge_touches"]),
                        float(native_pair[f"cue_{cue}_forward_edge_touches"]),
                    )
                )
                matched_rows.append(
                    {
                        "pair_index": candidate_pair["pair_index"],
                        "base_episode_id": candidate_pair["base_episode_id"],
                        "noise_stream_id": candidate_pair["noise_stream_id"],
                        "cue": cue,
                        "emitted_event_ratio": matched_event_ratios[-1],
                        "forward_edge_touch_ratio": matched_forward_ratios[-1],
                        "native_recurrent_dynamics_bitwise_equal": dynamics_equal,
                        "unit_events_sha256": candidate_pair[
                            f"cue_{cue}_unit_events_sha256"
                        ],
                        "activation_dynamics_sha256": candidate_pair[
                            f"cue_{cue}_activation_dynamics_sha256"
                        ],
                        "recurrent_dynamics_sha256": candidate_pair[
                            f"cue_{cue}_recurrent_dynamics_sha256"
                        ],
                        "legacy_ledger_delta_sha256": candidate_pair[
                            f"cue_{cue}_legacy_ledger_delta_sha256"
                        ],
                        "output_activation_difference": float(
                            candidate_pair[f"cue_{cue}_output_activation"]
                            - native_pair[f"cue_{cue}_output_activation"]
                        ),
                    }
                )
        if any(v is None for v in matched_event_ratios + matched_forward_ratios):
            raise RuntimeError("native episode had zero activity denominator")
        split_activity[split] = {
            "total_emitted_event_ratio": _safe_ratio(
                float(candidate_split["total_emitted_unit_events"]),
                float(native_split["total_emitted_unit_events"]),
            ),
            "maximum_matched_episode_event_count_ratio": max(matched_event_ratios),
            "forward_edge_touch_ratio": _safe_ratio(
                float(candidate_split["forward_edge_touches"]),
                float(native_split["forward_edge_touches"]),
            ),
            "maximum_matched_episode_forward_edge_touch_ratio": max(matched_forward_ratios),
        }
        matched_episode_rows[split] = matched_rows
    candidate_tail = condition["blank_tail_stability"]["cue_polarities"]
    native_tail = native["blank_tail_stability"]["cue_polarities"]
    blank_tail_recurrent_dynamics_equal = (
        len(candidate_tail) == len(native_tail) == 2
        and all(
            candidate_row["cue_input"] == native_row["cue_input"]
            and candidate_row["native_recurrent_dynamics_sha256"]
            == native_row["native_recurrent_dynamics_sha256"]
            for candidate_row, native_row in zip(candidate_tail, native_tail, strict=True)
        )
    )
    recurrent_activity_equal = (
        recurrent_activity_equal and blank_tail_recurrent_dynamics_equal
    )
    condition["paired_vs_native"] = {
        "eval_cue_to_noise_feature_ratio_multiple": _safe_ratio(
            candidate_ratio, native_ratio
        ),
        "eval_cue_to_noise_feature_ratio_difference": _paired_difference(
            candidate_ratio, native_ratio
        ),
        "eval_absolute_cue_feature_delta_rms_multiple": _safe_ratio(
            condition["eval"]["cue_feature_delta_rms"],
            native["eval"]["cue_feature_delta_rms"],
        ),
        "ridge_eval_accuracy_difference": float(
            condition["ridge_readout"]["eval_accuracy"]
            - native["ridge_readout"]["eval_accuracy"]
        ),
        "eval_output_cue_delta_to_sd_difference": _paired_difference(
            candidate_output, native_output
        ),
        "mean_event_count_inflation": _safe_ratio(candidate_events, native_events),
        "recurrent_activity_equals_native": recurrent_activity_equal,
        "blank_tail_recurrent_dynamics_equal": blank_tail_recurrent_dynamics_equal,
        "all_unit_events_activations_and_legacy_ledgers_equal_native": (
            recurrent_activity_equal
        ),
        "activity_by_split": split_activity,
        "matched_episode_rows": matched_episode_rows,
        "matched_episode_rows_sha256": _sha256_json(matched_episode_rows),
    }
    condition["invariants"]["recurrent_activity_equals_native"] = recurrent_activity_equal
    condition["seed_gate"] = _seed_gate(condition, eligible=gate_eligible)


def _run_seed(
    *, seed: int, retentions: tuple[float, ...], pairs_per_split: int, ridge_alpha: float
) -> dict[str, Any]:
    require_seed_role(seed, SeedRole.CUSTOM)
    stream_pair = generate_stream_pair(
        seed, episode_count=pairs_per_split, noise_events=NOISE_EVENTS,
        required_role=SeedRole.CUSTOM,
    )
    native_graph = build_experiment_000_graph(
        seed, mode="full", event_log_enabled=False, **FROZEN_GRAPH_OPTIONS
    )
    conditions = [
        _run_condition(
            native_graph=native_graph, retention=retention,
            stream_pair=stream_pair, ridge_alpha=ridge_alpha,
        )
        for retention in (None, *retentions)
    ]
    native = conditions[0]
    native_matched_rows = {
        split: [
            {
                "pair_index": pair_row["pair_index"],
                "base_episode_id": pair_row["base_episode_id"],
                "noise_stream_id": pair_row["noise_stream_id"],
                "cue": cue,
                "emitted_event_ratio": 1.0,
                "forward_edge_touch_ratio": 1.0,
                "native_recurrent_dynamics_bitwise_equal": True,
                "unit_events_sha256": pair_row[f"cue_{cue}_unit_events_sha256"],
                "activation_dynamics_sha256": pair_row[
                    f"cue_{cue}_activation_dynamics_sha256"
                ],
                "recurrent_dynamics_sha256": pair_row[
                    f"cue_{cue}_recurrent_dynamics_sha256"
                ],
                "legacy_ledger_delta_sha256": pair_row[
                    f"cue_{cue}_legacy_ledger_delta_sha256"
                ],
                "output_activation_difference": 0.0,
            }
            for pair_row in native["pair_rows"][split]
            for cue in (0, 1)
        ]
        for split in ("train", "eval")
    }
    native["paired_vs_native"] = {
        "eval_cue_to_noise_feature_ratio_multiple": 1.0,
        "eval_cue_to_noise_feature_ratio_difference": 0.0,
        "eval_absolute_cue_feature_delta_rms_multiple": 1.0,
        "ridge_eval_accuracy_difference": 0.0,
        "eval_output_cue_delta_to_sd_difference": 0.0,
        "mean_event_count_inflation": 1.0,
        "recurrent_activity_equals_native": True,
        "blank_tail_recurrent_dynamics_equal": True,
        "all_unit_events_activations_and_legacy_ledgers_equal_native": True,
        "activity_by_split": {
            split: {
                "total_emitted_event_ratio": 1.0,
                "maximum_matched_episode_event_count_ratio": 1.0,
                "forward_edge_touch_ratio": 1.0,
                "maximum_matched_episode_forward_edge_touch_ratio": 1.0,
            }
            for split in ("train", "eval")
        },
        "matched_episode_rows": native_matched_rows,
        "matched_episode_rows_sha256": _sha256_json(native_matched_rows),
    }
    native["seed_gate"] = {"eligible": False, "pass": False, "criteria": {}}
    for condition in conditions[1:]:
        _attach_native_comparison(
            condition,
            native,
            gate_eligible=pairs_per_split == FULL_PAIRS_PER_SPLIT,
        )
    return {
        "seed": seed,
        "seed_role": SeedRole.CUSTOM.value,
        "data_manifest": stream_pair.manifest.to_dict(),
        "condition_results": conditions,
    }


def _run_seed_job(
    job: tuple[int, tuple[float, ...], int, float],
) -> dict[str, Any]:
    seed, retentions, pairs_per_split, ridge_alpha = job
    with threadpool_limits(limits=1, user_api="blas"):
        blas_state = _single_thread_blas_state()
        result = _run_seed(
            seed=seed,
            retentions=retentions,
            pairs_per_split=pairs_per_split,
            ridge_alpha=ridge_alpha,
        )
    result["worker_backend"] = {
        "all_blas_pools_single_threaded": True,
        "numeric_library_threads_per_worker": 1,
    }
    result["_nondeterministic_worker_backend"] = {
        "process_id": os.getpid(),
        "process_instance_token": _PROCESS_INSTANCE_TOKEN,
        "blas_pool_count": len(blas_state),
        "blas_runtime": blas_state,
    }
    return result


def _ordered_seed_results(
    *,
    seeds: tuple[int, ...],
    retentions: tuple[float, ...],
    pairs_per_split: int,
    ridge_alpha: float,
    workers: int,
) -> list[dict[str, Any]]:
    """Run independent seeds and always return the requested deterministic order."""

    if workers not in {1, ORDERED_CPU_WORKERS}:
        raise ValueError("worker count must be one (parity only) or five")
    jobs = [
        (seed, retentions, pairs_per_split, ridge_alpha)
        for seed in seeds
    ]
    if workers == 1:
        results = [_run_seed_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=ORDERED_CPU_WORKERS,
            mp_context=multiprocessing.get_context("spawn"),
            max_tasks_per_child=1,
        ) as executor:
            results = list(executor.map(_run_seed_job, jobs, chunksize=1))
    if tuple(result["seed"] for result in results) != seeds:
        raise RuntimeError("worker results left the frozen seed order")
    return results


def _aggregate_conditions(
    seed_results: Sequence[Mapping[str, Any]],
    retentions: tuple[float, ...],
    *,
    pairs_per_split: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    condition_ids = tuple(condition_name(value) for value in (None, *retentions))
    by_condition: dict[str, list[Mapping[str, Any]]] = {c: [] for c in condition_ids}
    for seed_result in seed_results:
        rows = seed_result["condition_results"]
        if tuple(row["condition"] for row in rows) != condition_ids:
            raise RuntimeError("seed result condition order differs from the freeze")
        for row in rows:
            by_condition[row["condition"]].append(row)
    for condition, rows in by_condition.items():
        if len(rows) != len(seed_results):
            raise RuntimeError(f"condition {condition} missing from a seed")

    native_rows = by_condition[NATIVE_CONDITION]
    native_ratio_distribution = _distribution(
        e["eval"]["cue_to_noise_feature_ratio"] for e in native_rows
    )
    native_absolute_distribution = _distribution(
        e["eval"]["cue_feature_delta_rms"] for e in native_rows
    )
    native_ratio_median = native_ratio_distribution["median"]
    native_absolute_median = native_absolute_distribution["median"]
    native_control_valid = all(
        all(e["invariants"].values())
        and e["blank_tail_stability"]["all_tail_state_wake_gates_pass"]
        for e in native_rows
    )

    aggregate_rows: list[dict[str, Any]] = []
    for condition in condition_ids:
        rows = by_condition[condition]
        distributions = {
            "eval_cue_to_noise_feature_ratio": _distribution(
                e["eval"]["cue_to_noise_feature_ratio"] for e in rows
            ),
            "eval_cue_feature_delta_rms": _distribution(
                e["eval"]["cue_feature_delta_rms"] for e in rows
            ),
            "eval_noise_feature_rms": _distribution(
                e["eval"]["noise_feature_rms"] for e in rows
            ),
            "eval_output_cue_delta_to_sd": _distribution(
                e["eval"]["output_cue_delta_to_sd"] for e in rows
            ),
            "ridge_train_accuracy": _distribution(
                e["ridge_readout"]["train_accuracy"] for e in rows
            ),
            "ridge_eval_accuracy": _distribution(
                e["ridge_readout"]["eval_accuracy"] for e in rows
            ),
            "task_mean_emitted_unit_events": _distribution(
                e["activity"]["combined"]["mean_emitted_unit_events_per_forward_episode"]
                for e in rows
            ),
            "task_hidden_emission_density": _distribution(
                e["activity"]["combined"]["hidden_emission_density"] for e in rows
            ),
            "blank_tail_hidden_emission_density": _distribution(
                e["blank_tail_stability"]["hidden_emission_density"] for e in rows
            ),
        }
        all_invariants = all(all(e["invariants"].values()) for e in rows)
        all_tail = all(
            e["blank_tail_stability"]["all_tail_state_wake_gates_pass"] for e in rows
        )
        aggregate: dict[str, Any] = {
            "condition": condition,
            "target_retention": rows[0]["target_retention"],
            "seed_count": len(rows),
            "distributions": distributions,
            "all_invariants": all_invariants,
            "all_tail_effective_state_and_wake_gates": all_tail,
            "all_recurrent_activity_equals_native": all(
                e["invariants"]["recurrent_activity_equals_native"] for e in rows
            ),
        }
        if condition == NATIVE_CONDITION:
            aggregate["gate"] = {"eligible": False, "pass": False, "criteria": {}}
            aggregate["paired_vs_native"] = {
                "median_cue_ratio_multiple": 1.0,
                "median_absolute_cue_delta_multiple": 1.0,
                "within_seed_cue_ratio_threefold_count": 0,
                "within_seed_absolute_cue_delta_threefold_count": 0,
            }
        else:
            cue_multiples = [
                e["paired_vs_native"]["eval_cue_to_noise_feature_ratio_multiple"]
                for e in rows
            ]
            abs_multiples = [
                e["paired_vs_native"]["eval_absolute_cue_feature_delta_rms_multiple"]
                for e in rows
            ]
            cue_threefold = sum(
                v is not None
                and v >= GATE_THRESHOLDS["paired_native_cue_ratio_multiple_minimum"]
                for v in cue_multiples
            )
            abs_threefold = sum(
                v is not None
                and v
                >= GATE_THRESHOLDS["paired_native_absolute_cue_delta_multiple_minimum"]
                for v in abs_multiples
            )
            all_activity_guards = all(
                e["seed_gate"]["criteria"]["every_split_activity_guard"] for e in rows
            )
            median_ratio = distributions["eval_cue_to_noise_feature_ratio"]["median"]
            median_absolute = distributions["eval_cue_feature_delta_rms"]["median"]
            cue_multiple_distribution = _distribution(cue_multiples)
            absolute_multiple_distribution = _distribution(abs_multiples)
            paired = {
                "cue_ratio_multiple_distribution": cue_multiple_distribution,
                "absolute_cue_delta_multiple_distribution": (
                    absolute_multiple_distribution
                ),
                "ridge_eval_accuracy_difference_distribution": _distribution(
                    e["paired_vs_native"]["ridge_eval_accuracy_difference"]
                    for e in rows
                ),
                "output_cue_delta_to_sd_difference_distribution": _distribution(
                    e["paired_vs_native"][
                        "eval_output_cue_delta_to_sd_difference"
                    ]
                    for e in rows
                ),
                "mean_event_count_inflation_distribution": _distribution(
                    e["paired_vs_native"]["mean_event_count_inflation"]
                    for e in rows
                ),
                "median_cue_ratio_multiple": _safe_ratio(
                    median_ratio, native_ratio_median
                ),
                "median_absolute_cue_delta_multiple": _safe_ratio(
                    median_absolute, native_absolute_median
                ),
                "within_seed_cue_ratio_threefold_count": cue_threefold,
                "within_seed_absolute_cue_delta_threefold_count": abs_threefold,
                "all_seed_split_activity_guards": all_activity_guards,
                "maximum_split_total_event_ratio": max(
                    e["paired_vs_native"]["activity_by_split"][s][
                        "total_emitted_event_ratio"
                    ]
                    for e in rows
                    for s in ("train", "eval")
                ),
                "maximum_matched_episode_event_ratio": max(
                    e["paired_vs_native"]["activity_by_split"][s][
                        "maximum_matched_episode_event_count_ratio"
                    ]
                    for e in rows
                    for s in ("train", "eval")
                ),
            }
            ratio_values_all_defined = bool(
                distributions["eval_cue_to_noise_feature_ratio"][
                    "all_values_defined"
                ]
            )
            output_values_all_defined = bool(
                distributions["eval_output_cue_delta_to_sd"]["all_values_defined"]
            )
            native_ratio_values_all_defined = bool(
                native_ratio_distribution["all_values_defined"]
            )
            criteria = {
                "full_100_pair_run": pairs_per_split == FULL_PAIRS_PER_SPLIT,
                "native_control_valid": native_control_valid,
                "median_absolute_cue_delta_at_least_3x_median_native": (
                    paired["median_absolute_cue_delta_multiple"] is not None
                    and paired["median_absolute_cue_delta_multiple"]
                    >= GATE_THRESHOLDS["paired_native_absolute_cue_delta_multiple_minimum"]
                ),
                "at_least_4_of_5_seeds_absolute_cue_delta_at_least_3x_native": (
                    absolute_multiple_distribution["all_values_defined"]
                    and abs_threefold
                    >= int(GATE_THRESHOLDS["within_seed_threefold_improvement_count_minimum"])
                ),
                "median_eval_cue_ratio_at_least_0_50": (
                    ratio_values_all_defined
                    and median_ratio is not None
                    and median_ratio
                    >= GATE_THRESHOLDS["eval_cue_to_noise_feature_ratio_minimum"]
                ),
                "median_eval_cue_ratio_at_least_3x_median_native": (
                    ratio_values_all_defined
                    and native_ratio_values_all_defined
                    and paired["median_cue_ratio_multiple"] is not None
                    and paired["median_cue_ratio_multiple"]
                    >= GATE_THRESHOLDS["paired_native_cue_ratio_multiple_minimum"]
                ),
                "at_least_4_of_5_seeds_cue_ratio_at_least_3x_native": (
                    cue_multiple_distribution["all_values_defined"]
                    and cue_threefold
                    >= int(GATE_THRESHOLDS["within_seed_threefold_improvement_count_minimum"])
                ),
                "median_ridge_eval_accuracy_at_least_0_70": (
                    distributions["ridge_eval_accuracy"]["median"] is not None
                    and distributions["ridge_eval_accuracy"]["median"]
                    >= GATE_THRESHOLDS["ridge_eval_accuracy_minimum"]
                ),
                "median_eval_output_ratio_at_least_0_50": (
                    output_values_all_defined
                    and distributions["eval_output_cue_delta_to_sd"]["median"]
                    is not None
                    and distributions["eval_output_cue_delta_to_sd"]["median"]
                    >= GATE_THRESHOLDS["eval_output_cue_delta_to_sd_minimum"]
                ),
                "every_seed_split_activity_guard_passes": all_activity_guards,
                "tail_effective_state_and_wake_pass_for_every_seed": all_tail,
                "all_frozen_state_and_finite_invariants_pass": all_invariants,
            }
            aggregate["paired_vs_native"] = paired
            eligible = pairs_per_split == FULL_PAIRS_PER_SPLIT
            aggregate["gate"] = {
                "eligible": eligible,
                "criteria": criteria,
                "pass": eligible and all(criteria.values()),
            }
        aggregate_rows.append(aggregate)

    passing = sorted(
        (
            row
            for row in aggregate_rows
            if row["condition"] != NATIVE_CONDITION
            and row["gate"]["eligible"]
            and row["gate"]["pass"]
            and all(row["gate"]["criteria"].values())
        ),
        key=lambda row: float(row["target_retention"]),
    )
    selected = (
        passing[0]
        if pairs_per_split == FULL_PAIRS_PER_SPLIT
        and passing
        and native_control_valid
        else None
    )
    provisional_terminal_status = (
        "INVALID_NATIVE_CONTROL"
        if not native_control_valid
        else (f"SELECTED:{selected['condition']}" if selected else "NO_SELECTION")
    )
    report_status = (
        "PENDING_DETERMINISM_VERIFICATION"
        if pairs_per_split == FULL_PAIRS_PER_SPLIT
        else provisional_terminal_status
    )
    strongest_gate_reached = _strongest_retention_gate(aggregate_rows)
    selection = {
        "rule": "lowest retention satisfying every preregistered full-run gate",
        "final_status": report_status,
        "provisional_terminal_status": (
            provisional_terminal_status
            if pairs_per_split == FULL_PAIRS_PER_SPLIT
            else None
        ),
        "selected_condition": None if selected is None else selected["condition"],
        "selected_retention": None if selected is None else selected["target_retention"],
        "passing_conditions": [row["condition"] for row in passing],
        "any_target_passed": bool(passing),
        "strongest_retention_gate_reached": strongest_gate_reached,
        "smoke_selection_forbidden": pairs_per_split != FULL_PAIRS_PER_SPLIT,
    }
    if pairs_per_split != FULL_PAIRS_PER_SPLIT and (
        selection["selected_condition"] is not None
        or selection["passing_conditions"]
        or selection["any_target_passed"]
    ):
        raise RuntimeError("smoke selection invariant failed")
    return aggregate_rows, selection


_RETENTION_GATE_ORDER = (
    "median_absolute_cue_delta_at_least_3x_median_native",
    "at_least_4_of_5_seeds_absolute_cue_delta_at_least_3x_native",
    "median_eval_cue_ratio_at_least_0_50",
    "median_eval_cue_ratio_at_least_3x_median_native",
    "at_least_4_of_5_seeds_cue_ratio_at_least_3x_native",
    "median_ridge_eval_accuracy_at_least_0_70",
    "median_eval_output_ratio_at_least_0_50",
)


def _strongest_retention_gate(aggregate_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Identify the strongest candidate on the retention gates 1--7.

    Ranked by count of retention gates passed, then by higher retention
    coefficient. Reports every retention gate's pass/fail for that candidate and
    the continuous ridge, cue/noise, absolute-cue, and output values, so the
    next Gate-A decision (gated event memory vs. gain vs. abandon) is grounded in
    what actually fell short.
    """

    best_row: Mapping[str, Any] | None = None
    best_key: tuple[int, float] = (-1, -1.0)
    for row in aggregate_rows:
        if row["condition"] == NATIVE_CONDITION:
            continue
        criteria = row["gate"].get("criteria", {})
        passed = sum(bool(criteria.get(name)) for name in _RETENTION_GATE_ORDER)
        key = (passed, float(row["target_retention"]))
        if key > best_key:
            best_key = key
            best_row = row
    if best_row is None:
        return {"condition": None, "retention_gates_passed": 0, "of": len(_RETENTION_GATE_ORDER)}
    criteria = best_row["gate"].get("criteria", {})
    distributions = best_row["distributions"]
    paired = best_row["paired_vs_native"]
    return {
        "condition": best_row["condition"],
        "target_retention": best_row["target_retention"],
        "retention_gates_passed": best_key[0],
        "of": len(_RETENTION_GATE_ORDER),
        "retention_gate_results": {
            name: bool(criteria.get(name)) for name in _RETENTION_GATE_ORDER
        },
        "median_ridge_eval_accuracy": distributions["ridge_eval_accuracy"]["median"],
        "median_eval_cue_to_noise_ratio": distributions[
            "eval_cue_to_noise_feature_ratio"
        ]["median"],
        "median_eval_cue_to_noise_ratio_multiple": paired["median_cue_ratio_multiple"],
        "median_absolute_cue_delta_multiple": paired[
            "median_absolute_cue_delta_multiple"
        ],
        "median_eval_output_cue_delta_to_sd": distributions[
            "eval_output_cue_delta_to_sd"
        ]["median"],
        "all_stability_and_activity_gates_pass": bool(
            criteria.get("every_seed_split_activity_guard_passes")
            and criteria.get("tail_effective_state_and_wake_pass_for_every_seed")
            and criteria.get("all_frozen_state_and_finite_invariants_pass")
        ),
    }


# ----------------------------------------------------------------------
# rho=0 graph equivalence and determinism


def _query_result_from_forced_event(graph: RecurrentEventGraph) -> QueryResult:
    event_id = graph.audit["latest_output_event_id"]
    if not isinstance(event_id, str):
        raise TypeError("manual equivalence query did not force an output")
    event = graph.unit_events_by_id[event_id]
    return QueryResult(
        tick=event.step,
        event_id=event.event_id,
        activation=event.activation,
        prediction=1 if event.activation >= 0.0 else 0,
    )


def run_zero_retention_equivalence() -> dict[str, Any]:
    """Compare native and rho=0 at every official-smoke-example step."""

    current_manifest = _source_provenance()
    _validate_freeze_record(current_manifest=current_manifest)
    _validate_phase_prefix(_PHASE_ORDER[:2])
    rows: list[dict[str, Any]] = []
    compared_steps = 0
    public_episodes = 0
    for seed in REGISTERED_SEEDS:
        pair = generate_stream_pair(
            seed,
            episode_count=SMOKE_PAIRS_PER_SPLIT,
            noise_events=NOISE_EVENTS,
            required_role=SeedRole.CUSTOM,
        )
        native = build_experiment_000_graph(
            seed, mode="full", event_log_enabled=False, **FROZEN_GRAPH_OPTIONS
        )
        zero = clone_with_readout_trace(native, 0.0, event_log_enabled=False)
        for stream in (pair.train, pair.eval):
            for base in stream.episodes:
                for cue in (0, 1):
                    episode = _counterfactual_episode(
                        base, cue=cue, condition="rho_0_equivalence"
                    )
                    native.begin_episode(episode.episode_id)
                    zero.begin_episode(episode.episode_id)
                    visible_inputs = [
                        {"cue": 1.0 if cue else -1.0},
                        *(
                            {"noise": 1.0 if bit else -1.0}
                            for bit in episode.noise
                        ),
                    ]
                    for inputs in visible_inputs:
                        native_step = native.step(inputs)
                        zero_step = zero.step(inputs)
                        if native_step != zero_step:
                            raise RuntimeError("rho=0 differs at a visible step")
                        compared_steps += 1
                    for inputs, force_output in (
                        ({"query": 1.0}, False),
                        ({}, False),
                        ({}, True),
                    ):
                        native_step = native._advance(
                            inputs, force_output=force_output
                        )
                        zero_step = zero._advance(inputs, force_output=force_output)
                        if native_step != zero_step:
                            raise RuntimeError("rho=0 differs inside query path")
                        compared_steps += 1
                    native_query = _query_result_from_forced_event(native)
                    zero_query = _query_result_from_forced_event(zero)
                    native_features = extract_query_features(native, native_query)
                    zero_features = extract_query_features(zero, zero_query)
                    if (
                        native_query != zero_query
                        or native_features != zero_features
                        or native.unit_events != zero.unit_events
                        or native.ledger != zero.ledger
                    ):
                        raise RuntimeError("rho=0 manual event/readout mismatch")

                    native_public = forward_episode(native, episode)
                    zero_public = forward_episode(zero, episode)
                    if (
                        native_public != zero_public
                        or extract_query_features(native, native_public)
                        != extract_query_features(zero, zero_public)
                        or native.unit_events != zero.unit_events
                        or native.ledger != zero.ledger
                    ):
                        raise RuntimeError("rho=0 public path differs from native")
                    public_episodes += 1
                    trace_ledger = zero.trace_ledger
                    if any(trace_ledger[field] for field in _TRACE_TOUCH_FIELDS):
                        raise RuntimeError("rho=0 fast path touched local trace")
                    rows.append(
                        {
                            "seed": seed,
                            "split": stream.split,
                            "base_episode_id": base.episode_id,
                            "cue": cue,
                            "query_activation": native_public.activation,
                            "native_feature_sha256": _array_bundle_sha256(
                                [value for _, value in native_features]
                            ),
                            "legacy_ledger_sha256": _sha256_json(native.ledger),
                        }
                    )
    return {
        "pass": True,
        "seeds": list(REGISTERED_SEEDS),
        "base_pairs_per_split": SMOKE_PAIRS_PER_SPLIT,
        "compared_counterfactual_episodes": len(rows),
        "compared_step_result_count": compared_steps,
        "compared_public_episode_count": public_episodes,
        "step_results_bitwise_equal": True,
        "intermediate_activation_tuples_bitwise_equal": True,
        "emitted_events_equal": True,
        "query_results_bitwise_equal": True,
        "native_readout_features_equal": True,
        "native_forced_outputs_equal": True,
        "unit_events_bitwise_equal": True,
        "legacy_ledgers_equal": True,
        "all_trace_touches_zero": True,
        "rows_sha256": _sha256_json(rows),
    }


_REQUIRED_REPORT_INTEGRITY = (
    "all_seeds_custom_and_registered",
    "all_conditions_registered_in_order",
    "all_topologies_match_native",
    "all_structural_masks_match_native",
    "all_frozen_snapshots_match_native",
    "all_weight_kinds_match_native",
    "all_feature_reconstructions_pass",
    "all_output_reconstructions_pass",
    "all_recurrent_activity_equals_native",
    "all_credit_and_update_counts_zero",
    "all_finite",
    "all_trace_bounds_ok",
    "all_source_hashes_match",
    "source_manifests_match_freeze",
    "zero_retention_equivalence_pass",
    "native_controls_are_literal_recurrent_event_graphs",
)


def _paired_split_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_dimension: int,
    trace64: bool,
) -> _PairedSplit:
    key = "trace64_vector" if trace64 else "feature_vector"
    zero_features: list[np.ndarray] = []
    one_features: list[np.ndarray] = []
    zero_outputs: list[float] = []
    one_outputs: list[float] = []
    for row in rows:
        for cue, target in ((0, zero_features), (1, one_features)):
            raw = row.get(f"cue_{cue}_{key}")
            if not isinstance(raw, list) or len(raw) != feature_dimension:
                raise RuntimeError(f"raw {feature_dimension}-D {key} is incomplete")
            vector = np.asarray(raw, dtype=np.float64)
            if vector.shape != (feature_dimension,) or not np.all(np.isfinite(vector)):
                raise RuntimeError(f"raw {key} is non-finite or malformed")
            if row.get(f"cue_{cue}_{key}_sha256") != _array_bundle_sha256(vector):
                raise RuntimeError(f"raw {key} hash is invalid")
            target.append(vector)
        zero_output = float(row["cue_0_output_activation"])
        one_output = float(row["cue_1_output_activation"])
        if not math.isfinite(zero_output) or not math.isfinite(one_output):
            raise RuntimeError("raw output evidence is non-finite")
        zero_outputs.append(zero_output)
        one_outputs.append(one_output)
    return _PairedSplit(
        cue_zero_features=np.stack(zero_features),
        cue_one_features=np.stack(one_features),
        cue_zero_outputs=np.asarray(zero_outputs, dtype=np.float64),
        cue_one_outputs=np.asarray(one_outputs, dtype=np.float64),
    )


def _expected_activity_from_rows(
    rows: Sequence[Mapping[str, Any]], *, trace_active: bool
) -> dict[str, Any]:
    event_counts = [
        int(row[f"cue_{cue}_emitted_unit_events"])
        for row in rows
        for cue in (0, 1)
    ]
    hidden_counts = [
        int(row[f"cue_{cue}_hidden_unit_events"])
        for row in rows
        for cue in (0, 1)
    ]
    activation_counts = [
        int(row[f"cue_{cue}_activation_evaluations"])
        for row in rows
        for cue in (0, 1)
    ]
    hidden_activation_counts = [
        int(row[f"cue_{cue}_hidden_activation_evaluations"])
        for row in rows
        for cue in (0, 1)
    ]
    episodes = len(event_counts)
    hidden_opportunities = episodes * 64 * (NOISE_EVENTS + 3)
    total_hidden_activations = sum(hidden_activation_counts)
    saturation = {
        f"abs_ge_{threshold}": {
            "count": int(
                sum(
                    row[f"cue_{cue}_hidden_activation_saturation_counts"][
                        f"abs_ge_{threshold}"
                    ]
                    for row in rows
                    for cue in (0, 1)
                )
            ),
            "fraction": 0.0,
        }
        for threshold in SATURATION_THRESHOLDS
    }
    for value in saturation.values():
        value["fraction"] = float(value["count"] / total_hidden_activations)
    return {
        "counterfactual_forward_episodes": episodes,
        "total_emitted_unit_events": int(sum(event_counts)),
        "mean_emitted_unit_events_per_forward_episode": float(
            sum(event_counts) / episodes
        ),
        "minimum_emitted_unit_events_per_forward_episode": min(event_counts),
        "maximum_emitted_unit_events_per_forward_episode": max(event_counts),
        "event_counts_sha256": _array_bundle_sha256(event_counts),
        "hidden_event_counts_sha256": _array_bundle_sha256(hidden_counts),
        "pair_rows_sha256": _sha256_json(rows),
        "forward_edge_touches": int(
            sum(
                row[f"cue_{cue}_forward_edge_touches"]
                for row in rows
                for cue in (0, 1)
            )
        ),
        "total_hidden_unit_events": int(sum(hidden_counts)),
        "hidden_emission_opportunities": hidden_opportunities,
        "hidden_emission_density": float(sum(hidden_counts) / hidden_opportunities),
        "activation_evaluations": int(sum(activation_counts)),
        "hidden_activation_evaluations": int(total_hidden_activations),
        "hidden_activation_counts_sha256": _array_bundle_sha256(
            hidden_activation_counts
        ),
        "maximum_absolute_hidden_preactivation": max(
            float(row[f"cue_{cue}_maximum_absolute_hidden_preactivation"])
            for row in rows
            for cue in (0, 1)
        ),
        "hidden_activation_saturation": saturation,
        "local_trace_touches": {
            field: int(
                sum(
                    row[f"cue_{cue}_{field}"]
                    for row in rows
                    for cue in (0, 1)
                )
            )
            for field in _TRACE_TOUCH_FIELDS
        },
        "hidden_activation_evaluations_delta": (
            int(total_hidden_activations) if trace_active else 0
        ),
        "nonfinite_values": 0,
        "credit_event_touches": 0,
        "weight_write_touches": 0,
    }


def _validate_blank_tail_evidence(
    condition: Mapping[str, Any], feature_edge_ids: tuple[str, ...], weights: Mapping[str, float]
) -> None:
    tail = condition.get("blank_tail_stability")
    if not isinstance(tail, Mapping):
        raise TypeError("blank-tail evidence is absent")
    rows = tail.get("cue_polarities")
    if not isinstance(rows, list) or [row.get("cue_input") for row in rows] != [-1.0, 1.0]:
        raise RuntimeError("blank-tail cue rows are invalid")
    for row in rows:
        counts = row.get("hidden_emissions_per_blank_tick")
        if (
            not isinstance(counts, list)
            or len(counts) != BLANK_TAIL_TICKS
            or any(type(value) is not int or value < 0 or value > 64 for value in counts)
            or not isinstance(row.get("native_recurrent_step_rows_sha256"), str)
            or len(row["native_recurrent_step_rows_sha256"]) != 64
            or not isinstance(row.get("native_unit_events_sha256"), str)
            or len(row["native_unit_events_sha256"]) != 64
        ):
            raise RuntimeError("blank-tail compact trajectory commitment is incomplete")
        if row.get("hidden_counts_sha256") != _array_bundle_sha256(counts):
            raise RuntimeError("blank-tail hidden activity commitment is invalid")
        last_nonzero = next(
            (index for index in range(BLANK_TAIL_TICKS, 0, -1) if counts[index - 1]),
            None,
        )
        effective = float(row["effective_trace_maximum_absolute"])
        wake_feature = np.asarray(row.get("wake_feature"), dtype=np.float64)
        weighted_sum = math.fsum(
            weights[edge_id] * float(value)
            for edge_id, value in zip(feature_edge_ids, wake_feature, strict=True)
        )
        if (
            wake_feature.shape != (16,)
            or not np.all(np.isfinite(wake_feature))
            or row.get("wake_feature_sha256") != _array_bundle_sha256(wake_feature)
            or abs(float(row["wake_output_activation"]) - math.tanh(weighted_sum))
            > MATRIX_ABS_TOLERANCE
            + MATRIX_REL_TOLERANCE
            * max(abs(float(row["wake_output_activation"])), abs(math.tanh(weighted_sum)))
            or row.get("total_hidden_emissions") != sum(counts)
            or row.get("final_window_hidden_emissions")
            != sum(counts[-BLANK_TAIL_FINAL_WINDOW_TICKS:])
            or row.get("final_window_quiescent")
            is not (sum(counts[-BLANK_TAIL_FINAL_WINDOW_TICKS:]) == 0)
            or row.get("last_nonzero_hidden_blank_tick") != last_nonzero
            or row.get("effective_trace_gate_pass")
            is not (effective <= GATE_THRESHOLDS["effective_state_maximum_absolute"])
            or row.get("native_legacy_ledger_delta_sha256")
            != _sha256_json(row.get("native_legacy_ledger_delta"))
            or row.get("native_recurrent_dynamics_sha256")
            != _sha256_json(
                {
                    "step_rows_sha256": row.get("native_recurrent_step_rows_sha256"),
                    "unit_events_sha256": row.get("native_unit_events_sha256"),
                    "legacy_ledger_delta": row.get("native_legacy_ledger_delta"),
                }
            )
            or row.get("weights_unchanged") is not True
            or row.get("structural_unchanged") is not True
        ):
            raise RuntimeError("blank-tail derived evidence is incoherent")
    negative = np.asarray(rows[0]["wake_feature"], dtype=np.float64)
    positive = np.asarray(rows[1]["wake_feature"], dtype=np.float64)
    wake_l2 = float(np.linalg.norm(positive - negative) / 2.0)
    wake_output = float(
        abs(rows[1]["wake_output_activation"] - rows[0]["wake_output_activation"])
        / 2.0
    )
    all_gates = (
        all(
            row["final_window_quiescent"] and row["effective_trace_gate_pass"]
            for row in rows
        )
        and wake_l2 <= GATE_THRESHOLDS["wake_feature_half_difference_l2_maximum"]
        and wake_output
        <= GATE_THRESHOLDS["wake_output_half_difference_absolute_maximum"]
    )
    if (
        tail.get("polarity_bundle_sha256") != _sha256_json(rows)
        or tail.get("wake_feature_half_difference_l2") != wake_l2
        or tail.get("wake_output_half_difference_absolute") != wake_output
        or tail.get("all_tail_state_wake_gates_pass") is not all_gates
    ):
        raise RuntimeError("blank-tail aggregate evidence is incoherent")


def _validate_selection_coherence(report: Mapping[str, Any]) -> None:
    run_kind = report.get("run_kind")
    eligible = run_kind == "full"
    expected_conditions = [NATIVE_CONDITION] + [
        condition_name(value) for value in REGISTERED_RETENTIONS
    ]
    aggregates = report.get("aggregate_conditions")
    if not isinstance(aggregates, list) or [
        row.get("condition") for row in aggregates
    ] != expected_conditions:
        raise RuntimeError("aggregate condition order is invalid")
    passing: list[Mapping[str, Any]] = []
    for row in aggregates[1:]:
        gate = row.get("gate")
        if not isinstance(gate, Mapping) or not isinstance(
            gate.get("criteria"), Mapping
        ):
            raise TypeError("candidate gate object is incomplete")
        recomputed_pass = eligible and all(
            value is True for value in gate["criteria"].values()
        )
        if gate.get("eligible") is not eligible or gate.get("pass") is not recomputed_pass:
            raise RuntimeError("candidate gate eligibility/pass is incoherent")
        if recomputed_pass:
            passing.append(row)
    passing.sort(key=lambda row: float(row["target_retention"]))
    native_valid = report.get("integrity", {}).get("native_control_valid") is True
    selected = passing[0] if passing and native_valid and eligible else None
    expected_status = (
        "INVALID_NATIVE_CONTROL"
        if not native_valid
        else (f"SELECTED:{selected['condition']}" if selected else "NO_SELECTION")
    )
    expected_report_status = (
        "PENDING_DETERMINISM_VERIFICATION" if eligible else expected_status
    )
    expected_provisional = expected_status if eligible else None
    selection = report.get("selection")
    if not isinstance(selection, Mapping):
        raise TypeError("selection object is absent")
    if (
        selection.get("passing_conditions")
        != [row["condition"] for row in passing]
        or selection.get("selected_condition")
        != (None if selected is None else selected["condition"])
        or selection.get("selected_retention")
        != (None if selected is None else selected["target_retention"])
        or selection.get("any_target_passed") is not bool(passing)
        or selection.get("final_status") != expected_report_status
        or selection.get("provisional_terminal_status") != expected_provisional
        or report.get("status") != expected_report_status
    ):
        raise RuntimeError("selection/status fields are incoherent")
    if not eligible and (
        passing
        or selection.get("selected_condition") is not None
        or selection.get("passing_conditions") != []
    ):
        raise RuntimeError("smoke report selected a condition")


def _validate_report_mapping(
    report: Mapping[str, Any],
    *,
    expected_kind: str,
    freeze_info: Mapping[str, Any] | None = None,
    current_manifest: Mapping[str, Any] | None = None,
) -> None:
    """Validate a report under the frozen single-thread parent BLAS contract."""

    with threadpool_limits(limits=1, user_api="blas"):
        _single_thread_blas_state()
        _validate_report_mapping_inner(
            report,
            expected_kind=expected_kind,
            freeze_info=freeze_info,
            current_manifest=current_manifest,
        )


def _validate_report_mapping_inner(
    report: Mapping[str, Any],
    *,
    expected_kind: str,
    freeze_info: Mapping[str, Any] | None = None,
    current_manifest: Mapping[str, Any] | None = None,
) -> None:
    if expected_kind not in {"smoke", "full"}:
        raise ValueError("expected report kind must be 'smoke' or 'full'")
    expected_conditions = [NATIVE_CONDITION] + [
        condition_name(value) for value in REGISTERED_RETENTIONS
    ]
    expected_pairs = (
        SMOKE_PAIRS_PER_SPLIT if expected_kind == "smoke" else FULL_PAIRS_PER_SPLIT
    )
    expected_backend = {
        "engine": "numpy_cpu",
        "float_type": "float64",
        "gpu_used": False,
        "numeric_library_threads_per_worker": 1,
    }
    expected_size_limit = (
        MAX_SMOKE_REPORT_BYTES if expected_kind == "smoke" else MAX_FULL_REPORT_BYTES
    )
    expected_compact_contract = {
        "version": COMPACT_EVIDENCE_VERSION,
        "full_recurrent_dynamics_trees_persisted": False,
        "full_blank_tail_step_trees_persisted": False,
        "independent_episode_regeneration_required": True,
        "comparison_granularity": "one_compact_pair_row_at_a_time",
        "serialized_report_size_limit_bytes": expected_size_limit,
    }
    expected_seed_partition = {
        "registered_order": list(REGISTERED_SEEDS),
        "required_role": SeedRole.CUSTOM.value,
        "forbidden": {name: list(values) for name, values in FORBIDDEN_SEEDS.items()},
    }
    expected_formulas = {
        "trace_update": "m_i <- rho**(t-kappa_i) * m_i + (1-rho) * a_i(t)",
        "candidate_feature": "z_e = rho**(t_out - kappa_j(e)) * m_j(e)",
        "candidate_forced_output": "o = tanh(sum_e w_e * z_e)",
        "cue_to_noise_feature_ratio": (
            "R = sqrt(mean(||(z1-z0)/2||^2)) / "
            "sqrt(mean(||(z1+z0)/2 - mean_midpoint||^2))"
        ),
    }
    seeds = report.get("seeds")
    retentions = report.get("retention_coefficients")
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("protocol_name") != PROTOCOL_NAME
        or report.get("protocol_version") != PROTOCOL_VERSION
        or report.get("run_kind") != expected_kind
        or seeds != list(REGISTERED_SEEDS)
        or not isinstance(seeds, list)
        or any(type(seed) is not int for seed in seeds)
        or report.get("conditions") != expected_conditions
        or retentions != list(REGISTERED_RETENTIONS)
        or not isinstance(retentions, list)
        or any(type(retention) is not float for retention in retentions)
        or report.get("pairs_per_split") != expected_pairs
        or type(report.get("pairs_per_split")) is not int
        or report.get("ridge_alpha") != DEFAULT_RIDGE_ALPHA
        or type(report.get("ridge_alpha")) is not float
        or report.get("noise_events") != NOISE_EVENTS
        or type(report.get("noise_events")) is not int
        or report.get("worker_count") != ORDERED_CPU_WORKERS
        or type(report.get("worker_count")) is not int
        or report.get("ordered_parallel_execution") is not True
        or report.get("backend_contract") != expected_backend
        or report.get("compact_evidence_contract") != expected_compact_contract
        or _serialized_json_size(report) > expected_size_limit
        or report.get("confirmatory_executed") is not False
        or report.get("confirmatory_seed_count") != 0
        or type(report.get("confirmatory_seed_count")) is not int
        or report.get("counterfactual_forward_episodes_per_seed_per_condition")
        != 4 * expected_pairs
        or type(
            report.get("counterfactual_forward_episodes_per_seed_per_condition")
        )
        is not int
        or report.get("seed_partition") != expected_seed_partition
        or report.get("matrix_absolute_tolerance") != MATRIX_ABS_TOLERANCE
        or report.get("matrix_relative_tolerance") != MATRIX_REL_TOLERANCE
        or report.get("scope")
        != "custom_seed_architecture_diagnostic_procedural_recovery"
        or report.get("mechanism") != "CCF-v0 present but never called"
        or report.get("formulas") != expected_formulas
        or report.get("status_is_provisional_until_identical_full_rerun")
        is not (expected_kind == "full")
        or report.get("determinism_verification_status")
        != (
            "PENDING_IDENTICAL_FULL_RERUN"
            if expected_kind == "full"
            else "NOT_APPLICABLE_TO_SMOKE"
        )
        or report.get("gate_thresholds") != dict(GATE_THRESHOLDS)
        or report.get("graph_options") != dict(FROZEN_GRAPH_OPTIONS)
    ):
        raise RuntimeError("report frozen metadata is invalid")
    provenance = report.get("nondeterministic_provenance")
    worker_runtime = (
        provenance.get("worker_backend_runtime")
        if isinstance(provenance, Mapping)
        else None
    )
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("excluded_from_deterministic_payload_sha256") is not True
        or not isinstance(provenance.get("runtime_seconds"), (int, float))
        or isinstance(provenance.get("runtime_seconds"), bool)
        or not math.isfinite(float(provenance["runtime_seconds"]))
        or float(provenance["runtime_seconds"]) < 0.0
        or not isinstance(provenance.get("backend"), Mapping)
        or provenance["backend"].get("float_type") != "float64"
        or provenance["backend"].get("autograd_used") is not False
        or provenance["backend"].get("gpu_used") is not False
        or provenance["backend"].get("worker_count") != ORDERED_CPU_WORKERS
        or type(provenance.get("process_id")) is not int
        or provenance["process_id"] <= 0
        or not isinstance(provenance.get("process_instance_token"), str)
        or len(provenance["process_instance_token"]) != 64
        or not isinstance(provenance.get("run_started_utc"), str)
        or not isinstance(provenance.get("parent_blas_runtime"), list)
        or not provenance["parent_blas_runtime"]
        or any(
            not isinstance(pool, Mapping) or pool.get("num_threads") != 1
            for pool in provenance["parent_blas_runtime"]
        )
        or not isinstance(worker_runtime, list)
        or any(not isinstance(row, Mapping) for row in worker_runtime)
        or [row.get("seed") for row in worker_runtime] != list(REGISTERED_SEEDS)
        or any(type(row.get("seed")) is not int for row in worker_runtime)
        or any(type(row.get("process_id")) is not int for row in worker_runtime)
        or any(
            not isinstance(row.get("process_instance_token"), str)
            or len(row["process_instance_token"]) != 64
            for row in worker_runtime
        )
        or len({row["process_id"] for row in worker_runtime})
        != len(REGISTERED_SEEDS)
        or len({row["process_instance_token"] for row in worker_runtime})
        != len(REGISTERED_SEEDS)
        or any(
            not isinstance(row, Mapping)
            or not isinstance(row.get("blas_runtime"), list)
            or row.get("blas_pool_count") != len(row["blas_runtime"])
            or row.get("blas_pool_count", 0) < 1
            or any(
                not isinstance(pool, Mapping) or pool.get("num_threads") != 1
                for pool in row["blas_runtime"]
            )
            for row in worker_runtime
        )
    ):
        raise RuntimeError("report non-deterministic runtime provenance is invalid")
    try:
        run_started = datetime.fromisoformat(str(provenance["run_started_utc"]))
    except ValueError as exc:
        raise RuntimeError("report run-start timestamp is invalid") from exc
    if (
        not str(provenance["run_started_utc"]).endswith("Z")
        or run_started.utcoffset() != UTC.utcoffset(run_started)
    ):
        raise RuntimeError("report run-start timestamp is not canonical UTC")
    stored_hash = report.get("deterministic_payload_sha256")
    if not isinstance(stored_hash, str) or stored_hash != _recompute_report_payload_hash(
        report
    ):
        raise RuntimeError("report deterministic self-hash is invalid")
    source_integrity = report.get("frozen_source_integrity")
    if source_integrity != frozen_source_integrity():
        raise RuntimeError("report frozen source integrity is invalid")
    integrity = report.get("integrity")
    if not isinstance(integrity, Mapping) or not all(
        integrity.get(name) is True for name in _REQUIRED_REPORT_INTEGRITY
    ):
        raise RuntimeError("report implementation/global integrity is invalid")
    starts = report.get("source_manifest_start")
    ends = report.get("source_manifest_end")
    if not isinstance(starts, Mapping) or starts != ends:
        raise RuntimeError("report start/end source manifests differ")
    if freeze_info is not None:
        frozen_record = freeze_info["record"]
        if (
            report.get("freeze_record_sha256") != freeze_info["file_sha256"]
            or starts != frozen_record.get("source_manifest")
            or report.get("frozen_source_manifest_sha256")
            != frozen_record.get("source_manifest_sha256")
        ):
            raise RuntimeError("report is not bound to the current freeze")
    if current_manifest is not None and starts != current_manifest:
        raise RuntimeError("report source manifest differs from current source")
    prerequisite_binding = report.get("phase_prerequisite_binding")
    expected_binding_keys = {
        "freeze_record_sha256",
        "source_manifest_sha256",
        "pre_freeze_verification_sha256",
    }
    if expected_kind == "full":
        expected_binding_keys.update(
            {
                "smoke_report_sha256",
                "smoke_sidecar_sha256",
                "smoke_verification_file_sha256",
                "smoke_verification_payload_sha256",
            }
        )
    if (
        not isinstance(prerequisite_binding, Mapping)
        or set(prerequisite_binding) != expected_binding_keys
        or prerequisite_binding.get("freeze_record_sha256")
        != report.get("freeze_record_sha256")
        or prerequisite_binding.get("source_manifest_sha256")
        != starts.get("bundle_sha256")
        or any(
            not isinstance(prerequisite_binding.get(key), str)
            or len(prerequisite_binding[key]) != 64
            for key in expected_binding_keys
        )
    ):
        raise RuntimeError("report phase-prerequisite binding is invalid")
    if freeze_info is not None:
        expected_binding: dict[str, Any] = {
            "freeze_record_sha256": freeze_info["file_sha256"],
            "source_manifest_sha256": starts["bundle_sha256"],
            "pre_freeze_verification_sha256": freeze_info["record"][
                "pre_freeze_verification"
            ]["file_sha256"],
        }
        if expected_kind == "full":
            _, smoke_hashes = _read_report_with_valid_sidecar(
                DEFAULT_SMOKE_REPORT_PATH
            )
            verification_path = _project_path(SMOKE_VERIFICATION_PATH)
            if not verification_path.is_file():
                raise RuntimeError("A3 smoke verification is absent")
            smoke_verification = json.loads(
                verification_path.read_text(encoding="utf-8")
            )
            expected_binding.update(
                {
                    "smoke_report_sha256": smoke_hashes["report_file_sha256"],
                    "smoke_sidecar_sha256": smoke_hashes["sidecar_file_sha256"],
                    "smoke_verification_file_sha256": _file_sha256(
                        verification_path
                    ),
                    "smoke_verification_payload_sha256": smoke_verification.get(
                        "verification_payload_sha256"
                    ),
                }
            )
        if dict(prerequisite_binding) != expected_binding:
            raise RuntimeError("report phase artifacts differ from current prerequisites")
    seed_results = report.get("seed_results")
    if (
        not isinstance(seed_results, list)
        or any(not isinstance(row, Mapping) for row in seed_results)
        or [row.get("seed") for row in seed_results] != list(REGISTERED_SEEDS)
        or any(type(row.get("seed")) is not int for row in seed_results)
    ):
        raise RuntimeError("report seed result order is invalid")
    for seed_result in seed_results:
        if "_nondeterministic_worker_backend" in seed_result:
            raise RuntimeError("worker runtime leaked into the deterministic payload")
        seed = int(seed_result["seed"])
        stream_pair = generate_stream_pair(
            seed,
            episode_count=expected_pairs,
            noise_events=NOISE_EVENTS,
            required_role=SeedRole.CUSTOM,
        )
        if (
            seed_result.get("seed_role") != SeedRole.CUSTOM.value
            or seed_result.get("data_manifest") != stream_pair.manifest.to_dict()
        ):
            raise RuntimeError("report data manifest is not reproducible")
        worker_backend = seed_result.get("worker_backend")
        if (
            worker_backend
            != {
                "all_blas_pools_single_threaded": True,
                "numeric_library_threads_per_worker": 1,
            }
        ):
            raise RuntimeError("seed worker did not prove single-threaded BLAS")
        expected_graph = build_experiment_000_graph(
            seed, mode="full", event_log_enabled=False, **FROZEN_GRAPH_OPTIONS
        )
        expected_feature_ids = _output_feature_edge_ids(expected_graph)
        output_source_by_edge = {
            edge.edge_id: edge.source
            for edge in expected_graph.edges
            if edge.kind == "output" and edge.destination == expected_graph.output_node
        }
        hidden_index_by_node = {
            node: hidden_index
            for hidden_index, node in enumerate(expected_graph.hidden_nodes)
        }
        if (
            tuple(sorted(output_source_by_edge)) != expected_feature_ids
            or any(
                output_source_by_edge[edge_id] not in hidden_index_by_node
                for edge_id in expected_feature_ids
            )
        ):
            raise RuntimeError("registered output-edge source view is invalid")
        expected_trace64_view_indices = tuple(
            hidden_index_by_node[output_source_by_edge[edge_id]]
            for edge_id in expected_feature_ids
        )
        expected_snapshot = _frozen_graph_snapshot(expected_graph)
        conditions = seed_result.get("condition_results")
        if not isinstance(conditions, list) or [
            row.get("condition") for row in conditions
        ] != expected_conditions:
            raise RuntimeError("report per-seed condition order is invalid")
        for index, condition in enumerate(conditions):
            expected_retention = None if index == 0 else REGISTERED_RETENTIONS[index - 1]
            replay_native = build_experiment_000_graph(
                seed, mode="full", event_log_enabled=False, **FROZEN_GRAPH_OPTIONS
            )
            replay_graph: RecurrentEventGraph = (
                replay_native
                if index == 0
                else clone_with_readout_trace(
                    replay_native, float(expected_retention), event_log_enabled=False
                )
            )
            replay_native_diagnostic = (
                clone_with_readout_trace(
                    replay_native, 0.0, event_log_enabled=False
                )
                if index == 0
                else None
            )
            replay_feature_ids = _output_feature_edge_ids(replay_graph)
            graph = condition.get("graph", {})
            snapshots = graph.get("frozen_state_snapshots")
            invariants = condition.get("invariants")
            if (
                not isinstance(invariants, Mapping)
                or set(invariants) != _CONDITION_INVARIANT_KEYS
                or any(value is not True for value in invariants.values())
            ):
                raise RuntimeError("condition invariant schema or value is invalid")
            if (
                condition.get("target_retention") != expected_retention
                or (
                    expected_retention is not None
                    and type(condition.get("target_retention")) is not float
                )
                or condition.get("retention_coefficient")
                != (0.0 if expected_retention is None else expected_retention)
                or type(condition.get("retention_coefficient")) is not float
                or condition.get("retention_half_life_ticks")
                != (
                    None
                    if expected_retention is None
                    else float(math.log(0.5) / math.log(expected_retention))
                )
                or condition.get("retention_after_ten_ticks")
                != (
                    0.0
                    if expected_retention is None
                    else float(expected_retention**10)
                )
                or condition.get("data_pair_sha256") != stream_pair.manifest.pair_sha256
                or graph.get("feature_dimension") != 16
                or graph.get("feature_edge_ids") != list(expected_feature_ids)
                or graph.get("hidden_units") != 64
                or graph.get("topology_sha256") != expected_graph.topology_hash()
                or graph.get("structural_mask_sha256")
                != _structural_mask_sha256(expected_graph)
                or graph.get("initial_weights_sha256") != expected_graph.weights_hash()
                or graph.get("final_weights_sha256") != expected_graph.weights_hash()
                or graph.get("recurrent_matrix") != _recurrent_matrix_metrics(expected_graph)
                or graph.get("hyperparameters")
                != _constructor_hyperparameters(expected_graph)
                or graph.get("native_reference_snapshot") != expected_snapshot
                or not isinstance(snapshots, Mapping)
                or set(snapshots)
                != {
                    "before_train",
                    "after_train",
                    "after_eval",
                    "after_blank_tail",
                }
                or any(
                    snapshots[name] != expected_snapshot
                    for name in (
                        "before_train",
                        "after_train",
                        "after_eval",
                        "after_blank_tail",
                    )
                )
            ):
                raise RuntimeError("report graph/condition identity is invalid")
            if index == 0 and (
                graph.get("implementation_class") != "RecurrentEventGraph"
                or graph.get("native_control_is_literal_recurrent_event_graph") is not True
                or condition.get("invariants", {}).get(
                    "native_control_is_literal_recurrent_event_graph"
                )
                is not True
            ):
                raise RuntimeError("report native control is not the literal native graph")
            if index and graph.get("implementation_class") != "ReadoutTraceRecurrentEventGraph":
                raise RuntimeError("candidate graph did not use the registered trace wrapper")
            rebuilt_splits: dict[str, _PairedSplit] = {}
            for split in ("train", "eval"):
                rows = condition.get("pair_rows", {}).get(split)
                if not isinstance(rows, list) or len(rows) != expected_pairs:
                    raise RuntimeError("report raw pair rows are incomplete")
                expected_episodes = getattr(stream_pair, split).episodes
                if any(
                    row.get("pair_index") != episode.index
                    or row.get("base_episode_id") != episode.episode_id
                    or row.get("noise_stream_id") != episode.noise_stream_id
                    for row, episode in zip(rows, expected_episodes, strict=True)
                ):
                    raise RuntimeError("report pair identities differ from regenerated data")
                replayed_split = _collect_split(
                    replay_graph,
                    getattr(stream_pair, split),
                    condition=str(condition["condition"]),
                    is_native=index == 0,
                    feature_edge_ids=replay_feature_ids,
                    native_diagnostic_graph=replay_native_diagnostic,
                    expected_rows=rows,
                    retain_rows=False,
                )
                if replayed_split["activity"] != condition.get("activity", {}).get(split):
                    raise RuntimeError(
                        "stored compact activity differs from independently replayed episodes"
                    )
                for row in rows:
                    for cue in (0, 1):
                        if f"cue_{cue}_recurrent_dynamics" in row:
                            raise RuntimeError(
                                "A3 report retained a forbidden recurrent-dynamics tree"
                            )
                        dynamics_commitment = row.get(
                            f"cue_{cue}_recurrent_dynamics_sha256"
                        )
                        if (
                            not isinstance(dynamics_commitment, str)
                            or len(dynamics_commitment) != 64
                        ):
                            raise RuntimeError("compact recurrent commitment is invalid")
                        vector = np.asarray(
                            row.get(f"cue_{cue}_feature_vector"), dtype=np.float64
                        )
                        if index > 0:
                            raw_trace64 = row.get(f"cue_{cue}_trace64_vector")
                            if not isinstance(raw_trace64, list) or len(raw_trace64) != 64:
                                raise RuntimeError(
                                    "raw 64-D trace snapshot is incomplete"
                                )
                            trace64_vector = np.asarray(raw_trace64, dtype=np.float64)
                            if trace64_vector.shape != (64,) or not np.all(
                                np.isfinite(trace64_vector)
                            ):
                                raise RuntimeError(
                                    "raw 64-D trace snapshot is non-finite or malformed"
                                )
                            expected_selecting_view = trace64_vector[
                                list(expected_trace64_view_indices)
                            ]
                            if np.asarray(vector, dtype="<f8").tobytes() != np.asarray(
                                expected_selecting_view, dtype="<f8"
                            ).tobytes():
                                raise RuntimeError(
                                    "selecting 16-D features are not the output-edge "
                                    "source view of the same 64-D trace snapshot"
                                )
                        weighted_sum = math.fsum(
                            expected_graph.weights[edge_id] * float(value)
                            for edge_id, value in zip(
                                expected_feature_ids, vector, strict=True
                            )
                        )
                        expected_output = math.tanh(weighted_sum)
                        output = float(row[f"cue_{cue}_output_activation"])
                        expected_readout_preactivation = (
                            float(row[f"cue_{cue}_native_query_preactivation"])
                            if index == 0
                            else weighted_sum
                        )
                        feature_error = abs(
                            weighted_sum - expected_readout_preactivation
                        )
                        feature_tolerance = (
                            MATRIX_ABS_TOLERANCE
                            + MATRIX_REL_TOLERANCE
                            * max(
                                abs(weighted_sum),
                                abs(expected_readout_preactivation),
                            )
                        )
                        output_error = abs(output - expected_output)
                        output_tolerance = (
                            MATRIX_ABS_TOLERANCE
                            + MATRIX_REL_TOLERANCE
                            * max(abs(output), abs(expected_output))
                        )
                        if (
                            vector.shape != (16,)
                            or row.get(f"cue_{cue}_weighted_feature_sum") != weighted_sum
                            or row.get(f"cue_{cue}_readout_preactivation")
                            != expected_readout_preactivation
                            or row.get(f"cue_{cue}_feature_reconstruction_error")
                            != feature_error
                            or row.get(f"cue_{cue}_feature_reconstruction_tolerance")
                            != feature_tolerance
                            or row.get(f"cue_{cue}_feature_reconstruction_pass")
                            is not (feature_error <= feature_tolerance)
                            or row.get(f"cue_{cue}_output_reconstruction_error")
                            != output_error
                            or row.get(f"cue_{cue}_output_reconstruction_tolerance")
                            != output_tolerance
                            or row.get(f"cue_{cue}_output_reconstruction_pass")
                            is not (output_error <= output_tolerance)
                        ):
                            raise RuntimeError("raw feature/output reconstruction is invalid")
                rebuilt = _paired_split_from_rows(
                    rows, feature_dimension=16, trace64=False
                )
                rebuilt_splits[split] = rebuilt
                if condition.get(split) != paired_retention_metrics(rebuilt):
                    raise RuntimeError("stored retention metrics differ from raw features")
                expected_activity = _expected_activity_from_rows(
                    rows, trace_active=index > 0
                )
                if condition.get("activity", {}).get(split) != expected_activity:
                    raise RuntimeError("stored activity differs from raw rows")
            if condition.get("pair_rows", {}).get("bundle_sha256") != _sha256_json(
                [condition["pair_rows"]["train"], condition["pair_rows"]["eval"]]
            ):
                raise RuntimeError("pair-row bundle hash is invalid")
            expected_ridge = strict_ridge_readout(
                rebuilt_splits["train"],
                rebuilt_splits["eval"],
                alpha=DEFAULT_RIDGE_ALPHA,
            )
            if condition.get("ridge_readout") != expected_ridge:
                raise RuntimeError("stored ridge result differs from raw features")
            if condition.get("activity", {}).get("combined") != _combine_activity(
                condition["activity"]["train"], condition["activity"]["eval"]
            ):
                raise RuntimeError("combined activity is not reproducible")
            replayed_tail = _blank_tail_stability(
                replay_graph,
                condition=str(condition["condition"]),
                is_native=index == 0,
                feature_edge_ids=replay_feature_ids,
            )
            if replayed_tail != condition.get("blank_tail_stability"):
                raise RuntimeError(
                    "stored compact blank-tail evidence differs from independent replay"
                )
            _validate_blank_tail_evidence(
                condition, expected_feature_ids, expected_graph.weights
            )
            condition_ledger = condition.get("ledger")
            trace_ledger = (
                condition_ledger.get("trace")
                if isinstance(condition_ledger, Mapping)
                else None
            )
            expected_trace_touches = {
                field: sum(
                    int(row[f"cue_{cue}_{field}"])
                    for split in ("train", "eval")
                    for row in condition["pair_rows"][split]
                    for cue in (0, 1)
                )
                + sum(
                    int(row["trace_touch_deltas"][field])
                    for row in condition["blank_tail_stability"]["cue_polarities"]
                )
                for field in _TRACE_TOUCH_FIELDS
            }
            if (
                not isinstance(condition_ledger, Mapping)
                or set(condition_ledger)
                != {"credit_event_touches", "weight_write_touches", "nonfinite_values", "trace"}
                or any(
                    condition_ledger.get(name) != 0
                    for name in (
                        "credit_event_touches",
                        "weight_write_touches",
                        "nonfinite_values",
                    )
                )
                or not isinstance(trace_ledger, Mapping)
                or set(trace_ledger)
                != {"hidden_activation_evaluations", *_TRACE_TOUCH_FIELDS}
                or any(
                    trace_ledger.get(field) != expected_trace_touches[field]
                    for field in _TRACE_TOUCH_FIELDS
                )
                or (
                    index == 0
                    and trace_ledger.get("hidden_activation_evaluations") != 0
                )
                or (
                    index > 0
                    and trace_ledger.get("hidden_activation_evaluations")
                    != expected_trace_touches["local_trace_read_touches"]
                )
            ):
                raise RuntimeError("condition ledger/trace evidence is incoherent")
            diagnostic = condition.get("trace64_diagnostic_only")
            if not isinstance(diagnostic, Mapping) or diagnostic.get("selecting") is not False:
                raise RuntimeError("64-D diagnostic declaration is absent")
            if index and (
                diagnostic.get("applicable") is not True
                or diagnostic.get("feature_dimension") != 64
                or diagnostic.get("hidden_node_order") != list(expected_graph.hidden_nodes)
                or diagnostic.get("ridge_readout", {}).get("feature_count") != 64
            ):
                raise RuntimeError("candidate 64-D diagnostic is incomplete")
            if index:
                trace_train = _paired_split_from_rows(
                    condition["pair_rows"]["train"],
                    feature_dimension=64,
                    trace64=True,
                )
                trace_eval = _paired_split_from_rows(
                    condition["pair_rows"]["eval"],
                    feature_dimension=64,
                    trace64=True,
                )
                if (
                    diagnostic.get("train") != paired_retention_metrics(trace_train)
                    or diagnostic.get("eval") != paired_retention_metrics(trace_eval)
                    or diagnostic.get("ridge_readout")
                    != strict_ridge_readout(
                        trace_train, trace_eval, alpha=DEFAULT_RIDGE_ALPHA
                    )
                ):
                    raise RuntimeError("64-D diagnostic differs from raw trace evidence")
            else:
                expected_native_diagnostic = {
                    "applicable": False,
                    "selecting": False,
                    "reason": "native control has no passive trace",
                    "feature_dimension": 64,
                    "hidden_node_order": list(expected_graph.hidden_nodes),
                }
                if diagnostic != expected_native_diagnostic:
                    raise RuntimeError("native 64-D diagnostic declaration is invalid")
                for split in ("train", "eval"):
                    for row in condition["pair_rows"][split]:
                        if any(
                            row.get(f"cue_{cue}_{field}") is not None
                            for cue in (0, 1)
                            for field in (
                                "trace64_vector",
                                "trace64_vector_sha256",
                            )
                        ):
                            raise RuntimeError("native control contains candidate trace evidence")
        native = conditions[0]
        if native.get("seed_gate") != {"eligible": False, "pass": False, "criteria": {}}:
            raise RuntimeError("native seed gate is not ineligible")
        for condition in conditions[1:]:
            replay = copy.deepcopy(condition)
            _attach_native_comparison(
                replay,
                native,
                gate_eligible=expected_kind == "full",
            )
            if (
                replay["paired_vs_native"] != condition.get("paired_vs_native")
                or replay["seed_gate"] != condition.get("seed_gate")
                or replay["invariants"]["recurrent_activity_equals_native"]
                != condition.get("invariants", {}).get(
                    "recurrent_activity_equals_native"
                )
            ):
                raise RuntimeError("seed comparison/gate differs from raw evidence")
    recomputed_aggregates, recomputed_selection = _aggregate_conditions(
        seed_results,
        REGISTERED_RETENTIONS,
        pairs_per_split=expected_pairs,
    )
    if (
        report.get("aggregate_conditions") != recomputed_aggregates
        or report.get("selection") != recomputed_selection
    ):
        raise RuntimeError("aggregate gates/selection differ from seed evidence")
    native_valid = all(
        all(seed["condition_results"][0]["invariants"].values())
        and seed["condition_results"][0]["blank_tail_stability"][
            "all_tail_state_wake_gates_pass"
        ]
        for seed in seed_results
    )
    all_rows = [
        condition
        for seed_result in seed_results
        for condition in seed_result["condition_results"]
    ]
    zero_equivalence = report.get("zero_retention_native_equivalence")
    expected_integrity = {
        "all_seeds_custom_and_registered": True,
        "confirmatory_seed_count": 0,
        "all_conditions_registered": True,
        "all_conditions_registered_in_order": True,
        "all_topologies_match_native": True,
        "all_structural_masks_match_native": True,
        "all_frozen_snapshots_match_native": True,
        "all_weight_kinds_match_native": True,
        "all_recurrent_activity_equals_native": True,
        "all_feature_reconstructions_pass": True,
        "all_output_reconstructions_pass": True,
        "all_credit_and_update_counts_zero": True,
        "all_finite": True,
        "all_trace_bounds_ok": True,
        "all_tail_effective_state_and_wake_gates": all(
            row["blank_tail_stability"]["all_tail_state_wake_gates_pass"]
            is True
            for row in all_rows
        ),
        "all_source_hashes_match": True,
        "source_manifests_match_freeze": True,
        "native_control_valid": native_valid,
        "native_controls_are_literal_recurrent_event_graphs": True,
        "zero_retention_equivalence_pass": isinstance(zero_equivalence, Mapping)
        and zero_equivalence.get("pass") is True,
    }
    if dict(integrity) != expected_integrity:
        raise RuntimeError("report global integrity is not reproducible")
    if report.get("zero_retention_native_equivalence") != run_zero_retention_equivalence():
        raise RuntimeError("rho=0 equivalence evidence is not reproducible")
    _validate_selection_coherence(report)


def _validate_smoke_report(
    smoke_path: str | Path = DEFAULT_SMOKE_REPORT_PATH,
    *,
    freeze_info: Mapping[str, Any],
    current_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if _project_path(smoke_path).resolve() != _project_path(
        DEFAULT_SMOKE_REPORT_PATH
    ).resolve():
        raise ValueError("A3 smoke validation requires the registered smoke path")
    report, file_hashes = _read_report_with_valid_sidecar(smoke_path)
    _validate_report_mapping(
        report,
        expected_kind="smoke",
        freeze_info=freeze_info,
        current_manifest=current_manifest,
    )
    selection = report.get("selection", {})
    if (
        report.get("integrity", {}).get("native_control_valid") is not True
        or report.get("status") != "NO_SELECTION"
        or selection.get("final_status") != "NO_SELECTION"
        or selection.get("selected_condition") is not None
        or selection.get("passing_conditions") != []
        or selection.get("any_target_passed") is not False
        or selection.get("smoke_selection_forbidden") is not True
    ):
        raise RuntimeError("smoke report is not a valid nonselecting native control")
    return {"report": report, **file_hashes}


def create_smoke_verification(
    smoke_path: str | Path = DEFAULT_SMOKE_REPORT_PATH,
    output_path: str | Path = SMOKE_VERIFICATION_PATH,
) -> dict[str, Any]:
    if _project_path(smoke_path).resolve() != _project_path(
        DEFAULT_SMOKE_REPORT_PATH
    ).resolve() or _project_path(output_path).resolve() != _project_path(
        SMOKE_VERIFICATION_PATH
    ).resolve():
        raise ValueError("A3 smoke verification requires the registered paths")
    _validate_phase_sequence(_PHASE_ORDER[:3])
    current_manifest = _source_provenance()
    freeze_info = _validate_freeze_record(current_manifest=current_manifest)
    validated = _validate_smoke_report(
        smoke_path, freeze_info=freeze_info, current_manifest=current_manifest
    )
    record: dict[str, Any] = {
        "schema_version": "experiment-000-readout-trace-a3-smoke-verification-v1",
        "status": "PASS",
        "all_checks_pass": True,
        "smoke_report_sha256": validated["report_file_sha256"],
        "smoke_sidecar_sha256": validated["sidecar_file_sha256"],
        "smoke_deterministic_payload_sha256": validated["report"][
            "deterministic_payload_sha256"
        ],
        "freeze_record_sha256": freeze_info["file_sha256"],
        "source_manifest_sha256": current_manifest["bundle_sha256"],
    }
    record["verification_payload_sha256"] = _record_self_hash(
        record, "verification_payload_sha256"
    )
    final_manifest = _source_provenance()
    if final_manifest != current_manifest:
        raise RuntimeError("A3 source manifest changed before smoke-verification write")
    _validate_freeze_record(current_manifest=final_manifest)
    _validate_phase_sequence(_PHASE_ORDER[:3])
    _, final_smoke_hashes = _read_report_with_valid_sidecar(smoke_path)
    if final_smoke_hashes != {
        "report_file_sha256": validated["report_file_sha256"],
        "sidecar_file_sha256": validated["sidecar_file_sha256"],
    }:
        raise RuntimeError("smoke artifacts changed before verification write")
    _validate_phase_sequence(_PHASE_ORDER[:3])
    if _source_provenance() != final_manifest:
        raise RuntimeError("A3 source manifest changed at smoke-verification commit")
    _write_json(output_path, record)
    if _source_provenance() != final_manifest:
        raise RuntimeError("A3 source manifest changed during smoke-verification write")
    _record_phase("machine_smoke_verification", (output_path,))
    if _source_provenance() != final_manifest:
        raise RuntimeError(
            "A3 source manifest changed during smoke-verification phase binding"
        )
    return record


def _validate_smoke_verification(
    *,
    smoke_path: str | Path,
    verification_path: str | Path,
    freeze_info: Mapping[str, Any],
    current_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    validated_smoke = _validate_smoke_report(
        smoke_path, freeze_info=freeze_info, current_manifest=current_manifest
    )
    target = _project_path(verification_path)
    if not target.is_file():
        raise RuntimeError("machine smoke verification is absent")
    verification = json.loads(target.read_text(encoding="utf-8"))
    if (
        verification.get("schema_version")
        != "experiment-000-readout-trace-a3-smoke-verification-v1"
        or verification.get("status") != "PASS"
        or verification.get("all_checks_pass") is not True
        or verification.get("verification_payload_sha256")
        != _record_self_hash(verification, "verification_payload_sha256")
        or verification.get("smoke_report_sha256")
        != validated_smoke["report_file_sha256"]
        or verification.get("smoke_sidecar_sha256")
        != validated_smoke["sidecar_file_sha256"]
        or verification.get("smoke_deterministic_payload_sha256")
        != validated_smoke["report"]["deterministic_payload_sha256"]
        or verification.get("freeze_record_sha256") != freeze_info["file_sha256"]
        or verification.get("source_manifest_sha256")
        != current_manifest["bundle_sha256"]
    ):
        raise RuntimeError("machine smoke verification is invalid")
    return verification


def _load_full_input(
    value: str | Path,
    *,
    expected_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(value, Mapping):
        raise TypeError("A3 determinism verification requires persisted report files")
    if _project_path(value).resolve() != _project_path(expected_path).resolve():
        raise ValueError("A3 determinism verification received an unregistered report path")
    return _read_report_with_valid_sidecar(value)


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_canonical_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return (
        parsed.utcoffset() == UTC.utcoffset(parsed)
        and value
        == parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")
    )


def _coordinator_processes_are_fresh(
    left_provenance: Mapping[str, Any], right_provenance: Mapping[str, Any]
) -> bool:
    left_pid = left_provenance.get("process_id")
    right_pid = right_provenance.get("process_id")
    left_token = left_provenance.get("process_instance_token")
    right_token = right_provenance.get("process_instance_token")
    left_started = left_provenance.get("run_started_utc")
    right_started = right_provenance.get("run_started_utc")
    return (
        type(left_pid) is int
        and left_pid > 0
        and type(right_pid) is int
        and right_pid > 0
        and left_pid != right_pid
        and _is_lower_sha256(left_token)
        and _is_lower_sha256(right_token)
        and left_token != right_token
        and _is_canonical_utc_timestamp(left_started)
        and _is_canonical_utc_timestamp(right_started)
        and left_started != right_started
    )


def _require_distinct_rerun_coordinator(
    primary: Mapping[str, Any],
    *,
    current_process_id: int,
    current_process_instance_token: str,
    current_run_started_utc: str,
) -> None:
    primary_provenance = primary.get("nondeterministic_provenance")
    current_provenance = {
        "process_id": current_process_id,
        "process_instance_token": current_process_instance_token,
        "run_started_utc": current_run_started_utc,
    }
    if (
        not isinstance(primary_provenance, Mapping)
        or primary_provenance.get("rerun_prerequisite_artifacts") is not None
        or not _coordinator_processes_are_fresh(
            primary_provenance, current_provenance
        )
    ):
        raise RuntimeError(
            "A3 rerun requires a distinct coordinator PID, process token, "
            "and run-start timestamp"
        )


def _expected_determinism_verification(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    left_files: Mapping[str, Any],
    right_files: Mapping[str, Any],
    freeze_info: Mapping[str, Any],
    current_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    left_hash = left.get("deterministic_payload_sha256")
    right_hash = right.get("deterministic_payload_sha256")
    left_recomputed = _recompute_report_payload_hash(left)
    right_recomputed = _recompute_report_payload_hash(right)
    left_self_hash_valid = _is_lower_sha256(left_hash) and left_hash == left_recomputed
    right_self_hash_valid = (
        _is_lower_sha256(right_hash) and right_hash == right_recomputed
    )
    hashes_match = left_hash == right_hash
    canonical_payloads_match = _sha256_json(
        _deterministic_payload_object(left)
    ) == _sha256_json(_deterministic_payload_object(right)) and (
        _deterministic_payload_object(left) == _deterministic_payload_object(right)
    )
    selections_match = left.get("selection") == right.get("selection")
    statuses_match = left.get("status") == right.get("status")
    left_provenance = left.get("nondeterministic_provenance")
    right_provenance = right.get("nondeterministic_provenance")
    process_provenance_present = isinstance(left_provenance, Mapping) and isinstance(
        right_provenance, Mapping
    )
    fresh_processes = process_provenance_present and _coordinator_processes_are_fresh(
        left_provenance, right_provenance
    )
    expected_rerun_binding = {
        "primary_report_sha256": left_files.get("report_file_sha256"),
        "primary_sidecar_sha256": left_files.get("sidecar_file_sha256"),
    }
    primary_prerequisite_verified = bool(
        process_provenance_present
        and left_provenance.get("rerun_prerequisite_artifacts") is None
        and right_provenance.get("rerun_prerequisite_artifacts")
        == expected_rerun_binding
    )
    if not (
        left_self_hash_valid
        and right_self_hash_valid
        and hashes_match
        and canonical_payloads_match
        and selections_match
        and statuses_match
    ):
        status = "NONDETERMINISTIC_INVALID"
    elif not (fresh_processes and primary_prerequisite_verified):
        status = "INVALID_PROCEDURE_NONSELECTING"
    elif not (
        left.get("integrity", {}).get("native_control_valid") is True
        and right.get("integrity", {}).get("native_control_valid") is True
    ):
        status = "INVALID_NATIVE_CONTROL"
    else:
        status = str(left.get("selection", {}).get("provisional_terminal_status"))
    allowed_statuses = {"NO_SELECTION"} | {
        f"SELECTED:{condition_name(value)}" for value in REGISTERED_RETENTIONS
    }
    if status not in allowed_statuses and status not in {
        "NONDETERMINISTIC_INVALID",
        "INVALID_NATIVE_CONTROL",
        "INVALID_PROCEDURE_NONSELECTING",
    }:
        raise RuntimeError("full reports contain an unregistered terminal status")
    verification: dict[str, Any] = {
        "schema_version": "experiment-000-readout-trace-a3-determinism-v1",
        "status": status,
        "deterministic_payloads_match": hashes_match,
        "canonical_deterministic_payloads_equal": canonical_payloads_match,
        "selection_objects_match": selections_match,
        "selection_statuses_match": statuses_match,
        "fresh_processes_verified": fresh_processes,
        "primary_prerequisite_verified": primary_prerequisite_verified,
        "first_process_id": (
            left_provenance.get("process_id") if process_provenance_present else None
        ),
        "second_process_id": (
            right_provenance.get("process_id") if process_provenance_present else None
        ),
        "first_process_instance_token": (
            left_provenance.get("process_instance_token")
            if process_provenance_present
            else None
        ),
        "second_process_instance_token": (
            right_provenance.get("process_instance_token")
            if process_provenance_present
            else None
        ),
        "first_run_started_utc": (
            left_provenance.get("run_started_utc")
            if process_provenance_present
            else None
        ),
        "second_run_started_utc": (
            right_provenance.get("run_started_utc")
            if process_provenance_present
            else None
        ),
        "first_nondeterministic_provenance_sha256": (
            _sha256_json(left_provenance) if process_provenance_present else None
        ),
        "second_nondeterministic_provenance_sha256": (
            _sha256_json(right_provenance) if process_provenance_present else None
        ),
        "first_deterministic_payload_sha256": left_hash,
        "second_deterministic_payload_sha256": right_hash,
        "first_recomputed_payload_sha256": left_recomputed,
        "second_recomputed_payload_sha256": right_recomputed,
        "first_self_hash_valid": left_self_hash_valid,
        "second_self_hash_valid": right_self_hash_valid,
        "first_report_file_sha256": left_files.get("report_file_sha256"),
        "first_sidecar_file_sha256": left_files.get("sidecar_file_sha256"),
        "second_report_file_sha256": right_files.get("report_file_sha256"),
        "second_sidecar_file_sha256": right_files.get("sidecar_file_sha256"),
        "freeze_record_sha256": freeze_info.get("file_sha256"),
        "source_manifest_sha256": current_manifest.get("bundle_sha256"),
        "valid_terminal_status": status in allowed_statuses,
    }
    verification["verification_payload_sha256"] = _record_self_hash(
        verification, "verification_payload_sha256"
    )
    return verification


def _validate_determinism_verification(
    verification: Mapping[str, Any],
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    left_files: Mapping[str, Any],
    right_files: Mapping[str, Any],
    freeze_info: Mapping[str, Any],
    current_manifest: Mapping[str, Any],
    require_selected: bool,
) -> None:
    expected = _expected_determinism_verification(
        left,
        right,
        left_files=left_files,
        right_files=right_files,
        freeze_info=freeze_info,
        current_manifest=current_manifest,
    )
    if not isinstance(verification, dict) or json.dumps(
        verification,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) != json.dumps(
        expected,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ):
        raise RuntimeError("A3 determinism verification is inconsistent with reports")
    if require_selected and not str(verification["status"]).startswith("SELECTED:"):
        raise RuntimeError("A3 determinism verification is not a selected terminal state")


def _verify_deterministic_full_runs_validated(
    first: str | Path = PRIMARY_REPORT_PATH,
    second: str | Path = RERUN_REPORT_PATH,
    *,
    output_path: str | Path = DETERMINISM_VERIFICATION_PATH,
) -> dict[str, Any]:
    """Validate both complete reports before applying the A3 stop rule."""

    if _project_path(first).resolve() == _project_path(second).resolve():
        raise ValueError("primary and rerun reports must be distinct persisted files")
    if _project_path(output_path).resolve() != _project_path(
        DETERMINISM_VERIFICATION_PATH
    ).resolve():
        raise ValueError("A3 determinism verification requires its registered output")
    _validate_phase_sequence(_PHASE_ORDER[:6])
    left, left_files = _load_full_input(first, expected_path=PRIMARY_REPORT_PATH)
    right, right_files = _load_full_input(second, expected_path=RERUN_REPORT_PATH)
    # Reject malformed/tampered payloads before relying on external prerequisites.
    _validate_report_mapping(left, expected_kind="full")
    _validate_report_mapping(right, expected_kind="full")
    current_manifest = _source_provenance()
    freeze_info = _validate_freeze_record(current_manifest=current_manifest)
    _validate_report_mapping(
        left,
        expected_kind="full",
        freeze_info=freeze_info,
        current_manifest=current_manifest,
    )
    _validate_report_mapping(
        right,
        expected_kind="full",
        freeze_info=freeze_info,
        current_manifest=current_manifest,
    )
    expected_rerun_binding = {
        "primary_report_sha256": left_files["report_file_sha256"],
        "primary_sidecar_sha256": left_files["sidecar_file_sha256"],
    }
    if (
        left.get("nondeterministic_provenance", {}).get(
            "rerun_prerequisite_artifacts"
        )
        is not None
        or right.get("nondeterministic_provenance", {}).get(
            "rerun_prerequisite_artifacts"
        )
        != expected_rerun_binding
    ):
        raise RuntimeError("primary/rerun artifact provenance is invalid")
    _validate_smoke_verification(
        smoke_path=DEFAULT_SMOKE_REPORT_PATH,
        verification_path=SMOKE_VERIFICATION_PATH,
        freeze_info=freeze_info,
        current_manifest=current_manifest,
    )
    verification = _expected_determinism_verification(
        left,
        right,
        left_files=left_files,
        right_files=right_files,
        freeze_info=freeze_info,
        current_manifest=current_manifest,
    )
    _validate_determinism_verification(
        verification,
        left,
        right,
        left_files=left_files,
        right_files=right_files,
        freeze_info=freeze_info,
        current_manifest=current_manifest,
        require_selected=False,
    )
    final_manifest = _source_provenance()
    if final_manifest != current_manifest:
        raise RuntimeError("A3 source manifest changed before determinism write")
    _validate_freeze_record(current_manifest=final_manifest)
    _validate_phase_sequence(_PHASE_ORDER[:6])
    final_left, final_left_files = _load_full_input(
        first, expected_path=PRIMARY_REPORT_PATH
    )
    final_right, final_right_files = _load_full_input(
        second, expected_path=RERUN_REPORT_PATH
    )
    if (
        final_left != left
        or final_right != right
        or final_left_files != left_files
        or final_right_files != right_files
    ):
        raise RuntimeError("full report artifacts changed before determinism write")
    _validate_smoke_verification(
        smoke_path=DEFAULT_SMOKE_REPORT_PATH,
        verification_path=SMOKE_VERIFICATION_PATH,
        freeze_info=freeze_info,
        current_manifest=final_manifest,
    )
    _validate_phase_sequence(_PHASE_ORDER[:6])
    if _source_provenance() != final_manifest:
        raise RuntimeError("A3 source manifest changed at determinism commit")
    written_verification = _write_json(output_path, verification)
    persisted_verification = json.loads(
        written_verification.read_text(encoding="utf-8")
    )
    _validate_determinism_verification(
        persisted_verification,
        final_left,
        final_right,
        left_files=final_left_files,
        right_files=final_right_files,
        freeze_info=freeze_info,
        current_manifest=final_manifest,
        require_selected=False,
    )
    if _source_provenance() != final_manifest:
        raise RuntimeError("A3 source manifest changed during determinism write")
    _record_phase("determinism_verification", (output_path,))
    if _source_provenance() != final_manifest:
        raise RuntimeError("A3 source manifest changed during determinism phase binding")
    return verification


def verify_deterministic_full_runs(
    first: str | Path = PRIMARY_REPORT_PATH,
    second: str | Path = RERUN_REPORT_PATH,
    *,
    output_path: str | Path = DETERMINISM_VERIFICATION_PATH,
) -> dict[str, Any]:
    """Issue a terminal decision or a machine-readable nonselecting procedure failure."""

    if _project_path(first).resolve() == _project_path(second).resolve():
        raise ValueError("primary and rerun reports must be distinct persisted files")
    if (
        _project_path(first).resolve() != _project_path(PRIMARY_REPORT_PATH).resolve()
        or _project_path(second).resolve() != _project_path(RERUN_REPORT_PATH).resolve()
    ):
        raise ValueError("A3 determinism verification requires the registered reports")
    if _project_path(output_path).resolve() != _project_path(
        DETERMINISM_VERIFICATION_PATH
    ).resolve():
        raise ValueError("A3 determinism verification requires its registered output")
    try:
        with threadpool_limits(limits=1, user_api="blas"):
            _single_thread_blas_state()
            return _verify_deterministic_full_runs_validated(
                first, second, output_path=output_path
            )
    except (
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        LookupError,
        AttributeError,
        ArithmeticError,
    ) as exc:
        target = _project_path(output_path)
        artifact_will_be_written = not target.exists()
        verification: dict[str, Any] = {
            "schema_version": "experiment-000-readout-trace-a3-determinism-v1",
            "status": "INVALID_PROCEDURE_NONSELECTING",
            "procedure_valid": False,
            "valid_terminal_status": False,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "verification_artifact_written": artifact_will_be_written,
        }
        verification["verification_payload_sha256"] = _record_self_hash(
            verification, "verification_payload_sha256"
        )
        if artifact_will_be_written:
            _write_json(output_path, verification)
        return verification


def load_verified_a3_selection() -> dict[str, Any]:
    """Load one fully replay-validated A3 selection for downstream experiments.

    The loader refuses smoke/provisional/unselected/invalid state.  It revalidates
    the complete phase chain, current source freeze, report sidecars, both full
    reports, deterministic equality record, and every cross-artifact hash before
    returning the selected condition and its registered retention.
    """

    with threadpool_limits(limits=1, user_api="blas"):
        _single_thread_blas_state()
        current_manifest = _source_provenance()
        freeze_info = _validate_freeze_record(current_manifest=current_manifest)
        _validate_phase_sequence(_PHASE_ORDER)
        primary, primary_files = _load_full_input(
            PRIMARY_REPORT_PATH, expected_path=PRIMARY_REPORT_PATH
        )
        rerun, rerun_files = _load_full_input(
            RERUN_REPORT_PATH, expected_path=RERUN_REPORT_PATH
        )
        _validate_report_mapping(
            primary,
            expected_kind="full",
            freeze_info=freeze_info,
            current_manifest=current_manifest,
        )
        _validate_report_mapping(
            rerun,
            expected_kind="full",
            freeze_info=freeze_info,
            current_manifest=current_manifest,
        )

        determinism_path = _project_path(DETERMINISM_VERIFICATION_PATH)
        if not determinism_path.is_file():
            raise RuntimeError("A3 deterministic verification artifact is absent")
        determinism = json.loads(determinism_path.read_text(encoding="utf-8"))
        status = determinism.get("status")
        selected_condition = primary.get("selection", {}).get("selected_condition")
        retention_by_condition = {
            condition_name(value): value for value in REGISTERED_RETENTIONS
        }
        expected_status = (
            f"SELECTED:{selected_condition}"
            if selected_condition in retention_by_condition
            else None
        )
        _validate_determinism_verification(
            determinism,
            primary,
            rerun,
            left_files=primary_files,
            right_files=rerun_files,
            freeze_info=freeze_info,
            current_manifest=current_manifest,
            require_selected=True,
        )
        if status != expected_status:
            raise RuntimeError("A3 selection/determinism bindings are invalid")

        final_manifest = _source_provenance()
        if final_manifest != current_manifest:
            raise RuntimeError("A3 source manifest changed while loading selection")
        _validate_freeze_record(current_manifest=final_manifest)
        _validate_phase_sequence(_PHASE_ORDER)
        return {
            "schema_version": "experiment-000-readout-trace-a3-selection-binding-v1",
            "status": status,
            "selected_condition": selected_condition,
            "retention_coefficient": retention_by_condition[str(selected_condition)],
            "protocol_version": PROTOCOL_VERSION,
            "source_manifest_sha256": current_manifest["bundle_sha256"],
            "freeze_record_sha256": freeze_info["file_sha256"],
            "primary": {
                "path": str(PRIMARY_REPORT_PATH).replace("\\", "/"),
                "report_file_sha256": primary_files["report_file_sha256"],
                "sidecar_file_sha256": primary_files["sidecar_file_sha256"],
                "deterministic_payload_sha256": primary[
                    "deterministic_payload_sha256"
                ],
            },
            "rerun": {
                "path": str(RERUN_REPORT_PATH).replace("\\", "/"),
                "report_file_sha256": rerun_files["report_file_sha256"],
                "sidecar_file_sha256": rerun_files["sidecar_file_sha256"],
                "deterministic_payload_sha256": rerun[
                    "deterministic_payload_sha256"
                ],
            },
            "determinism": {
                "path": str(DETERMINISM_VERIFICATION_PATH).replace("\\", "/"),
                "file_sha256": _file_sha256(determinism_path),
                "verification_payload_sha256": determinism[
                    "verification_payload_sha256"
                ],
            },
        }


# ----------------------------------------------------------------------
# Top-level run


def run_readout_trace_experiment(
    *,
    seeds: Sequence[int] = REGISTERED_SEEDS,
    retentions: Sequence[float] = REGISTERED_RETENTIONS,
    pairs_per_split: int = DEFAULT_PAIRS_PER_SPLIT,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    output_path: str | Path | None = None,
    workers: int = ORDERED_CPU_WORKERS,
    smoke_report_path: str | Path = DEFAULT_SMOKE_REPORT_PATH,
    smoke_verification_path: str | Path = SMOKE_VERIFICATION_PATH,
) -> dict[str, Any]:
    """Run A3 with parent-side numerical work limited to one BLAS thread."""

    with threadpool_limits(limits=1, user_api="blas"):
        _single_thread_blas_state()
        return _run_readout_trace_experiment(
            seeds=seeds,
            retentions=retentions,
            pairs_per_split=pairs_per_split,
            ridge_alpha=ridge_alpha,
            output_path=output_path,
            workers=workers,
            smoke_report_path=smoke_report_path,
            smoke_verification_path=smoke_verification_path,
        )


def _run_readout_trace_experiment(
    *,
    seeds: Sequence[int] = REGISTERED_SEEDS,
    retentions: Sequence[float] = REGISTERED_RETENTIONS,
    pairs_per_split: int = DEFAULT_PAIRS_PER_SPLIT,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    output_path: str | Path | None = None,
    workers: int = ORDERED_CPU_WORKERS,
    smoke_report_path: str | Path = DEFAULT_SMOKE_REPORT_PATH,
    smoke_verification_path: str | Path = SMOKE_VERIFICATION_PATH,
) -> dict[str, Any]:
    started = perf_counter()
    run_started_utc = datetime.now(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )
    parent_blas_runtime = _single_thread_blas_state()
    resolved_seeds, resolved_retentions = _validate_arguments(
        seeds=seeds, retentions=retentions, pairs_per_split=pairs_per_split,
        ridge_alpha=ridge_alpha, workers=workers,
    )
    if output_path is None:
        raise ValueError("A3 official smoke/full execution requires a persisted output")
    target = _project_path(output_path)
    expected_target = (
        _project_path(DEFAULT_SMOKE_REPORT_PATH)
        if pairs_per_split == SMOKE_PAIRS_PER_SPLIT
        else None
    )
    if pairs_per_split == FULL_PAIRS_PER_SPLIT:
        valid_full_targets = {
            _project_path(PRIMARY_REPORT_PATH).resolve(),
            _project_path(RERUN_REPORT_PATH).resolve(),
        }
        if target.resolve() not in valid_full_targets:
            raise ValueError("full output must be the frozen primary or rerun path")
    elif target.resolve() != expected_target.resolve():
        raise ValueError("smoke output must be the frozen persisted smoke path")
    source_integrity = frozen_source_integrity()
    required_checks = (
        "recurrent_source_matches",
        "credit_method_bundle_matches",
        "ccf_document_matches",
        "protocol_matches",
        "config_matches",
    )
    if not all(source_integrity[key] for key in required_checks):
        raise RuntimeError(f"frozen source integrity failed: {source_integrity}")
    source_start = _source_provenance()
    freeze_info = _validate_freeze_record(current_manifest=source_start)
    prerequisite_binding: dict[str, Any] = {
        "freeze_record_sha256": freeze_info["file_sha256"],
        "source_manifest_sha256": source_start["bundle_sha256"],
        "pre_freeze_verification_sha256": freeze_info["record"][
            "pre_freeze_verification"
        ]["file_sha256"],
    }
    rerun_artifact_binding: dict[str, str] | None = None
    if pairs_per_split == SMOKE_PAIRS_PER_SPLIT:
        _validate_phase_sequence(_PHASE_ORDER[:2])
    if pairs_per_split == FULL_PAIRS_PER_SPLIT:
        if _project_path(smoke_report_path).resolve() != _project_path(
            DEFAULT_SMOKE_REPORT_PATH
        ).resolve() or _project_path(smoke_verification_path).resolve() != _project_path(
            SMOKE_VERIFICATION_PATH
        ).resolve():
            raise ValueError("full execution requires the registered A3 smoke artifacts")
        target_is_rerun = target.resolve() == _project_path(RERUN_REPORT_PATH).resolve()
        _validate_phase_sequence(
            _PHASE_ORDER[:5] if target_is_rerun else _PHASE_ORDER[:4]
        )
        smoke_verification = _validate_smoke_verification(
            smoke_path=smoke_report_path,
            verification_path=smoke_verification_path,
            freeze_info=freeze_info,
            current_manifest=source_start,
        )
        _, smoke_hashes = _read_report_with_valid_sidecar(smoke_report_path)
        prerequisite_binding.update(
            {
                "smoke_report_sha256": smoke_hashes["report_file_sha256"],
                "smoke_sidecar_sha256": smoke_hashes["sidecar_file_sha256"],
                "smoke_verification_file_sha256": _file_sha256(
                    _project_path(smoke_verification_path)
                ),
                "smoke_verification_payload_sha256": smoke_verification[
                    "verification_payload_sha256"
                ],
            }
        )
        if target_is_rerun:
            primary, primary_hashes = _read_report_with_valid_sidecar(
                PRIMARY_REPORT_PATH
            )
            _validate_report_mapping(
                primary,
                expected_kind="full",
                freeze_info=freeze_info,
                current_manifest=source_start,
            )
            _require_distinct_rerun_coordinator(
                primary,
                current_process_id=os.getpid(),
                current_process_instance_token=_PROCESS_INSTANCE_TOKEN,
                current_run_started_utc=run_started_utc,
            )
            rerun_artifact_binding = {
                "primary_report_sha256": primary_hashes["report_file_sha256"],
                "primary_sidecar_sha256": primary_hashes["sidecar_file_sha256"],
            }
    zero_equivalence = run_zero_retention_equivalence()

    seed_results = _ordered_seed_results(
        seeds=resolved_seeds,
        retentions=resolved_retentions,
        pairs_per_split=pairs_per_split,
        ridge_alpha=ridge_alpha,
        workers=workers,
    )
    worker_runtime_provenance = [
        {
            "seed": seed_result["seed"],
            **seed_result.pop("_nondeterministic_worker_backend"),
        }
        for seed_result in seed_results
    ]
    aggregate_conditions, selection = _aggregate_conditions(
        seed_results, resolved_retentions, pairs_per_split=pairs_per_split
    )
    all_rows = [
        row for seed_result in seed_results for row in seed_result["condition_results"]
    ]
    source_end = _source_provenance()
    if source_end != source_start or source_end != freeze_info["record"]["source_manifest"]:
        raise RuntimeError("A3 source manifest changed before report write")
    expected_condition_order = [NATIVE_CONDITION] + [
        condition_name(value) for value in resolved_retentions
    ]

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "run_kind": "full" if pairs_per_split == FULL_PAIRS_PER_SPLIT else "smoke",
        "scope": "custom_seed_architecture_diagnostic_procedural_recovery",
        "mechanism": "CCF-v0 present but never called",
        "seeds": list(resolved_seeds),
        "conditions": expected_condition_order,
        "retention_coefficients": list(resolved_retentions),
        "pairs_per_split": pairs_per_split,
        "ridge_alpha": ridge_alpha,
        "noise_events": NOISE_EVENTS,
        "confirmatory_executed": False,
        "confirmatory_seed_count": 0,
        "worker_count": workers,
        "ordered_parallel_execution": True,
        "backend_contract": {
            "engine": "numpy_cpu",
            "float_type": "float64",
            "gpu_used": False,
            "numeric_library_threads_per_worker": 1,
        },
        "compact_evidence_contract": {
            "version": COMPACT_EVIDENCE_VERSION,
            "full_recurrent_dynamics_trees_persisted": False,
            "full_blank_tail_step_trees_persisted": False,
            "independent_episode_regeneration_required": True,
            "comparison_granularity": "one_compact_pair_row_at_a_time",
            "serialized_report_size_limit_bytes": (
                MAX_FULL_REPORT_BYTES
                if pairs_per_split == FULL_PAIRS_PER_SPLIT
                else MAX_SMOKE_REPORT_BYTES
            ),
        },
        "counterfactual_forward_episodes_per_seed_per_condition": 4 * pairs_per_split,
        "seed_partition": {
            "registered_order": list(REGISTERED_SEEDS),
            "required_role": "custom",
            "forbidden": {k: list(v) for k, v in FORBIDDEN_SEEDS.items()},
        },
        "gate_thresholds": dict(GATE_THRESHOLDS),
        "matrix_absolute_tolerance": MATRIX_ABS_TOLERANCE,
        "matrix_relative_tolerance": MATRIX_REL_TOLERANCE,
        "formulas": {
            "trace_update": "m_i <- rho**(t-kappa_i) * m_i + (1-rho) * a_i(t)",
            "candidate_feature": "z_e = rho**(t_out - kappa_j(e)) * m_j(e)",
            "candidate_forced_output": "o = tanh(sum_e w_e * z_e)",
            "cue_to_noise_feature_ratio": (
                "R = sqrt(mean(||(z1-z0)/2||^2)) / "
                "sqrt(mean(||(z1+z0)/2 - mean_midpoint||^2))"
            ),
        },
        "graph_options": dict(FROZEN_GRAPH_OPTIONS),
        "frozen_source_integrity": source_integrity,
        "source_manifest_start": source_start,
        "source_manifest_end": source_end,
        "frozen_source_manifest_sha256": freeze_info["record"][
            "source_manifest_sha256"
        ],
        "freeze_record_sha256": freeze_info["file_sha256"],
        "phase_prerequisite_binding": prerequisite_binding,
        "zero_retention_native_equivalence": zero_equivalence,
        "seed_results": seed_results,
        "aggregate_conditions": aggregate_conditions,
        "selection": selection,
        "status": selection["final_status"],
        "status_is_provisional_until_identical_full_rerun": (
            pairs_per_split == FULL_PAIRS_PER_SPLIT
        ),
        "determinism_verification_status": (
            "PENDING_IDENTICAL_FULL_RERUN"
            if pairs_per_split == FULL_PAIRS_PER_SPLIT
            else "NOT_APPLICABLE_TO_SMOKE"
        ),
        "integrity": {
            "all_seeds_custom_and_registered": all(
                sr["seed"] in REGISTERED_SEEDS and sr["seed_role"] == SeedRole.CUSTOM.value
                for sr in seed_results
            ),
            "confirmatory_seed_count": 0,
            "all_conditions_registered": {row["condition"] for row in all_rows}
            == set([NATIVE_CONDITION] + [condition_name(v) for v in resolved_retentions]),
            "all_conditions_registered_in_order": all(
                [row["condition"] for row in seed_result["condition_results"]]
                == expected_condition_order
                for seed_result in seed_results
            ),
            "all_topologies_match_native": all(
                row["invariants"]["topology_matches_native"] for row in all_rows
            ),
            "all_structural_masks_match_native": all(
                row["invariants"]["structural_mask_matches_native"]
                for row in all_rows
            ),
            "all_frozen_snapshots_match_native": all(
                row["invariants"]["all_frozen_snapshots_equal_native"]
                for row in all_rows
            ),
            "all_weight_kinds_match_native": all(
                row["invariants"]["recurrent_weights_match_native"]
                and row["invariants"]["input_weights_match_native"]
                and row["invariants"]["output_weights_match_native"]
                for row in all_rows
            ),
            "all_recurrent_activity_equals_native": all(
                row["invariants"]["recurrent_activity_equals_native"] for row in all_rows
            ),
            "all_feature_reconstructions_pass": all(
                row["invariants"]["all_feature_reconstructions_pass"]
                for row in all_rows
            ),
            "all_output_reconstructions_pass": all(
                row["invariants"]["all_output_reconstructions_pass"]
                for row in all_rows
            ),
            "all_credit_and_update_counts_zero": all(
                row["invariants"]["credit_event_touches_zero"]
                and row["invariants"]["credit_edge_touches_zero"]
                and row["invariants"]["weight_write_touches_zero"]
                and row["invariants"]["credit_packets_zero"]
                for row in all_rows
            ),
            "all_finite": all(
                row["invariants"]["nonfinite_values_zero"] for row in all_rows
            ),
            "all_trace_bounds_ok": all(
                row["invariants"]["trace_bound_ok"] for row in all_rows
            ),
            "all_tail_effective_state_and_wake_gates": all(
                row["blank_tail_stability"]["all_tail_state_wake_gates_pass"]
                for row in all_rows
            ),
            "all_source_hashes_match": all(
                source_integrity[key] for key in required_checks
            ),
            "source_manifests_match_freeze": source_start
            == source_end
            == freeze_info["record"]["source_manifest"],
            "native_control_valid": all(
                all(sr["condition_results"][0]["invariants"].values())
                and sr["condition_results"][0]["blank_tail_stability"][
                    "all_tail_state_wake_gates_pass"
                ]
                for sr in seed_results
            ),
            "native_controls_are_literal_recurrent_event_graphs": all(
                sr["condition_results"][0]["graph"][
                    "native_control_is_literal_recurrent_event_graph"
                ]
                and sr["condition_results"][0]["graph"]["implementation_class"]
                == "RecurrentEventGraph"
                for sr in seed_results
            ),
            "zero_retention_equivalence_pass": zero_equivalence["pass"],
        },
    }
    report["deterministic_payload_sha256"] = _sha256_json(report)
    report["nondeterministic_provenance"] = {
        "excluded_from_deterministic_payload_sha256": True,
        "process_id": os.getpid(),
        "process_instance_token": _PROCESS_INSTANCE_TOKEN,
        "run_started_utc": run_started_utc,
        "parent_blas_runtime": parent_blas_runtime,
        "rerun_prerequisite_artifacts": rerun_artifact_binding,
        "worker_backend_runtime": worker_runtime_provenance,
        "runtime_seconds": perf_counter() - started,
        "backend": {
            "implementation": platform.python_implementation(),
            "python": platform.python_version(),
            "executable": sys.executable,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "float_type": "float64",
            "autograd_used": False,
            "gpu_used": False,
            "worker_count": workers,
        },
    }
    _validate_report_mapping(
        report,
        expected_kind=report["run_kind"],
        freeze_info=freeze_info,
        current_manifest=source_end,
    )
    final_manifest = _source_provenance()
    if final_manifest != source_end:
        raise RuntimeError("A3 source manifest changed before report write")
    _validate_freeze_record(current_manifest=final_manifest)
    if pairs_per_split == SMOKE_PAIRS_PER_SPLIT:
        _validate_phase_sequence(_PHASE_ORDER[:2])
    elif target.resolve() == _project_path(PRIMARY_REPORT_PATH).resolve():
        _validate_phase_sequence(_PHASE_ORDER[:4])
        _validate_smoke_verification(
            smoke_path=smoke_report_path,
            verification_path=smoke_verification_path,
            freeze_info=freeze_info,
            current_manifest=final_manifest,
        )
    else:
        _validate_phase_sequence(_PHASE_ORDER[:5])
        _validate_smoke_verification(
            smoke_path=smoke_report_path,
            verification_path=smoke_verification_path,
            freeze_info=freeze_info,
            current_manifest=final_manifest,
        )
        final_primary, final_primary_hashes = _read_report_with_valid_sidecar(
            PRIMARY_REPORT_PATH
        )
        _validate_report_mapping(
            final_primary,
            expected_kind="full",
            freeze_info=freeze_info,
            current_manifest=final_manifest,
        )
        if rerun_artifact_binding != {
            "primary_report_sha256": final_primary_hashes["report_file_sha256"],
            "primary_sidecar_sha256": final_primary_hashes["sidecar_file_sha256"],
        }:
            raise RuntimeError("primary artifacts changed before rerun write")
    required_phase = (
        _PHASE_ORDER[:2]
        if pairs_per_split == SMOKE_PAIRS_PER_SPLIT
        else (
            _PHASE_ORDER[:4]
            if target.resolve() == _project_path(PRIMARY_REPORT_PATH).resolve()
            else _PHASE_ORDER[:5]
        )
    )
    _validate_phase_sequence(required_phase)
    if _source_provenance() != final_manifest:
        raise RuntimeError("A3 source manifest changed at report commit")
    written_report, written_sidecar = _write_report_and_sidecar(target, report)
    if _source_provenance() != final_manifest:
        raise RuntimeError("A3 source manifest changed during report persistence")
    if pairs_per_split == SMOKE_PAIRS_PER_SPLIT:
        _record_phase(
            "persisted_smoke_and_sidecar", (written_report, written_sidecar)
        )
    elif target.resolve() == _project_path(PRIMARY_REPORT_PATH).resolve():
        _record_phase("full", (written_report, written_sidecar))
    else:
        _record_phase("rerun", (written_report, written_sidecar))
    if _source_provenance() != final_manifest:
        raise RuntimeError("A3 source manifest changed during report phase binding")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Experiment-000 readout-decoupled local-trace architecture "
            "diagnostic without learning or confirmatory seeds."
        )
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--retentions", type=float, nargs="+", required=True)
    parser.add_argument("--pairs-per-split", type=int, required=True)
    parser.add_argument("--ridge-alpha", type=float, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_readout_trace_experiment(
        seeds=tuple(args.seeds),
        retentions=tuple(args.retentions),
        pairs_per_split=args.pairs_per_split,
        ridge_alpha=args.ridge_alpha,
        output_path=args.output,
        workers=args.workers,
    )
    report_path = _project_path(args.output)
    sidecar_path = report_path.with_suffix(".sha256")
    summary = {
        "run_kind": report["run_kind"],
        "status": report["status"],
        "selection_effect": (
            "SMOKE_INELIGIBLE_NONSELECTING"
            if report["run_kind"] == "smoke"
            else "PROVISIONAL_PENDING_IDENTICAL_RERUN"
        ),
        "report_path": str(report_path),
        "report_file_sha256": _file_sha256(report_path),
        "sidecar_path": str(sidecar_path),
        "sidecar_file_sha256": _file_sha256(sidecar_path),
        "deterministic_payload_sha256": report["deterministic_payload_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FULL_PAIRS_PER_SPLIT",
    "PROTOCOL_VERSION",
    "REGISTERED_RETENTIONS",
    "REGISTERED_SEEDS",
    "SMOKE_PAIRS_PER_SPLIT",
    "clone_with_readout_trace",
    "condition_name",
    "frozen_source_integrity",
    "load_verified_a3_selection",
    "main",
    "run_readout_trace_experiment",
    "run_zero_retention_equivalence",
    "verify_deterministic_full_runs",
]
