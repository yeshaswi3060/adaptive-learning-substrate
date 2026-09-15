"""TRACE-HEAD-L1: exact alignment of a standalone selected-trace head.

The official runner is deliberately downstream of a terminal, independently
verified A3 representation selection. It explicitly rejects the resource-invalid
A2 run and never changes the recurrent graph:
only a separate 16-dimensional ``theta`` vector is updated.  Unit tests use the
explicit scratch entry point, which rejects the official seeds.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import struct
import tomllib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .experiment000_data import SeedRole, generate_stream, require_seed_role
from .experiment_000 import FROZEN_GRAPH_OPTIONS, forward_episode
from .experiment_000_memory_probe import _output_feature_edge_ids
from .readout_trace_recurrent import (
    ReadoutTraceRecurrentEventGraph,
    clone_with_readout_trace,
)
from .recurrent import RecurrentEventGraph, build_experiment_000_graph

SCHEMA_VERSION = "experiment-000-trace-head-l1-v1"
PROTOCOL_NAME = "experiment_000_trace_head_alignment"
PROTOCOL_VERSION = "trace-head-l1-v1"
OFFICIAL_SEEDS: tuple[int, ...] = (75, 76, 77, 78, 79)
EPISODES_PER_SEED = 32
NOISE_EVENTS = 8
FEATURE_DIMENSION = 16
FINITE_DIFFERENCE_EPSILON = 1e-6
ABSOLUTE_TOLERANCE = 1e-8
RELATIVE_TOLERANCE = 1e-6
DIRECTION_COSINE_MINIMUM = 0.999999
LEARNING_RATE = 0.5
ELEMENT_UPDATE_CLIP = 0.05
THETA_CLIP = 3.0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = Path("docs/EXPERIMENT_000_TRACE_HEAD_ALIGNMENT_PROTOCOL.md")
CONFIG_PATH = Path("configs/experiment_000_trace_head_alignment.toml")
ARTIFACT_DIRECTORY = Path("artifacts/experiment_000/trace_head_alignment")
PRIMARY_REPORT_PATH = (
    ARTIFACT_DIRECTORY / "alignment_seeds_75_79_episodes_32.json"
)
RERUN_REPORT_PATH = (
    ARTIFACT_DIRECTORY / "alignment_seeds_75_79_episodes_32_rerun.json"
)
DETERMINISM_VERIFICATION_PATH = ARTIFACT_DIRECTORY / "DETERMINISM_VERIFICATION.json"
A3_REQUIRED_PROTOCOL_VERSION = "readout-trace-v1a3"
A3_RESOURCE_INVALID_PROTOCOL_VERSIONS = frozenset({"readout-trace-v1a2"})
A3_REGISTERED_SEEDS: tuple[int, ...] = (105, 106, 107, 108, 109)
A3_REQUIRED_PHASE_ORDER: tuple[str, ...] = (
    "tests",
    "freeze",
    "persisted_smoke_and_sidecar",
    "machine_smoke_verification",
    "full",
    "rerun",
    "determinism_verification",
)
A3_PHASE_SEQUENCE_PATH = Path(
    "artifacts/experiment_000/readout_trace_a3/PHASE_SEQUENCE.json"
)
A3_PRIMARY_REPORT_PATH = Path(
    "artifacts/experiment_000/readout_trace_a3/custom_seeds_105_109_pairs_100.json"
)
A3_RERUN_REPORT_PATH = Path(
    "artifacts/experiment_000/readout_trace_a3/"
    "custom_seeds_105_109_pairs_100_rerun.json"
)
A3_DETERMINISM_VERIFICATION_PATH = Path(
    "artifacts/experiment_000/readout_trace_a3/DETERMINISM_VERIFICATION.json"
)

_PROCESS_INSTANCE_TOKEN = hashlib.sha256(os.urandom(32)).hexdigest()
_SOURCE_PATHS: tuple[Path, ...] = (
    PROTOCOL_PATH,
    CONFIG_PATH,
    Path("src/adaptive_learning_substrate/experiment_000_trace_head_alignment.py"),
    Path("tests/test_experiment_000_trace_head_alignment.py"),
    Path("src/adaptive_learning_substrate/experiment_000_readout_trace_a3.py"),
    Path("src/adaptive_learning_substrate/readout_trace_recurrent.py"),
    Path("src/adaptive_learning_substrate/recurrent.py"),
    Path("src/adaptive_learning_substrate/experiment_000.py"),
    Path("src/adaptive_learning_substrate/experiment000_data.py"),
    Path("src/adaptive_learning_substrate/experiment_000_memory_probe.py"),
)


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with _project_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _array_sha256(values: Sequence[float] | np.ndarray) -> str:
    array = np.asarray(values, dtype="<f8", order="C")
    hasher = hashlib.sha256()
    hasher.update(str(array.shape).encode("ascii"))
    hasher.update(b"\0")
    hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()


def _record_self_hash(record: Mapping[str, Any], field: str) -> str:
    payload = copy.deepcopy(dict(record))
    payload.pop(field, None)
    return _sha256_json(payload)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_canonical_utc_timestamp(value: object, *, require_z: bool = False) -> bool:
    if not isinstance(value, str):
        return False
    if require_z and not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        return False
    canonical = parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if value.endswith("Z"):
        return value == canonical
    return value == parsed.isoformat(timespec="microseconds")


def _head_contract() -> dict[str, Any]:
    return {
        "prediction": "q=tanh(theta^T z)",
        "loss": "L=0.5*(y-q)^2",
        "analytic_descent_direction": "g=(y-q)*(1-q^2)*z",
        "initial_theta": 0.0,
        "learning_rate": LEARNING_RATE,
        "normalizer": "max(1,||z||^2)",
        "element_update_clip": ELEMENT_UPDATE_CLIP,
        "theta_clip": THETA_CLIP,
    }


def _finite_difference_contract() -> dict[str, Any]:
    return {
        "method": "central_all_16_coordinates",
        "epsilon": FINITE_DIFFERENCE_EPSILON,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "coordinate_rule": "absolute_pass_or_relative_pass",
        "direction_cosine_minimum": DIRECTION_COSINE_MINIMUM,
        "positive_dot_required": True,
    }


def _validate_source_manifest(manifest: object) -> None:
    if not isinstance(manifest, dict) or set(manifest) != {
        "files",
        "bundle_sha256",
    }:
        raise RuntimeError("alignment source manifest schema is invalid")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise TypeError("alignment source manifest file list is invalid")
    paths: list[str] = []
    for row in files:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "bytes", "sha256"}
            or not isinstance(row.get("path"), str)
            or not row["path"]
            or type(row.get("bytes")) is not int
            or row["bytes"] < 0
            or not _is_sha256(row.get("sha256"))
        ):
            raise RuntimeError("alignment source manifest row is invalid")
        paths.append(row["path"])
    if (
        len(paths) != len(set(paths))
        or not _is_sha256(manifest.get("bundle_sha256"))
        or manifest["bundle_sha256"] != _sha256_json(files)
    ):
        raise RuntimeError("alignment source manifest digest is invalid")


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = _project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{_PROCESS_INSTANCE_TOKEN[:16]}.tmp"
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise RuntimeError(f"immutable artifact already exists: {target.name}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _write_report_and_sidecar(
    path: str | Path, report: Mapping[str, Any]
) -> tuple[Path, Path]:
    target = _atomic_write_json(path, report)
    sidecar = target.with_suffix(".sha256")
    _atomic_write_json(
        sidecar,
        {
            "algorithm": "sha256",
            "report_file": target.name,
            "report_sha256": _file_sha256(target),
        },
    )
    return target, sidecar


def _read_report_with_sidecar(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    target = _project_path(path)
    sidecar_path = target.with_suffix(".sha256")
    if not target.is_file() or not sidecar_path.is_file():
        raise RuntimeError(f"report or SHA-256 sidecar is absent: {target.name}")
    report_hash = _file_sha256(target)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if (
        sidecar.get("algorithm") != "sha256"
        or sidecar.get("report_file") != target.name
        or sidecar.get("report_sha256") != report_hash
    ):
        raise RuntimeError(f"report sidecar is invalid: {target.name}")
    report = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise TypeError("persisted alignment report must be a JSON object")
    return report, {
        "report_file_sha256": report_hash,
        "sidecar_file_sha256": _file_sha256(sidecar_path),
    }


def _source_manifest() -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for relative in _SOURCE_PATHS:
        target = _project_path(relative)
        if not target.is_file():
            raise RuntimeError(f"frozen alignment source is absent: {relative}")
        files.append(
            {
                "path": str(relative).replace("\\", "/"),
                "bytes": target.stat().st_size,
                "sha256": _file_sha256(target),
            }
        )
    return {"files": files, "bundle_sha256": _sha256_json(files)}


def validate_protocol_config() -> dict[str, Any]:
    """Load the registered TOML and reject drift in scientific fields."""

    protocol = _project_path(PROTOCOL_PATH)
    config = _project_path(CONFIG_PATH)
    if not protocol.is_file() or not config.is_file():
        raise RuntimeError("TRACE-HEAD-L1 protocol or config is absent")
    with config.open("rb") as handle:
        payload = tomllib.load(handle)
    expected: dict[str, Any] = {
        "protocol": {
            "name": PROTOCOL_NAME,
            "version": PROTOCOL_VERSION,
            "schema_version": SCHEMA_VERSION,
            "protocol_path": str(PROTOCOL_PATH).replace("\\", "/"),
            "scope": "standalone_selected_trace_head_exact_alignment",
            "claim": "head_alignment_only_not_recurrent_temporal_credit",
        },
        "a3_prerequisite": {
            "required_protocol_version": A3_REQUIRED_PROTOCOL_VERSION,
            "resource_invalid_protocol_versions": sorted(
                A3_RESOURCE_INVALID_PROTOCOL_VERSIONS
            ),
            "required_registered_seeds": list(A3_REGISTERED_SEEDS),
            "required_phase_count": 7,
            "required_terminal_prefix": "SELECTED:",
            "no_selection_allowed": False,
            "phase_sequence_path": str(A3_PHASE_SEQUENCE_PATH).replace("\\", "/"),
            "primary_path": str(A3_PRIMARY_REPORT_PATH).replace("\\", "/"),
            "rerun_path": str(A3_RERUN_REPORT_PATH).replace("\\", "/"),
            "determinism_path": str(A3_DETERMINISM_VERIFICATION_PATH).replace(
                "\\", "/"
            ),
        },
        "execution": {
            "enabled": True,
            "official_seeds": list(OFFICIAL_SEEDS),
            "episodes_per_seed": EPISODES_PER_SEED,
            "noise_events": NOISE_EVENTS,
            "data_split_namespace": "train",
            "target_encoding": "cue_0=-1,cue_1=+1",
            "feature_dimension": FEATURE_DIMENSION,
            "feature_order": "lexical_output_edge_id",
            "required_independent_runs": 2,
            "confirmatory_seeds_allowed": False,
        },
        "head": {
            "activation": "tanh",
            "initial_theta": 0.0,
            "loss": "0.5*(y-q)^2",
            "analytic_descent_direction": "(y-q)*(1-q^2)*z",
            "learning_rate": LEARNING_RATE,
            "normalizer": "max(1,||z||^2)",
            "element_update_clip": ELEMENT_UPDATE_CLIP,
            "theta_clip": THETA_CLIP,
        },
        "finite_difference": {
            "method": "central",
            "epsilon": FINITE_DIFFERENCE_EPSILON,
            "coordinates_per_case": FEATURE_DIMENSION,
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "relative_tolerance": RELATIVE_TOLERANCE,
            "direction_cosine_min": DIRECTION_COSINE_MINIMUM,
            "positive_dot_required": True,
            "case_pass_fraction": 1.0,
            "coordinate_pass_fraction": 1.0,
        },
        "invariants": {
            "topology_unchanged": True,
            "all_graph_weights_unchanged": True,
            "recurrent_weights_unchanged": True,
            "trace_read_only_to_head": True,
        },
        "artifacts": {
            "primary": str(PRIMARY_REPORT_PATH).replace("\\", "/"),
            "primary_sidecar": str(PRIMARY_REPORT_PATH.with_suffix(".sha256")).replace(
                "\\", "/"
            ),
            "rerun": str(RERUN_REPORT_PATH).replace("\\", "/"),
            "rerun_sidecar": str(RERUN_REPORT_PATH.with_suffix(".sha256")).replace(
                "\\", "/"
            ),
            "determinism": str(DETERMINISM_VERIFICATION_PATH).replace("\\", "/"),
        },
    }
    if payload != expected:
        raise RuntimeError("TRACE-HEAD-L1 config differs from the registered contract")
    return payload


def _parse_selected_condition(condition: str) -> float:
    if not condition.startswith("rho_"):
        raise ValueError("A3 terminal selection is not a registered rho condition")
    try:
        value = float(condition.removeprefix("rho_").replace("_", "."))
    except ValueError as exc:
        raise ValueError("A3 terminal selection has an invalid retention") from exc
    registered = {0.25, 0.50, 0.75, 0.90, 0.95}
    if value not in registered:
        raise ValueError("A3 terminal selection has an unregistered retention")
    expected = f"rho_{value:.2f}".replace(".", "_")
    if condition != expected:
        raise ValueError("A3 terminal selection is not canonically formatted")
    return value


def _validated_a3_selection_binding(
    selection: Mapping[str, Any], phase_info: Mapping[str, Any]
) -> dict[str, Any]:
    """Normalize the exact A3 loader result and bind its complete phase chain."""

    from . import experiment_000_readout_trace_a3 as a3

    if not isinstance(selection, dict):
        raise TypeError("A3 loader result must be an object")
    protocol_version = selection.get("protocol_version")
    if (
        isinstance(protocol_version, str)
        and protocol_version in A3_RESOURCE_INVALID_PROTOCOL_VERSIONS
    ):
        raise RuntimeError(
            f"resource-invalid predecessor {protocol_version!r} cannot satisfy "
            "the TRACE-HEAD-L1 prerequisite"
        )
    expected_module_paths = (
        (a3.PHASE_SEQUENCE_PATH, A3_PHASE_SEQUENCE_PATH),
        (a3.PRIMARY_REPORT_PATH, A3_PRIMARY_REPORT_PATH),
        (a3.RERUN_REPORT_PATH, A3_RERUN_REPORT_PATH),
        (a3.DETERMINISM_VERIFICATION_PATH, A3_DETERMINISM_VERIFICATION_PATH),
    )
    if (
        a3.PROTOCOL_VERSION != A3_REQUIRED_PROTOCOL_VERSION
        or tuple(a3.REGISTERED_SEEDS) != A3_REGISTERED_SEEDS
        or any(
            str(observed).replace("\\", "/")
            != str(expected).replace("\\", "/")
            for observed, expected in expected_module_paths
        )
    ):
        raise RuntimeError("A3 module identity differs from the registered prerequisite")

    selection_keys = {
        "schema_version",
        "status",
        "selected_condition",
        "retention_coefficient",
        "protocol_version",
        "source_manifest_sha256",
        "freeze_record_sha256",
        "primary",
        "rerun",
        "determinism",
    }
    primary_keys = {
        "path",
        "report_file_sha256",
        "sidecar_file_sha256",
        "deterministic_payload_sha256",
    }
    determinism_keys = {
        "path",
        "file_sha256",
        "verification_payload_sha256",
    }
    primary = selection.get("primary")
    rerun = selection.get("rerun")
    determinism = selection.get("determinism")
    selected = selection.get("selected_condition")
    if (
        set(selection) != selection_keys
        or selection.get("schema_version")
        != "experiment-000-readout-trace-a3-selection-binding-v1"
        or protocol_version != A3_REQUIRED_PROTOCOL_VERSION
        or not isinstance(selected, str)
        or selection.get("status") != f"SELECTED:{selected}"
        or not isinstance(primary, dict)
        or set(primary) != primary_keys
        or not isinstance(rerun, dict)
        or set(rerun) != primary_keys
        or not isinstance(determinism, dict)
        or set(determinism) != determinism_keys
        or primary.get("path") != str(A3_PRIMARY_REPORT_PATH).replace("\\", "/")
        or rerun.get("path") != str(A3_RERUN_REPORT_PATH).replace("\\", "/")
        or determinism.get("path")
        != str(A3_DETERMINISM_VERIFICATION_PATH).replace("\\", "/")
    ):
        raise RuntimeError("A3 loader result differs from its registered selection schema")
    retention = _parse_selected_condition(selected)
    if (
        type(selection.get("retention_coefficient")) is not float
        or selection.get("retention_coefficient") != retention
        or any(
            not _is_sha256(value)
            for value in (
                selection.get("source_manifest_sha256"),
                selection.get("freeze_record_sha256"),
                primary.get("report_file_sha256"),
                primary.get("sidecar_file_sha256"),
                primary.get("deterministic_payload_sha256"),
                rerun.get("report_file_sha256"),
                rerun.get("sidecar_file_sha256"),
                rerun.get("deterministic_payload_sha256"),
                determinism.get("file_sha256"),
                determinism.get("verification_payload_sha256"),
            )
        )
        or primary.get("deterministic_payload_sha256")
        != rerun.get("deterministic_payload_sha256")
    ):
        raise RuntimeError("A3 loader result has invalid selection or artifact identities")

    if not isinstance(phase_info, dict):
        raise TypeError("A3 phase validation result must be an object")
    phase_record = phase_info.get("record")
    phases = phase_record.get("phases") if isinstance(phase_record, Mapping) else None
    if (
        set(phase_info) != {"record", "file_sha256"}
        or not _is_sha256(phase_info.get("file_sha256"))
        or not isinstance(phase_record, dict)
        or phase_record.get("schema_version")
        != "experiment-000-readout-trace-a3-phases-v1"
        or phase_record.get("protocol_version") != A3_REQUIRED_PROTOCOL_VERSION
        or phase_record.get("phase_order") != list(A3_REQUIRED_PHASE_ORDER)
        or not isinstance(phases, list)
        or len(phases) != len(A3_REQUIRED_PHASE_ORDER)
        or any(not isinstance(phase, Mapping) for phase in phases)
        or [phase.get("name") for phase in phases]
        != list(A3_REQUIRED_PHASE_ORDER)
        or phase_record.get("phase_sequence_payload_sha256")
        != a3._record_self_hash(phase_record, "phase_sequence_payload_sha256")
    ):
        raise RuntimeError("A3 complete phase sequence binding is invalid")

    return {
        "a3_protocol_version": A3_REQUIRED_PROTOCOL_VERSION,
        "a3_registered_seeds": list(A3_REGISTERED_SEEDS),
        "a3_phase_sequence_file_sha256": phase_info["file_sha256"],
        "a3_phase_sequence_payload_sha256": phase_record[
            "phase_sequence_payload_sha256"
        ],
        "a3_source_manifest_sha256": selection["source_manifest_sha256"],
        "a3_freeze_record_file_sha256": selection["freeze_record_sha256"],
        "a3_primary_report_file_sha256": primary["report_file_sha256"],
        "a3_primary_sidecar_file_sha256": primary["sidecar_file_sha256"],
        "a3_rerun_report_file_sha256": rerun["report_file_sha256"],
        "a3_rerun_sidecar_file_sha256": rerun["sidecar_file_sha256"],
        "a3_determinism_verification_file_sha256": determinism["file_sha256"],
        "a3_determinism_verification_payload_sha256": determinism[
            "verification_payload_sha256"
        ],
        "a3_deterministic_payload_sha256": primary[
            "deterministic_payload_sha256"
        ],
        "selected_condition": selected,
        "trace_retention": retention,
    }


def load_verified_a3_selection() -> dict[str, Any]:
    """Accept only A3's fully validated terminal selection and seven phases."""

    from . import experiment_000_readout_trace_a3 as a3

    first_selection = a3.load_verified_a3_selection()
    phase_info = a3._validate_phase_sequence(a3._PHASE_ORDER)
    second_selection = a3.load_verified_a3_selection()
    second_phase_info = a3._validate_phase_sequence(a3._PHASE_ORDER)
    if first_selection != second_selection or phase_info != second_phase_info:
        raise RuntimeError("A3 selection or phase artifacts changed while binding")
    return _validated_a3_selection_binding(first_selection, phase_info)


def _as_vector(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (FEATURE_DIMENSION,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 16-dimensional vector")
    return vector


def _target_value(target: float) -> float:
    value = float(target)
    if value not in (-1.0, 1.0):
        raise ValueError("target must be exactly -1 or +1")
    return value


def _dot(left: np.ndarray, right: np.ndarray) -> float:
    return float(math.fsum(float(a) * float(b) for a, b in zip(left, right)))


def head_loss(
    theta: Sequence[float] | np.ndarray,
    features: Sequence[float] | np.ndarray,
    target: float,
) -> float:
    resolved_theta = _as_vector(theta, name="theta")
    resolved_features = _as_vector(features, name="features")
    y = _target_value(target)
    prediction = math.tanh(_dot(resolved_theta, resolved_features))
    return 0.5 * (y - prediction) ** 2


def analytic_descent_direction(
    theta: Sequence[float] | np.ndarray,
    features: Sequence[float] | np.ndarray,
    target: float,
) -> np.ndarray:
    """Return ``-(dL/dtheta) = (y-q)(1-q**2)z`` in float64."""

    resolved_theta = _as_vector(theta, name="theta")
    resolved_features = _as_vector(features, name="features")
    y = _target_value(target)
    prediction = math.tanh(_dot(resolved_theta, resolved_features))
    return np.asarray(
        (y - prediction) * (1.0 - prediction * prediction) * resolved_features,
        dtype=np.float64,
    )


def central_difference_descent_direction(
    theta: Sequence[float] | np.ndarray,
    features: Sequence[float] | np.ndarray,
    target: float,
    *,
    epsilon: float = FINITE_DIFFERENCE_EPSILON,
) -> np.ndarray:
    """Numerically compute the descent direction on every head coordinate."""

    resolved_theta = _as_vector(theta, name="theta")
    resolved_features = _as_vector(features, name="features")
    y = _target_value(target)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("finite-difference epsilon must be finite and positive")
    numerical = np.empty(FEATURE_DIMENSION, dtype=np.float64)
    for index in range(FEATURE_DIMENSION):
        plus = resolved_theta.copy()
        minus = resolved_theta.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical[index] = -(
            head_loss(plus, resolved_features, y)
            - head_loss(minus, resolved_features, y)
        ) / (2.0 * epsilon)
    if not np.all(np.isfinite(numerical)):
        raise FloatingPointError("central finite differences produced a nonfinite value")
    return numerical


def normalized_head_update(
    theta: Sequence[float] | np.ndarray,
    direction: Sequence[float] | np.ndarray,
    features: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply the frozen normalized, element-clipped, head-only update."""

    resolved_theta = _as_vector(theta, name="theta")
    resolved_direction = _as_vector(direction, name="direction")
    resolved_features = _as_vector(features, name="features")
    squared_norm = _dot(resolved_features, resolved_features)
    normalizer = max(1.0, squared_norm)
    delta = np.clip(
        LEARNING_RATE * resolved_direction / normalizer,
        -ELEMENT_UPDATE_CLIP,
        ELEMENT_UPDATE_CLIP,
    )
    updated = np.clip(resolved_theta + delta, -THETA_CLIP, THETA_CLIP)
    if not np.all(np.isfinite(updated)):
        raise FloatingPointError("head update produced a nonfinite value")
    return np.asarray(updated, dtype=np.float64), np.asarray(delta, dtype=np.float64), normalizer


def _weights_by_kind_sha256(graph: RecurrentEventGraph, kind: str) -> str:
    weights = graph.weights
    rows = [
        [edge.edge_id, struct.pack("<d", float(weights[edge.edge_id])).hex()]
        for edge in graph.edges
        if edge.kind == kind
    ]
    return _sha256_json(rows)


def extract_selected_trace_features(
    graph: ReadoutTraceRecurrentEventGraph,
    episode: Any,
    feature_edge_ids: tuple[str, ...],
) -> tuple[np.ndarray, Any]:
    """Run one episode and read the selected passive trace without graph mutation."""

    if len(feature_edge_ids) != FEATURE_DIMENSION:
        raise RuntimeError("TRACE-HEAD-L1 requires exactly 16 output edges")
    query = forward_episode(graph, episode)
    before = (
        graph.topology_hash(),
        graph.weights_hash(),
        _weights_by_kind_sha256(graph, "recurrent"),
    )
    features = graph.query_trace_features(feature_edge_ids, query.tick)
    after = (
        graph.topology_hash(),
        graph.weights_hash(),
        _weights_by_kind_sha256(graph, "recurrent"),
    )
    if before != after or tuple(edge_id for edge_id, _ in features) != feature_edge_ids:
        raise RuntimeError("selected passive-trace read mutated or reordered the graph")
    vector = _as_vector([value for _, value in features], name="features")
    return vector, query


def _gradient_case(
    theta: np.ndarray,
    features: np.ndarray,
    target: float,
) -> tuple[dict[str, Any], np.ndarray]:
    prediction = math.tanh(_dot(theta, features))
    loss = 0.5 * (target - prediction) ** 2
    analytic = analytic_descent_direction(theta, features, target)
    numerical = central_difference_descent_direction(theta, features, target)
    absolute_errors = np.abs(analytic - numerical)
    relative_denominator = np.maximum(
        np.maximum(np.abs(analytic), np.abs(numerical)),
        np.finfo(np.float64).tiny,
    )
    relative_errors = absolute_errors / relative_denominator
    coordinate_pass = np.logical_or(
        absolute_errors <= ABSOLUTE_TOLERANCE,
        relative_errors <= RELATIVE_TOLERANCE,
    )
    dot_product = _dot(analytic, numerical)
    analytic_norm = math.sqrt(max(0.0, _dot(analytic, analytic)))
    numerical_norm = math.sqrt(max(0.0, _dot(numerical, numerical)))
    denominator = analytic_norm * numerical_norm
    # A zero direction is a finite but failed alignment case.  Use a finite
    # sentinel so a scientifically failed report remains valid JSON evidence.
    cosine = dot_product / denominator if denominator > 0.0 else -1.0
    updated, delta, normalizer = normalized_head_update(theta, analytic, features)
    reconstructed, reconstructed_delta, reconstructed_normalizer = (
        normalized_head_update(theta, analytic, features)
    )
    update_reconstruction_pass = (
        normalizer == reconstructed_normalizer
        and np.array_equal(delta, reconstructed_delta)
        and np.array_equal(updated, reconstructed)
    )
    values_finite = all(
        math.isfinite(value)
        for value in (
            prediction,
            loss,
            dot_product,
            analytic_norm,
            numerical_norm,
            cosine,
            normalizer,
            float(np.max(absolute_errors)),
            float(np.max(relative_errors)),
        )
    ) and bool(
        np.all(np.isfinite(analytic))
        and np.all(np.isfinite(numerical))
        and np.all(np.isfinite(delta))
        and np.all(np.isfinite(updated))
    )
    case_pass = (
        bool(np.all(coordinate_pass))
        and cosine >= DIRECTION_COSINE_MINIMUM
        and dot_product > 0.0
        and update_reconstruction_pass
        and values_finite
    )
    return (
        {
            "target": target,
            "feature_vector": features.tolist(),
            "feature_vector_sha256": _array_sha256(features),
            "theta_before": theta.tolist(),
            "theta_before_sha256": _array_sha256(theta),
            "preactivation": _dot(theta, features),
            "prediction": prediction,
            "loss": loss,
            "analytic_descent_direction": analytic.tolist(),
            "analytic_descent_direction_sha256": _array_sha256(analytic),
            "numerical_descent_direction": numerical.tolist(),
            "numerical_descent_direction_sha256": _array_sha256(numerical),
            "coordinate_pass": coordinate_pass.tolist(),
            "coordinate_pass_count": int(np.count_nonzero(coordinate_pass)),
            "all_coordinates_pass": bool(np.all(coordinate_pass)),
            "maximum_absolute_error": float(np.max(absolute_errors)),
            "maximum_relative_error": float(np.max(relative_errors)),
            "direction_cosine": cosine,
            "direction_cosine_pass": cosine >= DIRECTION_COSINE_MINIMUM,
            "analytic_numeric_dot": dot_product,
            "positive_dot_pass": dot_product > 0.0,
            "normalizer": normalizer,
            "delta": delta.tolist(),
            "delta_sha256": _array_sha256(delta),
            "theta_after": updated.tolist(),
            "theta_after_sha256": _array_sha256(updated),
            "update_reconstruction_pass": update_reconstruction_pass,
            "values_finite": values_finite,
            "case_pass": case_pass,
        },
        updated,
    )


def _run_seed_alignment(
    *, seed: int, episodes_per_seed: int, retention: float
) -> dict[str, Any]:
    require_seed_role(seed, SeedRole.CUSTOM)
    stream = generate_stream(
        seed,
        "train",
        episode_count=episodes_per_seed,
        noise_events=NOISE_EVENTS,
    )
    native = build_experiment_000_graph(
        seed,
        mode="full",
        event_log_enabled=False,
        **FROZEN_GRAPH_OPTIONS,
    )
    graph = clone_with_readout_trace(native, retention, event_log_enabled=False)
    feature_edge_ids = _output_feature_edge_ids(graph)
    if len(feature_edge_ids) != FEATURE_DIMENSION:
        raise RuntimeError("frozen graph does not expose exactly 16 output features")
    topology_before = graph.topology_hash()
    weights_before = graph.weights_hash()
    recurrent_before = _weights_by_kind_sha256(graph, "recurrent")
    theta = np.zeros(FEATURE_DIMENSION, dtype=np.float64)
    if np.any(np.signbit(theta)):
        raise RuntimeError("theta must start as exact positive zero")
    rows: list[dict[str, Any]] = []
    for episode in stream.episodes:
        features, query = extract_selected_trace_features(
            graph, episode, feature_edge_ids
        )
        graph_before_head = (
            graph.topology_hash(),
            graph.weights_hash(),
            _weights_by_kind_sha256(graph, "recurrent"),
        )
        target = 1.0 if episode.target == 1 else -1.0
        row, theta = _gradient_case(theta, features, target)
        graph_after_head = (
            graph.topology_hash(),
            graph.weights_hash(),
            _weights_by_kind_sha256(graph, "recurrent"),
        )
        row.update(
            {
                "episode_id": episode.episode_id,
                "episode_index": episode.index,
                "cue": episode.cue,
                "query_tick": query.tick,
                "topology_unchanged_during_case": (
                    graph_before_head[0] == graph_after_head[0] == topology_before
                ),
                "all_graph_weights_unchanged_during_case": (
                    graph_before_head[1] == graph_after_head[1] == weights_before
                ),
                "recurrent_weights_unchanged_during_case": (
                    graph_before_head[2] == graph_after_head[2] == recurrent_before
                ),
            }
        )
        row["case_pass"] = bool(
            row["case_pass"]
            and row["topology_unchanged_during_case"]
            and row["all_graph_weights_unchanged_during_case"]
            and row["recurrent_weights_unchanged_during_case"]
        )
        rows.append(row)
    topology_after = graph.topology_hash()
    weights_after = graph.weights_hash()
    recurrent_after = _weights_by_kind_sha256(graph, "recurrent")
    coordinate_checks = episodes_per_seed * FEATURE_DIMENSION
    coordinate_passes = sum(row["coordinate_pass_count"] for row in rows)
    result: dict[str, Any] = {
        "seed": seed,
        "seed_role": SeedRole.CUSTOM.value,
        "episodes": episodes_per_seed,
        "cue_zeros": sum(episode.cue == 0 for episode in stream.episodes),
        "cue_ones": sum(episode.cue == 1 for episode in stream.episodes),
        "stream_sha256": stream.sha256,
        "trace_retention": retention,
        "feature_edge_ids": list(feature_edge_ids),
        "topology_sha256_before": topology_before,
        "topology_sha256_after": topology_after,
        "weights_sha256_before": weights_before,
        "weights_sha256_after": weights_after,
        "recurrent_weights_sha256_before": recurrent_before,
        "recurrent_weights_sha256_after": recurrent_after,
        "theta_initial": [0.0] * FEATURE_DIMENSION,
        "theta_final": theta.tolist(),
        "theta_final_sha256": _array_sha256(theta),
        "cases": rows,
        "case_count": len(rows),
        "case_pass_count": sum(bool(row["case_pass"]) for row in rows),
        "coordinate_check_count": coordinate_checks,
        "coordinate_pass_count": coordinate_passes,
        "all_cases_pass": all(bool(row["case_pass"]) for row in rows),
        "all_coordinates_pass": coordinate_passes == coordinate_checks,
        "all_direction_cosines_pass": all(
            bool(row["direction_cosine_pass"]) for row in rows
        ),
        "all_positive_dots_pass": all(bool(row["positive_dot_pass"]) for row in rows),
        "all_updates_reconstruct": all(
            bool(row["update_reconstruction_pass"]) for row in rows
        ),
        "topology_unchanged": topology_after == topology_before,
        "all_graph_weights_unchanged": weights_after == weights_before,
        "recurrent_weights_unchanged": recurrent_after == recurrent_before,
    }
    result["seed_payload_sha256"] = _record_self_hash(result, "seed_payload_sha256")
    return result


def _run_alignment_core(
    *, seeds: Sequence[int], episodes_per_seed: int, retention: float
) -> dict[str, Any]:
    seed_results = [
        _run_seed_alignment(
            seed=int(seed),
            episodes_per_seed=episodes_per_seed,
            retention=retention,
        )
        for seed in seeds
    ]
    total_cases = len(seed_results) * episodes_per_seed
    total_coordinates = total_cases * FEATURE_DIMENSION
    case_passes = sum(row["case_pass_count"] for row in seed_results)
    coordinate_passes = sum(row["coordinate_pass_count"] for row in seed_results)
    gates = {
        "case_pass_fraction_is_one": case_passes == total_cases,
        "coordinate_pass_fraction_is_one": coordinate_passes == total_coordinates,
        "all_direction_cosines_pass": all(
            row["all_direction_cosines_pass"] for row in seed_results
        ),
        "all_positive_dots_pass": all(
            row["all_positive_dots_pass"] for row in seed_results
        ),
        "all_updates_reconstruct": all(
            row["all_updates_reconstruct"] for row in seed_results
        ),
        "all_topologies_unchanged": all(
            row["topology_unchanged"] for row in seed_results
        ),
        "all_graph_weights_unchanged": all(
            row["all_graph_weights_unchanged"] for row in seed_results
        ),
        "all_recurrent_weights_unchanged": all(
            row["recurrent_weights_unchanged"] for row in seed_results
        ),
    }
    return {
        "seed_results": seed_results,
        "total_case_count": total_cases,
        "passed_case_count": case_passes,
        "total_coordinate_check_count": total_coordinates,
        "passed_coordinate_check_count": coordinate_passes,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def run_scratch_alignment(
    *,
    seeds: Sequence[int] = (9075,),
    episodes_per_seed: int = 4,
    retention: float = 0.75,
) -> dict[str, Any]:
    """Run a small, nonpersisting test instrument without reading A3 artifacts."""

    resolved = tuple(int(seed) for seed in seeds)
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("scratch seeds must be a nonempty unique sequence")
    if set(resolved).intersection(OFFICIAL_SEEDS):
        raise ValueError("scratch execution rejects official seeds 75--79")
    if (
        isinstance(episodes_per_seed, bool)
        or episodes_per_seed <= 0
        or episodes_per_seed % 2
    ):
        raise ValueError("scratch episodes must be a positive even integer")
    if not math.isfinite(retention) or not 0.0 < retention < 1.0:
        raise ValueError("scratch retention must be finite and strictly between 0 and 1")
    result = _run_alignment_core(
        seeds=resolved,
        episodes_per_seed=episodes_per_seed,
        retention=float(retention),
    )
    return {
        "schema_version": f"{SCHEMA_VERSION}-scratch",
        "scratch_nonselecting": True,
        "seeds": list(resolved),
        "episodes_per_seed": episodes_per_seed,
        "trace_retention": float(retention),
        **result,
    }


def _report_payload_object(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(report))
    payload.pop("nondeterministic_provenance", None)
    payload.pop("deterministic_payload_sha256", None)
    return payload


def _recompute_report_payload_sha256(report: Mapping[str, Any]) -> str:
    return _sha256_json(_report_payload_object(report))


def _validate_case(row: Mapping[str, Any]) -> None:
    theta = _as_vector(row.get("theta_before", ()), name="theta_before")
    features = _as_vector(row.get("feature_vector", ()), name="feature_vector")
    target = _target_value(row.get("target"))
    analytic = analytic_descent_direction(theta, features, target)
    numerical = central_difference_descent_direction(theta, features, target)
    expected, delta, normalizer = normalized_head_update(theta, analytic, features)
    preactivation = _dot(theta, features)
    prediction = math.tanh(preactivation)
    loss = 0.5 * (target - prediction) ** 2
    if (
        row.get("feature_vector_sha256") != _array_sha256(features)
        or row.get("theta_before_sha256") != _array_sha256(theta)
        or row.get("analytic_descent_direction") != analytic.tolist()
        or row.get("analytic_descent_direction_sha256") != _array_sha256(analytic)
        or row.get("numerical_descent_direction") != numerical.tolist()
        or row.get("numerical_descent_direction_sha256") != _array_sha256(numerical)
        or row.get("preactivation") != preactivation
        or row.get("prediction") != prediction
        or row.get("loss") != loss
        or row.get("normalizer") != normalizer
        or row.get("delta") != delta.tolist()
        or row.get("delta_sha256") != _array_sha256(delta)
        or row.get("theta_after") != expected.tolist()
        or row.get("theta_after_sha256") != _array_sha256(expected)
    ):
        raise RuntimeError("persisted alignment case does not reconstruct")
    absolute = np.abs(analytic - numerical)
    relative = absolute / np.maximum(
        np.maximum(np.abs(analytic), np.abs(numerical)),
        np.finfo(np.float64).tiny,
    )
    passes = np.logical_or(
        absolute <= ABSOLUTE_TOLERANCE, relative <= RELATIVE_TOLERANCE
    )
    dot_product = _dot(analytic, numerical)
    denominator = math.sqrt(_dot(analytic, analytic)) * math.sqrt(
        _dot(numerical, numerical)
    )
    cosine = dot_product / denominator if denominator > 0.0 else -1.0
    expected_pass = bool(
        np.all(passes)
        and dot_product > 0.0
        and cosine >= DIRECTION_COSINE_MINIMUM
        and row.get("topology_unchanged_during_case") is True
        and row.get("all_graph_weights_unchanged_during_case") is True
        and row.get("recurrent_weights_unchanged_during_case") is True
    )
    if (
        row.get("coordinate_pass") != passes.tolist()
        or row.get("coordinate_pass_count") != int(np.count_nonzero(passes))
        or row.get("all_coordinates_pass") is not bool(np.all(passes))
        or row.get("maximum_absolute_error") != float(np.max(absolute))
        or row.get("maximum_relative_error") != float(np.max(relative))
        or row.get("direction_cosine") != cosine
        or row.get("direction_cosine_pass")
        is not (cosine >= DIRECTION_COSINE_MINIMUM)
        or row.get("analytic_numeric_dot") != dot_product
        or row.get("positive_dot_pass") is not (dot_product > 0.0)
        or row.get("update_reconstruction_pass") is not True
        or row.get("values_finite") is not True
        or row.get("case_pass") is not expected_pass
    ):
        raise RuntimeError("persisted alignment gate values do not reconstruct")


def _validate_alignment_report(
    report: Mapping[str, Any],
    *,
    expected_a3_binding: Mapping[str, Any] | None = None,
    current_manifest: Mapping[str, Any] | None = None,
) -> None:
    report_keys = {
        "schema_version",
        "protocol_name",
        "protocol_version",
        "run_kind",
        "claim_scope",
        "seeds",
        "episodes_per_seed",
        "noise_events",
        "feature_dimension",
        "target_encoding",
        "head_contract",
        "finite_difference_contract",
        "a3_selection_binding",
        "source_manifest_start",
        "source_manifest_end",
        "seed_results",
        "total_case_count",
        "passed_case_count",
        "total_coordinate_check_count",
        "passed_coordinate_check_count",
        "gates",
        "all_gates_pass",
        "status",
        "deterministic_payload_sha256",
        "nondeterministic_provenance",
    }
    if (
        not isinstance(report, dict)
        or set(report) != report_keys
        or report.get("schema_version") != SCHEMA_VERSION
        or report.get("protocol_name") != PROTOCOL_NAME
        or report.get("protocol_version") != PROTOCOL_VERSION
        or report.get("run_kind") != "full"
        or report.get("claim_scope")
        != "standalone_head_alignment_only_not_recurrent_temporal_credit"
        or report.get("seeds") != list(OFFICIAL_SEEDS)
        or report.get("episodes_per_seed") != EPISODES_PER_SEED
        or type(report.get("episodes_per_seed")) is not int
        or report.get("noise_events") != NOISE_EVENTS
        or type(report.get("noise_events")) is not int
        or report.get("feature_dimension") != FEATURE_DIMENSION
        or type(report.get("feature_dimension")) is not int
        or _canonical_json(report.get("target_encoding"))
        != _canonical_json({"cue_0": -1.0, "cue_1": 1.0})
        or _canonical_json(report.get("head_contract"))
        != _canonical_json(_head_contract())
        or _canonical_json(report.get("finite_difference_contract"))
        != _canonical_json(_finite_difference_contract())
        or not _is_sha256(report.get("deterministic_payload_sha256"))
        or report.get("deterministic_payload_sha256")
        != _recompute_report_payload_sha256(report)
    ):
        raise RuntimeError("alignment report metadata or self-hash is invalid")
    binding = report.get("a3_selection_binding")
    binding_keys = {
        "a3_protocol_version",
        "a3_registered_seeds",
        "a3_phase_sequence_file_sha256",
        "a3_phase_sequence_payload_sha256",
        "a3_source_manifest_sha256",
        "a3_freeze_record_file_sha256",
        "a3_primary_report_file_sha256",
        "a3_primary_sidecar_file_sha256",
        "a3_rerun_report_file_sha256",
        "a3_rerun_sidecar_file_sha256",
        "a3_determinism_verification_file_sha256",
        "a3_determinism_verification_payload_sha256",
        "a3_deterministic_payload_sha256",
        "selected_condition",
        "trace_retention",
    }
    binding_protocol = (
        binding.get("a3_protocol_version") if isinstance(binding, Mapping) else None
    )
    if (
        isinstance(binding_protocol, str)
        and binding_protocol in A3_RESOURCE_INVALID_PROTOCOL_VERSIONS
    ):
        raise RuntimeError("alignment report contains resource-invalid v1a2 evidence")
    if (
        not isinstance(binding, dict)
        or set(binding) != binding_keys
        or binding.get("a3_protocol_version") != A3_REQUIRED_PROTOCOL_VERSION
        or binding.get("a3_registered_seeds") != list(A3_REGISTERED_SEEDS)
        or any(
            not _is_sha256(binding.get(field))
            for field in binding_keys
            if field.endswith("sha256")
        )
    ):
        raise TypeError("alignment report lacks its A3 selection binding")
    selected = binding.get("selected_condition")
    if not isinstance(selected, str):
        raise TypeError("alignment report selection is not a string")
    retention = _parse_selected_condition(selected)
    if (
        binding.get("trace_retention") != retention
        or type(binding.get("trace_retention")) is not float
    ):
        raise RuntimeError("alignment report retention disagrees with A3 selection")
    if expected_a3_binding is not None and dict(binding) != dict(expected_a3_binding):
        raise RuntimeError("alignment report A3 binding changed")
    source_start = report.get("source_manifest_start")
    source_end = report.get("source_manifest_end")
    _validate_source_manifest(source_start)
    _validate_source_manifest(source_end)
    if source_start != source_end:
        raise RuntimeError("alignment report start/end source manifests differ")
    if current_manifest is not None and (
        report.get("source_manifest_start") != dict(current_manifest)
        or report.get("source_manifest_end") != dict(current_manifest)
    ):
        raise RuntimeError("alignment report source manifest changed")
    provenance = report.get("nondeterministic_provenance")
    if (
        not isinstance(provenance, dict)
        or set(provenance)
        != {
            "excluded_from_deterministic_payload_sha256",
            "artifact_role",
            "process_id",
            "process_instance_token",
            "run_started_utc",
            "runtime_seconds",
            "primary_prerequisite",
        }
        or provenance.get("excluded_from_deterministic_payload_sha256") is not True
        or provenance.get("artifact_role") not in {"primary", "rerun"}
        or type(provenance.get("process_id")) is not int
        or provenance["process_id"] <= 0
        or not _is_sha256(provenance.get("process_instance_token"))
        or not _is_canonical_utc_timestamp(
            provenance.get("run_started_utc"), require_z=True
        )
        or type(provenance.get("runtime_seconds")) is not float
        or not math.isfinite(provenance["runtime_seconds"])
        or provenance["runtime_seconds"] < 0.0
    ):
        raise RuntimeError("alignment report process provenance is invalid")
    primary_prerequisite = provenance.get("primary_prerequisite")
    if provenance["artifact_role"] == "primary":
        if primary_prerequisite is not None:
            raise RuntimeError("alignment primary has a rerun prerequisite")
    elif (
        not isinstance(primary_prerequisite, dict)
        or set(primary_prerequisite)
        != {"report_file_sha256", "sidecar_file_sha256"}
        or not all(_is_sha256(value) for value in primary_prerequisite.values())
    ):
        raise RuntimeError("alignment rerun prerequisite is invalid")
    expected_core = _run_alignment_core(
        seeds=OFFICIAL_SEEDS,
        episodes_per_seed=EPISODES_PER_SEED,
        retention=retention,
    )
    if any(
        _canonical_json(report.get(field)) != _canonical_json(expected_core[field])
        for field in expected_core
    ):
        raise RuntimeError("alignment evidence does not replay exactly")
    seeds = report.get("seed_results")
    if not isinstance(seeds, list) or [row.get("seed") for row in seeds] != list(
        OFFICIAL_SEEDS
    ):
        raise RuntimeError("alignment report seed rows are absent or out of order")
    total_case_passes = 0
    total_coordinate_passes = 0
    for seed_row in seeds:
        if (
            seed_row.get("episodes") != EPISODES_PER_SEED
            or seed_row.get("cue_zeros") != EPISODES_PER_SEED // 2
            or seed_row.get("cue_ones") != EPISODES_PER_SEED // 2
            or seed_row.get("trace_retention") != retention
            or seed_row.get("feature_edge_ids")
            != sorted(seed_row.get("feature_edge_ids", ()))
            or len(seed_row.get("feature_edge_ids", ())) != FEATURE_DIMENSION
            or seed_row.get("seed_payload_sha256")
            != _record_self_hash(seed_row, "seed_payload_sha256")
        ):
            raise RuntimeError("alignment seed evidence is malformed")
        cases = seed_row.get("cases")
        if not isinstance(cases, list) or len(cases) != EPISODES_PER_SEED:
            raise RuntimeError("alignment seed has the wrong case count")
        theta_expected = np.zeros(FEATURE_DIMENSION, dtype=np.float64)
        for index, row in enumerate(cases):
            if (
                row.get("episode_index") != index
                or row.get("theta_before") != theta_expected.tolist()
                or row.get("cue") not in (0, 1)
                or row.get("target")
                != (1.0 if row.get("cue") == 1 else -1.0)
                or row.get("query_tick") != 11
            ):
                raise RuntimeError("alignment online head sequence is invalid")
            _validate_case(row)
            theta_expected = _as_vector(row["theta_after"], name="theta_after")
        case_passes = sum(bool(row["case_pass"]) for row in cases)
        coordinate_passes = sum(int(row["coordinate_pass_count"]) for row in cases)
        expected_seed_pass = case_passes == EPISODES_PER_SEED
        expected_cosines = all(bool(row["direction_cosine_pass"]) for row in cases)
        expected_dots = all(bool(row["positive_dot_pass"]) for row in cases)
        expected_updates = all(
            bool(row["update_reconstruction_pass"]) for row in cases
        )
        expected_topology = (
            seed_row.get("topology_sha256_before")
            == seed_row.get("topology_sha256_after")
        )
        expected_weights = (
            seed_row.get("weights_sha256_before")
            == seed_row.get("weights_sha256_after")
        )
        expected_recurrent = (
            seed_row.get("recurrent_weights_sha256_before")
            == seed_row.get("recurrent_weights_sha256_after")
        )
        if (
            seed_row.get("theta_final") != theta_expected.tolist()
            or seed_row.get("theta_final_sha256") != _array_sha256(theta_expected)
            or seed_row.get("case_count") != EPISODES_PER_SEED
            or seed_row.get("case_pass_count") != case_passes
            or seed_row.get("coordinate_check_count")
            != EPISODES_PER_SEED * FEATURE_DIMENSION
            or seed_row.get("coordinate_pass_count") != coordinate_passes
            or seed_row.get("all_cases_pass") is not expected_seed_pass
            or seed_row.get("all_coordinates_pass")
            is not (coordinate_passes == EPISODES_PER_SEED * FEATURE_DIMENSION)
            or seed_row.get("all_direction_cosines_pass") is not expected_cosines
            or seed_row.get("all_positive_dots_pass") is not expected_dots
            or seed_row.get("all_updates_reconstruct") is not expected_updates
            or seed_row.get("topology_unchanged") is not expected_topology
            or seed_row.get("all_graph_weights_unchanged") is not expected_weights
            or seed_row.get("recurrent_weights_unchanged") is not expected_recurrent
        ):
            raise RuntimeError("alignment seed aggregates do not reconstruct")
        total_case_passes += case_passes
        total_coordinate_passes += coordinate_passes
    total_cases = len(OFFICIAL_SEEDS) * EPISODES_PER_SEED
    total_coordinates = total_cases * FEATURE_DIMENSION
    expected_gates = {
        "case_pass_fraction_is_one": total_case_passes == total_cases,
        "coordinate_pass_fraction_is_one": total_coordinate_passes
        == total_coordinates,
        "all_direction_cosines_pass": all(
            all(case.get("direction_cosine_pass") is True for case in row["cases"])
            for row in seeds
        ),
        "all_positive_dots_pass": all(
            all(case.get("positive_dot_pass") is True for case in row["cases"])
            for row in seeds
        ),
        "all_updates_reconstruct": all(
            all(case.get("update_reconstruction_pass") is True for case in row["cases"])
            for row in seeds
        ),
        "all_topologies_unchanged": all(
            row.get("topology_sha256_before") == row.get("topology_sha256_after")
            for row in seeds
        ),
        "all_graph_weights_unchanged": all(
            row.get("weights_sha256_before") == row.get("weights_sha256_after")
            for row in seeds
        ),
        "all_recurrent_weights_unchanged": all(
            row.get("recurrent_weights_sha256_before")
            == row.get("recurrent_weights_sha256_after")
            for row in seeds
        ),
    }
    all_pass = all(expected_gates.values())
    expected_status = (
        "TRACE_HEAD_ALIGNMENT_PASS" if all_pass else "TRACE_HEAD_ALIGNMENT_FAIL"
    )
    if (
        report.get("total_case_count") != total_cases
        or report.get("passed_case_count") != total_case_passes
        or report.get("total_coordinate_check_count") != total_coordinates
        or report.get("passed_coordinate_check_count") != total_coordinate_passes
        or report.get("gates") != expected_gates
        or report.get("all_gates_pass") is not all_pass
        or report.get("status") != expected_status
    ):
        raise RuntimeError("alignment report aggregate gates do not reconstruct")


def _validate_official_arguments(
    *, seeds: Sequence[int], episodes_per_seed: int, output_path: str | Path
) -> tuple[int, ...]:
    resolved = tuple(seeds)
    if resolved != OFFICIAL_SEEDS:
        raise ValueError("official TRACE-HEAD-L1 seeds must be exactly 75--79 in order")
    if episodes_per_seed != EPISODES_PER_SEED:
        raise ValueError("official TRACE-HEAD-L1 requires exactly 32 episodes per seed")
    target = _project_path(output_path).resolve()
    valid = {
        _project_path(PRIMARY_REPORT_PATH).resolve(),
        _project_path(RERUN_REPORT_PATH).resolve(),
    }
    if target not in valid:
        raise ValueError("official TRACE-HEAD-L1 output path is not registered")
    for seed in resolved:
        require_seed_role(seed, SeedRole.CUSTOM)
    return resolved


def _expected_alignment_determinism_record(
    primary: Mapping[str, Any],
    rerun: Mapping[str, Any],
    *,
    primary_files: Mapping[str, str],
    rerun_files: Mapping[str, str],
    a3_binding: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    primary_provenance = primary.get("nondeterministic_provenance", {})
    rerun_provenance = rerun.get("nondeterministic_provenance", {})
    first_process_id = primary_provenance.get("process_id")
    second_process_id = rerun_provenance.get("process_id")
    first_token = primary_provenance.get("process_instance_token")
    second_token = rerun_provenance.get("process_instance_token")
    first_started = primary_provenance.get("run_started_utc")
    second_started = rerun_provenance.get("run_started_utc")
    fresh_processes = (
        type(first_process_id) is int
        and first_process_id > 0
        and type(second_process_id) is int
        and second_process_id > 0
        and first_process_id != second_process_id
        and _is_sha256(first_token)
        and _is_sha256(second_token)
        and first_token != second_token
        and _is_canonical_utc_timestamp(first_started, require_z=True)
        and _is_canonical_utc_timestamp(second_started, require_z=True)
        and first_started != second_started
        and primary_provenance.get("artifact_role") == "primary"
        and rerun_provenance.get("artifact_role") == "rerun"
    )
    primary_prerequisite_valid = (
        primary_provenance.get("primary_prerequisite") is None
        and rerun_provenance.get("primary_prerequisite") == dict(primary_files)
    )
    first_hash = primary.get("deterministic_payload_sha256")
    second_hash = rerun.get("deterministic_payload_sha256")
    first_recomputed = _recompute_report_payload_sha256(primary)
    second_recomputed = _recompute_report_payload_sha256(rerun)
    first_self_hash_valid = _is_sha256(first_hash) and first_hash == first_recomputed
    second_self_hash_valid = _is_sha256(second_hash) and second_hash == second_recomputed
    payload_hashes_match = first_hash == second_hash
    canonical_payloads_equal = _canonical_json(
        _report_payload_object(primary)
    ) == _canonical_json(_report_payload_object(rerun))
    statuses_match = primary.get("status") == rerun.get("status")
    if not (
        first_self_hash_valid
        and second_self_hash_valid
        and payload_hashes_match
        and canonical_payloads_equal
        and statuses_match
    ):
        status = "NONDETERMINISTIC_INVALID"
    elif not (fresh_processes and primary_prerequisite_valid):
        status = "INVALID_PROCEDURE_NONSELECTING"
    else:
        status = str(primary["status"])
    verification: dict[str, Any] = {
        "schema_version": "experiment-000-trace-head-l1-determinism-v1",
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "valid_terminal_status": status
        in {"TRACE_HEAD_ALIGNMENT_PASS", "TRACE_HEAD_ALIGNMENT_FAIL"},
        "deterministic_payload_hashes_match": payload_hashes_match,
        "canonical_deterministic_payloads_equal": canonical_payloads_equal,
        "scientific_statuses_match": statuses_match,
        "fresh_processes_verified": fresh_processes,
        "primary_prerequisite_verified": primary_prerequisite_valid,
        "first_process_id": first_process_id,
        "second_process_id": second_process_id,
        "first_process_instance_token": first_token,
        "second_process_instance_token": second_token,
        "first_run_started_utc": first_started,
        "second_run_started_utc": second_started,
        "first_deterministic_payload_sha256": first_hash,
        "second_deterministic_payload_sha256": second_hash,
        "first_recomputed_payload_sha256": first_recomputed,
        "second_recomputed_payload_sha256": second_recomputed,
        "first_self_hash_valid": first_self_hash_valid,
        "second_self_hash_valid": second_self_hash_valid,
        "first_report_file_sha256": primary_files.get("report_file_sha256"),
        "first_sidecar_file_sha256": primary_files.get("sidecar_file_sha256"),
        "second_report_file_sha256": rerun_files.get("report_file_sha256"),
        "second_sidecar_file_sha256": rerun_files.get("sidecar_file_sha256"),
        "a3_selection_binding_sha256": _sha256_json(a3_binding),
        "source_manifest_sha256": source_manifest.get("bundle_sha256"),
    }
    verification["verification_payload_sha256"] = _record_self_hash(
        verification, "verification_payload_sha256"
    )
    return verification


def _validate_alignment_determinism_record(
    verification: Mapping[str, Any],
    primary: Mapping[str, Any],
    rerun: Mapping[str, Any],
    *,
    primary_files: Mapping[str, str],
    rerun_files: Mapping[str, str],
    a3_binding: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    require_pass: bool,
) -> None:
    expected = _expected_alignment_determinism_record(
        primary,
        rerun,
        primary_files=primary_files,
        rerun_files=rerun_files,
        a3_binding=a3_binding,
        source_manifest=source_manifest,
    )
    if (
        not isinstance(verification, dict)
        or _canonical_json(verification) != _canonical_json(expected)
    ):
        raise RuntimeError("TRACE-HEAD-L1 determinism verification is inconsistent")
    if require_pass and not (
        primary.get("status")
        == rerun.get("status")
        == verification.get("status")
        == "TRACE_HEAD_ALIGNMENT_PASS"
    ):
        raise RuntimeError("TRACE-HEAD-L1 terminal status is not a verified pass")


def run_trace_head_alignment(
    *,
    seeds: Sequence[int],
    episodes_per_seed: int,
    output_path: str | Path,
) -> dict[str, Any]:
    """Run one official primary or rerun execution and persist its evidence."""

    resolved = _validate_official_arguments(
        seeds=seeds,
        episodes_per_seed=episodes_per_seed,
        output_path=output_path,
    )
    validate_protocol_config()
    a3_binding = load_verified_a3_selection()
    source_start = _source_manifest()
    target = _project_path(output_path)
    if target.exists() or target.with_suffix(".sha256").exists():
        raise RuntimeError("registered alignment output already exists and is immutable")
    target_is_rerun = target.resolve() == _project_path(RERUN_REPORT_PATH).resolve()
    primary_binding: dict[str, str] | None = None
    if target_is_rerun:
        primary, primary_files = _read_report_with_sidecar(PRIMARY_REPORT_PATH)
        _validate_alignment_report(
            primary,
            expected_a3_binding=a3_binding,
            current_manifest=source_start,
        )
        primary_token = primary.get("nondeterministic_provenance", {}).get(
            "process_instance_token"
        )
        primary_process_id = primary.get("nondeterministic_provenance", {}).get(
            "process_id"
        )
        primary_role = primary.get("nondeterministic_provenance", {}).get(
            "artifact_role"
        )
        primary_prerequisite = primary.get("nondeterministic_provenance", {}).get(
            "primary_prerequisite"
        )
        if (
            not isinstance(primary_token, str)
            or primary_token == _PROCESS_INSTANCE_TOKEN
            or primary_process_id == os.getpid()
            or primary_role != "primary"
            or primary_prerequisite is not None
        ):
            raise RuntimeError("alignment rerun requires a fresh process")
        primary_binding = dict(primary_files)
    started = perf_counter()
    run_started_utc = datetime.now(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )
    core = _run_alignment_core(
        seeds=resolved,
        episodes_per_seed=episodes_per_seed,
        retention=float(a3_binding["trace_retention"]),
    )
    source_end = _source_manifest()
    if source_end != source_start:
        raise RuntimeError("alignment source manifest changed during execution")
    if load_verified_a3_selection() != a3_binding:
        raise RuntimeError("A3 selection artifacts changed during alignment execution")
    status = (
        "TRACE_HEAD_ALIGNMENT_PASS"
        if core["all_gates_pass"]
        else "TRACE_HEAD_ALIGNMENT_FAIL"
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "run_kind": "full",
        "claim_scope": "standalone_head_alignment_only_not_recurrent_temporal_credit",
        "seeds": list(resolved),
        "episodes_per_seed": episodes_per_seed,
        "noise_events": NOISE_EVENTS,
        "feature_dimension": FEATURE_DIMENSION,
        "target_encoding": {"cue_0": -1.0, "cue_1": 1.0},
        "head_contract": _head_contract(),
        "finite_difference_contract": _finite_difference_contract(),
        "a3_selection_binding": a3_binding,
        "source_manifest_start": source_start,
        "source_manifest_end": source_end,
        **core,
        "status": status,
    }
    report["deterministic_payload_sha256"] = _sha256_json(report)
    report["nondeterministic_provenance"] = {
        "excluded_from_deterministic_payload_sha256": True,
        "artifact_role": "rerun" if target_is_rerun else "primary",
        "process_id": os.getpid(),
        "process_instance_token": _PROCESS_INSTANCE_TOKEN,
        "run_started_utc": run_started_utc,
        "runtime_seconds": perf_counter() - started,
        "primary_prerequisite": primary_binding,
    }
    _validate_alignment_report(
        report,
        expected_a3_binding=a3_binding,
        current_manifest=source_end,
    )
    if _source_manifest() != source_end or load_verified_a3_selection() != a3_binding:
        raise RuntimeError("alignment prerequisites changed before persistence")
    written, sidecar = _write_report_and_sidecar(target, report)
    persisted, persisted_files = _read_report_with_sidecar(written)
    _validate_alignment_report(
        persisted,
        expected_a3_binding=a3_binding,
        current_manifest=source_end,
    )
    if persisted != report or persisted_files["sidecar_file_sha256"] != _file_sha256(
        sidecar
    ):
        raise RuntimeError("persisted alignment report did not reopen exactly")
    if _source_manifest() != source_end or load_verified_a3_selection() != a3_binding:
        raise RuntimeError("alignment prerequisites changed during persistence")
    return report


def verify_deterministic_full_runs(
    first: str | Path = PRIMARY_REPORT_PATH,
    second: str | Path = RERUN_REPORT_PATH,
    *,
    output_path: str | Path = DETERMINISM_VERIFICATION_PATH,
) -> dict[str, Any]:
    """Verify independent official runs and persist the terminal head gate."""

    if (
        _project_path(first).resolve() != _project_path(PRIMARY_REPORT_PATH).resolve()
        or _project_path(second).resolve() != _project_path(RERUN_REPORT_PATH).resolve()
        or _project_path(output_path).resolve()
        != _project_path(DETERMINISM_VERIFICATION_PATH).resolve()
    ):
        raise ValueError("alignment determinism verification requires registered paths")
    validate_protocol_config()
    a3_binding = load_verified_a3_selection()
    source = _source_manifest()
    primary, primary_files = _read_report_with_sidecar(first)
    rerun, rerun_files = _read_report_with_sidecar(second)
    _validate_alignment_report(
        primary, expected_a3_binding=a3_binding, current_manifest=source
    )
    _validate_alignment_report(
        rerun, expected_a3_binding=a3_binding, current_manifest=source
    )
    verification = _expected_alignment_determinism_record(
        primary,
        rerun,
        primary_files=primary_files,
        rerun_files=rerun_files,
        a3_binding=a3_binding,
        source_manifest=source,
    )
    _validate_alignment_determinism_record(
        verification,
        primary,
        rerun,
        primary_files=primary_files,
        rerun_files=rerun_files,
        a3_binding=a3_binding,
        source_manifest=source,
        require_pass=False,
    )
    if _source_manifest() != source or load_verified_a3_selection() != a3_binding:
        raise RuntimeError("alignment prerequisites changed before verification write")
    target = _project_path(output_path)
    if target.exists():
        raise RuntimeError("alignment determinism verification is immutable")
    target = _atomic_write_json(target, verification)
    reopened = json.loads(target.read_text(encoding="utf-8"))
    _validate_alignment_determinism_record(
        reopened,
        primary,
        rerun,
        primary_files=primary_files,
        rerun_files=rerun_files,
        a3_binding=a3_binding,
        source_manifest=source,
        require_pass=False,
    )
    if reopened != verification:
        raise RuntimeError("alignment determinism verification did not reopen exactly")
    if _source_manifest() != source or load_verified_a3_selection() != a3_binding:
        raise RuntimeError("alignment prerequisites changed during verification write")
    return verification


def load_verified_trace_head_alignment() -> dict[str, Any]:
    """Revalidate and return the terminal TRACE-HEAD-L1 prerequisite binding."""

    validate_protocol_config()
    a3_binding = load_verified_a3_selection()
    source = _source_manifest()
    primary, primary_files = _read_report_with_sidecar(PRIMARY_REPORT_PATH)
    rerun, rerun_files = _read_report_with_sidecar(RERUN_REPORT_PATH)
    _validate_alignment_report(
        primary, expected_a3_binding=a3_binding, current_manifest=source
    )
    _validate_alignment_report(
        rerun, expected_a3_binding=a3_binding, current_manifest=source
    )
    target = _project_path(DETERMINISM_VERIFICATION_PATH)
    if not target.is_file():
        raise RuntimeError("TRACE-HEAD-L1 determinism verification is absent")
    verification = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(verification, dict):
        raise TypeError("TRACE-HEAD-L1 determinism verification must be an object")
    _validate_alignment_determinism_record(
        verification,
        primary,
        rerun,
        primary_files=primary_files,
        rerun_files=rerun_files,
        a3_binding=a3_binding,
        source_manifest=source,
        require_pass=True,
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "TRACE_HEAD_ALIGNMENT_PASS",
        "selected_condition": a3_binding["selected_condition"],
        "trace_retention": a3_binding["trace_retention"],
        "primary_report_file_sha256": primary_files["report_file_sha256"],
        "primary_sidecar_file_sha256": primary_files["sidecar_file_sha256"],
        "rerun_report_file_sha256": rerun_files["report_file_sha256"],
        "rerun_sidecar_file_sha256": rerun_files["sidecar_file_sha256"],
        "deterministic_payload_sha256": primary["deterministic_payload_sha256"],
        "determinism_verification_file_sha256": _file_sha256(target),
        "determinism_verification_payload_sha256": verification[
            "verification_payload_sha256"
        ],
        "source_manifest_sha256": source["bundle_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run or verify the preregistered TRACE-HEAD-L1 exact-alignment gate."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--seeds", type=int, nargs="+", required=True)
    run_parser.add_argument("--episodes-per-seed", type=int, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--first", type=Path, required=True)
    verify_parser.add_argument("--second", type=Path, required=True)
    verify_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        report = run_trace_head_alignment(
            seeds=tuple(args.seeds),
            episodes_per_seed=args.episodes_per_seed,
            output_path=args.output,
        )
        target = _project_path(args.output)
        summary = {
            "status": report["status"],
            "report_path": str(target),
            "report_file_sha256": _file_sha256(target),
            "sidecar_path": str(target.with_suffix(".sha256")),
            "sidecar_file_sha256": _file_sha256(target.with_suffix(".sha256")),
            "deterministic_payload_sha256": report["deterministic_payload_sha256"],
        }
    else:
        verification = verify_deterministic_full_runs(
            args.first, args.second, output_path=args.output
        )
        target = _project_path(args.output)
        summary = {
            "status": verification["status"],
            "verification_path": str(target),
            "verification_file_sha256": _file_sha256(target),
            "verification_payload_sha256": verification[
                "verification_payload_sha256"
            ],
        }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
