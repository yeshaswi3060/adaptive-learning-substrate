"""Prospectively registered LWOH-L1 V3 execution and evidence controls.

This module is intentionally separate from the historical V2 controller.  It
contains the complete post-A3 runner, but assigned-seed execution remains
fail-closed until the repaired-lineage handoff and pre-metric freeze validate.
"""

from __future__ import annotations

import argparse
import ast
import csv
import ctypes
import hashlib
import itertools
import json
import math
import multiprocessing
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .events import EventLog
from .experiment_000 import FROZEN_GRAPH_OPTIONS
from .recurrent import (
    EventMessage,
    QueryResult,
    RecurrentEventGraph,
    UnitEvent,
    build_experiment_000_graph,
)

PROTOCOL_VERSION = "experiment-000-lwoh-l1-v3"
SCIENTIFIC_RNG_NAMESPACE = "experiment-000-lwoh-l1-v1"
PROTOCOL_PATH = Path("docs/EXPERIMENT_000_LWOH_L1_V3_PROTOCOL.md")
CONFIG_PATH = Path("configs/experiment_000_lwoh_l1_v3.toml")
RUNNER_PATH = Path(
    "src/adaptive_learning_substrate/experiment_000_lwoh_l1_execution_v3.py"
)
TEST_PATH = Path("tests/test_experiment_000_lwoh_l1_execution_v3.py")
ARTIFACT_DIR = Path("artifacts/experiment_000/lwoh_l1_v3")
SCRATCH_ARTIFACT_DIR = Path("artifacts/experiment_000/lwoh_l1_v3_scratch")
LINEAGE_PATH = Path(
    "artifacts/evidence_integrity_repair_2026-09-04/PRE_METRIC_LINEAGE.json"
)
A3_TERMINAL_PATH = Path(
    "artifacts/experiment_000/readout_trace_a3/DETERMINISM_VERIFICATION.json"
)
V2_DIR = Path("artifacts/experiment_000/lwoh_l1_v2")

ACTIVATION_PATH = ARTIFACT_DIR / "ACTIVATION_VERIFICATION.json"
FREEZE_PATH = ARTIFACT_DIR / "PRE_METRIC_FREEZE.json"
PHASE_LEDGER_PATH = ARTIFACT_DIR / "PHASE_LEDGER.json"
TEST_VERIFICATION_PATH = ARTIFACT_DIR / "TEST_VERIFICATION.json"
SCRATCH_REPORT_PATH = SCRATCH_ARTIFACT_DIR / "smoke.json"
SCRATCH_VERIFICATION_PATH = SCRATCH_ARTIFACT_DIR / "SMOKE_VERIFICATION.json"
FREEZE_VERIFICATION_PATH = ARTIFACT_DIR / "FREEZE_VERIFICATION.json"
PHASE_RECEIPT_DIR = ARTIFACT_DIR / "phases"
ADMISSION_PRIMARY_PATH = ARTIFACT_DIR / "admission_seeds_90_94_pairs_100.json"
ADMISSION_RERUN_PATH = ARTIFACT_DIR / "admission_seeds_90_94_pairs_100_rerun.json"
ADMISSION_VERIFICATION_PATH = ARTIFACT_DIR / "ADMISSION_DETERMINISM_VERIFICATION.json"
IMPLEMENTATION_FREEZE_PATH = ARTIFACT_DIR / "IMPLEMENTATION_SOURCE_FREEZE.json"
LEARNING_PRIMARY_PATH = ARTIFACT_DIR / "learning_seeds_95_104.json"
LEARNING_RERUN_PATH = ARTIFACT_DIR / "learning_seeds_95_104_rerun.json"
LEARNING_VERIFICATION_PATH = ARTIFACT_DIR / "LEARNING_DETERMINISM_VERIFICATION.json"
READINESS_PATH = ARTIFACT_DIR / "READINESS_VERIFICATION.json"

ADMISSION_SEEDS = (90, 91, 92, 93, 94)
LEARNING_SEEDS = (95, 96, 97, 98, 99, 100, 101, 102, 103, 104)
SCRATCH_SEEDS = (9090, 9091)
CONFIRMATORY_SEEDS = tuple(range(1000, 1020))
BASELINES = (
    "random",
    "frozen_head",
    "latest_state",
    "independent_label",
    "ridge",
    "visible_cue",
)
CONDITION_IDS = (
    "lwoh_head",
    "latest_head",
    "lwoh_independent_label",
    "frozen_lwoh_head",
    "rand",
    "ridge_lwoh",
    "ridge_latest",
    "visible_cue_hold",
)
DELAYS = (4, 8, 16)
HORIZON = 32
ANALYSIS_SEED = 2026090302
RIDGE_ALPHA = 1e-3
RIDGE_ACTIVE_EPSILON = 1e-15
_PROCESS_INSTANCE_TOKEN = hashlib.sha256(os.urandom(32)).hexdigest()
_PROCESS_STARTED_UTC = datetime.now(UTC).isoformat()
_OFFICIAL_EXECUTION_AUTHORITY = object()

PHASES = (
    "activation",
    "freeze",
    "freeze_verification",
    "tests",
    "scratch_smoke",
    "scratch_verification",
    "admission_primary",
    "admission_rerun",
    "admission_verification",
    "implementation_source_revalidation",
    "learning_primary",
    "learning_rerun",
    "learning_verification",
    "readiness_verification",
)

_OFFICIAL_REPORTS = (
    ACTIVATION_PATH,
    FREEZE_PATH,
    FREEZE_VERIFICATION_PATH,
    PHASE_LEDGER_PATH,
    TEST_VERIFICATION_PATH,
    SCRATCH_REPORT_PATH,
    SCRATCH_VERIFICATION_PATH,
    ADMISSION_PRIMARY_PATH,
    ADMISSION_RERUN_PATH,
    ADMISSION_VERIFICATION_PATH,
    IMPLEMENTATION_FREEZE_PATH,
    LEARNING_PRIMARY_PATH,
    LEARNING_RERUN_PATH,
    LEARNING_VERIFICATION_PATH,
    READINESS_PATH,
)


def _project_root(project_root: str | Path | None = None) -> Path:
    return Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()


def _resolve(root: Path, path: str | Path) -> Path:
    resolved = (root / Path(path)).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path escapes project root: {path}")
    return resolved


def _canonical_object_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_bytes(value: Any) -> bytes:
    return _canonical_object_bytes(value) + b"\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_object_bytes(value))


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return _sha256_json(payload)


def _stable_bytes(path: Path) -> bytes:
    first = path.stat()
    data = path.read_bytes()
    second = path.stat()
    if (
        first.st_size != second.st_size
        or first.st_mtime_ns != second.st_mtime_ns
        or len(data) != first.st_size
    ):
        raise RuntimeError(f"file changed during read: {path}")
    return data


def _file_identity(root: Path, relative: str | Path) -> dict[str, Any]:
    path = _resolve(root, relative)
    data = _stable_bytes(path)
    return {
        "path": Path(relative).as_posix(),
        "bytes": len(data),
        "sha256": _sha256_bytes(data),
    }


def _read_json(path: Path) -> dict[str, Any]:
    data = _stable_bytes(path)
    value = json.loads(data)
    if not isinstance(value, dict):
        raise TypeError(f"JSON record must be an object: {path}")
    if _canonical_bytes(value) != data:
        raise RuntimeError(f"JSON is not canonical: {path}")
    return value


def _atomic_publish(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp, path)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite evidence: {path}") from None
        if _stable_bytes(path) != data:
            raise RuntimeError(f"published evidence differs on reopen: {path}")
    finally:
        temp.unlink(missing_ok=True)


def _atomic_publish_pair(path: Path, data: bytes, sidecar_data: bytes) -> None:
    """Publish a recoverable immutable report/sidecar transaction."""

    sidecar_path = path.with_suffix(".sha256")
    if path.exists() or sidecar_path.exists():
        raise FileExistsError(f"evidence pair already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_stage = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".pair-stage", dir=path.parent
    )
    stage = Path(raw_stage)
    sealed = stage.with_suffix(".sealed")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(stage, sealed)
        if sealed.read_bytes() != data:
            raise RuntimeError("staged report bytes differ")
        _atomic_publish(path, data)
        try:
            _atomic_publish(sidecar_path, sidecar_data)
        except BaseException:
            if sidecar_path.exists():
                path.unlink(missing_ok=True)
            raise
        if path.read_bytes() != data or sidecar_path.read_bytes() != sidecar_data:
            raise RuntimeError("published report/sidecar pair differs on reopen")
    finally:
        stage.unlink(missing_ok=True)
        sealed.unlink(missing_ok=True)


def _write_record(path: Path, record: dict[str, Any], *, self_field: str) -> dict[str, Any]:
    if self_field in record:
        raise ValueError(f"caller must not pre-populate {self_field}")
    final = dict(record)
    final[self_field] = _self_hash(final, self_field)
    data = _canonical_bytes(final)
    sidecar = {
        "algorithm": "sha256",
        "report_file": path.name,
        "report_sha256": _sha256_bytes(data),
    }
    sidecar_path = path.with_suffix(".sha256")
    sidecar_data = _canonical_bytes(sidecar)
    if sidecar_path.exists() and not path.exists():
        raise RuntimeError(f"orphan sidecar blocks evidence publication: {sidecar_path}")
    if path.exists():
        if _stable_bytes(path) != data:
            raise FileExistsError(f"refusing to overwrite evidence: {path}")
        if sidecar_path.exists():
            if _stable_bytes(sidecar_path) != sidecar_data:
                raise RuntimeError(f"existing sidecar differs: {sidecar_path}")
            return final
        _atomic_publish(sidecar_path, sidecar_data)
        return final
    _atomic_publish_pair(path, data, sidecar_data)
    return final


def _verify_record(path: Path, *, schema: str, self_field: str) -> dict[str, Any]:
    record = _read_json(path)
    if record.get("schema_version") != schema:
        raise RuntimeError(f"schema mismatch for {path}")
    if record.get(self_field) != _self_hash(record, self_field):
        raise RuntimeError(f"self-hash mismatch for {path}")
    sidecar = _read_json(path.with_suffix(".sha256"))
    expected = {
        "algorithm": "sha256",
        "report_file": path.name,
        "report_sha256": _sha256_bytes(_stable_bytes(path)),
    }
    if sidecar != expected:
        raise RuntimeError(f"sidecar mismatch for {path}")
    return record


@dataclass(frozen=True, slots=True)
class VisibleSample:
    sample_id: str
    cue: int | None
    noise: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.cue not in (-1, 1, None):
            raise ValueError("visible cue must be bipolar or absent")
        if any(value not in (-1, 1) for value in self.noise):
            raise ValueError("noise must be bipolar")


@dataclass(frozen=True, slots=True)
class Supervision:
    sample_id: str
    target: int

    def __post_init__(self) -> None:
        if self.target not in (-1, 1):
            raise ValueError("target must be bipolar")


@dataclass(frozen=True, slots=True)
class CellData:
    cell: str
    split: str
    samples: tuple[VisibleSample, ...]
    supervision: tuple[Supervision, ...]
    independent_labels: tuple[int, ...]
    random_predictions: tuple[int, ...]
    namespace_seeds: dict[str, int]


@dataclass(frozen=True, slots=True)
class EpisodeObservation:
    sample_id: str
    tick: int
    lwoh: tuple[float, ...]
    latest: tuple[float, ...]
    native_output: float
    structural_output: float
    emitted_events: int
    forward_edge_touches: int
    native_emitted_events: int
    native_forward_edge_touches: int
    observer_calls: int
    lwoh_kernel_calls: int
    latest_kernel_calls: int
    lwoh_writes: int
    latest_writes: int
    native_equivalent: bool
    native_event_sha256: str
    native_activation_sha256: str
    native_shadow_event_sha256: str
    native_shadow_activation_sha256: str


def _observation_record(observation: EpisodeObservation) -> dict[str, Any]:
    """Persist every registered raw observation and compute-ledger field."""

    return {
        "sample_id": observation.sample_id,
        "tick": observation.tick,
        "lwoh": list(observation.lwoh),
        "latest": list(observation.latest),
        "native_output": observation.native_output,
        "structural_output": observation.structural_output,
        "emitted_events": observation.emitted_events,
        "forward_edge_touches": observation.forward_edge_touches,
        "native_emitted_events": observation.native_emitted_events,
        "native_forward_edge_touches": observation.native_forward_edge_touches,
        "observer_calls": observation.observer_calls,
        "lwoh_kernel_calls": observation.lwoh_kernel_calls,
        "latest_kernel_calls": observation.latest_kernel_calls,
        "lwoh_writes": observation.lwoh_writes,
        "latest_writes": observation.latest_writes,
        "native_equivalent": observation.native_equivalent,
        "native_event_sha256": observation.native_event_sha256,
        "native_activation_sha256": observation.native_activation_sha256,
        "native_shadow_event_sha256": observation.native_shadow_event_sha256,
        "native_shadow_activation_sha256": observation.native_shadow_activation_sha256,
    }


@dataclass(frozen=True, slots=True)
class PendingPrediction:
    sequence: int
    feature: tuple[float, ...]
    score: float
    prediction: int


def _observer_kernel(
    *, gate: int, activation: float, memory: float, tick: int, memory_tick: int, written: bool
) -> tuple[float, int, bool]:
    """Apply the one registered scalar observer kernel for either gate choice."""

    if gate not in (0, 1):
        raise ValueError("observer gate must be zero or one")
    value = float(activation)
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise FloatingPointError("observer kernel received an invalid activation")
    complement = 1 - gate
    next_memory = float(np.clip(gate * value + complement * float(memory), -1.0, 1.0))
    next_tick = int(gate * int(tick) + complement * int(memory_tick))
    next_written = bool(written or gate == 1)
    return next_memory, next_tick, next_written


def _edge_state(graph: RecurrentEventGraph) -> list[dict[str, Any]]:
    weights = graph.weights
    return [
        {
            "edge_id": edge.edge_id,
            "source": edge.source,
            "destination": edge.destination,
            "kind": edge.kind,
            "delay_ticks": edge.delay_ticks,
            "plastic": edge.plastic,
            "weight": weights[edge.edge_id],
        }
        for edge in sorted(graph.edges, key=lambda item: item.edge_id)
    ]


def _recurrent_matrix_metrics(graph: RecurrentEventGraph) -> dict[str, Any]:
    ordered = tuple(sorted(graph.hidden_nodes))
    positions = {node: index for index, node in enumerate(ordered)}
    matrix = np.zeros((len(ordered), len(ordered)), dtype=np.float64)
    weights = graph.weights
    for edge in graph.edges:
        if edge.kind == "recurrent":
            matrix[positions[edge.destination], positions[edge.source]] = weights[edge.edge_id]
    eigenvalues = np.linalg.eigvals(matrix)
    spectral_radius = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    operator_norm = float(np.linalg.norm(matrix, ord=2)) if matrix.size else 0.0
    if not math.isfinite(spectral_radius) or not math.isfinite(operator_norm):
        raise FloatingPointError("recurrent matrix metric is non-finite")
    return {
        "hidden_nodes": list(ordered),
        "matrix_sha256": _array_sha256(matrix),
        "spectral_radius": spectral_radius,
        "operator_norm_2": operator_norm,
    }


def _graph_state(graph: RecurrentEventGraph) -> dict[str, Any]:
    edges = _edge_state(graph)
    return {
        "topology_sha256": graph.topology_hash(),
        "weights_sha256": graph.weights_hash(),
        "edge_state_sha256": _sha256_json(edges),
        "edge_count": len(edges),
        "edge_state": edges,
        "recurrent_matrix": _recurrent_matrix_metrics(graph),
    }


def _blas_state() -> dict[str, Any]:
    pools = [
        {
            "internal_api": row.get("internal_api"),
            "prefix": row.get("prefix"),
            "filepath": row.get("filepath"),
            "version": row.get("version"),
            "num_threads": row.get("num_threads"),
        }
        for row in threadpool_info()
    ]
    return {
        "pools": pools,
        "all_loaded_pools_single_threaded": all(row["num_threads"] == 1 for row in pools),
    }


def _peak_memory_bytes() -> int:
    if os.name != "nt":
        import resource

        usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return usage if sys.platform == "darwin" else usage * 1024

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("page_fault_count", ctypes.c_ulong),
            ("peak_working_set_size", ctypes.c_size_t),
            ("working_set_size", ctypes.c_size_t),
            ("quota_peak_paged_pool_usage", ctypes.c_size_t),
            ("quota_paged_pool_usage", ctypes.c_size_t),
            ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
            ("quota_non_paged_pool_usage", ctypes.c_size_t),
            ("pagefile_usage", ctypes.c_size_t),
            ("peak_pagefile_usage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(ProcessMemoryCounters)
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    process = get_current_process()
    get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    )
    get_process_memory_info.restype = ctypes.c_bool
    if not get_process_memory_info(process, ctypes.byref(counters), counters.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return int(counters.peak_working_set_size)


def _process_identity(source_manifest_sha256: str) -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    run_instance_token = uuid.uuid4().hex
    start_nonce = uuid.uuid4().hex
    return {
        "process_id": os.getpid(),
        "process_instance_token": _PROCESS_INSTANCE_TOKEN,
        "process_started_utc": _PROCESS_STARTED_UTC,
        "run_started_utc": datetime.now(UTC).isoformat(),
        "run_instance_token": run_instance_token,
        "start_nonce": start_nonce,
        "executable_sha256": _sha256_bytes(_stable_bytes(executable)),
        "source_manifest_sha256": source_manifest_sha256,
        "blas_state": _blas_state(),
        "peak_memory": _peak_memory_bytes(),
    }


def _runtime_source_sha256() -> str:
    frozen = os.environ.get("ALS_V3_SOURCE_MANIFEST_SHA256")
    return frozen if frozen else _sha256_bytes(_stable_bytes(Path(__file__).resolve()))


class LWOHObserverGraph(RecurrentEventGraph):
    """Native graph with two passive local observers on the same evaluations."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        cue_edges = tuple(
            sorted(
                (edge for edge in self.edges if edge.kind == "input" and edge.source == "cue"),
                key=lambda edge: edge.destination,
            )
        )
        if len(cue_edges) != 8 or len({edge.destination for edge in cue_edges}) != 8:
            raise RuntimeError("LWOH requires exactly eight distinct cue recipients")
        self.selected_nodes = tuple(edge.destination for edge in cue_edges)
        self.cue_edge_ids = tuple(edge.edge_id for edge in cue_edges)
        self.cue_edge_weights = tuple(self.weights[edge.edge_id] for edge in cue_edges)
        self._selected = frozenset(self.selected_nodes)
        self._lwoh_m: dict[str, float] = {}
        self._lwoh_kappa: dict[str, int] = {}
        self._lwoh_b: dict[str, bool] = {}
        self._latest_m: dict[str, float] = {}
        self._latest_kappa: dict[str, int] = {}
        self._latest_b: dict[str, bool] = {}
        self._observer_calls = 0
        self._lwoh_kernel_calls = 0
        self._latest_kernel_calls = 0
        self._lwoh_writes = 0
        self._latest_writes = 0
        self._observer_episode_resets = 0
        self._observer_calls_total = 0
        self._lwoh_kernel_calls_total = 0
        self._latest_kernel_calls_total = 0
        self._lwoh_writes_total = 0
        self._latest_writes_total = 0
        self._reset_observers(count_reset=False)

    @classmethod
    def from_seed(cls, seed: int) -> Self:
        base = build_experiment_000_graph(
            seed=int(seed), mode="full", event_log_enabled=False, **FROZEN_GRAPH_OPTIONS
        )
        result = cls(
            input_nodes=base.input_nodes,
            hidden_nodes=base.hidden_nodes,
            output_node=base.output_node,
            edges=base.edges,
            initial_weights=base.weights,
            mode=base.mode,
            learning_rate=base.learning_rate,
            trace_decay=base.trace_decay,
            route_gain=base.route_gain,
            emit_threshold=base.emit_threshold,
            epsilon_weight=base.epsilon_weight,
            epsilon_message=base.epsilon_message,
            epsilon_route=base.epsilon_route,
            credit_limit=base.credit_limit,
            credit_minimum=base.credit_minimum,
            trace_horizon=base.trace_horizon,
            hop_limit=base.hop_limit,
            max_update=base.max_update,
            weight_clip=base.weight_clip,
            event_log=EventLog(enabled=False),
        )
        if result.topology_hash() != base.topology_hash() or result.weights_hash() != base.weights_hash():
            raise RuntimeError("observer clone changed native graph identity")
        return result

    def _reset_observers(self, *, count_reset: bool = True) -> None:
        self._lwoh_m = {node: 0.0 for node in self.selected_nodes}
        self._lwoh_kappa = {node: 0 for node in self.selected_nodes}
        self._lwoh_b = {node: False for node in self.selected_nodes}
        self._latest_m = {node: 0.0 for node in self.selected_nodes}
        self._latest_kappa = {node: 0 for node in self.selected_nodes}
        self._latest_b = {node: False for node in self.selected_nodes}
        self._observer_calls = 0
        self._lwoh_kernel_calls = 0
        self._latest_kernel_calls = 0
        self._lwoh_writes = 0
        self._latest_writes = 0
        if count_reset:
            self._observer_episode_resets += 1

    def begin_episode(self, episode_id: str | int | None = None) -> str:
        resolved = super().begin_episode(episode_id)
        self._reset_observers()
        return resolved

    def _process_unit(
        self,
        node: str,
        messages: tuple[EventMessage, ...],
        *,
        forced_output: bool,
    ) -> tuple[UnitEvent | None, int]:
        result = super()._process_unit(node, messages, forced_output=forced_output)
        if node in self._selected:
            value = float(self._activations[node])
            self._observer_calls += 1
            self._observer_calls_total += 1
            lwoh_gate = 1 - int(self._lwoh_b[node])
            previous_lwoh_written = self._lwoh_b[node]
            (
                self._lwoh_m[node],
                self._lwoh_kappa[node],
                self._lwoh_b[node],
            ) = _observer_kernel(
                gate=lwoh_gate,
                activation=value,
                memory=self._lwoh_m[node],
                tick=self._tick,
                memory_tick=self._lwoh_kappa[node],
                written=self._lwoh_b[node],
            )
            self._lwoh_kernel_calls += 1
            self._lwoh_kernel_calls_total += 1
            if not previous_lwoh_written and self._lwoh_b[node]:
                self._lwoh_writes += 1
                self._lwoh_writes_total += 1
            (
                self._latest_m[node],
                self._latest_kappa[node],
                self._latest_b[node],
            ) = _observer_kernel(
                gate=1,
                activation=value,
                memory=self._latest_m[node],
                tick=self._tick,
                memory_tick=self._latest_kappa[node],
                written=self._latest_b[node],
            )
            self._latest_kernel_calls += 1
            self._latest_kernel_calls_total += 1
            self._latest_writes += 1
            self._latest_writes_total += 1
        return result

    def observer_features(
        self, kind: Literal["lwoh", "latest"], *, tick: int | None = None
    ) -> tuple[float, ...]:
        observed_tick = self._tick if tick is None else int(tick)
        if kind == "lwoh":
            values, times, written = self._lwoh_m, self._lwoh_kappa, self._lwoh_b
        elif kind == "latest":
            values, times, written = self._latest_m, self._latest_kappa, self._latest_b
        else:
            raise ValueError("observer kind must be lwoh or latest")
        result = tuple(
            float(values[node])
            if written[node] and 0 <= observed_tick - times[node] <= HORIZON
            else 0.0
            for node in self.selected_nodes
        )
        if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in result):
            raise FloatingPointError("observer produced an invalid feature")
        return result

    def structural_output(self, feature: Sequence[float]) -> float:
        if len(feature) != 8:
            raise ValueError("structural feature dimension must equal eight")
        return math.tanh(
            math.fsum(weight * float(value) for weight, value in zip(self.cue_edge_weights, feature, strict=True))
        )

    @property
    def observer_ledger(self) -> dict[str, int]:
        return {
            "kernel_calls": self._observer_calls,
            "lwoh_kernel_calls": self._lwoh_kernel_calls,
            "latest_kernel_calls": self._latest_kernel_calls,
            "lwoh_writes": self._lwoh_writes,
            "latest_writes": self._latest_writes,
            "episode_resets": self._observer_episode_resets,
            "kernel_calls_total": self._observer_calls_total,
            "lwoh_kernel_calls_total": self._lwoh_kernel_calls_total,
            "latest_kernel_calls_total": self._latest_kernel_calls_total,
            "lwoh_writes_total": self._lwoh_writes_total,
            "latest_writes_total": self._latest_writes_total,
            "state_float_coordinates_each": 8,
            "state_tick_coordinates_each": 8,
            "state_bit_coordinates_each": 8,
        }


def _derive_seed(master_seed: int, cell: str, split: str, purpose: str) -> int:
    payload = (
        f"{SCIENTIFIC_RNG_NAMESPACE}|{int(master_seed)}|{cell}|{split}|{purpose}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def _balanced_bipolar(count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 0 or count % 2:
        raise ValueError("balanced sample count must be positive and even")
    values = np.concatenate(
        (np.full(count // 2, -1, dtype=np.int8), np.full(count // 2, 1, dtype=np.int8))
    )
    rng.shuffle(values)
    return values


def generate_cell(
    *,
    master_seed: int,
    cell: str,
    split: str,
    count: int,
    noise_events: int,
    cue_visible: bool,
) -> CellData:
    if noise_events < 0:
        raise ValueError("noise event count must be non-negative")
    purposes = ("cue", "noise", "sample_identity", "independent_label", "random_prediction")
    seeds = {purpose: _derive_seed(master_seed, cell, split, purpose) for purpose in purposes}
    if len(set(seeds.values())) != len(seeds):
        raise RuntimeError("cell RNG namespaces collided")
    cue = _balanced_bipolar(count, np.random.Generator(np.random.PCG64(seeds["cue"])))
    noise_rng = np.random.Generator(np.random.PCG64(seeds["noise"]))
    noise = noise_rng.choice(np.asarray((-1, 1), dtype=np.int8), size=(count, noise_events))
    independent = _balanced_bipolar(
        count, np.random.Generator(np.random.PCG64(seeds["independent_label"]))
    )
    random_predictions = _balanced_bipolar(
        count,
        np.random.Generator(np.random.PCG64(seeds["random_prediction"])),
    )
    identity_rng = np.random.Generator(np.random.PCG64(seeds["sample_identity"]))
    samples: list[VisibleSample] = []
    supervision: list[Supervision] = []
    for index in range(count):
        token = int(identity_rng.integers(0, 2**63, dtype=np.int64))
        sample_id = f"{master_seed}:{cell}:{split}:{index:04d}:{token:016x}"
        target = int(cue[index])
        samples.append(
            VisibleSample(
                sample_id=sample_id,
                cue=target if cue_visible else None,
                noise=tuple(int(value) for value in noise[index]),
            )
        )
        supervision.append(Supervision(sample_id=sample_id, target=target))
    if sum(item.target == 1 for item in supervision) != count // 2:
        raise RuntimeError("cell target balance differs")
    return CellData(
        cell=cell,
        split=split,
        samples=tuple(samples),
        supervision=tuple(supervision),
        independent_labels=tuple(int(value) for value in independent),
        random_predictions=tuple(int(value) for value in random_predictions),
        namespace_seeds=seeds,
    )


def _forward_native_sample(graph: RecurrentEventGraph, sample: VisibleSample) -> QueryResult:
    graph.begin_episode(sample.sample_id)
    graph.step({} if sample.cue is None else {"cue": float(sample.cue)})
    for value in sample.noise:
        graph.step({"noise": float(value)})
    return graph.query()


def run_visible_sample(
    graph: LWOHObserverGraph,
    sample: VisibleSample,
    *,
    shadow: RecurrentEventGraph | None = None,
) -> EpisodeObservation:
    ledger_before = graph.ledger
    graph.begin_episode(sample.sample_id)
    graph.step({} if sample.cue is None else {"cue": float(sample.cue)})
    for value in sample.noise:
        graph.step({"noise": float(value)})
    query: QueryResult = graph.query()
    lwoh = graph.observer_features("lwoh", tick=query.tick)
    latest = graph.observer_features("latest", tick=query.tick)
    ledger_after = graph.ledger
    expected_tick = len(sample.noise) + 3
    if query.tick != expected_tick:
        raise RuntimeError(f"query tick {query.tick} differs from expected {expected_tick}")
    native_equivalent = shadow is None
    events = graph.unit_events
    shadow_events: tuple[UnitEvent, ...] = events
    native_emitted_events = ledger_after["emitted_unit_events"] - ledger_before["emitted_unit_events"]
    native_forward_edge_touches = (
        ledger_after["forward_edge_touches"] - ledger_before["forward_edge_touches"]
    )
    if shadow is not None:
        shadow_ledger_before = shadow.ledger
        shadow_query = _forward_native_sample(shadow, sample)
        shadow_ledger_after = shadow.ledger
        shadow_events = shadow.unit_events
        native_emitted_events = (
            shadow_ledger_after["emitted_unit_events"]
            - shadow_ledger_before["emitted_unit_events"]
        )
        native_forward_edge_touches = (
            shadow_ledger_after["forward_edge_touches"]
            - shadow_ledger_before["forward_edge_touches"]
        )
        native_equivalent = bool(
            query == shadow_query
            and events == shadow_events
            and graph.ledger == shadow.ledger
            and graph.topology_hash() == shadow.topology_hash()
            and graph.weights_hash() == shadow.weights_hash()
            and graph.edges == shadow.edges
            and graph.weights == shadow.weights
        )
        if not native_equivalent:
            raise RuntimeError("passive observer changed native execution")
    event_payload = repr(events).encode("utf-8")
    activation_payload = repr(
        tuple((event.event_id, event.preactivation, event.activation) for event in events)
    ).encode("utf-8")
    shadow_event_payload = repr(shadow_events).encode("utf-8")
    shadow_activation_payload = repr(
        tuple((event.event_id, event.preactivation, event.activation) for event in shadow_events)
    ).encode("utf-8")
    observer = graph.observer_ledger
    return EpisodeObservation(
        sample_id=sample.sample_id,
        tick=query.tick,
        lwoh=lwoh,
        latest=latest,
        native_output=float(query.activation),
        structural_output=graph.structural_output(lwoh),
        emitted_events=ledger_after["emitted_unit_events"] - ledger_before["emitted_unit_events"],
        forward_edge_touches=ledger_after["forward_edge_touches"] - ledger_before["forward_edge_touches"],
        native_emitted_events=native_emitted_events,
        native_forward_edge_touches=native_forward_edge_touches,
        observer_calls=observer["kernel_calls"],
        lwoh_kernel_calls=observer["lwoh_kernel_calls"],
        latest_kernel_calls=observer["latest_kernel_calls"],
        lwoh_writes=observer["lwoh_writes"],
        latest_writes=observer["latest_writes"],
        native_equivalent=native_equivalent,
        native_event_sha256=_sha256_bytes(event_payload),
        native_activation_sha256=_sha256_bytes(activation_payload),
        native_shadow_event_sha256=_sha256_bytes(shadow_event_payload),
        native_shadow_activation_sha256=_sha256_bytes(shadow_activation_payload),
    )


class OnlineHead:
    """Frozen eight-coordinate online tanh head from the V1 protocol."""

    def __init__(self) -> None:
        self.theta = np.zeros(8, dtype=np.float64)
        self.update_count = 0
        self.prediction_count = 0
        self._pending: PendingPrediction | None = None

    def score(self, feature: Sequence[float]) -> float:
        vector = np.asarray(feature, dtype=np.float64)
        if vector.shape != (8,) or not np.all(np.isfinite(vector)):
            raise ValueError("head feature must contain eight finite coordinates")
        value = math.tanh(
            math.fsum(float(weight) * float(item) for weight, item in zip(self.theta, vector, strict=True))
        )
        if not math.isfinite(value):
            raise FloatingPointError("head score is non-finite")
        return value

    def predict(self, feature: Sequence[float]) -> tuple[float, int]:
        score = self.score(feature)
        return score, 1 if score >= 0.0 else -1

    def predict_for_update(self, feature: Sequence[float]) -> PendingPrediction:
        if self._pending is not None:
            raise RuntimeError("target for the previous prediction was not consumed")
        score, prediction = self.predict(feature)
        vector = tuple(float(value) for value in feature)
        pending = PendingPrediction(
            sequence=self.prediction_count,
            feature=vector,
            score=score,
            prediction=prediction,
        )
        self._pending = pending
        self.prediction_count += 1
        return pending

    def update(self, feature: Sequence[float], target: int) -> np.ndarray:
        if target not in (-1, 1):
            raise ValueError("head target must be bipolar")
        vector = np.asarray(feature, dtype=np.float64)
        score = self.score(vector)
        normalizer = max(1.0, math.fsum(float(value) ** 2 for value in vector))
        scalar = 0.5 * (float(target) - score) * (1.0 - score * score) / normalizer
        delta = np.clip(scalar * vector, -0.05, 0.05)
        staged = np.clip(self.theta + delta, -3.0, 3.0)
        if not np.all(np.isfinite(staged)):
            raise FloatingPointError("head update produced a non-finite parameter")
        self.theta = staged.astype(np.float64, copy=False)
        self.update_count += 1
        return delta.astype(np.float64, copy=True)

    def reveal_target(self, pending: PendingPrediction, target: int) -> np.ndarray:
        if self._pending != pending:
            raise RuntimeError("target does not correspond to the pending prediction")
        self._pending = None
        return self.update(pending.feature, target)

    @property
    def chronology_ledger(self) -> dict[str, int | bool]:
        return {
            "predictions": self.prediction_count,
            "updates": self.update_count,
            "pending_prediction": self._pending is not None,
        }

    def digest(self) -> str:
        return _sha256_bytes(self.theta.astype("<f8", copy=False).tobytes(order="C"))


def _analytic_descent(theta: np.ndarray, feature: np.ndarray, target: float) -> np.ndarray:
    score = math.tanh(math.fsum(float(a) * float(b) for a, b in zip(theta, feature, strict=True)))
    normalizer = max(1.0, math.fsum(float(value) ** 2 for value in feature))
    return 0.5 * (target - score) * (1.0 - score * score) * feature / normalizer


def gradient_alignment_check() -> dict[str, Any]:
    theta = np.asarray((0.11, -0.07, 0.03, 0.09, -0.05, 0.02, -0.13, 0.04), dtype=np.float64)
    feature = np.asarray((0.3, -0.4, 0.2, 0.1, -0.25, 0.35, -0.15, 0.05), dtype=np.float64)
    target = 1.0
    analytic = _analytic_descent(theta, feature, target)
    epsilon = 1e-6

    def loss(value: np.ndarray) -> float:
        score = math.tanh(math.fsum(float(a) * float(b) for a, b in zip(value, feature, strict=True)))
        return 0.5 * (target - score) ** 2 / max(1.0, float(np.dot(feature, feature)))

    exact = np.zeros(8, dtype=np.float64)
    for index in range(8):
        plus, minus = theta.copy(), theta.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        exact[index] = -0.5 * (loss(plus) - loss(minus)) / (2.0 * epsilon)
    absolute = np.abs(analytic - exact)
    relative = absolute / np.maximum(np.abs(exact), 1e-30)
    denominator = float(np.linalg.norm(analytic) * np.linalg.norm(exact))
    cosine = float(np.dot(analytic, exact) / denominator)
    coordinate_pass = np.logical_or(absolute <= 1e-8, relative <= 1e-6)
    return {
        "coordinates": 8,
        "max_absolute_error": float(np.max(absolute)),
        "max_relative_error": float(np.max(relative)),
        "cosine": cosine,
        "positive_descent_dot_fraction": float(np.mean(analytic * exact > 0.0)),
        "all_coordinates_pass": bool(np.all(coordinate_pass)),
        "pass": bool(
            np.all(coordinate_pass)
            and cosine >= 0.999999
            and np.all(analytic * exact > 0.0)
        ),
    }


def _array_sha256(*arrays: np.ndarray) -> str:
    hasher = hashlib.sha256()
    for array in arrays:
        normalized = np.asarray(array, dtype="<f8", order="C")
        hasher.update(str(normalized.shape).encode("ascii"))
        hasher.update(normalized.tobytes(order="C"))
    return hasher.hexdigest()


def _paired_metrics(
    minus: np.ndarray, plus: np.ndarray, outputs_minus: np.ndarray, outputs_plus: np.ndarray
) -> dict[str, Any]:
    if minus.shape != plus.shape or minus.ndim != 2 or minus.shape[1] != 8:
        raise ValueError("paired feature arrays must have shape (pairs, 8)")
    delta = 0.5 * (plus - minus)
    midpoint = 0.5 * (plus + minus)
    cue_signal = float(np.sqrt(np.mean(np.sum(np.square(delta), axis=1))))
    noise = float(
        np.sqrt(np.mean(np.sum(np.square(midpoint - np.mean(midpoint, axis=0)), axis=1)))
    )
    output_delta = 0.5 * (outputs_plus - outputs_minus)
    output_cue = float(np.sqrt(np.mean(np.square(output_delta))))
    all_outputs = np.stack((outputs_minus, outputs_plus), axis=1).reshape(-1)
    output_sd = float(np.std(all_outputs, ddof=0))
    values = (cue_signal, noise, output_cue, output_sd)
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise FloatingPointError("paired metric is invalid")
    return {
        "C": cue_signal,
        "N": noise,
        "R": _guarded_ratio(cue_signal, noise),
        "output_cue_delta_rms": output_cue,
        "output_sd": output_sd,
        "O": _guarded_ratio(output_cue, output_sd),
        "feature_commitment_sha256": _array_sha256(minus, plus),
        "output_commitment_sha256": _array_sha256(outputs_minus, outputs_plus),
    }


def _guarded_ratio(numerator: float, denominator: float) -> dict[str, Any]:
    numerator, denominator = float(numerator), float(denominator)
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or numerator < 0.0
        or denominator < 0.0
    ):
        raise ValueError("guarded ratio requires finite non-negative inputs")
    if denominator != 0.0:
        return {
            "numerator": numerator,
            "denominator": denominator,
            "status": "finite",
            "value": numerator / denominator,
        }
    if numerator > 0.0:
        return {
            "numerator": numerator,
            "denominator": 0.0,
            "status": "positive_over_zero",
            "value": None,
        }
    return {
        "numerator": 0.0,
        "denominator": 0.0,
        "status": "zero_over_zero",
        "value": None,
    }


def _ratio_at_least(ratio: Mapping[str, Any], threshold: float) -> bool:
    status = ratio.get("status")
    if status == "zero_over_zero":
        return False
    if status == "positive_over_zero":
        return True
    if status != "finite":
        return False
    return bool(
        float(ratio["numerator"]) >= float(threshold) * float(ratio["denominator"])
    )


def _ratio_at_most(ratio: Mapping[str, Any], threshold: float) -> bool:
    status = ratio.get("status")
    if status == "zero_over_zero":
        return True
    if status == "positive_over_zero" or status != "finite":
        return False
    return bool(
        float(ratio["numerator"]) <= float(threshold) * float(ratio["denominator"])
    )


def _observed_ratio_at_most(value: Any, threshold: float) -> bool:
    """Accept persisted guarded ratios and legacy synthetic scalar fixtures."""

    if isinstance(value, Mapping):
        return _ratio_at_most(value, threshold)
    realized = float(value)
    return math.isfinite(realized) and realized <= threshold


def _ratio_multiple_at_least(
    candidate: Mapping[str, Any], native: Mapping[str, Any], multiple: float
) -> bool:
    c_num, c_den = float(candidate["numerator"]), float(candidate["denominator"])
    n_num, n_den = float(native["numerator"]), float(native["denominator"])
    candidate_status = candidate.get("status")
    native_status = native.get("status")
    if candidate_status == "zero_over_zero" or native_status == "zero_over_zero":
        return False
    if candidate_status == "positive_over_zero":
        return native_status == "finite"
    if native_status == "positive_over_zero":
        return False
    if candidate_status != "finite" or native_status != "finite":
        return False
    return c_num * n_den >= float(multiple) * n_num * c_den


def _median_guarded_ratio(
    rows: Sequence[Mapping[str, Any]], key: str
) -> dict[str, Any]:
    """Return the registered odd-sample median of per-seed guarded ratios."""

    ratios = [row[key] for row in rows]
    if not ratios or len(ratios) % 2 == 0:
        raise ValueError("guarded-ratio median requires a non-empty odd sample")
    statuses = [ratio.get("status") for ratio in ratios]
    if any(status == "zero_over_zero" for status in statuses):
        result = _guarded_ratio(0.0, 0.0)
        result["aggregation"] = "median_per_seed_guarded_ratio"
        result["component_statuses"] = statuses
        return result

    def order_key(item: tuple[int, Mapping[str, Any]]) -> tuple[int, Fraction, int]:
        index, ratio = item
        if ratio.get("status") == "positive_over_zero":
            return (1, Fraction(0, 1), index)
        if ratio.get("status") != "finite":
            raise ValueError("unknown guarded-ratio status")
        return (
            0,
            Fraction(float(ratio["numerator"])) / Fraction(float(ratio["denominator"])),
            index,
        )

    ordered = sorted(enumerate(ratios), key=order_key)
    source_index, source = ordered[len(ordered) // 2]
    result = dict(source)
    result["aggregation"] = "median_per_seed_guarded_ratio"
    result["source_seed_index"] = source_index
    result["component_statuses"] = statuses
    return result


def _ridge_fit(
    train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray, eval_y: np.ndarray
) -> dict[str, Any]:
    train_x = np.asarray(train_x, dtype=np.float64)
    eval_x = np.asarray(eval_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    eval_y = np.asarray(eval_y, dtype=np.float64)
    mean = np.mean(train_x, axis=0)
    scale = np.std(train_x, axis=0, ddof=0)
    active = scale != 0.0
    safe_scale = np.where(active, scale, 1.0)
    normalized_train = (train_x - mean) / safe_scale
    normalized_eval = (eval_x - mean) / safe_scale
    count = train_x.shape[0]
    gram = normalized_train.T @ normalized_train / count
    gram += RIDGE_ALPHA * np.eye(train_x.shape[1], dtype=np.float64)
    rhs = normalized_train.T @ train_y / count
    coefficients = np.linalg.solve(gram, rhs)
    intercept = float(np.mean(train_y))
    train_scores = normalized_train @ coefficients + intercept
    eval_scores = normalized_eval @ coefficients + intercept
    train_predictions = np.where(train_scores >= 0.0, 1, -1)
    eval_predictions = np.where(eval_scores >= 0.0, 1, -1)
    return {
        "alpha": RIDGE_ALPHA,
        "feature_count": int(train_x.shape[1]),
        "active_feature_count": int(np.count_nonzero(active)),
        "train_accuracy": float(np.mean(train_predictions == train_y)),
        "eval_accuracy": float(np.mean(eval_predictions == eval_y)),
        "coefficient_l2": float(np.linalg.norm(coefficients)),
        "coefficients": coefficients.tolist(),
        "intercept": intercept,
        "mean": mean.tolist(),
        "scale": safe_scale.tolist(),
        "eval_predictions": eval_predictions.astype(int).tolist(),
    }


def _paired_rows(
    seed: int, split: str, pairs: int, noise_events: int
) -> tuple[tuple[VisibleSample, VisibleSample], ...]:
    rng = np.random.Generator(
        np.random.PCG64(_derive_seed(seed, f"admission_d{noise_events}", split, "noise"))
    )
    noise = rng.choice(np.asarray((-1, 1), dtype=np.int8), size=(pairs, noise_events))
    rows: list[tuple[VisibleSample, VisibleSample]] = []
    for index in range(pairs):
        values = tuple(int(value) for value in noise[index])
        base = f"{seed}:admission:{split}:{index:04d}"
        rows.append(
            (
                VisibleSample(sample_id=f"{base}:minus", cue=-1, noise=values),
                VisibleSample(sample_id=f"{base}:plus", cue=1, noise=values),
            )
        )
    return tuple(rows)


def _collect_admission_split(
    graph: LWOHObserverGraph,
    shadow: RecurrentEventGraph,
    rows: Sequence[tuple[VisibleSample, VisibleSample]],
) -> dict[str, Any]:
    lwoh_minus: list[tuple[float, ...]] = []
    lwoh_plus: list[tuple[float, ...]] = []
    latest_minus: list[tuple[float, ...]] = []
    latest_plus: list[tuple[float, ...]] = []
    structural_minus: list[float] = []
    structural_plus: list[float] = []
    native_minus: list[float] = []
    native_plus: list[float] = []
    activity: list[dict[str, Any]] = []
    for minus, plus in rows:
        minus_observation = run_visible_sample(graph, minus, shadow=shadow)
        plus_observation = run_visible_sample(graph, plus, shadow=shadow)
        lwoh_minus.append(minus_observation.lwoh)
        lwoh_plus.append(plus_observation.lwoh)
        latest_minus.append(minus_observation.latest)
        latest_plus.append(plus_observation.latest)
        structural_minus.append(minus_observation.structural_output)
        structural_plus.append(plus_observation.structural_output)
        native_minus.append(minus_observation.native_output)
        native_plus.append(plus_observation.native_output)
        for observation in (minus_observation, plus_observation):
            event_ratio = _guarded_ratio(
                observation.emitted_events, observation.native_emitted_events
            )
            touch_ratio = _guarded_ratio(
                observation.forward_edge_touches,
                observation.native_forward_edge_touches,
            )
            activity.append(
                {
                    **_observation_record(observation),
                    "candidate_events": observation.emitted_events,
                    "native_events": observation.native_emitted_events,
                    "candidate_forward_edge_touches": observation.forward_edge_touches,
                    "native_forward_edge_touches": observation.native_forward_edge_touches,
                    "candidate_event_ratio": event_ratio,
                    "candidate_forward_touch_ratio": touch_ratio,
                }
            )
    arrays = {
        "lwoh_minus": np.asarray(lwoh_minus, dtype=np.float64),
        "lwoh_plus": np.asarray(lwoh_plus, dtype=np.float64),
        "latest_minus": np.asarray(latest_minus, dtype=np.float64),
        "latest_plus": np.asarray(latest_plus, dtype=np.float64),
        "structural_minus": np.asarray(structural_minus, dtype=np.float64),
        "structural_plus": np.asarray(structural_plus, dtype=np.float64),
        "native_minus": np.asarray(native_minus, dtype=np.float64),
        "native_plus": np.asarray(native_plus, dtype=np.float64),
    }
    split_event_ratio = _guarded_ratio(
        sum(int(row["candidate_events"]) for row in activity),
        sum(int(row["native_events"]) for row in activity),
    )
    split_touch_ratio = _guarded_ratio(
        sum(int(row["candidate_forward_edge_touches"]) for row in activity),
        sum(int(row["native_forward_edge_touches"]) for row in activity),
    )
    return {
        "arrays": {key: value.tolist() for key, value in arrays.items()},
        "pair_sample_ids": [[minus.sample_id, plus.sample_id] for minus, plus in rows],
        "paired_noise": [list(minus.noise) for minus, _ in rows],
        "candidate_metrics": _paired_metrics(
            arrays["lwoh_minus"],
            arrays["lwoh_plus"],
            arrays["structural_minus"],
            arrays["structural_plus"],
        ),
        "native_metrics": _paired_metrics(
            arrays["latest_minus"],
            arrays["latest_plus"],
            arrays["native_minus"],
            arrays["native_plus"],
        ),
        "activity": activity,
        "split_event_ratio": split_event_ratio,
        "split_forward_touch_ratio": split_touch_ratio,
    }


def _tail_assay(seed: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for cue in (-1, 1):
        graph = LWOHObserverGraph.from_seed(seed)
        shadow = build_experiment_000_graph(
            seed=seed, mode="full", event_log_enabled=False, **FROZEN_GRAPH_OPTIONS
        )
        graph.begin_episode(f"{seed}:tail:{cue}")
        shadow.begin_episode(f"{seed}:tail:{cue}")
        graph.step({"cue": float(cue)})
        shadow.step({"cue": float(cue)})
        final_window_hidden_emissions = 0
        for _ in range(1, 257):
            step = graph.step({})
            shadow_step = shadow.step({})
            if step != shadow_step or graph.unit_events != shadow.unit_events:
                raise RuntimeError("tail observer changed native execution")
            if 193 <= step.tick <= 256:
                final_window_hidden_emissions += sum(
                    event_id.rsplit(":", 2)[-2] in graph.hidden_nodes
                    for event_id in step.emitted_event_ids
                )
        effective = graph.observer_features("lwoh", tick=256)
        query = graph.query()
        shadow_query = shadow.query()
        if (
            query != shadow_query
            or graph.unit_events != shadow.unit_events
            or graph.ledger != shadow.ledger
            or graph.edges != shadow.edges
            or graph.weights != shadow.weights
        ):
            raise RuntimeError("tail wake observer changed native execution")
        wake_feature = graph.observer_features("lwoh", tick=query.tick)
        wake_output = graph.structural_output(wake_feature)
        rows.append(
            {
                "cue": cue,
                "final_window_hidden_emissions": final_window_hidden_emissions,
                "effective_state_at_256": list(effective),
                "effective_state_abs_max": max(abs(value) for value in effective),
                "wake_feature": list(wake_feature),
                "wake_output": wake_output,
            }
        )
    minus, plus = rows
    wake_feature_half = float(
        np.linalg.norm(np.asarray(plus["wake_feature"]) - np.asarray(minus["wake_feature"])) / 2.0
    )
    wake_output_half = abs(float(plus["wake_output"]) - float(minus["wake_output"])) / 2.0
    passed = bool(
        all(row["final_window_hidden_emissions"] == 0 for row in rows)
        and all(row["effective_state_abs_max"] == 0.0 for row in rows)
        and wake_feature_half <= 1e-3
        and wake_output_half <= 1e-3
    )
    return {
        "rows": rows,
        "wake_feature_half_difference_l2": wake_feature_half,
        "wake_output_half_difference_abs": wake_output_half,
        "pass": passed,
    }


def _admission_seed(seed: int, *, pairs: int = 100) -> dict[str, Any]:
    if seed in CONFIRMATORY_SEEDS:
        raise ValueError("confirmatory seeds are closed")
    graph = LWOHObserverGraph.from_seed(seed)
    shadow = build_experiment_000_graph(
        seed=seed, mode="full", event_log_enabled=False, **FROZEN_GRAPH_OPTIONS
    )
    graph_state_start = _graph_state(graph)
    shadow_state_start = _graph_state(shadow)
    topology_start, weights_start = graph.topology_hash(), graph.weights_hash()
    train = _collect_admission_split(graph, shadow, _paired_rows(seed, "train", pairs, 8))
    evaluation = _collect_admission_split(
        graph, shadow, _paired_rows(seed, "eval", pairs, 8)
    )
    train_targets = np.tile(np.asarray((-1.0, 1.0), dtype=np.float64), pairs)
    train_lwoh = np.stack(
        (np.asarray(train["arrays"]["lwoh_minus"]), np.asarray(train["arrays"]["lwoh_plus"])),
        axis=1,
    ).reshape(-1, 8)
    eval_lwoh = np.stack(
        (
            np.asarray(evaluation["arrays"]["lwoh_minus"]),
            np.asarray(evaluation["arrays"]["lwoh_plus"]),
        ),
        axis=1,
    ).reshape(-1, 8)
    train_latest = np.stack(
        (
            np.asarray(train["arrays"]["latest_minus"]),
            np.asarray(train["arrays"]["latest_plus"]),
        ),
        axis=1,
    ).reshape(-1, 8)
    eval_latest = np.stack(
        (
            np.asarray(evaluation["arrays"]["latest_minus"]),
            np.asarray(evaluation["arrays"]["latest_plus"]),
        ),
        axis=1,
    ).reshape(-1, 8)
    ridge_candidate = _ridge_fit(train_lwoh, train_targets, eval_lwoh, train_targets)
    ridge_native = _ridge_fit(train_latest, train_targets, eval_latest, train_targets)
    tail = _tail_assay(seed)
    invariants = {
        "finite": all(
            math.isfinite(value)
            for split in (train, evaluation)
            for key in ("candidate_metrics", "native_metrics")
            for metric, value in split[key].items()
            if metric in {"C", "N", "output_cue_delta_rms", "output_sd"}
        ),
        "feature_dimension_eight": len(graph.selected_nodes) == 8,
        "selected_nodes_lexical": tuple(sorted(graph.selected_nodes)) == graph.selected_nodes,
        "topology_unchanged": graph.topology_hash() == topology_start,
        "weights_unchanged": graph.weights_hash() == weights_start,
        "credit_packets_zero": len(graph.credit_packets) == 0,
        "ccf_calls_zero": len(graph.credit_packets) == 0,
        "observer_passive": graph.topology_hash() == topology_start
        and graph.weights_hash() == weights_start
        and all(
            row["native_equivalent"]
            for split in (train, evaluation)
            for row in split["activity"]
        ),
        "write_once_bound": graph.observer_ledger["lwoh_writes"] <= 8,
        "activity_ratios_exact_one": all(
            row["candidate_event_ratio"].get("status") == "finite"
            and row["candidate_event_ratio"].get("value") == 1.0
            and row["candidate_forward_touch_ratio"].get("status") == "finite"
            and row["candidate_forward_touch_ratio"].get("value") == 1.0
            for split in (train, evaluation)
            for row in split["activity"]
        ),
        "native_equivalence_all_episodes": all(
            row["native_equivalent"]
            for split in (train, evaluation)
            for row in split["activity"]
        ),
        "tail_pass": tail["pass"],
    }
    worker_identity = _process_identity(_runtime_source_sha256())
    return {
        "seed": seed,
        "worker_pid": os.getpid(),
        "worker_nonce": f"{os.getpid()}:{time.time_ns()}",
        "process_identity": worker_identity,
        "graph": {
            "selected_nodes": list(graph.selected_nodes),
            "cue_edge_ids": list(graph.cue_edge_ids),
            "cue_edge_weights": list(graph.cue_edge_weights),
            "state_start": graph_state_start,
            "state_end": _graph_state(graph),
            "native_shadow_state_start": shadow_state_start,
            "native_shadow_state_end": _graph_state(shadow),
        },
        "train": train,
        "eval": evaluation,
        "ridge_candidate": ridge_candidate,
        "ridge_native": ridge_native,
        "tail": tail,
        "invariants": invariants,
        "all_invariants_pass": all(invariants.values()),
    }


def _median(values: Iterable[float]) -> float:
    realized = tuple(float(value) for value in values)
    if not realized or not all(math.isfinite(value) for value in realized):
        raise ValueError("median requires finite values")
    return float(statistics.median(realized))


def _aggregate_guarded_ratio(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    return _guarded_ratio(
        _median(float(row[key]["numerator"]) for row in rows),
        _median(float(row[key]["denominator"]) for row in rows),
    )


def recompute_admission_gates(seed_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if tuple(int(row["seed"]) for row in seed_results) != ADMISSION_SEEDS:
        raise RuntimeError("admission seed rows are incomplete or out of order")
    candidate = [row["eval"]["candidate_metrics"] for row in seed_results]
    native = [row["eval"]["native_metrics"] for row in seed_results]
    candidate_c = [float(row["C"]) for row in candidate]
    native_c = [float(row["C"]) for row in native]
    candidate_r = [row["R"] for row in candidate]
    native_r = [row["R"] for row in native]
    aggregate_candidate_r = _median_guarded_ratio(candidate, "R")
    aggregate_native_r = _median_guarded_ratio(native, "R")
    aggregate_candidate_o = _median_guarded_ratio(candidate, "O")
    gates = {
        "A01": _median(candidate_c) >= 3.0 * _median(native_c),
        "A02": sum(a >= 3.0 * b for a, b in zip(candidate_c, native_c, strict=True)) >= 4,
        "A03": _ratio_at_least(aggregate_candidate_r, 0.50),
        "A04": _ratio_multiple_at_least(aggregate_candidate_r, aggregate_native_r, 3.0),
        "A05": sum(
            _ratio_multiple_at_least(a, b, 3.0)
            for a, b in zip(candidate_r, native_r, strict=True)
        )
        >= 4,
        "A06": _median(float(row["ridge_candidate"]["eval_accuracy"]) for row in seed_results)
        >= 0.70,
        "A07": aggregate_candidate_o.get("status") != "zero_over_zero"
        and _ratio_at_least(aggregate_candidate_o, 0.50),
        "A08": all(
            _observed_ratio_at_most(split["split_event_ratio"], 1.10)
            and _observed_ratio_at_most(split["split_forward_touch_ratio"], 1.10)
            and all(
                _observed_ratio_at_most(activity["candidate_event_ratio"], 1.15)
                and _observed_ratio_at_most(
                    activity["candidate_forward_touch_ratio"], 1.15
                )
                for activity in split["activity"]
            )
            for row in seed_results
            for split in (row["train"], row["eval"])
        ),
        "A09": all(bool(row["tail"]["pass"]) for row in seed_results),
        "A10": all(bool(row["all_invariants_pass"]) for row in seed_results),
    }
    failed = [key for key in (f"A{index:02d}" for index in range(1, 11)) if not gates[key]]
    return {
        "gates": gates,
        "all_admission_gates_pass": not failed,
        "failed_gate_ids": failed,
        "terminal": (
            "LWOH_ADMITTED:write_once_ttl32"
            if not failed
            else "LWOH_ADMISSION_FAIL:" + ",".join(failed)
        ),
        "aggregate": {
            "median_candidate_C": _median(candidate_c),
            "median_native_C": _median(native_c),
            "candidate_R": aggregate_candidate_r,
            "native_R": aggregate_native_r,
            "candidate_O": aggregate_candidate_o,
            "median_ridge_eval_accuracy": _median(
                float(row["ridge_candidate"]["eval_accuracy"]) for row in seed_results
            ),
        },
    }


def _admission_worker(argument: tuple[int, int]) -> dict[str, Any]:
    seed, pairs = argument
    with threadpool_limits(limits=1):
        return _admission_seed(seed, pairs=pairs)


def _strip_nondeterministic(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_nondeterministic(item)
            for key, item in value.items()
            if key
            not in {
                "worker_pid",
                "worker_nonce",
                "process_identity",
                "coordinator_pid",
                "coordinator_nonce",
                "phase",
                "phase_label",
                "output_path",
                "output_name",
                "runtime_seconds",
                "created_at_unix_ns",
                "nondeterministic_provenance",
                "report_payload_sha256",
                "deterministic_payload_sha256",
            }
        }
    if isinstance(value, list):
        return [_strip_nondeterministic(item) for item in value]
    return value


def _finalize_report(report: dict[str, Any]) -> dict[str, Any]:
    final = dict(report)
    final["deterministic_payload_sha256"] = _sha256_json(_strip_nondeterministic(final))
    return final


def run_admission(
    *,
    seeds: Sequence[int],
    workers: int,
    pairs: int,
    phase: str,
    _authorization: object | None = None,
) -> dict[str, Any]:
    if any(type(seed) is not int for seed in seeds) or type(workers) is not int or type(pairs) is not int:
        raise TypeError("seeds, workers, and pairs must be built-in integers")
    selected = tuple(seeds)
    if phase in {"admission", "admission-rerun"}:
        if selected != ADMISSION_SEEDS or workers != 5 or pairs != 100:
            raise ValueError("official admission requires seeds 90-94, five workers, 100 pairs")
        if _authorization is not _OFFICIAL_EXECUTION_AUTHORITY:
            raise RuntimeError("official admission lacks verified execution authority")
    elif phase == "scratch":
        if selected != SCRATCH_SEEDS or workers != 2 or not 1 <= pairs <= 8:
            raise ValueError(
                "scratch admission requires seeds 9090 9091, two workers, and 1..8 pairs"
            )
    else:
        raise ValueError("unknown admission phase")
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        if workers == 1:
            rows = [_admission_seed(seed, pairs=pairs) for seed in selected]
        else:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=context, max_tasks_per_child=1
            ) as executor:
                rows = list(
                    executor.map(_admission_worker, ((seed, pairs) for seed in selected))
                )
        coordinator_identity = _process_identity(_runtime_source_sha256())
    rows.sort(key=lambda row: int(row["seed"]))
    if tuple(int(row["seed"]) for row in rows) != selected:
        raise RuntimeError("admission workers returned unordered or missing seeds")
    decision = (
        recompute_admission_gates(rows)
        if selected == ADMISSION_SEEDS
        else {
            "gates": {},
            "all_admission_gates_pass": False,
            "failed_gate_ids": [],
            "terminal": "SCRATCH_NONSELECTING",
            "aggregate": {},
        }
    )
    report = {
        "schema_version": "experiment-000-lwoh-l1-v3-admission-report-v1",
        "protocol_version": PROTOCOL_VERSION,
        "scientific_rng_namespace": SCIENTIFIC_RNG_NAMESPACE,
        "phase": phase,
        "status": decision["terminal"],
        "seeds": list(selected),
        "workers": workers,
        "pairs_per_split": pairs,
        "condition_ids": [
            "lwoh_head",
            "latest_head",
            "lwoh_independent_label",
            "frozen_lwoh_head",
            "rand",
            "ridge_lwoh",
            "ridge_latest",
            "visible_cue_hold",
        ],
        "seed_results": rows,
        "decision": decision,
        "integrity": {
            "assigned_seeds_exact": selected == ADMISSION_SEEDS if phase != "scratch" else True,
            "forbidden_a3_seeds_absent": not set(selected).intersection(range(105, 110)),
            "confirmatory_seeds_absent": not set(selected).intersection(CONFIRMATORY_SEEDS),
            "no_ccf_calls": all(
                bool(row["invariants"]["ccf_calls_zero"]) for row in rows
            ),
            "no_learning_generated": True,
            "ordered_seed_results": True,
            "unique_worker_processes": len({int(row["worker_pid"]) for row in rows}),
        },
        "nondeterministic_provenance": {
            "coordinator_identity": coordinator_identity,
            "runtime_seconds": time.perf_counter() - started,
            "worker_pids": sorted({int(row["worker_pid"]) for row in rows}),
        },
    }
    return _finalize_report(report)


def _accuracy(predictions: Sequence[int], targets: Sequence[int]) -> float:
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("accuracy inputs differ or are empty")
    return float(np.mean(np.asarray(predictions, dtype=np.int8) == np.asarray(targets, dtype=np.int8)))


def _ridge_predict(model: Mapping[str, Any], features: np.ndarray) -> list[int]:
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    scores = ((features - mean) / scale) @ coefficients + float(model["intercept"])
    return np.where(scores >= 0.0, 1, -1).astype(int).tolist()


def _evaluate_cell(
    graph: LWOHObserverGraph,
    shadow: RecurrentEventGraph,
    data: CellData,
    *,
    lwoh_head: OnlineHead,
    latest_head: OnlineHead,
    independent_head: OnlineHead,
    ridge_lwoh: Mapping[str, Any],
    ridge_latest: Mapping[str, Any],
) -> dict[str, Any]:
    theta_before = {
        "lwoh_head": lwoh_head.digest(),
        "latest_head": latest_head.digest(),
        "lwoh_independent_label": independent_head.digest(),
    }
    lwoh_features: list[tuple[float, ...]] = []
    latest_features: list[tuple[float, ...]] = []
    observation_ledger: list[dict[str, Any]] = []
    targets: list[int] = []
    predictions: dict[str, list[int]] = {
        key: []
        for key in (
            "lwoh_head",
            "latest_head",
            "lwoh_independent_label",
            "frozen_lwoh_head",
            "rand",
            "ridge_lwoh",
            "ridge_latest",
            "visible_cue_hold",
        )
    }
    for index, sample in enumerate(data.samples):
        observation = run_visible_sample(graph, sample, shadow=shadow)
        observation_ledger.append(_observation_record(observation))
        lwoh_features.append(observation.lwoh)
        latest_features.append(observation.latest)
        predictions["lwoh_head"].append(lwoh_head.predict(observation.lwoh)[1])
        predictions["latest_head"].append(latest_head.predict(observation.latest)[1])
        predictions["lwoh_independent_label"].append(independent_head.predict(observation.lwoh)[1])
        predictions["frozen_lwoh_head"].append(1)
        predictions["rand"].append(int(data.random_predictions[index]))
        predictions["visible_cue_hold"].append(sample.cue if sample.cue is not None else 1)
        supervision = data.supervision[index]
        if supervision.sample_id != sample.sample_id:
            raise RuntimeError("target/sample identity mismatch")
        targets.append(supervision.target)
    lwoh_array = np.asarray(lwoh_features, dtype=np.float64)
    latest_array = np.asarray(latest_features, dtype=np.float64)
    predictions["ridge_lwoh"] = _ridge_predict(ridge_lwoh, lwoh_array)
    predictions["ridge_latest"] = _ridge_predict(ridge_latest, latest_array)
    theta_after = {
        "lwoh_head": lwoh_head.digest(),
        "latest_head": latest_head.digest(),
        "lwoh_independent_label": independent_head.digest(),
    }
    accuracies = {key: _accuracy(value, targets) for key, value in predictions.items()}
    return {
        "cell": data.cell,
        "count": len(data.samples),
        "namespace_seeds": data.namespace_seeds,
        "sample_ids": [sample.sample_id for sample in data.samples],
        "visible_cues": [sample.cue for sample in data.samples],
        "noise": [list(sample.noise) for sample in data.samples],
        "targets": targets,
        "lwoh_features": lwoh_array.tolist(),
        "latest_features": latest_array.tolist(),
        "observation_ledger": observation_ledger,
        "predictions": predictions,
        "accuracies": accuracies,
        "target_sha256": _array_sha256(np.asarray(targets, dtype=np.float64)),
        "lwoh_feature_sha256": _array_sha256(lwoh_array),
        "latest_feature_sha256": _array_sha256(latest_array),
        "theta_before": theta_before,
        "theta_after": theta_after,
        "evaluation_writes": 0,
        "theta_unchanged": theta_before == theta_after,
    }


def _learning_seed(seed: int, *, sample_scale: float = 1.0) -> dict[str, Any]:
    if seed in CONFIRMATORY_SEEDS:
        raise ValueError("confirmatory seeds are closed")
    counts = {
        "d8_clean_train": max(2, int(512 * sample_scale)),
        "d8_clean_eval": max(2, int(512 * sample_scale)),
        "d4_clean_eval": max(2, int(256 * sample_scale)),
        "d16_clean_eval": max(2, int(256 * sample_scale)),
        "d8_cue_removed_eval": max(2, int(256 * sample_scale)),
    }
    counts = {key: value + value % 2 for key, value in counts.items()}
    graph = LWOHObserverGraph.from_seed(seed)
    shadow = build_experiment_000_graph(
        seed=seed, mode="full", event_log_enabled=False, **FROZEN_GRAPH_OPTIONS
    )
    graph_state_start = _graph_state(graph)
    shadow_state_start = _graph_state(shadow)
    topology_start, weights_start = graph.topology_hash(), graph.weights_hash()
    train = generate_cell(
        master_seed=seed,
        cell="d8_clean_train",
        split="train",
        count=counts["d8_clean_train"],
        noise_events=8,
        cue_visible=True,
    )
    lwoh_head, latest_head, independent_head = OnlineHead(), OnlineHead(), OnlineHead()
    lwoh_features: list[tuple[float, ...]] = []
    latest_features: list[tuple[float, ...]] = []
    targets: list[int] = []
    independent_labels: list[int] = []
    preupdate_predictions: dict[str, list[int]] = {
        "lwoh_head": [],
        "latest_head": [],
        "lwoh_independent_label": [],
    }
    update_commitments: dict[str, list[str]] = {key: [] for key in preupdate_predictions}
    training_ledger: list[dict[str, Any]] = []
    target_access_sequence_valid = True
    for index, sample in enumerate(train.samples):
        observation = run_visible_sample(graph, sample, shadow=shadow)
        lwoh_features.append(observation.lwoh)
        latest_features.append(observation.latest)
        lwoh_pending = lwoh_head.predict_for_update(observation.lwoh)
        latest_pending = latest_head.predict_for_update(observation.latest)
        independent_pending = independent_head.predict_for_update(observation.lwoh)
        chronology_before_target = {
            "lwoh_head": lwoh_head.chronology_ledger,
            "latest_head": latest_head.chronology_ledger,
            "lwoh_independent_label": independent_head.chronology_ledger,
        }
        preupdate_predictions["lwoh_head"].append(lwoh_pending.prediction)
        preupdate_predictions["latest_head"].append(latest_pending.prediction)
        preupdate_predictions["lwoh_independent_label"].append(
            independent_pending.prediction
        )
        supervision = train.supervision[index]
        if supervision.sample_id != sample.sample_id:
            target_access_sequence_valid = False
            raise RuntimeError("target identity differs after prediction")
        target = int(supervision.target)
        independent_target = int(train.independent_labels[index])
        targets.append(target)
        independent_labels.append(independent_target)
        for key, delta in (
            ("lwoh_head", lwoh_head.reveal_target(lwoh_pending, target)),
            ("latest_head", latest_head.reveal_target(latest_pending, target)),
            (
                "lwoh_independent_label",
                independent_head.reveal_target(independent_pending, independent_target),
            ),
        ):
            update_commitments[key].append(_array_sha256(delta))
        training_ledger.append(
            {
                "sample_id": sample.sample_id,
                "observation": _observation_record(observation),
                "preupdate_predictions": {
                    "lwoh_head": lwoh_pending.prediction,
                    "latest_head": latest_pending.prediction,
                    "lwoh_independent_label": independent_pending.prediction,
                },
                "target": target,
                "independent_target": independent_target,
                "chronology_before_target": chronology_before_target,
                "chronology_after_update": {
                    "lwoh_head": lwoh_head.chronology_ledger,
                    "latest_head": latest_head.chronology_ledger,
                    "lwoh_independent_label": independent_head.chronology_ledger,
                },
            }
        )
    lwoh_array = np.asarray(lwoh_features, dtype=np.float64)
    latest_array = np.asarray(latest_features, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    ridge_lwoh = _ridge_fit(lwoh_array, target_array, lwoh_array, target_array)
    ridge_latest = _ridge_fit(latest_array, target_array, latest_array, target_array)
    cells = (
        ("d8_clean_eval", 8, True),
        ("d4_clean_eval", 4, True),
        ("d16_clean_eval", 16, True),
        ("d8_cue_removed_eval", 8, False),
    )
    evaluations: dict[str, Any] = {}
    for cell, delay, visible in cells:
        data = generate_cell(
            master_seed=seed,
            cell=cell,
            split="eval",
            count=counts[cell],
            noise_events=delay,
            cue_visible=visible,
        )
        evaluations[cell] = _evaluate_cell(
            graph,
            shadow,
            data,
            lwoh_head=lwoh_head,
            latest_head=latest_head,
            independent_head=independent_head,
            ridge_lwoh=ridge_lwoh,
            ridge_latest=ridge_latest,
        )
    invariants = {
        "prediction_before_target": target_access_sequence_valid,
        "one_training_pass": True,
        "training_updates_exact": all(
            head.update_count == counts["d8_clean_train"]
            for head in (lwoh_head, latest_head, independent_head)
        ),
        "true_label_heads_changed": bool(
            np.any(lwoh_head.theta != 0.0) and np.any(latest_head.theta != 0.0)
        ),
        "evaluation_writes_zero": all(
            row["evaluation_writes"] == 0 for row in evaluations.values()
        ),
        "evaluation_theta_unchanged": all(
            bool(row["theta_unchanged"]) for row in evaluations.values()
        ),
        "topology_unchanged": graph.topology_hash() == topology_start,
        "weights_unchanged": graph.weights_hash() == weights_start,
        "credit_packets_zero": len(graph.credit_packets) == 0,
        "ccf_calls_zero": len(graph.credit_packets) == 0,
        "feature_dimension_eight": lwoh_array.shape[1] == latest_array.shape[1] == 8,
        "all_finite": bool(
            np.all(np.isfinite(lwoh_array))
            and np.all(np.isfinite(latest_array))
            and np.all(np.isfinite(lwoh_head.theta))
            and np.all(np.isfinite(latest_head.theta))
            and np.all(np.isfinite(independent_head.theta))
        ),
        "exact_target_balance": sum(value == 1 for value in targets) == len(targets) // 2,
        "exact_independent_label_balance": sum(value == 1 for value in independent_labels)
        == len(independent_labels) // 2,
        "namespace_disjoint": len(
            {
                value
                for row in evaluations.values()
                for value in row["namespace_seeds"].values()
            }.union(train.namespace_seeds.values())
        )
        == 5 * 5,
    }
    worker_identity = _process_identity(_runtime_source_sha256())
    return {
        "seed": seed,
        "worker_pid": os.getpid(),
        "worker_nonce": f"{os.getpid()}:{time.time_ns()}",
        "process_identity": worker_identity,
        "graph": {
            "selected_nodes": list(graph.selected_nodes),
            "cue_edge_ids": list(graph.cue_edge_ids),
            "cue_edge_weights": list(graph.cue_edge_weights),
            "state_start": graph_state_start,
            "state_end": _graph_state(graph),
            "native_shadow_state_start": shadow_state_start,
            "native_shadow_state_end": _graph_state(shadow),
        },
        "training": {
            "cell": train.cell,
            "count": len(train.samples),
            "namespace_seeds": train.namespace_seeds,
            "sample_ids": [sample.sample_id for sample in train.samples],
            "visible_cues": [sample.cue for sample in train.samples],
            "noise": [list(sample.noise) for sample in train.samples],
            "targets": targets,
            "independent_labels": independent_labels,
            "lwoh_features": lwoh_array.tolist(),
            "latest_features": latest_array.tolist(),
            "preupdate_predictions": preupdate_predictions,
            "update_commitments": update_commitments,
            "observation_and_chronology_ledger": training_ledger,
            "target_sha256": _array_sha256(target_array),
            "lwoh_feature_sha256": _array_sha256(lwoh_array),
            "latest_feature_sha256": _array_sha256(latest_array),
        },
        "heads": {
            "lwoh_head": {
                "theta": lwoh_head.theta.tolist(),
                "theta_sha256": lwoh_head.digest(),
                "updates": lwoh_head.update_count,
                "chronology_ledger": lwoh_head.chronology_ledger,
            },
            "latest_head": {
                "theta": latest_head.theta.tolist(),
                "theta_sha256": latest_head.digest(),
                "updates": latest_head.update_count,
                "chronology_ledger": latest_head.chronology_ledger,
            },
            "lwoh_independent_label": {
                "theta": independent_head.theta.tolist(),
                "theta_sha256": independent_head.digest(),
                "updates": independent_head.update_count,
                "chronology_ledger": independent_head.chronology_ledger,
            },
        },
        "ridge_lwoh": ridge_lwoh,
        "ridge_latest": ridge_latest,
        "evaluations": evaluations,
        "invariants": invariants,
        "all_invariants_pass": all(invariants.values()),
    }


def _learning_bucket(argument: tuple[tuple[int, ...], float]) -> list[dict[str, Any]]:
    seeds, sample_scale = argument
    with threadpool_limits(limits=1):
        identity = _process_identity(_runtime_source_sha256())
        rows = [_learning_seed(seed, sample_scale=sample_scale) for seed in seeds]
    for row in rows:
        row["process_identity"] = identity
        row["worker_pid"] = identity["process_id"]
    return rows


def _exact_sign_flip(values: Sequence[float]) -> float:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (10,):
        raise ValueError("exact sign flip requires ten seed contrasts")
    observed = math.fsum(float(value) for value in vector) / 10.0
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=10):
        statistic = math.fsum(
            float(value) * float(sign)
            for value, sign in zip(vector, signs, strict=True)
        ) / 10.0
        if statistic >= observed:
            extreme += 1
    return extreme / 1024.0


def _holm_adjust(raw: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(raw.items(), key=lambda item: (float(item[1]), item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * float(value)))
        adjusted[name] = running
    return adjusted


def _bootstrap_lower_bounds(contrasts: Mapping[str, Sequence[float]]) -> dict[str, float]:
    rng = np.random.Generator(np.random.PCG64(ANALYSIS_SEED))
    indices = rng.integers(0, 10, size=(20_000, 10))
    result: dict[str, float] = {}
    for name, values in contrasts.items():
        vector = np.asarray(values, dtype=np.float64)
        means = np.mean(vector[indices], axis=1)
        result[name] = float(np.quantile(means, 0.025, method="linear"))
    return result


def recompute_learning_gates(seed_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if tuple(int(row["seed"]) for row in seed_results) != LEARNING_SEEDS:
        raise RuntimeError("learning seed rows are incomplete or out of order")

    def values(cell: str, condition: str) -> list[float]:
        return [float(row["evaluations"][cell]["accuracies"][condition]) for row in seed_results]

    d8 = values("d8_clean_eval", "lwoh_head")
    d4 = values("d4_clean_eval", "lwoh_head")
    d16 = values("d16_clean_eval", "lwoh_head")
    frozen = values("d8_clean_eval", "frozen_lwoh_head")
    latest = values("d8_clean_eval", "latest_head")
    independent = values("d8_clean_eval", "lwoh_independent_label")
    ridge = values("d8_clean_eval", "ridge_lwoh")
    contrasts = {
        "lwoh_minus_frozen": [a - b for a, b in zip(d8, frozen, strict=True)],
        "lwoh_minus_latest": [a - b for a, b in zip(d8, latest, strict=True)],
        "lwoh_minus_independent_label": [
            a - b for a, b in zip(d8, independent, strict=True)
        ],
    }
    raw_p = {name: _exact_sign_flip(row) for name, row in contrasts.items()}
    adjusted = _holm_adjust(raw_p)
    lower = _bootstrap_lower_bounds(contrasts)
    primary = {
        "P01": float(np.mean(d8)) >= 0.75,
        "P02": sum(value >= 0.70 for value in d8) >= 8,
        "P03_frozen": float(np.mean(contrasts["lwoh_minus_frozen"])) >= 0.20,
        "P03_latest": float(np.mean(contrasts["lwoh_minus_latest"])) >= 0.10,
        "P03_independent": float(np.mean(contrasts["lwoh_minus_independent_label"]))
        >= 0.15,
        "P04": float(np.mean(d8)) >= float(np.mean(ridge)) - 0.05,
    }
    statistical = {
        "exact_assignments_1024": True,
        "all_holm_adjusted_p_strictly_below_0_05": all(value < 0.05 for value in adjusted.values()),
        "all_bootstrap_lower_bounds_positive": all(value > 0.0 for value in lower.values()),
    }
    cue_removed = values("d8_cue_removed_eval", "lwoh_head")
    robustness = {
        "R01_d4": float(np.mean(d4)) >= 0.75,
        "R01_d16": float(np.mean(d16)) >= 0.65,
        "R02_d4": sum(value >= 0.60 for value in d4) >= 8,
        "R02_d16": sum(value >= 0.60 for value in d16) >= 8,
        "R03": float(np.mean(d8)) - float(np.mean(d16)) <= 0.15,
        "R04": 0.45 <= float(np.mean(cue_removed)) <= 0.55,
        "R05_independent": 0.45 <= float(np.mean(independent)) <= 0.55,
        "R05_frozen": 0.45 <= float(np.mean(frozen)) <= 0.55,
    }
    controls = {
        "visible_cue_clean_exact_one": all(
            float(row["evaluations"][cell]["accuracies"]["visible_cue_hold"]) == 1.0
            for row in seed_results
            for cell in ("d8_clean_eval", "d4_clean_eval", "d16_clean_eval")
        ),
        "visible_cue_removed_exact_half": all(
            float(row["evaluations"]["d8_cue_removed_eval"]["accuracies"]["visible_cue_hold"])
            == 0.5
            for row in seed_results
        ),
        "all_seed_integrity": all(bool(row["all_invariants_pass"]) for row in seed_results),
    }
    failed: list[str] = []
    failed.extend(key for key, value in primary.items() if not value)
    failed.extend(key for key, value in statistical.items() if not value)
    failed.extend(key for key, value in robustness.items() if not value)
    failed.extend(key for key, value in controls.items() if not value)
    return {
        "primary_gates": primary,
        "statistical_gates": statistical,
        "robustness_gates": robustness,
        "control_gates": controls,
        "all_learning_gates_pass": all(primary.values()),
        "all_statistical_gates_pass": all(statistical.values()),
        "all_robustness_gates_pass": all(robustness.values()),
        "all_control_gates_pass": all(controls.values()),
        "all_integrity_gates_pass": controls["all_seed_integrity"],
        "failed_gate_ids": failed,
        "terminal": "LWOH_LEARNING_PASS" if not failed else "LWOH_LEARNING_FAIL:" + ",".join(failed),
        "aggregates": {
            "d8_lwoh_mean": float(np.mean(d8)),
            "d4_lwoh_mean": float(np.mean(d4)),
            "d16_lwoh_mean": float(np.mean(d16)),
            "cue_removed_lwoh_mean": float(np.mean(cue_removed)),
            "independent_label_mean": float(np.mean(independent)),
            "frozen_head_mean": float(np.mean(frozen)),
            "ridge_lwoh_mean": float(np.mean(ridge)),
            "contrast_means": {name: float(np.mean(row)) for name, row in contrasts.items()},
            "exact_sign_flip_raw_p": raw_p,
            "holm_adjusted_p": adjusted,
            "bootstrap_95_lower": lower,
        },
    }


def run_learning(
    *,
    seeds: Sequence[int],
    workers: int,
    sample_scale: float,
    phase: str,
    _authorization: object | None = None,
) -> dict[str, Any]:
    if any(type(seed) is not int for seed in seeds) or type(workers) is not int:
        raise TypeError("seeds and workers must be built-in integers")
    selected = tuple(seeds)
    official = phase in {"learning", "learning-rerun"}
    if official and (selected != LEARNING_SEEDS or workers != 5 or sample_scale != 1.0):
        raise ValueError("official learning requires seeds 95-104, five workers, full samples")
    if official and _authorization is not _OFFICIAL_EXECUTION_AUTHORITY:
        raise RuntimeError("official learning lacks verified execution authority")
    if not official and any(seed not in SCRATCH_SEEDS for seed in selected):
        raise ValueError("nonofficial learning must use scratch seeds")
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        if workers == 1:
            rows = [_learning_seed(seed, sample_scale=sample_scale) for seed in selected]
        else:
            buckets = tuple(tuple(selected[index::workers]) for index in range(workers))
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=context, max_tasks_per_child=1
            ) as executor:
                nested = list(
                    executor.map(
                        _learning_bucket,
                        ((bucket, sample_scale) for bucket in buckets),
                    )
                )
            rows = [row for bucket in nested for row in bucket]
        coordinator_identity = _process_identity(_runtime_source_sha256())
    rows.sort(key=lambda row: int(row["seed"]))
    if tuple(int(row["seed"]) for row in rows) != selected:
        raise RuntimeError("learning workers returned unordered or missing seeds")
    decision = recompute_learning_gates(rows) if official else {
        "primary_gates": {},
        "statistical_gates": {},
        "robustness_gates": {},
        "control_gates": {},
        "all_learning_gates_pass": False,
        "all_statistical_gates_pass": False,
        "all_robustness_gates_pass": False,
        "all_control_gates_pass": False,
        "all_integrity_gates_pass": all(bool(row["all_invariants_pass"]) for row in rows),
        "failed_gate_ids": [],
        "terminal": "SCRATCH_NONSELECTING",
        "aggregates": {},
    }
    report = {
        "schema_version": "experiment-000-lwoh-l1-v3-learning-report-v1",
        "protocol_version": PROTOCOL_VERSION,
        "scientific_rng_namespace": SCIENTIFIC_RNG_NAMESPACE,
        "phase": phase,
        "status": decision["terminal"],
        "seeds": list(selected),
        "workers": workers,
        "sample_scale": sample_scale,
        "condition_ids": [
            "lwoh_head",
            "latest_head",
            "lwoh_independent_label",
            "frozen_lwoh_head",
            "rand",
            "ridge_lwoh",
            "ridge_latest",
            "visible_cue_hold",
        ],
        "seed_results": rows,
        "decision": decision,
        "integrity": {
            "assigned_seeds_exact": selected == LEARNING_SEEDS if official else True,
            "forbidden_a3_seeds_absent": not set(selected).intersection(range(105, 110)),
            "confirmatory_seeds_absent": not set(selected).intersection(CONFIRMATORY_SEEDS),
            "ordered_seed_results": True,
            "no_ccf_calls": all(
                bool(row["invariants"]["ccf_calls_zero"]) for row in rows
            ),
            "unique_worker_processes": len({int(row["worker_pid"]) for row in rows}),
        },
        "nondeterministic_provenance": {
            "coordinator_identity": coordinator_identity,
            "runtime_seconds": time.perf_counter() - started,
            "worker_pids": sorted({int(row["worker_pid"]) for row in rows}),
        },
    }
    return _finalize_report(report)


def _load_config(root: Path) -> dict[str, Any]:
    with _resolve(root, CONFIG_PATH).open("rb") as handle:
        value = tomllib.load(handle)
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("V3 protocol version differs")
    return value


def _hash_json_object(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return _sha256_bytes(encoded)


def _local_import_closure(root: Path, entry: Path) -> tuple[str, ...]:
    package_root = _resolve(root, Path("src/adaptive_learning_substrate"))
    pending = [_resolve(root, entry)]
    realized: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in realized:
            continue
        realized.add(path)
        tree = ast.parse(_stable_bytes(path), filename=str(path))
        for node in ast.walk(tree):
            candidates: list[str] = []
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.module:
                        candidates.append(node.module.split(".")[0])
                    else:
                        candidates.extend(alias.name.split(".")[0] for alias in node.names)
                elif node.module and node.module.startswith("adaptive_learning_substrate."):
                    candidates.append(node.module.rsplit(".", 1)[-1])
            elif isinstance(node, ast.Import):
                candidates.extend(
                    alias.name.rsplit(".", 1)[-1]
                    for alias in node.names
                    if alias.name.startswith("adaptive_learning_substrate.")
                )
            for name in candidates:
                candidate = package_root / f"{name}.py"
                if candidate.exists() and candidate not in realized:
                    pending.append(candidate)
    return tuple(
        sorted(path.relative_to(root).as_posix() for path in realized if path.is_file())
    )


def _validate_scientific_binding(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config["v1_scientific_binding"]
    for key in ("protocol", "config"):
        path = _resolve(root, binding[f"{key}_path"])
        if _sha256_bytes(_stable_bytes(path)) != binding[f"{key}_sha256"]:
            raise RuntimeError(f"V1 {key} binding differs")
    with _resolve(root, binding["config_path"]).open("rb") as handle:
        v1 = tomllib.load(handle)
    sections = tuple(binding["section_sha256"])
    if set(sections) != {
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
    }:
        raise RuntimeError("V1 scientific section registry differs")
    for section in sections:
        if _hash_json_object(v1[section]) != binding["section_sha256"][section]:
            raise RuntimeError(f"V1 scientific section hash differs: {section}")
        if config[section] != v1[section]:
            raise RuntimeError(f"V3 changed frozen V1 scientific section: {section}")
    scientific = {section: v1[section] for section in sections}
    if _hash_json_object(scientific) != binding["scientific_payload_sha256"]:
        raise RuntimeError("V1 scientific payload hash differs")
    return {
        "v1_protocol_sha256": binding["protocol_sha256"],
        "v1_config_sha256": binding["config_sha256"],
        "v1_scientific_payload_sha256": binding["scientific_payload_sha256"],
        "section_sha256": dict(binding["section_sha256"]),
    }


def _validate_historical_bindings(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    v2 = config["v2_historical_binding"]
    for field in ("protocol", "config", "freeze", "freeze_sidecar"):
        if _file_identity(root, v2[f"{field}_path"])["sha256"] != v2[f"{field}_sha256"]:
            raise RuntimeError(f"historical V2 {field} binding differs")
    if v2["execution_authority"] is not False or v2["metrics_generated"] is not False:
        raise RuntimeError("historical V2 status differs")
    a3 = config["a3_predecessor_binding"]
    for field in (
        "freeze_record",
        "phase_sequence",
        "primary_report",
        "primary_sidecar",
        "rerun_report",
        "rerun_sidecar",
        "terminal",
    ):
        if _file_identity(root, a3[f"{field}_path"])["sha256"] != a3[f"{field}_sha256"]:
            raise RuntimeError(f"A3 {field} binding differs")
    terminal = _read_json(_resolve(root, a3["terminal_path"]))
    if terminal.get("status") != "NO_SELECTION":
        raise RuntimeError("A3 terminal is not NO_SELECTION")
    if tuple(a3["registered_seeds"]) != tuple(range(105, 110)):
        raise RuntimeError("A3 registered seed binding differs")
    return {
        "a3_terminal_sha256": a3["terminal_sha256"],
        "a3_deterministic_payload_sha256": a3["deterministic_payload_sha256"],
        "v2_freeze_sha256": v2["freeze_sha256"],
        "v2_execution_authority": False,
    }


def _source_manifest(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    registered = tuple(str(path) for path in config["source_freeze"]["paths"])
    closure = _local_import_closure(root, RUNNER_PATH)
    missing_from_registry = sorted(set(closure).difference(registered))
    if missing_from_registry:
        raise RuntimeError(f"transitive local imports missing from freeze: {missing_from_registry}")
    manifest: dict[str, dict[str, Any]] = {}
    for relative in registered:
        manifest[Path(relative).as_posix()] = _file_identity(root, relative)
    return dict(sorted(manifest.items()))


def _observed_metric_evidence(root: Path) -> list[str]:
    paths = (
        ADMISSION_PRIMARY_PATH,
        ADMISSION_RERUN_PATH,
        ADMISSION_VERIFICATION_PATH,
        IMPLEMENTATION_FREEZE_PATH,
        LEARNING_PRIMARY_PATH,
        LEARNING_RERUN_PATH,
        LEARNING_VERIFICATION_PATH,
        READINESS_PATH,
    )
    observed: list[str] = []
    for relative in paths:
        for path in (_resolve(root, relative), _resolve(root, relative).with_suffix(".sha256")):
            if path.exists():
                observed.append(path.relative_to(root).as_posix())
    for index in range(PHASES.index("admission_primary"), len(PHASES)):
        relative = _phase_receipt_path(index, PHASES[index])
        for path in (_resolve(root, relative), _resolve(root, relative).with_suffix(".sha256")):
            if path.exists():
                observed.append(path.relative_to(root).as_posix())
    return sorted(observed)


def _fresh_authoritative_v3_probe(root: Path) -> dict[str, Any]:
    script = (
        "import json,sys; "
        "from adaptive_learning_substrate.artifact_validation import "
        "validate_lwoh_v3_state; "
        "root=sys.argv[1]; "
        "\ntry:\n r=validate_lwoh_v3_state(project_root=root); "
        "print(json.dumps({'outcome':'RETURNED','result':r},sort_keys=True))"
        "\nexcept Exception as e:\n print(json.dumps({'outcome':'REJECTED',"
        "'exception_type':type(e).__name__,'message':str(e)},sort_keys=True))"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(_resolve(root, "src"))
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script, str(root)],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or len(lines) != 1:
        raise RuntimeError(
            "fresh authoritative V3 probe failed: "
            f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )
    result = json.loads(lines[0])
    if not isinstance(result, dict):
        raise TypeError("fresh authoritative V3 probe returned a non-object")
    return {
        "command": [sys.executable, "-B", "-c", "<authoritative-v3-probe>", str(root)],
        "exit_status": completed.returncode,
        "literal_output": lines[0],
        **result,
    }


def validate_preregistration(
    project_root: str | Path | None = None,
    *,
    require_official_absence: bool = False,
) -> dict[str, Any]:
    root = _project_root(project_root)
    config = _load_config(root)
    protocol = _file_identity(root, PROTOCOL_PATH)
    if protocol["sha256"] != config["protocol_sha256"]:
        raise RuntimeError("V3 protocol hash differs from configuration")
    scientific = _validate_scientific_binding(root, config)
    history = _validate_historical_bindings(root, config)
    if tuple(config["v3_seed_attestation"]["admission"]) != ADMISSION_SEEDS:
        raise RuntimeError("V3 admission seed binding differs")
    if tuple(config["v3_seed_attestation"]["learning"]) != LEARNING_SEEDS:
        raise RuntimeError("V3 learning seed binding differs")
    if tuple(config["v3_seed_attestation"]["forbidden_a3"]) != tuple(range(105, 110)):
        raise RuntimeError("V3 forbidden A3 seed binding differs")
    if config["v3_rng"]["scientific_namespace"] != SCIENTIFIC_RNG_NAMESPACE:
        raise RuntimeError("V3 scientific RNG namespace differs")
    if tuple(config["v3_global_readiness"]["registered_condition_ids"]) != CONDITION_IDS:
        raise RuntimeError("V3 condition registry differs")
    observed_metric_evidence = _observed_metric_evidence(root)
    if require_official_absence and observed_metric_evidence:
        raise RuntimeError(
            "official metric evidence exists before preregistration freeze: "
            f"{observed_metric_evidence}"
        )
    manifest = _source_manifest(root, config)
    return {
        "status": "PASS_PRE_METRIC_ONLY",
        "protocol_sha256": protocol["sha256"],
        "config_sha256": _file_identity(root, CONFIG_PATH)["sha256"],
        "scientific_binding": scientific,
        "historical_binding": history,
        "source_manifest": manifest,
        "source_manifest_sha256": _hash_json_object(manifest),
        "transitive_import_closure": list(_local_import_closure(root, RUNNER_PATH)),
        "official_metrics_observed": bool(observed_metric_evidence),
        "observed_metric_evidence": observed_metric_evidence,
    }


def _verify_lineage(root: Path) -> dict[str, Any]:
    lineage_path = _resolve(root, LINEAGE_PATH)
    lineage = _read_json(lineage_path)
    if (
        lineage.get("schema_version")
        != "experiment-000-lwoh-l1-v3-pre-metric-lineage-v1"
        or lineage.get("status") != "PASS"
    ):
        raise RuntimeError("repaired lineage handoff is invalid")
    sidecar = _read_json(lineage_path.with_suffix(".sha256"))
    expected = {
        "algorithm": "sha256",
        "report_file": lineage_path.name,
        "report_sha256": _sha256_bytes(_stable_bytes(lineage_path)),
    }
    if sidecar != expected:
        raise RuntimeError("repaired lineage sidecar differs")
    payload_fields = (
        "lineage_payload_sha256",
        "verification_payload_sha256",
        "payload_sha256",
    )
    present = [field for field in payload_fields if field in lineage]
    if len(present) != 1 or lineage[present[0]] != _self_hash(lineage, present[0]):
        raise RuntimeError("repaired lineage self-hash differs")
    config = _load_config(root)
    trigger = config["trigger"]
    if (
        expected["report_sha256"] != trigger["lineage_handoff_sha256"]
        or present[0] != trigger["lineage_handoff_payload_field"]
        or lineage[present[0]] != trigger["lineage_handoff_payload_sha256"]
        or _sha256_bytes(_stable_bytes(lineage_path.with_suffix(".sha256")))
        != trigger["lineage_handoff_sidecar_sha256"]
        or lineage.get("repaired_source_manifest_sha256")
        != trigger["repaired_source_manifest_sha256"]
        or lineage.get("validator_import_closure", {}).get("closure_sha256")
        != trigger["validator_import_closure_sha256"]
    ):
        raise RuntimeError("repaired lineage configuration binding differs")
    return {
        "path": LINEAGE_PATH.as_posix(),
        "sha256": expected["report_sha256"],
        "sidecar_sha256": _sha256_bytes(_stable_bytes(lineage_path.with_suffix(".sha256"))),
        "payload_field": present[0],
        "payload_sha256": lineage[present[0]],
    }


def _write_static_phase_ledger(root: Path) -> dict[str, Any]:
    path = _resolve(root, PHASE_LEDGER_PATH)
    return _write_record(
        path,
        {
            "schema_version": "experiment-000-lwoh-l1-v3-phase-ledger-v1",
            "protocol_version": PROTOCOL_VERSION,
            "status": "WRITE_ONCE_RECEIPT_CHAIN",
            "phase_order": list(PHASES),
            "receipt_directory": PHASE_RECEIPT_DIR.as_posix(),
            "ledger_is_immutable": True,
        },
        self_field="ledger_payload_sha256",
    )


def _phase_receipt_path(index: int, phase: str) -> Path:
    return PHASE_RECEIPT_DIR / f"{index:02d}_{phase}.json"


def _record_phase(root: Path, phase: str, artifact: Path) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError("unregistered V3 phase")
    ledger = _verify_record(
        _resolve(root, PHASE_LEDGER_PATH),
        schema="experiment-000-lwoh-l1-v3-phase-ledger-v1",
        self_field="ledger_payload_sha256",
    )
    if tuple(ledger["phase_order"]) != PHASES:
        raise RuntimeError("phase order differs")
    index = PHASES.index(phase)
    required_previous = list(range(index))
    if phase == "readiness_verification":
        admission = _verify_record(
            _resolve(root, ADMISSION_VERIFICATION_PATH),
            schema="experiment-000-lwoh-l1-v3-admission-determinism-verification-v1",
            self_field="verification_payload_sha256",
        )
        if admission["status"] == "LWOH_ADMISSION_FAIL":
            required_previous = list(range(PHASES.index("admission_verification") + 1))
            for skipped in range(
                PHASES.index("implementation_source_revalidation"), index
            ):
                if _resolve(
                    root, _phase_receipt_path(skipped, PHASES[skipped])
                ).exists():
                    raise RuntimeError("learning receipt exists after admission failure")
    receipt_path = _resolve(root, _phase_receipt_path(index, phase))
    if receipt_path.exists():
        existing = _read_json(receipt_path)
        artifact_identity = _file_identity(root, artifact)
        predecessor = None
        if required_previous:
            previous = required_previous[-1]
            previous_path = _resolve(
                root, _phase_receipt_path(previous, PHASES[previous])
            )
            predecessor = _sha256_bytes(_stable_bytes(previous_path))
        expected = {
            "schema_version": "experiment-000-lwoh-l1-v3-phase-receipt-v1",
            "protocol_version": PROTOCOL_VERSION,
            "phase_index": index,
            "phase": phase,
            "status": "COMMITTED",
            "artifact": artifact_identity,
            "predecessor_receipt_sha256": predecessor,
        }
        if existing.get("receipt_payload_sha256") != _self_hash(
            existing, "receipt_payload_sha256"
        ) or {key: value for key, value in existing.items() if key != "receipt_payload_sha256"} != expected:
            raise RuntimeError("existing phase receipt differs")
        final = _write_record(
            receipt_path, expected, self_field="receipt_payload_sha256"
        )
        _verify_phase_chain(root, phase)
        return final
    for previous in required_previous:
        path = _resolve(root, _phase_receipt_path(previous, PHASES[previous]))
        if not path.exists():
            raise RuntimeError(f"missing predecessor phase receipt: {PHASES[previous]}")
    for later in range(index, len(PHASES)):
        path = _resolve(root, _phase_receipt_path(later, PHASES[later]))
        if path.exists():
            raise RuntimeError(f"phase receipt already exists: {PHASES[later]}")
    predecessor = None
    if required_previous:
        previous = required_previous[-1]
        previous_path = _resolve(root, _phase_receipt_path(previous, PHASES[previous]))
        predecessor = _sha256_bytes(_stable_bytes(previous_path))
    artifact_identity = _file_identity(root, artifact)
    frozen_sha256 = (
        _source_checkpoint(root)
        if _resolve(root, FREEZE_PATH).with_suffix(".sha256").exists()
        else None
    )
    result = _write_record(
        receipt_path,
        {
            "schema_version": "experiment-000-lwoh-l1-v3-phase-receipt-v1",
            "protocol_version": PROTOCOL_VERSION,
            "phase_index": index,
            "phase": phase,
            "status": "COMMITTED",
            "artifact": artifact_identity,
            "predecessor_receipt_sha256": predecessor,
        },
        self_field="receipt_payload_sha256",
    )
    if frozen_sha256 is not None:
        _source_checkpoint(root, expected=frozen_sha256)
    return result


def _verify_phase_chain(root: Path, through: str) -> bool:
    if through not in PHASES:
        raise ValueError("unregistered phase")
    terminal_index = PHASES.index(through)
    indices = list(range(terminal_index + 1))
    if through == "readiness_verification":
        admission = _verify_record(
            _resolve(root, ADMISSION_VERIFICATION_PATH),
            schema="experiment-000-lwoh-l1-v3-admission-determinism-verification-v1",
            self_field="verification_payload_sha256",
        )
        if admission["status"] == "LWOH_ADMISSION_FAIL":
            indices = list(range(PHASES.index("admission_verification") + 1)) + [
                terminal_index
            ]
    for position, index in enumerate(indices):
        phase = PHASES[index]
        record = _verify_record(
            _resolve(root, _phase_receipt_path(index, phase)),
            schema="experiment-000-lwoh-l1-v3-phase-receipt-v1",
            self_field="receipt_payload_sha256",
        )
        expected_previous = None
        if position:
            previous_index = indices[position - 1]
            previous = _resolve(
                root,
                _phase_receipt_path(previous_index, PHASES[previous_index]),
            )
            expected_previous = _sha256_bytes(_stable_bytes(previous))
        if (
            record["phase_index"] != index
            or record["phase"] != phase
            or record["predecessor_receipt_sha256"] != expected_previous
        ):
            raise RuntimeError("phase receipt chain differs")
        artifact = _file_identity(root, record["artifact"]["path"])
        if artifact != record["artifact"]:
            raise RuntimeError("phase artifact changed after receipt")
    return True


def commit_activation(project_root: str | Path | None = None) -> dict[str, Any]:
    root = _project_root(project_root)
    forbidden = [
        path
        for path in _OFFICIAL_REPORTS
        if path not in {ACTIVATION_PATH, PHASE_LEDGER_PATH}
        and (
            _resolve(root, path).exists()
            or _resolve(root, path).with_suffix(".sha256").exists()
        )
    ]
    if forbidden:
        raise RuntimeError(f"V3 namespace contains post-activation evidence: {forbidden}")
    lineage = _verify_lineage(root)
    preregistration = validate_preregistration(root, require_official_absence=True)
    absent_state = _fresh_authoritative_v3_probe(root)
    if (
        absent_state.get("outcome") != "RETURNED"
        or absent_state.get("result", {}).get("state") != "ABSENT"
        or absent_state.get("result", {}).get("readiness_after") != 21
    ):
        raise RuntimeError("authoritative global validator did not observe absent V3 state")
    _write_static_phase_ledger(root)
    record = _write_record(
        _resolve(root, ACTIVATION_PATH),
        {
            "schema_version": "experiment-000-lwoh-l1-v3-activation-verification-v1",
            "protocol_version": PROTOCOL_VERSION,
            "status": "ACTIVATED_A3_NO_SELECTION_V3",
            "action": "AUTHORIZE_V3_FREEZE_ONLY",
            "predecessor_status": "NO_SELECTION",
            "lineage_handoff": lineage,
            "a3_terminal_sha256": preregistration["historical_binding"]["a3_terminal_sha256"],
            "a3_deterministic_payload_sha256": preregistration["historical_binding"][
                "a3_deterministic_payload_sha256"
            ],
            "v2_status": "INVALIDATED_SOURCE_DRIFT_NO_METRICS",
            "no_lwoh_metric_observed": True,
            "admission_seeds_unused": True,
            "learning_seeds_unused": True,
            "confirmatory_seeds_used": 0,
            "authoritative_absent_state": absent_state,
        },
        self_field="verification_payload_sha256",
    )
    _record_phase(root, "activation", ACTIVATION_PATH)
    return record


def commit_pre_metric_freeze(project_root: str | Path | None = None) -> dict[str, Any]:
    root = _project_root(project_root)
    activation = _verify_record(
        _resolve(root, ACTIVATION_PATH),
        schema="experiment-000-lwoh-l1-v3-activation-verification-v1",
        self_field="verification_payload_sha256",
    )
    _verify_phase_chain(root, "activation")
    preregistration = validate_preregistration(root, require_official_absence=True)
    config = _load_config(root)
    record = _write_record(
        _resolve(root, FREEZE_PATH),
        {
            "schema_version": "experiment-000-lwoh-l1-v3-pre-metric-freeze-v1",
            "protocol_version": PROTOCOL_VERSION,
            "status": "FROZEN_PRE_METRIC",
            "protocol_path": PROTOCOL_PATH.as_posix(),
            "protocol_sha256": preregistration["protocol_sha256"],
            "config_path": CONFIG_PATH.as_posix(),
            "config_sha256": preregistration["config_sha256"],
            "config_canonical_object_sha256": _hash_json_object(config),
            "readiness_baseline_path": config["v3_score_contract"][
                "readiness_baseline_path"
            ],
            "readiness_baseline_sha256": config["v3_score_contract"][
                "readiness_baseline_sha256"
            ],
            "activation_file_sha256": _sha256_bytes(_stable_bytes(_resolve(root, ACTIVATION_PATH))),
            "activation_payload_sha256": activation["verification_payload_sha256"],
            "lineage_handoff_sha256": activation["lineage_handoff"]["sha256"],
            "source_manifest": preregistration["source_manifest"],
            "source_manifest_sha256": preregistration["source_manifest_sha256"],
            "scientific_binding": preregistration["scientific_binding"],
            "historical_binding": preregistration["historical_binding"],
            "admission_seeds": list(ADMISSION_SEEDS),
            "learning_seeds": list(LEARNING_SEEDS),
            "scratch_seeds": list(SCRATCH_SEEDS),
            "forbidden_a3_seeds": list(range(105, 110)),
            "confirmatory_seeds_closed": list(CONFIRMATORY_SEEDS),
            "condition_ids": list(CONDITION_IDS),
            "score_before": 21,
            "score_if_complete_pass": 25,
            "score_if_any_failure": 21,
            "partial_score": False,
            "official_metric_count_at_freeze": 0,
            "v2_execution_authority": False,
            "validator_fail_closed": True,
            "claim_level": "custom_seed_development_only",
        },
        self_field="freeze_payload_sha256",
    )
    _record_phase(root, "freeze", FREEZE_PATH)
    return record


def verify_pre_metric_freeze(project_root: str | Path | None = None) -> dict[str, Any]:
    root = _project_root(project_root)
    _verify_phase_chain(root, "freeze")
    freeze = _verify_record(
        _resolve(root, FREEZE_PATH),
        schema="experiment-000-lwoh-l1-v3-pre-metric-freeze-v1",
        self_field="freeze_payload_sha256",
    )
    current = validate_preregistration(root)
    if current["source_manifest"] != freeze["source_manifest"]:
        raise RuntimeError("source manifest differs from pre-metric freeze")
    record = _write_record(
        _resolve(root, FREEZE_VERIFICATION_PATH),
        {
            "schema_version": "experiment-000-lwoh-l1-v3-freeze-verification-v1",
            "status": "PASS",
            "freeze_file_sha256": _sha256_bytes(_stable_bytes(_resolve(root, FREEZE_PATH))),
            "freeze_payload_sha256": freeze["freeze_payload_sha256"],
            "source_manifest_sha256": freeze["source_manifest_sha256"],
            "activation_valid": True,
            "source_manifest_valid": True,
            "source_manifest_stable": True,
            "scientific_values_equal_v1": True,
            "assigned_seeds_unused_attested": True,
            "official_outputs_absent": all(
                not _resolve(root, path).exists()
                for path in _OFFICIAL_REPORTS
                if path not in {ACTIVATION_PATH, FREEZE_PATH, FREEZE_VERIFICATION_PATH, PHASE_LEDGER_PATH}
            ),
            "validator_fail_closed_at_21": True,
            "fresh_process_verified": True,
        },
        self_field="verification_payload_sha256",
    )
    _record_phase(root, "freeze_verification", FREEZE_VERIFICATION_PATH)
    return record


def _verify_raw_report(
    root: Path,
    relative: Path,
    *,
    schema: str,
) -> dict[str, Any]:
    report = _verify_record(
        _resolve(root, relative), schema=schema, self_field="report_payload_sha256"
    )
    expected = _sha256_json(_strip_nondeterministic(report))
    if report.get("deterministic_payload_sha256") != expected:
        raise RuntimeError(f"deterministic payload hash differs: {relative}")
    return report


def _publish_raw_report(root: Path, relative: Path, report: dict[str, Any]) -> dict[str, Any]:
    """Publish exact bytes; phase-entry recovery owns existing raw evidence."""
    frozen_sha256 = _source_checkpoint(root)
    _write_record(_resolve(root, relative), report, self_field="report_payload_sha256")
    final = _verify_raw_report(root, relative, schema=str(report["schema_version"]))
    _source_checkpoint(root, expected=frozen_sha256)
    return final


def _boundary_input_hashes(root: Path, predecessor: str) -> dict[str, str]:
    """Reopen predecessor reports, sidecars, and receipts as one retry input set."""
    _verify_phase_chain(root, predecessor)
    paths = {PHASE_LEDGER_PATH}
    for index in range(PHASES.index(predecessor) + 1):
        receipt = _phase_receipt_path(index, PHASES[index])
        paths.add(receipt)
        paths.add(Path(_read_json(_resolve(root, receipt))["artifact"]["path"]))
    hashes = {}
    for path in sorted(paths):
        identity = _file_identity(root, path)
        sidecar = path.with_suffix(".sha256")
        if _read_json(_resolve(root, sidecar)) != {
            "algorithm": "sha256", "report_file": path.name,
            "report_sha256": identity["sha256"],
        }:
            raise RuntimeError("boundary prerequisite sidecar differs")
        hashes[path.as_posix()] = identity["sha256"]
        hashes[sidecar.as_posix()] = _file_identity(root, sidecar)["sha256"]
    return hashes


def _validate_boundary_report(
    report: Mapping[str, Any], *, kind: str, phase: str, source_sha256: str
) -> None:
    """Validate an existing boundary without regenerating scientific output."""
    if report.get("phase") != phase:
        raise RuntimeError("existing boundary phase differs")
    if not isinstance(report.get("nondeterministic_provenance"), Mapping):
        raise TypeError("existing boundary provenance is malformed")
    if phase == "scratch":
        rows = report.get("seed_results")
        if (
            kind != "admission"
            or report.get("schema_version") != "experiment-000-lwoh-l1-v3-admission-report-v1"
            or report.get("protocol_version") != PROTOCOL_VERSION
            or report.get("scientific_rng_namespace") != SCIENTIFIC_RNG_NAMESPACE
            or report.get("seeds") != list(SCRATCH_SEEDS)
            or any(type(seed) is not int for seed in report["seeds"])
            or type(report.get("workers")) is not int or report["workers"] != 2
            or type(report.get("pairs_per_split")) is not int or report["pairs_per_split"] != 8
            or report.get("condition_ids") != list(CONDITION_IDS)
            or report.get("status") != "SCRATCH_NONSELECTING"
            or not isinstance(report.get("decision"), Mapping)
            or report["decision"].get("terminal") != "SCRATCH_NONSELECTING"
            or not isinstance(rows, list) or len(rows) != len(SCRATCH_SEEDS)
            or any(not isinstance(row, Mapping) or type(row.get("seed")) is not int
                   or row["seed"] != seed or row.get("all_invariants_pass") is not True
                   for row, seed in zip(rows, SCRATCH_SEEDS, strict=True))
        ):
            raise RuntimeError("scratch boundary registry or invariant differs")
        _validate_admission_raw_evidence(report)
        expected_workers = 2
    else:
        _validate_replay_registry(report, kind)
        recompute = recompute_admission_gates if kind == "admission" else recompute_learning_gates
        if recompute(report["seed_results"]) != report["decision"]:
            raise RuntimeError("existing boundary decision differs from raw rows")
        expected_workers = 5
        host = report.get("nondeterministic_provenance", {}).get("host_preflight", {})
        limits = {
            "available_memory_bytes": (8 * 1024**3, float("inf")),
            "free_artifact_volume_bytes": (50 * 1024**3, float("inf")),
            "pagefile_used_fraction": (0.0, 0.10),
            "cpu_mean_percent": (0.0, 20.0),
            "cpu_sample_seconds": (60.0, 60.0),
        }
        if not isinstance(host, Mapping) or host.get("pass") is not True or host.get("ac_power") is not True:
            raise RuntimeError("existing boundary host attestation differs")
        for name, (lower, upper) in limits.items():
            value = host.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
                raise RuntimeError(f"existing boundary host measurement differs: {name}")
    _report_process_sets(report, expected_workers=expected_workers)
    identities = [report["nondeterministic_provenance"]["coordinator_identity"]]
    identities.extend(row["process_identity"] for row in report["seed_results"])
    executable_sha256 = _sha256_bytes(_stable_bytes(Path(sys.executable)))
    if any(identity["source_manifest_sha256"] != source_sha256
           or identity["executable_sha256"] != executable_sha256 for identity in identities):
        raise RuntimeError("existing boundary source or executable binding differs")
    if report.get("deterministic_payload_sha256") != _sha256_json(_strip_nondeterministic(report)):
        raise RuntimeError("existing boundary deterministic hash differs")


def _recover_raw_boundary(
    root: Path, relative: Path, *, kind: str, phase: str,
    predecessor: str, ledger_phase: str, source_sha256: str,
) -> dict[str, Any] | None:
    """Complete only missing publication legs; never run a producer on retry.

    This is boundary recovery, not independent replay or a scientific PASS.
    An interruption before a canonical report exists still starts a fresh run.
    """
    path = _resolve(root, relative)
    sidecar = path.with_suffix(".sha256")
    if not path.exists():
        if sidecar.exists():
            raise RuntimeError("orphan sidecar blocks boundary recovery")
        return None
    _source_checkpoint(root, expected=source_sha256)
    inputs = _boundary_input_hashes(root, predecessor)
    record = _read_json(path)
    if record.get("report_payload_sha256") != _self_hash(record, "report_payload_sha256"):
        raise RuntimeError("existing boundary self-hash differs")
    try:
        _validate_boundary_report(record, kind=kind, phase=phase, source_sha256=source_sha256)
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise RuntimeError("existing boundary evidence is malformed") from error
    payload = dict(record)
    payload.pop("report_payload_sha256")
    # _write_record compares the entire original record, including provenance,
    # before deriving a missing sidecar. A concurrent report change is rejected.
    _source_checkpoint(root, expected=source_sha256)
    if _boundary_input_hashes(root, predecessor) != inputs:
        raise RuntimeError("boundary prerequisites changed before recovery publication")
    _write_record(path, payload, self_field="report_payload_sha256")
    final = _verify_raw_report(root, relative, schema=record["schema_version"])
    _source_checkpoint(root, expected=source_sha256)
    if _boundary_input_hashes(root, predecessor) != inputs:
        raise RuntimeError("boundary prerequisites changed after recovery publication")
    _record_phase(root, ledger_phase, relative)
    _verify_phase_chain(root, ledger_phase)
    if _boundary_input_hashes(root, predecessor) != inputs:
        raise RuntimeError("boundary prerequisites changed after recovery receipt")
    _source_checkpoint(root, expected=source_sha256)
    if _verify_raw_report(root, relative, schema=record["schema_version"]) != record:
        raise RuntimeError("boundary report changed during recovery")
    return final


def _identity_key(identity: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return only stable process-instance fields after validating run provenance."""

    required = (
        "process_id",
        "process_instance_token",
        "process_started_utc",
        "run_started_utc",
        "run_instance_token",
        "start_nonce",
        "executable_sha256",
        "source_manifest_sha256",
    )
    if any(key not in identity for key in required):
        raise RuntimeError("process identity is incomplete")
    if not identity.get("blas_state", {}).get("all_loaded_pools_single_threaded"):
        raise RuntimeError("process BLAS pools are not single threaded")
    if type(identity["process_id"]) is not int or identity["process_id"] <= 0:
        raise RuntimeError("process id is invalid")
    if any(
        not isinstance(identity[field], str) or not identity[field]
        for field in required
        if field != "process_id"
    ):
        raise RuntimeError("process identity string is invalid")
    return (
        identity["process_id"],
        identity["process_instance_token"],
        identity["process_started_utc"],
    )


def _report_process_sets(
    report: Mapping[str, Any], *, expected_workers: int
) -> tuple[tuple[Any, ...], set[tuple[Any, ...]]]:
    provenance = report.get("nondeterministic_provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError("coordinator provenance is absent")
    coordinator = provenance.get("coordinator_identity")
    if not isinstance(coordinator, Mapping):
        raise TypeError("coordinator identity is absent")
    coordinator_key = _identity_key(coordinator)
    rows = report.get("seed_results")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("seed results are absent")
    workers = set()
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("process_identity"), Mapping):
            raise TypeError("worker identity is absent from a seed row")
        key = _identity_key(row["process_identity"])
        if type(row.get("worker_pid")) is not int or row["worker_pid"] != key[0]:
            raise RuntimeError("worker PID differs from its process identity")
        workers.add(key)
    if len(workers) != expected_workers or coordinator_key in workers:
        raise RuntimeError("worker identities are not exact and distinct")
    return coordinator_key, workers


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _validate_graph_evidence(graph: Mapping[str, Any]) -> None:
    required_states = (
        "state_start",
        "state_end",
        "native_shadow_state_start",
        "native_shadow_state_end",
    )
    if any(not isinstance(graph.get(key), Mapping) for key in required_states):
        raise RuntimeError("graph state evidence is incomplete")
    states = [graph[key] for key in required_states]
    if not all(state == states[0] for state in states[1:]):
        raise RuntimeError("graph or native-shadow state changed")
    state = states[0]
    edges = state.get("edge_state")
    recurrent = state.get("recurrent_matrix")
    # The native graph API emits uppercase hexadecimal; do not rewrite its
    # recorded commitments or confuse their representation with file sidecars.
    native_hashes = (state.get("topology_sha256"), state.get("weights_sha256"))
    if (
        not isinstance(edges, list)
        or state.get("edge_count") != len(edges)
        or any(not isinstance(value, str) or not _is_sha256(value.lower()) for value in native_hashes)
        or state.get("edge_state_sha256") != _sha256_json(edges)
        or not isinstance(recurrent, Mapping)
    ):
        raise RuntimeError("graph state commitments differ")
    edge_fields = {
        "edge_id",
        "source",
        "destination",
        "kind",
        "delay_ticks",
        "plastic",
        "weight",
    }
    if any(set(edge) != edge_fields for edge in edges):
        raise RuntimeError("registered edge-state fields differ")
    if any(
        type(edge["plastic"]) is not bool
        or type(edge["delay_ticks"]) is not int
        or not math.isfinite(float(edge["weight"]))
        for edge in edges
    ):
        raise RuntimeError("edge-state value is invalid")
    if (
        not _is_sha256(recurrent.get("matrix_sha256"))
        or not math.isfinite(float(recurrent.get("spectral_radius", math.nan)))
        or not math.isfinite(float(recurrent.get("operator_norm_2", math.nan)))
    ):
        raise RuntimeError("recurrent matrix evidence is invalid")


def _validate_observation_evidence(observation: Mapping[str, Any]) -> None:
    sha_pairs = (
        ("native_event_sha256", "native_shadow_event_sha256"),
        ("native_activation_sha256", "native_shadow_activation_sha256"),
    )
    if observation.get("native_equivalent") is not True or any(
        not _is_sha256(observation.get(left))
        or observation.get(left) != observation.get(right)
        for left, right in sha_pairs
    ):
        raise RuntimeError("native event or activation evidence differs")
    counters = (
        "emitted_events",
        "forward_edge_touches",
        "native_emitted_events",
        "native_forward_edge_touches",
        "observer_calls",
        "lwoh_kernel_calls",
        "latest_kernel_calls",
        "lwoh_writes",
        "latest_writes",
    )
    if any(type(observation.get(key)) is not int or observation[key] < 0 for key in counters):
        raise RuntimeError("observation compute ledger is invalid")
    if (
        observation["emitted_events"] != observation["native_emitted_events"]
        or observation["forward_edge_touches"]
        != observation["native_forward_edge_touches"]
        or observation["observer_calls"]
        != observation["lwoh_kernel_calls"]
        or observation["observer_calls"] != observation["latest_kernel_calls"]
        or observation["lwoh_writes"] > 8
        or observation["latest_writes"] != observation["latest_kernel_calls"]
    ):
        raise RuntimeError("observation activity contract differs")
    if (
        len(observation.get("lwoh", ())) != 8
        or len(observation.get("latest", ())) != 8
        or not all(
            math.isfinite(float(value))
            for key in ("lwoh", "latest")
            for value in observation[key]
        )
    ):
        raise RuntimeError("observation feature evidence is invalid")


def _validate_admission_raw_evidence(report: Mapping[str, Any]) -> None:
    for seed_row in report["seed_results"]:
        _validate_graph_evidence(seed_row["graph"])
        for split in (seed_row["train"], seed_row["eval"]):
            arrays = {
                key: np.asarray(value, dtype=np.float64)
                for key, value in split["arrays"].items()
            }
            candidate = _paired_metrics(
                arrays["lwoh_minus"],
                arrays["lwoh_plus"],
                arrays["structural_minus"],
                arrays["structural_plus"],
            )
            native = _paired_metrics(
                arrays["latest_minus"],
                arrays["latest_plus"],
                arrays["native_minus"],
                arrays["native_plus"],
            )
            if candidate != split["candidate_metrics"] or native != split["native_metrics"]:
                raise RuntimeError("admission metrics do not reproduce from raw arrays")
            for observation in split["activity"]:
                _validate_observation_evidence(observation)


def _validate_learning_raw_evidence(report: Mapping[str, Any]) -> None:
    for seed_row in report["seed_results"]:
        _validate_graph_evidence(seed_row["graph"])
        training = seed_row["training"]
        lwoh = np.asarray(training["lwoh_features"], dtype=np.float64)
        latest = np.asarray(training["latest_features"], dtype=np.float64)
        targets = np.asarray(training["targets"], dtype=np.float64)
        if (
            training["lwoh_feature_sha256"] != _array_sha256(lwoh)
            or training["latest_feature_sha256"] != _array_sha256(latest)
            or training["target_sha256"] != _array_sha256(targets)
        ):
            raise RuntimeError("learning training commitments differ")
        ledger = training.get("observation_and_chronology_ledger")
        if not isinstance(ledger, list) or len(ledger) != training["count"]:
            raise RuntimeError("training observation ledger is incomplete")
        for index, entry in enumerate(ledger, start=1):
            _validate_observation_evidence(entry["observation"])
            for key in ("lwoh_head", "latest_head", "lwoh_independent_label"):
                before = entry["chronology_before_target"][key]
                after = entry["chronology_after_update"][key]
                if before != {
                    "predictions": index,
                    "updates": index - 1,
                    "pending_prediction": True,
                } or after != {
                    "predictions": index,
                    "updates": index,
                    "pending_prediction": False,
                }:
                    raise RuntimeError("prediction/target chronology differs")
        for head in seed_row["heads"].values():
            if head["chronology_ledger"] != {
                "predictions": training["count"],
                "updates": training["count"],
                "pending_prediction": False,
            }:
                raise RuntimeError("final head chronology differs")
        for evaluation in seed_row["evaluations"].values():
            observations = evaluation.get("observation_ledger")
            if not isinstance(observations, list) or len(observations) != evaluation["count"]:
                raise RuntimeError("evaluation observation ledger is incomplete")
            for observation in observations:
                _validate_observation_evidence(observation)
            evaluation_lwoh = np.asarray(evaluation["lwoh_features"], dtype=np.float64)
            evaluation_latest = np.asarray(evaluation["latest_features"], dtype=np.float64)
            evaluation_targets = np.asarray(evaluation["targets"], dtype=np.float64)
            if (
                evaluation["lwoh_feature_sha256"] != _array_sha256(evaluation_lwoh)
                or evaluation["latest_feature_sha256"] != _array_sha256(evaluation_latest)
                or evaluation["target_sha256"] != _array_sha256(evaluation_targets)
            ):
                raise RuntimeError("learning evaluation commitments differ")
            for condition, predictions in evaluation["predictions"].items():
                if evaluation["accuracies"][condition] != _accuracy(
                    predictions, evaluation["targets"]
                ):
                    raise RuntimeError("learning accuracy does not reproduce")


def _validate_replay_registry(report: Mapping[str, Any], kind: str) -> None:
    """Reject partial/foreign rows before any assigned-seed regeneration."""
    if kind not in {"admission", "learning"}:
        raise ValueError("unknown replay kind")
    seeds = ADMISSION_SEEDS if kind == "admission" else LEARNING_SEEDS
    rows = report.get("seed_results")
    if (
        report.get("schema_version") != f"experiment-000-lwoh-l1-v3-{kind}-report-v1"
        or report.get("protocol_version") != PROTOCOL_VERSION
        or report.get("scientific_rng_namespace") != SCIENTIFIC_RNG_NAMESPACE
        or not isinstance(report.get("decision"), Mapping)
        or report.get("status") != report["decision"].get("terminal")
        or report.get("seeds") != list(seeds)
        or any(type(seed) is not int for seed in report["seeds"])
        or type(report.get("workers")) is not int or report["workers"] != 5
        or report.get("condition_ids") != list(CONDITION_IDS)
        or not isinstance(rows, list) or len(rows) != len(seeds)
        or any(not isinstance(row, Mapping) or type(row.get("seed")) is not int
               or row["seed"] != seed for row, seed in zip(rows, seeds, strict=True))
    ):
        raise RuntimeError(f"{kind} replay registry or seed rows differ")
    if kind == "admission":
        if type(report.get("pairs_per_split")) is not int or report["pairs_per_split"] != 100:
            raise RuntimeError("admission replay sample count differs")
        validator = _validate_admission_raw_evidence
    else:
        if type(report.get("sample_scale")) not in (int, float) or report["sample_scale"] != 1.0:
            raise RuntimeError("learning replay sample scale differs")
        validator = _validate_learning_raw_evidence
    try:
        validator(report)
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise RuntimeError(f"{kind} raw evidence is incomplete or malformed") from error
    if report.get("deterministic_payload_sha256") != _sha256_json(_strip_nondeterministic(report)):
        raise RuntimeError(f"{kind} replay input payload hash differs")


def _assert_replay_processes(
    reports: Sequence[Mapping[str, Any]], verifier: Mapping[str, Any], source_sha256: str
) -> None:
    """Check all roles, including cross coordinator/worker collisions."""
    seen = {_identity_key(verifier)}
    for report in reports:
        coordinator, workers = _report_process_sets(report, expected_workers=5)
        identities = workers | {coordinator}
        if not seen.isdisjoint(identities):
            raise RuntimeError("replay verifier or archived process sets overlap")
        seen.update(identities)
        raw_identities = [report["nondeterministic_provenance"]["coordinator_identity"]]
        raw_identities.extend(row["process_identity"] for row in report["seed_results"])
        for identity in raw_identities:
            if (identity["source_manifest_sha256"] != source_sha256
                    or identity["executable_sha256"] != verifier["executable_sha256"]):
                raise RuntimeError("replay process source or executable binding differs")
    if verifier["source_manifest_sha256"] != source_sha256:
        raise RuntimeError("replay verifier source binding differs")


def _compare_replayed_seed(
    observed: Mapping[str, Any], regenerated: Mapping[str, Any], *, kind: str
) -> str:
    """Compare every deterministic field, not only aggregate metrics or gates."""
    expected = _sha256_json(_strip_nondeterministic(regenerated))
    if _sha256_json(_strip_nondeterministic(observed)) != expected:
        raise RuntimeError(f"{kind} replay differs for seed {observed.get('seed')}")
    return expected


def _verify_fresh_replay(
    root: Path, reports: Sequence[Mapping[str, Any]], *, kind: str, source_sha256: str
) -> dict[str, Any]:
    """Regenerate full rows in a measured verifier process distinct from both runs.

    The public callers have already reopened the bound reports. This helper also
    checks phase/source/resource authority before the first scientific operation,
    then reopens input commitments and source/phase prerequisites at completion.
    It never publishes or reuses a trained head from an archived report.
    """
    if kind not in {"admission", "learning"} or len(reports) != 2:
        raise ValueError("replay requires an admission or learning primary/rerun pair")
    for index, report in enumerate(reports):
        _validate_replay_registry(report, kind)
        if report.get("phase") != kind + ("-rerun" if index else ""):
            raise RuntimeError("replay primary/rerun phase order differs")
    if reports[0]["deterministic_payload_sha256"] != reports[1]["deterministic_payload_sha256"]:
        raise RuntimeError("replay input payloads differ")
    _verify_phase_chain(root, kind + "_rerun")
    _source_checkpoint(root, expected=source_sha256)
    paths = ((ADMISSION_PRIMARY_PATH, ADMISSION_RERUN_PATH) if kind == "admission"
             else (LEARNING_PRIMARY_PATH, LEARNING_RERUN_PATH))
    bound_paths = [entry for path in paths for entry in (path, path.with_suffix(".sha256"))]
    before = [_file_identity(root, path) for path in bound_paths]
    with threadpool_limits(limits=1):
        verifier = _process_identity(source_sha256)
        _assert_replay_processes(reports, verifier, source_sha256)
        _host_resource_preflight(root)
        checks = []
        for index, seed in enumerate(reports[0]["seeds"]):
            _source_checkpoint(root, expected=source_sha256)
            regenerated = (_admission_seed(seed, pairs=100) if kind == "admission"
                           else _learning_seed(seed, sample_scale=1.0))
            for report in reports:
                digest = _compare_replayed_seed(report["seed_results"][index], regenerated, kind=kind)
            checks.append({"seed": seed, "deterministic_row_sha256": digest})
    if before != [_file_identity(root, path) for path in bound_paths]:
        raise RuntimeError("replay input artifacts changed during regeneration")
    _verify_phase_chain(root, kind + "_rerun")
    _source_checkpoint(root, expected=source_sha256)
    return {"verified": True, "verifier_process_identity": verifier,
            "source_manifest_sha256": source_sha256, "seed_rows": checks}


def _validate_replay_attestation(
    stage: Mapping[str, Any], reports: Sequence[Mapping[str, Any]], realized: Mapping[str, Any]
) -> None:
    decision = reports[0]["decision"]
    if "pairs_per_split" in reports[0]:
        expected_status = "LWOH_ADMISSION_PASS" if decision["all_admission_gates_pass"] else "LWOH_ADMISSION_FAIL"
    else:
        expected_status = "LWOH_LEARNING_PASS" if decision["terminal"] == "LWOH_LEARNING_PASS" else "LWOH_LEARNING_FAIL"
    if (stage.get("status") != expected_status
            or stage.get("terminal") != decision["terminal"]
            or stage.get("failed_gate_ids") != decision["failed_gate_ids"]):
        raise RuntimeError("stage terminal differs from replayed scientific gates")
    archived = stage.get("replay_evidence")
    if (not isinstance(archived, Mapping) or archived.get("verified") is not True
            or stage.get("replay_verified") is not True
            or archived.get("source_manifest_sha256") != realized["source_manifest_sha256"]
            or archived.get("seed_rows") != realized["seed_rows"]
            or not isinstance(archived.get("verifier_process_identity"), Mapping)):
        raise RuntimeError("stage replay attestation differs from fresh regeneration")
    _assert_replay_processes(reports, archived["verifier_process_identity"], realized["source_manifest_sha256"])
    if (_identity_key(archived["verifier_process_identity"])
            == _identity_key(realized["verifier_process_identity"])):
        raise RuntimeError("terminal replay reused the stage verifier process")


def _source_freeze(root: Path) -> dict[str, Any]:
    freeze = _verify_record(
        _resolve(root, FREEZE_PATH),
        schema="experiment-000-lwoh-l1-v3-pre-metric-freeze-v1",
        self_field="freeze_payload_sha256",
    )
    current = validate_preregistration(root)
    if (
        current["source_manifest"] != freeze["source_manifest"]
        or current["source_manifest_sha256"] != freeze["source_manifest_sha256"]
    ):
        raise RuntimeError("current source differs from the pre-metric freeze")
    return freeze


def _source_checkpoint(root: Path, *, expected: str | None = None) -> str:
    realized = str(_source_freeze(root)["source_manifest_sha256"])
    if expected is not None and realized != expected:
        raise RuntimeError("frozen source changed across evidence publication")
    return realized


def _write_frozen_record(
    root: Path,
    relative: Path,
    record: dict[str, Any],
    *,
    self_field: str,
) -> dict[str, Any]:
    frozen_sha256 = _source_checkpoint(root)
    final = _write_record(_resolve(root, relative), record, self_field=self_field)
    _source_checkpoint(root, expected=frozen_sha256)
    return final


def _parse_pagefile_counter(output: str) -> float:
    """Read the single registered sample, not typeperf's trailing status text."""
    # typeperf emits unquoted status prose, including "Exiting, please wait...".
    # Only its quoted CSV lines belong to the counter data table.
    rows = list(csv.reader(line for line in output.splitlines() if line.lstrip().startswith('"')))
    headers = [
        index for index, row in enumerate(rows)
        if len(row) == 2 and row[1].casefold().endswith(
            r"\paging file(_total)\% usage"
        )
    ]
    if len(headers) != 1:
        raise OSError("typeperf paging-file counter header is missing or ambiguous")
    samples = [row for row in rows[headers[0] + 1:] if len(row) == 2]
    if len(samples) != 1:
        raise OSError("typeperf requires exactly one paging-file sample")
    try:
        percent = float(samples[0][1])
    except ValueError as exc:
        raise OSError("typeperf paging-file percentage is not numeric") from exc
    if not math.isfinite(percent) or not 0.0 <= percent <= 100.0:
        raise OSError("typeperf paging-file percentage is invalid")
    return percent


def _windows_host_snapshot(root: Path) -> dict[str, Any]:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_pagefile", ctypes.c_ulonglong),
            ("available_pagefile", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    class PowerStatus(ctypes.Structure):
        _fields_ = [
            ("ac_line_status", ctypes.c_ubyte),
            ("battery_flag", ctypes.c_ubyte),
            ("battery_life_percent", ctypes.c_ubyte),
            ("system_status_flag", ctypes.c_ubyte),
            ("battery_life_time", ctypes.c_ulong),
            ("battery_full_life_time", ctypes.c_ulong),
        ]

    memory = MemoryStatus()
    memory.length = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise OSError("GlobalMemoryStatusEx failed")
    power = PowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power)):
        raise OSError("GetSystemPowerStatus failed")
    completed = subprocess.run(
        ["typeperf", r"\Paging File(_Total)\% Usage", "-sc", "1"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise OSError(f"typeperf paging-file query failed: {completed.stderr.strip()}")
    pagefile_percent = _parse_pagefile_counter(completed.stdout)
    return {
        "available_memory_bytes": int(memory.available_physical),
        "free_artifact_volume_bytes": int(shutil.disk_usage(root).free),
        "pagefile_used_fraction": pagefile_percent / 100.0,
        "pagefile_counter": r"\Paging File(_Total)\% Usage",
        "pagefile_percent": pagefile_percent,
        "ac_power": power.ac_line_status == 1,
    }


def _host_resource_preflight(
    root: Path, *, cpu_sample_seconds: float = 60.0
) -> dict[str, Any]:
    if os.name != "nt":
        available = int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        snapshot = {
            "available_memory_bytes": available,
            "free_artifact_volume_bytes": int(shutil.disk_usage(root).free),
            "pagefile_used_fraction": 0.0,
            "ac_power": True,
        }
        cpu_mean = os.getloadavg()[0] * 100.0 / max(1, os.cpu_count() or 1)
    else:
        snapshot = _windows_host_snapshot(root)

        def cpu_times() -> tuple[int, int, int]:
            idle = ctypes.c_ulonglong()
            kernel = ctypes.c_ulonglong()
            user = ctypes.c_ulonglong()
            if not ctypes.windll.kernel32.GetSystemTimes(
                ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
            ):
                raise OSError("GetSystemTimes failed")
            return idle.value, kernel.value, user.value

        before = cpu_times()
        time.sleep(cpu_sample_seconds)
        after = cpu_times()
        idle_delta = after[0] - before[0]
        total_delta = after[1] - before[1] + after[2] - before[2]
        cpu_mean = 100.0 * (1.0 - idle_delta / total_delta) if total_delta else 100.0
    result = {
        **snapshot,
        "cpu_sample_seconds": cpu_sample_seconds,
        "cpu_mean_percent": cpu_mean,
        "minimum_available_memory_bytes": 8 * 1024**3,
        "minimum_free_artifact_volume_bytes": 50 * 1024**3,
        "maximum_pagefile_used_fraction": 0.10,
        "maximum_cpu_mean_percent": 20.0,
        "peak_memory": _peak_memory_bytes(),
    }
    result["peak_memory_ratio"] = result["peak_memory"] / max(
        1, result["available_memory_bytes"]
    )
    result["pass"] = bool(
        result["available_memory_bytes"] >= result["minimum_available_memory_bytes"]
        and result["free_artifact_volume_bytes"]
        >= result["minimum_free_artifact_volume_bytes"]
        and result["pagefile_used_fraction"] <= 0.10
        and result["cpu_mean_percent"] <= 20.0
        and result["ac_power"]
    )
    if not result["pass"]:
        raise RuntimeError(f"official host resource preflight failed: {result}")
    return result


def _verify_pytest_junit(path: Path) -> int:
    """Require real, non-skipped passing cases rather than a fixed console count."""
    root = ET.parse(path).getroot()
    suites = list(root.iter("testsuite"))
    cases = list(root.iter("testcase"))
    if not suites or not cases:
        raise RuntimeError("focused suite produced no test cases")
    if any(
        int(suite.get(field, "0")) != 0
        for suite in suites for field in ("failures", "errors", "skipped")
    ) or any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped")):
        raise RuntimeError("focused suite contains failed, errored, or skipped tests")
    if sum(int(suite.get("tests", "0")) for suite in suites) != len(cases):
        raise RuntimeError("focused suite test counts differ")
    return len(cases)


def commit_test_verification(
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = _project_root(project_root)
    _verify_phase_chain(root, "freeze_verification")
    _source_freeze(root)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        TEST_PATH.as_posix(),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(_resolve(root, "src"))
    with tempfile.TemporaryDirectory(prefix="als-v3-tests-") as temporary:
        junit = Path(temporary) / "results.xml"
        command.extend(["--junitxml", str(junit)])
        completed = subprocess.run(
            command, cwd=root, env=environment, capture_output=True,
            text=True, check=False,
        )
        literal = (completed.stdout + completed.stderr).strip()
        if completed.returncode != 0:
            raise RuntimeError(f"focused V3 suite failed: {literal}")
        _verify_pytest_junit(junit)
    record = _write_frozen_record(
        root, TEST_VERIFICATION_PATH,
        {
            "schema_version": "experiment-000-lwoh-l1-v3-test-verification-v1",
            "status": "PASS",
            "command": command,
            "literal_output": literal,
            "exit_status": completed.returncode,
            "test_file_sha256": _file_identity(root, TEST_PATH)["sha256"],
            "source_manifest_sha256": _source_freeze(root)[
                "source_manifest_sha256"
            ],
        },
        self_field="verification_payload_sha256",
    )
    _record_phase(root, "tests", TEST_VERIFICATION_PATH)
    return record


def commit_scratch_smoke(project_root: str | Path | None = None) -> dict[str, Any]:
    root = _project_root(project_root)
    _verify_phase_chain(root, "tests")
    freeze = _source_freeze(root)
    recovered = _recover_raw_boundary(
        root, SCRATCH_REPORT_PATH, kind="admission", phase="scratch",
        predecessor="tests", ledger_phase="scratch_smoke",
        source_sha256=freeze["source_manifest_sha256"],
    )
    if recovered is not None:
        return recovered
    os.environ["ALS_V3_SOURCE_MANIFEST_SHA256"] = freeze["source_manifest_sha256"]
    report = run_admission(seeds=SCRATCH_SEEDS, workers=2, pairs=8, phase="scratch")
    if not all(bool(row["all_invariants_pass"]) for row in report["seed_results"]):
        raise RuntimeError("scratch observer invariants failed")
    final = _publish_raw_report(root, SCRATCH_REPORT_PATH, report)
    _record_phase(root, "scratch_smoke", SCRATCH_REPORT_PATH)
    return final


def verify_scratch_smoke(project_root: str | Path | None = None) -> dict[str, Any]:
    root = _project_root(project_root)
    _verify_phase_chain(root, "scratch_smoke")
    freeze = _source_freeze(root)
    primary = _verify_raw_report(
        root,
        SCRATCH_REPORT_PATH,
        schema="experiment-000-lwoh-l1-v3-admission-report-v1",
    )
    os.environ["ALS_V3_SOURCE_MANIFEST_SHA256"] = freeze["source_manifest_sha256"]
    replay = run_admission(seeds=SCRATCH_SEEDS, workers=2, pairs=8, phase="scratch")
    if replay["deterministic_payload_sha256"] != primary["deterministic_payload_sha256"]:
        raise RuntimeError("scratch deterministic replay differs")
    record = _write_record(
        _resolve(root, SCRATCH_VERIFICATION_PATH),
        {
            "schema_version": "experiment-000-lwoh-l1-v3-scratch-verification-v1",
            "status": "PASS_NONSELECTING",
            "scratch_report_sha256": _file_identity(root, SCRATCH_REPORT_PATH)["sha256"],
            "deterministic_payload_sha256": primary[
                "deterministic_payload_sha256"
            ],
            "replay_verified": True,
            "assigned_seeds_touched": False,
        },
        self_field="verification_payload_sha256",
    )
    _record_phase(root, "scratch_verification", SCRATCH_VERIFICATION_PATH)
    return record


def _run_official_admission(
    root: Path, *, phase: str, output: Path
) -> dict[str, Any]:
    expected_phase = "admission" if phase == "admission" else "admission-rerun"
    expected_output = ADMISSION_PRIMARY_PATH if phase == "admission" else ADMISSION_RERUN_PATH
    if expected_phase != phase or output != expected_output:
        raise ValueError("official admission phase or output differs")
    predecessor = "scratch_verification" if phase == "admission" else "admission_primary"
    _verify_phase_chain(root, predecessor)
    freeze = _source_freeze(root)
    ledger_phase = "admission_primary" if phase == "admission" else "admission_rerun"
    recovered = _recover_raw_boundary(
        root, output, kind="admission", phase=phase,
        predecessor=predecessor, ledger_phase=ledger_phase,
        source_sha256=freeze["source_manifest_sha256"],
    )
    if recovered is not None:
        return recovered
    preflight = _host_resource_preflight(root)
    os.environ["ALS_V3_SOURCE_MANIFEST_SHA256"] = freeze["source_manifest_sha256"]
    report = run_admission(
        seeds=ADMISSION_SEEDS,
        workers=5,
        pairs=100,
        phase=phase,
        _authorization=_OFFICIAL_EXECUTION_AUTHORITY,
    )
    coordinator, workers = _report_process_sets(report, expected_workers=5)
    unique_worker_processes = len(workers)
    if unique_worker_processes != 5:
        raise RuntimeError("official admission did not use exactly five workers")
    report["nondeterministic_provenance"]["host_preflight"] = preflight
    report["nondeterministic_provenance"]["coordinator_identity_key"] = list(coordinator)
    report["nondeterministic_provenance"]["worker_identity_count"] = len(workers)
    final = _publish_raw_report(root, output, report)
    ledger_phase = "admission_primary" if phase == "admission" else "admission_rerun"
    _record_phase(root, ledger_phase, output)
    return final


def verify_admission_runs(project_root: str | Path | None = None) -> dict[str, Any]:
    root = _project_root(project_root)
    _verify_phase_chain(root, "admission_rerun")
    freeze = _source_freeze(root)
    primary = _verify_raw_report(
        root,
        ADMISSION_PRIMARY_PATH,
        schema="experiment-000-lwoh-l1-v3-admission-report-v1",
    )
    rerun = _verify_raw_report(
        root,
        ADMISSION_RERUN_PATH,
        schema="experiment-000-lwoh-l1-v3-admission-report-v1",
    )
    if primary["deterministic_payload_sha256"] != rerun["deterministic_payload_sha256"]:
        raise RuntimeError("admission deterministic payloads differ")
    decisions = []
    for report in (primary, rerun):
        if (
            tuple(report["seeds"]) != ADMISSION_SEEDS
            or report["workers"] != 5
            or report["pairs_per_split"] != 100
            or tuple(report["condition_ids"]) != CONDITION_IDS
        ):
            raise RuntimeError("admission report registry differs")
        decision = recompute_admission_gates(report["seed_results"])
        if decision != report["decision"]:
            raise RuntimeError("admission gates differ on recomputation")
        decisions.append(decision)
    primary_coordinator, primary_workers = _report_process_sets(
        primary, expected_workers=5
    )
    rerun_coordinator, rerun_workers = _report_process_sets(rerun, expected_workers=5)
    if (
        primary_coordinator == rerun_coordinator
        or not primary_workers.isdisjoint(rerun_workers)
    ):
        raise RuntimeError("admission primary and rerun processes overlap")
    replay = _verify_fresh_replay(
        root, (primary, rerun), kind="admission", source_sha256=freeze["source_manifest_sha256"]
    )
    passed = bool(decisions[0]["all_admission_gates_pass"])
    record = _write_frozen_record(
        root, ADMISSION_VERIFICATION_PATH,
        {
            "schema_version": "experiment-000-lwoh-l1-v3-admission-determinism-verification-v1",
            "status": "LWOH_ADMISSION_PASS" if passed else "LWOH_ADMISSION_FAIL",
            "terminal": decisions[0]["terminal"],
            "failed_gate_ids": decisions[0]["failed_gate_ids"],
            "deterministic_payload_sha256": primary[
                "deterministic_payload_sha256"
            ],
            "primary_report_sha256": _file_identity(root, ADMISSION_PRIMARY_PATH)[
                "sha256"
            ],
            "rerun_report_sha256": _file_identity(root, ADMISSION_RERUN_PATH)[
                "sha256"
            ],
            "source_manifest_sha256": freeze["source_manifest_sha256"],
            "admission_gates_recomputed": True,
            "deterministic_payloads_equal": True,
            "distinct_processes_verified": True,
            "replay_verified": replay["verified"],
            "replay_evidence": replay,
        },
        self_field="verification_payload_sha256",
    )
    _record_phase(root, "admission_verification", ADMISSION_VERIFICATION_PATH)
    return record


def commit_implementation_source_freeze(
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = _project_root(project_root)
    _verify_phase_chain(root, "admission_verification")
    admission = _verify_record(
        _resolve(root, ADMISSION_VERIFICATION_PATH),
        schema="experiment-000-lwoh-l1-v3-admission-determinism-verification-v1",
        self_field="verification_payload_sha256",
    )
    if admission["status"] != "LWOH_ADMISSION_PASS":
        raise RuntimeError("learning is blocked by admission failure")
    freeze = _source_freeze(root)
    record = _write_record(
        _resolve(root, IMPLEMENTATION_FREEZE_PATH),
        {
            "schema_version": "experiment-000-lwoh-l1-v3-implementation-source-freeze-v1",
            "status": "FROZEN_FOR_LEARNING",
            "pre_metric_freeze_sha256": _file_identity(root, FREEZE_PATH)["sha256"],
            "admission_verification_sha256": _file_identity(
                root, ADMISSION_VERIFICATION_PATH
            )["sha256"],
            "source_manifest": freeze["source_manifest"],
            "source_manifest_sha256": freeze["source_manifest_sha256"],
            "manifest_equal_pre_metric_manifest": True,
        },
        self_field="freeze_payload_sha256",
    )
    _record_phase(
        root, "implementation_source_revalidation", IMPLEMENTATION_FREEZE_PATH
    )
    return record


def _run_official_learning(
    root: Path, *, phase: str, output: Path
) -> dict[str, Any]:
    expected_output = LEARNING_PRIMARY_PATH if phase == "learning" else LEARNING_RERUN_PATH
    if phase not in {"learning", "learning-rerun"} or output != expected_output:
        raise ValueError("official learning phase or output differs")
    predecessor = (
        "implementation_source_revalidation" if phase == "learning" else "learning_primary"
    )
    _verify_phase_chain(root, predecessor)
    freeze = _source_freeze(root)
    ledger_phase = "learning_primary" if phase == "learning" else "learning_rerun"
    recovered = _recover_raw_boundary(
        root, output, kind="learning", phase=phase,
        predecessor=predecessor, ledger_phase=ledger_phase,
        source_sha256=freeze["source_manifest_sha256"],
    )
    if recovered is not None:
        return recovered
    preflight = _host_resource_preflight(root)
    os.environ["ALS_V3_SOURCE_MANIFEST_SHA256"] = freeze["source_manifest_sha256"]
    report = run_learning(
        seeds=LEARNING_SEEDS,
        workers=5,
        sample_scale=1.0,
        phase=phase,
        _authorization=_OFFICIAL_EXECUTION_AUTHORITY,
    )
    coordinator, workers = _report_process_sets(report, expected_workers=5)
    unique_worker_processes = len(workers)
    if unique_worker_processes != 5:
        raise RuntimeError("official learning did not use exactly five workers")
    report["nondeterministic_provenance"]["host_preflight"] = preflight
    report["nondeterministic_provenance"]["coordinator_identity_key"] = list(coordinator)
    report["nondeterministic_provenance"]["worker_identity_count"] = len(workers)
    final = _publish_raw_report(root, output, report)
    ledger_phase = "learning_primary" if phase == "learning" else "learning_rerun"
    _record_phase(root, ledger_phase, output)
    return final


def verify_learning_runs(project_root: str | Path | None = None) -> dict[str, Any]:
    root = _project_root(project_root)
    _verify_phase_chain(root, "learning_rerun")
    freeze = _source_freeze(root)
    primary = _verify_raw_report(
        root,
        LEARNING_PRIMARY_PATH,
        schema="experiment-000-lwoh-l1-v3-learning-report-v1",
    )
    rerun = _verify_raw_report(
        root,
        LEARNING_RERUN_PATH,
        schema="experiment-000-lwoh-l1-v3-learning-report-v1",
    )
    if primary["deterministic_payload_sha256"] != rerun["deterministic_payload_sha256"]:
        raise RuntimeError("learning deterministic payloads differ")
    decisions = []
    for report in (primary, rerun):
        if (
            tuple(report["seeds"]) != LEARNING_SEEDS
            or report["workers"] != 5
            or report["sample_scale"] != 1.0
            or tuple(report["condition_ids"]) != CONDITION_IDS
        ):
            raise RuntimeError("learning report registry differs")
        decision = recompute_learning_gates(report["seed_results"])
        if decision != report["decision"]:
            raise RuntimeError("learning gates differ on recomputation")
        decisions.append(decision)
    primary_coordinator, primary_workers = _report_process_sets(
        primary, expected_workers=5
    )
    rerun_coordinator, rerun_workers = _report_process_sets(rerun, expected_workers=5)
    if (
        primary_coordinator == rerun_coordinator
        or not primary_workers.isdisjoint(rerun_workers)
    ):
        raise RuntimeError("learning primary and rerun processes overlap")
    replay = _verify_fresh_replay(
        root, (primary, rerun), kind="learning", source_sha256=freeze["source_manifest_sha256"]
    )
    passed = decisions[0]["terminal"] == "LWOH_LEARNING_PASS"
    record = _write_frozen_record(
        root, LEARNING_VERIFICATION_PATH,
        {
            "schema_version": "experiment-000-lwoh-l1-v3-learning-determinism-verification-v1",
            "status": "LWOH_LEARNING_PASS" if passed else "LWOH_LEARNING_FAIL",
            "terminal": decisions[0]["terminal"],
            "failed_gate_ids": decisions[0]["failed_gate_ids"],
            "deterministic_payload_sha256": primary[
                "deterministic_payload_sha256"
            ],
            "primary_report_sha256": _file_identity(root, LEARNING_PRIMARY_PATH)[
                "sha256"
            ],
            "rerun_report_sha256": _file_identity(root, LEARNING_RERUN_PATH)[
                "sha256"
            ],
            "source_manifest_sha256": freeze["source_manifest_sha256"],
            "learning_gates_recomputed": True,
            "statistical_gates_recomputed": True,
            "robustness_gates_recomputed": True,
            "controls_recomputed": True,
            "deterministic_payloads_equal": True,
            "distinct_processes_verified": True,
            "replay_verified": replay["verified"],
            "replay_evidence": replay,
        },
        self_field="verification_payload_sha256",
    )
    _record_phase(root, "learning_verification", LEARNING_VERIFICATION_PATH)
    return record


def _readiness_chain_artifacts(root: Path, *, admission_pass: bool) -> dict[str, str]:
    paths = [FREEZE_PATH, ACTIVATION_PATH, ADMISSION_VERIFICATION_PATH]
    if admission_pass:
        paths.extend([IMPLEMENTATION_FREEZE_PATH, LEARNING_VERIFICATION_PATH])
    return {path.as_posix(): _file_identity(root, path)["sha256"] for path in paths}


def commit_readiness_verification(
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = _project_root(project_root)
    admission = _verify_record(
        _resolve(root, ADMISSION_VERIFICATION_PATH),
        schema="experiment-000-lwoh-l1-v3-admission-determinism-verification-v1",
        self_field="verification_payload_sha256",
    )
    admission_pass = admission["status"] == "LWOH_ADMISSION_PASS"
    learning: dict[str, Any] | None = None
    if admission_pass:
        _verify_phase_chain(root, "learning_verification")
        learning = _verify_record(
            _resolve(root, LEARNING_VERIFICATION_PATH),
            schema="experiment-000-lwoh-l1-v3-learning-determinism-verification-v1",
            self_field="verification_payload_sha256",
        )
    else:
        _verify_phase_chain(root, "admission_verification")
        for path in (
            IMPLEMENTATION_FREEZE_PATH,
            LEARNING_PRIMARY_PATH,
            LEARNING_RERUN_PATH,
            LEARNING_VERIFICATION_PATH,
        ):
            if _resolve(root, path).exists() or _resolve(root, path).with_suffix(
                ".sha256"
            ).exists():
                raise RuntimeError("learning evidence exists after admission failure")
    lineage = _verify_lineage(root)
    learning_pass = bool(learning and learning["status"] == "LWOH_LEARNING_PASS")
    learning_report = (
        _verify_raw_report(
            root,
            LEARNING_PRIMARY_PATH,
            schema="experiment-000-lwoh-l1-v3-learning-report-v1",
        )
        if learning
        else None
    )
    decision = learning_report["decision"] if learning_report else None
    status = (
        "LWOH_LEARNING_PASS"
        if learning_pass
        else "LWOH_LEARNING_FAIL"
        if admission_pass
        else "LWOH_ADMISSION_FAIL"
    )
    chain_artifacts = _readiness_chain_artifacts(root, admission_pass=admission_pass)
    record = _write_frozen_record(
        root, READINESS_PATH,
        {
            "schema_version": "experiment-000-lwoh-l1-v3-readiness-v1",
            "status": status,
            "readiness_before": 21,
            "readiness_after": 25 if learning_pass else 21,
            "partial_score": False,
            "admission_seeds": list(ADMISSION_SEEDS),
            "learning_seeds": list(LEARNING_SEEDS),
            "delays": list(DELAYS),
            "baselines": list(BASELINES),
            "deterministic_admission": True,
            "deterministic_learning": learning is not None,
            "all_admission_gates_pass": admission_pass,
            "all_learning_gates_pass": bool(
                decision and decision["all_learning_gates_pass"]
            ),
            "all_robustness_gates_pass": bool(
                decision and decision["all_robustness_gates_pass"]
            ),
            "all_integrity_gates_pass": bool(
                decision
                and decision["all_integrity_gates_pass"]
                and decision["all_control_gates_pass"]
                and decision["all_statistical_gates_pass"]
            ),
            "lineage_handoff_path": lineage["path"],
            "lineage_handoff_sha256": lineage["sha256"],
            "chain_artifacts": chain_artifacts,
        },
        self_field="verification_payload_sha256",
    )
    _record_phase(root, "readiness_verification", READINESS_PATH)
    validate_official_v3_chain(root)
    return record


def validate_official_v3_chain(
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Reopen and recompute the complete official V3 evidence chain."""
    root = _project_root(project_root)
    readiness = _verify_record(
        _resolve(root, READINESS_PATH),
        schema="experiment-000-lwoh-l1-v3-readiness-v1",
        self_field="verification_payload_sha256",
    )
    _verify_phase_chain(root, "readiness_verification")
    config = _load_config(root)
    exact_readiness_keys = set(config["v3_global_readiness"]["exact_record_keys"])
    if set(readiness) != exact_readiness_keys:
        raise RuntimeError("readiness key set differs")
    if (
        readiness["readiness_before"] != 21
        or readiness["readiness_after"] not in {21, 25}
        or readiness["partial_score"] is not False
        or tuple(readiness["admission_seeds"]) != ADMISSION_SEEDS
        or tuple(readiness["learning_seeds"]) != LEARNING_SEEDS
        or tuple(readiness["delays"]) != DELAYS
        or tuple(readiness["baselines"]) != BASELINES
    ):
        raise RuntimeError("readiness fixed registry differs")
    freeze = _source_freeze(root)
    admission_stage = _verify_record(
        _resolve(root, ADMISSION_VERIFICATION_PATH),
        schema="experiment-000-lwoh-l1-v3-admission-determinism-verification-v1",
        self_field="verification_payload_sha256",
    )
    admission_reports = [
        _verify_raw_report(
            root,
            path,
            schema="experiment-000-lwoh-l1-v3-admission-report-v1",
        )
        for path in (ADMISSION_PRIMARY_PATH, ADMISSION_RERUN_PATH)
    ]
    admission_decisions = [
        recompute_admission_gates(report["seed_results"])
        for report in admission_reports
    ]
    if any(
        decision != report["decision"]
        for decision, report in zip(
            admission_decisions, admission_reports, strict=True
        )
    ):
        raise RuntimeError("admission raw rows do not reproduce their decisions")
    admission_deterministic = (
        admission_reports[0]["deterministic_payload_sha256"]
        == admission_reports[1]["deterministic_payload_sha256"]
        == admission_stage["deterministic_payload_sha256"]
    )
    admission_processes = [
        _report_process_sets(report, expected_workers=5)
        for report in admission_reports
    ]
    if (
        admission_processes[0][0] == admission_processes[1][0]
        or not admission_processes[0][1].isdisjoint(admission_processes[1][1])
    ):
        raise RuntimeError("admission process sets overlap")
    admission_pass = admission_stage["status"] == "LWOH_ADMISSION_PASS"
    if readiness["chain_artifacts"] != _readiness_chain_artifacts(
        root, admission_pass=admission_pass
    ):
        raise RuntimeError("readiness chain hash mapping differs")
    learning_reports: list[dict[str, Any]] = []
    learning_decisions: list[dict[str, Any]] = []
    learning_processes: list[tuple[tuple[Any, ...], set[tuple[Any, ...]]]] = []
    learning_deterministic = False
    absence_valid = True
    if admission_pass:
        implementation = _verify_record(
            _resolve(root, IMPLEMENTATION_FREEZE_PATH),
            schema="experiment-000-lwoh-l1-v3-implementation-source-freeze-v1",
            self_field="freeze_payload_sha256",
        )
        if (
            implementation["source_manifest"] != freeze["source_manifest"]
            or implementation["source_manifest_sha256"]
            != freeze["source_manifest_sha256"]
        ):
            raise RuntimeError("implementation source revalidation differs")
        learning_stage = _verify_record(
            _resolve(root, LEARNING_VERIFICATION_PATH),
            schema="experiment-000-lwoh-l1-v3-learning-determinism-verification-v1",
            self_field="verification_payload_sha256",
        )
        learning_reports = [
            _verify_raw_report(
                root,
                path,
                schema="experiment-000-lwoh-l1-v3-learning-report-v1",
            )
            for path in (LEARNING_PRIMARY_PATH, LEARNING_RERUN_PATH)
        ]
        learning_decisions = [
            recompute_learning_gates(report["seed_results"])
            for report in learning_reports
        ]
        if any(
            decision != report["decision"]
            for decision, report in zip(
                learning_decisions, learning_reports, strict=True
            )
        ):
            raise RuntimeError("learning raw rows do not reproduce their decisions")
        learning_deterministic = (
            learning_reports[0]["deterministic_payload_sha256"]
            == learning_reports[1]["deterministic_payload_sha256"]
            == learning_stage["deterministic_payload_sha256"]
        )
        learning_processes = [
            _report_process_sets(report, expected_workers=5)
            for report in learning_reports
        ]
        all_metric_identities = (
            {admission_processes[0][0], admission_processes[1][0]}
            | admission_processes[0][1]
            | admission_processes[1][1]
        )
        for coordinator, workers in learning_processes:
            if coordinator in all_metric_identities or not workers.isdisjoint(
                all_metric_identities
            ):
                raise RuntimeError("admission and learning process sets overlap")
            all_metric_identities.add(coordinator)
            all_metric_identities.update(workers)
        if (
            learning_processes[0][0] == learning_processes[1][0]
            or not learning_processes[0][1].isdisjoint(learning_processes[1][1])
        ):
            raise RuntimeError("learning process sets overlap")
    else:
        for path in (
            IMPLEMENTATION_FREEZE_PATH,
            LEARNING_PRIMARY_PATH,
            LEARNING_RERUN_PATH,
            LEARNING_VERIFICATION_PATH,
        ):
            absence_valid &= not _resolve(root, path).exists()
            absence_valid &= not _resolve(root, path).with_suffix(".sha256").exists()
        if not absence_valid:
            raise RuntimeError("admission-failure absence branch differs")
    condition_ids_exact = all(
        tuple(report["condition_ids"]) == CONDITION_IDS
        for report in admission_reports + learning_reports
    )
    assigned_seeds_exact = all(
        tuple(report["seeds"]) == ADMISSION_SEEDS
        for report in admission_reports
    ) and all(
        tuple(report["seeds"]) == LEARNING_SEEDS for report in learning_reports
    )
    per_seed_rows_complete = all(
        len(report["seed_results"]) == len(report["seeds"])
        for report in admission_reports + learning_reports
    )
    official_and_forbidden_seeds_exact = assigned_seeds_exact and all(
        not set(report["seeds"]).intersection(
            set(range(105, 110)).union(CONFIRMATORY_SEEDS)
        )
        for report in admission_reports + learning_reports
    )
    learning_recomputed = bool(learning_reports)
    with threadpool_limits(limits=1):
        verifier_identity = _process_identity(freeze["source_manifest_sha256"])
    archived_identity_sets = [
        workers | {coordinator}
        for coordinator, workers in admission_processes + learning_processes
    ]
    verifier_key = _identity_key(verifier_identity)
    pairwise_disjoint = all(
        left.isdisjoint(right)
        for index, left in enumerate(archived_identity_sets)
        for right in archived_identity_sets[index + 1 :]
    ) and all(verifier_key not in identities for identities in archived_identity_sets)
    distinct_processes = pairwise_disjoint
    deterministic_equal = admission_deterministic and (
        learning_deterministic if admission_pass else True
    )
    if not deterministic_equal or not distinct_processes:
        raise RuntimeError("terminal replay requires equal payloads and fresh process sets")
    admission_replay = _verify_fresh_replay(
        root, admission_reports, kind="admission", source_sha256=freeze["source_manifest_sha256"]
    )
    _validate_replay_attestation(admission_stage, admission_reports, admission_replay)
    learning_replay = None
    if admission_pass:
        learning_replay = _verify_fresh_replay(
            root, learning_reports, kind="learning", source_sha256=freeze["source_manifest_sha256"]
        )
        _validate_replay_attestation(learning_stage, learning_reports, learning_replay)
    expected_terminal_status = learning_stage["status"] if admission_pass else admission_stage["status"]
    if readiness["status"] != expected_terminal_status:
        raise RuntimeError("readiness terminal differs from replayed stage status")
    _source_checkpoint(root, expected=freeze["source_manifest_sha256"])
    _verify_phase_chain(root, "readiness_verification")
    result = {
        "status": readiness["status"],
        "readiness_before": readiness["readiness_before"],
        "readiness_after": readiness["readiness_after"],
        "raw_admission_primary_valid": True,
        "raw_admission_rerun_valid": True,
        "raw_learning_primary_valid": bool(learning_reports),
        "raw_learning_rerun_valid": bool(learning_reports),
        "all_sidecars_valid": True,
        "assigned_seeds_exact": assigned_seeds_exact,
        "per_seed_rows_complete": per_seed_rows_complete,
        "admission_gates_recomputed": True,
        "learning_gates_recomputed": learning_recomputed,
        "statistical_gates_recomputed": learning_recomputed,
        "robustness_gates_recomputed": learning_recomputed,
        "controls_recomputed": True,
        "distinct_processes_verified": distinct_processes,
        "phase_ledger_valid": True,
        "source_manifest_valid": True,
        "source_manifest_stable": True,
        "deterministic_payloads_equal": deterministic_equal,
        "absence_branches_valid": absence_valid,
        "condition_ids_exact": condition_ids_exact,
        "official_and_forbidden_seeds_exact": official_and_forbidden_seeds_exact,
        "replay_verified": admission_replay["verified"] and (
            learning_replay["verified"] if learning_replay is not None else True
        ),
        "process_freshness_verified": distinct_processes,
    }
    expected_keys = set(config["v3_deep_verifier"]["exact_result_keys"])
    if set(result) != expected_keys:
        raise RuntimeError("deep-verifier result key set differs")
    required_true = (
        config["v3_deep_verifier"]["boolean_fields"]
        if admission_pass
        else config["v3_deep_verifier"]["admission_fail_required_true"]
    )
    if any(result[field] is not True for field in required_true):
        raise RuntimeError("deep-verifier required boolean is false")
    expected_after = 25 if readiness["status"] == "LWOH_LEARNING_PASS" else 21
    if readiness["readiness_after"] != expected_after:
        raise RuntimeError("readiness score does not match the terminal")
    if readiness["status"] == "LWOH_LEARNING_PASS" and not all(
        (
            readiness["deterministic_admission"],
            readiness["deterministic_learning"],
            readiness["all_admission_gates_pass"],
            readiness["all_learning_gates_pass"],
            readiness["all_robustness_gates_pass"],
            readiness["all_integrity_gates_pass"],
        )
    ):
        raise RuntimeError("passing readiness contains a false scientific gate")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute and verify preregistered LWOH-L1 V3 phases"
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            "activation",
            "freeze",
            "verify-freeze",
            "tests",
            "scratch",
            "verify-scratch",
            "admission",
            "admission-rerun",
            "verify-admission",
            "implementation-freeze",
            "learning",
            "learning-rerun",
            "verify-learning",
            "readiness",
            "validate",
            "validate-preregistration",
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--pairs", type=int)
    parser.add_argument("--sample-scale", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--project-root", type=Path)
    return parser


def _require_no_phase_arguments(args: argparse.Namespace) -> None:
    if any(
        value is not None
        for value in (
            args.seeds,
            args.workers,
            args.pairs,
            args.sample_scale,
            args.output,
        )
    ):
        raise ValueError("this phase accepts no scientific or output arguments")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = _project_root(args.project_root)
    result: dict[str, Any]
    artifact: Path | None
    if args.phase == "activation":
        _require_no_phase_arguments(args)
        result, artifact = commit_activation(root), ACTIVATION_PATH
    elif args.phase == "freeze":
        _require_no_phase_arguments(args)
        result, artifact = commit_pre_metric_freeze(root), FREEZE_PATH
    elif args.phase == "verify-freeze":
        _require_no_phase_arguments(args)
        result, artifact = verify_pre_metric_freeze(root), FREEZE_VERIFICATION_PATH
    elif args.phase == "tests":
        _require_no_phase_arguments(args)
        result, artifact = commit_test_verification(root), TEST_VERIFICATION_PATH
    elif args.phase == "scratch":
        if (
            tuple(args.seeds or ()) != SCRATCH_SEEDS
            or args.workers != 2
            or args.output != SCRATCH_REPORT_PATH
            or args.sample_scale is not None
            or args.pairs != 8
        ):
            raise ValueError("scratch arguments differ from the registered command")
        result, artifact = commit_scratch_smoke(root), SCRATCH_REPORT_PATH
    elif args.phase == "verify-scratch":
        _require_no_phase_arguments(args)
        result, artifact = verify_scratch_smoke(root), SCRATCH_VERIFICATION_PATH
    elif args.phase in {"admission", "admission-rerun"}:
        expected = (
            ADMISSION_PRIMARY_PATH
            if args.phase == "admission"
            else ADMISSION_RERUN_PATH
        )
        if (
            tuple(args.seeds or ()) != ADMISSION_SEEDS
            or args.workers != 5
            or args.pairs not in {None, 100}
            or args.sample_scale is not None
            or args.output != expected
        ):
            raise ValueError("official admission arguments differ")
        result, artifact = _run_official_admission(
            root, phase=args.phase, output=expected
        ), expected
    elif args.phase == "verify-admission":
        _require_no_phase_arguments(args)
        result, artifact = verify_admission_runs(root), ADMISSION_VERIFICATION_PATH
    elif args.phase == "implementation-freeze":
        _require_no_phase_arguments(args)
        result, artifact = (
            commit_implementation_source_freeze(root),
            IMPLEMENTATION_FREEZE_PATH,
        )
    elif args.phase in {"learning", "learning-rerun"}:
        expected = LEARNING_PRIMARY_PATH if args.phase == "learning" else LEARNING_RERUN_PATH
        if (
            tuple(args.seeds or ()) != LEARNING_SEEDS
            or args.workers != 5
            or args.sample_scale not in {None, 1.0}
            or args.pairs is not None
            or args.output != expected
        ):
            raise ValueError("official learning arguments differ")
        result, artifact = _run_official_learning(
            root, phase=args.phase, output=expected
        ), expected
    elif args.phase == "verify-learning":
        _require_no_phase_arguments(args)
        result, artifact = verify_learning_runs(root), LEARNING_VERIFICATION_PATH
    elif args.phase == "readiness":
        _require_no_phase_arguments(args)
        result, artifact = commit_readiness_verification(root), READINESS_PATH
    elif args.phase == "validate":
        _require_no_phase_arguments(args)
        result, artifact = validate_official_v3_chain(root), None
    elif args.phase == "validate-preregistration":
        _require_no_phase_arguments(args)
        result, artifact = validate_preregistration(root), None
    else:  # pragma: no cover
        raise AssertionError(args.phase)
    if artifact is None:
        if args.phase == "validate-preregistration":
            print(
                f"{result['status']} source_manifest_sha256="
                f"{result['source_manifest_sha256']}"
            )
            return 0
        print(
            f"{result['status']} readiness={result['readiness_after']} "
            f"verified={all(value for key, value in result.items() if key not in {'status', 'readiness_before', 'readiness_after'})}"
        )
    else:
        identity = _file_identity(root, artifact)
        terminal = result.get("terminal", result.get("status", "PASS"))
        print(f"{terminal} {identity['path']} sha256={identity['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
