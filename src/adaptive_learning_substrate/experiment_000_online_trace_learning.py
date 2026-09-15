"""Preregistered online learning gate for the A3-selected trace representation.

``TRACE-HEAD-L1`` freezes the recurrent graph and trains only a persistent
16-parameter tanh head.  Official execution is impossible until both the A3
selection and the separate TRACE-HEAD-L1 alignment result have been strictly
verified.  The explicit scratch entry point rejects official seeds and paths.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import multiprocessing
import os
import platform
import stat
import struct
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

for _thread_environment_name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_environment_name] = "1"

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .experiment000_data import (
    Experiment000Episode,
    Experiment000Stream,
    SeedRole,
    generate_stream_pair,
    require_seed_role,
)
from .experiment_000 import FROZEN_GRAPH_OPTIONS
from .experiment_000_memory_probe import (
    _output_feature_edge_ids,
    extract_query_features,
)
from .readout_trace_recurrent import (
    ReadoutTraceRecurrentEventGraph,
    clone_with_readout_trace,
)
from .recurrent import QueryResult, RecurrentEventGraph, build_experiment_000_graph

SCHEMA_VERSION = "experiment-000-online-trace-learning-v1"
PROTOCOL_NAME = "experiment_000_online_trace_learning"
PROTOCOL_VERSION = "trace-head-l1-v1"
REGISTERED_SEEDS: tuple[int, ...] = tuple(range(80, 90))
SCRATCH_SMOKE_SEEDS: tuple[int, ...] = (900, 901)
SMOKE_EXAMPLES_PER_SPLIT = 2
ORDERED_CONDITIONS: tuple[str, ...] = (
    "trace_head_v1",
    "native_head",
    "trace_independent_label",
    "frozen_trace_head",
    "rand",
    "ridge_trace",
    "ridge_native",
)
FEATURE_DIMENSION = 16
HIDDEN_DIMENSION = 64
LEARNING_RATE = 0.5
ELEMENT_UPDATE_CLIP = 0.05
THETA_CLIP = 3.0
DEFAULT_RIDGE_ALPHA = 1e-3
ORDERED_CPU_WORKERS = 5
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_CONFIDENCE = 0.95
READINESS_BEFORE = 21
READINESS_AFTER_PASS = 25
READINESS_AFTER_FAIL = 21

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = Path("docs/EXPERIMENT_000_ONLINE_TRACE_LEARNING_PROTOCOL.md")
CONFIG_PATH = Path("configs/experiment_000_online_trace_learning.toml")
CANONICAL_FREEZE_RECORD_PATH = Path(
    "configs/experiment_000_online_trace_learning.freeze.json"
)
FREEZE_RECORD_PATH = CANONICAL_FREEZE_RECORD_PATH
MODULE_PATH = Path(
    "src/adaptive_learning_substrate/experiment_000_online_trace_learning.py"
)
TEST_PATH = Path("tests/test_experiment_000_online_trace_learning.py")
ARTIFACT_DIRECTORY = Path("artifacts/experiment_000/online_trace_learning")
PRIMARY_REPORT_PATH = ARTIFACT_DIRECTORY / "custom_seeds_80_89_full.json"
RERUN_REPORT_PATH = ARTIFACT_DIRECTORY / "custom_seeds_80_89_full_rerun.json"
DETERMINISM_VERIFICATION_PATH = ARTIFACT_DIRECTORY / "DETERMINISM_VERIFICATION.json"
SCRATCH_DIRECTORY = Path("artifacts/scratch/experiment_000_online_trace_learning")

_PROCESS_INSTANCE_TOKEN = hashlib.sha256(os.urandom(32)).hexdigest()
_PROCESS_STARTED_UTC = datetime.now(UTC).isoformat(timespec="microseconds").replace(
    "+00:00", "Z"
)
FREEZE_SCHEMA_VERSION = "experiment-000-online-trace-learning-freeze-v1"
FREEZE_ANCHOR_ENV = "TRACE_HEAD_L1_FREEZE_SHA256"
EVIDENCE_SCHEMA_VERSION = "experiment-000-online-trace-learning-evidence-v1"
CHALLENGE_SCHEMA_VERSION = (
    "experiment-000-online-trace-learning-reconstruction-challenge-v1"
)
_PACKAGE_INIT_PATH = Path("src/adaptive_learning_substrate/__init__.py")
_STATIC_FREEZE_PATHS: tuple[Path, ...] = (
    PROTOCOL_PATH,
    CONFIG_PATH,
    TEST_PATH,
    Path("docs/EXPERIMENT_000_READOUT_TRACE_A3_PROTOCOL.md"),
    Path("configs/experiment_000_readout_trace_a3.toml"),
    Path("tests/test_experiment_000_readout_trace_a3.py"),
    Path("docs/EXPERIMENT_000_TRACE_HEAD_ALIGNMENT_PROTOCOL.md"),
    Path("configs/experiment_000_trace_head_alignment.toml"),
    Path("tests/test_experiment_000_trace_head_alignment.py"),
)


@dataclass(frozen=True, slots=True)
class CellSpec:
    """One independently trained delay/control cell."""

    name: str
    distractor_count: int
    cue_removed: bool
    train_examples: int
    eval_examples: int

    def validate(self, *, official: bool) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("cell name must be nonempty")
        if (
            isinstance(self.distractor_count, bool)
            or type(self.distractor_count) is not int
            or self.distractor_count <= 0
        ):
            raise ValueError("distractor_count must be a positive exact integer")
        if type(self.cue_removed) is not bool:
            raise TypeError("cue_removed must be an exact bool")
        for field_name, count in (
            ("train_examples", self.train_examples),
            ("eval_examples", self.eval_examples),
        ):
            if (
                isinstance(count, bool)
                or type(count) is not int
                or count <= 0
                or count % 2
            ):
                raise ValueError(f"{field_name} must be a positive even exact integer")
        if self.train_examples != self.eval_examples:
            raise ValueError("the frozen data generator requires equal train/eval sizes")
        if not official and max(self.train_examples, self.eval_examples) > 16:
            raise ValueError("scratch smoke cells are capped at 16 examples per split")


FULL_CELLS: tuple[CellSpec, ...] = (
    CellSpec("d8_clean", 8, False, 512, 512),
    CellSpec("d4_clean", 4, False, 256, 256),
    CellSpec("d16_clean", 16, False, 256, 256),
    CellSpec("d8_cue_removed", 8, True, 256, 256),
)
SMOKE_CELLS: tuple[CellSpec, ...] = tuple(
    CellSpec(
        cell.name,
        cell.distractor_count,
        cell.cue_removed,
        SMOKE_EXAMPLES_PER_SPLIT,
        SMOKE_EXAMPLES_PER_SPLIT,
    )
    for cell in FULL_CELLS
)
_FULL_CELL_NAMES = tuple(cell.name for cell in FULL_CELLS)

GATE_THRESHOLDS: dict[str, float | int] = {
    "d8_trace_mean_accuracy_min": 0.75,
    "d8_trace_seed_accuracy_min": 0.70,
    "d8_trace_seed_count_min": 8,
    "d8_advantage_vs_frozen_min": 0.20,
    "d8_advantage_vs_native_min": 0.10,
    "d8_advantage_vs_independent_label_min": 0.15,
    "d8_online_below_trace_ridge_max": 0.05,
    "d4_trace_mean_accuracy_min": 0.75,
    "d16_trace_mean_accuracy_min": 0.65,
    "d4_d16_seed_accuracy_min": 0.60,
    "d4_d16_seed_count_min": 8,
    "d16_drop_from_d8_max": 0.15,
    "chance_accuracy_min": 0.45,
    "chance_accuracy_max": 0.55,
    "holm_adjusted_p_max_exclusive": 0.05,
}

_GATE_ORDER: tuple[str, ...] = (
    "P01_D8_TRACE_MEAN_ACCURACY",
    "P02_D8_TRACE_SEED_COUNT",
    "P03_D8_ADVANTAGE_VS_FROZEN",
    "P04_D8_ADVANTAGE_VS_NATIVE",
    "P05_D8_ADVANTAGE_VS_INDEPENDENT_LABEL",
    "P06_D8_ONLINE_WITHIN_TRACE_RIDGE",
    "S01_TRACE_VS_FROZEN_HOLM_P",
    "S02_TRACE_VS_NATIVE_HOLM_P",
    "S03_TRACE_VS_INDEPENDENT_HOLM_P",
    "B01_TRACE_VS_FROZEN_BOOTSTRAP_LOWER",
    "B02_TRACE_VS_NATIVE_BOOTSTRAP_LOWER",
    "B03_TRACE_VS_INDEPENDENT_BOOTSTRAP_LOWER",
    "R01_D4_TRACE_MEAN_ACCURACY",
    "R02_D16_TRACE_MEAN_ACCURACY",
    "R03_D4_TRACE_SEED_COUNT",
    "R04_D16_TRACE_SEED_COUNT",
    "R05_D16_DROP_FROM_D8",
    "N01_D8_CUE_REMOVED_CHANCE",
    "N02_D8_INDEPENDENT_LABEL_CHANCE",
    "N03_D8_FROZEN_HEAD_CHANCE",
    "I01_ALL_INTEGRITY_GATES",
)

_CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("trace_vs_frozen", "frozen_trace_head", "S01_TRACE_VS_FROZEN_HOLM_P"),
    ("trace_vs_native", "native_head", "S02_TRACE_VS_NATIVE_HOLM_P"),
    (
        "trace_vs_independent_label",
        "trace_independent_label",
        "S03_TRACE_VS_INDEPENDENT_HOLM_P",
    ),
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


def _structural_mask_sha256(graph: RecurrentEventGraph) -> str:
    """Hash endpoints, delays, kinds, and the otherwise easy-to-miss plastic flag."""

    return _sha256_json(
        {
            "input_nodes": list(graph.input_nodes),
            "hidden_nodes": list(graph.hidden_nodes),
            "output_node": graph.output_node,
            "edges": [
                (
                    edge.edge_id,
                    edge.source,
                    edge.destination,
                    edge.kind,
                    edge.delay_ticks,
                    edge.plastic,
                )
                for edge in graph.edges
            ],
        }
    )


def _record_self_hash(record: Mapping[str, Any], field: str) -> str:
    payload = copy.deepcopy(dict(record))
    payload.pop(field, None)
    return _sha256_json(payload)


def _derive_seed(master_seed: int, namespace: str) -> int:
    material = f"{PROTOCOL_VERSION}|{master_seed}|{namespace}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:16], "big")


def derive_cell_data_seed(master_seed: int, cell_name: str) -> tuple[int, str]:
    """Derive a cell-specific data root while leaving the graph seed frozen."""

    namespace = f"{PROTOCOL_VERSION}/seed-{master_seed}/data-stream/{cell_name}"
    seed = _derive_seed(master_seed, namespace)
    require_seed_role(seed, SeedRole.CUSTOM)
    return seed, namespace


def _single_thread_blas_state() -> list[dict[str, Any]]:
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
        raise RuntimeError(f"TRACE-HEAD-L1 requires single-thread BLAS: {state}")
    return state


def _local_module_source(module_name: str) -> Path | None:
    if not module_name.startswith("adaptive_learning_substrate"):
        return None
    root = _project_path("src")
    stem = root.joinpath(*module_name.split("."))
    candidates = (stem.with_suffix(".py"), stem / "__init__.py")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.relative_to(PROJECT_ROOT)
    return None


def _scientific_import_closure() -> tuple[Path, ...]:
    """Return every local Python source transitively imported by this runner."""

    pending = [_PACKAGE_INIT_PATH, MODULE_PATH]
    discovered: set[Path] = set()
    while pending:
        relative = pending.pop()
        if relative in discovered:
            continue
        target = _project_path(relative)
        if not target.is_file():
            raise RuntimeError(f"scientific import source is absent: {relative}")
        discovered.add(relative)
        tree = ast.parse(target.read_text(encoding="utf-8"), filename=str(relative))
        module_parts = list(relative.relative_to("src").with_suffix("").parts)
        is_package = module_parts[-1] == "__init__"
        if is_package:
            module_parts.pop()
        module_name = ".".join(module_parts)
        package_name = (
            module_name if is_package else module_name.rsplit(".", maxsplit=1)[0]
        )
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package_parts = package_name.split(".")
                    keep = len(package_parts) - node.level + 1
                    if keep < 0:
                        raise RuntimeError(f"invalid local import in {relative}")
                    base = package_parts[:keep]
                    if node.module:
                        imported.append(".".join((*base, node.module)))
                    else:
                        imported.extend(
                            ".".join((*base, alias.name)) for alias in node.names
                        )
                elif node.module:
                    imported.append(node.module)
            for imported_name in imported:
                imported_path = _local_module_source(imported_name)
                if imported_path is not None and imported_path not in discovered:
                    pending.append(imported_path)
    return tuple(sorted(discovered, key=lambda path: path.as_posix()))


def _source_paths() -> tuple[Path, ...]:
    paths = (*_STATIC_FREEZE_PATHS, *_scientific_import_closure())
    unique: dict[str, Path] = {}
    for path in paths:
        unique.setdefault(path.as_posix(), path)
    return tuple(unique.values())


def _source_manifest() -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for relative in _source_paths():
        target = _project_path(relative)
        if not target.is_file():
            raise RuntimeError(f"frozen learning source is absent: {relative}")
        source_bytes = target.read_bytes()
        files.append(
            {
                "path": relative.as_posix(),
                "bytes": len(source_bytes),
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
            }
        )
    return {"files": files, "bundle_sha256": _sha256_json(files)}


def _freeze_file_is_read_only(path: Path) -> bool:
    file_stat = path.stat()
    if os.name == "nt":
        readonly_flag = getattr(stat, "FILE_ATTRIBUTE_READONLY", 0x1)
        return bool(getattr(file_stat, "st_file_attributes", 0) & readonly_flag)
    return not bool(stat.S_IMODE(file_stat.st_mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def create_pre_official_source_freeze(
    output_path: str | Path = FREEZE_RECORD_PATH,
    *,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Create once, self-hash, reopen, and make the external freeze read-only."""

    target = _project_path(output_path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite source freeze: {target}")
    official_evidence_paths = tuple(
        candidate
        for report_path in (
            _project_path(PRIMARY_REPORT_PATH),
            _project_path(RERUN_REPORT_PATH),
            _project_path(DETERMINISM_VERIFICATION_PATH),
        )
        for candidate in (report_path, report_path.with_suffix(".sha256"))
    )
    if (
        target.resolve() == _project_path(CANONICAL_FREEZE_RECORD_PATH).resolve()
        and any(path.exists() for path in official_evidence_paths)
    ):
        raise RuntimeError("official evidence exists before the source freeze")
    timestamp = (
        datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
        if created_utc is None
        else created_utc
    )
    if not _is_utc_timestamp(timestamp):
        raise ValueError("source-freeze timestamp must be an ISO UTC value")
    manifest = _source_manifest()
    record: dict[str, Any] = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "status": "FROZEN_PRE_OFFICIAL",
        "created_utc": timestamp,
        "recorded_before_official_metrics": True,
        "official_artifacts_absent_at_creation": True,
        "entry_points": [_PACKAGE_INIT_PATH.as_posix(), MODULE_PATH.as_posix()],
        "scientific_import_closure": [
            path.as_posix() for path in _scientific_import_closure()
        ],
        "source_manifest": manifest,
    }
    record["record_sha256"] = _record_self_hash(record, "record_sha256")
    written = _atomic_write_json(target, record)
    written.chmod(stat.S_IREAD)
    reopened = json.loads(written.read_text(encoding="utf-8"))
    if reopened != record or not _freeze_file_is_read_only(written):
        raise RuntimeError("source freeze did not persist immutably")
    return record


def validate_pre_official_source_freeze() -> dict[str, Any]:
    """Validate the external immutable freeze and every covered source byte."""

    target = _project_path(FREEZE_RECORD_PATH)
    if not target.is_file() or not _freeze_file_is_read_only(target):
        raise RuntimeError("pre-official source-freeze record is absent or mutable")
    freeze_bytes = target.read_bytes()
    freeze_file_sha256 = hashlib.sha256(freeze_bytes).hexdigest()
    external_anchor = os.environ.get(FREEZE_ANCHOR_ENV)
    if not _is_sha256(external_anchor) or external_anchor != freeze_file_sha256:
        raise RuntimeError("pre-official source-freeze external anchor changed")
    record = json.loads(freeze_bytes)
    closure = [path.as_posix() for path in _scientific_import_closure()]
    current_manifest = _source_manifest()
    if (
        record.get("schema_version") != FREEZE_SCHEMA_VERSION
        or record.get("protocol_name") != PROTOCOL_NAME
        or record.get("protocol_version") != PROTOCOL_VERSION
        or record.get("status") != "FROZEN_PRE_OFFICIAL"
        or record.get("recorded_before_official_metrics") is not True
        or record.get("official_artifacts_absent_at_creation") is not True
        or not _is_utc_timestamp(record.get("created_utc"))
        or record.get("entry_points")
        != [_PACKAGE_INIT_PATH.as_posix(), MODULE_PATH.as_posix()]
        or record.get("scientific_import_closure") != closure
        or record.get("source_manifest") != current_manifest
        or record.get("record_sha256")
        != _record_self_hash(record, "record_sha256")
    ):
        raise RuntimeError("pre-official source freeze or covered bytes changed")
    result = copy.deepcopy(record)
    result["freeze_file_sha256"] = freeze_file_sha256
    result["external_anchor_environment"] = FREEZE_ANCHOR_ENV
    return result


def _as_feature_vector(values: Sequence[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (FEATURE_DIMENSION,) or not np.all(np.isfinite(vector)):
        raise ValueError("features must be a finite 16-dimensional vector")
    return vector


def _bipolar_target(bit_or_target: float) -> float:
    value = float(bit_or_target)
    if value == 0.0:
        return -1.0
    if value == 1.0:
        return 1.0
    if value == -1.0:
        return -1.0
    raise ValueError("target must be binary or bipolar")


def _dot(left: np.ndarray, right: np.ndarray) -> float:
    return float(math.fsum(float(a) * float(b) for a, b in zip(left, right)))


@dataclass(frozen=True, slots=True)
class PendingPrediction:
    """Feature/prediction state captured before a training target is revealed."""

    sequence: int
    features: np.ndarray
    activation: float
    prediction: float
    theta_before_sha256: str


class OnlineTanhHead:
    """Strictly ordered zero-initialized 16-parameter TRACE-HEAD-L1 learner."""

    def __init__(self) -> None:
        self._theta = np.zeros(FEATURE_DIMENSION, dtype=np.float64)
        self._pending: PendingPrediction | None = None
        self._prediction_count = 0
        self._target_reveal_count = 0
        self._update_count = 0
        self._parameter_read_touches = 0
        self._parameter_write_touches = 0
        self._maximum_absolute_delta = 0.0

    @property
    def theta(self) -> np.ndarray:
        return self._theta.copy()

    @property
    def pending(self) -> bool:
        return self._pending is not None

    @property
    def ledger(self) -> dict[str, int | float]:
        return {
            "prediction_count": self._prediction_count,
            "target_reveal_count": self._target_reveal_count,
            "update_count": self._update_count,
            "parameter_read_touches": self._parameter_read_touches,
            "parameter_write_touches": self._parameter_write_touches,
            "maximum_absolute_delta": self._maximum_absolute_delta,
        }

    def _predict(self, features: Sequence[float] | np.ndarray) -> tuple[np.ndarray, float, float]:
        vector = _as_feature_vector(features).copy()
        score = _dot(self._theta, vector)
        activation = math.tanh(score)
        prediction = 1.0 if activation >= 0.0 else -1.0
        if not math.isfinite(activation):
            raise FloatingPointError("online head produced a non-finite activation")
        self._prediction_count += 1
        self._parameter_read_touches += FEATURE_DIMENSION
        return vector, activation, prediction

    def predict_for_training(
        self, features: Sequence[float] | np.ndarray
    ) -> PendingPrediction:
        """Predict and retain only pre-reveal state for one subsequent update."""

        if self._pending is not None:
            raise RuntimeError("a target must resolve the pending prediction first")
        vector, activation, prediction = self._predict(features)
        pending = PendingPrediction(
            sequence=self._prediction_count - 1,
            features=vector,
            activation=activation,
            prediction=prediction,
            theta_before_sha256=_array_sha256(self._theta),
        )
        self._pending = pending
        return pending

    def reveal_target_and_update(self, target: float) -> dict[str, Any]:
        """Reveal a bipolar target and apply the frozen update to its prediction."""

        pending = self._pending
        if pending is None:
            raise RuntimeError("target reveal requires a pending pre-reveal prediction")
        value = _bipolar_target(target)
        normalizer = max(1.0, _dot(pending.features, pending.features))
        raw_delta = (
            LEARNING_RATE
            * (value - pending.activation)
            * (1.0 - pending.activation * pending.activation)
            * pending.features
            / normalizer
        )
        delta = np.clip(raw_delta, -ELEMENT_UPDATE_CLIP, ELEMENT_UPDATE_CLIP)
        theta_after = np.clip(self._theta + delta, -THETA_CLIP, THETA_CLIP)
        if not np.all(np.isfinite(theta_after)):
            raise FloatingPointError("online head update produced non-finite parameters")
        self._theta = theta_after
        self._pending = None
        self._target_reveal_count += 1
        self._update_count += 1
        self._parameter_write_touches += FEATURE_DIMENSION
        self._maximum_absolute_delta = max(
            self._maximum_absolute_delta,
            float(np.max(np.abs(delta))),
        )
        return {
            "normalizer": normalizer,
            "raw_delta_sha256": _array_sha256(raw_delta),
            "delta_sha256": _array_sha256(delta),
            "theta_before_sha256": pending.theta_before_sha256,
            "theta_after_sha256": _array_sha256(theta_after),
            "maximum_absolute_delta": float(np.max(np.abs(delta))),
            "target": value,
        }

    def predict_for_evaluation(
        self, features: Sequence[float] | np.ndarray
    ) -> tuple[float, float]:
        """Predict without creating an update-capable pending state."""

        if self._pending is not None:
            raise RuntimeError("evaluation cannot begin with a pending training target")
        _, activation, prediction = self._predict(features)
        return activation, prediction


def normalized_online_update(
    theta: Sequence[float] | np.ndarray,
    features: Sequence[float] | np.ndarray,
    target: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Pure reference implementation of the frozen TRACE-HEAD-L1 update."""

    current = _as_feature_vector(theta)
    vector = _as_feature_vector(features)
    value = _bipolar_target(target)
    activation = math.tanh(_dot(current, vector))
    normalizer = max(1.0, _dot(vector, vector))
    raw = (
        LEARNING_RATE
        * (value - activation)
        * (1.0 - activation * activation)
        * vector
        / normalizer
    )
    delta = np.clip(raw, -ELEMENT_UPDATE_CLIP, ELEMENT_UPDATE_CLIP)
    updated = np.clip(current + delta, -THETA_CLIP, THETA_CLIP)
    return updated, delta, activation, normalizer


class _IndependentReplayHead:
    """Small equation-level replay state independent of ``OnlineTanhHead``."""

    def __init__(self) -> None:
        self.theta = np.zeros(FEATURE_DIMENSION, dtype=np.float64)
        self.prediction_count = 0
        self.target_reveal_count = 0
        self.update_count = 0
        self.maximum_absolute_delta = 0.0

    def predict(
        self, features: Sequence[float] | np.ndarray
    ) -> tuple[float, float, str]:
        vector = _as_feature_vector(features)
        activation = math.tanh(_dot(self.theta, vector))
        prediction = 1.0 if activation >= 0.0 else -1.0
        before = _array_sha256(self.theta)
        self.prediction_count += 1
        return activation, prediction, before

    def update(self, features: Sequence[float] | np.ndarray, target: float) -> str:
        updated, delta, _, _ = normalized_online_update(
            self.theta, features, target
        )
        self.theta = updated
        self.target_reveal_count += 1
        self.update_count += 1
        self.maximum_absolute_delta = max(
            self.maximum_absolute_delta, float(np.max(np.abs(delta)))
        )
        return _array_sha256(self.theta)

    def summary(self) -> dict[str, Any]:
        return {
            "feature_dimension": FEATURE_DIMENSION,
            "parameter_count": FEATURE_DIMENSION,
            "bias_parameter_count": 0,
            "initialization": "exact_positive_zero",
            "theta": self.theta.tolist(),
            "theta_sha256": _array_sha256(self.theta),
            "maximum_absolute_theta": float(np.max(np.abs(self.theta))),
            "all_parameters_finite": bool(np.all(np.isfinite(self.theta))),
            "all_parameters_within_clip": bool(
                np.all(np.abs(self.theta) <= THETA_CLIP)
            ),
            "pending_prediction": False,
            "ledger": {
                "prediction_count": self.prediction_count,
                "target_reveal_count": self.target_reveal_count,
                "update_count": self.update_count,
                "parameter_read_touches": self.prediction_count
                * FEATURE_DIMENSION,
                "parameter_write_touches": self.update_count * FEATURE_DIMENSION,
                "maximum_absolute_delta": self.maximum_absolute_delta,
            },
        }


def load_verified_learning_prerequisites() -> dict[str, Any]:
    """Verify A3 first, then the A3-bound selected-trace alignment chain."""

    from .experiment_000_readout_trace_a3 import load_verified_a3_selection
    from .experiment_000_trace_head_alignment import load_verified_trace_head_alignment

    a3 = load_verified_a3_selection()
    alignment = load_verified_trace_head_alignment()
    retention = float(a3.get("retention_coefficient", math.nan))
    if (
        not str(a3.get("status", "")).startswith("SELECTED:")
        or alignment.get("status") != "TRACE_HEAD_ALIGNMENT_PASS"
        or alignment.get("selected_condition") != a3.get("selected_condition")
        or alignment.get("trace_retention") != retention
    ):
        raise RuntimeError("selected-trace alignment does not bind the verified A3 choice")
    if not math.isfinite(retention) or not 0.0 < retention < 1.0:
        raise RuntimeError("verified A3 retention is outside the learning domain")
    return {
        "a3": a3,
        "trace_head_alignment": alignment,
        "selected_condition": a3["selected_condition"],
        "trace_retention": retention,
        "binding_sha256": _sha256_json(
            {"a3": a3, "trace_head_alignment": alignment}
        ),
    }


def _forward_without_target(
    graph: RecurrentEventGraph,
    episode: Experiment000Episode,
    *,
    cue_removed: bool,
) -> QueryResult:
    """Forward one episode while never reading its separately stored target."""

    graph.begin_episode(episode.episode_id)
    if cue_removed:
        graph.step({})
    else:
        graph.step({"cue": 1.0 if episode.cue else -1.0})
    for bit in episode.noise:
        graph.step({"noise": 1.0 if bit else -1.0})
    query = graph.query()
    expected_tick = len(episode.noise) + 3
    if query.tick != expected_tick:
        raise RuntimeError(
            f"query returned at tick {query.tick}; expected {expected_tick}"
        )
    if graph.audit["queried_output_event_id"] != query.event_id:
        raise RuntimeError("query identity does not match the forced output event")
    return query


@dataclass(frozen=True, slots=True)
class EpisodeObservation:
    """Both 16-D representations created before the target is read."""

    trace_features: np.ndarray
    native_features: np.ndarray
    trace_snapshot: np.ndarray
    query_tick: int
    event_payload_sha256: str
    trace_weighted_output: float
    native_output: float


_TRACE_COMPUTE_FIELDS: tuple[str, ...] = (
    "hidden_activation_evaluations",
    "local_trace_read_touches",
    "local_trace_decay_touches",
    "local_trace_write_touches",
    "local_trace_reset_touches",
    "local_trace_observation_touches",
)

_CELL_INTEGRITY_KEYS = frozenset(
    {
        "train_eval_episode_ids_disjoint",
        "train_episode_ids_unique",
        "eval_episode_ids_unique",
        "train_eval_noise_stream_ids_disjoint",
        "data_rng_seeds_disjoint",
        "baseline_rng_seeds_disjoint",
        "data_and_baseline_rng_seeds_disjoint",
        "rng_namespaces_disjoint",
        "one_train_pass",
        "one_eval_pass",
        "no_replay",
        "prediction_precedes_target_reveal",
        "no_eval_parameter_writes",
        "native_topology_unchanged",
        "trace_topology_unchanged",
        "topologies_match",
        "native_structural_mask_unchanged",
        "trace_structural_mask_unchanged",
        "structural_masks_match",
        "native_weights_unchanged",
        "trace_weights_unchanged",
        "weights_match",
        "candidate_native_events_exact",
        "native_trace_ledgers_reconcile",
        "no_recurrent_weight_writes",
        "trace_bound_ok",
        "trace_resets_valid",
        "trace_reset_count_exact",
        "trace_reset_touches_exact",
        "trace_observation_touches_exact",
        "trace_read_decay_write_touches_reconcile",
        "head_update_counts_exact",
        "head_target_reveal_counts_exact",
        "native_trace_compute_budget_matched",
        "native_dummy_trace_work_exact",
        "native_dummy_trace_state_exact",
        "native_dummy_trace_history_exact",
        "native_dummy_trace_effective_snapshot_exact",
        "native_dummy_trace_weighted_output_exact",
        "native_dummy_trace_diagnostics_exact",
        "head_parameter_and_feature_dimensions_exact",
        "head_values_finite_and_bounded",
        "ridge_train_only",
        "ridge_evaluation_fit_count_zero",
        "all_values_finite",
    }
)


class NativeDummyTraceState:
    """Faithfully replay the candidate trace as blind native-baseline work.

    The dummy consumes only native-hidden activation records which have already
    been proven identical between the literal native graph and trace clone.  It
    independently repeats the registered state/timestamp updates, history
    construction, query snapshot, feature gather, weighted dot, and ``tanh``.
    Its values are never returned to or read by the native prediction head.
    """

    def __init__(self, trace_retention: float) -> None:
        if isinstance(trace_retention, bool) or not math.isfinite(
            float(trace_retention)
        ):
            raise ValueError("dummy trace retention must be finite")
        self._retention = float(trace_retention)
        if not 0.0 <= self._retention <= 1.0:
            raise ValueError("dummy trace retention must be in [0, 1]")
        self._active = self._retention > 0.0
        self._hidden_nodes: tuple[str, ...] | None = None
        self._state: dict[str, float] = {}
        self._timestamps: dict[str, int | None] = {}
        self._ledger = {field: 0 for field in _TRACE_COMPUTE_FIELDS}
        self._reset_count = 0
        self._resets_valid = True
        self._bound_ok = True
        self._state_exact = True
        self._history_exact = True
        self._effective_snapshot_exact = True
        self._weighted_outputs_exact = True
        self._dummy_state_transcript: list[str] = []
        self._candidate_state_transcript: list[str] = []
        self._dummy_history_transcript: list[str] = []
        self._candidate_history_transcript: list[str] = []
        self._dummy_effective_transcript: list[str] = []
        self._candidate_effective_transcript: list[str] = []
        self._dummy_weighted_outputs: list[float] = []
        self._candidate_weighted_outputs: list[float] = []

    @property
    def ledger(self) -> dict[str, int]:
        return dict(self._ledger)

    @property
    def state_sha256(self) -> str:
        return _sha256_json(self._dummy_state_transcript)

    @property
    def candidate_state_sha256(self) -> str:
        return _sha256_json(self._candidate_state_transcript)

    @property
    def history_sha256(self) -> str:
        return _sha256_json(self._dummy_history_transcript)

    @property
    def candidate_history_sha256(self) -> str:
        return _sha256_json(self._candidate_history_transcript)

    @property
    def effective_snapshot_sha256(self) -> str:
        return _sha256_json(self._dummy_effective_transcript)

    @property
    def candidate_effective_snapshot_sha256(self) -> str:
        return _sha256_json(self._candidate_effective_transcript)

    @property
    def weighted_outputs_sha256(self) -> str:
        return _array_sha256(np.asarray(self._dummy_weighted_outputs, dtype=np.float64))

    @property
    def candidate_weighted_outputs_sha256(self) -> str:
        return _array_sha256(
            np.asarray(self._candidate_weighted_outputs, dtype=np.float64)
        )

    @property
    def state_exact(self) -> bool:
        return self._state_exact

    @property
    def history_exact(self) -> bool:
        return self._history_exact

    @property
    def effective_snapshot_exact(self) -> bool:
        return self._effective_snapshot_exact

    @property
    def weighted_outputs_exact(self) -> bool:
        return self._weighted_outputs_exact

    @property
    def reset_count(self) -> int:
        return self._reset_count

    @property
    def resets_valid(self) -> bool:
        return self._resets_valid

    @property
    def bound_ok(self) -> bool:
        return self._bound_ok

    @staticmethod
    def _same_float(left: float, right: float) -> bool:
        """Use IEEE-754 bytes so signed zero and every mantissa bit matter."""

        return struct.pack("<d", float(left)) == struct.pack("<d", float(right))

    def mirror_episode(
        self,
        trace_graph: ReadoutTraceRecurrentEventGraph,
        observation: EpisodeObservation,
        *,
        feature_edge_ids: tuple[str, ...],
    ) -> None:
        """Repeat one complete passive-trace episode without feeding the head."""

        hidden_nodes = tuple(trace_graph.hidden_nodes)
        if len(hidden_nodes) != HIDDEN_DIMENSION or len(set(hidden_nodes)) != len(
            hidden_nodes
        ):
            raise RuntimeError("dummy trace requires the frozen 64 hidden nodes")
        if self._hidden_nodes is None:
            self._hidden_nodes = hidden_nodes
        elif self._hidden_nodes != hidden_nodes:
            raise RuntimeError("dummy trace hidden-node identity changed")

        records = trace_graph.trace_activation_records
        candidate_history = [asdict(record) for record in records]
        dummy_history: list[dict[str, Any]] = []

        if self._active:
            self._state = {node: 0.0 for node in hidden_nodes}
            self._timestamps = {node: None for node in hidden_nodes}
            self._ledger["local_trace_reset_touches"] += len(hidden_nodes)
            self._reset_count += 1
            reset_valid = all(
                value == 0.0 and math.copysign(1.0, value) == 1.0
                for value in self._state.values()
            ) and all(value is None for value in self._timestamps.values())
            self._resets_valid = self._resets_valid and reset_valid

        for record in records:
            self._ledger["hidden_activation_evaluations"] += 1
            if self._active:
                if record.node not in self._state:
                    raise RuntimeError("dummy trace activation names an unknown node")
                self._ledger["local_trace_read_touches"] += 1
                previous_tick = self._timestamps[record.node]
                if previous_tick is None:
                    retained = 0.0
                else:
                    gap = int(record.tick) - previous_tick
                    if gap <= 0:
                        raise RuntimeError("dummy trace processing gap must be positive")
                    retained = float(
                        (self._retention**gap) * self._state[record.node]
                    )
                self._ledger["local_trace_decay_touches"] += 1
                updated = retained + (1.0 - self._retention) * float(
                    record.activation
                )
                if not math.isfinite(updated):
                    raise FloatingPointError("dummy trace produced a non-finite value")
                if abs(updated) > 1.0 + 1e-12:
                    self._bound_ok = False
                    raise FloatingPointError("dummy trace left the frozen [-1, 1] bound")
                self._state[record.node] = updated
                self._timestamps[record.node] = int(record.tick)
                self._ledger["local_trace_write_touches"] += 1
            else:
                retained = 0.0
                updated = 0.0
            self._history_exact = self._history_exact and self._same_float(
                retained, record.retained_trace
            ) and self._same_float(updated, record.trace_after)
            dummy_history.append(
                {
                    "node": record.node,
                    "tick": int(record.tick),
                    "activation": float(record.activation),
                    "retained_trace": retained,
                    "trace_after": updated,
                    "emitted": bool(record.emitted),
                }
            )

        candidate_history_sha = _sha256_json(candidate_history)
        dummy_history_sha = _sha256_json(dummy_history)
        self._candidate_history_transcript.append(candidate_history_sha)
        self._dummy_history_transcript.append(dummy_history_sha)
        self._history_exact = self._history_exact and (
            dummy_history_sha == candidate_history_sha
        )

        candidate_raw = trace_graph.trace_snapshot
        dummy_raw = (
            {node: (self._state[node], self._timestamps[node]) for node in hidden_nodes}
            if self._active
            else {}
        )
        candidate_state_rows = [
            [node, float(candidate_raw[node][0]), candidate_raw[node][1]]
            for node in hidden_nodes
        ] if self._active else []
        dummy_state_rows = [
            [node, float(dummy_raw[node][0]), dummy_raw[node][1]]
            for node in hidden_nodes
        ] if self._active else []
        candidate_state_sha = _sha256_json(candidate_state_rows)
        dummy_state_sha = _sha256_json(dummy_state_rows)
        self._candidate_state_transcript.append(candidate_state_sha)
        self._dummy_state_transcript.append(dummy_state_sha)
        self._state_exact = self._state_exact and dummy_state_sha == candidate_state_sha

        effective: list[float] = []
        if self._active:
            for node in hidden_nodes:
                previous_tick = self._timestamps[node]
                if previous_tick is None:
                    value = 0.0
                else:
                    gap = observation.query_tick - previous_tick
                    if gap < 0:
                        raise RuntimeError("dummy trace timestamp exceeds query tick")
                    value = float((self._retention**gap) * self._state[node])
                effective.append(value)
                self._ledger["local_trace_observation_touches"] += 1
        else:
            effective = [0.0 for _ in hidden_nodes]

        candidate_effective = observation.trace_snapshot.tolist()
        candidate_effective_sha = _array_sha256(observation.trace_snapshot)
        dummy_effective_sha = _array_sha256(
            np.asarray(effective, dtype=np.float64)
        )
        self._candidate_effective_transcript.append(candidate_effective_sha)
        self._dummy_effective_transcript.append(dummy_effective_sha)
        bit_exact_effective = len(effective) == len(candidate_effective) and all(
            self._same_float(left, right)
            for left, right in zip(effective, candidate_effective, strict=True)
        )
        self._effective_snapshot_exact = (
            self._effective_snapshot_exact
            and bit_exact_effective
            and dummy_effective_sha == candidate_effective_sha
        )

        source_by_edge = {
            edge.edge_id: edge.source
            for edge in trace_graph.edges
            if edge.kind == "output" and edge.destination == trace_graph.output_node
        }
        if tuple(sorted(source_by_edge)) != feature_edge_ids:
            raise RuntimeError("dummy trace output-feature identity changed")
        effective_by_node = dict(zip(hidden_nodes, effective, strict=True))
        dummy_features = [
            effective_by_node[source_by_edge[edge_id]] for edge_id in feature_edge_ids
        ]
        dummy_weighted = math.tanh(
            math.fsum(
                trace_graph.weights[edge_id] * dummy_features[index]
                for index, edge_id in enumerate(feature_edge_ids)
            )
        )
        if not math.isfinite(dummy_weighted):
            raise FloatingPointError("dummy weighted trace output is non-finite")
        self._dummy_weighted_outputs.append(dummy_weighted)
        self._candidate_weighted_outputs.append(observation.trace_weighted_output)
        self._weighted_outputs_exact = (
            self._weighted_outputs_exact
            and self._same_float(dummy_weighted, observation.trace_weighted_output)
        )

        if self._ledger != trace_graph.trace_ledger:
            raise RuntimeError("dummy and candidate trace ledgers differ")


def _event_payload(graph: RecurrentEventGraph) -> list[dict[str, Any]]:
    return [asdict(event) for event in graph.unit_events]


def observe_episode(
    native_graph: RecurrentEventGraph,
    trace_graph: ReadoutTraceRecurrentEventGraph,
    episode: Experiment000Episode,
    *,
    cue_removed: bool,
    feature_edge_ids: tuple[str, ...],
) -> EpisodeObservation:
    """Run literal native and passive-trace graphs exactly once and compare them."""

    observation_touches_before = trace_graph.trace_ledger[
        "local_trace_observation_touches"
    ]
    native_query = _forward_without_target(
        native_graph, episode, cue_removed=cue_removed
    )
    trace_query = _forward_without_target(
        trace_graph, episode, cue_removed=cue_removed
    )
    native_events = native_graph.unit_events
    trace_events = trace_graph.unit_events
    if native_query != trace_query or native_events != trace_events:
        raise RuntimeError("passive trace changed native events or query activation")
    extracted = extract_query_features(native_graph, native_query)
    if tuple(edge_id for edge_id, _ in extracted) != feature_edge_ids:
        raise RuntimeError("native output-feature identity changed")
    native_features = np.asarray(
        [value for _, value in extracted], dtype=np.float64
    )
    effective = trace_graph.effective_trace(trace_query.tick)
    if tuple(effective) != trace_graph.hidden_nodes or len(effective) != HIDDEN_DIMENSION:
        raise RuntimeError("trace observation is not one ordered 64-D snapshot")
    observation_touches_after = trace_graph.trace_ledger[
        "local_trace_observation_touches"
    ]
    if observation_touches_after - observation_touches_before != HIDDEN_DIMENSION:
        raise RuntimeError("one episode must perform exactly one 64-D trace observation")
    trace_snapshot = np.asarray(
        [effective[node] for node in trace_graph.hidden_nodes], dtype=np.float64
    )
    source_by_edge = {
        edge.edge_id: edge.source
        for edge in trace_graph.edges
        if edge.kind == "output" and edge.destination == trace_graph.output_node
    }
    if tuple(sorted(source_by_edge)) != feature_edge_ids:
        raise RuntimeError("trace output-feature identity changed")
    trace_features = np.asarray(
        [effective[source_by_edge[edge_id]] for edge_id in feature_edge_ids],
        dtype=np.float64,
    )
    if (
        native_features.shape != (FEATURE_DIMENSION,)
        or trace_features.shape != (FEATURE_DIMENSION,)
        or trace_snapshot.shape != (HIDDEN_DIMENSION,)
        or not np.all(np.isfinite(native_features))
        or not np.all(np.isfinite(trace_features))
        or not np.all(np.isfinite(trace_snapshot))
    ):
        raise FloatingPointError("query representations are malformed or non-finite")
    weighted = math.tanh(
        math.fsum(
            trace_graph.weights[edge_id] * float(trace_features[index])
            for index, edge_id in enumerate(feature_edge_ids)
        )
    )
    if not math.isfinite(weighted):
        raise FloatingPointError("trace-weighted diagnostic output is non-finite")
    payload = _event_payload(native_graph)
    if payload != _event_payload(trace_graph):
        raise RuntimeError("serialized candidate/native events differ")
    return EpisodeObservation(
        trace_features=trace_features,
        native_features=native_features,
        trace_snapshot=trace_snapshot,
        query_tick=trace_query.tick,
        event_payload_sha256=_sha256_json(payload),
        trace_weighted_output=weighted,
        native_output=float(native_query.activation),
    )


def balanced_independent_labels(
    *, master_seed: int, cell_name: str, count: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return an exactly balanced train-label permutation from its own namespace."""

    if type(count) is not int or count <= 0 or count % 2:
        raise ValueError("independent-label count must be positive and even")
    namespace = (
        f"{PROTOCOL_VERSION}/seed-{master_seed}/"
        f"independent-label/{cell_name}/train"
    )
    seed = _derive_seed(master_seed, namespace)
    labels = np.concatenate(
        (
            np.full(count // 2, -1.0, dtype=np.float64),
            np.full(count // 2, 1.0, dtype=np.float64),
        )
    )
    generator = np.random.Generator(np.random.PCG64(seed))
    labels = labels[generator.permutation(count)]
    if int(np.count_nonzero(labels == -1.0)) != count // 2 or int(
        np.count_nonzero(labels == 1.0)
    ) != count // 2:
        raise RuntimeError("independent labels lost exact balance")
    return labels, {
        "namespace": namespace,
        "seed": seed,
        "bit_generator": "numpy.random.PCG64",
        "labels_sha256": _array_sha256(labels),
        "negative_count": count // 2,
        "positive_count": count // 2,
    }


def deterministic_random_predictions(
    *, master_seed: int, cell_name: str, split: str, count: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return unfitted deterministic random predictions from a disjoint namespace."""

    if split not in {"train", "eval"}:
        raise ValueError("random prediction split must be train or eval")
    if type(count) is not int or count <= 0:
        raise ValueError("random prediction count must be a positive exact integer")
    namespace = (
        f"{PROTOCOL_VERSION}/seed-{master_seed}/"
        f"random-prediction/{cell_name}/{split}"
    )
    seed = _derive_seed(master_seed, namespace)
    generator = np.random.Generator(np.random.PCG64(seed))
    predictions = np.where(
        generator.integers(0, 2, size=count, dtype=np.uint8) == 0,
        -1.0,
        1.0,
    ).astype(np.float64)
    return predictions, {
        "namespace": namespace,
        "seed": seed,
        "bit_generator": "numpy.random.PCG64",
        "predictions_sha256": _array_sha256(predictions),
    }


@dataclass(frozen=True, slots=True)
class RidgeModel:
    """Train-only standardized ridge model used only as a diagnostic."""

    mean: np.ndarray
    scale: np.ndarray
    active: np.ndarray
    coefficients: np.ndarray
    intercept: float
    alpha: float
    train_examples: int
    train_features_sha256: str
    train_targets_sha256: str

    def scores(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != FEATURE_DIMENSION:
            raise ValueError("ridge features must have shape (n, 16)")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("ridge features must be finite")
        normalized = np.zeros_like(matrix)
        normalized[:, self.active] = (
            matrix[:, self.active] - self.mean[self.active]
        ) / self.scale[self.active]
        scores = normalized @ self.coefficients + self.intercept
        if not np.all(np.isfinite(scores)):
            raise FloatingPointError("ridge produced non-finite scores")
        return scores

    def predictions(
        self, features: Sequence[Sequence[float]] | np.ndarray
    ) -> np.ndarray:
        return np.where(self.scores(features) >= 0.0, 1.0, -1.0)

    def diagnostic(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "feature_dimension": FEATURE_DIMENSION,
            "train_examples": self.train_examples,
            "fit_split": "train_only",
            "evaluation_fit_count": 0,
            "feature_standardization": "train_mean_and_population_sd_only",
            "active_feature_count": int(np.count_nonzero(self.active)),
            "inactive_feature_count": int(np.count_nonzero(~self.active)),
            "train_feature_mean": self.mean.tolist(),
            "train_feature_mean_sha256": _array_sha256(self.mean),
            "train_feature_scale": self.scale.tolist(),
            "train_feature_scale_sha256": _array_sha256(self.scale),
            "active_mask": self.active.tolist(),
            "active_mask_sha256": _array_sha256(self.active.astype(np.float64)),
            "coefficients": self.coefficients.tolist(),
            "coefficient_sha256": _array_sha256(self.coefficients),
            "coefficient_l2": float(np.linalg.norm(self.coefficients)),
            "intercept": self.intercept,
            "train_features_sha256": self.train_features_sha256,
            "train_targets_sha256": self.train_targets_sha256,
        }


def fit_train_only_ridge(
    features: Sequence[Sequence[float]] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
    *,
    alpha: float = DEFAULT_RIDGE_ALPHA,
) -> RidgeModel:
    """Fit a deterministic ridge model without accepting evaluation inputs."""

    matrix = np.asarray(features, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != FEATURE_DIMENSION
        or labels.shape != (matrix.shape[0],)
        or matrix.shape[0] == 0
    ):
        raise ValueError("ridge training data must have shapes (n, 16) and (n,)")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isin(labels, (-1.0, 1.0))):
        raise ValueError("ridge training values must be finite and targets bipolar")
    if isinstance(alpha, bool) or not math.isfinite(float(alpha)) or alpha <= 0.0:
        raise ValueError("ridge alpha must be finite and positive")
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0, ddof=0)
    active = scale > 1e-15
    normalized = np.zeros_like(matrix)
    normalized[:, active] = (matrix[:, active] - mean[active]) / scale[active]
    coefficients = np.zeros(FEATURE_DIMENSION, dtype=np.float64)
    active_count = int(np.count_nonzero(active))
    if active_count:
        active_matrix = normalized[:, active]
        gram = (active_matrix.T @ active_matrix) / matrix.shape[0]
        gram += float(alpha) * np.eye(active_count, dtype=np.float64)
        right = (active_matrix.T @ labels) / matrix.shape[0]
        coefficients[active] = np.linalg.solve(gram, right)
    intercept = float(np.mean(labels))
    if not np.all(np.isfinite(coefficients)) or not math.isfinite(intercept):
        raise FloatingPointError("ridge fit produced non-finite parameters")
    if not np.all(coefficients[~active] == 0.0):
        raise RuntimeError("inactive ridge coordinates must stay exact zero")
    return RidgeModel(
        mean=mean,
        scale=scale,
        active=active,
        coefficients=coefficients,
        intercept=intercept,
        alpha=float(alpha),
        train_examples=matrix.shape[0],
        train_features_sha256=_array_sha256(matrix),
        train_targets_sha256=_array_sha256(labels),
    )


def _metric(
    predictions: Sequence[float] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    predicted = np.asarray(predictions, dtype=np.float64)
    expected = np.asarray(targets, dtype=np.float64)
    if predicted.shape != expected.shape or predicted.ndim != 1 or not predicted.size:
        raise ValueError("metric predictions and targets must be nonempty paired vectors")
    if not np.all(np.isin(predicted, (-1.0, 1.0))) or not np.all(
        np.isin(expected, (-1.0, 1.0))
    ):
        raise ValueError("accuracy inputs must be bipolar")
    correct = int(np.count_nonzero(predicted == expected))
    return {
        "examples": int(expected.size),
        "correct": correct,
        "accuracy": float(correct / expected.size),
        "predictions_sha256": _array_sha256(predicted),
        "targets_sha256": _array_sha256(expected),
    }


def _head_summary(head: OnlineTanhHead) -> dict[str, Any]:
    theta = head.theta
    return {
        "feature_dimension": FEATURE_DIMENSION,
        "parameter_count": FEATURE_DIMENSION,
        "bias_parameter_count": 0,
        "initialization": "exact_positive_zero",
        "theta": theta.tolist(),
        "theta_sha256": _array_sha256(theta),
        "maximum_absolute_theta": float(np.max(np.abs(theta))),
        "all_parameters_finite": bool(np.all(np.isfinite(theta))),
        "all_parameters_within_clip": bool(np.all(np.abs(theta) <= THETA_CLIP)),
        "pending_prediction": head.pending,
        "ledger": head.ledger,
    }


def _rng_identity_seeds(stream: Experiment000Stream) -> tuple[int, int, int]:
    substreams = stream.substreams
    return (
        substreams.cue_order_seed,
        substreams.noise_seed,
        substreams.identity_seed,
    )


def _rng_identity_namespaces(stream: Experiment000Stream) -> tuple[str, str, str]:
    base = stream.substreams.namespace
    return (
        f"{base}/cue-order",
        f"{base}/noise",
        f"{base}/identity",
    )


def _run_cell(
    *,
    seed: int,
    cell: CellSpec,
    trace_retention: float,
    ridge_alpha: float,
) -> dict[str, Any]:
    """Run one cell in one pass, sharing observations across paired conditions."""

    cell.validate(official=cell in FULL_CELLS)
    if cell.train_examples != cell.eval_examples:
        raise ValueError("cell stream sizes must match")
    cell_data_seed, cell_data_namespace = derive_cell_data_seed(seed, cell.name)
    stream_pair = generate_stream_pair(
        cell_data_seed,
        episode_count=cell.train_examples,
        noise_events=cell.distractor_count,
        required_role=SeedRole.CUSTOM,
    )
    native_graph = build_experiment_000_graph(
        seed,
        mode="full",
        event_log_enabled=False,
        **FROZEN_GRAPH_OPTIONS,
    )
    trace_graph = clone_with_readout_trace(
        native_graph,
        trace_retention,
        event_log_enabled=False,
    )
    native_dummy_trace = NativeDummyTraceState(trace_retention)
    feature_edge_ids = _output_feature_edge_ids(native_graph)
    if len(feature_edge_ids) != FEATURE_DIMENSION or feature_edge_ids != tuple(
        sorted(feature_edge_ids)
    ):
        raise RuntimeError("frozen graph does not expose 16 ordered output features")

    native_topology_before = native_graph.topology_hash()
    trace_topology_before = trace_graph.topology_hash()
    native_weights_before = native_graph.weights_hash()
    trace_weights_before = trace_graph.weights_hash()
    native_structural_before = _structural_mask_sha256(native_graph)
    trace_structural_before = _structural_mask_sha256(trace_graph)

    trace_head = OnlineTanhHead()
    native_head = OnlineTanhHead()
    independent_head = OnlineTanhHead()
    frozen_head = OnlineTanhHead()

    independent_labels, independent_manifest = balanced_independent_labels(
        master_seed=seed,
        cell_name=cell.name,
        count=cell.train_examples,
    )
    train_random, train_random_manifest = deterministic_random_predictions(
        master_seed=seed,
        cell_name=cell.name,
        split="train",
        count=cell.train_examples,
    )
    eval_random, eval_random_manifest = deterministic_random_predictions(
        master_seed=seed,
        cell_name=cell.name,
        split="eval",
        count=cell.eval_examples,
    )

    train_features_trace: list[np.ndarray] = []
    train_features_native: list[np.ndarray] = []
    train_targets: list[float] = []
    train_predictions: dict[str, list[float]] = {
        condition: []
        for condition in ORDERED_CONDITIONS
        if not condition.startswith("ridge_")
    }
    train_event_hashes: list[str] = []
    train_trace_snapshot_hashes: list[str] = []
    train_order_rows: list[tuple[str, str]] = []
    train_ids_seen: list[str] = []
    train_example_evidence: list[dict[str, Any]] = []

    for index, episode in enumerate(stream_pair.train.episodes):
        observation = observe_episode(
            native_graph,
            trace_graph,
            episode,
            cue_removed=cell.cue_removed,
            feature_edge_ids=feature_edge_ids,
        )
        native_dummy_trace.mirror_episode(
            trace_graph,
            observation,
            feature_edge_ids=feature_edge_ids,
        )
        train_order_rows.extend(
            (
                (episode.episode_id, "features"),
                (episode.episode_id, "prediction"),
            )
        )
        trace_pending = trace_head.predict_for_training(observation.trace_features)
        native_pending = native_head.predict_for_training(observation.native_features)
        independent_pending = independent_head.predict_for_training(
            observation.trace_features
        )
        frozen_activation, frozen_prediction = frozen_head.predict_for_evaluation(
            observation.trace_features
        )
        random_prediction = float(train_random[index])

        # This is the first target access in the episode. Every online and
        # baseline prediction above is therefore pre-reveal by construction.
        target = _bipolar_target(episode.target)
        train_order_rows.append((episode.episode_id, "target_reveal"))
        trace_update = trace_head.reveal_target_and_update(target)
        native_update = native_head.reveal_target_and_update(target)
        independent_update = independent_head.reveal_target_and_update(
            independent_labels[index]
        )
        train_order_rows.append((episode.episode_id, "update"))

        train_features_trace.append(observation.trace_features)
        train_features_native.append(observation.native_features)
        train_targets.append(target)
        train_predictions["trace_head_v1"].append(trace_pending.prediction)
        train_predictions["native_head"].append(native_pending.prediction)
        train_predictions["trace_independent_label"].append(
            independent_pending.prediction
        )
        train_predictions["frozen_trace_head"].append(frozen_prediction)
        train_predictions["rand"].append(random_prediction)
        train_event_hashes.append(observation.event_payload_sha256)
        train_trace_snapshot_hashes.append(_array_sha256(observation.trace_snapshot))
        train_ids_seen.append(episode.episode_id)
        train_example_evidence.append(
            {
                "index": index,
                "episode_id": episode.episode_id,
                "noise_stream_id": episode.noise_stream_id,
                "query_tick": observation.query_tick,
                "target": target,
                "trace_features_sha256": _array_sha256(observation.trace_features),
                "native_features_sha256": _array_sha256(observation.native_features),
                "trace_snapshot_sha256": _array_sha256(observation.trace_snapshot),
                "event_payload_sha256": observation.event_payload_sha256,
                "predictions": {},
                "activations": {
                    "trace_head_v1": trace_pending.activation,
                    "native_head": native_pending.activation,
                    "trace_independent_label": independent_pending.activation,
                    "frozen_trace_head": frozen_activation,
                },
                "update_targets": {
                    "trace_head_v1": target,
                    "native_head": target,
                    "trace_independent_label": float(independent_labels[index]),
                },
                "theta_before_sha256": {
                    "trace_head_v1": trace_pending.theta_before_sha256,
                    "native_head": native_pending.theta_before_sha256,
                    "trace_independent_label": independent_pending.theta_before_sha256,
                },
                "theta_after_sha256": {
                    "trace_head_v1": trace_update["theta_after_sha256"],
                    "native_head": native_update["theta_after_sha256"],
                    "trace_independent_label": independent_update[
                        "theta_after_sha256"
                    ],
                },
            }
        )

    trace_train_matrix = np.stack(train_features_trace)
    native_train_matrix = np.stack(train_features_native)
    train_target_vector = np.asarray(train_targets, dtype=np.float64)
    ridge_trace = fit_train_only_ridge(
        trace_train_matrix, train_target_vector, alpha=ridge_alpha
    )
    ridge_native = fit_train_only_ridge(
        native_train_matrix, train_target_vector, alpha=ridge_alpha
    )
    train_predictions["ridge_trace"] = ridge_trace.predictions(
        trace_train_matrix
    ).tolist()
    train_predictions["ridge_native"] = ridge_native.predictions(
        native_train_matrix
    ).tolist()
    for index, evidence in enumerate(train_example_evidence):
        evidence["predictions"] = {
            condition: float(train_predictions[condition][index])
            for condition in ORDERED_CONDITIONS
        }
        evidence["evidence_sha256"] = _record_self_hash(
            evidence, "evidence_sha256"
        )

    theta_before_eval = {
        "trace_head_v1": _array_sha256(trace_head.theta),
        "native_head": _array_sha256(native_head.theta),
        "trace_independent_label": _array_sha256(independent_head.theta),
        "frozen_trace_head": _array_sha256(frozen_head.theta),
    }
    update_counts_before_eval = {
        "trace_head_v1": int(trace_head.ledger["update_count"]),
        "native_head": int(native_head.ledger["update_count"]),
        "trace_independent_label": int(independent_head.ledger["update_count"]),
        "frozen_trace_head": int(frozen_head.ledger["update_count"]),
    }

    eval_features_trace: list[np.ndarray] = []
    eval_features_native: list[np.ndarray] = []
    eval_targets: list[float] = []
    eval_predictions: dict[str, list[float]] = {
        condition: [] for condition in ORDERED_CONDITIONS
    }
    eval_event_hashes: list[str] = []
    eval_trace_snapshot_hashes: list[str] = []
    eval_order_rows: list[tuple[str, str]] = []
    eval_ids_seen: list[str] = []
    eval_example_evidence: list[dict[str, Any]] = []

    for index, episode in enumerate(stream_pair.eval.episodes):
        observation = observe_episode(
            native_graph,
            trace_graph,
            episode,
            cue_removed=cell.cue_removed,
            feature_edge_ids=feature_edge_ids,
        )
        native_dummy_trace.mirror_episode(
            trace_graph,
            observation,
            feature_edge_ids=feature_edge_ids,
        )
        eval_order_rows.extend(
            (
                (episode.episode_id, "features"),
                (episode.episode_id, "prediction"),
            )
        )
        trace_activation, trace_prediction = trace_head.predict_for_evaluation(
            observation.trace_features
        )
        native_activation, native_prediction = native_head.predict_for_evaluation(
            observation.native_features
        )
        independent_activation, independent_prediction = independent_head.predict_for_evaluation(
            observation.trace_features
        )
        frozen_activation, frozen_prediction = frozen_head.predict_for_evaluation(
            observation.trace_features
        )
        ridge_trace_prediction = float(
            ridge_trace.predictions(observation.trace_features[None, :])[0]
        )
        ridge_native_prediction = float(
            ridge_native.predictions(observation.native_features[None, :])[0]
        )
        random_prediction = float(eval_random[index])

        # Evaluation target access follows every condition's prediction.
        target = _bipolar_target(episode.target)
        eval_order_rows.append((episode.episode_id, "target_reveal"))
        eval_features_trace.append(observation.trace_features)
        eval_features_native.append(observation.native_features)
        eval_targets.append(target)
        eval_predictions["trace_head_v1"].append(trace_prediction)
        eval_predictions["native_head"].append(native_prediction)
        eval_predictions["trace_independent_label"].append(
            independent_prediction
        )
        eval_predictions["frozen_trace_head"].append(frozen_prediction)
        eval_predictions["rand"].append(random_prediction)
        eval_predictions["ridge_trace"].append(ridge_trace_prediction)
        eval_predictions["ridge_native"].append(ridge_native_prediction)
        eval_event_hashes.append(observation.event_payload_sha256)
        eval_trace_snapshot_hashes.append(_array_sha256(observation.trace_snapshot))
        eval_ids_seen.append(episode.episode_id)
        evidence: dict[str, Any] = {
            "index": index,
            "episode_id": episode.episode_id,
            "noise_stream_id": episode.noise_stream_id,
            "query_tick": observation.query_tick,
            "target": target,
            "trace_features_sha256": _array_sha256(observation.trace_features),
            "native_features_sha256": _array_sha256(observation.native_features),
            "trace_snapshot_sha256": _array_sha256(observation.trace_snapshot),
            "event_payload_sha256": observation.event_payload_sha256,
            "predictions": {
                condition: float(eval_predictions[condition][index])
                for condition in ORDERED_CONDITIONS
            },
            "activations": {
                "trace_head_v1": trace_activation,
                "native_head": native_activation,
                "trace_independent_label": independent_activation,
                "frozen_trace_head": frozen_activation,
            },
            "theta_sha256": {
                "trace_head_v1": _array_sha256(trace_head.theta),
                "native_head": _array_sha256(native_head.theta),
                "trace_independent_label": _array_sha256(independent_head.theta),
            },
        }
        evidence["evidence_sha256"] = _record_self_hash(evidence, "evidence_sha256")
        eval_example_evidence.append(evidence)

    eval_target_vector = np.asarray(eval_targets, dtype=np.float64)
    theta_after_eval = {
        "trace_head_v1": _array_sha256(trace_head.theta),
        "native_head": _array_sha256(native_head.theta),
        "trace_independent_label": _array_sha256(independent_head.theta),
        "frozen_trace_head": _array_sha256(frozen_head.theta),
    }
    update_counts_after_eval = {
        "trace_head_v1": int(trace_head.ledger["update_count"]),
        "native_head": int(native_head.ledger["update_count"]),
        "trace_independent_label": int(independent_head.ledger["update_count"]),
        "frozen_trace_head": int(frozen_head.ledger["update_count"]),
    }

    condition_results: dict[str, dict[str, Any]] = {}
    for condition in ORDERED_CONDITIONS:
        condition_results[condition] = {
            "train": _metric(train_predictions[condition], train_target_vector),
            "eval": _metric(eval_predictions[condition], eval_target_vector),
        }
    condition_results["trace_head_v1"]["head"] = _head_summary(trace_head)
    condition_results["native_head"]["head"] = _head_summary(native_head)
    condition_results["trace_independent_label"]["head"] = _head_summary(
        independent_head
    )
    condition_results["trace_independent_label"][
        "train_update_labels_sha256"
    ] = independent_manifest["labels_sha256"]
    condition_results["frozen_trace_head"]["head"] = _head_summary(frozen_head)
    condition_results["rand"]["manifest"] = {
        "train": train_random_manifest,
        "eval": eval_random_manifest,
        "fitted_parameter_count": 0,
    }
    condition_results["ridge_trace"]["diagnostic"] = ridge_trace.diagnostic()
    condition_results["ridge_native"]["diagnostic"] = ridge_native.diagnostic()

    total_examples = cell.train_examples + cell.eval_examples
    per_example_evidence: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "condition_order": list(ORDERED_CONDITIONS),
        "train": train_example_evidence,
        "eval": eval_example_evidence,
    }
    per_example_evidence["evidence_sha256"] = _record_self_hash(
        per_example_evidence, "evidence_sha256"
    )
    train_ids = set(train_ids_seen)
    eval_ids = set(eval_ids_seen)
    data_rng_seeds = (
        *_rng_identity_seeds(stream_pair.train),
        *_rng_identity_seeds(stream_pair.eval),
    )
    baseline_rng_seeds = (
        int(independent_manifest["seed"]),
        int(train_random_manifest["seed"]),
        int(eval_random_manifest["seed"]),
    )
    rng_namespaces = (
        *_rng_identity_namespaces(stream_pair.train),
        *_rng_identity_namespaces(stream_pair.eval),
        str(independent_manifest["namespace"]),
        str(train_random_manifest["namespace"]),
        str(eval_random_manifest["namespace"]),
    )
    heads = (trace_head, native_head, independent_head, frozen_head)
    integrity = {
        "train_eval_episode_ids_disjoint": train_ids.isdisjoint(eval_ids),
        "train_episode_ids_unique": len(train_ids) == cell.train_examples,
        "eval_episode_ids_unique": len(eval_ids) == cell.eval_examples,
        "train_eval_noise_stream_ids_disjoint": {
            episode.noise_stream_id for episode in stream_pair.train.episodes
        }.isdisjoint(
            episode.noise_stream_id for episode in stream_pair.eval.episodes
        ),
        "data_rng_seeds_disjoint": len(set(data_rng_seeds)) == len(data_rng_seeds),
        "baseline_rng_seeds_disjoint": len(set(baseline_rng_seeds))
        == len(baseline_rng_seeds),
        "data_and_baseline_rng_seeds_disjoint": set(data_rng_seeds).isdisjoint(
            baseline_rng_seeds
        ),
        "rng_namespaces_disjoint": len(rng_namespaces)
        == len(set(rng_namespaces)),
        "one_train_pass": len(train_ids_seen) == cell.train_examples,
        "one_eval_pass": len(eval_ids_seen) == cell.eval_examples,
        "no_replay": len(train_ids_seen + eval_ids_seen)
        == len(set(train_ids_seen + eval_ids_seen)),
        "prediction_precedes_target_reveal": all(
            train_order_rows[index : index + 4][1][1] == "prediction"
            and train_order_rows[index : index + 4][2][1] == "target_reveal"
            and train_order_rows[index : index + 4][3][1] == "update"
            for index in range(0, len(train_order_rows), 4)
        )
        and all(
            eval_order_rows[index : index + 3][1][1] == "prediction"
            and eval_order_rows[index : index + 3][2][1] == "target_reveal"
            for index in range(0, len(eval_order_rows), 3)
        ),
        "no_eval_parameter_writes": theta_before_eval == theta_after_eval
        and update_counts_before_eval == update_counts_after_eval,
        "native_topology_unchanged": native_graph.topology_hash()
        == native_topology_before,
        "trace_topology_unchanged": trace_graph.topology_hash()
        == trace_topology_before,
        "topologies_match": native_topology_before == trace_topology_before,
        "native_structural_mask_unchanged": _structural_mask_sha256(native_graph)
        == native_structural_before,
        "trace_structural_mask_unchanged": _structural_mask_sha256(trace_graph)
        == trace_structural_before,
        "structural_masks_match": native_structural_before
        == trace_structural_before,
        "native_weights_unchanged": native_graph.weights_hash()
        == native_weights_before,
        "trace_weights_unchanged": trace_graph.weights_hash()
        == trace_weights_before,
        "weights_match": native_weights_before == trace_weights_before,
        "candidate_native_events_exact": len(train_event_hashes)
        + len(eval_event_hashes)
        == total_examples,
        "native_trace_ledgers_reconcile": native_graph.ledger == trace_graph.ledger,
        "no_recurrent_weight_writes": native_graph.ledger["weight_write_touches"] == 0
        and trace_graph.ledger["weight_write_touches"] == 0,
        "trace_bound_ok": trace_graph.trace_bound_ok,
        "trace_resets_valid": trace_graph.trace_resets_valid,
        "trace_reset_count_exact": trace_graph.trace_reset_count == total_examples,
        "trace_reset_touches_exact": trace_graph.trace_ledger[
            "local_trace_reset_touches"
        ]
        == total_examples * HIDDEN_DIMENSION,
        "trace_observation_touches_exact": trace_graph.trace_ledger[
            "local_trace_observation_touches"
        ]
        == total_examples * HIDDEN_DIMENSION,
        "trace_read_decay_write_touches_reconcile": trace_graph.trace_ledger[
            "local_trace_read_touches"
        ]
        == trace_graph.trace_ledger["local_trace_decay_touches"]
        == trace_graph.trace_ledger["local_trace_write_touches"],
        "head_update_counts_exact": int(trace_head.ledger["update_count"])
        == cell.train_examples
        and int(native_head.ledger["update_count"]) == cell.train_examples
        and int(independent_head.ledger["update_count"]) == cell.train_examples
        and int(frozen_head.ledger["update_count"]) == 0,
        "head_target_reveal_counts_exact": int(trace_head.ledger["target_reveal_count"])
        == cell.train_examples
        and int(native_head.ledger["target_reveal_count"]) == cell.train_examples
        and int(independent_head.ledger["target_reveal_count"])
        == cell.train_examples
        and int(frozen_head.ledger["target_reveal_count"]) == 0,
        "native_trace_compute_budget_matched": native_graph.ledger
        == trace_graph.ledger
        and native_dummy_trace.ledger == trace_graph.trace_ledger
        and native_dummy_trace.state_exact
        and native_dummy_trace.history_exact
        and native_dummy_trace.effective_snapshot_exact
        and native_dummy_trace.weighted_outputs_exact
        and native_dummy_trace.resets_valid
        and native_dummy_trace.bound_ok
        and all(
            trace_head.ledger[field] == native_head.ledger[field]
            for field in (
                "prediction_count",
                "target_reveal_count",
                "update_count",
                "parameter_read_touches",
                "parameter_write_touches",
            )
        ),
        "native_dummy_trace_work_exact": native_dummy_trace.ledger
        == trace_graph.trace_ledger,
        "native_dummy_trace_state_exact": native_dummy_trace.state_exact
        and native_dummy_trace.state_sha256
        == native_dummy_trace.candidate_state_sha256,
        "native_dummy_trace_history_exact": native_dummy_trace.history_exact
        and native_dummy_trace.history_sha256
        == native_dummy_trace.candidate_history_sha256,
        "native_dummy_trace_effective_snapshot_exact": (
            native_dummy_trace.effective_snapshot_exact
            and native_dummy_trace.effective_snapshot_sha256
            == native_dummy_trace.candidate_effective_snapshot_sha256
        ),
        "native_dummy_trace_weighted_output_exact": (
            native_dummy_trace.weighted_outputs_exact
            and native_dummy_trace.weighted_outputs_sha256
            == native_dummy_trace.candidate_weighted_outputs_sha256
        ),
        "native_dummy_trace_diagnostics_exact": native_dummy_trace.bound_ok
        and native_dummy_trace.resets_valid
        and native_dummy_trace.reset_count == trace_graph.trace_reset_count,
        "head_parameter_and_feature_dimensions_exact": all(
            head.theta.shape == (FEATURE_DIMENSION,) for head in heads
        )
        and len(feature_edge_ids) == FEATURE_DIMENSION,
        "head_values_finite_and_bounded": all(
            np.all(np.isfinite(head.theta))
            and np.all(np.abs(head.theta) <= THETA_CLIP)
            and not head.pending
            for head in heads
        ),
        "ridge_train_only": ridge_trace.train_examples == cell.train_examples
        and ridge_native.train_examples == cell.train_examples,
        "ridge_evaluation_fit_count_zero": ridge_trace.diagnostic()[
            "evaluation_fit_count"
        ]
        == 0
        and ridge_native.diagnostic()["evaluation_fit_count"] == 0,
        "all_values_finite": all(
            math.isfinite(condition_results[condition][split]["accuracy"])
            for condition in ORDERED_CONDITIONS
            for split in ("train", "eval")
        )
        and native_graph.ledger["nonfinite_values"] == 0
        and trace_graph.ledger["nonfinite_values"] == 0,
    }
    if set(integrity) != _CELL_INTEGRITY_KEYS:
        raise RuntimeError("cell integrity schema changed")
    if not all(integrity.values()):
        failed = sorted(name for name, passed in integrity.items() if not passed)
        raise RuntimeError(f"cell integrity failed: {failed}")

    return {
        "cell": cell.name,
        "distractor_count": cell.distractor_count,
        "cue_removed": cell.cue_removed,
        "train_examples": cell.train_examples,
        "eval_examples": cell.eval_examples,
        "stream_manifest": stream_pair.manifest.to_dict(),
        "cell_data_root": {
            "registered_master_seed": seed,
            "namespace": cell_data_namespace,
            "derived_seed": cell_data_seed,
        },
        "identity_evidence": {
            "train_episode_ids": [
                episode.episode_id for episode in stream_pair.train.episodes
            ],
            "eval_episode_ids": [
                episode.episode_id for episode in stream_pair.eval.episodes
            ],
            "train_noise_stream_ids": [
                episode.noise_stream_id for episode in stream_pair.train.episodes
            ],
            "eval_noise_stream_ids": [
                episode.noise_stream_id for episode in stream_pair.eval.episodes
            ],
            "data_rng_seeds": list(data_rng_seeds),
            "baseline_rng_seeds": list(baseline_rng_seeds),
            "rng_namespaces": list(rng_namespaces),
        },
        "feature_edge_ids": list(feature_edge_ids),
        "feature_edge_ids_sha256": _sha256_json(list(feature_edge_ids)),
        "trace_retention": trace_retention,
        "condition_results": condition_results,
        "independent_label_manifest": independent_manifest,
        "per_example_evidence": per_example_evidence,
        "feature_evidence": {
            "train_trace_features_sha256": _array_sha256(trace_train_matrix),
            "train_native_features_sha256": _array_sha256(native_train_matrix),
            "eval_trace_features_sha256": _array_sha256(
                np.stack(eval_features_trace)
            ),
            "eval_native_features_sha256": _array_sha256(
                np.stack(eval_features_native)
            ),
            "train_trace_snapshots_sha256": _sha256_json(
                train_trace_snapshot_hashes
            ),
            "eval_trace_snapshots_sha256": _sha256_json(
                eval_trace_snapshot_hashes
            ),
            "train_candidate_native_events_sha256": _sha256_json(
                train_event_hashes
            ),
            "eval_candidate_native_events_sha256": _sha256_json(eval_event_hashes),
        },
        "temporal_order_sha256": _sha256_json(
            {"train": train_order_rows, "eval": eval_order_rows}
        ),
        "graph": {
            "topology_sha256": native_topology_before,
            "structural_mask_sha256": native_structural_before,
            "weights_sha256": native_weights_before,
            "native_ledger": native_graph.ledger,
            "trace_native_ledger": trace_graph.ledger,
            "trace_ledger": trace_graph.trace_ledger,
            "native_dummy_trace_ledger": native_dummy_trace.ledger,
            "native_dummy_trace_state_sha256": native_dummy_trace.state_sha256,
            "candidate_trace_state_sha256": (
                native_dummy_trace.candidate_state_sha256
            ),
            "native_dummy_trace_history_sha256": (
                native_dummy_trace.history_sha256
            ),
            "candidate_trace_history_sha256": (
                native_dummy_trace.candidate_history_sha256
            ),
            "native_dummy_trace_effective_snapshot_sha256": (
                native_dummy_trace.effective_snapshot_sha256
            ),
            "candidate_trace_effective_snapshot_sha256": (
                native_dummy_trace.candidate_effective_snapshot_sha256
            ),
            "native_dummy_trace_weighted_outputs_sha256": (
                native_dummy_trace.weighted_outputs_sha256
            ),
            "candidate_trace_weighted_outputs_sha256": (
                native_dummy_trace.candidate_weighted_outputs_sha256
            ),
            "native_dummy_trace_reset_count": native_dummy_trace.reset_count,
            "native_dummy_trace_resets_valid": native_dummy_trace.resets_valid,
            "native_dummy_trace_bound_ok": native_dummy_trace.bound_ok,
        },
        "integrity": integrity,
    }


def _run_seed(
    *,
    seed: int,
    cells: Sequence[CellSpec],
    trace_retention: float,
    ridge_alpha: float,
) -> dict[str, Any]:
    if isinstance(seed, bool) or type(seed) is not int:
        raise TypeError("seed must be an exact built-in integer")
    require_seed_role(seed, SeedRole.CUSTOM)
    resolved_cells = tuple(cells)
    if not resolved_cells or len({cell.name for cell in resolved_cells}) != len(
        resolved_cells
    ):
        raise ValueError("cells must be nonempty and uniquely named")
    results = [
        _run_cell(
            seed=seed,
            cell=cell,
            trace_retention=trace_retention,
            ridge_alpha=ridge_alpha,
        )
        for cell in resolved_cells
    ]
    all_episode_ids = [
        episode_id
        for result in results
        for split in ("train_episode_ids", "eval_episode_ids")
        for episode_id in result["identity_evidence"][split]
    ]
    all_noise_stream_ids = [
        stream_id
        for result in results
        for split in ("train_noise_stream_ids", "eval_noise_stream_ids")
        for stream_id in result["identity_evidence"][split]
    ]
    all_rng_seeds = [
        int(rng_seed)
        for result in results
        for group in ("data_rng_seeds", "baseline_rng_seeds")
        for rng_seed in result["identity_evidence"][group]
    ]
    all_rng_namespaces = [
        namespace
        for result in results
        for namespace in result["identity_evidence"]["rng_namespaces"]
    ]
    seed_integrity = {
        "cross_cell_episode_ids_disjoint": len(all_episode_ids)
        == len(set(all_episode_ids)),
        "cross_cell_noise_stream_ids_disjoint": len(all_noise_stream_ids)
        == len(set(all_noise_stream_ids)),
        "cross_cell_rng_seeds_disjoint": len(all_rng_seeds) == len(set(all_rng_seeds)),
        "cross_cell_rng_namespaces_disjoint": len(all_rng_namespaces)
        == len(set(all_rng_namespaces)),
        "cell_data_namespaces_disjoint": len(
            {result["cell_data_root"]["namespace"] for result in results}
        )
        == len(results),
        "cell_data_roots_disjoint": len(
            {result["cell_data_root"]["derived_seed"] for result in results}
        )
        == len(results),
    }
    if not all(seed_integrity.values()):
        raise RuntimeError("cross-cell identities or RNG namespaces collided")
    return {
        "seed": seed,
        "seed_role": SeedRole.CUSTOM.value,
        "cell_results": results,
        "integrity": seed_integrity,
        "all_cell_integrity_pass": all(
            all(cell_result["integrity"].values()) for cell_result in results
        )
        and all(seed_integrity.values()),
    }


def _execution_binding(payload: Mapping[str, Any]) -> str:
    return _sha256_json(dict(payload))


def _run_seed_job(
    job: tuple[int, tuple[CellSpec, ...], float, float, Mapping[str, Any]]
) -> dict[str, Any]:
    seed, cells, trace_retention, ridge_alpha, coordinator = job
    started = perf_counter()
    job_started_utc = datetime.now(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )
    with threadpool_limits(limits=1, user_api="blas"):
        blas = _single_thread_blas_state()
        deterministic = _run_seed(
            seed=seed,
            cells=cells,
            trace_retention=trace_retention,
            ridge_alpha=ridge_alpha,
        )
    worker_identity = {
        "seed": seed,
        "process_id": os.getpid(),
        "parent_process_id": os.getppid(),
        "process_instance_token": _PROCESS_INSTANCE_TOKEN,
        "process_started_utc": _PROCESS_STARTED_UTC,
        "job_started_utc": job_started_utc,
        "coordinator_binding_sha256": coordinator["binding_sha256"],
    }
    return {
        "deterministic": deterministic,
        "runtime": {
            **worker_identity,
            "worker_binding_sha256": _execution_binding(worker_identity),
            "runtime_seconds": perf_counter() - started,
            "blas": blas,
        },
    }


def _ordered_seed_results(
    *,
    seeds: Sequence[int],
    cells: Sequence[CellSpec],
    trace_retention: float,
    ridge_alpha: float,
    workers: int,
    coordinator: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run seed jobs in a five-worker spawn pool and restore registered order."""

    if workers != ORDERED_CPU_WORKERS:
        raise ValueError("official ordered execution requires exactly five workers")
    if coordinator is None:
        coordinator_payload: dict[str, Any] = {
            "process_id": os.getpid(),
            "process_instance_token": _PROCESS_INSTANCE_TOKEN,
            "process_started_utc": _PROCESS_STARTED_UTC,
            "run_instance_token": hashlib.sha256(os.urandom(32)).hexdigest(),
            "run_started_utc": datetime.now(UTC)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z"),
        }
        coordinator_payload["binding_sha256"] = _execution_binding(
            coordinator_payload
        )
    else:
        coordinator_payload = dict(coordinator)
        binding = coordinator_payload.pop("binding_sha256", None)
        if binding != _execution_binding(coordinator_payload):
            raise RuntimeError("coordinator execution binding is invalid")
        coordinator_payload["binding_sha256"] = binding
    jobs = [
        (seed, tuple(cells), trace_retention, ridge_alpha, coordinator_payload)
        for seed in tuple(seeds)
    ]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        rows = list(executor.map(_run_seed_job, jobs, chunksize=1))
    deterministic = [row["deterministic"] for row in rows]
    runtime = [row["runtime"] for row in rows]
    if [row["seed"] for row in deterministic] != list(seeds):
        raise RuntimeError("worker results are not in registered seed order")
    worker_process_ids = {int(row["process_id"]) for row in runtime}
    worker_process_tokens = {str(row["process_instance_token"]) for row in runtime}
    if len(worker_process_ids) != ORDERED_CPU_WORKERS:
        raise RuntimeError(
            "official execution did not realize exactly five worker processes"
        )
    if len(worker_process_tokens) != ORDERED_CPU_WORKERS:
        raise RuntimeError(
            "official execution did not realize five distinct worker instances"
        )
    if len(
        {
            (
                row["process_id"],
                row["process_instance_token"],
                row["process_started_utc"],
            )
            for row in runtime
        }
    ) != ORDERED_CPU_WORKERS:
        raise RuntimeError("worker process identity changed between seed jobs")
    if any(
        row["parent_process_id"] != coordinator_payload["process_id"]
        or row["coordinator_binding_sha256"]
        != coordinator_payload["binding_sha256"]
        or row["worker_binding_sha256"]
        != _execution_binding(
            {
                key: row[key]
                for key in (
                    "seed",
                    "process_id",
                    "parent_process_id",
                    "process_instance_token",
                    "process_started_utc",
                    "job_started_utc",
                    "coordinator_binding_sha256",
                )
            }
        )
        for row in runtime
    ):
        raise RuntimeError("worker execution bindings do not reconcile")
    if any(
        not row["blas"] or any(pool["num_threads"] != 1 for pool in row["blas"])
        for row in runtime
    ):
        raise RuntimeError("a worker used a multithreaded numerical backend")
    return deterministic, runtime


def exact_sign_flip_test(differences: Sequence[float]) -> dict[str, Any]:
    """Exact one-sided paired sign-flip test over every ``2**n`` assignment."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("sign-flip differences must be a finite nonempty vector")
    if values.size > 20:
        raise ValueError("exact sign-flip enumeration is capped at 20 pairs")
    observed = float(np.mean(values))
    assignments = 1 << int(values.size)
    extreme = 0
    for mask in range(assignments):
        signed_sum = math.fsum(
            float(value) if mask & (1 << index) else -float(value)
            for index, value in enumerate(values)
        )
        if signed_sum / values.size >= observed:
            extreme += 1
    return {
        "unit": "seed",
        "alternative": "mean_paired_difference_greater_than_zero",
        "pair_count": int(values.size),
        "assignments": assignments,
        "extreme_assignments": extreme,
        "observed_mean_difference": observed,
        "p_value": float(extreme / assignments),
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return monotone Holm family-wise adjusted p-values."""

    if not p_values:
        raise ValueError("Holm adjustment requires at least one p-value")
    parsed = {name: float(value) for name, value in p_values.items()}
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in parsed.values()):
        raise ValueError("Holm p-values must be finite values in [0, 1]")
    ordered = sorted(parsed.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    running = 0.0
    adjusted: dict[str, float] = {}
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}


def bootstrap_seed_mean_ci(
    differences: Sequence[float],
    *,
    namespace: str,
    replicates: int = BOOTSTRAP_REPLICATES,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> dict[str, Any]:
    """Fixed-PCG64 percentile bootstrap over the seed-level paired differences."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap differences must be a finite nonempty vector")
    if isinstance(replicates, bool) or type(replicates) is not int or replicates <= 0:
        raise ValueError("bootstrap replicates must be a positive exact integer")
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must lie strictly between zero and one")
    seed = _derive_seed(0, f"seed-bootstrap/{namespace}/{replicates}")
    generator = np.random.Generator(np.random.PCG64(seed))
    indices = generator.integers(
        0,
        values.size,
        size=(replicates, values.size),
        dtype=np.int64,
    )
    means = np.mean(values[indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, (tail, 1.0 - tail), method="linear")
    return {
        "unit": "seed",
        "bit_generator": "numpy.random.PCG64",
        "namespace": namespace,
        "seed": seed,
        "replicates": replicates,
        "confidence": confidence,
        "observed_mean_difference": float(np.mean(values)),
        "lower": float(lower),
        "upper": float(upper),
        "bootstrap_means_sha256": _array_sha256(means),
    }


def _cell_lookup(seed_result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = seed_result.get("cell_results")
    if not isinstance(rows, list):
        raise TypeError("seed result cell rows are absent")
    resolved = {str(row["cell"]): row for row in rows}
    if len(resolved) != len(rows):
        raise RuntimeError("seed result has duplicate cell rows")
    return resolved


def aggregate_seed_results(
    seed_results: Sequence[Mapping[str, Any]],
    *,
    cell_names: Sequence[str],
) -> dict[str, Any]:
    """Aggregate only seed-level accuracies; samples never become statistical units."""

    results = tuple(seed_results)
    if not results:
        raise ValueError("at least one seed result is required")
    if len({row.get("seed") for row in results}) != len(results):
        raise ValueError("seed results must have unique seed identifiers")
    cells: dict[str, Any] = {}
    for cell_name in cell_names:
        condition_rows: dict[str, Any] = {}
        for condition in ORDERED_CONDITIONS:
            values = [
                float(
                    _cell_lookup(seed_result)[cell_name]["condition_results"][
                        condition
                    ]["eval"]["accuracy"]
                )
                for seed_result in results
            ]
            if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
                raise FloatingPointError("aggregate accuracy is invalid")
            condition_rows[condition] = {
                "per_seed_eval_accuracy": values,
                "mean_eval_accuracy": float(np.mean(values)),
                "minimum_eval_accuracy": float(np.min(values)),
                "maximum_eval_accuracy": float(np.max(values)),
            }
        cells[cell_name] = {"conditions": condition_rows}
    return {
        "unit": "seed",
        "seed_count": len(results),
        "seed_order": [int(row["seed"]) for row in results],
        "cells": cells,
    }


def _accuracy_vector(
    aggregate: Mapping[str, Any], cell: str, condition: str
) -> np.ndarray:
    values = aggregate["cells"][cell]["conditions"][condition][
        "per_seed_eval_accuracy"
    ]
    return np.asarray(values, dtype=np.float64)


def compute_registered_statistics(
    aggregate: Mapping[str, Any],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Compute the three predeclared seed-paired D8 contrasts."""

    trace = _accuracy_vector(aggregate, "d8_clean", "trace_head_v1")
    if trace.size != len(REGISTERED_SEEDS):
        raise ValueError("registered inference requires exactly ten seed units")
    tests: dict[str, dict[str, Any]] = {}
    raw: dict[str, float] = {}
    for name, baseline, _ in _CONTRASTS:
        differences = trace - _accuracy_vector(aggregate, "d8_clean", baseline)
        sign_flip = exact_sign_flip_test(differences)
        bootstrap = bootstrap_seed_mean_ci(
            differences,
            namespace=name,
            replicates=bootstrap_replicates,
        )
        tests[name] = {
            "baseline": baseline,
            "paired_differences": differences.tolist(),
            "paired_differences_sha256": _array_sha256(differences),
            "sign_flip": sign_flip,
            "bootstrap": bootstrap,
        }
        raw[name] = float(sign_flip["p_value"])
    adjusted = holm_adjust(raw)
    for name, test in tests.items():
        test["holm_adjusted_p"] = adjusted[name]
    return {
        "unit": "seed",
        "predeclared_contrast_order": [name for name, _, _ in _CONTRASTS],
        "exact_sign_flip_assignments_per_contrast": 1 << len(REGISTERED_SEEDS),
        "holm_family_size": len(_CONTRASTS),
        "bootstrap_replicates_per_contrast": bootstrap_replicates,
        "tests": tests,
    }


def _gate(value: float | bool, relation: str, threshold: Any, passed: bool) -> dict[str, Any]:
    return {
        "value": value,
        "relation": relation,
        "threshold": threshold,
        "pass": bool(passed),
    }


def evaluate_registered_gates(
    aggregate: Mapping[str, Any],
    statistics: Mapping[str, Any] | None,
    *,
    integrity_pass: bool,
    eligible: bool,
) -> dict[str, Any]:
    """Apply the frozen full-run gates; scratch inputs remain nonselecting."""

    if not eligible:
        return {
            "eligible": False,
            "criteria": {},
            "all_pass": False,
            "failed_gate_ids": [],
            "provisional_terminal_status": "SMOKE_NONSELECTING",
        }
    if statistics is None:
        raise ValueError("eligible gate evaluation requires registered statistics")
    d8_trace = _accuracy_vector(aggregate, "d8_clean", "trace_head_v1")
    d8_frozen = _accuracy_vector(aggregate, "d8_clean", "frozen_trace_head")
    d8_native = _accuracy_vector(aggregate, "d8_clean", "native_head")
    d8_independent = _accuracy_vector(
        aggregate, "d8_clean", "trace_independent_label"
    )
    d8_ridge = _accuracy_vector(aggregate, "d8_clean", "ridge_trace")
    d4_trace = _accuracy_vector(aggregate, "d4_clean", "trace_head_v1")
    d16_trace = _accuracy_vector(aggregate, "d16_clean", "trace_head_v1")
    removed = _accuracy_vector(
        aggregate, "d8_cue_removed", "trace_head_v1"
    )
    criteria: dict[str, dict[str, Any]] = {}
    d8_mean = float(np.mean(d8_trace))
    d4_mean = float(np.mean(d4_trace))
    d16_mean = float(np.mean(d16_trace))
    frozen_advantage = float(np.mean(d8_trace - d8_frozen))
    native_advantage = float(np.mean(d8_trace - d8_native))
    independent_advantage = float(np.mean(d8_trace - d8_independent))
    ridge_shortfall = float(np.mean(d8_ridge) - d8_mean)
    d8_seed_count = int(np.count_nonzero(d8_trace >= 0.70))
    d4_seed_count = int(np.count_nonzero(d4_trace >= 0.60))
    d16_seed_count = int(np.count_nonzero(d16_trace >= 0.60))
    d16_drop = d8_mean - d16_mean
    removed_mean = float(np.mean(removed))
    independent_mean = float(np.mean(d8_independent))
    frozen_mean = float(np.mean(d8_frozen))

    criteria["P01_D8_TRACE_MEAN_ACCURACY"] = _gate(
        d8_mean, ">=", 0.75, d8_mean >= 0.75
    )
    criteria["P02_D8_TRACE_SEED_COUNT"] = _gate(
        d8_seed_count, ">=", 8, d8_seed_count >= 8
    )
    criteria["P03_D8_ADVANTAGE_VS_FROZEN"] = _gate(
        frozen_advantage, ">=", 0.20, frozen_advantage >= 0.20
    )
    criteria["P04_D8_ADVANTAGE_VS_NATIVE"] = _gate(
        native_advantage, ">=", 0.10, native_advantage >= 0.10
    )
    criteria["P05_D8_ADVANTAGE_VS_INDEPENDENT_LABEL"] = _gate(
        independent_advantage, ">=", 0.15, independent_advantage >= 0.15
    )
    criteria["P06_D8_ONLINE_WITHIN_TRACE_RIDGE"] = _gate(
        ridge_shortfall, "<=", 0.05, ridge_shortfall <= 0.05
    )
    tests = statistics["tests"]
    bootstrap_gate_by_contrast = {
        "trace_vs_frozen": "B01_TRACE_VS_FROZEN_BOOTSTRAP_LOWER",
        "trace_vs_native": "B02_TRACE_VS_NATIVE_BOOTSTRAP_LOWER",
        "trace_vs_independent_label": "B03_TRACE_VS_INDEPENDENT_BOOTSTRAP_LOWER",
    }
    for contrast, _, p_gate_id in _CONTRASTS:
        adjusted = float(tests[contrast]["holm_adjusted_p"])
        criteria[p_gate_id] = _gate(
            adjusted, "<", 0.05, adjusted < 0.05
        )
    for contrast, _, _ in _CONTRASTS:
        lower = float(tests[contrast]["bootstrap"]["lower"])
        bootstrap_gate_id = bootstrap_gate_by_contrast[contrast]
        criteria[bootstrap_gate_id] = _gate(lower, ">", 0.0, lower > 0.0)
    criteria["R01_D4_TRACE_MEAN_ACCURACY"] = _gate(
        d4_mean, ">=", 0.75, d4_mean >= 0.75
    )
    criteria["R02_D16_TRACE_MEAN_ACCURACY"] = _gate(
        d16_mean, ">=", 0.65, d16_mean >= 0.65
    )
    criteria["R03_D4_TRACE_SEED_COUNT"] = _gate(
        d4_seed_count, ">=", 8, d4_seed_count >= 8
    )
    criteria["R04_D16_TRACE_SEED_COUNT"] = _gate(
        d16_seed_count, ">=", 8, d16_seed_count >= 8
    )
    criteria["R05_D16_DROP_FROM_D8"] = _gate(
        d16_drop, "<=", 0.15, d16_drop <= 0.15
    )
    criteria["N01_D8_CUE_REMOVED_CHANCE"] = _gate(
        removed_mean,
        "in_closed_interval",
        [0.45, 0.55],
        0.45 <= removed_mean <= 0.55,
    )
    criteria["N02_D8_INDEPENDENT_LABEL_CHANCE"] = _gate(
        independent_mean,
        "in_closed_interval",
        [0.45, 0.55],
        0.45 <= independent_mean <= 0.55,
    )
    criteria["N03_D8_FROZEN_HEAD_CHANCE"] = _gate(
        frozen_mean,
        "in_closed_interval",
        [0.45, 0.55],
        0.45 <= frozen_mean <= 0.55,
    )
    criteria["I01_ALL_INTEGRITY_GATES"] = _gate(
        integrity_pass, "is", True, integrity_pass
    )
    if tuple(criteria) != _GATE_ORDER:
        raise RuntimeError("gate order drifted from the frozen protocol")
    failed = [gate_id for gate_id in _GATE_ORDER if not criteria[gate_id]["pass"]]
    all_pass = not failed
    terminal = (
        "LEARNING_RESULT_PASS"
        if all_pass
        else f"LEARNING_RESULT_FAIL:{','.join(failed)}"
    )
    return {
        "eligible": True,
        "criteria": criteria,
        "all_pass": all_pass,
        "failed_gate_ids": failed,
        "provisional_terminal_status": terminal,
    }


def _validate_official_arguments(
    *,
    seeds: Sequence[int],
    cells: Sequence[CellSpec],
    ridge_alpha: float,
    workers: int,
    output_path: str | Path,
) -> tuple[tuple[int, ...], tuple[CellSpec, ...], Path]:
    if any(isinstance(seed, bool) or type(seed) is not int for seed in seeds):
        raise TypeError("official seeds must be exact built-in integers")
    resolved_seeds = tuple(seeds)
    if resolved_seeds != REGISTERED_SEEDS:
        raise ValueError(
            f"official learning seeds must be exactly {REGISTERED_SEEDS} in order"
        )
    for seed in resolved_seeds:
        require_seed_role(seed, SeedRole.CUSTOM)
    resolved_cells = tuple(cells)
    if resolved_cells != FULL_CELLS:
        raise ValueError("official learning cells or sizes differ from the frozen matrix")
    for cell in resolved_cells:
        cell.validate(official=True)
    if type(ridge_alpha) is not float or ridge_alpha != DEFAULT_RIDGE_ALPHA:
        raise ValueError("official ridge alpha must be exact float 0.001")
    if type(workers) is not int or workers != ORDERED_CPU_WORKERS:
        raise ValueError("official execution requires exactly five workers")
    target = _project_path(output_path)
    valid_targets = {
        _project_path(PRIMARY_REPORT_PATH).resolve(),
        _project_path(RERUN_REPORT_PATH).resolve(),
    }
    if target.resolve() not in valid_targets:
        raise ValueError("official output must be the registered primary or rerun path")
    return resolved_seeds, resolved_cells, target


def _validate_scratch_arguments(
    *,
    seeds: Sequence[int],
    cells: Sequence[CellSpec],
    trace_retention: float,
    ridge_alpha: float,
    output_path: str | Path,
) -> tuple[tuple[int, ...], tuple[CellSpec, ...], Path]:
    if any(isinstance(seed, bool) or type(seed) is not int for seed in seeds):
        raise TypeError("scratch seeds must be exact built-in integers")
    resolved_seeds = tuple(seeds)
    if set(resolved_seeds).intersection(REGISTERED_SEEDS):
        raise ValueError("scratch smoke must never consume official seeds 80--89")
    if resolved_seeds != SCRATCH_SMOKE_SEEDS:
        raise ValueError(
            f"scratch smoke seeds must be exactly {SCRATCH_SMOKE_SEEDS} in order"
        )
    for seed in resolved_seeds:
        require_seed_role(seed, SeedRole.CUSTOM)
    resolved_cells = tuple(cells)
    if resolved_cells != SMOKE_CELLS:
        raise ValueError("scratch smoke cells must use the frozen 2/2 matrix")
    for cell in resolved_cells:
        cell.validate(official=False)
    if (
        isinstance(trace_retention, bool)
        or not math.isfinite(float(trace_retention))
        or not 0.0 < float(trace_retention) < 1.0
    ):
        raise ValueError("scratch trace retention must lie strictly between zero and one")
    if isinstance(ridge_alpha, bool) or not math.isfinite(ridge_alpha) or ridge_alpha <= 0:
        raise ValueError("scratch ridge alpha must be finite and positive")
    target = _project_path(output_path)
    canonical = _project_path(ARTIFACT_DIRECTORY).resolve()
    try:
        target.resolve().relative_to(canonical)
    except ValueError:
        pass
    else:
        raise ValueError("scratch smoke cannot write in the official artifact directory")
    if "scratch" not in str(target).lower():
        raise ValueError("scratch smoke output path must be visibly marked scratch")
    return resolved_seeds, resolved_cells, target


def _atomic_write_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    refuse_overwrite: bool = True,
) -> Path:
    target = _project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{_PROCESS_INSTANCE_TOKEN[:16]}.tmp"
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        # Exclusive temporary creation protects two writers in the same
        # process as well as distinct processes.  Hard-link publication is an
        # atomic create-if-absent operation on the target volume; unlike
        # ``Path.replace`` it can never overwrite evidence that won a race.
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if refuse_overwrite:
            try:
                os.link(temporary, target)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"refusing to overwrite evidence artifact: {target}"
                ) from exc
        else:
            os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    if target.read_text(encoding="utf-8") != encoded:
        raise RuntimeError(f"evidence artifact changed during publication: {target}")
    return target


def _write_report_and_sidecar(
    path: str | Path,
    report: Mapping[str, Any],
    *,
    refuse_overwrite: bool = True,
) -> tuple[Path, Path]:
    target = _project_path(path)
    sidecar = target.with_suffix(".sha256")
    if refuse_overwrite and sidecar.exists():
        raise FileExistsError(f"refusing to overwrite evidence sidecar: {sidecar}")
    written = _atomic_write_json(
        target, report, refuse_overwrite=refuse_overwrite
    )
    try:
        _atomic_write_json(
            sidecar,
            {
                "algorithm": "sha256",
                "report_file": target.name,
                "report_sha256": _file_sha256(written),
            },
            refuse_overwrite=refuse_overwrite,
        )
    except Exception:
        if written.exists():
            written.unlink()
        raise
    return written, sidecar


def _read_report_with_sidecar(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    target = _project_path(path)
    sidecar = target.with_suffix(".sha256")
    if not target.is_file() or not sidecar.is_file():
        raise RuntimeError(f"report or SHA-256 sidecar is absent: {target.name}")
    # Hash and parse the same immutable byte snapshots.  Hashing a path and
    # reopening it later would permit a replacement race between validation
    # and parsing; higher-level verifiers additionally recheck the live paths.
    report_bytes = target.read_bytes()
    sidecar_bytes = sidecar.read_bytes()
    report_hash = hashlib.sha256(report_bytes).hexdigest()
    sidecar_payload = json.loads(sidecar_bytes)
    if (
        not isinstance(sidecar_payload, dict)
        or sidecar_payload.get("algorithm") != "sha256"
        or sidecar_payload.get("report_file") != target.name
        or sidecar_payload.get("report_sha256") != report_hash
    ):
        raise RuntimeError(f"report SHA-256 sidecar is invalid: {target.name}")
    report = json.loads(report_bytes)
    if not isinstance(report, dict):
        raise TypeError("persisted learning report must be a JSON object")
    return report, {
        "report_file_sha256": report_hash,
        "sidecar_file_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
    }


def _report_payload_object(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(report))
    payload.pop("nondeterministic_provenance", None)
    payload.pop("deterministic_payload_sha256", None)
    return payload


def _recompute_report_payload_sha256(report: Mapping[str, Any]) -> str:
    return _sha256_json(_report_payload_object(report))


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return False


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _independently_reconstruct_cell(
    reported: Mapping[str, Any],
    *,
    seed: int,
    spec: CellSpec,
    trace_retention: float,
    ridge_alpha: float,
) -> None:
    """Regenerate a cell and replay every prediction/update from its transcript.

    This intentionally does not call ``_run_cell`` or use reported counts as
    inputs.  Registered streams and frozen equations are the sole authority.
    """

    cell_data_seed, cell_data_namespace = derive_cell_data_seed(seed, spec.name)
    stream_pair = generate_stream_pair(
        cell_data_seed,
        episode_count=spec.train_examples,
        noise_events=spec.distractor_count,
        required_role=SeedRole.CUSTOM,
    )
    native_graph = build_experiment_000_graph(
        seed,
        mode="full",
        event_log_enabled=False,
        **FROZEN_GRAPH_OPTIONS,
    )
    trace_graph = clone_with_readout_trace(
        native_graph,
        trace_retention,
        event_log_enabled=False,
    )
    native_dummy_trace = NativeDummyTraceState(trace_retention)
    feature_edge_ids = _output_feature_edge_ids(native_graph)
    topology = native_graph.topology_hash()
    structure = _structural_mask_sha256(native_graph)
    weights = native_graph.weights_hash()

    independent_labels, independent_manifest = balanced_independent_labels(
        master_seed=seed,
        cell_name=spec.name,
        count=spec.train_examples,
    )
    train_random, train_random_manifest = deterministic_random_predictions(
        master_seed=seed,
        cell_name=spec.name,
        split="train",
        count=spec.train_examples,
    )
    eval_random, eval_random_manifest = deterministic_random_predictions(
        master_seed=seed,
        cell_name=spec.name,
        split="eval",
        count=spec.eval_examples,
    )
    heads = {
        "trace_head_v1": _IndependentReplayHead(),
        "native_head": _IndependentReplayHead(),
        "trace_independent_label": _IndependentReplayHead(),
        "frozen_trace_head": _IndependentReplayHead(),
    }
    predictions: dict[str, dict[str, list[float]]] = {
        condition: {"train": [], "eval": []} for condition in ORDERED_CONDITIONS
    }
    train_trace: list[np.ndarray] = []
    train_native: list[np.ndarray] = []
    eval_trace: list[np.ndarray] = []
    eval_native: list[np.ndarray] = []
    train_targets: list[float] = []
    eval_targets: list[float] = []
    train_events: list[str] = []
    eval_events: list[str] = []
    train_snapshots: list[str] = []
    eval_snapshots: list[str] = []
    train_order: list[tuple[str, str]] = []
    eval_order: list[tuple[str, str]] = []
    reconstructed_train_rows: list[dict[str, Any]] = []
    reconstructed_eval_rows: list[dict[str, Any]] = []

    for index, episode in enumerate(stream_pair.train.episodes):
        observation = observe_episode(
            native_graph,
            trace_graph,
            episode,
            cue_removed=spec.cue_removed,
            feature_edge_ids=feature_edge_ids,
        )
        native_dummy_trace.mirror_episode(
            trace_graph,
            observation,
            feature_edge_ids=feature_edge_ids,
        )
        train_order.extend(
            (
                (episode.episode_id, "features"),
                (episode.episode_id, "prediction"),
            )
        )
        trace_activation, trace_prediction, trace_before_sha = heads[
            "trace_head_v1"
        ].predict(observation.trace_features)
        native_activation, native_prediction, native_before_sha = heads[
            "native_head"
        ].predict(observation.native_features)
        independent_activation, independent_prediction, independent_before_sha = heads[
            "trace_independent_label"
        ].predict(observation.trace_features)
        frozen_activation, frozen_prediction, _ = heads[
            "frozen_trace_head"
        ].predict(observation.trace_features)
        target = _bipolar_target(episode.target)
        train_order.append((episode.episode_id, "target_reveal"))
        trace_after_sha = heads["trace_head_v1"].update(
            observation.trace_features, target
        )
        native_after_sha = heads["native_head"].update(
            observation.native_features, target
        )
        independent_after_sha = heads["trace_independent_label"].update(
            observation.trace_features, float(independent_labels[index])
        )
        train_order.append((episode.episode_id, "update"))
        for condition, prediction in (
            ("trace_head_v1", trace_prediction),
            ("native_head", native_prediction),
            ("trace_independent_label", independent_prediction),
            ("frozen_trace_head", frozen_prediction),
            ("rand", float(train_random[index])),
        ):
            predictions[condition]["train"].append(float(prediction))
        train_trace.append(observation.trace_features)
        train_native.append(observation.native_features)
        train_targets.append(target)
        train_events.append(observation.event_payload_sha256)
        snapshot_sha = _array_sha256(observation.trace_snapshot)
        train_snapshots.append(snapshot_sha)
        reconstructed_train_rows.append(
            {
                "index": index,
                "episode_id": episode.episode_id,
                "noise_stream_id": episode.noise_stream_id,
                "query_tick": observation.query_tick,
                "target": target,
                "trace_features_sha256": _array_sha256(
                    observation.trace_features
                ),
                "native_features_sha256": _array_sha256(
                    observation.native_features
                ),
                "trace_snapshot_sha256": snapshot_sha,
                "event_payload_sha256": observation.event_payload_sha256,
                "predictions": {},
                "activations": {
                    "trace_head_v1": trace_activation,
                    "native_head": native_activation,
                    "trace_independent_label": independent_activation,
                    "frozen_trace_head": frozen_activation,
                },
                "update_targets": {
                    "trace_head_v1": target,
                    "native_head": target,
                    "trace_independent_label": float(independent_labels[index]),
                },
                "theta_before_sha256": {
                    "trace_head_v1": trace_before_sha,
                    "native_head": native_before_sha,
                    "trace_independent_label": independent_before_sha,
                },
                "theta_after_sha256": {
                    "trace_head_v1": trace_after_sha,
                    "native_head": native_after_sha,
                    "trace_independent_label": independent_after_sha,
                },
            }
        )

    train_trace_matrix = np.stack(train_trace)
    train_native_matrix = np.stack(train_native)
    train_target_vector = np.asarray(train_targets, dtype=np.float64)
    ridge_trace = fit_train_only_ridge(
        train_trace_matrix, train_target_vector, alpha=ridge_alpha
    )
    ridge_native = fit_train_only_ridge(
        train_native_matrix, train_target_vector, alpha=ridge_alpha
    )
    predictions["ridge_trace"]["train"] = ridge_trace.predictions(
        train_trace_matrix
    ).tolist()
    predictions["ridge_native"]["train"] = ridge_native.predictions(
        train_native_matrix
    ).tolist()
    for index, evidence in enumerate(reconstructed_train_rows):
        evidence["predictions"] = {
            condition: float(predictions[condition]["train"][index])
            for condition in ORDERED_CONDITIONS
        }
        evidence["evidence_sha256"] = _record_self_hash(
            evidence, "evidence_sha256"
        )

    for index, episode in enumerate(stream_pair.eval.episodes):
        observation = observe_episode(
            native_graph,
            trace_graph,
            episode,
            cue_removed=spec.cue_removed,
            feature_edge_ids=feature_edge_ids,
        )
        native_dummy_trace.mirror_episode(
            trace_graph,
            observation,
            feature_edge_ids=feature_edge_ids,
        )
        eval_order.extend(
            (
                (episode.episode_id, "features"),
                (episode.episode_id, "prediction"),
            )
        )
        activations: dict[str, float] = {}
        for condition, features in (
            ("trace_head_v1", observation.trace_features),
            ("native_head", observation.native_features),
            ("trace_independent_label", observation.trace_features),
            ("frozen_trace_head", observation.trace_features),
        ):
            activation, prediction, _ = heads[condition].predict(features)
            activations[condition] = activation
            predictions[condition]["eval"].append(prediction)
        predictions["rand"]["eval"].append(float(eval_random[index]))
        predictions["ridge_trace"]["eval"].append(
            float(ridge_trace.predictions(observation.trace_features[None, :])[0])
        )
        predictions["ridge_native"]["eval"].append(
            float(ridge_native.predictions(observation.native_features[None, :])[0])
        )
        target = _bipolar_target(episode.target)
        eval_order.append((episode.episode_id, "target_reveal"))
        eval_trace.append(observation.trace_features)
        eval_native.append(observation.native_features)
        eval_targets.append(target)
        eval_events.append(observation.event_payload_sha256)
        snapshot_sha = _array_sha256(observation.trace_snapshot)
        eval_snapshots.append(snapshot_sha)
        evidence = {
            "index": index,
            "episode_id": episode.episode_id,
            "noise_stream_id": episode.noise_stream_id,
            "query_tick": observation.query_tick,
            "target": target,
            "trace_features_sha256": _array_sha256(observation.trace_features),
            "native_features_sha256": _array_sha256(observation.native_features),
            "trace_snapshot_sha256": snapshot_sha,
            "event_payload_sha256": observation.event_payload_sha256,
            "predictions": {
                condition: float(predictions[condition]["eval"][index])
                for condition in ORDERED_CONDITIONS
            },
            "activations": activations,
            "theta_sha256": {
                condition: _array_sha256(heads[condition].theta)
                for condition in (
                    "trace_head_v1",
                    "native_head",
                    "trace_independent_label",
                )
            },
        }
        evidence["evidence_sha256"] = _record_self_hash(evidence, "evidence_sha256")
        reconstructed_eval_rows.append(evidence)

    eval_target_vector = np.asarray(eval_targets, dtype=np.float64)
    expected_conditions: dict[str, dict[str, Any]] = {}
    for condition in ORDERED_CONDITIONS:
        expected_conditions[condition] = {
            "train": _metric(predictions[condition]["train"], train_target_vector),
            "eval": _metric(predictions[condition]["eval"], eval_target_vector),
        }
    for condition, head in heads.items():
        expected_conditions[condition]["head"] = head.summary()
    expected_conditions["trace_independent_label"][
        "train_update_labels_sha256"
    ] = independent_manifest["labels_sha256"]
    expected_conditions["rand"]["manifest"] = {
        "train": train_random_manifest,
        "eval": eval_random_manifest,
        "fitted_parameter_count": 0,
    }
    expected_conditions["ridge_trace"]["diagnostic"] = ridge_trace.diagnostic()
    expected_conditions["ridge_native"]["diagnostic"] = ridge_native.diagnostic()

    expected_evidence: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "condition_order": list(ORDERED_CONDITIONS),
        "train": reconstructed_train_rows,
        "eval": reconstructed_eval_rows,
    }
    expected_evidence["evidence_sha256"] = _record_self_hash(
        expected_evidence, "evidence_sha256"
    )
    expected_identities = {
        "train_episode_ids": [
            episode.episode_id for episode in stream_pair.train.episodes
        ],
        "eval_episode_ids": [
            episode.episode_id for episode in stream_pair.eval.episodes
        ],
        "train_noise_stream_ids": [
            episode.noise_stream_id for episode in stream_pair.train.episodes
        ],
        "eval_noise_stream_ids": [
            episode.noise_stream_id for episode in stream_pair.eval.episodes
        ],
        "data_rng_seeds": [
            *_rng_identity_seeds(stream_pair.train),
            *_rng_identity_seeds(stream_pair.eval),
        ],
        "baseline_rng_seeds": [
            independent_manifest["seed"],
            train_random_manifest["seed"],
            eval_random_manifest["seed"],
        ],
        "rng_namespaces": [
            *_rng_identity_namespaces(stream_pair.train),
            *_rng_identity_namespaces(stream_pair.eval),
            independent_manifest["namespace"],
            train_random_manifest["namespace"],
            eval_random_manifest["namespace"],
        ],
    }
    expected_features = {
        "train_trace_features_sha256": _array_sha256(train_trace_matrix),
        "train_native_features_sha256": _array_sha256(train_native_matrix),
        "eval_trace_features_sha256": _array_sha256(np.stack(eval_trace)),
        "eval_native_features_sha256": _array_sha256(np.stack(eval_native)),
        "train_trace_snapshots_sha256": _sha256_json(train_snapshots),
        "eval_trace_snapshots_sha256": _sha256_json(eval_snapshots),
        "train_candidate_native_events_sha256": _sha256_json(train_events),
        "eval_candidate_native_events_sha256": _sha256_json(eval_events),
    }
    expected_graph = {
        "topology_sha256": topology,
        "structural_mask_sha256": structure,
        "weights_sha256": weights,
        "native_ledger": native_graph.ledger,
        "trace_native_ledger": trace_graph.ledger,
        "trace_ledger": trace_graph.trace_ledger,
        "native_dummy_trace_ledger": native_dummy_trace.ledger,
        "native_dummy_trace_state_sha256": native_dummy_trace.state_sha256,
        "candidate_trace_state_sha256": native_dummy_trace.candidate_state_sha256,
        "native_dummy_trace_history_sha256": native_dummy_trace.history_sha256,
        "candidate_trace_history_sha256": (
            native_dummy_trace.candidate_history_sha256
        ),
        "native_dummy_trace_effective_snapshot_sha256": (
            native_dummy_trace.effective_snapshot_sha256
        ),
        "candidate_trace_effective_snapshot_sha256": (
            native_dummy_trace.candidate_effective_snapshot_sha256
        ),
        "native_dummy_trace_weighted_outputs_sha256": (
            native_dummy_trace.weighted_outputs_sha256
        ),
        "candidate_trace_weighted_outputs_sha256": (
            native_dummy_trace.candidate_weighted_outputs_sha256
        ),
        "native_dummy_trace_reset_count": native_dummy_trace.reset_count,
        "native_dummy_trace_resets_valid": native_dummy_trace.resets_valid,
        "native_dummy_trace_bound_ok": native_dummy_trace.bound_ok,
    }
    exact_fields = {
        "stream_manifest": stream_pair.manifest.to_dict(),
        "cell_data_root": {
            "registered_master_seed": seed,
            "namespace": cell_data_namespace,
            "derived_seed": cell_data_seed,
        },
        "identity_evidence": expected_identities,
        "feature_edge_ids": list(feature_edge_ids),
        "feature_edge_ids_sha256": _sha256_json(list(feature_edge_ids)),
        "condition_results": expected_conditions,
        "independent_label_manifest": independent_manifest,
        "per_example_evidence": expected_evidence,
        "feature_evidence": expected_features,
        "temporal_order_sha256": _sha256_json(
            {"train": train_order, "eval": eval_order}
        ),
        "graph": expected_graph,
        "integrity": {key: True for key in sorted(_CELL_INTEGRITY_KEYS)},
    }
    mismatched = [
        field for field, expected in exact_fields.items() if reported.get(field) != expected
    ]
    if mismatched:
        raise RuntimeError(
            "independent learning reconstruction mismatch: "
            + ",".join(sorted(mismatched))
        )


def _validate_seed_result_mapping(seed_result: Mapping[str, Any], seed: int) -> None:
    if (
        seed_result.get("seed") != seed
        or type(seed_result.get("seed")) is not int
        or seed_result.get("seed_role") != SeedRole.CUSTOM.value
        or seed_result.get("all_cell_integrity_pass") is not True
        or not isinstance(seed_result.get("integrity"), Mapping)
        or not seed_result["integrity"]
        or not all(value is True for value in seed_result["integrity"].values())
    ):
        raise RuntimeError("learning seed metadata or integrity is invalid")
    cells = seed_result.get("cell_results")
    if not isinstance(cells, list) or [row.get("cell") for row in cells] != list(
        _FULL_CELL_NAMES
    ):
        raise RuntimeError("learning seed cell order is invalid")
    all_episode_ids: list[str] = []
    all_noise_ids: list[str] = []
    all_rng_seeds: list[int] = []
    all_rng_namespaces: list[str] = []
    data_namespaces: list[str] = []
    data_roots: list[int] = []
    for spec, row in zip(FULL_CELLS, cells):
        expected_data_seed, expected_data_namespace = derive_cell_data_seed(
            seed, spec.name
        )
        data_root = row.get("cell_data_root")
        identities = row.get("identity_evidence")
        manifest = row.get("stream_manifest")
        if (
            row.get("distractor_count") != spec.distractor_count
            or row.get("cue_removed") is not spec.cue_removed
            or row.get("train_examples") != spec.train_examples
            or row.get("eval_examples") != spec.eval_examples
            or row.get("feature_edge_ids")
            != sorted(row.get("feature_edge_ids", ()))
            or len(row.get("feature_edge_ids", ())) != FEATURE_DIMENSION
            or row.get("feature_edge_ids_sha256")
            != _sha256_json(row["feature_edge_ids"])
            or not isinstance(row.get("integrity"), Mapping)
            or not row["integrity"]
            or set(row["integrity"]) != _CELL_INTEGRITY_KEYS
            or not all(value is True for value in row["integrity"].values())
            or data_root
            != {
                "registered_master_seed": seed,
                "namespace": expected_data_namespace,
                "derived_seed": expected_data_seed,
            }
            or not isinstance(identities, Mapping)
            or not isinstance(manifest, Mapping)
            or manifest.get("master_seed") != expected_data_seed
            or manifest.get("seed_role") != SeedRole.CUSTOM.value
            or manifest.get("episodes_per_split") != spec.train_examples
            or manifest.get("noise_events") != spec.distractor_count
            or not isinstance(manifest.get("train_sha256"), str)
            or not isinstance(manifest.get("eval_sha256"), str)
            or not isinstance(manifest.get("pair_sha256"), str)
        ):
            raise RuntimeError("learning cell metadata or integrity is invalid")
        train_episode_ids = identities.get("train_episode_ids")
        eval_episode_ids = identities.get("eval_episode_ids")
        train_noise_ids = identities.get("train_noise_stream_ids")
        eval_noise_ids = identities.get("eval_noise_stream_ids")
        data_rng_seeds = identities.get("data_rng_seeds")
        baseline_rng_seeds = identities.get("baseline_rng_seeds")
        rng_namespaces = identities.get("rng_namespaces")
        if (
            not isinstance(train_episode_ids, list)
            or not isinstance(eval_episode_ids, list)
            or not isinstance(train_noise_ids, list)
            or not isinstance(eval_noise_ids, list)
            or len(train_episode_ids) != spec.train_examples
            or len(eval_episode_ids) != spec.eval_examples
            or len(train_noise_ids) != spec.train_examples
            or len(eval_noise_ids) != spec.eval_examples
            or len(set(train_episode_ids + eval_episode_ids))
            != spec.train_examples + spec.eval_examples
            or len(set(train_noise_ids + eval_noise_ids))
            != spec.train_examples + spec.eval_examples
            or not isinstance(data_rng_seeds, list)
            or not isinstance(baseline_rng_seeds, list)
            or len(data_rng_seeds) != 6
            or len(baseline_rng_seeds) != 3
            or len(set(data_rng_seeds + baseline_rng_seeds)) != 9
            or not isinstance(rng_namespaces, list)
            or len(rng_namespaces) != 9
            or len(set(rng_namespaces)) != 9
        ):
            raise RuntimeError("learning identity/RNG evidence is malformed")
        expected_data_rng_seeds = [
            manifest[split][field]
            for split in ("train_substreams", "eval_substreams")
            for field in ("cue_order_seed", "noise_seed", "identity_seed")
        ]
        condition_results = row["condition_results"]
        expected_baseline_rng_seeds = [
            row["independent_label_manifest"]["seed"],
            condition_results["rand"]["manifest"]["train"]["seed"],
            condition_results["rand"]["manifest"]["eval"]["seed"],
        ]
        expected_rng_namespaces = [
            f"{manifest[split]['namespace']}/{field}"
            for split in ("train_substreams", "eval_substreams")
            for field in ("cue-order", "noise", "identity")
        ] + [
            row["independent_label_manifest"]["namespace"],
            condition_results["rand"]["manifest"]["train"]["namespace"],
            condition_results["rand"]["manifest"]["eval"]["namespace"],
        ]
        if (
            data_rng_seeds != expected_data_rng_seeds
            or baseline_rng_seeds != expected_baseline_rng_seeds
            or rng_namespaces != expected_rng_namespaces
        ):
            raise RuntimeError("learning RNG manifests do not reconcile")
        all_episode_ids.extend(train_episode_ids + eval_episode_ids)
        all_noise_ids.extend(train_noise_ids + eval_noise_ids)
        all_rng_seeds.extend(data_rng_seeds + baseline_rng_seeds)
        all_rng_namespaces.extend(rng_namespaces)
        data_namespaces.append(expected_data_namespace)
        data_roots.append(expected_data_seed)
        conditions = row.get("condition_results")
        if not isinstance(conditions, Mapping) or set(conditions) != set(
            ORDERED_CONDITIONS
        ):
            raise RuntimeError("learning cell condition results are invalid")
        target_hashes_by_split = {
            split: {
                conditions[condition][split]["targets_sha256"]
                for condition in ORDERED_CONDITIONS
            }
            for split in ("train", "eval")
        }
        if any(len(hashes) != 1 for hashes in target_hashes_by_split.values()):
            raise RuntimeError("paired conditions do not share exact target order")
        for condition in ORDERED_CONDITIONS:
            for split, expected_count in (
                ("train", spec.train_examples),
                ("eval", spec.eval_examples),
            ):
                metric = conditions[condition].get(split)
                if (
                    not isinstance(metric, Mapping)
                    or metric.get("examples") != expected_count
                    or type(metric.get("correct")) is not int
                    or not 0 <= metric["correct"] <= expected_count
                    or metric.get("accuracy")
                    != float(metric["correct"] / expected_count)
                    or not isinstance(metric.get("predictions_sha256"), str)
                    or not isinstance(metric.get("targets_sha256"), str)
                ):
                    raise RuntimeError("learning condition metric does not reconstruct")
        for condition in (
            "trace_head_v1",
            "native_head",
            "trace_independent_label",
            "frozen_trace_head",
        ):
            head = conditions[condition].get("head")
            if (
                not isinstance(head, Mapping)
                or head.get("feature_dimension") != FEATURE_DIMENSION
                or head.get("parameter_count") != FEATURE_DIMENSION
                or head.get("bias_parameter_count") != 0
                or len(head.get("theta", ())) != FEATURE_DIMENSION
                or head.get("theta_sha256") != _array_sha256(head["theta"])
                or head.get("pending_prediction") is not False
                or head.get("all_parameters_finite") is not True
                or head.get("all_parameters_within_clip") is not True
            ):
                raise RuntimeError("learning head evidence is malformed")
            theta = np.asarray(head["theta"], dtype=np.float64)
            ledger = head.get("ledger")
            trainable = condition != "frozen_trace_head"
            expected_updates = spec.train_examples if trainable else 0
            if (
                head.get("maximum_absolute_theta")
                != float(np.max(np.abs(theta)))
                or not isinstance(ledger, Mapping)
                or ledger.get("prediction_count")
                != spec.train_examples + spec.eval_examples
                or ledger.get("target_reveal_count") != expected_updates
                or ledger.get("update_count") != expected_updates
                or ledger.get("parameter_read_touches")
                != (spec.train_examples + spec.eval_examples) * FEATURE_DIMENSION
                or ledger.get("parameter_write_touches")
                != expected_updates * FEATURE_DIMENSION
                or not 0.0
                <= float(ledger.get("maximum_absolute_delta", math.inf))
                <= ELEMENT_UPDATE_CLIP
            ):
                raise RuntimeError("learning head ledger does not reconcile")
        frozen_head = conditions["frozen_trace_head"]["head"]
        if not all(
            float(value) == 0.0 and math.copysign(1.0, float(value)) == 1.0
            for value in frozen_head["theta"]
        ):
            raise RuntimeError("frozen head is not exact positive zero")
        expected_frozen_train_hash = _array_sha256(
            np.ones(spec.train_examples, dtype=np.float64)
        )
        expected_frozen_eval_hash = _array_sha256(
            np.ones(spec.eval_examples, dtype=np.float64)
        )
        if (
            conditions["frozen_trace_head"]["train"]["predictions_sha256"]
            != expected_frozen_train_hash
            or conditions["frozen_trace_head"]["eval"]["predictions_sha256"]
            != expected_frozen_eval_hash
            or conditions["frozen_trace_head"]["train"]["accuracy"] != 0.5
            or conditions["frozen_trace_head"]["eval"]["accuracy"] != 0.5
        ):
            raise RuntimeError("frozen-head control does not reconstruct")
        independent_manifest = row.get("independent_label_manifest")
        _, expected_independent_manifest = balanced_independent_labels(
            master_seed=seed,
            cell_name=spec.name,
            count=spec.train_examples,
        )
        if (
            not isinstance(independent_manifest, Mapping)
            or dict(independent_manifest) != expected_independent_manifest
            or independent_manifest.get("negative_count") != spec.train_examples // 2
            or independent_manifest.get("positive_count") != spec.train_examples // 2
            or independent_manifest.get("bit_generator") != "numpy.random.PCG64"
            or conditions["trace_independent_label"].get(
                "train_update_labels_sha256"
            )
            != independent_manifest.get("labels_sha256")
        ):
            raise RuntimeError("independent-label control is malformed")
        random_manifest = conditions["rand"].get("manifest")
        _, expected_random_train = deterministic_random_predictions(
            master_seed=seed,
            cell_name=spec.name,
            split="train",
            count=spec.train_examples,
        )
        _, expected_random_eval = deterministic_random_predictions(
            master_seed=seed,
            cell_name=spec.name,
            split="eval",
            count=spec.eval_examples,
        )
        if (
            not isinstance(random_manifest, Mapping)
            or random_manifest.get("fitted_parameter_count") != 0
            or random_manifest.get("train") != expected_random_train
            or random_manifest.get("eval") != expected_random_eval
            or random_manifest.get("train", {}).get("predictions_sha256")
            != conditions["rand"]["train"]["predictions_sha256"]
            or random_manifest.get("eval", {}).get("predictions_sha256")
            != conditions["rand"]["eval"]["predictions_sha256"]
        ):
            raise RuntimeError("deterministic-random control is malformed")
        features = row.get("feature_evidence")
        for condition, feature_key in (
            ("ridge_trace", "train_trace_features_sha256"),
            ("ridge_native", "train_native_features_sha256"),
        ):
            diagnostic = conditions[condition].get("diagnostic")
            coefficient = np.asarray(
                diagnostic.get("coefficients", ())
                if isinstance(diagnostic, Mapping)
                else (),
                dtype=np.float64,
            )
            feature_mean = np.asarray(
                diagnostic.get("train_feature_mean", ())
                if isinstance(diagnostic, Mapping)
                else (),
                dtype=np.float64,
            )
            feature_scale = np.asarray(
                diagnostic.get("train_feature_scale", ())
                if isinstance(diagnostic, Mapping)
                else (),
                dtype=np.float64,
            )
            active_mask = np.asarray(
                diagnostic.get("active_mask", ())
                if isinstance(diagnostic, Mapping)
                else (),
                dtype=np.bool_,
            )
            if (
                not isinstance(diagnostic, Mapping)
                or diagnostic.get("fit_split") != "train_only"
                or diagnostic.get("evaluation_fit_count") != 0
                or diagnostic.get("feature_dimension") != FEATURE_DIMENSION
                or diagnostic.get("train_examples") != spec.train_examples
                or not isinstance(features, Mapping)
                or diagnostic.get("train_features_sha256")
                != features.get(feature_key)
                or coefficient.shape != (FEATURE_DIMENSION,)
                or feature_mean.shape != (FEATURE_DIMENSION,)
                or feature_scale.shape != (FEATURE_DIMENSION,)
                or active_mask.shape != (FEATURE_DIMENSION,)
                or not np.all(np.isfinite(coefficient))
                or not np.all(np.isfinite(feature_mean))
                or not np.all(np.isfinite(feature_scale))
                or diagnostic.get("coefficient_sha256")
                != _array_sha256(coefficient)
                or diagnostic.get("train_feature_mean_sha256")
                != _array_sha256(feature_mean)
                or diagnostic.get("train_feature_scale_sha256")
                != _array_sha256(feature_scale)
                or diagnostic.get("active_mask_sha256")
                != _array_sha256(active_mask.astype(np.float64))
                or diagnostic.get("active_feature_count")
                != int(np.count_nonzero(active_mask))
                or diagnostic.get("inactive_feature_count")
                != int(np.count_nonzero(~active_mask))
                or diagnostic.get("coefficient_l2")
                != float(np.linalg.norm(coefficient))
                or not np.all(coefficient[~active_mask] == 0.0)
            ):
                raise RuntimeError("train-only ridge evidence is malformed")
        graph = row.get("graph")
        total_examples = spec.train_examples + spec.eval_examples
        native_ledger = graph.get("native_ledger", {}) if isinstance(graph, Mapping) else {}
        trace_ledger = graph.get("trace_ledger", {}) if isinstance(graph, Mapping) else {}
        dummy_trace_ledger = (
            graph.get("native_dummy_trace_ledger", {})
            if isinstance(graph, Mapping)
            else {}
        )
        if (
            not isinstance(graph, Mapping)
            or graph.get("native_ledger") != graph.get("trace_native_ledger")
            or native_ledger.get("episodes") != total_examples
            or native_ledger.get("ticks")
            != total_examples * (spec.distractor_count + 4)
            or native_ledger.get("credit_event_touches") != 0
            or native_ledger.get("credit_edge_touches") != 0
            or native_ledger.get("weight_write_touches") != 0
            or native_ledger.get("nonfinite_values") != 0
            or trace_ledger.get("local_trace_reset_touches")
            != total_examples * HIDDEN_DIMENSION
            or trace_ledger.get("local_trace_observation_touches")
            != total_examples * HIDDEN_DIMENSION
            or trace_ledger.get("hidden_activation_evaluations")
            != trace_ledger.get("local_trace_read_touches")
            or trace_ledger.get("local_trace_read_touches")
            != trace_ledger.get("local_trace_decay_touches")
            or trace_ledger.get("local_trace_decay_touches")
            != trace_ledger.get("local_trace_write_touches")
            or dummy_trace_ledger != trace_ledger
            or graph.get("native_dummy_trace_state_sha256")
            != graph.get("candidate_trace_state_sha256")
            or graph.get("native_dummy_trace_history_sha256")
            != graph.get("candidate_trace_history_sha256")
            or graph.get("native_dummy_trace_effective_snapshot_sha256")
            != graph.get("candidate_trace_effective_snapshot_sha256")
            or graph.get("native_dummy_trace_weighted_outputs_sha256")
            != graph.get("candidate_trace_weighted_outputs_sha256")
            or not all(
                _is_sha256(graph.get(key))
                for key in (
                    "native_dummy_trace_state_sha256",
                    "candidate_trace_state_sha256",
                    "native_dummy_trace_history_sha256",
                    "candidate_trace_history_sha256",
                    "native_dummy_trace_effective_snapshot_sha256",
                    "candidate_trace_effective_snapshot_sha256",
                    "native_dummy_trace_weighted_outputs_sha256",
                    "candidate_trace_weighted_outputs_sha256",
                )
            )
            or graph.get("native_dummy_trace_reset_count") != total_examples
            or graph.get("native_dummy_trace_resets_valid") is not True
            or graph.get("native_dummy_trace_bound_ok") is not True
        ):
            raise RuntimeError("learning graph ledger evidence is malformed")
        evidence = row.get("per_example_evidence")
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
            or evidence.get("condition_order") != list(ORDERED_CONDITIONS)
            or not isinstance(evidence.get("train"), list)
            or not isinstance(evidence.get("eval"), list)
            or len(evidence["train"]) != spec.train_examples
            or len(evidence["eval"]) != spec.eval_examples
            or evidence.get("evidence_sha256")
            != _record_self_hash(evidence, "evidence_sha256")
            or any(
                not isinstance(example, Mapping)
                or example.get("evidence_sha256")
                != _record_self_hash(example, "evidence_sha256")
                for split in ("train", "eval")
                for example in evidence[split]
            )
        ):
            raise RuntimeError("per-example learning evidence is malformed")
    expected_seed_integrity = {
        "cross_cell_episode_ids_disjoint": len(all_episode_ids)
        == len(set(all_episode_ids)),
        "cross_cell_noise_stream_ids_disjoint": len(all_noise_ids)
        == len(set(all_noise_ids)),
        "cross_cell_rng_seeds_disjoint": len(all_rng_seeds) == len(set(all_rng_seeds)),
        "cross_cell_rng_namespaces_disjoint": len(all_rng_namespaces)
        == len(set(all_rng_namespaces)),
        "cell_data_namespaces_disjoint": len(data_namespaces)
        == len(set(data_namespaces)),
        "cell_data_roots_disjoint": len(data_roots) == len(set(data_roots)),
    }
    if seed_result.get("integrity") != expected_seed_integrity or not all(
        expected_seed_integrity.values()
    ):
        raise RuntimeError("learning cross-cell integrity does not reconstruct")
    if not _all_finite(seed_result):
        raise FloatingPointError("learning seed result contains a non-finite value")


def _report_integrity(
    seed_results: Sequence[Mapping[str, Any]],
    *,
    workers: int,
    worker_runtime: Sequence[Mapping[str, Any]],
    source_unchanged: bool,
    prerequisites_unchanged: bool,
) -> dict[str, bool | int]:
    process_ids = {int(row["process_id"]) for row in worker_runtime}
    global_identity = _global_identity_integrity(seed_results)
    return {
        "all_registered_custom_seeds": all(
            row["seed"] in REGISTERED_SEEDS
            and row["seed_role"] == SeedRole.CUSTOM.value
            for row in seed_results
        ),
        "seed_order_exact": [row["seed"] for row in seed_results]
        == list(REGISTERED_SEEDS),
        "confirmatory_seed_count": 0,
        "all_cell_integrity_pass": all(
            row["all_cell_integrity_pass"] is True for row in seed_results
        ),
        **global_identity,
        "worker_count_exactly_five": workers == ORDERED_CPU_WORKERS,
        "worker_process_count_exactly_five": len(process_ids)
        == ORDERED_CPU_WORKERS,
        "all_workers_single_thread": all(
            row["blas"]
            and all(pool["num_threads"] == 1 for pool in row["blas"])
            for row in worker_runtime
        ),
        "source_manifest_unchanged": source_unchanged,
        "prerequisites_unchanged": prerequisites_unchanged,
    }


def _global_identity_integrity(
    seed_results: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    episode_ids: list[str] = []
    noise_stream_ids: list[str] = []
    rng_seeds: list[int] = []
    rng_namespaces: list[str] = []
    for seed_result in seed_results:
        for cell in seed_result["cell_results"]:
            identities = cell["identity_evidence"]
            episode_ids.extend(identities["train_episode_ids"])
            episode_ids.extend(identities["eval_episode_ids"])
            noise_stream_ids.extend(identities["train_noise_stream_ids"])
            noise_stream_ids.extend(identities["eval_noise_stream_ids"])
            rng_seeds.extend(int(value) for value in identities["data_rng_seeds"])
            rng_seeds.extend(int(value) for value in identities["baseline_rng_seeds"])
            rng_namespaces.extend(
                str(value) for value in identities["rng_namespaces"]
            )
    return {
        "global_episode_ids_disjoint": len(episode_ids) == len(set(episode_ids)),
        "global_noise_stream_ids_disjoint": len(noise_stream_ids)
        == len(set(noise_stream_ids)),
        "global_rng_seeds_disjoint": len(rng_seeds) == len(set(rng_seeds)),
        "global_rng_namespaces_disjoint": len(rng_namespaces)
        == len(set(rng_namespaces)),
    }


def _assemble_full_report(
    *,
    seed_results: Sequence[Mapping[str, Any]],
    worker_runtime: Sequence[Mapping[str, Any]],
    prerequisites: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    source_freeze_record: Mapping[str, Any],
    workers: int,
    run_started_utc: str,
    runtime_seconds: float,
    artifact_role: str,
    primary_binding: Mapping[str, str] | None,
    coordinator: Mapping[str, Any],
) -> dict[str, Any]:
    aggregate = aggregate_seed_results(seed_results, cell_names=_FULL_CELL_NAMES)
    statistics = compute_registered_statistics(aggregate)
    integrity = _report_integrity(
        seed_results,
        workers=workers,
        worker_runtime=worker_runtime,
        source_unchanged=True,
        prerequisites_unchanged=True,
    )
    integrity_pass = _report_integrity_passes(integrity)
    gates = evaluate_registered_gates(
        aggregate,
        statistics,
        integrity_pass=integrity_pass,
        eligible=True,
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "run_kind": "full",
        "scope": "custom_seed_true_learning_result",
        "seeds": list(REGISTERED_SEEDS),
        "conditions": list(ORDERED_CONDITIONS),
        "cell_specs": [asdict(cell) for cell in FULL_CELLS],
        "trace_retention": float(prerequisites["trace_retention"]),
        "selected_a3_condition": prerequisites["selected_condition"],
        "ridge_alpha": DEFAULT_RIDGE_ALPHA,
        "worker_count": workers,
        "ordered_parallel_execution": True,
        "confirmatory_executed": False,
        "confirmatory_seed_count": 0,
        "backend_contract": {
            "engine": "numpy_cpu",
            "float_type": "float64",
            "gpu_used": False,
            "numeric_library_threads_per_worker": 1,
        },
        "head_contract": {
            "parameter_count": FEATURE_DIMENSION,
            "bias": False,
            "initialization": "exact_positive_zero",
            "activation": "tanh",
            "loss": "0.5*(y-q)^2",
            "learning_rate": LEARNING_RATE,
            "normalizer": "max(1,||z||^2)",
            "element_update_clip": ELEMENT_UPDATE_CLIP,
            "parameter_clip": THETA_CLIP,
            "target_encoding": [-1.0, 1.0],
        },
        "temporal_contract": [
            "cue_and_distractors",
            "query",
            "features",
            "prediction",
            "target_reveal",
            "update_train_only",
        ],
        "graph_options": dict(FROZEN_GRAPH_OPTIONS),
        "gate_thresholds": dict(GATE_THRESHOLDS),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "independent_reconstruction_required": True,
        "per_example_evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "compute_match_contract": (
            "native graph plus faithful blind 64-state state/timestamp/history/"
            "snapshot/feature/dot/tanh replay and identical online-head budget"
        ),
        "prerequisites": copy.deepcopy(dict(prerequisites)),
        "pre_official_source_freeze": copy.deepcopy(dict(source_freeze_record)),
        "source_manifest_start": copy.deepcopy(dict(source_manifest)),
        "source_manifest_end": copy.deepcopy(dict(source_manifest)),
        "seed_results": list(seed_results),
        "aggregate": aggregate,
        "statistics": statistics,
        "integrity": integrity,
        "gates": gates,
        "status": "PENDING_SEPARATE_PROCESS_DETERMINISM_VERIFICATION",
        "provisional_terminal_status": gates["provisional_terminal_status"],
        "readiness": {
            "score_before": READINESS_BEFORE,
            "score_after_pass": READINESS_AFTER_PASS,
            "score_after_fail": READINESS_AFTER_FAIL,
            "provisional_score_if_verified": (
                READINESS_AFTER_PASS
                if gates["provisional_terminal_status"] == "LEARNING_RESULT_PASS"
                else READINESS_AFTER_FAIL
            ),
            "score_change_deferred_until_terminal_determinism_verification": True,
        },
        "nondeterministic_provenance": {
            "excluded_from_deterministic_payload_sha256": True,
            "artifact_role": artifact_role,
            "process_id": os.getpid(),
            "process_instance_token": _PROCESS_INSTANCE_TOKEN,
            "process_started_utc": coordinator["process_started_utc"],
            "run_instance_token": coordinator["run_instance_token"],
            "coordinator_binding_sha256": coordinator["binding_sha256"],
            "run_started_utc": run_started_utc,
            "runtime_seconds": runtime_seconds,
            "implementation": platform.python_implementation(),
            "python": platform.python_version(),
            "executable": sys.executable,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "worker_runtime": list(worker_runtime),
            "rerun_primary_artifact_binding": (
                None if primary_binding is None else dict(primary_binding)
            ),
        },
    }
    report["nondeterministic_provenance"]["provenance_sha256"] = _record_self_hash(
        report["nondeterministic_provenance"], "provenance_sha256"
    )
    report["deterministic_payload_sha256"] = _recompute_report_payload_sha256(
        report
    )
    return report


def validate_protocol_config() -> dict[str, Any]:
    """Load the frozen TOML and reject scientific-field drift."""

    # The external record covers the complete bytes of both files; selected
    # field checks below additionally produce precise scientific-drift errors.
    validate_pre_official_source_freeze()
    with _project_path(CONFIG_PATH).open("rb") as handle:
        payload = tomllib.load(handle)
    expected = {
        ("protocol", "name"): PROTOCOL_NAME,
        ("protocol", "version"): PROTOCOL_VERSION,
        ("protocol", "schema_version"): SCHEMA_VERSION,
        ("protocol", "readiness_score_before"): READINESS_BEFORE,
        ("protocol", "readiness_score_after_pass"): READINESS_AFTER_PASS,
        ("protocol", "readiness_score_after_fail"): READINESS_AFTER_FAIL,
        ("execution", "official_seeds"): list(REGISTERED_SEEDS),
        ("execution", "scratch_smoke_seeds"): list(SCRATCH_SMOKE_SEEDS),
        ("execution", "scratch_smoke_examples_per_split"): SMOKE_EXAMPLES_PER_SPLIT,
        ("execution", "worker_count"): ORDERED_CPU_WORKERS,
        ("head", "feature_count"): FEATURE_DIMENSION,
        ("head", "learning_rate"): LEARNING_RATE,
        ("head", "coordinate_update_clip"): ELEMENT_UPDATE_CLIP,
        ("head", "parameter_clip"): THETA_CLIP,
        ("ridge", "alpha"): DEFAULT_RIDGE_ALPHA,
        ("statistics", "exact_sign_flip_assignments"): 1
        << len(REGISTERED_SEEDS),
        ("statistics", "bootstrap_replicates"): BOOTSTRAP_REPLICATES,
        ("statistics", "alternative"): "trace_head_v1_greater",
        ("statistics", "predeclared_contrasts"): [
            "trace_vs_frozen",
            "trace_vs_native",
            "trace_vs_independent_label",
        ],
        ("statistics", "multiple_testing"): "Holm",
        ("statistics", "adjusted_p_max_exclusive"): 0.05,
        ("statistics", "bootstrap_confidence"): BOOTSTRAP_CONFIDENCE,
        ("statistics", "bootstrap_rng"): "numpy.random.PCG64",
        ("freeze", "schema_version"): FREEZE_SCHEMA_VERSION,
        ("freeze", "recorded_before_official_metrics"): True,
        ("freeze", "exact_protocol_and_config_bytes_required"): True,
        ("freeze", "all_source_hashes_required"): True,
        ("freeze", "record_path"): CANONICAL_FREEZE_RECORD_PATH.as_posix(),
        ("freeze", "record_mode"): "external_read_only_self_hashed",
        ("freeze", "external_anchor_environment"): FREEZE_ANCHOR_ENV,
        ("freeze", "transitive_local_import_closure_required"): True,
        ("freeze", "upstream_protocol_config_test_bindings_required"): True,
        ("compute_matching", "scope"): "total_mechanism",
        ("compute_matching", "native_dummy_trace_state_count"): HIDDEN_DIMENSION,
        ("compute_matching", "native_dummy_trace_work_must_equal_candidate"): True,
        (
            "compute_matching",
            "native_dummy_state_timestamp_history_replay_required",
        ): True,
        (
            "compute_matching",
            "native_dummy_query_feature_dot_tanh_replay_required",
        ): True,
        (
            "compute_matching",
            "native_dummy_values_must_not_enter_native_prediction",
        ): True,
        ("compute_matching", "graph_ledgers_must_match"): True,
        ("compute_matching", "online_head_ledgers_must_match"): True,
        ("verification", "per_example_evidence_schema"): EVIDENCE_SCHEMA_VERSION,
        ("verification", "registered_stream_reconstruction_required"): True,
        ("verification", "equation_level_replay_required"): True,
        (
            "verification",
            "verifier_orchestrated_reconstruction_subprocesses",
        ): 2,
        ("verification", "fresh_nonce_binding_required"): True,
        (
            "verification",
            "cross_run_coordinator_worker_identities_disjoint",
        ): True,
        ("verification", "coordinator_worker_bindings_required"): True,
        ("verification", "terminal_sha256_sidecar_required"): True,
        ("verification", "post_persistence_input_recheck_required"): True,
        ("decision", "pass_terminal"): "LEARNING_RESULT_PASS",
        ("decision", "fail_terminal_prefix"): "LEARNING_RESULT_FAIL:",
    }
    for (section, field), expected_value in expected.items():
        if payload.get(section, {}).get(field) != expected_value:
            raise RuntimeError(f"frozen config drift: {section}.{field}")
    expected_prerequisites = {
        "a3": {
            "protocol_version": "readout-trace-v1a3",
            "registered_seeds": [105, 106, 107, 108, 109],
            "phase_ledger": (
                "artifacts/experiment_000/readout_trace_a3/PHASE_SEQUENCE.json"
            ),
            "primary_report": (
                "artifacts/experiment_000/readout_trace_a3/"
                "custom_seeds_105_109_pairs_100.json"
            ),
            "rerun_report": (
                "artifacts/experiment_000/readout_trace_a3/"
                "custom_seeds_105_109_pairs_100_rerun.json"
            ),
            "determinism_report": (
                "artifacts/experiment_000/readout_trace_a3/"
                "DETERMINISM_VERIFICATION.json"
            ),
            "required_terminal_prefix": "SELECTED:",
            "resource_invalid_predecessor": "readout-trace-v1a2",
        },
        "alignment": {
            "primary_report": (
                "artifacts/experiment_000/trace_head_alignment/"
                "alignment_seeds_75_79_episodes_32.json"
            ),
            "rerun_report": (
                "artifacts/experiment_000/trace_head_alignment/"
                "alignment_seeds_75_79_episodes_32_rerun.json"
            ),
            "determinism_report": (
                "artifacts/experiment_000/trace_head_alignment/"
                "DETERMINISM_VERIFICATION.json"
            ),
            "required_schema": "experiment-000-trace-head-l1-v1",
            "required_terminal": "TRACE_HEAD_ALIGNMENT_PASS",
            "selected_condition_and_retention_must_match_a3": True,
            "canonical_payload_equality_required": True,
            "fresh_process_required": True,
        },
    }
    if payload.get("prerequisites") != expected_prerequisites:
        raise RuntimeError("frozen config prerequisite chain drifted")
    config_cells = payload.get("cells")
    if config_cells != [asdict(cell) for cell in FULL_CELLS]:
        raise RuntimeError("frozen config cell matrix drifted")
    if payload.get("conditions", {}).get("ordered") != list(ORDERED_CONDITIONS):
        raise RuntimeError("frozen config condition order drifted")
    configured_thresholds = {
        **payload.get("gates", {}).get("primary", {}),
        **payload.get("gates", {}).get("robustness", {}),
    }
    expected_thresholds = {
        key: value
        for key, value in GATE_THRESHOLDS.items()
        if key != "holm_adjusted_p_max_exclusive"
    }
    # The cue-removed interval is named separately in TOML but shares chance bounds.
    expected_thresholds |= {
        "cue_removed_accuracy_min": GATE_THRESHOLDS["chance_accuracy_min"],
        "cue_removed_accuracy_max": GATE_THRESHOLDS["chance_accuracy_max"],
    }
    if configured_thresholds != expected_thresholds:
        raise RuntimeError("frozen config gate matrix drifted")
    return payload


def _expected_report_integrity(seed_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "all_registered_custom_seeds": all(
            row["seed"] in REGISTERED_SEEDS
            and row["seed_role"] == SeedRole.CUSTOM.value
            for row in seed_results
        ),
        "seed_order_exact": [row["seed"] for row in seed_results]
        == list(REGISTERED_SEEDS),
        "confirmatory_seed_count": 0,
        "all_cell_integrity_pass": all(
            row["all_cell_integrity_pass"] is True for row in seed_results
        ),
        **_global_identity_integrity(seed_results),
        "worker_count_exactly_five": True,
        "worker_process_count_exactly_five": True,
        "all_workers_single_thread": True,
        "source_manifest_unchanged": True,
        "prerequisites_unchanged": True,
    }


def _report_integrity_passes(integrity: Mapping[str, Any]) -> bool:
    """Apply the sole frozen I01 predicate used by builder and verifier."""

    return bool(integrity) and all(
        value == 0 if key == "confirmatory_seed_count" else value is True
        for key, value in integrity.items()
    )


def validate_full_report(
    report: Mapping[str, Any],
    *,
    expected_prerequisites: Mapping[str, Any] | None = None,
    current_source_manifest: Mapping[str, Any] | None = None,
) -> None:
    """Strictly reconstruct metadata, aggregates, statistics, gates, and hashes."""

    expected_backend = {
        "engine": "numpy_cpu",
        "float_type": "float64",
        "gpu_used": False,
        "numeric_library_threads_per_worker": 1,
    }
    expected_head_contract = {
        "parameter_count": FEATURE_DIMENSION,
        "bias": False,
        "initialization": "exact_positive_zero",
        "activation": "tanh",
        "loss": "0.5*(y-q)^2",
        "learning_rate": LEARNING_RATE,
        "normalizer": "max(1,||z||^2)",
        "element_update_clip": ELEMENT_UPDATE_CLIP,
        "parameter_clip": THETA_CLIP,
        "target_encoding": [-1.0, 1.0],
    }
    expected_temporal_contract = [
        "cue_and_distractors",
        "query",
        "features",
        "prediction",
        "target_reveal",
        "update_train_only",
    ]
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("protocol_name") != PROTOCOL_NAME
        or report.get("protocol_version") != PROTOCOL_VERSION
        or report.get("run_kind") != "full"
        or report.get("scope") != "custom_seed_true_learning_result"
        or report.get("seeds") != list(REGISTERED_SEEDS)
        or not isinstance(report.get("seeds"), list)
        or any(type(seed) is not int for seed in report["seeds"])
        or report.get("conditions") != list(ORDERED_CONDITIONS)
        or report.get("cell_specs") != [asdict(cell) for cell in FULL_CELLS]
        or report.get("ridge_alpha") != DEFAULT_RIDGE_ALPHA
        or type(report.get("ridge_alpha")) is not float
        or report.get("worker_count") != ORDERED_CPU_WORKERS
        or type(report.get("worker_count")) is not int
        or report.get("ordered_parallel_execution") is not True
        or report.get("confirmatory_executed") is not False
        or report.get("confirmatory_seed_count") != 0
        or report.get("backend_contract") != expected_backend
        or report.get("head_contract") != expected_head_contract
        or report.get("temporal_contract") != expected_temporal_contract
        or report.get("graph_options") != dict(FROZEN_GRAPH_OPTIONS)
        or report.get("gate_thresholds") != dict(GATE_THRESHOLDS)
        or report.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES
        or report.get("independent_reconstruction_required") is not True
        or report.get("per_example_evidence_schema_version")
        != EVIDENCE_SCHEMA_VERSION
        or report.get("compute_match_contract")
        != (
            "native graph plus faithful blind 64-state state/timestamp/history/"
            "snapshot/feature/dot/tanh replay and identical online-head budget"
        )
        or report.get("status")
        != "PENDING_SEPARATE_PROCESS_DETERMINISM_VERIFICATION"
        or report.get("deterministic_payload_sha256")
        != _recompute_report_payload_sha256(report)
    ):
        raise RuntimeError("full learning report metadata or self-hash is invalid")
    expected_freeze = validate_pre_official_source_freeze()
    if report.get("pre_official_source_freeze") != expected_freeze:
        raise RuntimeError("full learning report source-freeze binding changed")
    if expected_prerequisites is not None and report.get("prerequisites") != dict(
        expected_prerequisites
    ):
        raise RuntimeError("full learning report prerequisite binding changed")
    source = current_source_manifest
    if source is not None and (
        report.get("source_manifest_start") != dict(source)
        or report.get("source_manifest_end") != dict(source)
    ):
        raise RuntimeError("full learning report source manifest changed")
    prerequisites = report.get("prerequisites")
    if not isinstance(prerequisites, Mapping):
        raise TypeError("full learning report prerequisite binding is absent")
    if (
        report.get("selected_a3_condition") != prerequisites.get("selected_condition")
        or report.get("trace_retention") != prerequisites.get("trace_retention")
        or not isinstance(report.get("selected_a3_condition"), str)
        or not math.isfinite(float(report.get("trace_retention")))
    ):
        raise RuntimeError("full learning report selected trace binding is invalid")
    seed_results = report.get("seed_results")
    if not isinstance(seed_results, list) or [row.get("seed") for row in seed_results] != list(
        REGISTERED_SEEDS
    ):
        raise RuntimeError("full learning report seed rows are absent or out of order")
    for seed, seed_result in zip(REGISTERED_SEEDS, seed_results):
        _validate_seed_result_mapping(seed_result, seed)
        if any(
            cell_result.get("trace_retention") != report["trace_retention"]
            for cell_result in seed_result["cell_results"]
        ):
            raise RuntimeError("seed cell retention differs from the A3 selection")
        for spec, cell_result in zip(FULL_CELLS, seed_result["cell_results"]):
            _independently_reconstruct_cell(
                cell_result,
                seed=seed,
                spec=spec,
                trace_retention=float(report["trace_retention"]),
                ridge_alpha=float(report["ridge_alpha"]),
            )
    expected_aggregate = aggregate_seed_results(
        seed_results, cell_names=_FULL_CELL_NAMES
    )
    if report.get("aggregate") != expected_aggregate:
        raise RuntimeError("full learning report aggregate does not reconstruct")
    expected_statistics = compute_registered_statistics(expected_aggregate)
    if report.get("statistics") != expected_statistics:
        raise RuntimeError("full learning report statistics do not reconstruct")
    expected_integrity = _expected_report_integrity(seed_results)
    if report.get("integrity") != expected_integrity:
        raise RuntimeError("full learning report integrity does not reconstruct")
    expected_gates = evaluate_registered_gates(
        expected_aggregate,
        expected_statistics,
        integrity_pass=_report_integrity_passes(expected_integrity),
        eligible=True,
    )
    if (
        report.get("gates") != expected_gates
        or report.get("provisional_terminal_status")
        != expected_gates["provisional_terminal_status"]
    ):
        raise RuntimeError("full learning report gates do not reconstruct")
    expected_readiness = {
        "score_before": READINESS_BEFORE,
        "score_after_pass": READINESS_AFTER_PASS,
        "score_after_fail": READINESS_AFTER_FAIL,
        "provisional_score_if_verified": (
            READINESS_AFTER_PASS
            if expected_gates["provisional_terminal_status"]
            == "LEARNING_RESULT_PASS"
            else READINESS_AFTER_FAIL
        ),
        "score_change_deferred_until_terminal_determinism_verification": True,
    }
    if report.get("readiness") != expected_readiness:
        raise RuntimeError("full learning report readiness decision is incoherent")
    provenance = report.get("nondeterministic_provenance")
    worker_runtime = (
        provenance.get("worker_runtime")
        if isinstance(provenance, Mapping)
        else None
    )
    coordinator_identity = (
        {
            key: provenance.get(key)
            for key in (
                "process_id",
                "process_instance_token",
                "process_started_utc",
                "run_instance_token",
                "run_started_utc",
            )
        }
        if isinstance(provenance, Mapping)
        else {}
    )
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("excluded_from_deterministic_payload_sha256") is not True
        or provenance.get("artifact_role") not in {"primary", "rerun"}
        or type(provenance.get("process_id")) is not int
        or provenance["process_id"] <= 0
        or not _is_sha256(provenance.get("process_instance_token"))
        or not _is_sha256(provenance.get("run_instance_token"))
        or not _is_utc_timestamp(provenance.get("process_started_utc"))
        or not _is_utc_timestamp(provenance.get("run_started_utc"))
        or provenance.get("coordinator_binding_sha256")
        != _execution_binding(coordinator_identity)
        or provenance.get("provenance_sha256")
        != _record_self_hash(provenance, "provenance_sha256")
        or not isinstance(provenance.get("runtime_seconds"), (int, float))
        or isinstance(provenance.get("runtime_seconds"), bool)
        or not math.isfinite(float(provenance["runtime_seconds"]))
        or float(provenance["runtime_seconds"]) < 0.0
        or not isinstance(worker_runtime, list)
        or [row.get("seed") for row in worker_runtime] != list(REGISTERED_SEEDS)
        or len({row.get("process_id") for row in worker_runtime})
        != ORDERED_CPU_WORKERS
        or len({row.get("process_instance_token") for row in worker_runtime})
        != ORDERED_CPU_WORKERS
        or len(
            {
                (
                    row.get("process_id"),
                    row.get("process_instance_token"),
                    row.get("process_started_utc"),
                    row.get("parent_process_id"),
                    row.get("coordinator_binding_sha256"),
                )
                for row in worker_runtime
            }
        )
        != ORDERED_CPU_WORKERS
        or provenance.get("process_id")
        in {row.get("process_id") for row in worker_runtime}
        or not isinstance(provenance.get("parent_blas"), list)
        or not provenance["parent_blas"]
        or any(
            pool.get("num_threads") != 1 for pool in provenance["parent_blas"]
        )
        or any(
            type(row.get("seed")) is not int
            or type(row.get("process_id")) is not int
            or row["process_id"] <= 0
            or row.get("parent_process_id") != provenance.get("process_id")
            or not _is_sha256(row.get("process_instance_token"))
            or not _is_utc_timestamp(row.get("process_started_utc"))
            or not _is_utc_timestamp(row.get("job_started_utc"))
            or row.get("coordinator_binding_sha256")
            != provenance.get("coordinator_binding_sha256")
            or row.get("worker_binding_sha256")
            != _execution_binding(
                {
                    key: row.get(key)
                    for key in (
                        "seed",
                        "process_id",
                        "parent_process_id",
                        "process_instance_token",
                        "process_started_utc",
                        "job_started_utc",
                        "coordinator_binding_sha256",
                    )
                }
            )
            or not isinstance(row.get("runtime_seconds"), (int, float))
            or isinstance(row.get("runtime_seconds"), bool)
            or not math.isfinite(float(row["runtime_seconds"]))
            or float(row["runtime_seconds"]) < 0.0
            or not row.get("blas")
            or any(pool.get("num_threads") != 1 for pool in row["blas"])
            for row in worker_runtime
        )
    ):
        raise RuntimeError("full learning report runtime provenance is invalid")
    if provenance["artifact_role"] == "primary" and provenance.get(
        "rerun_primary_artifact_binding"
    ) is not None:
        raise RuntimeError("primary report unexpectedly binds another primary")
    if not _all_finite(report):
        raise FloatingPointError("full learning report contains a non-finite value")


def run_online_trace_learning(
    *,
    seeds: Sequence[int] = REGISTERED_SEEDS,
    cells: Sequence[CellSpec] = FULL_CELLS,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    workers: int = ORDERED_CPU_WORKERS,
    output_path: str | Path,
) -> dict[str, Any]:
    """Run one official full execution after all upstream evidence is verified."""

    with threadpool_limits(limits=1, user_api="blas"):
        parent_blas = _single_thread_blas_state()
        return _run_online_trace_learning(
            seeds=seeds,
            cells=cells,
            ridge_alpha=ridge_alpha,
            workers=workers,
            output_path=output_path,
            parent_blas=parent_blas,
        )


def _run_online_trace_learning(
    *,
    seeds: Sequence[int],
    cells: Sequence[CellSpec],
    ridge_alpha: float,
    workers: int,
    output_path: str | Path,
    parent_blas: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    started = perf_counter()
    run_started_utc = datetime.now(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )
    coordinator: dict[str, Any] = {
        "process_id": os.getpid(),
        "process_instance_token": _PROCESS_INSTANCE_TOKEN,
        "process_started_utc": _PROCESS_STARTED_UTC,
        "run_instance_token": hashlib.sha256(os.urandom(32)).hexdigest(),
        "run_started_utc": run_started_utc,
    }
    coordinator["binding_sha256"] = _execution_binding(coordinator)
    resolved_seeds, resolved_cells, target = _validate_official_arguments(
        seeds=seeds,
        cells=cells,
        ridge_alpha=ridge_alpha,
        workers=workers,
        output_path=output_path,
    )
    validate_protocol_config()
    source_freeze_record = validate_pre_official_source_freeze()
    # These calls are read-only and precede generation or path creation.
    prerequisites = load_verified_learning_prerequisites()
    source = _source_manifest()
    if target.exists() or target.with_suffix(".sha256").exists():
        raise FileExistsError(f"refusing to overwrite official evidence: {target}")
    artifact_role = (
        "primary"
        if target.resolve() == _project_path(PRIMARY_REPORT_PATH).resolve()
        else "rerun"
    )
    primary_binding: dict[str, str] | None = None
    if artifact_role == "rerun":
        primary, primary_files = _read_report_with_sidecar(PRIMARY_REPORT_PATH)
        validate_full_report(
            primary,
            expected_prerequisites=prerequisites,
            current_source_manifest=source,
        )
        primary_binding = {
            "primary_report_file_sha256": primary_files["report_file_sha256"],
            "primary_sidecar_file_sha256": primary_files["sidecar_file_sha256"],
        }
    seed_results, worker_runtime = _ordered_seed_results(
        seeds=resolved_seeds,
        cells=resolved_cells,
        trace_retention=float(prerequisites["trace_retention"]),
        ridge_alpha=ridge_alpha,
        workers=workers,
        coordinator=coordinator,
    )
    source_end = _source_manifest()
    source_freeze_end = validate_pre_official_source_freeze()
    prerequisites_end = load_verified_learning_prerequisites()
    if source_end != source:
        raise RuntimeError("learning source manifest changed during official execution")
    if prerequisites_end != prerequisites:
        raise RuntimeError("learning prerequisites changed during official execution")
    if source_freeze_end != source_freeze_record:
        raise RuntimeError("pre-official source freeze changed during execution")
    report = _assemble_full_report(
        seed_results=seed_results,
        worker_runtime=worker_runtime,
        prerequisites=prerequisites,
        source_manifest=source,
        source_freeze_record=source_freeze_record,
        workers=workers,
        run_started_utc=run_started_utc,
        runtime_seconds=perf_counter() - started,
        artifact_role=artifact_role,
        primary_binding=primary_binding,
        coordinator=coordinator,
    )
    report["nondeterministic_provenance"]["parent_blas"] = list(parent_blas)
    report["nondeterministic_provenance"]["provenance_sha256"] = _record_self_hash(
        report["nondeterministic_provenance"], "provenance_sha256"
    )
    # The excluded provenance was extended; the deterministic hash remains valid.
    if report["deterministic_payload_sha256"] != _recompute_report_payload_sha256(
        report
    ):
        raise RuntimeError("excluded runtime provenance changed deterministic payload")
    validate_full_report(
        report,
        expected_prerequisites=prerequisites,
        current_source_manifest=source,
    )
    if (
        _source_manifest() != source
        or validate_pre_official_source_freeze() != source_freeze_record
        or load_verified_learning_prerequisites() != prerequisites
    ):
        raise RuntimeError("learning inputs changed before official persistence")
    written, sidecar = _write_report_and_sidecar(target, report)
    try:
        reopened, reopened_files = _read_report_with_sidecar(written)
        validate_full_report(
            reopened,
            expected_prerequisites=prerequisites,
            current_source_manifest=source,
        )
        if (
            reopened != report
            or reopened_files["report_file_sha256"] != _file_sha256(written)
            or reopened_files["sidecar_file_sha256"] != _file_sha256(sidecar)
            or _source_manifest() != source
            or validate_pre_official_source_freeze() != source_freeze_record
            or load_verified_learning_prerequisites() != prerequisites
        ):
            raise RuntimeError(
                "official learning evidence or its immutable inputs changed"
            )
    except Exception:
        sidecar.unlink(missing_ok=True)
        written.unlink(missing_ok=True)
        raise
    return report


def run_scratch_smoke(
    *,
    seeds: Sequence[int],
    cells: Sequence[CellSpec],
    trace_retention: float,
    output_path: str | Path,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
) -> dict[str, Any]:
    """Run a reduced, sequential, explicitly nonselecting scratch smoke."""

    resolved_seeds, resolved_cells, target = _validate_scratch_arguments(
        seeds=seeds,
        cells=cells,
        trace_retention=trace_retention,
        ridge_alpha=ridge_alpha,
        output_path=output_path,
    )
    with threadpool_limits(limits=1, user_api="blas"):
        _single_thread_blas_state()
        seed_results = [
            _run_seed(
                seed=seed,
                cells=resolved_cells,
                trace_retention=float(trace_retention),
                ridge_alpha=float(ridge_alpha),
            )
            for seed in resolved_seeds
        ]
    cell_names = tuple(cell.name for cell in resolved_cells)
    aggregate = aggregate_seed_results(seed_results, cell_names=cell_names)
    gates = evaluate_registered_gates(
        aggregate,
        None,
        integrity_pass=all(
            row["all_cell_integrity_pass"] is True for row in seed_results
        ),
        eligible=False,
    )
    report: dict[str, Any] = {
        "schema_version": f"{SCHEMA_VERSION}-scratch",
        "protocol_version": PROTOCOL_VERSION,
        "run_kind": "scratch_smoke",
        "scratch_nonselecting": True,
        "official_seeds_executed": False,
        "seeds": list(resolved_seeds),
        "conditions": list(ORDERED_CONDITIONS),
        "cell_specs": [asdict(cell) for cell in resolved_cells],
        "trace_retention": float(trace_retention),
        "ridge_alpha": float(ridge_alpha),
        "worker_count": 1,
        "ordered_parallel_execution": False,
        "confirmatory_executed": False,
        "confirmatory_seed_count": 0,
        "seed_results": seed_results,
        "aggregate": aggregate,
        "statistics": None,
        "gates": gates,
        "status": "SMOKE_NONSELECTING",
        "readiness_score": READINESS_BEFORE,
    }
    report["deterministic_payload_sha256"] = _sha256_json(report)
    written, sidecar = _write_report_and_sidecar(target, report)
    reopened, hashes = _read_report_with_sidecar(written)
    if (
        reopened != report
        or reopened["status"] != "SMOKE_NONSELECTING"
        or reopened["deterministic_payload_sha256"]
        != _record_self_hash(reopened, "deterministic_payload_sha256")
        or hashes["sidecar_file_sha256"] != _file_sha256(sidecar)
    ):
        raise RuntimeError("scratch smoke report did not reopen exactly")
    return report


def _run_reconstruction_challenge_worker(
    report_path: str | Path, nonce: str
) -> dict[str, Any]:
    """Independently replay one full report in a verifier-created process."""

    if not _is_sha256(nonce):
        raise ValueError("reconstruction challenge nonce must be a SHA-256 token")
    validate_protocol_config()
    freeze = validate_pre_official_source_freeze()
    prerequisites = load_verified_learning_prerequisites()
    source = _source_manifest()
    report, files = _read_report_with_sidecar(report_path)
    validate_full_report(
        report,
        expected_prerequisites=prerequisites,
        current_source_manifest=source,
    )
    report_after, files_after = _read_report_with_sidecar(report_path)
    if (
        report_after != report
        or files_after != files
        or _source_manifest() != source
        or validate_pre_official_source_freeze() != freeze
        or load_verified_learning_prerequisites() != prerequisites
    ):
        raise RuntimeError("reconstruction challenge inputs changed")
    result: dict[str, Any] = {
        "schema_version": CHALLENGE_SCHEMA_VERSION,
        "nonce": nonce,
        "process_id": os.getpid(),
        "process_instance_token": _PROCESS_INSTANCE_TOKEN,
        "process_started_utc": _PROCESS_STARTED_UTC,
        "report_path": str(_project_path(report_path).resolve()),
        "report_file_sha256": files["report_file_sha256"],
        "sidecar_file_sha256": files["sidecar_file_sha256"],
        "deterministic_payload_sha256": report["deterministic_payload_sha256"],
        "reconstructed_evidence_sha256": _sha256_json(
            {
                "seed_results": report["seed_results"],
                "aggregate": report["aggregate"],
                "statistics": report["statistics"],
                "gates": report["gates"],
            }
        ),
        "source_freeze_file_sha256": freeze["freeze_file_sha256"],
        "prerequisite_binding_sha256": prerequisites["binding_sha256"],
    }
    result["challenge_payload_sha256"] = _record_self_hash(
        result, "challenge_payload_sha256"
    )
    return result


def _run_orchestrated_reconstruction_challenges(
    first: str | Path,
    second: str | Path,
    *,
    first_files: Mapping[str, str],
    second_files: Mapping[str, str],
    deterministic_payload_sha256es: tuple[str, str],
) -> dict[str, Any]:
    """Launch two nonce-bound replay subprocesses and verify observed PIDs."""

    paths = (_project_path(first).resolve(), _project_path(second).resolve())
    file_bindings = (dict(first_files), dict(second_files))
    nonces = tuple(hashlib.sha256(os.urandom(32)).hexdigest() for _ in paths)
    environment = os.environ.copy()
    source_root = str(_project_path("src"))
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root
        if not existing_pythonpath
        else os.pathsep.join((source_root, existing_pythonpath))
    )
    command_prefix = (
        sys.executable,
        "-m",
        "adaptive_learning_substrate.experiment_000_online_trace_learning",
        "challenge-replay",
    )
    processes: list[subprocess.Popen[str]] = []
    rows: list[dict[str, Any]] = []
    try:
        for path, nonce in zip(paths, nonces, strict=True):
            processes.append(
                subprocess.Popen(
                    (*command_prefix, "--report", str(path), "--nonce", nonce),
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                )
            )
        for process, path, nonce, binding, expected_payload_sha256 in zip(
            processes,
            paths,
            nonces,
            file_bindings,
            deterministic_payload_sha256es,
            strict=True,
        ):
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                raise RuntimeError(
                    "reconstruction challenge subprocess failed: "
                    + stderr.strip()[-2000:]
                )
            try:
                row = json.loads(stdout)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    "reconstruction challenge emitted malformed JSON"
                ) from error
            if (
                row.get("schema_version") != CHALLENGE_SCHEMA_VERSION
                or row.get("nonce") != nonce
                or row.get("process_id") != process.pid
                or row.get("report_path") != str(path)
                or row.get("report_file_sha256") != binding["report_file_sha256"]
                or row.get("sidecar_file_sha256")
                != binding["sidecar_file_sha256"]
                or row.get("deterministic_payload_sha256")
                != expected_payload_sha256
                or not _is_sha256(row.get("process_instance_token"))
                or not _is_utc_timestamp(row.get("process_started_utc"))
                or row.get("challenge_payload_sha256")
                != _record_self_hash(row, "challenge_payload_sha256")
            ):
                raise RuntimeError("reconstruction challenge binding is invalid")
            rows.append(row)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
    verified = (
        len(rows) == 2
        and rows[0]["process_id"] != rows[1]["process_id"]
        and rows[0]["process_instance_token"]
        != rows[1]["process_instance_token"]
        and rows[0]["process_started_utc"] != rows[1]["process_started_utc"]
        and rows[0]["nonce"] != rows[1]["nonce"]
        and rows[0]["reconstructed_evidence_sha256"]
        == rows[1]["reconstructed_evidence_sha256"]
    )
    evidence: dict[str, Any] = {
        "schema_version": CHALLENGE_SCHEMA_VERSION,
        "verified": verified,
        "verifier_process_id": os.getpid(),
        "verifier_process_instance_token": _PROCESS_INSTANCE_TOKEN,
        "challenges": rows,
    }
    evidence["orchestration_payload_sha256"] = _record_self_hash(
        evidence, "orchestration_payload_sha256"
    )
    return evidence


def _separate_execution_evidence_is_fresh(
    primary: Mapping[str, Any],
    rerun: Mapping[str, Any],
    *,
    expected_primary_binding: Mapping[str, str],
    orchestrated_reconstruction: Mapping[str, Any] | None = None,
) -> bool:
    """Require verifier-observed replays plus consistent archived run records.

    Archived coordinator/worker fields are consistency evidence, not an
    attestation of their historical origin.  The non-forgeable part of this
    gate is the two nonce-bound subprocesses observed by the live verifier.
    """

    try:
        primary_workers = primary["worker_runtime"]
        rerun_workers = rerun["worker_runtime"]
        return bool(
            isinstance(orchestrated_reconstruction, Mapping)
            and orchestrated_reconstruction.get("verified") is True
            and orchestrated_reconstruction.get("orchestration_payload_sha256")
            == _record_self_hash(
                orchestrated_reconstruction, "orchestration_payload_sha256"
            )
            and primary["artifact_role"] == "primary"
            and rerun["artifact_role"] == "rerun"
            and primary["process_instance_token"]
            != rerun["process_instance_token"]
            and primary["run_instance_token"] != rerun["run_instance_token"]
            and primary["process_id"] != rerun["process_id"]
            and primary["process_started_utc"] != rerun["process_started_utc"]
            and primary["run_started_utc"] != rerun["run_started_utc"]
            and primary["rerun_primary_artifact_binding"] is None
            and rerun["rerun_primary_artifact_binding"]
            == dict(expected_primary_binding)
            and {row["process_id"] for row in primary_workers}.isdisjoint(
                row["process_id"] for row in rerun_workers
            )
            and {
                row["process_instance_token"] for row in primary_workers
            }.isdisjoint(row["process_instance_token"] for row in rerun_workers)
            and {row["process_started_utc"] for row in primary_workers}.isdisjoint(
                row["process_started_utc"] for row in rerun_workers
            )
            and {row["job_started_utc"] for row in primary_workers}.isdisjoint(
                row["job_started_utc"] for row in rerun_workers
            )
        )
    except (KeyError, TypeError):
        return False


def verify_deterministic_full_runs(
    first: str | Path = PRIMARY_REPORT_PATH,
    second: str | Path = RERUN_REPORT_PATH,
    *,
    output_path: str | Path = DETERMINISM_VERIFICATION_PATH,
) -> dict[str, Any]:
    """Strictly verify full reports and persist the only terminal learning result."""

    if (
        _project_path(first).resolve() != _project_path(PRIMARY_REPORT_PATH).resolve()
        or _project_path(second).resolve()
        != _project_path(RERUN_REPORT_PATH).resolve()
        or _project_path(output_path).resolve()
        != _project_path(DETERMINISM_VERIFICATION_PATH).resolve()
    ):
        raise ValueError("learning determinism verification requires registered paths")
    if _project_path(first).resolve() == _project_path(second).resolve():
        raise ValueError("primary and rerun reports must be distinct files")
    target = _project_path(output_path)
    if target.exists() or target.with_suffix(".sha256").exists():
        raise FileExistsError(f"refusing to overwrite terminal verification: {target}")
    validate_protocol_config()
    source_freeze = validate_pre_official_source_freeze()
    prerequisites = load_verified_learning_prerequisites()
    source = _source_manifest()
    primary, primary_files = _read_report_with_sidecar(first)
    rerun, rerun_files = _read_report_with_sidecar(second)
    validate_full_report(
        primary,
        expected_prerequisites=prerequisites,
        current_source_manifest=source,
    )
    validate_full_report(
        rerun,
        expected_prerequisites=prerequisites,
        current_source_manifest=source,
    )
    primary_provenance = primary["nondeterministic_provenance"]
    rerun_provenance = rerun["nondeterministic_provenance"]
    expected_primary_binding = {
        "primary_report_file_sha256": primary_files["report_file_sha256"],
        "primary_sidecar_file_sha256": primary_files["sidecar_file_sha256"],
    }
    orchestrated_reconstruction = _run_orchestrated_reconstruction_challenges(
        first,
        second,
        first_files=primary_files,
        second_files=rerun_files,
        deterministic_payload_sha256es=(
            primary["deterministic_payload_sha256"],
            rerun["deterministic_payload_sha256"],
        ),
    )
    payload_hashes_match = (
        primary["deterministic_payload_sha256"]
        == rerun["deterministic_payload_sha256"]
    )
    canonical_payloads_equal = _report_payload_object(
        primary
    ) == _report_payload_object(rerun)
    decisions_match = (
        primary["provisional_terminal_status"]
        == rerun["provisional_terminal_status"]
    )
    fresh_replay_processes = _separate_execution_evidence_is_fresh(
        primary_provenance,
        rerun_provenance,
        expected_primary_binding=expected_primary_binding,
        orchestrated_reconstruction=orchestrated_reconstruction,
    )
    prerequisites_stable = (
        _source_manifest() == source
        and load_verified_learning_prerequisites() == prerequisites
        and primary["prerequisites"] == rerun["prerequisites"] == prerequisites
    )
    determinism_gates = {
        "D01_DETERMINISTIC_PAYLOAD_EQUALITY": payload_hashes_match
        and canonical_payloads_equal,
        "D02_VERIFIER_ORCHESTRATED_REPLAYS": fresh_replay_processes,
        "D03_REPORT_DECISIONS_EQUAL": decisions_match,
        "D04_PREREQUISITES_AND_SOURCE_STABLE": prerequisites_stable,
    }
    scientific = str(primary["provisional_terminal_status"])
    scientific_failures = (
        []
        if scientific == "LEARNING_RESULT_PASS"
        else scientific.removeprefix("LEARNING_RESULT_FAIL:").split(",")
    )
    deterministic_failures = [
        gate_id for gate_id, passed in determinism_gates.items() if not passed
    ]
    failures = [
        gate_id
        for gate_id in (*scientific_failures, *deterministic_failures)
        if gate_id
    ]
    status = (
        "LEARNING_RESULT_PASS"
        if not failures and scientific == "LEARNING_RESULT_PASS"
        else f"LEARNING_RESULT_FAIL:{','.join(failures)}"
    )
    readiness_after = (
        READINESS_AFTER_PASS
        if status == "LEARNING_RESULT_PASS"
        else READINESS_AFTER_FAIL
    )
    verification: dict[str, Any] = {
        "schema_version": "experiment-000-online-trace-learning-determinism-v1",
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "valid_terminal_status": status == "LEARNING_RESULT_PASS"
        or status.startswith("LEARNING_RESULT_FAIL:"),
        "determinism_gates": determinism_gates,
        "failed_gate_ids": failures,
        "deterministic_payload_hashes_match": payload_hashes_match,
        "canonical_deterministic_payloads_equal": canonical_payloads_equal,
        "scientific_decisions_match": decisions_match,
        "fresh_replay_processes_verified": fresh_replay_processes,
        "orchestrated_reconstruction": orchestrated_reconstruction,
        "prerequisites_and_source_stable": prerequisites_stable,
        "first_deterministic_payload_sha256": primary[
            "deterministic_payload_sha256"
        ],
        "second_deterministic_payload_sha256": rerun[
            "deterministic_payload_sha256"
        ],
        "first_report_file_sha256": primary_files["report_file_sha256"],
        "first_sidecar_file_sha256": primary_files["sidecar_file_sha256"],
        "second_report_file_sha256": rerun_files["report_file_sha256"],
        "second_sidecar_file_sha256": rerun_files["sidecar_file_sha256"],
        "first_process_id": primary_provenance["process_id"],
        "second_process_id": rerun_provenance["process_id"],
        "first_process_instance_token": primary_provenance[
            "process_instance_token"
        ],
        "second_process_instance_token": rerun_provenance[
            "process_instance_token"
        ],
        "prerequisite_binding_sha256": prerequisites["binding_sha256"],
        "source_manifest_sha256": source["bundle_sha256"],
        "readiness_score_before": READINESS_BEFORE,
        "readiness_score_after": readiness_after,
        "readiness_delta": readiness_after - READINESS_BEFORE,
    }
    verification["verification_payload_sha256"] = _record_self_hash(
        verification, "verification_payload_sha256"
    )
    if (
        _source_manifest() != source
        or validate_pre_official_source_freeze() != source_freeze
        or load_verified_learning_prerequisites() != prerequisites
    ):
        raise RuntimeError("learning inputs changed before terminal verification write")
    written, sidecar = _write_report_and_sidecar(target, verification)
    try:
        reopened, reopened_files = _read_report_with_sidecar(written)
        primary_after, primary_files_after = _read_report_with_sidecar(first)
        rerun_after, rerun_files_after = _read_report_with_sidecar(second)
        if (
            reopened != verification
            or reopened["verification_payload_sha256"]
            != _record_self_hash(reopened, "verification_payload_sha256")
            or reopened_files["report_file_sha256"] != _file_sha256(written)
            or reopened_files["sidecar_file_sha256"] != _file_sha256(sidecar)
            or primary_after != primary
            or rerun_after != rerun
            or primary_files_after != primary_files
            or rerun_files_after != rerun_files
            or _source_manifest() != source
            or validate_pre_official_source_freeze() != source_freeze
            or load_verified_learning_prerequisites() != prerequisites
        ):
            raise RuntimeError(
                "terminal learning verification or its immutable inputs changed"
            )
    except Exception:
        # A failed post-persistence transaction must not leave a valid-looking
        # terminal decision or binding sidecar behind.
        sidecar.unlink(missing_ok=True)
        written.unlink(missing_ok=True)
        raise
    return verification


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run or verify the preregistered online trace learning gate."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(REGISTERED_SEEDS)
    )
    run_parser.add_argument("--output", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--first", type=Path, required=True)
    verify_parser.add_argument("--second", type=Path, required=True)
    verify_parser.add_argument("--output", type=Path, required=True)
    smoke_parser = subparsers.add_parser("scratch-smoke")
    smoke_parser.add_argument("--seeds", type=int, nargs="+", required=True)
    smoke_parser.add_argument("--trace-retention", type=float, required=True)
    smoke_parser.add_argument("--examples-per-split", type=int, default=2)
    smoke_parser.add_argument("--output", type=Path, required=True)
    challenge_parser = subparsers.add_parser("challenge-replay")
    challenge_parser.add_argument("--report", type=Path, required=True)
    challenge_parser.add_argument("--nonce", type=str, required=True)
    freeze_parser = subparsers.add_parser("freeze-source")
    freeze_parser.add_argument(
        "--output", type=Path, default=CANONICAL_FREEZE_RECORD_PATH
    )
    args = parser.parse_args(argv)
    if args.command == "challenge-replay":
        challenge = _run_reconstruction_challenge_worker(args.report, args.nonce)
        print(json.dumps(challenge, sort_keys=True, separators=(",", ":")))
        return 0
    if args.command == "freeze-source":
        freeze = create_pre_official_source_freeze(args.output)
        summary = {
            "status": freeze["status"],
            "record_sha256": freeze["record_sha256"],
            "freeze_file_sha256": _file_sha256(args.output),
            "external_anchor_environment": FREEZE_ANCHOR_ENV,
            "freeze_path": str(_project_path(args.output)),
        }
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.command == "run":
        report = run_online_trace_learning(
            seeds=tuple(args.seeds),
            output_path=args.output,
        )
        summary = {
            "status": report["status"],
            "provisional_terminal_status": report[
                "provisional_terminal_status"
            ],
            "deterministic_payload_sha256": report[
                "deterministic_payload_sha256"
            ],
            "report_path": str(_project_path(args.output)),
        }
    elif args.command == "verify":
        verification = verify_deterministic_full_runs(
            args.first,
            args.second,
            output_path=args.output,
        )
        summary = {
            "status": verification["status"],
            "readiness_score_after": verification["readiness_score_after"],
            "verification_payload_sha256": verification[
                "verification_payload_sha256"
            ],
            "verification_path": str(_project_path(args.output)),
        }
    else:
        if args.examples_per_split != SMOKE_EXAMPLES_PER_SPLIT:
            parser.error("scratch smoke requires exactly 2 examples per split")
        report = run_scratch_smoke(
            seeds=tuple(args.seeds),
            cells=SMOKE_CELLS,
            trace_retention=args.trace_retention,
            output_path=args.output,
        )
        summary = {
            "status": report["status"],
            "scratch_nonselecting": True,
            "deterministic_payload_sha256": report[
                "deterministic_payload_sha256"
            ],
            "report_path": str(_project_path(args.output)),
        }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "CANONICAL_FREEZE_RECORD_PATH",
    "DETERMINISM_VERIFICATION_PATH",
    "FREEZE_ANCHOR_ENV",
    "FULL_CELLS",
    "ORDERED_CONDITIONS",
    "PRIMARY_REPORT_PATH",
    "PROTOCOL_VERSION",
    "REGISTERED_SEEDS",
    "RERUN_REPORT_PATH",
    "SCHEMA_VERSION",
    "SMOKE_CELLS",
    "CellSpec",
    "NativeDummyTraceState",
    "OnlineTanhHead",
    "aggregate_seed_results",
    "balanced_independent_labels",
    "bootstrap_seed_mean_ci",
    "compute_registered_statistics",
    "create_pre_official_source_freeze",
    "deterministic_random_predictions",
    "evaluate_registered_gates",
    "exact_sign_flip_test",
    "fit_train_only_ridge",
    "holm_adjust",
    "load_verified_learning_prerequisites",
    "normalized_online_update",
    "observe_episode",
    "run_online_trace_learning",
    "run_scratch_smoke",
    "validate_full_report",
    "validate_pre_official_source_freeze",
    "verify_deterministic_full_runs",
]
