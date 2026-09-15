"""Preregistered architecture-only slow local-state experiment.

The native simulator is a frozen external control. Candidate graphs preserve
its exact explicit edges and float64 weights while a hidden unit retains its own
last processed activation with one fixed lazy exponential coefficient. CCF is
never called. Only custom seeds 60--64 and the frozen grid are accepted.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import platform
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .experiment000_data import (
    Experiment000Episode,
    Experiment000Stream,
    SeedRole,
    generate_stream_pair,
    require_seed_role,
)
from .experiment_000 import FROZEN_GRAPH_OPTIONS, forward_episode
from .experiment_000_memory_probe import (
    _PairedSplit,
    extract_query_features,
    paired_retention_metrics,
    recurrent_spectral_radius,
)
from .recurrent import QueryResult, RecurrentEventGraph, build_experiment_000_graph
from .slow_state_recurrent import (
    SlowStateRecurrentEventGraph,
    clone_with_local_state_retention,
)

SLOW_STATE_SCHEMA_VERSION = "experiment-000-slow-local-state-v1-a1"
PROTOCOL_NAME = "experiment_000_slow_local_state"
PROTOCOL_VERSION = "slow-state-v1-a1"
REGISTERED_SEEDS: tuple[int, ...] = (60, 61, 62, 63, 64)
REGISTERED_RETENTIONS: tuple[float, ...] = (0.25, 0.50, 0.75, 0.90, 0.95)
NATIVE_CONDITION = "native"
SMOKE_PAIRS_PER_SPLIT = 2
FULL_PAIRS_PER_SPLIT = 100
DEFAULT_PAIRS_PER_SPLIT = SMOKE_PAIRS_PER_SPLIT
DEFAULT_RIDGE_ALPHA = 1e-3
DEFAULT_OUTPUT_PATH = Path("artifacts/experiment_000/slow_state/report.json")
MATRIX_ABS_TOLERANCE = 1e-12
MATRIX_REL_TOLERANCE = 1e-12
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
    "b33be9b4fffab70cf472720d17b04e710166cb0f719cf8e8d34595476d9cd74b"
)
FROZEN_CONFIG_SHA256 = (
    "dc65068a0dfefc0ffc6dead61e762a4564e4f4639bcd810f2a034172eaf0c26d"
)
_RATIO_EPSILON = 1e-15
BLANK_TAIL_TICKS = 256
BLANK_TAIL_FINAL_WINDOW_TICKS = 64

GATE_THRESHOLDS: dict[str, float] = {
    "eval_cue_to_noise_feature_ratio_minimum": 0.50,
    "paired_native_cue_ratio_multiple_minimum": 3.0,
    "ridge_eval_accuracy_minimum": 0.70,
    "eval_output_cue_delta_to_sd_minimum": 0.50,
    "paired_native_absolute_cue_delta_multiple_minimum": 3.0,
    "within_seed_threefold_improvement_count_minimum": 4.0,
    "event_activity_total_ratio_maximum": 1.10,
    "matched_episode_event_count_ratio_maximum": 1.15,
    "forward_edge_touch_total_ratio_maximum": 1.10,
    "matched_episode_forward_edge_touch_ratio_maximum": 1.15,
    "blank_tail_ticks": float(BLANK_TAIL_TICKS),
    "blank_tail_final_window_ticks": float(BLANK_TAIL_FINAL_WINDOW_TICKS),
    "blank_tail_final_window_hidden_events_maximum": 0.0,
    "effective_state_maximum_absolute": 1e-4,
    "wake_feature_half_difference_l2_maximum": 1e-3,
    "wake_output_half_difference_absolute_maximum": 1e-3,
}
SATURATION_THRESHOLDS: tuple[float, ...] = (0.95, 0.99, 0.999)

_CLONED_HYPERPARAMETERS: tuple[str, ...] = (
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


@dataclass(frozen=True, slots=True)
class _CollectedSplit:
    paired: _PairedSplit
    activity: dict[str, Any]
    pair_rows: tuple[dict[str, Any], ...]


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


def _array_bundle_sha256(*arrays: Sequence[float | int]) -> str:
    hasher = hashlib.sha256()
    for array in arrays:
        normalized = np.asarray(array, dtype="<f8", order="C")
        hasher.update(str(normalized.shape).encode("ascii"))
        hasher.update(b"\0")
        hasher.update(normalized.tobytes(order="C"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _source_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "src/adaptive_learning_substrate/experiment_000_slow_state.py",
        "src/adaptive_learning_substrate/slow_state_recurrent.py",
        "src/adaptive_learning_substrate/experiment_000_memory_probe.py",
        "src/adaptive_learning_substrate/recurrent.py",
        "src/adaptive_learning_substrate/experiment_000.py",
        "src/adaptive_learning_substrate/experiment000_data.py",
        "docs/EXPERIMENT_000_SLOW_STATE_PROTOCOL.md",
        "configs/experiment_000_slow_state.toml",
    )
    files: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"slow-state provenance file is missing: {relative}")
        files[relative] = {"sha256": _file_sha256(path), "bytes": path.stat().st_size}
    return {"files": files, "bundle_sha256": _sha256_json(files)}


def condition_name(retention: float | None) -> str:
    """Return the stable report identifier for one registered condition."""

    if retention is None:
        return NATIVE_CONDITION
    return f"lambda_{retention:.2f}".replace(".", "_")


def _registered_retention(value: float) -> float | None:
    for registered in REGISTERED_RETENTIONS:
        if value == registered:
            return registered
    return None


def _validate_arguments(
    *,
    seeds: Sequence[int],
    retentions: Sequence[float],
    pairs_per_split: int,
    ridge_alpha: float,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    resolved_seeds = tuple(seeds)
    if not resolved_seeds:
        raise ValueError("the exact registered slow-state seed list is required")
    if len(set(resolved_seeds)) != len(resolved_seeds):
        raise ValueError("slow-state seeds must be unique")
    for seed in resolved_seeds:
        require_seed_role(seed, SeedRole.CUSTOM)
        if seed not in REGISTERED_SEEDS:
            raise ValueError(
                f"seed {seed} is custom but not one of the frozen slow-state "
                f"seeds {REGISTERED_SEEDS}"
            )
    if resolved_seeds != REGISTERED_SEEDS:
        raise ValueError(
            "the official smoke and full slow-state experiment require the exact ordered "
            f"seed list {REGISTERED_SEEDS}"
        )

    if not retentions:
        raise ValueError("at least one registered retention is required")
    resolved_retentions: list[float] = []
    for raw_retention in retentions:
        if isinstance(raw_retention, bool):
            raise TypeError("retentions must be finite registered numbers")
        retention = float(raw_retention)
        if not math.isfinite(retention):
            raise ValueError("retentions must be finite registered numbers")
        registered = _registered_retention(retention)
        if registered is None:
            raise ValueError(
                f"retention {retention} is not one of the frozen slow-state "
                f"coefficients {REGISTERED_RETENTIONS}"
            )
        resolved_retentions.append(registered)
    if len(set(resolved_retentions)) != len(resolved_retentions):
        raise ValueError("retentions must be unique")
    ordered_retentions = tuple(
        value for value in REGISTERED_RETENTIONS if value in resolved_retentions
    )
    if tuple(resolved_retentions) != REGISTERED_RETENTIONS:
        raise ValueError(
            "the slow-state experiment requires the exact ordered retention list "
            f"{REGISTERED_RETENTIONS}"
        )

    if (
        isinstance(pairs_per_split, bool)
        or not isinstance(pairs_per_split, int)
        or pairs_per_split not in {SMOKE_PAIRS_PER_SPLIT, FULL_PAIRS_PER_SPLIT}
    ):
        raise ValueError(
            "pairs_per_split must be the frozen smoke value 2 or full value 100"
        )
    if (
        not math.isfinite(ridge_alpha)
        or ridge_alpha != DEFAULT_RIDGE_ALPHA
    ):
        raise ValueError(
            f"ridge_alpha must equal the frozen value {DEFAULT_RIDGE_ALPHA}"
        )
    return resolved_seeds, ordered_retentions


def _within_matrix_tolerance(achieved: float, expected: float) -> bool:
    tolerance = MATRIX_ABS_TOLERANCE + (
        MATRIX_REL_TOLERANCE * max(abs(achieved), abs(expected))
    )
    return abs(achieved - expected) <= tolerance


def _constructor_hyperparameters(graph: RecurrentEventGraph) -> dict[str, Any]:
    return {field: getattr(graph, field) for field in _CLONED_HYPERPARAMETERS}


def _edge_weight_sha256(
    graph: RecurrentEventGraph, *, kinds: Iterable[str]
) -> str:
    selected = set(kinds)
    weights = graph.weights
    rows = [
        edge
        for edge in sorted(graph.edges, key=lambda item: item.edge_id)
        if edge.kind in selected
    ]
    hasher = hashlib.sha256()
    hasher.update(str((len(rows),)).encode("ascii"))
    hasher.update(b"\0")
    for edge in rows:
        hasher.update(edge.edge_id.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(np.asarray([weights[edge.edge_id]], dtype="<f8").tobytes())
        hasher.update(b"\n")
    return hasher.hexdigest()


def _structural_mask_sha256(graph: RecurrentEventGraph) -> str:
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


def _recurrent_matrix_metrics(graph: RecurrentEventGraph) -> dict[str, float]:
    index = {node: position for position, node in enumerate(graph.hidden_nodes)}
    matrix = np.zeros((len(index), len(index)), dtype=np.float64)
    weights = graph.weights
    for edge in graph.edges:
        if edge.kind == "recurrent":
            matrix[index[edge.destination], index[edge.source]] = weights[
                edge.edge_id
            ]
    radius = recurrent_spectral_radius(graph)
    operator_norm = float(np.linalg.norm(matrix, ord=2))
    ratio = _safe_ratio(operator_norm, radius)
    if ratio is None or not math.isfinite(operator_norm):
        raise FloatingPointError("invalid recurrent matrix norm")
    return {
        "spectral_radius": radius,
        "operator_2_norm": operator_norm,
        "nonnormality_ratio": ratio,
        "operator_2_norm_to_spectral_radius": ratio,
    }


def _credit_method_bundle_sha256(source_path: Path) -> str:
    """Reproduce the A1 byte-exact CCF method-source bundle hash."""

    text = source_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    recurrent_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RecurrentEventGraph"
    )
    names = ("_credit_target", "_sign", "_record_packet", "apply_supervised_credit")
    segments: list[str] = []
    for name in names:
        method = next(
            node
            for node in recurrent_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        )
        first_line = min(
            [method.lineno, *(decorator.lineno for decorator in method.decorator_list)]
        )
        if method.end_lineno is None:  # pragma: no cover - Python parser invariant
            raise RuntimeError(f"AST end line missing for {name}")
        segments.append("".join(lines[first_line - 1 : method.end_lineno]))
    payload = "\n---METHOD---\n".join(segments).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def frozen_source_integrity() -> dict[str, Any]:
    """Verify the native recurrent control, CCF methods, and frozen CCF document."""

    root = Path(__file__).resolve().parents[2]
    recurrent_path = root / "src/adaptive_learning_substrate/recurrent.py"
    ccf_path = root / "docs/CCF_V0.md"
    protocol_path = root / "docs/EXPERIMENT_000_SLOW_STATE_PROTOCOL.md"
    config_path = root / "configs/experiment_000_slow_state.toml"
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


def _query_result_from_forced_event(graph: RecurrentEventGraph) -> QueryResult:
    event_id = graph.audit["latest_output_event_id"]
    if event_id is None:
        raise RuntimeError("manual equivalence query did not force an output")
    assert isinstance(event_id, str)
    event = graph.unit_events_by_id[event_id]
    return QueryResult(
        tick=event.step,
        event_id=event.event_id,
        activation=event.activation,
        prediction=1 if event.activation >= 0.0 else 0,
    )


def run_zero_retention_equivalence() -> dict[str, Any]:
    """Compare native and λ=0 at every smoke-example forward step."""

    rows: list[dict[str, Any]] = []
    step_count = 0
    public_episode_count = 0
    for seed in REGISTERED_SEEDS:
        pair = generate_stream_pair(
            seed, episode_count=SMOKE_PAIRS_PER_SPLIT, noise_events=8,
            required_role=SeedRole.CUSTOM,
        )
        native = build_experiment_000_graph(
            seed, mode="full", event_log_enabled=False, **FROZEN_GRAPH_OPTIONS
        )
        zero = clone_with_local_state_retention(
            native, 0.0, event_log_enabled=False
        )
        for stream in (pair.train, pair.eval):
            for base in stream.episodes:
                for cue in (0, 1):
                    episode = _counterfactual_episode(
                        base, cue=cue, condition="lambda_0_equivalence"
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
                            raise RuntimeError("λ=0 differs from native at a visible step")
                        step_count += 1
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
                            raise RuntimeError("λ=0 differs inside the query path")
                        step_count += 1
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
                        raise RuntimeError("λ=0 manual query/event/ledger mismatch")

                    # Independently exercise the same public path used officially.
                    native_public = forward_episode(native, episode)
                    zero_public = forward_episode(zero, episode)
                    if (
                        native_public != zero_public
                        or extract_query_features(native, native_public)
                        != extract_query_features(zero, zero_public)
                        or native.unit_events != zero.unit_events
                        or native.ledger != zero.ledger
                    ):
                        raise RuntimeError("λ=0 public forward path differs from native")
                    public_episode_count += 1
                    state_ledger = zero.slow_state_ledger
                    state_touch_fields = (
                        "local_state_read_touches",
                        "local_state_decay_touches",
                        "local_state_write_touches",
                        "local_state_reset_touches",
                        "local_state_observation_touches",
                    )
                    if any(state_ledger[field] != 0 for field in state_touch_fields):
                        raise RuntimeError("λ=0 equivalence path touched local state")
                    rows.append(
                        {
                            "seed": seed,
                            "split": stream.split,
                            "base_episode_id": base.episode_id,
                            "cue": cue,
                            "query_activation": native_public.activation,
                            "feature_sha256": _array_bundle_sha256(
                                [value for _, value in native_features]
                            ),
                            "legacy_ledger_sha256": _sha256_json(native.ledger),
                        }
                    )
    return {
        "pass": True,
        "seeds": list(REGISTERED_SEEDS),
        "base_pairs_per_split": SMOKE_PAIRS_PER_SPLIT,
        "counterfactual_episode_count": len(rows),
        "compared_step_result_count": step_count,
        "compared_public_episode_count": public_episode_count,
        "step_results_bitwise_equal": True,
        "episode_id_normalization": "identity_because_both_graphs_use_same_episode_id",
        "intermediate_activation_tuples_bitwise_equal": True,
        "emitted_event_ids_equal": True,
        "query_results_bitwise_equal": True,
        "features_bitwise_equal": True,
        "unit_events_bitwise_equal": True,
        "legacy_additive_ledgers_equal": True,
        "all_local_state_touches_zero": True,
        "rows_sha256": _sha256_json(rows),
    }


def _counterfactual_episode(
    base: Experiment000Episode, *, cue: int, condition: str
) -> Experiment000Episode:
    return replace(
        base,
        episode_id=f"{base.episode_id}-slow-state-{condition}-cue-{cue}",
        cue=cue,
        target=cue,
    )


def _collect_split(
    graph: RecurrentEventGraph,
    stream: Experiment000Stream,
    *,
    condition: str,
    feature_edge_ids: tuple[str, ...],
    native_diagnostic_graph: SlowStateRecurrentEventGraph | None = None,
) -> _CollectedSplit:
    cue_zero_features: list[np.ndarray] = []
    cue_one_features: list[np.ndarray] = []
    cue_zero_outputs: list[float] = []
    cue_one_outputs: list[float] = []
    event_counts: list[int] = []
    hidden_event_counts: list[int] = []
    hidden_activation_counts: list[int] = []
    saturation_counts = {threshold: 0 for threshold in SATURATION_THRESHOLDS}
    maximum_absolute_preactivation = 0.0
    pair_rows: list[dict[str, Any]] = []
    ledger_before = graph.ledger
    state_ledger_before = (
        graph.slow_state_ledger
        if isinstance(graph, SlowStateRecurrentEventGraph)
        else {key: 0 for key in (
            "hidden_activation_evaluations",
            "local_state_read_touches",
            "local_state_decay_touches",
            "local_state_write_touches",
            "local_state_reset_touches",
            "local_state_observation_touches",
        )}
    )

    for base in stream.episodes:
        row: dict[str, Any] = {
            "pair_index": int(base.index),
            "base_episode_id": base.episode_id,
            "noise_stream_id": base.noise_stream_id,
        }
        for cue in (0, 1):
            episode = _counterfactual_episode(base, cue=cue, condition=condition)
            episode_ledger_before = graph.ledger
            episode_state_before = (
                graph.slow_state_ledger
                if isinstance(graph, SlowStateRecurrentEventGraph)
                else {key: 0 for key in state_ledger_before}
            )
            query = forward_episode(graph, episode)
            episode_ledger_after = graph.ledger
            extracted = extract_query_features(graph, query)
            if tuple(edge_id for edge_id, _ in extracted) != feature_edge_ids:
                raise RuntimeError("query feature order changed during slow-state experiment")
            diagnostics_graph: SlowStateRecurrentEventGraph
            if native_diagnostic_graph is not None:
                shadow_query = forward_episode(native_diagnostic_graph, episode)
                shadow_extracted = extract_query_features(
                    native_diagnostic_graph, shadow_query
                )
                if query != shadow_query:
                    raise RuntimeError("zero-retention query differs from native")
                if graph.unit_events != native_diagnostic_graph.unit_events:
                    raise RuntimeError("zero-retention events differ from native")
                if graph.ledger != native_diagnostic_graph.ledger:
                    raise RuntimeError("zero-retention legacy ledger differs from native")
                if extracted != shadow_extracted:
                    raise RuntimeError("zero-retention features differ from native")
                if any(native_diagnostic_graph.slow_state_ledger[key] != 0 for key in (
                    "local_state_read_touches",
                    "local_state_decay_touches",
                    "local_state_write_touches",
                    "local_state_reset_touches",
                    "local_state_observation_touches",
                )):
                    raise RuntimeError("zero-retention fast path touched local state")
                diagnostics_graph = native_diagnostic_graph
            elif isinstance(graph, SlowStateRecurrentEventGraph):
                diagnostics_graph = graph
            else:  # pragma: no cover - native always receives its λ=0 observer
                raise RuntimeError("native condition lacks its diagnostic observer")
            feature = np.asarray([value for _, value in extracted], dtype=np.float64)
            output = float(query.activation)
            if not np.all(np.isfinite(feature)) or not math.isfinite(output):
                raise FloatingPointError("non-finite slow-state forward value")
            if any(
                not math.isfinite(float(event.preactivation))
                or not math.isfinite(float(event.activation))
                or abs(float(event.activation)) > 1.0 + 1e-15
                for event in graph.unit_events
            ):
                raise FloatingPointError("event left the finite tanh/source range")
            event_count = len(graph.unit_events)
            hidden_event_count = sum(
                event.node in graph.hidden_nodes for event in graph.unit_events
            )
            activation_records = diagnostics_graph.slow_activation_records
            hidden_activation_count = len(activation_records)
            if hidden_activation_count <= 0:
                raise RuntimeError("episode produced no hidden activation evaluations")
            if any(
                not all(
                    math.isfinite(value)
                    for value in (
                        record.incoming_drive,
                        record.retained_state,
                        record.preactivation,
                        record.activation,
                    )
                )
                or abs(record.activation) > 1.0
                or abs(record.retained_state) > 1.0
                for record in activation_records
            ):
                raise FloatingPointError("hidden activation diagnostic is invalid")
            episode_max_preactivation = max(
                abs(record.preactivation) for record in activation_records
            )
            maximum_absolute_preactivation = max(
                maximum_absolute_preactivation, episode_max_preactivation
            )
            episode_saturation = {
                threshold: sum(
                    abs(record.activation) >= threshold
                    for record in activation_records
                )
                for threshold in SATURATION_THRESHOLDS
            }
            for threshold, count in episode_saturation.items():
                saturation_counts[threshold] += count
            event_counts.append(event_count)
            hidden_event_counts.append(hidden_event_count)
            hidden_activation_counts.append(hidden_activation_count)
            row[f"cue_{cue}_output_activation"] = output
            row[f"cue_{cue}_emitted_unit_events"] = event_count
            row[f"cue_{cue}_hidden_unit_events"] = hidden_event_count
            row[f"cue_{cue}_hidden_emission_density"] = float(
                hidden_event_count
                / (len(graph.hidden_nodes) * (stream.noise_events + 3))
            )
            row[f"cue_{cue}_forward_edge_touches"] = int(
                episode_ledger_after["forward_edge_touches"]
                - episode_ledger_before["forward_edge_touches"]
            )
            row[f"cue_{cue}_activation_evaluations"] = int(
                episode_ledger_after["activation_evaluations"]
                - episode_ledger_before["activation_evaluations"]
            )
            row[f"cue_{cue}_hidden_activation_evaluations"] = (
                hidden_activation_count
            )
            episode_state_after = (
                graph.slow_state_ledger
                if isinstance(graph, SlowStateRecurrentEventGraph)
                else episode_state_before
            )
            for field in (
                "local_state_read_touches",
                "local_state_decay_touches",
                "local_state_write_touches",
                "local_state_reset_touches",
                "local_state_observation_touches",
            ):
                row[f"cue_{cue}_{field}"] = int(
                    episode_state_after[field] - episode_state_before[field]
                )
            row[f"cue_{cue}_maximum_absolute_hidden_preactivation"] = float(
                episode_max_preactivation
            )
            row[f"cue_{cue}_hidden_saturation_fractions"] = {
                f"abs_ge_{threshold}": float(
                    episode_saturation[threshold] / hidden_activation_count
                )
                for threshold in SATURATION_THRESHOLDS
            }
            if cue == 0:
                cue_zero_features.append(feature)
                cue_zero_outputs.append(output)
            else:
                cue_one_features.append(feature)
                cue_one_outputs.append(output)
        pair_rows.append(row)

    ledger_after = graph.ledger
    state_ledger_after = (
        graph.slow_state_ledger
        if isinstance(graph, SlowStateRecurrentEventGraph)
        else state_ledger_before
    )
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
        raise RuntimeError("no-learning slow-state experiment wrote graph weights")
    if ledger_after["credit_event_touches"] != ledger_before["credit_event_touches"]:
        raise RuntimeError("no-learning slow-state experiment processed credit")

    paired = _PairedSplit(
        cue_zero_features=np.stack(cue_zero_features),
        cue_one_features=np.stack(cue_one_features),
        cue_zero_outputs=np.asarray(cue_zero_outputs, dtype=np.float64),
        cue_one_outputs=np.asarray(cue_one_outputs, dtype=np.float64),
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
        "pair_rows_sha256": _sha256_json(pair_rows),
        "forward_edge_touches": int(
            ledger_after["forward_edge_touches"] - ledger_before["forward_edge_touches"]
        ),
        "total_hidden_unit_events": int(sum(hidden_event_counts)),
        "mean_hidden_unit_events_per_forward_episode": float(
            np.mean(hidden_event_counts)
        ),
        "maximum_hidden_unit_events_per_forward_episode": max(
            hidden_event_counts
        ),
        "hidden_event_fraction": float(sum(hidden_event_counts) / total_events),
        "hidden_emission_opportunities": hidden_opportunities,
        "hidden_emission_density": float(
            sum(hidden_event_counts) / hidden_opportunities
        ),
        "activation_evaluations": int(
            ledger_after["activation_evaluations"]
            - ledger_before["activation_evaluations"]
        ),
        "hidden_activation_evaluations": int(sum(hidden_activation_counts)),
        "hidden_activation_counts_sha256": _array_bundle_sha256(
            hidden_activation_counts
        ),
        "maximum_absolute_hidden_preactivation": float(
            maximum_absolute_preactivation
        ),
        "hidden_activation_saturation": {
            f"abs_ge_{threshold}": {
                "count": int(saturation_counts[threshold]),
                "fraction": float(
                    saturation_counts[threshold] / sum(hidden_activation_counts)
                ),
            }
            for threshold in SATURATION_THRESHOLDS
        },
        "local_state_touches": {
            field: int(state_ledger_after[field] - state_ledger_before[field])
            for field in (
                "local_state_read_touches",
                "local_state_decay_touches",
                "local_state_write_touches",
                "local_state_reset_touches",
                "local_state_observation_touches",
            )
        },
        "nonfinite_values": int(
            ledger_after["nonfinite_values"] - ledger_before["nonfinite_values"]
        ),
        "credit_event_touches": int(
            ledger_after["credit_event_touches"]
            - ledger_before["credit_event_touches"]
        ),
        "weight_write_touches": int(
            ledger_after["weight_write_touches"]
            - ledger_before["weight_write_touches"]
        ),
    }
    return _CollectedSplit(
        paired=paired, activity=activity, pair_rows=tuple(pair_rows)
    )


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
        "nonfinite_values": int(train["nonfinite_values"])
        + int(evaluation["nonfinite_values"]),
        "credit_event_touches": int(train["credit_event_touches"])
        + int(evaluation["credit_event_touches"]),
        "weight_write_touches": int(train["weight_write_touches"])
        + int(evaluation["weight_write_touches"]),
        "split_event_counts_sha256": _sha256_json(
            [train["event_counts_sha256"], evaluation["event_counts_sha256"]]
        ),
        "total_hidden_unit_events": total_hidden,
        "mean_hidden_unit_events_per_forward_episode": float(
            total_hidden / episodes
        ),
        "maximum_hidden_unit_events_per_forward_episode": max(
            int(train["maximum_hidden_unit_events_per_forward_episode"]),
            int(evaluation["maximum_hidden_unit_events_per_forward_episode"]),
        ),
        "hidden_event_fraction": float(total_hidden / total),
        "hidden_emission_opportunities": hidden_opportunities,
        "hidden_emission_density": float(total_hidden / hidden_opportunities),
        "maximum_absolute_hidden_preactivation": max(
            float(train["maximum_absolute_hidden_preactivation"]),
            float(evaluation["maximum_absolute_hidden_preactivation"]),
        ),
        "hidden_activation_saturation": {
            f"abs_ge_{threshold}": {
                "count": int(
                    train["hidden_activation_saturation"][f"abs_ge_{threshold}"][
                        "count"
                    ]
                )
                + int(
                    evaluation["hidden_activation_saturation"][
                        f"abs_ge_{threshold}"
                    ]["count"]
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
        "local_state_touches": {
            field: int(train["local_state_touches"][field])
            + int(evaluation["local_state_touches"][field])
            for field in (
                "local_state_read_touches",
                "local_state_decay_touches",
                "local_state_write_touches",
                "local_state_reset_touches",
                "local_state_observation_touches",
            )
        },
    }


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) <= _RATIO_EPSILON:
        return None
    value = float(numerator) / float(denominator)
    if not math.isfinite(value):
        raise FloatingPointError("non-finite paired ratio")
    return value


def _paired_difference(candidate: float | None, native: float | None) -> float | None:
    if candidate is None or native is None:
        return None
    value = float(candidate) - float(native)
    if not math.isfinite(value):
        raise FloatingPointError("non-finite paired difference")
    return value


def _ridge_rows(split: _PairedSplit) -> tuple[np.ndarray, np.ndarray]:
    feature_count = split.cue_zero_features.shape[1]
    features = np.stack(
        (split.cue_zero_features, split.cue_one_features), axis=1
    ).reshape(-1, feature_count)
    targets = np.tile(np.asarray((-1.0, 1.0), dtype=np.float64), split.pair_count)
    return features, targets


def strict_ridge_readout(
    train: _PairedSplit, evaluation: _PairedSplit, *, alpha: float
) -> dict[str, Any]:
    """Frozen ridge diagnostic with inactive coordinates excluded exactly."""

    train_x, train_y = _ridge_rows(train)
    eval_x, eval_y = _ridge_rows(evaluation)
    train_mean = np.mean(train_x, axis=0)
    train_scale = np.std(train_x, axis=0, ddof=0)
    active = train_scale > _RATIO_EPSILON
    active_indices = np.flatnonzero(active)
    coefficients = np.zeros(train_x.shape[1], dtype=np.float64)
    intercept = float(np.mean(train_y))
    if active_indices.size:
        normalized_train = (
            train_x[:, active_indices] - train_mean[active_indices]
        ) / train_scale[active_indices]
        normalized_eval = (
            eval_x[:, active_indices] - train_mean[active_indices]
        ) / train_scale[active_indices]
        sample_count = train_x.shape[0]
        gram = (normalized_train.T @ normalized_train) / sample_count
        gram += float(alpha) * np.eye(gram.shape[0], dtype=np.float64)
        right_hand_side = (normalized_train.T @ train_y) / sample_count
        active_coefficients = np.linalg.solve(gram, right_hand_side)
        coefficients[active_indices] = active_coefficients
        train_scores = normalized_train @ active_coefficients + intercept
        eval_scores = normalized_eval @ active_coefficients + intercept
    else:
        train_scores = np.full(train_y.shape, intercept, dtype=np.float64)
        eval_scores = np.full(eval_y.shape, intercept, dtype=np.float64)
    if not np.all(coefficients[~active] == 0.0):
        raise RuntimeError("inactive ridge coordinates are not exact zero")
    train_predictions = np.where(train_scores >= 0.0, 1.0, -1.0)
    eval_predictions = np.where(eval_scores >= 0.0, 1.0, -1.0)
    train_accuracy = float(np.mean(train_predictions == train_y))
    eval_accuracy = float(np.mean(eval_predictions == eval_y))
    if not all(
        math.isfinite(value)
        for value in (
            train_accuracy,
            eval_accuracy,
            float(np.linalg.norm(coefficients)),
        )
    ):
        raise FloatingPointError("non-finite strict ridge result")
    return {
        "alpha": float(alpha),
        "target_encoding": {"cue_0": -1.0, "cue_1": 1.0},
        "feature_standardization": "train_mean_and_population_sd_only",
        "zero_variance_threshold": _RATIO_EPSILON,
        "inactive_coordinates_standardized_to_zero": True,
        "inactive_coordinates_excluded_from_solve": True,
        "inactive_embedded_coefficients_exact_zero": True,
        "score_tie_prediction": "cue_1_when_score_greater_than_or_equal_to_zero",
        "active_feature_count": int(active_indices.size),
        "inactive_feature_count": int(np.count_nonzero(~active)),
        "active_feature_indices": active_indices.tolist(),
        "active_mask_sha256": _array_bundle_sha256(active.astype(np.float64)),
        "embedded_coefficients": coefficients.tolist(),
        "embedded_coefficients_sha256": _array_bundle_sha256(coefficients),
        "feature_count": int(train_x.shape[1]),
        "train_examples": int(train_x.shape[0]),
        "eval_examples": int(eval_x.shape[0]),
        "train_accuracy": train_accuracy,
        "eval_accuracy": eval_accuracy,
        "coefficient_l2": float(np.linalg.norm(coefficients)),
    }


def _blank_tail_stability(
    graph: RecurrentEventGraph, *, condition: str
) -> dict[str, Any]:
    """Run the frozen both-polarity blank-tail, state, and wake assay."""

    ledger_before = graph.ledger
    state_before = (
        graph.slow_state_ledger
        if isinstance(graph, SlowStateRecurrentEventGraph)
        else None
    )
    weights_before = graph.weights_hash()
    structural_before = _structural_mask_sha256(graph)
    hidden_nodes = set(graph.hidden_nodes)
    polarity_rows: list[dict[str, Any]] = []
    for cue_input in (-1.0, 1.0):
        polarity_ledger_before = graph.ledger
        polarity_state_before = (
            graph.slow_state_ledger
            if isinstance(graph, SlowStateRecurrentEventGraph)
            else None
        )
        polarity_weights_before = graph.weights_hash()
        polarity_topology_before = graph.topology_hash()
        polarity_structural_before = _structural_mask_sha256(graph)
        cue_label = "negative" if cue_input < 0.0 else "positive"
        graph.begin_episode(f"slow-state-blank-tail-{condition}-{cue_label}")
        graph.step({"cue": cue_input})
        hidden_counts: list[int] = []
        total_counts: list[int] = []
        for _ in range(BLANK_TAIL_TICKS):
            result = graph.step({})
            emitted_nodes = tuple(
                event_id.rsplit(":", 2)[1] for event_id in result.emitted_event_ids
            )
            hidden_counts.append(sum(node in hidden_nodes for node in emitted_nodes))
            total_counts.append(len(emitted_nodes))

        state_snapshot_before = (
            graph.local_state_snapshot
            if isinstance(graph, SlowStateRecurrentEventGraph)
            else None
        )
        state_observation_ledger_before = (
            graph.slow_state_ledger
            if isinstance(graph, SlowStateRecurrentEventGraph)
            else None
        )
        effective_states = (
            graph.effective_local_states(BLANK_TAIL_TICKS)
            if isinstance(graph, SlowStateRecurrentEventGraph)
            else {node: 0.0 for node in graph.hidden_nodes}
        )
        effective_state_maximum = max(abs(value) for value in effective_states.values())
        state_snapshot_after = (
            graph.local_state_snapshot
            if isinstance(graph, SlowStateRecurrentEventGraph)
            else None
        )
        state_observation_ledger_after = (
            graph.slow_state_ledger
            if isinstance(graph, SlowStateRecurrentEventGraph)
            else None
        )
        observation_touches = (
            0
            if state_observation_ledger_before is None
            else int(
                state_observation_ledger_after["local_state_observation_touches"]
                - state_observation_ledger_before["local_state_observation_touches"]
            )
        )
        if state_snapshot_before != state_snapshot_after:
            raise RuntimeError("effective-state observation mutated local state")

        wake_query = graph.query()
        wake_extracted = extract_query_features(graph, wake_query)
        wake_feature = [float(value) for _, value in wake_extracted]
        polarity_ledger_after = graph.ledger
        polarity_state_after = (
            graph.slow_state_ledger
            if isinstance(graph, SlowStateRecurrentEventGraph)
            else None
        )
        polarity_weights_after = graph.weights_hash()
        polarity_topology_after = graph.topology_hash()
        polarity_structural_after = _structural_mask_sha256(graph)
        last_nonzero = next(
            (
                index
                for index in range(BLANK_TAIL_TICKS, 0, -1)
                if hidden_counts[index - 1] > 0
            ),
            None,
        )
        final_window = hidden_counts[-BLANK_TAIL_FINAL_WINDOW_TICKS:]
        state_touch_deltas = {
            field: 0
            if polarity_state_before is None
            else int(polarity_state_after[field] - polarity_state_before[field])
            for field in (
                "hidden_activation_evaluations",
                "local_state_read_touches",
                "local_state_decay_touches",
                "local_state_write_touches",
                "local_state_reset_touches",
                "local_state_observation_touches",
            )
        }
        row = {
            "cue_input": cue_input,
            "blank_ticks": BLANK_TAIL_TICKS,
            "final_window_ticks": BLANK_TAIL_FINAL_WINDOW_TICKS,
            "hidden_emissions_per_blank_tick": hidden_counts,
            "total_emissions_per_blank_tick": total_counts,
            "hidden_counts_sha256": _array_bundle_sha256(hidden_counts),
            "total_counts_sha256": _array_bundle_sha256(total_counts),
            "total_hidden_emissions": int(sum(hidden_counts)),
            "hidden_emission_density": float(
                sum(hidden_counts) / (BLANK_TAIL_TICKS * len(graph.hidden_nodes))
            ),
            "maximum_hidden_emissions_in_one_tick": max(hidden_counts),
            "last_nonzero_hidden_blank_tick": last_nonzero,
            "final_window_hidden_emissions": int(sum(final_window)),
            "final_window_quiescent": sum(final_window) == 0,
            "effective_state_observation_tick": BLANK_TAIL_TICKS,
            "effective_state_maximum_absolute": float(effective_state_maximum),
            "effective_state_sha256": _array_bundle_sha256(
                [effective_states[node] for node in graph.hidden_nodes]
            ),
            "effective_state_gate_pass": effective_state_maximum
            <= GATE_THRESHOLDS["effective_state_maximum_absolute"],
            "observation_touches": observation_touches,
            "observation_nonmutating": state_snapshot_before == state_snapshot_after,
            "wake_feature_edge_ids": [edge_id for edge_id, _ in wake_extracted],
            "wake_feature": wake_feature,
            "wake_feature_sha256": _array_bundle_sha256(wake_feature),
            "wake_output_activation": float(wake_query.activation),
            "forward_edge_touches": int(
                polarity_ledger_after["forward_edge_touches"]
                - polarity_ledger_before["forward_edge_touches"]
            ),
            "state_touch_deltas": state_touch_deltas,
            "weights_before_sha256": polarity_weights_before,
            "weights_after_sha256": polarity_weights_after,
            "weights_unchanged": polarity_weights_before == polarity_weights_after,
            "topology_before_sha256": polarity_topology_before,
            "topology_after_sha256": polarity_topology_after,
            "topology_unchanged": polarity_topology_before == polarity_topology_after,
            "structural_before_sha256": polarity_structural_before,
            "structural_after_sha256": polarity_structural_after,
            "structural_unchanged": polarity_structural_before
            == polarity_structural_after,
        }
        polarity_rows.append(row)
        if not row["weights_unchanged"] or not row["topology_unchanged"] or not row["structural_unchanged"]:
            raise RuntimeError("one tail/wake polarity changed frozen graph state")

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
    state_after = (
        graph.slow_state_ledger
        if isinstance(graph, SlowStateRecurrentEventGraph)
        else None
    )
    weights_after = graph.weights_hash()
    if weights_before != weights_after:
        raise RuntimeError("tail/wake assay changed graph weights")
    if ledger_after["weight_write_touches"] != ledger_before["weight_write_touches"]:
        raise RuntimeError("tail/wake assay wrote graph weights")
    if ledger_after["credit_event_touches"] != ledger_before["credit_event_touches"]:
        raise RuntimeError("tail/wake assay processed credit")
    total_hidden = sum(row["total_hidden_emissions"] for row in polarity_rows)
    return {
        "cue_polarities": polarity_rows,
        "polarity_bundle_sha256": _sha256_json(polarity_rows),
        "all_polarities_final_window_quiescent": all(
            row["final_window_quiescent"] for row in polarity_rows
        ),
        "all_polarities_effective_state_pass": all(
            row["effective_state_gate_pass"] for row in polarity_rows
        ),
        "wake_feature_half_difference_l2": wake_feature_half_l2,
        "wake_feature_gate_pass": wake_feature_half_l2
        <= GATE_THRESHOLDS["wake_feature_half_difference_l2_maximum"],
        "wake_output_half_difference_absolute": wake_output_half_abs,
        "wake_output_gate_pass": wake_output_half_abs
        <= GATE_THRESHOLDS["wake_output_half_difference_absolute_maximum"],
        "all_tail_state_wake_gates_pass": all(
            row["final_window_quiescent"] and row["effective_state_gate_pass"]
            for row in polarity_rows
        ) and wake_feature_half_l2
        <= GATE_THRESHOLDS["wake_feature_half_difference_l2_maximum"]
        and wake_output_half_abs
        <= GATE_THRESHOLDS["wake_output_half_difference_absolute_maximum"],
        "total_hidden_emissions": int(total_hidden),
        "hidden_emission_density": float(
            total_hidden / (2 * BLANK_TAIL_TICKS * len(graph.hidden_nodes))
        ),
        "forward_edge_touches": int(
            ledger_after["forward_edge_touches"] - ledger_before["forward_edge_touches"]
        ),
        "state_touch_deltas": {
            field: 0 if state_before is None else int(state_after[field] - state_before[field])
            for field in (
                "hidden_activation_evaluations",
                "local_state_read_touches",
                "local_state_decay_touches",
                "local_state_write_touches",
                "local_state_reset_touches",
                "local_state_observation_touches",
            )
        },
        "weights_unchanged": weights_before == weights_after,
        "structural_unchanged": structural_before == _structural_mask_sha256(graph),
        "credit_event_touches": int(
            ledger_after["credit_event_touches"] - ledger_before["credit_event_touches"]
        ),
        "weight_write_touches": int(
            ledger_after["weight_write_touches"] - ledger_before["weight_write_touches"]
        ),
        "nonfinite_values": int(
            ledger_after["nonfinite_values"] - ledger_before["nonfinite_values"]
        ),
    }


def _condition_graph(
    native_graph: RecurrentEventGraph, retention: float | None
) -> RecurrentEventGraph:
    if retention is None:
        return native_graph
    return clone_with_local_state_retention(
        native_graph, retention, event_log_enabled=False
    )


def _run_condition(
    *,
    native_graph: RecurrentEventGraph,
    retention: float | None,
    stream_pair: Any,
    ridge_alpha: float,
) -> dict[str, Any]:
    condition = condition_name(retention)
    graph = _condition_graph(native_graph, retention)
    native_matrix_metrics = _recurrent_matrix_metrics(native_graph)
    recurrent_matrix_metrics = _recurrent_matrix_metrics(graph)
    topology_before = graph.topology_hash()
    structural_mask_before = _structural_mask_sha256(graph)
    weights_before = graph.weights_hash()
    framed_weights_before = _edge_weight_sha256(
        graph, kinds=("input", "recurrent", "output")
    )
    recurrent_weights_before = _edge_weight_sha256(graph, kinds=("recurrent",))
    input_weights_before = _edge_weight_sha256(graph, kinds=("input",))
    output_weights_before = _edge_weight_sha256(graph, kinds=("output",))
    native_weights_hash = native_graph.weights_hash()
    native_framed_weights_hash = _edge_weight_sha256(
        native_graph, kinds=("input", "recurrent", "output")
    )
    native_recurrent_hash = _edge_weight_sha256(native_graph, kinds=("recurrent",))
    native_input_hash = _edge_weight_sha256(native_graph, kinds=("input",))
    native_output_hash = _edge_weight_sha256(native_graph, kinds=("output",))
    feature_edge_ids = tuple(
        sorted(
            edge.edge_id
            for edge in graph.edges
            if edge.kind == "output" and edge.destination == graph.output_node
        )
    )
    native_diagnostic_graph = (
        clone_with_local_state_retention(
            native_graph, 0.0, event_log_enabled=False
        )
        if retention is None
        else None
    )
    train = _collect_split(
        graph,
        stream_pair.train,
        condition=condition,
        feature_edge_ids=feature_edge_ids,
        native_diagnostic_graph=native_diagnostic_graph,
    )
    weights_after_train = graph.weights_hash()
    framed_weights_after_train = _edge_weight_sha256(
        graph, kinds=("input", "recurrent", "output")
    )
    topology_after_train = graph.topology_hash()
    structural_mask_after_train = _structural_mask_sha256(graph)
    recurrent_weights_after_train = _edge_weight_sha256(
        graph, kinds=("recurrent",)
    )
    input_weights_after_train = _edge_weight_sha256(graph, kinds=("input",))
    output_weights_after_train = _edge_weight_sha256(graph, kinds=("output",))
    evaluation = _collect_split(
        graph,
        stream_pair.eval,
        condition=condition,
        feature_edge_ids=feature_edge_ids,
        native_diagnostic_graph=native_diagnostic_graph,
    )
    weights_after_eval = graph.weights_hash()
    topology_after_eval = graph.topology_hash()
    structural_mask_after_eval = _structural_mask_sha256(graph)
    framed_weights_after_eval = _edge_weight_sha256(
        graph, kinds=("input", "recurrent", "output")
    )
    recurrent_weights_after_eval = _edge_weight_sha256(
        graph, kinds=("recurrent",)
    )
    input_weights_after_eval = _edge_weight_sha256(graph, kinds=("input",))
    output_weights_after_eval = _edge_weight_sha256(graph, kinds=("output",))
    blank_tail = _blank_tail_stability(graph, condition=condition)
    weights_after = graph.weights_hash()
    framed_weights_after = _edge_weight_sha256(
        graph, kinds=("input", "recurrent", "output")
    )
    topology_after = graph.topology_hash()
    structural_mask_after = _structural_mask_sha256(graph)
    recurrent_weights_after = _edge_weight_sha256(graph, kinds=("recurrent",))
    input_weights_after = _edge_weight_sha256(graph, kinds=("input",))
    output_weights_after = _edge_weight_sha256(graph, kinds=("output",))
    ledger = graph.ledger
    train_metrics = paired_retention_metrics(train.paired)
    eval_metrics = paired_retention_metrics(evaluation.paired)
    ridge = strict_ridge_readout(
        train.paired, evaluation.paired, alpha=ridge_alpha
    )
    combined_activity = _combine_activity(train.activity, evaluation.activity)

    native_weights = native_graph.weights
    condition_weights = graph.weights
    all_weights_match_native = all(
        np.asarray([condition_weights[edge.edge_id]], dtype="<f8").tobytes()
        == np.asarray([native_weights[edge.edge_id]], dtype="<f8").tobytes()
        for edge in graph.edges
    )
    native_feature_edge_ids = tuple(
        sorted(
            edge.edge_id
            for edge in native_graph.edges
            if edge.kind == "output"
            and edge.destination == native_graph.output_node
        )
    )
    state_ledger = (
        graph.slow_state_ledger
        if isinstance(graph, SlowStateRecurrentEventGraph)
        else {
            "hidden_activation_evaluations": 0,
            "local_state_read_touches": 0,
            "local_state_decay_touches": 0,
            "local_state_write_touches": 0,
            "local_state_reset_touches": 0,
            "local_state_observation_touches": 0,
        }
    )
    is_candidate = retention is not None
    expected_reset_touches = ledger["episodes"] * len(graph.hidden_nodes)
    state_touch_equality = (
        state_ledger["local_state_read_touches"]
        == state_ledger["local_state_decay_touches"]
        == state_ledger["local_state_write_touches"]
        == state_ledger["hidden_activation_evaluations"]
    ) if is_candidate else all(value == 0 for value in state_ledger.values())
    current_states = (
        graph.local_state_snapshot
        if isinstance(graph, SlowStateRecurrentEventGraph)
        else {}
    )
    invariant = {
        "topology_unchanged": topology_before == topology_after,
        "topology_matches_native": topology_before == native_graph.topology_hash(),
        "structural_mask_unchanged": structural_mask_before == structural_mask_after,
        "topology_unchanged_after_each_split": topology_before
        == topology_after_train
        == topology_after_eval
        == topology_after,
        "structural_mask_unchanged_after_each_split": structural_mask_before
        == structural_mask_after_train
        == structural_mask_after_eval
        == structural_mask_after,
        "structural_mask_matches_native": (
            structural_mask_before == _structural_mask_sha256(native_graph)
        ),
        "weights_unchanged_during_probe": (
            weights_before
            == weights_after_train
            == weights_after_eval
            == weights_after
        ),
        "framed_all_kind_weights_unchanged_during_probe": (
            framed_weights_before
            == framed_weights_after_train
            == framed_weights_after_eval
            == framed_weights_after
        ),
        "recurrent_weights_unchanged_during_probe": (
            recurrent_weights_before
            == recurrent_weights_after_train
            == recurrent_weights_after_eval
            == recurrent_weights_after
        ),
        "input_weights_unchanged_during_probe": (
            input_weights_before
            == input_weights_after_train
            == input_weights_after_eval
            == input_weights_after
        ),
        "output_weights_unchanged_during_probe": (
            output_weights_before
            == output_weights_after_train
            == output_weights_after_eval
            == output_weights_after
        ),
        "all_weights_match_native": weights_before == native_weights_hash
        and framed_weights_before == native_framed_weights_hash
        and all_weights_match_native,
        "recurrent_weights_match_native": recurrent_weights_before
        == native_recurrent_hash,
        "input_weights_match_native": input_weights_before == native_input_hash,
        "output_weights_match_native": output_weights_before == native_output_hash,
        "spectral_radius_matches_native": _within_matrix_tolerance(
            recurrent_matrix_metrics["spectral_radius"],
            native_matrix_metrics["spectral_radius"],
        ),
        "operator_norm_matches_native": _within_matrix_tolerance(
            recurrent_matrix_metrics["operator_2_norm"],
            native_matrix_metrics["operator_2_norm"],
        ),
        "registered_retention_immutable": (
            not is_candidate
            or (
                isinstance(graph, SlowStateRecurrentEventGraph)
                and graph.local_state_retention == retention
            )
        ),
        "candidate_state_forward_touches_equal_hidden_evaluations": state_touch_equality,
        "reset_touches_exact": (
            state_ledger["local_state_reset_touches"] == expected_reset_touches
            if is_candidate
            else state_ledger["local_state_reset_touches"] == 0
        ),
        "observation_touches_exact": (
            state_ledger["local_state_observation_touches"] == 128
            if is_candidate
            else state_ledger["local_state_observation_touches"] == 0
        ),
        "episode_resets_valid": (
            not is_candidate
            or (
                isinstance(graph, SlowStateRecurrentEventGraph)
                and graph.local_state_resets_valid
                and graph.local_state_reset_count == ledger["episodes"]
            )
        ),
        "stored_states_finite_and_in_tanh_range": all(
            math.isfinite(value) and abs(value) <= 1.0
            for value, _ in current_states.values()
        ),
        "feature_edges_match_native": feature_edge_ids == native_feature_edge_ids,
        "credit_event_touches_zero": ledger["credit_event_touches"] == 0,
        "credit_edge_touches_zero": ledger["credit_edge_touches"] == 0,
        "weight_write_touches_zero": ledger["weight_write_touches"] == 0,
        "credit_packets_zero": len(graph.credit_packets) == 0,
        "nonfinite_values_zero": ledger["nonfinite_values"] == 0,
        "all_weights_finite": all(
            math.isfinite(value) for value in condition_weights.values()
        ),
        "all_weights_within_configured_clip": all(
            abs(value) <= graph.weight_clip for value in condition_weights.values()
        ),
    }
    if not all(invariant.values()):
        failed = sorted(key for key, value in invariant.items() if not value)
        raise RuntimeError(f"slow-state condition invariant failed: {failed}")

    return {
        "condition": condition,
        "retention_coefficient": 0.0 if retention is None else retention,
        "target_retention": retention,
        "retention_half_life_ticks": (
            None if retention is None else float(math.log(0.5) / math.log(retention))
        ),
        "retention_after_ten_ticks": (
            0.0 if retention is None else float(retention**10)
        ),
        "data_pair_sha256": stream_pair.manifest.pair_sha256,
        "graph": {
            "hidden_units": len(graph.hidden_nodes),
            "recurrent_edge_count": sum(edge.kind == "recurrent" for edge in graph.edges),
            "feature_dimension": len(feature_edge_ids),
            "feature_edge_ids": list(feature_edge_ids),
            "topology_sha256": topology_before,
            "structural_mask_sha256": structural_mask_before,
            "initial_legacy_json_weights_sha256": weights_before,
            "after_train_legacy_json_weights_sha256": weights_after_train,
            "after_eval_legacy_json_weights_sha256": weights_after_eval,
            "final_legacy_json_weights_sha256": weights_after,
            "initial_complete_weights_sha256": framed_weights_before,
            "after_train_complete_weights_sha256": (
                framed_weights_after_train
            ),
            "after_eval_complete_weights_sha256": framed_weights_after_eval,
            "final_complete_weights_sha256": framed_weights_after,
            "topology_after_train_sha256": topology_after_train,
            "topology_after_eval_sha256": topology_after_eval,
            "topology_final_sha256": topology_after,
            "structural_mask_after_train_sha256": structural_mask_after_train,
            "structural_mask_after_eval_sha256": structural_mask_after_eval,
            "structural_mask_final_sha256": structural_mask_after,
            "initial_recurrent_weights_sha256": recurrent_weights_before,
            "after_train_recurrent_weights_sha256": recurrent_weights_after_train,
            "after_eval_recurrent_weights_sha256": recurrent_weights_after_eval,
            "final_recurrent_weights_sha256": recurrent_weights_after,
            "initial_input_weights_sha256": input_weights_before,
            "after_train_input_weights_sha256": input_weights_after_train,
            "after_eval_input_weights_sha256": input_weights_after_eval,
            "final_input_weights_sha256": input_weights_after,
            "initial_output_weights_sha256": output_weights_before,
            "after_train_output_weights_sha256": output_weights_after_train,
            "after_eval_output_weights_sha256": output_weights_after_eval,
            "final_output_weights_sha256": output_weights_after,
            "native_legacy_json_weights_sha256": native_weights_hash,
            "native_complete_weights_sha256": native_framed_weights_hash,
            "native_recurrent_weights_sha256": native_recurrent_hash,
            "native_input_weights_sha256": native_input_hash,
            "native_output_weights_sha256": native_output_hash,
            "recurrent_matrix": recurrent_matrix_metrics,
            "hyperparameters": _constructor_hyperparameters(graph),
        },
        "train": train_metrics,
        "eval": eval_metrics,
        "ridge_readout": ridge,
        "activity": {
            "train": train.activity,
            "eval": evaluation.activity,
            "combined": combined_activity,
        },
        "blank_tail_stability": blank_tail,
        "pair_rows": {
            "train": list(train.pair_rows),
            "eval": list(evaluation.pair_rows),
            "train_sha256": _sha256_json(train.pair_rows),
            "eval_sha256": _sha256_json(evaluation.pair_rows),
            "bundle_sha256": _sha256_json(
                [list(train.pair_rows), list(evaluation.pair_rows)]
            ),
        },
        "ledger": {
            "credit_event_touches": ledger["credit_event_touches"],
            "credit_edge_touches": ledger["credit_edge_touches"],
            "weight_write_touches": ledger["weight_write_touches"],
            "nonfinite_values": ledger["nonfinite_values"],
            "state": state_ledger,
        },
        "invariants": invariant,
    }


def _seed_gate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    paired = candidate["paired_vs_native"]
    split_activity_pass = all(
        paired["activity_by_split"][split]["total_emitted_event_ratio"]
        is not None
        and paired["activity_by_split"][split]["total_emitted_event_ratio"]
        <= GATE_THRESHOLDS["event_activity_total_ratio_maximum"]
        and paired["activity_by_split"][split][
            "maximum_matched_episode_event_count_ratio"
        ]
        is not None
        and paired["activity_by_split"][split][
            "maximum_matched_episode_event_count_ratio"
        ]
        <= GATE_THRESHOLDS["matched_episode_event_count_ratio_maximum"]
        and paired["activity_by_split"][split]["forward_edge_touch_ratio"]
        is not None
        and paired["activity_by_split"][split]["forward_edge_touch_ratio"]
        <= GATE_THRESHOLDS["forward_edge_touch_total_ratio_maximum"]
        and paired["activity_by_split"][split][
            "maximum_matched_episode_forward_edge_touch_ratio"
        ]
        <= GATE_THRESHOLDS[
            "matched_episode_forward_edge_touch_ratio_maximum"
        ]
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
            >= GATE_THRESHOLDS[
                "paired_native_absolute_cue_delta_multiple_minimum"
            ]
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
        "every_split_event_and_forward_totals_at_most_1_10x_and_matched_episodes_at_most_1_15x_native": (
            split_activity_pass
        ),
        "tail_effective_state_and_wake_gates": candidate[
            "blank_tail_stability"
        ]["all_tail_state_wake_gates_pass"],
        "finite": candidate["invariants"]["nonfinite_values_zero"],
        "all_implementation_and_graph_invariants": all(
            candidate["invariants"].values()
        ),
        "no_ccf_updates": all(
            candidate["invariants"][key]
            for key in (
                "credit_event_touches_zero",
                "credit_edge_touches_zero",
                "weight_write_touches_zero",
                "credit_packets_zero",
            )
        ),
        "spectral_radius_and_operator_norm_match_native": (
            candidate["invariants"]["spectral_radius_matches_native"]
            and candidate["invariants"]["operator_norm_matches_native"]
        ),
    }
    return {"criteria": criteria, "pass": all(criteria.values())}


def _attach_native_comparison(
    condition: dict[str, Any], native: Mapping[str, Any]
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
    for split in ("train", "eval"):
        candidate_split = condition["activity"][split]
        native_split = native["activity"][split]
        candidate_pair_rows = condition["pair_rows"][split]
        native_pair_rows = native["pair_rows"][split]
        if len(candidate_pair_rows) != len(native_pair_rows):
            raise RuntimeError("candidate/native pair rows have different lengths")
        matched_episode_ratios: list[float] = []
        matched_forward_ratios: list[float] = []
        for candidate_pair, native_pair in zip(
            candidate_pair_rows, native_pair_rows, strict=True
        ):
            if (
                candidate_pair["base_episode_id"]
                != native_pair["base_episode_id"]
                or candidate_pair["noise_stream_id"]
                != native_pair["noise_stream_id"]
            ):
                raise RuntimeError("candidate/native activity rows are not paired")
            for cue in (0, 1):
                matched_ratio = _safe_ratio(
                    float(candidate_pair[f"cue_{cue}_emitted_unit_events"]),
                    float(native_pair[f"cue_{cue}_emitted_unit_events"]),
                )
                if matched_ratio is None:
                    raise RuntimeError("native episode emitted zero unit events")
                matched_episode_ratios.append(matched_ratio)
                forward_ratio = _safe_ratio(
                    float(candidate_pair[f"cue_{cue}_forward_edge_touches"]),
                    float(native_pair[f"cue_{cue}_forward_edge_touches"]),
                )
                if forward_ratio is None:
                    raise RuntimeError("native episode touched zero forward edges")
                matched_forward_ratios.append(forward_ratio)
        split_activity[split] = {
            "total_emitted_event_ratio": _safe_ratio(
                float(candidate_split["total_emitted_unit_events"]),
                float(native_split["total_emitted_unit_events"]),
            ),
            "maximum_episode_event_count_ratio": _safe_ratio(
                float(
                    candidate_split[
                        "maximum_emitted_unit_events_per_forward_episode"
                    ]
                ),
                float(
                    native_split[
                        "maximum_emitted_unit_events_per_forward_episode"
                    ]
                ),
            ),
            "maximum_matched_episode_event_count_ratio": max(
                matched_episode_ratios
            ),
            "matched_episode_event_count_ratios": matched_episode_ratios,
            "matched_episode_event_count_ratios_sha256": _array_bundle_sha256(
                matched_episode_ratios
            ),
            "hidden_event_total_ratio": _safe_ratio(
                float(candidate_split["total_hidden_unit_events"]),
                float(native_split["total_hidden_unit_events"]),
            ),
            "forward_edge_touch_ratio": _safe_ratio(
                float(candidate_split["forward_edge_touches"]),
                float(native_split["forward_edge_touches"]),
            ),
            "maximum_matched_episode_forward_edge_touch_ratio": max(
                matched_forward_ratios
            ),
            "matched_episode_forward_edge_touch_ratios": matched_forward_ratios,
            "matched_episode_forward_edge_touch_ratios_sha256": (
                _array_bundle_sha256(matched_forward_ratios)
            ),
            "total_emitted_event_difference": int(
                candidate_split["total_emitted_unit_events"]
                - native_split["total_emitted_unit_events"]
            ),
        }
    condition["paired_vs_native"] = {
        "eval_cue_to_noise_feature_ratio_difference": _paired_difference(
            candidate_ratio, native_ratio
        ),
        "eval_cue_to_noise_feature_ratio_multiple": _safe_ratio(
            candidate_ratio, native_ratio
        ),
        "eval_absolute_cue_feature_delta_rms_difference": _paired_difference(
            condition["eval"]["cue_feature_delta_rms"],
            native["eval"]["cue_feature_delta_rms"],
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
        "mean_event_count_difference": float(candidate_events - native_events),
        "mean_event_count_inflation": _safe_ratio(candidate_events, native_events),
        "maximum_event_count_inflation": _safe_ratio(
            float(
                condition["activity"]["combined"][
                    "maximum_emitted_unit_events_per_forward_episode"
                ]
            ),
            float(
                native["activity"]["combined"][
                    "maximum_emitted_unit_events_per_forward_episode"
                ]
            ),
        ),
        "activity_by_split": split_activity,
    }
    condition["seed_gate"] = _seed_gate(condition)


def _run_seed(
    *,
    seed: int,
    retentions: tuple[float, ...],
    pairs_per_split: int,
    ridge_alpha: float,
) -> dict[str, Any]:
    require_seed_role(seed, SeedRole.CUSTOM)
    if seed not in REGISTERED_SEEDS:
        raise ValueError(f"seed {seed} is not a registered slow-state seed")
    stream_pair = generate_stream_pair(
        seed,
        episode_count=pairs_per_split,
        noise_events=8,
        required_role=SeedRole.CUSTOM,
    )
    native_graph = build_experiment_000_graph(
        seed,
        mode="full",
        event_log_enabled=False,
        **FROZEN_GRAPH_OPTIONS,
    )
    conditions = [
        _run_condition(
            native_graph=native_graph,
            retention=retention,
            stream_pair=stream_pair,
            ridge_alpha=ridge_alpha,
        )
        for retention in (None, *retentions)
    ]
    native = conditions[0]
    native["paired_vs_native"] = {
        "eval_cue_to_noise_feature_ratio_difference": 0.0,
        "eval_cue_to_noise_feature_ratio_multiple": 1.0,
        "eval_absolute_cue_feature_delta_rms_difference": 0.0,
        "eval_absolute_cue_feature_delta_rms_multiple": 1.0,
        "ridge_eval_accuracy_difference": 0.0,
        "eval_output_cue_delta_to_sd_difference": 0.0,
        "mean_event_count_difference": 0.0,
        "mean_event_count_inflation": 1.0,
        "maximum_event_count_inflation": 1.0,
        "activity_by_split": {
            "train": {
                "total_emitted_event_ratio": 1.0,
                "maximum_episode_event_count_ratio": 1.0,
                "maximum_matched_episode_event_count_ratio": 1.0,
                "matched_episode_event_count_ratios": [1.0]
                * (2 * pairs_per_split),
                "matched_episode_event_count_ratios_sha256": _array_bundle_sha256(
                    [1.0] * (2 * pairs_per_split)
                ),
                "hidden_event_total_ratio": 1.0,
                "forward_edge_touch_ratio": 1.0,
                "maximum_matched_episode_forward_edge_touch_ratio": 1.0,
                "matched_episode_forward_edge_touch_ratios": [1.0]
                * (2 * pairs_per_split),
                "matched_episode_forward_edge_touch_ratios_sha256": (
                    _array_bundle_sha256([1.0] * (2 * pairs_per_split))
                ),
                "total_emitted_event_difference": 0,
            },
            "eval": {
                "total_emitted_event_ratio": 1.0,
                "maximum_episode_event_count_ratio": 1.0,
                "maximum_matched_episode_event_count_ratio": 1.0,
                "matched_episode_event_count_ratios": [1.0]
                * (2 * pairs_per_split),
                "matched_episode_event_count_ratios_sha256": _array_bundle_sha256(
                    [1.0] * (2 * pairs_per_split)
                ),
                "hidden_event_total_ratio": 1.0,
                "forward_edge_touch_ratio": 1.0,
                "maximum_matched_episode_forward_edge_touch_ratio": 1.0,
                "matched_episode_forward_edge_touch_ratios": [1.0]
                * (2 * pairs_per_split),
                "matched_episode_forward_edge_touch_ratios_sha256": (
                    _array_bundle_sha256([1.0] * (2 * pairs_per_split))
                ),
                "total_emitted_event_difference": 0,
            },
        },
    }
    native["seed_gate"] = {"eligible": False, "pass": False, "criteria": {}}
    for condition in conditions[1:]:
        _attach_native_comparison(condition, native)
    return {
        "seed": seed,
        "seed_role": SeedRole.CUSTOM.value,
        "data_manifest": stream_pair.manifest.to_dict(),
        "condition_results": conditions,
    }


def _mean_optional(values: Iterable[float | None]) -> float | None:
    realized = [float(value) for value in values if value is not None]
    return float(np.mean(realized)) if realized else None


def _distribution(values: Iterable[float | None]) -> dict[str, Any]:
    supplied = list(values)
    realized = np.asarray(
        [float(value) for value in supplied if value is not None], dtype=np.float64
    )
    null_count = len(supplied) - int(realized.size)
    if not realized.size:
        return {
            "count": 0,
            "total_count": len(supplied),
            "null_count": null_count,
            "all_values_defined": False,
            "minimum": None,
            "q1": None,
            "median": None,
            "q3": None,
            "maximum": None,
            "mean": None,
        }
    if not np.all(np.isfinite(realized)):
        raise FloatingPointError("aggregate distribution contains non-finite values")
    return {
        "count": int(realized.size),
        "total_count": len(supplied),
        "null_count": null_count,
        "all_values_defined": null_count == 0,
        "minimum": float(np.min(realized)),
        "q1": float(np.quantile(realized, 0.25)),
        "median": float(np.median(realized)),
        "q3": float(np.quantile(realized, 0.75)),
        "maximum": float(np.max(realized)),
        "mean": float(np.mean(realized)),
    }


def _aggregate_conditions(
    seed_results: Sequence[Mapping[str, Any]],
    retentions: tuple[float, ...],
    *,
    pairs_per_split: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    condition_ids = tuple(condition_name(value) for value in (None, *retentions))
    by_condition: dict[str, list[Mapping[str, Any]]] = {
        condition: [] for condition in condition_ids
    }
    for seed_result in seed_results:
        seen: set[str] = set()
        for condition_row in seed_result["condition_results"]:
            condition = condition_row["condition"]
            if condition not in by_condition or condition in seen:
                raise RuntimeError("seed result contains invalid condition rows")
            by_condition[condition].append(condition_row)
            seen.add(condition)
        if seen != set(condition_ids):
            raise RuntimeError("seed result is missing a registered condition")

    native_rows = by_condition[NATIVE_CONDITION]
    native_ratio_distribution = _distribution(
        entry["eval"]["cue_to_noise_feature_ratio"] for entry in native_rows
    )
    native_absolute_distribution = _distribution(
        entry["eval"]["cue_feature_delta_rms"] for entry in native_rows
    )
    native_control_valid = all(
        all(entry["invariants"].values())
        and entry["blank_tail_stability"]["all_tail_state_wake_gates_pass"]
        for entry in native_rows
    )
    aggregate_rows: list[dict[str, Any]] = []
    for condition in condition_ids:
        rows = by_condition[condition]
        distributions = {
            "recurrent_spectral_radius": _distribution(
                entry["graph"]["recurrent_matrix"]["spectral_radius"]
                for entry in rows
            ),
            "eval_cue_to_noise_feature_ratio": _distribution(
                entry["eval"]["cue_to_noise_feature_ratio"] for entry in rows
            ),
            "eval_cue_feature_delta_rms": _distribution(
                entry["eval"]["cue_feature_delta_rms"] for entry in rows
            ),
            "eval_noise_feature_rms": _distribution(
                entry["eval"]["noise_feature_rms"] for entry in rows
            ),
            "ridge_train_accuracy": _distribution(
                entry["ridge_readout"]["train_accuracy"] for entry in rows
            ),
            "ridge_eval_accuracy": _distribution(
                entry["ridge_readout"]["eval_accuracy"] for entry in rows
            ),
            "eval_output_cue_delta_to_sd": _distribution(
                entry["eval"]["output_cue_delta_to_sd"] for entry in rows
            ),
            "task_mean_emitted_unit_events": _distribution(
                entry["activity"]["combined"][
                    "mean_emitted_unit_events_per_forward_episode"
                ]
                for entry in rows
            ),
            "task_hidden_emission_density": _distribution(
                entry["activity"]["combined"]["hidden_emission_density"]
                for entry in rows
            ),
            "blank_tail_hidden_emission_density": _distribution(
                entry["blank_tail_stability"]["hidden_emission_density"]
                for entry in rows
            ),
            "recurrent_operator_2_norm": _distribution(
                entry["graph"]["recurrent_matrix"]["operator_2_norm"]
                for entry in rows
            ),
            "recurrent_nonnormality_ratio": _distribution(
                entry["graph"]["recurrent_matrix"][
                    "operator_2_norm_to_spectral_radius"
                ]
                for entry in rows
            ),
        }
        all_invariants = all(all(entry["invariants"].values()) for entry in rows)
        all_tail_state_wake_gates = all(
            entry["blank_tail_stability"]["all_tail_state_wake_gates_pass"]
            for entry in rows
        )
        aggregate: dict[str, Any] = {
            "condition": condition,
            "target_retention": rows[0]["target_retention"],
            "seed_count": len(rows),
            "distributions": distributions,
            "all_invariants": all_invariants,
            "all_tail_effective_state_and_wake_gates": all_tail_state_wake_gates,
            "all_weights_unchanged": all(
                entry["invariants"]["weights_unchanged_during_probe"]
                for entry in rows
            ),
            "all_topologies_match_native": all(
                entry["invariants"]["topology_matches_native"] for entry in rows
            ),
            "all_weight_kinds_match_native": all(
                entry["invariants"]["recurrent_weights_match_native"]
                and entry["invariants"]["input_weights_match_native"]
                and entry["invariants"]["output_weights_match_native"]
                for entry in rows
            ),
            "all_update_counts_zero": all(
                entry["invariants"]["credit_event_touches_zero"]
                and entry["invariants"]["credit_edge_touches_zero"]
                and entry["invariants"]["weight_write_touches_zero"]
                and entry["invariants"]["credit_packets_zero"]
                for entry in rows
            ),
        }
        if condition == NATIVE_CONDITION:
            aggregate["paired_vs_native"] = {
                "median_cue_ratio_multiple": 1.0,
                "within_seed_cue_ratio_threefold_count": 0,
                "median_absolute_cue_delta_multiple": 1.0,
                "within_seed_absolute_cue_delta_threefold_count": 0,
            }
            aggregate["gate"] = {
                "eligible": False,
                "pass": False,
                "criteria": {},
            }
        else:
            cue_multiple_rows = [
                entry["paired_vs_native"][
                    "eval_cue_to_noise_feature_ratio_multiple"
                ]
                for entry in rows
            ]
            absolute_multiple_rows = [
                entry["paired_vs_native"][
                    "eval_absolute_cue_feature_delta_rms_multiple"
                ]
                for entry in rows
            ]
            cue_threefold_count = sum(
                value is not None
                and value
                >= GATE_THRESHOLDS["paired_native_cue_ratio_multiple_minimum"]
                for value in cue_multiple_rows
            )
            absolute_threefold_count = sum(
                value is not None
                and value
                >= GATE_THRESHOLDS[
                    "paired_native_absolute_cue_delta_multiple_minimum"
                ]
                for value in absolute_multiple_rows
            )
            all_activity_guards = all(
                entry["seed_gate"]["criteria"][
                    "every_split_event_and_forward_totals_at_most_1_10x_and_matched_episodes_at_most_1_15x_native"
                ]
                for entry in rows
            )
            median_ratio = distributions["eval_cue_to_noise_feature_ratio"][
                "median"
            ]
            median_absolute = distributions["eval_cue_feature_delta_rms"][
                "median"
            ]
            paired = {
                "cue_ratio_multiple_distribution": _distribution(
                    cue_multiple_rows
                ),
                "median_cue_ratio_multiple": _safe_ratio(
                    median_ratio, native_ratio_distribution["median"]
                ),
                "within_seed_cue_ratio_threefold_count": cue_threefold_count,
                "absolute_cue_delta_multiple_distribution": _distribution(
                    absolute_multiple_rows
                ),
                "median_absolute_cue_delta_multiple": _safe_ratio(
                    median_absolute, native_absolute_distribution["median"]
                ),
                "within_seed_absolute_cue_delta_threefold_count": (
                    absolute_threefold_count
                ),
                "all_seed_split_activity_guards": all_activity_guards,
                "maximum_split_total_event_ratio": max(
                    entry["paired_vs_native"]["activity_by_split"][split][
                        "total_emitted_event_ratio"
                    ]
                    for entry in rows
                    for split in ("train", "eval")
                ),
                "maximum_matched_episode_event_ratio": max(
                    entry["paired_vs_native"]["activity_by_split"][split][
                        "maximum_matched_episode_event_count_ratio"
                    ]
                    for entry in rows
                    for split in ("train", "eval")
                ),
                "maximum_split_total_forward_edge_touch_ratio": max(
                    entry["paired_vs_native"]["activity_by_split"][split][
                        "forward_edge_touch_ratio"
                    ]
                    for entry in rows
                    for split in ("train", "eval")
                ),
                "maximum_matched_episode_forward_edge_touch_ratio": max(
                    entry["paired_vs_native"]["activity_by_split"][split][
                        "maximum_matched_episode_forward_edge_touch_ratio"
                    ]
                    for entry in rows
                    for split in ("train", "eval")
                ),
            }
            aggregate["paired_vs_native"] = paired
            criteria = {
                "full_100_pair_run": pairs_per_split == FULL_PAIRS_PER_SPLIT,
                "native_control_all_invariants_and_blank_tails_valid": (
                    native_control_valid
                ),
                "all_five_seed_ratio_output_and_multiplier_values_defined": (
                    distributions["eval_cue_to_noise_feature_ratio"]["count"]
                    == len(rows)
                    and distributions["eval_output_cue_delta_to_sd"]["count"]
                    == len(rows)
                    and native_ratio_distribution["count"] == len(native_rows)
                    and paired["cue_ratio_multiple_distribution"]["count"]
                    == len(rows)
                    and paired["absolute_cue_delta_multiple_distribution"][
                        "count"
                    ]
                    == len(rows)
                ),
                "median_eval_cue_ratio_at_least_0_50": (
                    median_ratio is not None
                    and median_ratio
                    >= GATE_THRESHOLDS[
                        "eval_cue_to_noise_feature_ratio_minimum"
                    ]
                ),
                "median_eval_cue_ratio_at_least_3x_median_native": (
                    paired["median_cue_ratio_multiple"] is not None
                    and paired["median_cue_ratio_multiple"]
                    >= GATE_THRESHOLDS[
                        "paired_native_cue_ratio_multiple_minimum"
                    ]
                ),
                "at_least_4_of_5_seeds_cue_ratio_at_least_3x_native": (
                    cue_threefold_count
                    >= int(
                        GATE_THRESHOLDS[
                            "within_seed_threefold_improvement_count_minimum"
                        ]
                    )
                ),
                "median_absolute_cue_delta_at_least_3x_median_native": (
                    paired["median_absolute_cue_delta_multiple"] is not None
                    and paired["median_absolute_cue_delta_multiple"]
                    >= GATE_THRESHOLDS[
                        "paired_native_absolute_cue_delta_multiple_minimum"
                    ]
                ),
                "at_least_4_of_5_seeds_absolute_cue_delta_at_least_3x_native": (
                    absolute_threefold_count
                    >= int(
                        GATE_THRESHOLDS[
                            "within_seed_threefold_improvement_count_minimum"
                        ]
                    )
                ),
                "median_ridge_eval_accuracy_at_least_0_70": (
                    distributions["ridge_eval_accuracy"]["median"]
                    >= GATE_THRESHOLDS["ridge_eval_accuracy_minimum"]
                ),
                "median_eval_output_ratio_at_least_0_50": (
                    distributions["eval_output_cue_delta_to_sd"]["median"]
                    is not None
                    and distributions["eval_output_cue_delta_to_sd"]["median"]
                    >= GATE_THRESHOLDS[
                        "eval_output_cue_delta_to_sd_minimum"
                    ]
                ),
                "every_seed_split_activity_guard_passes": all_activity_guards,
                "tail_effective_state_and_wake_pass_for_every_seed": (
                    all_tail_state_wake_gates
                ),
                "all_frozen_state_and_finite_invariants_pass": all_invariants,
            }
            aggregate["gate"] = {
                "eligible": pairs_per_split == FULL_PAIRS_PER_SPLIT,
                "criteria": criteria,
                "pass": all(criteria.values()),
            }
        aggregate_rows.append(aggregate)

    passing = sorted(
        (
            row
            for row in aggregate_rows
            if row["condition"] != NATIVE_CONDITION and row["gate"]["pass"]
        ),
        key=lambda row: float(row["target_retention"]),
    )
    selected = passing[0] if passing and native_control_valid else None
    final_status = (
        "INVALID_NATIVE_CONTROL"
        if not native_control_valid
        else (
            f"SELECTED:{selected['condition']}"
            if selected is not None
            else "NO_SELECTION"
        )
    )
    selection = {
        "rule": "lowest retention satisfying every preregistered full-run gate",
        "final_status": final_status,
        "selected_condition": None if selected is None else selected["condition"],
        "selected_retention": (
            None if selected is None else selected["target_retention"]
        ),
        "passing_conditions": [row["condition"] for row in passing],
        "any_target_passed": bool(passing),
        "smoke_selection_forbidden": pairs_per_split != FULL_PAIRS_PER_SPLIT,
    }
    return aggregate_rows, selection


def verify_deterministic_full_runs(
    first: Mapping[str, Any] | str | Path,
    second: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Apply the frozen two-run determinism stop rule to full reports."""

    def load(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return value
        return json.loads(Path(value).read_text(encoding="utf-8"))

    left = load(first)
    right = load(second)
    if left.get("run_kind") != "full" or right.get("run_kind") != "full":
        raise ValueError("determinism verification requires two full reports")
    left_hash = left.get("deterministic_payload_sha256")
    right_hash = right.get("deterministic_payload_sha256")
    def recompute(report: Mapping[str, Any]) -> str:
        payload = dict(report)
        payload.pop("nondeterministic_provenance", None)
        payload.pop("deterministic_payload_sha256", None)
        return _sha256_json(payload)

    left_recomputed = recompute(left)
    right_recomputed = recompute(right)
    left_self_valid = isinstance(left_hash, str) and left_hash == left_recomputed
    right_self_valid = isinstance(right_hash, str) and right_hash == right_recomputed
    hashes_match = (
        left_self_valid and right_self_valid and left_hash == right_hash
    )
    if not hashes_match:
        status = "NONDETERMINISTIC_INVALID"
    elif not (
        bool(left.get("integrity", {}).get("native_control_valid"))
        and bool(right.get("integrity", {}).get("native_control_valid"))
    ):
        status = "INVALID_NATIVE_CONTROL"
    elif left.get("status") != right.get("status"):
        status = "NONDETERMINISTIC_INVALID"
        hashes_match = False
    else:
        status = str(left["status"])
    return {
        "status": status,
        "deterministic_payloads_match": hashes_match,
        "first_deterministic_payload_sha256": left_hash,
        "second_deterministic_payload_sha256": right_hash,
        "first_recomputed_payload_sha256": left_recomputed,
        "second_recomputed_payload_sha256": right_recomputed,
        "first_self_hash_valid": left_self_valid,
        "second_self_hash_valid": right_self_valid,
        "selection_statuses_match": left.get("status") == right.get("status"),
        "valid_terminal_status": status == "NO_SELECTION"
        or status.startswith("SELECTED:"),
    }


def run_slow_state_experiment(
    *,
    seeds: Sequence[int] = REGISTERED_SEEDS,
    retentions: Sequence[float] = REGISTERED_RETENTIONS,
    pairs_per_split: int = DEFAULT_PAIRS_PER_SPLIT,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the deterministic, custom-seed, no-learning architecture sweep."""

    resolved_seeds, resolved_retentions = _validate_arguments(
        seeds=seeds,
        retentions=retentions,
        pairs_per_split=pairs_per_split,
        ridge_alpha=ridge_alpha,
    )
    source_integrity = frozen_source_integrity()
    required_source_checks = (
        "recurrent_source_matches",
        "credit_method_bundle_matches",
        "ccf_document_matches",
        "protocol_matches",
        "config_matches",
    )
    if not all(source_integrity[key] for key in required_source_checks):
        failed = [key for key in required_source_checks if not source_integrity[key]]
        raise RuntimeError(f"frozen native source integrity failed: {failed}")
    zero_retention_equivalence = run_zero_retention_equivalence()
    started = perf_counter()
    seed_results = [
        _run_seed(
            seed=seed,
            retentions=resolved_retentions,
            pairs_per_split=pairs_per_split,
            ridge_alpha=ridge_alpha,
        )
        for seed in resolved_seeds
    ]
    aggregate_conditions, selection = _aggregate_conditions(
        seed_results, resolved_retentions, pairs_per_split=pairs_per_split
    )
    all_rows = [
        condition
        for seed_result in seed_results
        for condition in seed_result["condition_results"]
    ]
    provenance = _source_provenance()
    report: dict[str, Any] = {
        "schema_version": SLOW_STATE_SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "scope": "custom_seed_architecture_only_no_learning_slow_state",
        "mechanism": "CCF-v0 unchanged and never invoked",
        "run_kind": (
            "full" if pairs_per_split == FULL_PAIRS_PER_SPLIT else "smoke"
        ),
        "confirmatory_executed": False,
        "confirmatory_seed_count": 0,
        "seeds": list(resolved_seeds),
        "seed_partition": {
            "required_role": SeedRole.CUSTOM.value,
            "registered_order": list(REGISTERED_SEEDS),
            "forbidden_development_seeds": [0, 1, 2, 3, 4],
            "forbidden_alignment_seeds": [42, 43, 44, 45, 46],
            "forbidden_memory_sweep_seeds": [50, 51, 52, 53, 54],
            "forbidden_confirmatory_seeds": list(range(1000, 1020)),
            "forbidden_seed_intersection": [],
            "exact_order_validated_before_output": True,
        },
        "conditions": [condition_name(None)]
        + [condition_name(value) for value in resolved_retentions],
        "retention_coefficients": list(resolved_retentions),
        "pairs_per_split": pairs_per_split,
        "counterfactual_forward_episodes_per_seed_per_condition": 4
        * pairs_per_split,
        "required_identical_full_runs": 2,
        "sparse_efficiency_claim_allowed": False,
        "noise_events": 8,
        "ridge_alpha": float(ridge_alpha),
        "matrix_absolute_tolerance": MATRIX_ABS_TOLERANCE,
        "matrix_relative_tolerance": MATRIX_REL_TOLERANCE,
        "blank_tail_assay": {
            "cue_polarities": [-1.0, 1.0],
            "blank_ticks": BLANK_TAIL_TICKS,
            "final_window_first_tick": 193,
            "final_window_last_tick": 256,
            "required_final_window_hidden_emissions": 0,
            "effective_state_maximum_absolute": GATE_THRESHOLDS[
                "effective_state_maximum_absolute"
            ],
            "wake_feature_half_difference_l2_maximum": GATE_THRESHOLDS[
                "wake_feature_half_difference_l2_maximum"
            ],
            "wake_output_half_difference_absolute_maximum": GATE_THRESHOLDS[
                "wake_output_half_difference_absolute_maximum"
            ],
        },
        "graph_options": dict(FROZEN_GRAPH_OPTIONS),
        "gate_thresholds": dict(GATE_THRESHOLDS),
        "formulas": {
            "spectral_radius": "rho(W_rec)=max(abs(eigvals(W_rec)))",
            "local_retained_state": "r_i(t)=lambda**(t-tau_i)*s_i",
            "hidden_activation": "a_i(t)=tanh(sum_e(w_e*v_e)+r_i(t))",
            "cue_to_noise_feature_ratio": (
                "R=sqrt(mean(||(z_1-z_0)/2||^2))/"
                "sqrt(mean(||(z_1+z_0)/2-mean_pair_midpoint||^2))"
            ),
            "paired_native_multiple": "R_target(seed)/R_native(seed)",
            "absolute_cue_multiple": "C_target(seed)/C_native(seed)",
            "task_total_activity_ratio": (
                "sum emitted unit events target / matched native sum"
            ),
            "matched_episode_activity_ratio": "E_target(k,c)/E_native(k,c)",
            "hidden_emission_density": "hidden UnitEvents/(64*11)=hidden/704",
            "nonnormality_ratio": "operator_2_norm(W_rec)/rho(W_rec)",
        },
        "source_provenance": provenance,
        "frozen_source_integrity": source_integrity,
        "zero_retention_native_equivalence": zero_retention_equivalence,
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
                seed_result["seed"] in REGISTERED_SEEDS
                and seed_result["seed_role"] == SeedRole.CUSTOM.value
                for seed_result in seed_results
            ),
            "confirmatory_seed_count": 0,
            "all_conditions_registered": {row["condition"] for row in all_rows}
            == set(
                [NATIVE_CONDITION]
                + [condition_name(value) for value in resolved_retentions]
            ),
            "all_topologies_match_native_and_remain_unchanged": all(
                row["invariants"]["topology_unchanged"]
                and row["invariants"]["topology_matches_native"]
                for row in all_rows
            ),
            "all_weight_kinds_match_native": all(
                row["invariants"]["recurrent_weights_match_native"]
                and row["invariants"]["input_weights_match_native"]
                and row["invariants"]["output_weights_match_native"]
                for row in all_rows
            ),
            "all_condition_weights_unchanged": all(
                row["invariants"]["weights_unchanged_during_probe"]
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
            "all_recurrent_matrix_metrics_match_native": all(
                row["invariants"]["spectral_radius_matches_native"]
                and row["invariants"]["operator_norm_matches_native"]
                for row in all_rows
            ),
            "all_weights_within_clip": all(
                row["invariants"]["all_weights_within_configured_clip"]
                for row in all_rows
            ),
            "all_tail_effective_state_and_wake_gates": all(
                row["blank_tail_stability"]["all_tail_state_wake_gates_pass"]
                for row in all_rows
            ),
            "all_source_hashes_match": all(
                source_integrity[key] for key in required_source_checks
            ),
            "native_control_valid": all(
                all(row["invariants"].values())
                and row["blank_tail_stability"]["all_tail_state_wake_gates_pass"]
                for row in (
                    seed_result["condition_results"][0]
                    for seed_result in seed_results
                )
            ),
            "pair_manifest_count": len(seed_results),
        },
    }
    report["deterministic_payload_sha256"] = _sha256_json(report)
    report["nondeterministic_provenance"] = {
        "excluded_from_deterministic_payload_sha256": True,
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
        },
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            report, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
        payload_bytes = serialized.encode("utf-8")
        target.write_bytes(payload_bytes)
        report_sha256 = hashlib.sha256(payload_bytes).hexdigest()
        sidecar = target.with_suffix(".sha256")
        sidecar_payload = (
            json.dumps(
                {
                    "algorithm": "sha256",
                    "report_file": target.name,
                    "report_sha256": report_sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        sidecar.write_bytes(sidecar_payload.encode("utf-8"))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Experiment-000 recurrent local-state-retention slow-state experiment "
            "without learning or confirmatory seeds."
        )
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="required exact custom seeds: 60 61 62 63 64",
    )
    parser.add_argument(
        "--retentions",
        type=float,
        nargs="+",
        required=True,
        help="required exact retentions: 0.25 0.50 0.75 0.90 0.95",
    )
    parser.add_argument(
        "--pairs-per-split",
        type=int,
        default=DEFAULT_PAIRS_PER_SPLIT,
        help="frozen smoke/full values are 2 and 100 (default: 2)",
    )
    parser.add_argument("--ridge-alpha", type=float, default=DEFAULT_RIDGE_ALPHA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)
    report = run_slow_state_experiment(
        seeds=tuple(args.seeds),
        retentions=tuple(args.retentions),
        pairs_per_split=args.pairs_per_split,
        ridge_alpha=args.ridge_alpha,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_PAIRS_PER_SPLIT",
    "FULL_PAIRS_PER_SPLIT",
    "GATE_THRESHOLDS",
    "MATRIX_ABS_TOLERANCE",
    "MATRIX_REL_TOLERANCE",
    "NATIVE_CONDITION",
    "REGISTERED_RETENTIONS",
    "REGISTERED_SEEDS",
    "SLOW_STATE_SCHEMA_VERSION",
    "SMOKE_PAIRS_PER_SPLIT",
    "clone_with_local_state_retention",
    "condition_name",
    "frozen_source_integrity",
    "main",
    "run_slow_state_experiment",
    "run_zero_retention_equivalence",
    "strict_ridge_readout",
    "verify_deterministic_full_runs",
]
