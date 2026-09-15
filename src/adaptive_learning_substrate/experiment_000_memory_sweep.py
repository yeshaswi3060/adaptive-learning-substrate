"""Preregistered architecture-only memory-strength sweep for Experiment 000.

The sweep changes exactly one architectural quantity: every recurrent weight is
multiplied by ``target_radius / native_radius``.  Input and output weights, the
structural mask, all simulator hyperparameters, the CCF-v0 implementation, and
the paired data are held fixed.  No credit event is delivered, so every
condition is a pure forward-memory measurement rather than a learning run.

Only the custom seeds 50--54 and the registered recurrent spectral-radius
targets are accepted.  Arguments are validated before an output path is
created, preventing accidental use of development or confirmatory seeds.
"""

from __future__ import annotations

import argparse
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

from .events import EventLog
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
    ridge_readout,
)
from .recurrent import RecurrentEventGraph, build_experiment_000_graph

MEMORY_SWEEP_SCHEMA_VERSION = "experiment-000-memory-strength-sweep-v1-a1"
PROTOCOL_NAME = "experiment_000_recurrent_memory_strength_sweep"
PROTOCOL_VERSION = "memory-sweep-v1-a1"
REGISTERED_SEEDS: tuple[int, ...] = (50, 51, 52, 53, 54)
REGISTERED_TARGET_RADII: tuple[float, ...] = (0.60, 0.80, 0.95, 1.05)
NATIVE_CONDITION = "native"
SMOKE_PAIRS_PER_SPLIT = 2
FULL_PAIRS_PER_SPLIT = 100
DEFAULT_PAIRS_PER_SPLIT = SMOKE_PAIRS_PER_SPLIT
DEFAULT_RIDGE_ALPHA = 1e-3
DEFAULT_OUTPUT_PATH = Path("artifacts/experiment_000/memory_sweep/report.json")
SPECTRAL_RADIUS_ABS_TOLERANCE = 1e-10
SPECTRAL_RADIUS_REL_TOLERANCE = 1e-10
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
    "blank_tail_ticks": float(BLANK_TAIL_TICKS),
    "blank_tail_final_window_ticks": float(BLANK_TAIL_FINAL_WINDOW_TICKS),
    "blank_tail_final_window_hidden_events_maximum": 0.0,
}

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
        "src/adaptive_learning_substrate/experiment_000_memory_sweep.py",
        "src/adaptive_learning_substrate/experiment_000_memory_probe.py",
        "src/adaptive_learning_substrate/recurrent.py",
        "src/adaptive_learning_substrate/experiment_000.py",
        "src/adaptive_learning_substrate/experiment000_data.py",
        "docs/EXPERIMENT_000_MEMORY_SWEEP_PROTOCOL.md",
        "configs/experiment_000_memory_sweep.toml",
    )
    files: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"memory-sweep provenance file is missing: {relative}")
        files[relative] = {"sha256": _file_sha256(path), "bytes": path.stat().st_size}
    return {"files": files, "bundle_sha256": _sha256_json(files)}


def condition_name(target_radius: float | None) -> str:
    """Return the stable report identifier for one registered condition."""

    if target_radius is None:
        return NATIVE_CONDITION
    return f"rho_{target_radius:.2f}".replace(".", "_")


def _registered_target(value: float) -> float | None:
    for registered in REGISTERED_TARGET_RADII:
        if math.isclose(value, registered, rel_tol=0.0, abs_tol=1e-12):
            return registered
    return None


def _validate_arguments(
    *,
    seeds: Sequence[int],
    target_radii: Sequence[float],
    pairs_per_split: int,
    ridge_alpha: float,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    resolved_seeds = tuple(seeds)
    if not resolved_seeds:
        raise ValueError("the exact registered memory-sweep seed list is required")
    if len(set(resolved_seeds)) != len(resolved_seeds):
        raise ValueError("memory-sweep seeds must be unique")
    for seed in resolved_seeds:
        require_seed_role(seed, SeedRole.CUSTOM)
        if seed not in REGISTERED_SEEDS:
            raise ValueError(
                f"seed {seed} is custom but not one of the frozen memory-sweep "
                f"seeds {REGISTERED_SEEDS}"
            )
    if resolved_seeds != REGISTERED_SEEDS:
        raise ValueError(
            "the official smoke and full memory sweep require the exact ordered "
            f"seed list {REGISTERED_SEEDS}"
        )

    if not target_radii:
        raise ValueError("at least one registered target radius is required")
    resolved_targets: list[float] = []
    for raw_target in target_radii:
        if isinstance(raw_target, bool):
            raise TypeError("target radii must be finite registered numbers")
        target = float(raw_target)
        if not math.isfinite(target):
            raise ValueError("target radii must be finite registered numbers")
        registered = _registered_target(target)
        if registered is None:
            raise ValueError(
                f"target radius {target} is not one of the frozen memory-sweep "
                f"targets {REGISTERED_TARGET_RADII}"
            )
        resolved_targets.append(registered)
    if len(set(resolved_targets)) != len(resolved_targets):
        raise ValueError("target radii must be unique")
    # Canonical registered order makes a run independent of CLI argument order.
    ordered_targets = tuple(
        target for target in REGISTERED_TARGET_RADII if target in resolved_targets
    )
    if tuple(resolved_targets) != REGISTERED_TARGET_RADII:
        raise ValueError(
            "the memory sweep requires the exact ordered target list "
            f"{REGISTERED_TARGET_RADII}"
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
    return resolved_seeds, ordered_targets


def _radius_within_tolerance(achieved: float, expected: float) -> bool:
    tolerance = SPECTRAL_RADIUS_ABS_TOLERANCE + (
        SPECTRAL_RADIUS_REL_TOLERANCE * max(abs(achieved), abs(expected))
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
        [
            (
                edge.edge_id,
                edge.source,
                edge.destination,
                edge.kind,
                edge.delay_ticks,
                edge.plastic,
            )
            for edge in graph.edges
        ]
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


def clone_with_recurrent_spectral_radius(
    graph: RecurrentEventGraph,
    target_radius: float,
    *,
    event_log_enabled: bool | None = None,
) -> RecurrentEventGraph:
    """Return a fresh public graph clone with only recurrent weights rescaled.

    Multiplying the complete recurrent matrix by ``target/native`` scales all
    eigenvalues by that factor.  The function constructs a new
    :class:`RecurrentEventGraph`; it never writes a private weight mapping.
    """

    target = float(target_radius)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("target_radius must be finite and positive")
    native_radius = recurrent_spectral_radius(graph)
    if native_radius <= _RATIO_EPSILON:
        raise ValueError("cannot scale a graph with zero recurrent spectral radius")
    scale_factor = target / native_radius
    native_weights = graph.weights
    scaled_weights = {
        edge.edge_id: (
            float(native_weights[edge.edge_id] * scale_factor)
            if edge.kind == "recurrent"
            else native_weights[edge.edge_id]
        )
        for edge in graph.edges
    }
    if not all(math.isfinite(value) for value in scaled_weights.values()):
        raise FloatingPointError("target recurrent rescaling produced non-finite weights")
    if any(abs(value) > graph.weight_clip for value in scaled_weights.values()):
        raise ValueError(
            "target recurrent rescaling would exceed the configured weight clip"
        )
    log_enabled = graph.event_log.enabled if event_log_enabled is None else bool(event_log_enabled)
    clone = RecurrentEventGraph(
        input_nodes=graph.input_nodes,
        hidden_nodes=graph.hidden_nodes,
        output_node=graph.output_node,
        edges=graph.edges,
        initial_weights=scaled_weights,
        event_log=EventLog(enabled=log_enabled),
        **_constructor_hyperparameters(graph),
    )

    if clone.topology_hash() != graph.topology_hash():
        raise RuntimeError("spectral rescaling changed graph topology")
    for field in _CLONED_HYPERPARAMETERS:
        if getattr(clone, field) != getattr(graph, field):
            raise RuntimeError(f"spectral rescaling failed to preserve {field}")
    clone_weights = clone.weights
    for edge in graph.edges:
        expected = scaled_weights[edge.edge_id]
        if clone_weights[edge.edge_id] != expected:
            raise RuntimeError(f"clone changed edge weight {edge.edge_id}")
        if edge.kind != "recurrent" and expected != native_weights[edge.edge_id]:
            raise RuntimeError("spectral rescaling changed a non-recurrent weight")

    achieved = recurrent_spectral_radius(clone)
    if not _radius_within_tolerance(achieved, target):
        raise RuntimeError(
            f"target recurrent spectral radius {target} was not achieved; got {achieved}"
        )
    return clone


def _counterfactual_episode(
    base: Experiment000Episode, *, cue: int, condition: str
) -> Experiment000Episode:
    return replace(
        base,
        episode_id=f"{base.episode_id}-memory-sweep-{condition}-cue-{cue}",
        cue=cue,
        target=cue,
    )


def _collect_split(
    graph: RecurrentEventGraph,
    stream: Experiment000Stream,
    *,
    condition: str,
    feature_edge_ids: tuple[str, ...],
) -> _CollectedSplit:
    cue_zero_features: list[np.ndarray] = []
    cue_one_features: list[np.ndarray] = []
    cue_zero_outputs: list[float] = []
    cue_one_outputs: list[float] = []
    event_counts: list[int] = []
    hidden_event_counts: list[int] = []
    pair_rows: list[dict[str, Any]] = []
    ledger_before = graph.ledger

    for base in stream.episodes:
        row: dict[str, Any] = {
            "pair_index": int(base.index),
            "base_episode_id": base.episode_id,
            "noise_stream_id": base.noise_stream_id,
        }
        for cue in (0, 1):
            episode = _counterfactual_episode(base, cue=cue, condition=condition)
            episode_ledger_before = graph.ledger
            query = forward_episode(graph, episode)
            episode_ledger_after = graph.ledger
            extracted = extract_query_features(graph, query)
            if tuple(edge_id for edge_id, _ in extracted) != feature_edge_ids:
                raise RuntimeError("query feature order changed during memory sweep")
            feature = np.asarray([value for _, value in extracted], dtype=np.float64)
            output = float(query.activation)
            if not np.all(np.isfinite(feature)) or not math.isfinite(output):
                raise FloatingPointError("non-finite memory-sweep forward value")
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
            event_counts.append(event_count)
            hidden_event_counts.append(hidden_event_count)
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
            if cue == 0:
                cue_zero_features.append(feature)
                cue_zero_outputs.append(output)
            else:
                cue_one_features.append(feature)
                cue_one_outputs.append(output)
        pair_rows.append(row)

    ledger_after = graph.ledger
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
        raise RuntimeError("no-learning memory sweep wrote graph weights")
    if ledger_after["credit_event_touches"] != ledger_before["credit_event_touches"]:
        raise RuntimeError("no-learning memory sweep processed credit")

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


def _blank_tail_stability(
    graph: RecurrentEventGraph, *, condition: str
) -> dict[str, Any]:
    """Measure autonomous hidden emissions after both bipolar cue polarities."""

    ledger_before = graph.ledger
    weights_before = graph.weights_hash()
    hidden_nodes = set(graph.hidden_nodes)
    polarity_rows: list[dict[str, Any]] = []
    for cue_input in (-1.0, 1.0):
        polarity_ledger_before = graph.ledger
        polarity_weights_before = graph.weights_hash()
        polarity_topology_before = graph.topology_hash()
        cue_label = "negative" if cue_input < 0.0 else "positive"
        graph.begin_episode(f"memory-sweep-blank-tail-{condition}-{cue_label}")
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
        polarity_ledger_after = graph.ledger
        polarity_weights_after = graph.weights_hash()
        polarity_topology_after = graph.topology_hash()
        last_nonzero = next(
            (
                index
                for index in range(BLANK_TAIL_TICKS, 0, -1)
                if hidden_counts[index - 1] > 0
            ),
            None,
        )
        final_window = hidden_counts[-BLANK_TAIL_FINAL_WINDOW_TICKS:]
        polarity_rows.append(
            {
                "cue_input": cue_input,
                "blank_ticks": BLANK_TAIL_TICKS,
                "final_window_ticks": BLANK_TAIL_FINAL_WINDOW_TICKS,
                "hidden_emissions_per_blank_tick": hidden_counts,
                "total_emissions_per_blank_tick": total_counts,
                "hidden_counts_sha256": _array_bundle_sha256(hidden_counts),
                "total_counts_sha256": _array_bundle_sha256(total_counts),
                "total_hidden_emissions": int(sum(hidden_counts)),
                "hidden_emission_density": float(
                    sum(hidden_counts)
                    / (BLANK_TAIL_TICKS * len(graph.hidden_nodes))
                ),
                "maximum_hidden_emissions_in_one_tick": max(hidden_counts),
                "last_nonzero_hidden_blank_tick": last_nonzero,
                "final_window_hidden_emissions": int(sum(final_window)),
                "final_window_quiescent": sum(final_window) == 0,
                "forward_edge_touches": int(
                    polarity_ledger_after["forward_edge_touches"]
                    - polarity_ledger_before["forward_edge_touches"]
                ),
                "weights_before_sha256": polarity_weights_before,
                "weights_after_sha256": polarity_weights_after,
                "weights_unchanged": (
                    polarity_weights_before == polarity_weights_after
                ),
                "topology_before_sha256": polarity_topology_before,
                "topology_after_sha256": polarity_topology_after,
                "topology_unchanged": (
                    polarity_topology_before == polarity_topology_after
                ),
            }
        )
        if polarity_weights_before != polarity_weights_after:
            raise RuntimeError("one blank-tail polarity changed graph weights")
        if polarity_topology_before != polarity_topology_after:
            raise RuntimeError("one blank-tail polarity changed graph topology")
    ledger_after = graph.ledger
    weights_after = graph.weights_hash()
    if weights_before != weights_after:
        raise RuntimeError("blank-tail stability assay changed graph weights")
    if ledger_after["weight_write_touches"] != ledger_before["weight_write_touches"]:
        raise RuntimeError("blank-tail stability assay wrote graph weights")
    if ledger_after["credit_event_touches"] != ledger_before["credit_event_touches"]:
        raise RuntimeError("blank-tail stability assay processed credit")
    total_hidden = sum(row["total_hidden_emissions"] for row in polarity_rows)
    return {
        "cue_polarities": polarity_rows,
        "polarity_bundle_sha256": _sha256_json(polarity_rows),
        "all_polarities_final_window_quiescent": all(
            row["final_window_quiescent"] for row in polarity_rows
        ),
        "total_hidden_emissions": int(total_hidden),
        "hidden_emission_density": float(
            total_hidden
            / (2 * BLANK_TAIL_TICKS * len(graph.hidden_nodes))
        ),
        "forward_edge_touches": int(
            ledger_after["forward_edge_touches"] - ledger_before["forward_edge_touches"]
        ),
        "weights_unchanged": weights_before == weights_after,
        "credit_event_touches": int(
            ledger_after["credit_event_touches"]
            - ledger_before["credit_event_touches"]
        ),
        "weight_write_touches": int(
            ledger_after["weight_write_touches"]
            - ledger_before["weight_write_touches"]
        ),
        "nonfinite_values": int(
            ledger_after["nonfinite_values"] - ledger_before["nonfinite_values"]
        ),
    }


def _condition_graph(
    native_graph: RecurrentEventGraph, target_radius: float | None
) -> tuple[RecurrentEventGraph, float, float]:
    native_radius = recurrent_spectral_radius(native_graph)
    if target_radius is None:
        return native_graph, native_radius, 1.0
    return (
        clone_with_recurrent_spectral_radius(
            native_graph, target_radius, event_log_enabled=False
        ),
        float(target_radius),
        float(target_radius / native_radius),
    )


def _run_condition(
    *,
    native_graph: RecurrentEventGraph,
    target_radius: float | None,
    stream_pair: Any,
    ridge_alpha: float,
) -> dict[str, Any]:
    condition = condition_name(target_radius)
    graph, expected_radius, scale_factor = _condition_graph(native_graph, target_radius)
    native_radius = recurrent_spectral_radius(native_graph)
    achieved_radius = recurrent_spectral_radius(graph)
    recurrent_matrix_metrics = _recurrent_matrix_metrics(graph)
    topology_before = graph.topology_hash()
    structural_mask_before = _structural_mask_sha256(graph)
    weights_before = graph.weights_hash()
    recurrent_weights_before = _edge_weight_sha256(graph, kinds=("recurrent",))
    input_output_weights_before = _edge_weight_sha256(
        graph, kinds=("input", "output")
    )
    native_input_output_hash = _edge_weight_sha256(
        native_graph, kinds=("input", "output")
    )
    feature_edge_ids = tuple(
        sorted(
            edge.edge_id
            for edge in graph.edges
            if edge.kind == "output" and edge.destination == graph.output_node
        )
    )
    train = _collect_split(
        graph,
        stream_pair.train,
        condition=condition,
        feature_edge_ids=feature_edge_ids,
    )
    weights_after_train = graph.weights_hash()
    recurrent_weights_after_train = _edge_weight_sha256(
        graph, kinds=("recurrent",)
    )
    input_output_weights_after_train = _edge_weight_sha256(
        graph, kinds=("input", "output")
    )
    evaluation = _collect_split(
        graph,
        stream_pair.eval,
        condition=condition,
        feature_edge_ids=feature_edge_ids,
    )
    weights_after_eval = graph.weights_hash()
    recurrent_weights_after_eval = _edge_weight_sha256(
        graph, kinds=("recurrent",)
    )
    input_output_weights_after_eval = _edge_weight_sha256(
        graph, kinds=("input", "output")
    )
    blank_tail = _blank_tail_stability(graph, condition=condition)
    weights_after = graph.weights_hash()
    topology_after = graph.topology_hash()
    structural_mask_after = _structural_mask_sha256(graph)
    recurrent_weights_after = _edge_weight_sha256(graph, kinds=("recurrent",))
    input_output_weights_after = _edge_weight_sha256(
        graph, kinds=("input", "output")
    )
    ledger = graph.ledger
    train_metrics = paired_retention_metrics(train.paired)
    eval_metrics = paired_retention_metrics(evaluation.paired)
    ridge = ridge_readout(train.paired, evaluation.paired, alpha=ridge_alpha)
    combined_activity = _combine_activity(train.activity, evaluation.activity)

    achieved_within_tolerance = _radius_within_tolerance(
        achieved_radius, expected_radius
    )
    native_weights = native_graph.weights
    condition_weights = graph.weights
    recurrent_scale_exact = all(
        condition_weights[edge.edge_id]
        == float(native_weights[edge.edge_id] * scale_factor)
        for edge in graph.edges
        if edge.kind == "recurrent"
    )
    recurrent_signs_preserved = all(
        math.copysign(1.0, condition_weights[edge.edge_id])
        == math.copysign(1.0, native_weights[edge.edge_id])
        for edge in graph.edges
        if edge.kind == "recurrent"
    )
    native_feature_edge_ids = tuple(
        sorted(
            edge.edge_id
            for edge in native_graph.edges
            if edge.kind == "output"
            and edge.destination == native_graph.output_node
        )
    )
    invariant = {
        "topology_unchanged": topology_before == topology_after,
        "topology_matches_native": topology_before == native_graph.topology_hash(),
        "structural_mask_unchanged": structural_mask_before == structural_mask_after,
        "structural_mask_matches_native": (
            structural_mask_before == _structural_mask_sha256(native_graph)
        ),
        "weights_unchanged_during_probe": (
            weights_before
            == weights_after_train
            == weights_after_eval
            == weights_after
        ),
        "recurrent_weights_unchanged_during_probe": (
            recurrent_weights_before
            == recurrent_weights_after_train
            == recurrent_weights_after_eval
            == recurrent_weights_after
        ),
        "input_output_weights_unchanged_during_probe": (
            input_output_weights_before
            == input_output_weights_after_train
            == input_output_weights_after_eval
            == input_output_weights_after
        ),
        "input_output_weights_match_native": (
            input_output_weights_before == native_input_output_hash
        ),
        "achieved_radius_within_tolerance": achieved_within_tolerance,
        "recurrent_scale_exact": recurrent_scale_exact,
        "recurrent_signs_preserved": recurrent_signs_preserved,
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
        raise RuntimeError(f"memory-sweep condition invariant failed: {failed}")

    return {
        "condition": condition,
        "target_spectral_radius": target_radius,
        "native_spectral_radius": native_radius,
        "expected_spectral_radius": expected_radius,
        "achieved_spectral_radius": achieved_radius,
        "achieved_radius_absolute_error": abs(achieved_radius - expected_radius),
        "recurrent_scale_factor": scale_factor,
        "data_pair_sha256": stream_pair.manifest.pair_sha256,
        "graph": {
            "hidden_units": len(graph.hidden_nodes),
            "recurrent_edge_count": sum(edge.kind == "recurrent" for edge in graph.edges),
            "feature_dimension": len(feature_edge_ids),
            "feature_edge_ids": list(feature_edge_ids),
            "topology_sha256": topology_before,
            "structural_mask_sha256": structural_mask_before,
            "initial_weights_sha256": weights_before,
            "after_train_weights_sha256": weights_after_train,
            "after_eval_weights_sha256": weights_after_eval,
            "final_weights_sha256": weights_after,
            "initial_recurrent_weights_sha256": recurrent_weights_before,
            "after_train_recurrent_weights_sha256": recurrent_weights_after_train,
            "after_eval_recurrent_weights_sha256": recurrent_weights_after_eval,
            "final_recurrent_weights_sha256": recurrent_weights_after,
            "initial_input_output_weights_sha256": input_output_weights_before,
            "after_train_input_output_weights_sha256": (
                input_output_weights_after_train
            ),
            "after_eval_input_output_weights_sha256": (
                input_output_weights_after_eval
            ),
            "final_input_output_weights_sha256": input_output_weights_after,
            "native_input_output_weights_sha256": native_input_output_hash,
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
        "every_split_total_at_most_1_10x_and_matched_episode_at_most_1_15x_native": (
            split_activity_pass
        ),
        "both_blank_tails_final_64_quiescent": candidate[
            "blank_tail_stability"
        ]["all_polarities_final_window_quiescent"],
        "finite": candidate["invariants"]["nonfinite_values_zero"],
        "weights_and_topology_unchanged": all(
            candidate["invariants"][key]
            for key in (
                "topology_unchanged",
                "topology_matches_native",
                "weights_unchanged_during_probe",
                "input_output_weights_match_native",
            )
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
        "spectral_radius_achieved": candidate["invariants"][
            "achieved_radius_within_tolerance"
        ],
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
    target_radii: tuple[float, ...],
    pairs_per_split: int,
    ridge_alpha: float,
) -> dict[str, Any]:
    require_seed_role(seed, SeedRole.CUSTOM)
    if seed not in REGISTERED_SEEDS:
        raise ValueError(f"seed {seed} is not a registered memory-sweep seed")
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
            target_radius=target,
            stream_pair=stream_pair,
            ridge_alpha=ridge_alpha,
        )
        for target in (None, *target_radii)
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
    realized = np.asarray(
        [float(value) for value in values if value is not None], dtype=np.float64
    )
    if not realized.size:
        return {
            "count": 0,
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
        "minimum": float(np.min(realized)),
        "q1": float(np.quantile(realized, 0.25)),
        "median": float(np.median(realized)),
        "q3": float(np.quantile(realized, 0.75)),
        "maximum": float(np.max(realized)),
        "mean": float(np.mean(realized)),
    }


def _aggregate_conditions(
    seed_results: Sequence[Mapping[str, Any]],
    target_radii: tuple[float, ...],
    *,
    pairs_per_split: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    condition_ids = tuple(condition_name(target) for target in (None, *target_radii))
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
        and entry["blank_tail_stability"][
            "all_polarities_final_window_quiescent"
        ]
        for entry in native_rows
    )
    aggregate_rows: list[dict[str, Any]] = []
    for condition in condition_ids:
        rows = by_condition[condition]
        distributions = {
            "achieved_spectral_radius": _distribution(
                entry["achieved_spectral_radius"] for entry in rows
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
        all_blank_tails_quiescent = all(
            entry["blank_tail_stability"][
                "all_polarities_final_window_quiescent"
            ]
            for entry in rows
        )
        aggregate: dict[str, Any] = {
            "condition": condition,
            "target_spectral_radius": rows[0]["target_spectral_radius"],
            "seed_count": len(rows),
            "distributions": distributions,
            "maximum_radius_absolute_error": max(
                entry["achieved_radius_absolute_error"] for entry in rows
            ),
            "all_invariants": all_invariants,
            "all_blank_tails_final_64_quiescent": all_blank_tails_quiescent,
            "all_weights_unchanged": all(
                entry["invariants"]["weights_unchanged_during_probe"]
                for entry in rows
            ),
            "all_topologies_match_native": all(
                entry["invariants"]["topology_matches_native"] for entry in rows
            ),
            "all_input_output_weights_match_native": all(
                entry["invariants"]["input_output_weights_match_native"]
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
                    "every_split_total_at_most_1_10x_and_matched_episode_at_most_1_15x_native"
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
                "both_blank_tails_quiescent_for_every_seed": (
                    all_blank_tails_quiescent
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
        key=lambda row: float(row["target_spectral_radius"]),
    )
    selected = passing[0] if passing else None
    final_status = (
        f"SELECTED:{selected['condition']}" if selected is not None else "NO_SELECTION"
    )
    selection = {
        "rule": "lowest target satisfying every preregistered full-run gate",
        "final_status": final_status,
        "selected_condition": None if selected is None else selected["condition"],
        "selected_target_spectral_radius": (
            None if selected is None else selected["target_spectral_radius"]
        ),
        "passing_conditions": [row["condition"] for row in passing],
        "any_target_passed": bool(passing),
        "smoke_selection_forbidden": pairs_per_split != FULL_PAIRS_PER_SPLIT,
    }
    return aggregate_rows, selection


def run_memory_strength_sweep(
    *,
    seeds: Sequence[int] = REGISTERED_SEEDS,
    target_radii: Sequence[float] = REGISTERED_TARGET_RADII,
    pairs_per_split: int = DEFAULT_PAIRS_PER_SPLIT,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the deterministic, custom-seed, no-learning architecture sweep."""

    resolved_seeds, resolved_targets = _validate_arguments(
        seeds=seeds,
        target_radii=target_radii,
        pairs_per_split=pairs_per_split,
        ridge_alpha=ridge_alpha,
    )
    started = perf_counter()
    seed_results = [
        _run_seed(
            seed=seed,
            target_radii=resolved_targets,
            pairs_per_split=pairs_per_split,
            ridge_alpha=ridge_alpha,
        )
        for seed in resolved_seeds
    ]
    aggregate_conditions, selection = _aggregate_conditions(
        seed_results, resolved_targets, pairs_per_split=pairs_per_split
    )
    all_rows = [
        condition
        for seed_result in seed_results
        for condition in seed_result["condition_results"]
    ]
    provenance = _source_provenance()
    report: dict[str, Any] = {
        "schema_version": MEMORY_SWEEP_SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "scope": "custom_seed_architecture_only_no_learning_memory_sweep",
        "mechanism": "CCF-v0 unchanged and never invoked",
        "run_kind": (
            "full" if pairs_per_split == FULL_PAIRS_PER_SPLIT else "smoke"
        ),
        "confirmatory_executed": False,
        "confirmatory_seed_count": 0,
        "seeds": list(resolved_seeds),
        "conditions": [condition_name(None)]
        + [condition_name(target) for target in resolved_targets],
        "target_spectral_radii": list(resolved_targets),
        "pairs_per_split": pairs_per_split,
        "counterfactual_forward_episodes_per_seed_per_condition": 4
        * pairs_per_split,
        "required_identical_full_runs": 2,
        "sparse_efficiency_claim_allowed": False,
        "noise_events": 8,
        "ridge_alpha": float(ridge_alpha),
        "spectral_radius_absolute_tolerance": SPECTRAL_RADIUS_ABS_TOLERANCE,
        "spectral_radius_relative_tolerance": SPECTRAL_RADIUS_REL_TOLERANCE,
        "blank_tail_assay": {
            "cue_polarities": [-1.0, 1.0],
            "blank_ticks": BLANK_TAIL_TICKS,
            "final_window_first_tick": 193,
            "final_window_last_tick": 256,
            "required_final_window_hidden_emissions": 0,
        },
        "graph_options": dict(FROZEN_GRAPH_OPTIONS),
        "gate_thresholds": dict(GATE_THRESHOLDS),
        "formulas": {
            "spectral_radius": "rho(W_rec)=max(abs(eigvals(W_rec)))",
            "recurrent_scale": "W_rec,target=(rho_target/rho_native)*W_rec,native",
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
        "seed_results": seed_results,
        "aggregate_conditions": aggregate_conditions,
        "selection": selection,
        "status": selection["final_status"],
        "integrity": {
            "all_seeds_custom_and_registered": all(
                seed_result["seed"] in REGISTERED_SEEDS
                and seed_result["seed_role"] == SeedRole.CUSTOM.value
                for seed_result in seed_results
            ),
            "confirmatory_seed_count": 0,
            "all_conditions_registered": {
                row["condition"] for row in all_rows
            }
            == set(
                [NATIVE_CONDITION]
                + [condition_name(target) for target in resolved_targets]
            ),
            "all_topologies_match_native_and_remain_unchanged": all(
                row["invariants"]["topology_unchanged"]
                and row["invariants"]["topology_matches_native"]
                for row in all_rows
            ),
            "all_input_output_weights_match_native": all(
                row["invariants"]["input_output_weights_match_native"]
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
            "all_spectral_radii_within_tolerance": all(
                row["invariants"]["achieved_radius_within_tolerance"]
                for row in all_rows
            ),
            "all_scaled_weights_within_clip": all(
                row["invariants"]["all_weights_within_configured_clip"]
                for row in all_rows
            ),
            "all_blank_tail_polarities_final_64_quiescent": all(
                row["blank_tail_stability"][
                    "all_polarities_final_window_quiescent"
                ]
                for row in all_rows
            ),
            "native_control_valid": all(
                all(row["invariants"].values())
                and row["blank_tail_stability"][
                    "all_polarities_final_window_quiescent"
                ]
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
            "Run the Experiment-000 recurrent spectral-radius memory sweep "
            "without learning or confirmatory seeds."
        )
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(REGISTERED_SEEDS),
        help="registered custom seeds only (default: 50 51 52 53 54)",
    )
    parser.add_argument(
        "--target-radii",
        type=float,
        nargs="+",
        default=list(REGISTERED_TARGET_RADII),
        help="registered targets only (default: 0.60 0.80 0.95 1.05)",
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
    report = run_memory_strength_sweep(
        seeds=tuple(args.seeds),
        target_radii=tuple(args.target_radii),
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
    "MEMORY_SWEEP_SCHEMA_VERSION",
    "NATIVE_CONDITION",
    "REGISTERED_SEEDS",
    "REGISTERED_TARGET_RADII",
    "SMOKE_PAIRS_PER_SPLIT",
    "SPECTRAL_RADIUS_ABS_TOLERANCE",
    "clone_with_recurrent_spectral_radius",
    "condition_name",
    "main",
    "run_memory_strength_sweep",
]
