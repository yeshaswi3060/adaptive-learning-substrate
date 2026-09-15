"""Exact fixed-event-DAG update-alignment diagnostic for Experiment 000.

This development diagnostic differentiates the immutable event ancestry that
was actually realised by the sparse recurrent graph.  Event emission decisions
are held fixed, shared weights accumulate every temporal occurrence, and no
automatic-differentiation runtime is used.  The exact local descent direction
is compared with the total parameter change produced by CCF-v0.

The default seeds are custom diagnostic seeds 42--46.  Confirmatory seeds are
rejected before an output path is created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np

from .events import EventLog, UnitEvent
from .experiment000_data import (
    Experiment000Episode,
    SeedRole,
    generate_stream_pair,
    require_seed_role,
)
from .experiment_000 import FROZEN_GRAPH_OPTIONS, forward_episode
from .recurrent import QueryResult, RecurrentEventGraph, build_experiment_000_graph

ALIGNMENT_SCHEMA_VERSION = "experiment-000-fixed-event-alignment-v1"
DEFAULT_ALIGNMENT_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
FULL_EPISODES_PER_SEED = 20
DEFAULT_EPISODES_PER_SEED = 2
DEFAULT_FINITE_DIFFERENCE_COORDINATES = 32
DEFAULT_FINITE_DIFFERENCE_STEP = 1e-6
DEFAULT_OUTPUT_PATH = Path("artifacts/experiment_000/alignment/report.json")
GROUP_ORDER: tuple[str, ...] = ("cue", "noise", "query", "recurrent", "output")
_METRIC_EPSILON = 1e-15


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


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _source_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "src/adaptive_learning_substrate/experiment_000_alignment.py",
        "src/adaptive_learning_substrate/recurrent.py",
        "src/adaptive_learning_substrate/experiment_000.py",
        "src/adaptive_learning_substrate/experiment000_data.py",
        "docs/EXPERIMENT_000_ALIGNMENT_PROTOCOL.md",
        "configs/experiment_000_alignment.toml",
    )
    files: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"alignment provenance file is missing: {relative}")
        files[relative] = {"sha256": _file_sha256(path), "bytes": path.stat().st_size}
    return {
        "project_root": str(root),
        "files": files,
        "bundle_sha256": _sha256_json(files),
    }


def _event_dag_sha256(events: Sequence[UnitEvent]) -> str:
    return _sha256_json([asdict(event) for event in events])


def _event_mask_signature(events: Sequence[UnitEvent]) -> tuple[object, ...]:
    """Return event/ancestry structure without activation values."""

    return tuple(
        (
            event.event_id,
            event.step,
            event.node,
            event.forced_output,
            tuple(
                (trace.edge_id, trace.parent_event_id)
                for trace in event.edge_traces
            ),
        )
        for event in events
    )


def _validate_arguments(
    *,
    seeds: Sequence[int],
    episodes_per_seed: int,
    finite_difference_coordinates: int,
    finite_difference_step: float,
) -> tuple[int, ...]:
    resolved = tuple(seeds)
    if not resolved:
        raise ValueError("at least one custom diagnostic seed is required")
    if len(set(resolved)) != len(resolved):
        raise ValueError("custom diagnostic seeds must be unique")
    for seed in resolved:
        require_seed_role(seed, SeedRole.CUSTOM)
        if seed not in DEFAULT_ALIGNMENT_SEEDS:
            raise ValueError(
                f"seed {seed} is custom but not one of the frozen alignment seeds "
                f"{DEFAULT_ALIGNMENT_SEEDS}"
            )
    if (
        isinstance(episodes_per_seed, bool)
        or not isinstance(episodes_per_seed, int)
        or episodes_per_seed <= 0
        or episodes_per_seed % 2
    ):
        raise ValueError("episodes_per_seed must be a positive even integer")
    if episodes_per_seed not in {DEFAULT_EPISODES_PER_SEED, FULL_EPISODES_PER_SEED}:
        raise ValueError("episodes_per_seed must be the frozen smoke value 2 or full value 20")
    if (
        isinstance(finite_difference_coordinates, bool)
        or not isinstance(finite_difference_coordinates, int)
        or finite_difference_coordinates < len(GROUP_ORDER)
    ):
        raise ValueError(
            f"finite_difference_coordinates must be an integer >= {len(GROUP_ORDER)}"
        )
    if not math.isfinite(finite_difference_step) or finite_difference_step <= 0.0:
        raise ValueError("finite_difference_step must be finite and positive")
    return resolved


def _target_activation(target: bool | float) -> float:
    value = float(target)
    if value == 0.0:
        return -1.0
    if value == 1.0:
        return 1.0
    if value == -1.0:
        return -1.0
    raise ValueError("alignment targets must be binary or bipolar")


def fixed_event_dag_output_gradient(
    *,
    events: Sequence[UnitEvent],
    weights: Mapping[str, float],
    edge_ids: Sequence[str],
    root_event_id: str,
) -> dict[str, float]:
    """Reverse-differentiate a realised event DAG with emission gates fixed.

    Adjoints are keyed by temporal event ID, while gradient coordinates are
    keyed by shared structural edge ID.  Thus every temporal occurrence uses
    ``gradient[edge_id] += dz * parent_activation`` and reconvergent event
    adjoints use ``+=`` as required.
    """

    ordered_edges = tuple(edge_ids)
    if len(set(ordered_edges)) != len(ordered_edges):
        raise ValueError("edge_ids must be unique")
    if set(ordered_edges) != set(weights):
        raise ValueError("edge_ids and weights must name exactly the same parameters")
    ordered_events = tuple(sorted(events, key=lambda event: (event.step, event.event_id)))
    events_by_id: dict[str, UnitEvent] = {}
    for event in ordered_events:
        if event.event_id in events_by_id:
            raise ValueError(f"duplicate event identifier {event.event_id!r}")
        events_by_id[event.event_id] = event
        for trace in event.edge_traces:
            if trace.edge_id not in weights:
                raise KeyError(f"trace names unknown edge {trace.edge_id!r}")
            parent = events_by_id.get(trace.parent_event_id)
            if parent is None:
                raise ValueError("event ancestry is not in strict chronological order")
            if parent.step >= event.step:
                raise ValueError("event trace does not point to an earlier event")
    if root_event_id not in events_by_id:
        raise KeyError(f"root event {root_event_id!r} is absent from the event DAG")

    adjoints = {event.event_id: 0.0 for event in ordered_events}
    adjoints[root_event_id] = 1.0
    gradient = {edge_id: 0.0 for edge_id in ordered_edges}
    for event in reversed(ordered_events):
        if not event.edge_traces:
            continue
        dz = adjoints[event.event_id] * (1.0 - float(event.activation) ** 2)
        for trace in event.edge_traces:
            edge_id = trace.edge_id
            gradient[edge_id] += dz * float(trace.message_value)
            adjoints[trace.parent_event_id] += dz * float(weights[edge_id])
    if not all(math.isfinite(value) for value in gradient.values()):
        raise FloatingPointError("non-finite fixed-event derivative")
    return gradient


def fixed_event_dag_output(
    *,
    events: Sequence[UnitEvent],
    weights: Mapping[str, float],
    root_event_id: str,
) -> float:
    """Evaluate the recorded computational DAG without changing its gates."""

    activations: dict[str, float] = {}
    for event in sorted(events, key=lambda item: (item.step, item.event_id)):
        if not event.edge_traces:
            activation = float(event.activation)
        else:
            total = math.fsum(
                float(weights[trace.edge_id]) * activations[trace.parent_event_id]
                for trace in event.edge_traces
            )
            activation = math.tanh(total)
        if not math.isfinite(activation):
            raise FloatingPointError("non-finite fixed-event replay activation")
        activations[event.event_id] = activation
    if root_event_id not in activations:
        raise KeyError(f"root event {root_event_id!r} is absent from the event DAG")
    return activations[root_event_id]


def _validate_realized_dag(
    *,
    graph: RecurrentEventGraph,
    events: Sequence[UnitEvent],
    weights: Mapping[str, float],
    query: QueryResult,
) -> dict[str, Any]:
    events_by_id = {event.event_id: event for event in events}
    if len(events_by_id) != len(events):
        raise RuntimeError("realized event DAG contains duplicate event IDs")
    if graph.audit["queried_output_event_id"] != query.event_id:
        raise RuntimeError("queried output ID does not match the event-DAG root")
    edges = graph.edges_by_id
    max_message_error = 0.0
    max_preactivation_error = 0.0
    max_activation_error = 0.0
    for event in events:
        contributions: list[float] = []
        for trace in event.edge_traces:
            parent = events_by_id.get(trace.parent_event_id)
            if parent is None or parent.step >= event.step:
                raise RuntimeError("event trace does not name a strict earlier parent")
            edge = edges[trace.edge_id]
            if edge.source != parent.node or edge.destination != event.node:
                raise RuntimeError("event trace endpoints disagree with structural edge")
            message_error = abs(float(trace.message_value) - float(parent.activation))
            max_message_error = max(max_message_error, message_error)
            contributions.append(float(weights[trace.edge_id]) * float(parent.activation))
        if event.edge_traces:
            reconstructed_u = math.fsum(contributions)
            reconstructed_a = math.tanh(reconstructed_u)
            max_preactivation_error = max(
                max_preactivation_error, abs(reconstructed_u - event.preactivation)
            )
            max_activation_error = max(
                max_activation_error, abs(reconstructed_a - event.activation)
            )
    fixed_output = fixed_event_dag_output(
        events=events, weights=weights, root_event_id=query.event_id
    )
    root_error = abs(fixed_output - query.activation)
    tolerance = 1e-12
    if max(max_message_error, max_preactivation_error, max_activation_error, root_error) > tolerance:
        raise RuntimeError("realized event DAG failed float64 reconstruction")

    ancestors: set[str] = set()
    pending = [query.event_id]
    while pending:
        event_id = pending.pop()
        if event_id in ancestors:
            continue
        ancestors.add(event_id)
        pending.extend(trace.parent_event_id for trace in events_by_id[event_id].edge_traces)
    current_tick = int(graph.audit["tick"])
    if any(current_tick - events_by_id[event_id].step > graph.trace_horizon for event_id in ancestors):
        raise RuntimeError("a query ancestor exceeds the configured trace horizon")
    return {
        "root_ancestor_count": len(ancestors),
        "max_message_error": max_message_error,
        "max_preactivation_error": max_preactivation_error,
        "max_activation_error": max_activation_error,
        "fixed_dag_root_error": root_error,
        "within_float64_tolerance": True,
    }


def _clone_with_weights(
    template: RecurrentEventGraph, weights: Mapping[str, float]
) -> RecurrentEventGraph:
    return RecurrentEventGraph(
        input_nodes=template.input_nodes,
        hidden_nodes=template.hidden_nodes,
        output_node=template.output_node,
        edges=template.edges,
        initial_weights=weights,
        mode="full",
        learning_rate=template.learning_rate,
        trace_decay=template.trace_decay,
        route_gain=template.route_gain,
        emit_threshold=template.emit_threshold,
        epsilon_weight=template.epsilon_weight,
        epsilon_message=template.epsilon_message,
        epsilon_route=template.epsilon_route,
        credit_limit=template.credit_limit,
        credit_minimum=template.credit_minimum,
        trace_horizon=template.trace_horizon,
        hop_limit=template.hop_limit,
        max_update=template.max_update,
        weight_clip=template.weight_clip,
        event_log=EventLog(enabled=False),
    )


def _forward_with_weights(
    template: RecurrentEventGraph,
    weights: Mapping[str, float],
    episode: Experiment000Episode,
) -> tuple[RecurrentEventGraph, QueryResult]:
    template_hash = template.weights_hash()
    graph = _clone_with_weights(template, weights)
    if graph.topology_hash() != template.topology_hash():
        raise RuntimeError("replay clone topology differs from the live graph")
    if graph.weights != {str(key): float(value) for key, value in weights.items()}:
        raise RuntimeError("replay clone did not preserve requested weights")
    query = forward_episode(graph, episode)
    if template.weights_hash() != template_hash:
        raise RuntimeError("replay construction mutated the live graph")
    return graph, query


def _edge_groups(
    graph: RecurrentEventGraph, edge_ids: Sequence[str]
) -> tuple[dict[str, str], dict[str, tuple[int, ...]]]:
    edges = graph.edges_by_id
    group_by_edge: dict[str, str] = {}
    indices: dict[str, list[int]] = {group: [] for group in GROUP_ORDER}
    for position, edge_id in enumerate(edge_ids):
        edge = edges[edge_id]
        if edge.kind == "input" and edge.source in {"cue", "noise", "query"}:
            group = edge.source
        elif edge.kind in {"recurrent", "output"}:
            group = edge.kind
        else:
            raise ValueError(f"edge {edge_id!r} has no alignment group")
        group_by_edge[edge_id] = group
        indices[group].append(position)
    if any(not indices[group] for group in GROUP_ORDER):
        raise RuntimeError("every alignment group must contain at least one edge")
    return group_by_edge, {
        group: tuple(group_indices) for group, group_indices in indices.items()
    }


def _stratified_coordinates(
    *,
    edge_ids: Sequence[str],
    group_indices: Mapping[str, Sequence[int]],
    exact_direction: np.ndarray,
    count: int,
) -> tuple[int, ...]:
    ranked: dict[str, list[int]] = {}
    for group in GROUP_ORDER:
        ranked[group] = sorted(
            group_indices[group],
            key=lambda position: (-abs(float(exact_direction[position])), edge_ids[position]),
        )
    if count == 32:
        quotas = {"cue": 4, "noise": 4, "query": 4, "recurrent": 16, "output": 4}
    else:
        base, remainder = divmod(count, len(GROUP_ORDER))
        quotas = {
            group: base + int(position < remainder)
            for position, group in enumerate(GROUP_ORDER)
        }
    selected = [
        position
        for group in GROUP_ORDER
        for position in ranked[group][: quotas[group]]
    ]
    return tuple(selected)


def _finite_difference_validation(
    *,
    template: RecurrentEventGraph,
    episode: Experiment000Episode,
    weights: Mapping[str, float],
    edge_ids: Sequence[str],
    group_by_edge: Mapping[str, str],
    exact_output_gradient: np.ndarray,
    exact_direction: np.ndarray,
    events: Sequence[UnitEvent],
    root_event_id: str,
    baseline_mask: tuple[object, ...],
    coordinate_count: int,
    step: float,
) -> dict[str, Any]:
    selected = _stratified_coordinates(
        edge_ids=edge_ids,
        group_indices=_edge_groups(template, edge_ids)[1],
        exact_direction=exact_direction,
        count=coordinate_count,
    )
    records: list[dict[str, Any]] = []
    for position in selected:
        edge_id = edge_ids[position]
        coordinate_step = step * max(1.0, abs(float(weights[edge_id])))
        plus_weights = dict(weights)
        minus_weights = dict(weights)
        plus_weights[edge_id] += coordinate_step
        minus_weights[edge_id] -= coordinate_step
        fixed_plus = fixed_event_dag_output(
            events=events, weights=plus_weights, root_event_id=root_event_id
        )
        fixed_minus = fixed_event_dag_output(
            events=events, weights=minus_weights, root_event_id=root_event_id
        )
        fixed_numerical = (fixed_plus - fixed_minus) / (2.0 * coordinate_step)
        fixed_absolute_error = abs(
            fixed_numerical - float(exact_output_gradient[position])
        )
        fixed_scale = max(
            abs(fixed_numerical), abs(float(exact_output_gradient[position]))
        )
        fixed_tolerance = 1e-7 + 1e-4 * fixed_scale
        fixed_relative_error = fixed_absolute_error / max(fixed_scale, 1e-12)
        plus_graph, plus_query = _forward_with_weights(template, plus_weights, episode)
        minus_graph, minus_query = _forward_with_weights(template, minus_weights, episode)
        plus_match = _event_mask_signature(plus_graph.unit_events) == baseline_mask
        minus_match = _event_mask_signature(minus_graph.unit_events) == baseline_mask
        mask_match = plus_match and minus_match
        record: dict[str, Any] = {
            "edge_id": edge_id,
            "group": group_by_edge[edge_id],
            "analytic_derivative": float(exact_output_gradient[position]),
            "coordinate_step": coordinate_step,
            "fixed_dag_derivative": float(fixed_numerical),
            "fixed_dag_absolute_error": float(fixed_absolute_error),
            "fixed_dag_relative_error": float(fixed_relative_error),
            "fixed_dag_tolerance": float(fixed_tolerance),
            "fixed_dag_within_tolerance": bool(
                fixed_absolute_error <= fixed_tolerance
            ),
            "plus_mask_match": plus_match,
            "minus_mask_match": minus_match,
            "emission_masks_match": mask_match,
        }
        if mask_match:
            numerical = (plus_query.activation - minus_query.activation) / (
                2.0 * coordinate_step
            )
            absolute_error = abs(numerical - exact_output_gradient[position])
            scale = max(abs(numerical), abs(float(exact_output_gradient[position])))
            relative_error = absolute_error / max(scale, 1e-12)
            tolerance = 1e-7 + 1e-4 * scale
            record.update(
                {
                    "dynamic_derivative": float(numerical),
                    "dynamic_absolute_error": float(absolute_error),
                    "dynamic_relative_error": float(relative_error),
                    "dynamic_tolerance": float(tolerance),
                    "dynamic_within_tolerance": bool(absolute_error <= tolerance),
                }
            )
        else:
            record.update(
                {
                    "dynamic_derivative": None,
                    "dynamic_absolute_error": None,
                    "dynamic_relative_error": None,
                    "dynamic_tolerance": None,
                    "dynamic_within_tolerance": None,
                }
            )
        records.append(record)

    stable = [record for record in records if record["emission_masks_match"]]
    fixed_passes = sum(record["fixed_dag_within_tolerance"] is True for record in records)
    dynamic_passes = sum(
        record["dynamic_within_tolerance"] is True for record in stable
    )
    stable_fraction = len(stable) / len(records) if records else 0.0
    fixed_pass_fraction = fixed_passes / len(records) if records else 0.0
    dynamic_pass_fraction = dynamic_passes / len(stable) if stable else 0.0
    return {
        "base_step": float(step),
        "step_rule": "h=base_step*max(1,abs(weight))",
        "requested_coordinates": coordinate_count,
        "selected_coordinates": len(records),
        "fixed_dag_within_tolerance": fixed_passes,
        "fixed_dag_pass_fraction": fixed_pass_fraction,
        "fixed_dag_max_absolute_error": (
            max(float(record["fixed_dag_absolute_error"]) for record in records)
            if records
            else None
        ),
        "fixed_dag_max_relative_error": (
            max(float(record["fixed_dag_relative_error"]) for record in records)
            if records
            else None
        ),
        "stable_coordinates": len(stable),
        "stable_fraction": stable_fraction,
        "mask_changed_coordinates": len(records) - len(stable),
        "dynamic_stable_within_tolerance": dynamic_passes,
        "dynamic_pass_fraction": dynamic_pass_fraction,
        "dynamic_max_absolute_error": (
            max(float(record["dynamic_absolute_error"]) for record in stable)
            if stable
            else None
        ),
        "dynamic_max_relative_error": (
            max(float(record["dynamic_relative_error"]) for record in stable)
            if stable
            else None
        ),
        "minimum_gate_stable_fraction": 0.80,
        "minimum_derivative_pass_fraction": 0.99,
        "validation_pass": bool(
            stable_fraction >= 0.80
            and fixed_pass_fraction >= 0.99
            and (not stable or dynamic_pass_fraction >= 0.99)
        ),
        "coordinates": records,
    }


def _optional_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= _METRIC_EPSILON:
        return None
    value = numerator / denominator
    if not math.isfinite(value):
        raise FloatingPointError("non-finite alignment ratio")
    return float(value)


def _vector_metrics(
    *,
    actual: np.ndarray,
    exact_direction: np.ndarray,
    learning_rate: float,
    global_actual_norm_squared: float,
    global_exact_norm_squared: float,
) -> dict[str, Any]:
    actual_norm = float(np.linalg.norm(actual))
    exact_norm = float(np.linalg.norm(exact_direction))
    dot_product = float(actual @ exact_direction)
    cosine = _optional_ratio(dot_product, actual_norm * exact_norm)
    compared = (np.abs(actual) > _METRIC_EPSILON) & (
        np.abs(exact_direction) > _METRIC_EPSILON
    )
    compared_count = int(np.count_nonzero(compared))
    sign_agreement = (
        float(np.mean(np.sign(actual[compared]) == np.sign(exact_direction[compared])))
        if compared_count
        else None
    )
    scaled_exact_norm = learning_rate * exact_norm
    return {
        "coordinate_count": int(actual.size),
        "actual_update_norm": actual_norm,
        "exact_direction_norm": exact_norm,
        "learning_rate_scaled_exact_norm": scaled_exact_norm,
        "cosine": cosine,
        "sign_compared_coordinates": compared_count,
        "sign_agreement": sign_agreement,
        "dot_product": dot_product,
        "norm_ratio": _optional_ratio(actual_norm, exact_norm),
        "learning_rate_scaled_norm_ratio": _optional_ratio(
            actual_norm, scaled_exact_norm
        ),
        "actual_update_norm_fraction": _optional_ratio(
            actual_norm**2, global_actual_norm_squared
        ),
        "exact_direction_norm_fraction": _optional_ratio(
            exact_norm**2, global_exact_norm_squared
        ),
    }


def _alignment_metrics(
    *,
    actual: np.ndarray,
    exact_direction: np.ndarray,
    learning_rate: float,
    group_indices: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    actual_norm_squared = float(actual @ actual)
    exact_norm_squared = float(exact_direction @ exact_direction)
    global_metrics = _vector_metrics(
        actual=actual,
        exact_direction=exact_direction,
        learning_rate=learning_rate,
        global_actual_norm_squared=actual_norm_squared,
        global_exact_norm_squared=exact_norm_squared,
    )
    groups: dict[str, dict[str, Any]] = {}
    for group in GROUP_ORDER:
        selection = np.asarray(group_indices[group], dtype=np.int64)
        groups[group] = _vector_metrics(
            actual=actual[selection],
            exact_direction=exact_direction[selection],
            learning_rate=learning_rate,
            global_actual_norm_squared=actual_norm_squared,
            global_exact_norm_squared=exact_norm_squared,
        )
    return {"global": global_metrics, "groups": groups}


def _loss(target_activation: float, output: float) -> float:
    return 0.5 * (target_activation - output) ** 2


def _replay(
    *,
    template: RecurrentEventGraph,
    weights: Mapping[str, float],
    episode: Experiment000Episode,
    target_activation: float,
    baseline_mask: tuple[object, ...],
) -> dict[str, Any]:
    replay_graph, replay_query = _forward_with_weights(template, weights, episode)
    replay_loss = _loss(target_activation, replay_query.activation)
    return {
        "output": float(replay_query.activation),
        "loss": float(replay_loss),
        "emission_mask_matches_baseline": (
            _event_mask_signature(replay_graph.unit_events) == baseline_mask
        ),
    }


def _replay_comparison(
    *,
    template: RecurrentEventGraph,
    episode: Experiment000Episode,
    target_activation: float,
    baseline_output: float,
    baseline_mask: tuple[object, ...],
    events: Sequence[UnitEvent],
    root_event_id: str,
    edge_ids: Sequence[str],
    weights_before: Mapping[str, float],
    weights_after_ccf: Mapping[str, float],
    actual_delta: np.ndarray,
    exact_direction: np.ndarray,
) -> dict[str, Any]:
    baseline_loss = _loss(target_activation, baseline_output)
    actual_norm = float(np.linalg.norm(actual_delta))
    exact_norm = float(np.linalg.norm(exact_direction))
    if exact_norm > 1e-30 and actual_norm > 0.0:
        proposed_exact_delta = exact_direction * (actual_norm / exact_norm)
    else:
        proposed_exact_delta = np.zeros_like(exact_direction)
    unclipped = np.asarray(
        [weights_before[edge_id] for edge_id in edge_ids], dtype=np.float64
    ) + proposed_exact_delta
    clipped = np.clip(unclipped, -template.weight_clip, template.weight_clip)
    clip_count = int(np.count_nonzero(clipped != unclipped))
    realized_exact_delta = clipped - np.asarray(
        [weights_before[edge_id] for edge_id in edge_ids], dtype=np.float64
    )
    exact_weights = {str(edge_id): float(value) for edge_id, value in weights_before.items()}
    for position, edge_id in enumerate(edge_ids):
        exact_weights[edge_id] = float(clipped[position])
    ccf_simulator = _replay(
        template=template,
        weights=weights_after_ccf,
        episode=episode,
        target_activation=target_activation,
        baseline_mask=baseline_mask,
    )
    exact_simulator = _replay(
        template=template,
        weights=exact_weights,
        episode=episode,
        target_activation=target_activation,
        baseline_mask=baseline_mask,
    )
    ccf_fixed_output = fixed_event_dag_output(
        events=events, weights=weights_after_ccf, root_event_id=root_event_id
    )
    exact_fixed_output = fixed_event_dag_output(
        events=events, weights=exact_weights, root_event_id=root_event_id
    )

    def result_record(fixed_output: float, simulator: Mapping[str, Any]) -> dict[str, Any]:
        fixed_loss = _loss(target_activation, fixed_output)
        simulator_loss = float(simulator["loss"])
        return {
            "fixed_dag_output": float(fixed_output),
            "fixed_dag_loss": float(fixed_loss),
            "fixed_dag_loss_change": float(fixed_loss - baseline_loss),
            "fixed_dag_loss_improvement": float(baseline_loss - fixed_loss),
            "simulator_output": float(simulator["output"]),
            "simulator_loss": simulator_loss,
            "simulator_loss_change": float(simulator_loss - baseline_loss),
            "simulator_loss_improvement": float(baseline_loss - simulator_loss),
            "simulator_emission_mask_matches_baseline": bool(
                simulator["emission_mask_matches_baseline"]
            ),
        }

    ccf = result_record(ccf_fixed_output, ccf_simulator)
    exact = result_record(exact_fixed_output, exact_simulator)
    realized_exact_norm = float(np.linalg.norm(realized_exact_delta))
    return {
        "baseline_output": float(baseline_output),
        "baseline_loss": float(baseline_loss),
        "primary_replay": "fixed_event_dag",
        "ccf": ccf,
        "equal_norm_exact": exact,
        "requested_update_norm": actual_norm,
        "realized_exact_update_norm": realized_exact_norm,
        "weight_clip_count": clip_count,
        "equal_norm_achieved": math.isclose(
            realized_exact_norm, actual_norm, rel_tol=1e-10, abs_tol=1e-12
        ),
    }


def _episode_alignment(
    *,
    graph: RecurrentEventGraph,
    episode: Experiment000Episode,
    all_edge_ids: Sequence[str],
    edge_ids: Sequence[str],
    group_by_edge: Mapping[str, str],
    group_indices: Mapping[str, Sequence[int]],
    finite_difference_coordinates: int,
    finite_difference_step: float,
) -> dict[str, Any]:
    topology_before = graph.topology_hash()
    weights_before_forward_hash = graph.weights_hash()
    weights_before = graph.weights
    query = forward_episode(graph, episode)
    if graph.weights_hash() != weights_before_forward_hash:
        raise RuntimeError("forward pass changed graph weights")
    events = tuple(graph.unit_events)
    dag_hash = _event_dag_sha256(events)
    baseline_mask = _event_mask_signature(events)
    dag_validation = _validate_realized_dag(
        graph=graph,
        events=events,
        weights=weights_before,
        query=query,
    )
    exact_gradient_map = fixed_event_dag_output_gradient(
        events=events,
        weights=weights_before,
        edge_ids=all_edge_ids,
        root_event_id=query.event_id,
    )
    exact_output_gradient = np.asarray(
        [exact_gradient_map[edge_id] for edge_id in edge_ids], dtype=np.float64
    )
    target_activation = _target_activation(episode.target)
    root_signal = float(
        np.clip(
            target_activation - query.activation,
            -graph.credit_limit,
            graph.credit_limit,
        )
    )
    exact_direction = root_signal * exact_output_gradient
    finite_difference = _finite_difference_validation(
        template=graph,
        episode=episode,
        weights=weights_before,
        edge_ids=edge_ids,
        group_by_edge=group_by_edge,
        exact_output_gradient=exact_output_gradient,
        exact_direction=exact_direction,
        events=events,
        root_event_id=query.event_id,
        baseline_mask=baseline_mask,
        coordinate_count=finite_difference_coordinates,
        step=finite_difference_step,
    )

    updates = graph.apply_supervised_credit(episode.target)
    weights_after = graph.weights
    actual_delta = np.asarray(
        [weights_after[edge_id] - weights_before[edge_id] for edge_id in edge_ids],
        dtype=np.float64,
    )
    event_deltas: dict[str, list[float]] = {edge_id: [] for edge_id in edge_ids}
    for update in updates:
        if update.edge_id in event_deltas:
            event_deltas[update.edge_id].append(update.delta)
    event_delta_vector = np.asarray(
        [math.fsum(event_deltas[edge_id]) for edge_id in edge_ids], dtype=np.float64
    )
    if not np.allclose(actual_delta, event_delta_vector, rtol=1e-10, atol=1e-12):
        raise RuntimeError("UpdateEvent deltas do not reconstruct the weight change")
    if _event_dag_sha256(graph.unit_events) != dag_hash:
        raise RuntimeError("credit application mutated the immutable event DAG")
    if graph.topology_hash() != topology_before:
        raise RuntimeError("credit application changed graph topology")
    if not np.all(np.isfinite(actual_delta)):
        raise FloatingPointError("CCF produced a non-finite total update")

    alignment = _alignment_metrics(
        actual=actual_delta,
        exact_direction=exact_direction,
        learning_rate=graph.learning_rate,
        group_indices=group_indices,
    )
    replay = _replay_comparison(
        template=graph,
        episode=episode,
        target_activation=target_activation,
        baseline_output=query.activation,
        baseline_mask=baseline_mask,
        events=events,
        root_event_id=query.event_id,
        edge_ids=edge_ids,
        weights_before=weights_before,
        weights_after_ccf=weights_after,
        actual_delta=actual_delta,
        exact_direction=exact_direction,
    )
    return {
        "episode_id": episode.episode_id,
        "episode_index": episode.index,
        "cue": episode.cue,
        "target": episode.target,
        "query_event_id": query.event_id,
        "query_activation": float(query.activation),
        "prediction": query.prediction,
        "target_activation": target_activation,
        "root_signal": root_signal,
        "event_count": len(events),
        "event_dag_sha256": dag_hash,
        "event_dag_validation": dag_validation,
        "event_dag_unchanged_after_credit": _event_dag_sha256(graph.unit_events) == dag_hash,
        "topology_sha256": topology_before,
        "weights_before_sha256": weights_before_forward_hash,
        "weights_after_sha256": graph.weights_hash(),
        "update_event_count": len(updates),
        "update_audit": {
            "unique_plastic_edges_with_writes": sum(
                bool(event_deltas[edge_id]) for edge_id in edge_ids
            ),
            "plastic_edges_with_multiple_temporal_writes": sum(
                len(event_deltas[edge_id]) > 1 for edge_id in edge_ids
            ),
            "maximum_temporal_writes_to_one_edge": max(
                (len(event_deltas[edge_id]) for edge_id in edge_ids), default=0
            ),
            "event_deltas_reconstruct_final_difference": True,
        },
        "alignment": alignment,
        "finite_difference": finite_difference,
        "replay": replay,
    }


def _mean_optional(values: Iterable[float | None]) -> float | None:
    realized = [float(value) for value in values if value is not None]
    return float(np.mean(realized)) if realized else None


def _median_optional(values: Iterable[float | None]) -> float | None:
    realized = [float(value) for value in values if value is not None]
    return float(median(realized)) if realized else None


def _distribution(values: Iterable[float | None]) -> dict[str, float | None]:
    realized = np.asarray(
        [float(value) for value in values if value is not None], dtype=np.float64
    )
    if not realized.size:
        return {"mean": None, "median": None, "q1": None, "q3": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(realized)),
        "median": float(np.median(realized)),
        "q1": float(np.quantile(realized, 0.25)),
        "q3": float(np.quantile(realized, 0.75)),
        "min": float(np.min(realized)),
        "max": float(np.max(realized)),
    }


def _summarize_episodes(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    group_summary: dict[str, Any] = {}
    for group in GROUP_ORDER:
        metrics = [episode["alignment"]["groups"][group] for episode in episodes]
        group_summary[group] = {
            "mean_cosine": _mean_optional(metric["cosine"] for metric in metrics),
            "mean_sign_agreement": _mean_optional(
                metric["sign_agreement"] for metric in metrics
            ),
            "mean_dot_product": _mean_optional(
                metric["dot_product"] for metric in metrics
            ),
            "mean_norm_ratio": _mean_optional(metric["norm_ratio"] for metric in metrics),
            "mean_actual_update_norm_fraction": _mean_optional(
                metric["actual_update_norm_fraction"] for metric in metrics
            ),
            "mean_exact_direction_norm_fraction": _mean_optional(
                metric["exact_direction_norm_fraction"] for metric in metrics
            ),
        }
    global_metrics = [episode["alignment"]["global"] for episode in episodes]
    stable_fd = sum(
        episode["finite_difference"]["stable_coordinates"] for episode in episodes
    )
    dynamic_passing_fd = sum(
        episode["finite_difference"]["dynamic_stable_within_tolerance"]
        for episode in episodes
    )
    fixed_passing_fd = sum(
        episode["finite_difference"]["fixed_dag_within_tolerance"]
        for episode in episodes
    )
    selected_fd = sum(
        episode["finite_difference"]["selected_coordinates"] for episode in episodes
    )
    fixed_errors = [
        float(record["fixed_dag_absolute_error"])
        for episode in episodes
        for record in episode["finite_difference"]["coordinates"]
    ]
    dynamic_errors = [
        float(record["dynamic_absolute_error"])
        for episode in episodes
        for record in episode["finite_difference"]["coordinates"]
        if record["dynamic_absolute_error"] is not None
    ]
    return {
        "episodes": len(episodes),
        "global": {
            "mean_cosine": _mean_optional(
                metric["cosine"] for metric in global_metrics
            ),
            "median_cosine": _median_optional(
                metric["cosine"] for metric in global_metrics
            ),
            "mean_sign_agreement": _mean_optional(
                metric["sign_agreement"] for metric in global_metrics
            ),
            "mean_dot_product": _mean_optional(
                metric["dot_product"] for metric in global_metrics
            ),
            "fraction_positive_dot_product": float(
                np.mean([metric["dot_product"] > 0.0 for metric in global_metrics])
            ),
            "mean_norm_ratio": _mean_optional(
                metric["norm_ratio"] for metric in global_metrics
            ),
            "cosine_distribution": _distribution(
                metric["cosine"] for metric in global_metrics
            ),
            "dot_product_distribution": _distribution(
                metric["dot_product"] for metric in global_metrics
            ),
            "norm_ratio_distribution": _distribution(
                metric["norm_ratio"] for metric in global_metrics
            ),
        },
        "groups": group_summary,
        "finite_difference": {
            "selected_coordinates": selected_fd,
            "fixed_dag_within_tolerance": fixed_passing_fd,
            "fixed_dag_pass_fraction": (
                fixed_passing_fd / selected_fd if selected_fd else 0.0
            ),
            "stable_coordinates": stable_fd,
            "stable_fraction": stable_fd / selected_fd if selected_fd else 0.0,
            "dynamic_stable_within_tolerance": dynamic_passing_fd,
            "dynamic_pass_fraction": (
                dynamic_passing_fd / stable_fd if stable_fd else 0.0
            ),
            "all_fixed_within_tolerance": selected_fd == fixed_passing_fd,
            "all_stable_dynamic_within_tolerance": stable_fd == dynamic_passing_fd,
            "max_fixed_dag_absolute_error": max(fixed_errors, default=0.0),
            "max_dynamic_absolute_error": max(dynamic_errors, default=0.0),
            "validation_pass": bool(
                selected_fd > 0
                and stable_fd / selected_fd >= 0.80
                and fixed_passing_fd / selected_fd >= 0.99
                and (not stable_fd or dynamic_passing_fd / stable_fd >= 0.99)
            ),
        },
        "replay": {
            "primary_replay": "fixed_event_dag",
            "mean_ccf_fixed_dag_loss_change": _mean_optional(
                episode["replay"]["ccf"]["fixed_dag_loss_change"]
                for episode in episodes
            ),
            "mean_equal_norm_exact_fixed_dag_loss_change": _mean_optional(
                episode["replay"]["equal_norm_exact"]["fixed_dag_loss_change"]
                for episode in episodes
            ),
            "fraction_ccf_fixed_dag_loss_improved": float(
                np.mean(
                    [
                        episode["replay"]["ccf"]["fixed_dag_loss_improvement"] > 0.0
                        for episode in episodes
                    ]
                )
            ),
            "fraction_equal_norm_exact_fixed_dag_loss_improved": float(
                np.mean(
                    [
                        episode["replay"]["equal_norm_exact"][
                            "fixed_dag_loss_improvement"
                        ]
                        > 0.0
                        for episode in episodes
                    ]
                )
            ),
            "fraction_ccf_simulator_mask_stable": float(
                np.mean(
                    [
                        episode["replay"]["ccf"][
                            "simulator_emission_mask_matches_baseline"
                        ]
                        for episode in episodes
                    ]
                )
            ),
            "fraction_exact_simulator_mask_stable": float(
                np.mean(
                    [
                        episode["replay"]["equal_norm_exact"][
                            "simulator_emission_mask_matches_baseline"
                        ]
                        for episode in episodes
                    ]
                )
            ),
            "all_equal_norm_achieved": all(
                episode["replay"]["equal_norm_achieved"] for episode in episodes
            ),
        },
    }


def _run_seed(
    *,
    seed: int,
    episodes_per_seed: int,
    finite_difference_coordinates: int,
    finite_difference_step: float,
) -> dict[str, Any]:
    require_seed_role(seed, SeedRole.CUSTOM)
    pair = generate_stream_pair(
        seed,
        episode_count=episodes_per_seed,
        noise_events=8,
        required_role=SeedRole.CUSTOM,
    )
    graph = build_experiment_000_graph(
        seed,
        mode="full",
        event_log_enabled=False,
        **FROZEN_GRAPH_OPTIONS,
    )
    topology_before = graph.topology_hash()
    initial_weights_hash = graph.weights_hash()
    all_edge_ids = tuple(sorted(graph.weights))
    edge_ids = tuple(
        sorted(edge.edge_id for edge in graph.edges if edge.plastic)
    )
    group_by_edge, group_indices = _edge_groups(graph, edge_ids)
    episode_results = [
        _episode_alignment(
            graph=graph,
            episode=episode,
            all_edge_ids=all_edge_ids,
            edge_ids=edge_ids,
            group_by_edge=group_by_edge,
            group_indices=group_indices,
            finite_difference_coordinates=finite_difference_coordinates,
            finite_difference_step=finite_difference_step,
        )
        for episode in pair.train.episodes
    ]
    final_weights_hash = graph.weights_hash()
    if graph.topology_hash() != topology_before:
        raise RuntimeError("alignment run changed graph topology")
    if graph.ledger["nonfinite_values"]:
        raise FloatingPointError("alignment run recorded non-finite graph values")
    return {
        "seed": seed,
        "seed_role": SeedRole.CUSTOM.value,
        "data_split_used": "train",
        "data_manifest": pair.manifest.to_dict(),
        "graph": {
            "edge_count": len(all_edge_ids),
            "plastic_edge_count": len(edge_ids),
            "parameter_order_sha256": _sha256_json(list(edge_ids)),
            "group_edge_counts": {
                group: len(group_indices[group]) for group in GROUP_ORDER
            },
            "topology_sha256": topology_before,
            "topology_unchanged": graph.topology_hash() == topology_before,
            "initial_weights_sha256": initial_weights_hash,
            "final_weights_sha256": final_weights_hash,
            "weights_changed_by_ccf": initial_weights_hash != final_weights_hash,
            "nonfinite_values": graph.ledger["nonfinite_values"],
        },
        "episodes": episode_results,
        "summary": _summarize_episodes(episode_results),
    }


def run_alignment_diagnostic(
    *,
    seeds: Sequence[int] = DEFAULT_ALIGNMENT_SEEDS,
    episodes_per_seed: int = DEFAULT_EPISODES_PER_SEED,
    finite_difference_coordinates: int = DEFAULT_FINITE_DIFFERENCE_COORDINATES,
    finite_difference_step: float = DEFAULT_FINITE_DIFFERENCE_STEP,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run exact fixed-event alignment on custom diagnostic seeds only."""

    started = perf_counter()
    resolved_seeds = _validate_arguments(
        seeds=seeds,
        episodes_per_seed=episodes_per_seed,
        finite_difference_coordinates=finite_difference_coordinates,
        finite_difference_step=finite_difference_step,
    )
    seed_results = [
        _run_seed(
            seed=seed,
            episodes_per_seed=episodes_per_seed,
            finite_difference_coordinates=finite_difference_coordinates,
            finite_difference_step=finite_difference_step,
        )
        for seed in resolved_seeds
    ]
    all_episodes = [
        episode
        for seed_result in seed_results
        for episode in seed_result["episodes"]
    ]
    aggregate = _summarize_episodes(all_episodes)
    provenance = _source_provenance()
    report: dict[str, Any] = {
        "schema_version": ALIGNMENT_SCHEMA_VERSION,
        "protocol_name": "experiment_000_update_alignment",
        "protocol_version": "alignment-v1",
        "mechanism": "CCF-v0",
        "implementation_revision": "C000.1",
        "scope": "custom_seed_fixed_event_alignment_diagnostic",
        "run_kind": (
            "full" if episodes_per_seed == FULL_EPISODES_PER_SEED else "smoke"
        ),
        "confirmatory_executed": False,
        "seeds": list(resolved_seeds),
        "episodes_per_seed": episodes_per_seed,
        "finite_difference_coordinates_per_episode": finite_difference_coordinates,
        "finite_difference_step": float(finite_difference_step),
        "graph_options": dict(FROZEN_GRAPH_OPTIONS),
        "source_provenance": provenance,
        "backend": {
            "implementation": platform.python_implementation(),
            "python": platform.python_version(),
            "executable": sys.executable,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "float_type": "float64",
            "autograd_used": False,
        },
        "formulas": {
            "event_derivative": (
                "adj[root]=1; dz=adj[event]*(1-a_event^2); "
                "grad[edge]+=dz*a_parent; adj[parent]+=dz*w_edge"
            ),
            "exact_descent_direction": (
                "d_exact=clip(y-a_output,-credit_limit,credit_limit)*D_output"
            ),
            "cosine": "dot(delta_ccf,d_exact)/(||delta_ccf||*||d_exact||)",
            "norm_ratio": "||delta_ccf||/||d_exact||",
            "update_norm_fraction": "||selected_group||^2/||global_vector||^2",
            "loss": "0.5*(y-a_output)^2",
            "equal_norm_exact_step": (
                "delta_exact=d_exact*||delta_ccf||/||d_exact||"
            ),
            "fixed_dag_central_finite_difference": (
                "(a_fixed_DAG(w+h*e)-a_fixed_DAG(w-h*e))/(2*h), "
                "h=1e-6*max(1,abs(w_e))"
            ),
            "dynamic_central_finite_difference": (
                "same central difference through the full simulator; derivative "
                "error accepted only when both emission masks equal baseline"
            ),
        },
        "seed_results": seed_results,
        "aggregate": aggregate,
        "integrity": {
            "all_custom_seeds": True,
            "confirmatory_seed_count": 0,
            "all_topologies_unchanged": all(
                result["graph"]["topology_unchanged"] for result in seed_results
            ),
            "all_event_dags_immutable": all(
                episode["event_dag_unchanged_after_credit"]
                for episode in all_episodes
            ),
            "all_finite": all(
                result["graph"]["nonfinite_values"] == 0 for result in seed_results
            ),
            "finite_difference_validation_pass": aggregate["finite_difference"][
                "validation_pass"
            ],
        },
    }
    report["deterministic_payload_sha256"] = _sha256_json(report)
    report["runtime"] = {
        "seconds": perf_counter() - started,
        "excluded_from_deterministic_payload_sha256": True,
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare CCF-v0 updates with exact fixed-event-DAG descent."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_ALIGNMENT_SEEDS),
        help="custom diagnostic seeds only (default: 42 43 44 45 46)",
    )
    parser.add_argument(
        "--episodes-per-seed", type=int, default=DEFAULT_EPISODES_PER_SEED
    )
    parser.add_argument(
        "--finite-difference-coordinates",
        type=int,
        default=DEFAULT_FINITE_DIFFERENCE_COORDINATES,
    )
    parser.add_argument(
        "--finite-difference-step",
        type=float,
        default=DEFAULT_FINITE_DIFFERENCE_STEP,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)
    report = run_alignment_diagnostic(
        seeds=tuple(args.seeds),
        episodes_per_seed=args.episodes_per_seed,
        finite_difference_coordinates=args.finite_difference_coordinates,
        finite_difference_step=args.finite_difference_step,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALIGNMENT_SCHEMA_VERSION",
    "DEFAULT_ALIGNMENT_SEEDS",
    "DEFAULT_EPISODES_PER_SEED",
    "DEFAULT_FINITE_DIFFERENCE_COORDINATES",
    "DEFAULT_FINITE_DIFFERENCE_STEP",
    "DEFAULT_OUTPUT_PATH",
    "FULL_EPISODES_PER_SEED",
    "fixed_event_dag_output",
    "fixed_event_dag_output_gradient",
    "main",
    "run_alignment_diagnostic",
]
