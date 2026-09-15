"""Preregistered architecture-only readout-decoupled local-trace experiment.

Implements ``readout-trace-v1a`` (``docs/EXPERIMENT_000_READOUT_TRACE_PROTOCOL.md``).
The native recurrent forward/emission path is byte-identical to ``recurrent.py``;
each hidden unit keeps one bounded passive EMA of its own activations that is
read only at ``QUERY`` through the 16 frozen output edges. CCF is never called.
Only custom seeds 65--69 and the frozen ``0.25..0.95`` grid are accepted.

The stable probe, ridge, distribution, matrix, hashing, gate-threshold, and
determinism helpers are imported from :mod:`experiment_000_slow_state` and
:mod:`experiment_000_memory_probe` so the numerical contract matches the
slow-state run one-for-one.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
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
from .recurrent import RecurrentEventGraph, build_experiment_000_graph

SCHEMA_VERSION = "experiment-000-readout-decoupled-local-trace-v1a"
PROTOCOL_NAME = "experiment_000_readout_decoupled_local_trace"
PROTOCOL_VERSION = "readout-trace-v1a"
REGISTERED_SEEDS: tuple[int, ...] = (65, 66, 67, 68, 69)
REGISTERED_RETENTIONS: tuple[float, ...] = (0.25, 0.50, 0.75, 0.90, 0.95)
NATIVE_CONDITION = "native"
SMOKE_PAIRS_PER_SPLIT = 2
FULL_PAIRS_PER_SPLIT = 100
DEFAULT_PAIRS_PER_SPLIT = SMOKE_PAIRS_PER_SPLIT
DEFAULT_RIDGE_ALPHA = 1e-3
DEFAULT_OUTPUT_PATH = Path("artifacts/experiment_000/readout_trace/report.json")
BLANK_TAIL_TICKS = 256
BLANK_TAIL_FINAL_WINDOW_TICKS = 64
NOISE_EVENTS = 8

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
    "61b9b32d0baf86ad65285f183dbd2d9b550afca12032cb2f53ff64d5bdc861aa"
)
FROZEN_CONFIG_SHA256 = (
    "3242e96b77539a96bf800ae58230bc2235422b3b2fa7e1a596fe9f5db6c6665c"
)

FORBIDDEN_SEEDS: dict[str, tuple[int, ...]] = {
    "development_0_4": (0, 1, 2, 3, 4),
    "alignment_42_46": (42, 43, 44, 45, 46),
    "memory_sweep_50_54": (50, 51, 52, 53, 54),
    "slow_state_60_64": (60, 61, 62, 63, 64),
    "confirmatory_1000_1019": tuple(range(1000, 1020)),
}

_TRACE_TOUCH_FIELDS = (
    "local_trace_read_touches",
    "local_trace_decay_touches",
    "local_trace_write_touches",
    "local_trace_reset_touches",
    "local_trace_observation_touches",
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
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    resolved_seeds = tuple(int(value) for value in seeds)
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
    return resolved_seeds, resolved_retentions


def frozen_source_integrity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    recurrent_path = root / "src/adaptive_learning_substrate/recurrent.py"
    ccf_path = root / "docs/CCF_V0.md"
    protocol_path = root / "docs/EXPERIMENT_000_READOUT_TRACE_PROTOCOL.md"
    config_path = root / "configs/experiment_000_readout_trace.toml"
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
        "src/adaptive_learning_substrate/experiment_000_readout_trace.py",
        "src/adaptive_learning_substrate/readout_trace_recurrent.py",
        "src/adaptive_learning_substrate/experiment_000_slow_state.py",
        "src/adaptive_learning_substrate/experiment_000_memory_probe.py",
        "src/adaptive_learning_substrate/recurrent.py",
        "src/adaptive_learning_substrate/experiment_000.py",
        "src/adaptive_learning_substrate/experiment000_data.py",
        "docs/EXPERIMENT_000_READOUT_TRACE_PROTOCOL.md",
        "configs/experiment_000_readout_trace.toml",
    )
    files: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"readout-trace provenance file missing: {relative}")
        files[relative] = {"sha256": _file_sha256(path), "bytes": path.stat().st_size}
    return {"files": files, "bundle_sha256": _sha256_json(files)}


# ----------------------------------------------------------------------
# Forward collection


def _counterfactual_episode(
    base: Experiment000Episode, *, cue: int, condition: str
) -> Experiment000Episode:
    return replace(
        base,
        episode_id=f"{base.episode_id}-readout-trace-{condition}-cue-{cue}",
        cue=cue,
        target=cue,
    )


def _candidate_features_and_output(
    graph: ReadoutTraceRecurrentEventGraph,
    query_tick: int,
    feature_edge_ids: tuple[str, ...],
) -> tuple[list[tuple[str, float]], float]:
    features = list(graph.query_trace_features(feature_edge_ids, query_tick))
    weights = graph.weights
    preactivation = math.fsum(weights[edge_id] * value for edge_id, value in features)
    return features, math.tanh(preactivation)


def _collect_split(
    graph: ReadoutTraceRecurrentEventGraph,
    stream: Experiment000Stream,
    *,
    condition: str,
    is_native: bool,
    feature_edge_ids: tuple[str, ...],
) -> dict[str, Any]:
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
    trace_before = graph.trace_ledger

    for base in stream.episodes:
        row: dict[str, Any] = {
            "pair_index": int(base.index),
            "base_episode_id": base.episode_id,
            "noise_stream_id": base.noise_stream_id,
        }
        for cue in (0, 1):
            episode = _counterfactual_episode(base, cue=cue, condition=condition)
            episode_ledger_before = graph.ledger
            episode_trace_before = graph.trace_ledger
            query = forward_episode(graph, episode)
            episode_ledger_after = graph.ledger

            if is_native:
                extracted = list(extract_query_features(graph, query))
                output = float(query.activation)
            else:
                extracted, output = _candidate_features_and_output(
                    graph, query.tick, feature_edge_ids
                )
            if tuple(edge_id for edge_id, _ in extracted) != feature_edge_ids:
                raise RuntimeError("query feature order changed during readout-trace run")
            feature = np.asarray([value for _, value in extracted], dtype=np.float64)
            if not np.all(np.isfinite(feature)) or not math.isfinite(output):
                raise FloatingPointError("non-finite readout-trace forward value")
            if any(
                not math.isfinite(float(event.preactivation))
                or not math.isfinite(float(event.activation))
                or abs(float(event.activation)) > 1.0 + 1e-15
                for event in graph.unit_events
            ):
                raise FloatingPointError("event left the finite tanh range")

            records = graph.trace_activation_records
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

            trace_after = graph.trace_ledger
            row[f"cue_{cue}_output_activation"] = output
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

            if cue == 0:
                cue_zero_features.append(feature)
                cue_zero_outputs.append(output)
            else:
                cue_one_features.append(feature)
                cue_one_outputs.append(output)
        pair_rows.append(row)

    ledger_after = graph.ledger
    trace_ledger_after = graph.trace_ledger
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
    return {"paired": paired, "activity": activity, "pair_rows": pair_rows}


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
    graph: ReadoutTraceRecurrentEventGraph,
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
        polarity_trace_before = graph.trace_ledger
        cue_label = "negative" if cue_input < 0.0 else "positive"
        graph.begin_episode(f"readout-trace-blank-tail-{condition}-{cue_label}")
        graph.step({"cue": cue_input})
        hidden_counts: list[int] = []
        for _ in range(BLANK_TAIL_TICKS):
            result = graph.step({})
            emitted_nodes = tuple(
                event_id.rsplit(":", 2)[1] for event_id in result.emitted_event_ids
            )
            hidden_counts.append(sum(node in hidden_nodes for node in emitted_nodes))

        snapshot_before = graph.trace_snapshot
        effective = graph.effective_trace(BLANK_TAIL_TICKS)
        effective_maximum = max(abs(value) for value in effective.values())
        snapshot_after = graph.trace_snapshot
        if snapshot_before != snapshot_after:
            raise RuntimeError("effective-trace observation mutated local trace")

        # Advance to the forced-output tick and read the condition's own readout.
        graph._advance({"query": 1.0})
        graph._advance({})
        graph._advance({}, force_output=True)
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
            extracted, wake_output = _candidate_features_and_output(
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
                field: int(graph.trace_ledger[field] - polarity_trace_before[field])
                for field in _TRACE_TOUCH_FIELDS
            },
            "weights_unchanged": weights_before == graph.weights_hash(),
            "structural_unchanged": structural_before == _structural_mask_sha256(graph),
        }
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


def _run_condition(
    *,
    native_graph: RecurrentEventGraph,
    retention: float | None,
    stream_pair: Any,
    ridge_alpha: float,
) -> dict[str, Any]:
    condition = condition_name(retention)
    is_native = retention is None
    graph = clone_with_readout_trace(
        native_graph, 0.0 if retention is None else retention, event_log_enabled=False
    )
    native_matrix = _recurrent_matrix_metrics(native_graph)
    condition_matrix = _recurrent_matrix_metrics(graph)
    feature_edge_ids = _output_feature_edge_ids(graph)
    native_feature_edge_ids = _output_feature_edge_ids(native_graph)

    topology_before = graph.topology_hash()
    structural_before = _structural_mask_sha256(graph)
    weights_before = graph.weights_hash()
    framed_before = _edge_weight_sha256(graph, kinds=("input", "recurrent", "output"))

    train = _collect_split(
        graph, stream_pair.train, condition=condition, is_native=is_native,
        feature_edge_ids=feature_edge_ids,
    )
    weights_mid = graph.weights_hash()
    evaluation = _collect_split(
        graph, stream_pair.eval, condition=condition, is_native=is_native,
        feature_edge_ids=feature_edge_ids,
    )
    weights_after_eval = graph.weights_hash()
    blank_tail = _blank_tail_stability(
        graph, condition=condition, is_native=is_native,
        feature_edge_ids=feature_edge_ids,
    )
    weights_after = graph.weights_hash()
    framed_after = _edge_weight_sha256(graph, kinds=("input", "recurrent", "output"))
    topology_after = graph.topology_hash()
    structural_after = _structural_mask_sha256(graph)

    train_metrics = paired_retention_metrics(train["paired"])
    eval_metrics = paired_retention_metrics(evaluation["paired"])
    ridge = strict_ridge_readout(train["paired"], evaluation["paired"], alpha=ridge_alpha)
    combined_activity = _combine_activity(train["activity"], evaluation["activity"])

    ledger = graph.ledger
    trace_ledger = graph.trace_ledger
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
        "retention_immutable": is_native or graph.trace_retention == retention,
        "trace_bound_ok": graph.trace_bound_ok,
        "trace_resets_valid": is_native or graph.trace_resets_valid,
        "trace_reset_count_matches_episodes": (
            graph.trace_reset_count == episodes if not is_native
            else graph.trace_reset_count == 0
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
            "hidden_units": len(graph.hidden_nodes),
            "feature_dimension": len(feature_edge_ids),
            "feature_edge_ids": list(feature_edge_ids),
            "topology_sha256": topology_before,
            "structural_mask_sha256": structural_before,
            "initial_weights_sha256": weights_before,
            "final_weights_sha256": weights_after,
            "recurrent_matrix": condition_matrix,
            "hyperparameters": _constructor_hyperparameters(graph),
        },
        "train": train_metrics,
        "eval": eval_metrics,
        "ridge_readout": ridge,
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


def _seed_gate(candidate: Mapping[str, Any]) -> dict[str, Any]:
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
    return {"criteria": criteria, "pass": all(criteria.values()), "eligible": True}


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
    recurrent_activity_equal = True
    for split in ("train", "eval"):
        candidate_split = condition["activity"][split]
        native_split = native["activity"][split]
        candidate_pair_rows = condition["pair_rows"][split]
        native_pair_rows = native["pair_rows"][split]
        matched_event_ratios: list[float] = []
        matched_forward_ratios: list[float] = []
        for candidate_pair, native_pair in zip(
            candidate_pair_rows, native_pair_rows, strict=True
        ):
            if (
                candidate_pair["base_episode_id"] != native_pair["base_episode_id"]
                or candidate_pair["noise_stream_id"] != native_pair["noise_stream_id"]
            ):
                raise RuntimeError("candidate/native activity rows are not paired")
            for cue in (0, 1):
                if (
                    candidate_pair[f"cue_{cue}_emitted_unit_events"]
                    != native_pair[f"cue_{cue}_emitted_unit_events"]
                    or candidate_pair[f"cue_{cue}_forward_edge_touches"]
                    != native_pair[f"cue_{cue}_forward_edge_touches"]
                ):
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
        "activity_by_split": split_activity,
    }
    condition["invariants"]["recurrent_activity_equals_native"] = recurrent_activity_equal
    condition["seed_gate"] = _seed_gate(condition)


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
    native["paired_vs_native"] = {
        "eval_cue_to_noise_feature_ratio_multiple": 1.0,
        "eval_cue_to_noise_feature_ratio_difference": 0.0,
        "eval_absolute_cue_feature_delta_rms_multiple": 1.0,
        "ridge_eval_accuracy_difference": 0.0,
        "eval_output_cue_delta_to_sd_difference": 0.0,
        "mean_event_count_inflation": 1.0,
        "recurrent_activity_equals_native": True,
        "activity_by_split": {
            split: {
                "total_emitted_event_ratio": 1.0,
                "maximum_matched_episode_event_count_ratio": 1.0,
                "forward_edge_touch_ratio": 1.0,
                "maximum_matched_episode_forward_edge_touch_ratio": 1.0,
            }
            for split in ("train", "eval")
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


def _aggregate_conditions(
    seed_results: Sequence[Mapping[str, Any]],
    retentions: tuple[float, ...],
    *,
    pairs_per_split: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    condition_ids = tuple(condition_name(value) for value in (None, *retentions))
    by_condition: dict[str, list[Mapping[str, Any]]] = {c: [] for c in condition_ids}
    for seed_result in seed_results:
        for row in seed_result["condition_results"]:
            by_condition[row["condition"]].append(row)
    for condition, rows in by_condition.items():
        if len(rows) != len(seed_results):
            raise RuntimeError(f"condition {condition} missing from a seed")

    native_rows = by_condition[NATIVE_CONDITION]
    native_ratio_median = _distribution(
        e["eval"]["cue_to_noise_feature_ratio"] for e in native_rows
    )["median"]
    native_absolute_median = _distribution(
        e["eval"]["cue_feature_delta_rms"] for e in native_rows
    )["median"]
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
            paired = {
                "cue_ratio_multiple_distribution": _distribution(cue_multiples),
                "absolute_cue_delta_multiple_distribution": _distribution(abs_multiples),
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
            criteria = {
                "full_100_pair_run": pairs_per_split == FULL_PAIRS_PER_SPLIT,
                "native_control_valid": native_control_valid,
                "median_absolute_cue_delta_at_least_3x_median_native": (
                    paired["median_absolute_cue_delta_multiple"] is not None
                    and paired["median_absolute_cue_delta_multiple"]
                    >= GATE_THRESHOLDS["paired_native_absolute_cue_delta_multiple_minimum"]
                ),
                "at_least_4_of_5_seeds_absolute_cue_delta_at_least_3x_native": (
                    abs_threefold
                    >= int(GATE_THRESHOLDS["within_seed_threefold_improvement_count_minimum"])
                ),
                "median_eval_cue_ratio_at_least_0_50": (
                    median_ratio is not None
                    and median_ratio
                    >= GATE_THRESHOLDS["eval_cue_to_noise_feature_ratio_minimum"]
                ),
                "median_eval_cue_ratio_at_least_3x_median_native": (
                    paired["median_cue_ratio_multiple"] is not None
                    and paired["median_cue_ratio_multiple"]
                    >= GATE_THRESHOLDS["paired_native_cue_ratio_multiple_minimum"]
                ),
                "at_least_4_of_5_seeds_cue_ratio_at_least_3x_native": (
                    cue_threefold
                    >= int(GATE_THRESHOLDS["within_seed_threefold_improvement_count_minimum"])
                ),
                "median_ridge_eval_accuracy_at_least_0_70": (
                    distributions["ridge_eval_accuracy"]["median"] is not None
                    and distributions["ridge_eval_accuracy"]["median"]
                    >= GATE_THRESHOLDS["ridge_eval_accuracy_minimum"]
                ),
                "median_eval_output_ratio_at_least_0_50": (
                    distributions["eval_output_cue_delta_to_sd"]["median"] is not None
                    and distributions["eval_output_cue_delta_to_sd"]["median"]
                    >= GATE_THRESHOLDS["eval_output_cue_delta_to_sd_minimum"]
                ),
                "every_seed_split_activity_guard_passes": all_activity_guards,
                "tail_effective_state_and_wake_pass_for_every_seed": all_tail,
                "all_frozen_state_and_finite_invariants_pass": all_invariants,
            }
            aggregate["paired_vs_native"] = paired
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
        else (f"SELECTED:{selected['condition']}" if selected else "NO_SELECTION")
    )
    strongest_gate_reached = _strongest_retention_gate(aggregate_rows)
    selection = {
        "rule": "lowest retention satisfying every preregistered full-run gate",
        "final_status": final_status,
        "selected_condition": None if selected is None else selected["condition"],
        "selected_retention": None if selected is None else selected["target_retention"],
        "passing_conditions": [row["condition"] for row in passing],
        "any_target_passed": bool(passing),
        "strongest_retention_gate_reached": strongest_gate_reached,
        "smoke_selection_forbidden": pairs_per_split != FULL_PAIRS_PER_SPLIT,
    }
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


def run_zero_retention_equivalence() -> dict[str, Any]:
    """Compare the rho=0 instance and the native graph at every smoke step."""

    rows: list[dict[str, Any]] = []
    for seed in REGISTERED_SEEDS:
        pair = generate_stream_pair(
            seed, episode_count=SMOKE_PAIRS_PER_SPLIT, noise_events=NOISE_EVENTS,
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
                    qn = forward_episode(native, episode)
                    qz = forward_episode(zero, episode)
                    fn = extract_query_features(native, qn)
                    fz = extract_query_features(zero, qz)
                    if (
                        (qn.activation, qn.prediction) != (qz.activation, qz.prediction)
                        or native.unit_events != zero.unit_events
                        or native.ledger != zero.ledger
                        or fn != fz
                    ):
                        raise RuntimeError("rho=0 instance differs from native")
                    trace_ledger = zero.trace_ledger
                    if any(trace_ledger[field] != 0 for field in _TRACE_TOUCH_FIELDS):
                        raise RuntimeError("rho=0 fast path touched local trace")
                    rows.append(
                        {
                            "seed": seed,
                            "split": stream.split,
                            "base_episode_id": base.episode_id,
                            "cue": cue,
                            "native_feature_sha256": _array_bundle_sha256(
                                [v for _, v in fn]
                            ),
                        }
                    )
    return {
        "pass": True,
        "seeds": list(REGISTERED_SEEDS),
        "compared_counterfactual_episodes": len(rows),
        "step_results_bitwise_equal": True,
        "emitted_events_equal": True,
        "native_readout_features_equal": True,
        "legacy_ledgers_equal": True,
        "all_trace_touches_zero": True,
        "rows_sha256": _sha256_json(rows),
    }


def verify_deterministic_full_runs(
    first: Mapping[str, Any] | str | Path,
    second: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    def load(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return value
        return json.loads(Path(value).read_text(encoding="utf-8"))

    left = load(first)
    right = load(second)
    if left.get("run_kind") != "full" or right.get("run_kind") != "full":
        raise ValueError("determinism verification requires two full reports")

    def recompute(report: Mapping[str, Any]) -> str:
        payload = dict(report)
        payload.pop("nondeterministic_provenance", None)
        payload.pop("deterministic_payload_sha256", None)
        return _sha256_json(payload)

    left_hash = left.get("deterministic_payload_sha256")
    right_hash = right.get("deterministic_payload_sha256")
    left_recomputed = recompute(left)
    right_recomputed = recompute(right)
    left_ok = isinstance(left_hash, str) and left_hash == left_recomputed
    right_ok = isinstance(right_hash, str) and right_hash == right_recomputed
    hashes_match = left_ok and right_ok and left_hash == right_hash
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
        "first_self_hash_valid": left_ok,
        "second_self_hash_valid": right_ok,
        "selection_statuses_match": left.get("status") == right.get("status"),
        "valid_terminal_status": status == "NO_SELECTION"
        or status.startswith("SELECTED:"),
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
) -> dict[str, Any]:
    started = perf_counter()
    resolved_seeds, resolved_retentions = _validate_arguments(
        seeds=seeds, retentions=retentions, pairs_per_split=pairs_per_split,
        ridge_alpha=ridge_alpha,
    )
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
    provenance = _source_provenance()
    zero_equivalence = run_zero_retention_equivalence()

    seed_results = [
        _run_seed(
            seed=seed, retentions=resolved_retentions,
            pairs_per_split=pairs_per_split, ridge_alpha=ridge_alpha,
        )
        for seed in resolved_seeds
    ]
    aggregate_conditions, selection = _aggregate_conditions(
        seed_results, resolved_retentions, pairs_per_split=pairs_per_split
    )
    all_rows = [
        row for seed_result in seed_results for row in seed_result["condition_results"]
    ]

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "run_kind": "full" if pairs_per_split == FULL_PAIRS_PER_SPLIT else "smoke",
        "scope": "custom_seed_architecture_diagnostic",
        "mechanism": "CCF-v0 present but never called",
        "seeds": list(resolved_seeds),
        "retention_coefficients": list(resolved_retentions),
        "pairs_per_split": pairs_per_split,
        "ridge_alpha": ridge_alpha,
        "noise_events": NOISE_EVENTS,
        "confirmatory_executed": False,
        "confirmatory_seed_count": 0,
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
        "source_provenance": provenance,
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
            "all_topologies_match_native": all(
                row["invariants"]["topology_matches_native"] for row in all_rows
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
            "native_control_valid": all(
                all(sr["condition_results"][0]["invariants"].values())
                and sr["condition_results"][0]["blank_tail_stability"][
                    "all_tail_state_wake_gates_pass"
                ]
                for sr in seed_results
            ),
            "zero_retention_equivalence_pass": zero_equivalence["pass"],
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
        payload_bytes = (
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        target.write_bytes(payload_bytes)
        import hashlib

        sidecar = target.with_suffix(".sha256")
        sidecar.write_bytes(
            (
                json.dumps(
                    {
                        "algorithm": "sha256",
                        "report_file": target.name,
                        "report_sha256": hashlib.sha256(payload_bytes).hexdigest(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        )
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
    parser.add_argument(
        "--pairs-per-split", type=int, default=DEFAULT_PAIRS_PER_SPLIT
    )
    parser.add_argument("--ridge-alpha", type=float, default=DEFAULT_RIDGE_ALPHA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)
    report = run_readout_trace_experiment(
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
    "FULL_PAIRS_PER_SPLIT",
    "PROTOCOL_VERSION",
    "REGISTERED_RETENTIONS",
    "REGISTERED_SEEDS",
    "SMOKE_PAIRS_PER_SPLIT",
    "clone_with_readout_trace",
    "condition_name",
    "frozen_source_integrity",
    "main",
    "run_readout_trace_experiment",
    "run_zero_retention_equivalence",
    "verify_deterministic_full_runs",
]
