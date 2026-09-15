"""Development runner for the Experiment-000 delayed-cue gate.

This module deliberately accepts development seeds only.  The confirmatory
partition is blocked in code as well as in its TOML configuration so that a
debugging run cannot consume preregistered evidence by accident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, stdev
from time import perf_counter
from typing import Any

import numpy as np

from .experiment000_data import (
    DEVELOPMENT_SEEDS,
    PROTOCOL_VERSION,
    Experiment000Episode,
    Experiment000Stream,
    SeedRole,
    StreamPair,
    generate_stream_pair,
    require_seed_role,
)
from .recurrent import QueryResult, RecurrentEventGraph, build_experiment_000_graph

REPORT_SCHEMA_VERSION = "experiment-000-development-report-v1"
IMPLEMENTATION_REVISION = "C000.1"
SUPPORTED_METHODS = ("ccf_v0", "ccf_no_trace", "rand")
FROZEN_GRAPH_OPTIONS: dict[str, int | float] = {
    "hidden_count": 64,
    "recurrent_in_degree": 8,
    "input_fan_out": 8,
    "output_fan_in": 16,
    "initial_weight_scale": 0.35,
    "emit_threshold": 1e-3,
    "learning_rate": 0.01,
    "trace_decay": 0.97,
    "route_gain": 0.90,
    "credit_limit": 1.0,
    "max_update": 0.05,
    "weight_clip": 3.0,
    "trace_horizon": 32,
    "hop_limit": 16,
    "epsilon_weight": 1e-12,
    "epsilon_message": 1e-12,
    "epsilon_route": 1e-12,
    "credit_minimum": 1e-8,
}
FROZEN_DEVELOPMENT_PROTOCOL = {
    "implementation_revision": IMPLEMENTATION_REVISION,
    "seeds": list(DEVELOPMENT_SEEDS),
    "methods": list(SUPPORTED_METHODS),
    "episodes_per_split": 2_000,
    "noise_events": 8,
    "edge_delay_ticks": 1,
    "query_readout_latency_ticks": 2,
    "graph_options": FROZEN_GRAPH_OPTIONS,
}
_LEDGER_GAUGES = {"peak_retained_event_records", "peak_pending_messages"}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a TOML configuration without mutating it."""

    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


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
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _bipolar(bit: int) -> float:
    if bit not in (0, 1):
        raise ValueError("Experiment 000 events must be binary")
    return 1.0 if bit else -1.0


def forward_episode(
    graph: RecurrentEventGraph, episode: Experiment000Episode
) -> QueryResult:
    """Run CUE, every NOISE event, and QUERY without exposing the target."""

    graph.begin_episode(episode.episode_id)
    query: QueryResult | None = None
    events = episode.events
    for position, event in enumerate(events):
        if event.kind == "CUE":
            if event.time != 0 or event.value is None:
                raise ValueError("CUE must be the first binary event")
            graph.step({"cue": _bipolar(event.value)})
        elif event.kind == "NOISE":
            if event.value is None:
                raise ValueError("NOISE must carry a binary value")
            graph.step({"noise": _bipolar(event.value)})
        elif event.kind == "QUERY":
            if event.value is not None or position != len(events) - 1:
                raise ValueError("QUERY must be last and must not expose a target")
            query = graph.query()
        else:  # pragma: no cover - the immutable data protocol prevents this
            raise ValueError(f"unknown Experiment 000 event kind: {event.kind}")

    if query is None:
        raise RuntimeError("episode did not contain QUERY")
    expected_tick = len(episode.noise) + 3
    if query.tick != expected_tick:
        raise RuntimeError(
            f"QUERY returned at tick {query.tick}; expected {expected_tick}"
        )
    if graph.audit["queried_output_event_id"] != query.event_id:
        raise RuntimeError("queried output identity does not match the returned event")
    return query


def _ledger_difference(
    later: Mapping[str, int], earlier: Mapping[str, int]
) -> dict[str, int]:
    # Peak fields are high-water gauges, not additive counters.  Subtracting
    # them produces meaningless "split deltas", so they are reported only as
    # cumulative high-water values in the method result.
    keys = sorted(set(later).union(earlier).difference(_LEDGER_GAUGES))
    return {key: int(later.get(key, 0) - earlier.get(key, 0)) for key in keys}


def _new_metric_accumulator() -> dict[str, Any]:
    return {
        "episodes": 0,
        "correct": 0,
        "prediction_0": 0,
        "prediction_1": 0,
        "cue_0_correct": 0,
        "cue_0_total": 0,
        "cue_1_correct": 0,
        "cue_1_total": 0,
        "activation_sum": 0.0,
        "activation_min": math.inf,
        "activation_max": -math.inf,
    }


def _add_metric(
    metrics: dict[str, Any],
    *,
    cue: int,
    prediction: int,
    activation: float | None,
    correct: bool,
) -> None:
    if prediction not in (0, 1):
        raise FloatingPointError("prediction is invalid")
    if activation is not None and not math.isfinite(activation):
        raise FloatingPointError("prediction or activation is invalid")
    metrics["episodes"] += 1
    metrics["correct"] += int(correct)
    metrics[f"prediction_{prediction}"] += 1
    metrics[f"cue_{cue}_correct"] += int(correct)
    metrics[f"cue_{cue}_total"] += 1
    if activation is not None:
        metrics["activation_sum"] += activation
        metrics["activation_min"] = min(metrics["activation_min"], activation)
        metrics["activation_max"] = max(metrics["activation_max"], activation)


def _finish_metrics(
    metrics: Mapping[str, Any], *, training: bool, has_activation: bool = True
) -> dict[str, Any]:
    episodes = int(metrics["episodes"])
    if episodes <= 0:
        raise ValueError("a stream must contain at least one episode")
    result: dict[str, Any] = {
        "episodes": episodes,
        "correct": int(metrics["correct"]),
        "prediction_0": int(metrics["prediction_0"]),
        "prediction_1": int(metrics["prediction_1"]),
        "accuracy": float(metrics["correct"]) / episodes,
        "per_cue_accuracy": {
            "0": float(metrics["cue_0_correct"]) / int(metrics["cue_0_total"]),
            "1": float(metrics["cue_1_correct"]) / int(metrics["cue_1_total"]),
        },
    }
    if has_activation:
        result.update(
            {
                "activation_mean": float(metrics["activation_sum"]) / episodes,
                "activation_min": float(metrics["activation_min"]),
                "activation_max": float(metrics["activation_max"]),
            }
        )
    else:
        result["activation"] = None
    if training:
        result["prequential_accuracy"] = result.pop("accuracy")
    return result


def _episode_record(
    *,
    seed: int,
    method: str,
    split: str,
    episode: Experiment000Episode,
    query: QueryResult,
    update_count: int,
    sum_abs_delta: float,
    max_abs_delta: float,
) -> dict[str, Any]:
    # The target is deliberately first read by the caller only after query()
    # returned.  This record is therefore an audit of terminal reveal, not an
    # input to the forward graph.
    return {
        "seed": seed,
        "method": method,
        "split": split,
        "episode_id": episode.episode_id,
        "noise_stream_id": episode.noise_stream_id,
        "index": episode.index,
        "query_tick": query.tick,
        "query_event_id": query.event_id,
        "activation": query.activation,
        "prediction": query.prediction,
        "target": episode.target,
        "correct": query.prediction == episode.target,
        "update_count": update_count,
        "sum_abs_delta": sum_abs_delta,
        "max_abs_delta": max_abs_delta,
    }


def _write_jsonl(handle: Any | None, record: Mapping[str, Any]) -> None:
    if handle is not None:
        handle.write(_canonical_json(record))
        handle.write("\n")


def _run_learned_method(
    *,
    seed: int,
    method: str,
    pair: StreamPair,
    graph_options: Mapping[str, Any],
    raw_handle: Any | None,
    audit_sample_path: Path | None,
) -> dict[str, Any]:
    mode = "full" if method == "ccf_v0" else "no_trace"
    graph = build_experiment_000_graph(
        seed,
        mode=mode,
        event_log_enabled=audit_sample_path is not None,
        **dict(graph_options),
    )
    topology_before = graph.topology_hash()
    initial_weights = graph.weights_hash()
    start_ledger = graph.ledger
    train_metrics = _new_metric_accumulator()
    update_count = 0
    episodes_with_updates = 0
    total_abs_delta = 0.0
    largest_abs_delta = 0.0
    silent_root_events = 0
    eval_silent_root_events = 0
    peak_max_abs_weight = max(abs(value) for value in graph.weights.values())
    audit_sample: dict[str, Any] | None = None
    started = perf_counter()

    for episode_index, episode in enumerate(pair.train.episodes):
        ledger_before_episode = graph.ledger
        query = forward_episode(graph, episode)
        prediction = query.prediction
        target = episode.target  # terminal reveal occurs after query returned
        correct = prediction == target
        root_event = graph.unit_events_by_id[query.event_id]
        silent_root_events += int(not root_event.edge_traces)
        updates = graph.apply_supervised_credit(target)
        if episode_index == 0 and audit_sample_path is not None:
            audit_sample_path.parent.mkdir(parents=True, exist_ok=True)
            graph.event_log.write_jsonl(audit_sample_path)
            audit_sample = {
                "episode_id": episode.episode_id,
                "path": audit_sample_path.as_posix(),
                "record_count": len(graph.event_log.records),
                "sha256": _file_sha256(audit_sample_path),
            }
            graph.event_log.enabled = False
        peak_max_abs_weight = max(
            peak_max_abs_weight, max(abs(value) for value in graph.weights.values())
        )
        abs_deltas = tuple(abs(update.delta) for update in updates)
        episode_sum = math.fsum(abs_deltas)
        episode_max = max(abs_deltas, default=0.0)
        update_count += len(updates)
        episodes_with_updates += int(bool(updates))
        total_abs_delta += episode_sum
        largest_abs_delta = max(largest_abs_delta, episode_max)
        _add_metric(
            train_metrics,
            cue=episode.cue,
            prediction=prediction,
            activation=query.activation,
            correct=correct,
        )
        tick_delta = graph.ledger["ticks"] - ledger_before_episode["ticks"]
        if tick_delta != len(episode.noise) + 4:
            raise RuntimeError(f"episode consumed {tick_delta} ticks instead of 12")
        _write_jsonl(
            raw_handle,
            _episode_record(
                seed=seed,
                method=method,
                split="train",
                episode=episode,
                query=query,
                update_count=len(updates),
                sum_abs_delta=episode_sum,
                max_abs_delta=episode_max,
            ),
        )

    after_train_ledger = graph.ledger
    trained_weights = graph.weights_hash()
    topology_after_train = graph.topology_hash()
    evaluation_weights_before = graph.weights_hash()
    eval_metrics = _new_metric_accumulator()

    for episode in pair.eval.episodes:
        ledger_before_episode = graph.ledger
        query = forward_episode(graph, episode)
        prediction = query.prediction
        target = episode.target
        correct = prediction == target
        root_event = graph.unit_events_by_id[query.event_id]
        eval_silent_root_events += int(not root_event.edge_traces)
        _add_metric(
            eval_metrics,
            cue=episode.cue,
            prediction=prediction,
            activation=query.activation,
            correct=correct,
        )
        tick_delta = graph.ledger["ticks"] - ledger_before_episode["ticks"]
        if tick_delta != len(episode.noise) + 4:
            raise RuntimeError(f"evaluation episode consumed {tick_delta} ticks instead of 12")
        _write_jsonl(
            raw_handle,
            _episode_record(
                seed=seed,
                method=method,
                split="eval",
                episode=episode,
                query=query,
                update_count=0,
                sum_abs_delta=0.0,
                max_abs_delta=0.0,
            ),
        )

    final_ledger = graph.ledger
    post_eval_weights = graph.weights_hash()
    topology_after_eval = graph.topology_hash()
    if evaluation_weights_before != post_eval_weights:
        raise RuntimeError("evaluation changed learned weights")
    topology_unchanged = (
        topology_before == topology_after_train == topology_after_eval
    )
    if not topology_unchanged:
        raise RuntimeError("the frozen graph topology changed")
    weights = graph.weights.values()
    if not all(math.isfinite(value) for value in weights):
        raise FloatingPointError("non-finite learned weight")

    elapsed = perf_counter() - started
    return {
        "seed": seed,
        "method": method,
        "status": "complete",
        "train": _finish_metrics(train_metrics, training=True),
        "eval": _finish_metrics(eval_metrics, training=False),
        "updates": {
            "count": update_count,
            "episodes_with_updates": episodes_with_updates,
            "sum_abs_delta": total_abs_delta,
            "max_abs_delta": largest_abs_delta,
        },
        "graph": {
            "topology_before": topology_before,
            "topology_after_train": topology_after_train,
            "topology_after_eval": topology_after_eval,
            "topology_unchanged": topology_unchanged,
            "initial_weights": initial_weights,
            "trained_weights": trained_weights,
            "post_eval_weights": post_eval_weights,
            "evaluation_weights_unchanged": evaluation_weights_before == post_eval_weights,
            "final_max_abs_weight": max(abs(value) for value in graph.weights.values()),
            "peak_max_abs_weight": peak_max_abs_weight,
        },
        "stability": {
            "nonfinite_values": final_ledger["nonfinite_values"],
            "clipped_updates": final_ledger["clipped_updates"],
            "train_silent_root_events": silent_root_events,
            "eval_silent_root_events": eval_silent_root_events,
        },
        "ledger": {
            "train_delta": _ledger_difference(after_train_ledger, start_ledger),
            "eval_delta": _ledger_difference(final_ledger, after_train_ledger),
            "cumulative_high_water": {
                key: final_ledger[key] for key in sorted(_LEDGER_GAUGES)
            },
            "total": final_ledger,
        },
        "mechanism_audit_sample": audit_sample,
        "runtime_seconds": elapsed,
    }


def _rand_prediction(seed: int, episode_id: str) -> int:
    material = f"{PROTOCOL_VERSION}|{seed}|rand/eval|{episode_id}".encode()
    return hashlib.sha256(material).digest()[0] & 1


def _run_random_method(
    *, seed: int, stream: Experiment000Stream, raw_handle: Any | None
) -> dict[str, Any]:
    metrics = _new_metric_accumulator()
    action_hasher = hashlib.sha256()
    started = perf_counter()
    for episode in stream.episodes:
        prediction = _rand_prediction(seed, episode.episode_id)
        correct = prediction == episode.target
        _add_metric(
            metrics,
            cue=episode.cue,
            prediction=prediction,
            activation=None,
            correct=correct,
        )
        action_hasher.update(f"{episode.episode_id}:{prediction}\n".encode())
        _write_jsonl(
            raw_handle,
            {
                "seed": seed,
                "method": "rand",
                "split": "eval",
                "episode_id": episode.episode_id,
                "noise_stream_id": episode.noise_stream_id,
                "index": episode.index,
                "query_tick": None,
                "query_event_id": None,
                "activation": None,
                "prediction": prediction,
                "target": episode.target,
                "correct": correct,
                "update_count": 0,
                "sum_abs_delta": 0.0,
                "max_abs_delta": 0.0,
            },
        )
    return {
        "seed": seed,
        "method": "rand",
        "status": "complete",
        "train": None,
        "eval": _finish_metrics(metrics, training=False, has_activation=False),
        "updates": {"count": 0, "episodes_with_updates": 0, "sum_abs_delta": 0.0, "max_abs_delta": 0.0},
        "graph": None,
        "stability": {"nonfinite_values": 0, "clipped_updates": 0, "silent_root_events": 0},
        "ledger": None,
        "action_stream_sha256": action_hasher.hexdigest(),
        "runtime_seconds": perf_counter() - started,
    }


def _aggregate(values: Sequence[float]) -> dict[str, Any]:
    return {
        "values_by_seed": [float(value) for value in values],
        "mean": fmean(values),
        "sample_sd": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "count_at_or_above_0_85": sum(value >= 0.85 for value in values),
    }


def _aggregate_results(
    seeds: Sequence[int], results: Sequence[Mapping[str, Any]], methods: Sequence[str]
) -> dict[str, Any]:
    lookup = {(int(row["seed"]), str(row["method"])): row for row in results}
    by_method: dict[str, Any] = {}
    for method in methods:
        accuracies = [float(lookup[(seed, method)]["eval"]["accuracy"]) for seed in seeds]
        summary = _aggregate(accuracies)
        summary["values_by_seed"] = {
            str(seed): accuracy for seed, accuracy in zip(seeds, accuracies, strict=True)
        }
        by_method[method] = summary

    paired: dict[str, Any] = {}
    if "ccf_v0" in methods and "rand" in methods:
        differences = [
            float(lookup[(seed, "ccf_v0")]["eval"]["accuracy"])
            - float(lookup[(seed, "rand")]["eval"]["accuracy"])
            for seed in seeds
        ]
        paired["ccf_v0_minus_rand"] = {
            "values_by_seed": dict(zip(map(str, seeds), differences, strict=True)),
            "mean": fmean(differences),
        }
    if "ccf_v0" in methods and "ccf_no_trace" in methods:
        differences = [
            float(lookup[(seed, "ccf_v0")]["eval"]["accuracy"])
            - float(lookup[(seed, "ccf_no_trace")]["eval"]["accuracy"])
            for seed in seeds
        ]
        paired["ccf_v0_minus_ccf_no_trace"] = {
            "values_by_seed": dict(zip(map(str, seeds), differences, strict=True)),
            "mean": fmean(differences),
        }
    return {"methods": by_method, "paired_differences": paired}


def _effective_settings(
    config: Mapping[str, Any], *, seeds: Sequence[int], episodes_per_split: int
) -> dict[str, Any]:
    return {
        "protocol": dict(config.get("protocol", {})),
        "seeds": list(seeds),
        "methods": list(config.get("execution", {}).get("paired_methods", SUPPORTED_METHODS)),
        "data": {
            **dict(config.get("data", {})),
            "episodes_per_split": episodes_per_split,
        },
        "graph": dict(config.get("graph", {})),
        "ccf": dict(config.get("ccf", {})),
    }


def _graph_options(config: Mapping[str, Any]) -> dict[str, Any]:
    graph = dict(config.get("graph", {}))
    ccf = dict(config.get("ccf", {}))
    if int(graph.get("edge_delay_ticks", 1)) != 1:
        raise ValueError("Experiment 000 requires one-tick graph edges")
    if int(graph.get("query_readout_latency_ticks", 2)) != 2:
        raise ValueError("Experiment 000 uses a two-tick QUERY readout latency")
    if str(ccf.get("activation", "tanh")) != "tanh":
        raise ValueError("the frozen recurrent implementation uses tanh")
    if str(ccf.get("float_type", "float64")) != "float64":
        raise ValueError("the frozen recurrent implementation uses float64")
    return {
        "hidden_count": int(graph.get("hidden_count", 64)),
        "recurrent_in_degree": int(graph.get("recurrent_in_degree", 8)),
        "input_fan_out": int(graph.get("input_fan_out", 8)),
        "output_fan_in": int(graph.get("output_fan_in", 16)),
        "initial_weight_scale": float(graph.get("initial_weight_scale", 0.35)),
        "emit_threshold": float(ccf.get("emit_threshold", 1e-3)),
        "learning_rate": float(ccf.get("learning_rate", 0.01)),
        "trace_decay": float(ccf.get("trace_decay", 0.97)),
        "route_gain": float(ccf.get("route_gain", 0.90)),
        "credit_limit": float(ccf.get("credit_limit", 1.0)),
        "max_update": float(ccf.get("max_update", 0.05)),
        "weight_clip": float(ccf.get("weight_clip", 3.0)),
        "trace_horizon": int(ccf.get("trace_horizon", 32)),
        "hop_limit": int(ccf.get("hop_limit", 16)),
        "epsilon_weight": float(ccf.get("epsilon_weight", 1e-12)),
        "epsilon_message": float(ccf.get("epsilon_message", 1e-12)),
        "epsilon_route": float(ccf.get("epsilon_route", 1e-12)),
        "credit_minimum": float(ccf.get("credit_minimum", 1e-8)),
    }


def _graph_manifest(seed: int, graph_options: Mapping[str, Any]) -> dict[str, Any]:
    graph = build_experiment_000_graph(
        seed, mode="full", event_log_enabled=False, **dict(graph_options)
    )
    manifest = {
        "seed": seed,
        "nodes": {
            "inputs": list(graph.input_nodes),
            "hidden": list(graph.hidden_nodes),
            "output": graph.output_node,
        },
        "graph_options": dict(graph_options),
        "topology_sha256": graph.topology_hash(),
        "initial_weights_sha256": graph.weights_hash(),
        "edges": [
            {**asdict(edge), "initial_weight": graph.weights[edge.edge_id]}
            for edge in graph.edges
        ],
    }
    manifest["manifest_sha256"] = _sha256_json(manifest)
    return manifest


def _source_provenance() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "src/adaptive_learning_substrate/experiment_000.py",
        "src/adaptive_learning_substrate/experiment000_data.py",
        "src/adaptive_learning_substrate/recurrent.py",
        "src/adaptive_learning_substrate/events.py",
        "docs/CCF_V0.md",
        "docs/EXPERIMENT_001.md",
    )
    files = {
        relative: {
            "sha256": _file_sha256(project_root / relative),
            "bytes": (project_root / relative).stat().st_size,
        }
        for relative in relative_paths
    }
    return {
        "mechanism": "CCF-v0",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "files": files,
        "bundle_sha256": _sha256_json(files),
    }


def _hardware_record() -> dict[str, Any]:
    gpu: str | None = None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if completed.returncode == 0:
            gpu = completed.stdout.strip() or None
    except (FileNotFoundError, subprocess.SubprocessError):
        gpu = None
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "gpu_detected": gpu,
        "backend": "numpy_cpu_float64",
        "gpu_used": False,
    }


def run_experiment000_development(
    config: Mapping[str, Any],
    output_dir: str | Path | None = None,
    *,
    seeds: Iterable[int] | None = None,
    episodes_per_split: int | None = None,
    methods: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Run paired development diagnostics while leaving confirmatory seeds sealed."""

    protocol = dict(config.get("protocol", {}))
    execution = dict(config.get("execution", {}))
    if bool(protocol.get("confirmatory", False)) or protocol.get("scope") != "development_only":
        raise ValueError("this runner accepts development-only configuration")
    if not bool(execution.get("enabled", False)):
        raise ValueError("development execution is disabled")
    if execution.get("required_seed_role") != "development":
        raise ValueError("required_seed_role must be development")

    selected_seeds = tuple(int(seed) for seed in (seeds if seeds is not None else execution.get("seeds", ())))
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("development seeds must be non-empty and unique")
    # This entire validation pass happens before data generation, graph creation,
    # output-directory creation, or any other observable run action.
    for seed in selected_seeds:
        require_seed_role(seed, SeedRole.DEVELOPMENT)

    selected_methods = tuple(
        str(method) for method in (methods if methods is not None else execution.get("paired_methods", ()))
    )
    if not selected_methods or len(set(selected_methods)) != len(selected_methods):
        raise ValueError("methods must be non-empty and unique")
    unknown = tuple(method for method in selected_methods if method not in SUPPORTED_METHODS)
    if unknown:
        raise ValueError(f"unsupported Experiment 000 methods: {unknown}")

    configured_episodes = int(config.get("data", {}).get("episodes_per_split", 2_000))
    episode_count = configured_episodes if episodes_per_split is None else int(episodes_per_split)
    if episode_count <= 0 or episode_count % 2:
        raise ValueError("episodes_per_split must be a positive even integer")
    noise_events = int(config.get("data", {}).get("noise_events", 8))
    if noise_events != 8:
        raise ValueError("Experiment 000 primary development uses exactly eight noise events")

    graph_options = _graph_options(config)
    effective = _effective_settings(
        config, seeds=selected_seeds, episodes_per_split=episode_count
    )
    effective["methods"] = list(selected_methods)
    config_hash = _sha256_json(effective)
    frozen_protocol_sha256 = _sha256_json(FROZEN_DEVELOPMENT_PROTOCOL)
    canonical_development = (
        selected_seeds == DEVELOPMENT_SEEDS
        and selected_methods == SUPPORTED_METHODS
        and episode_count == 2_000
        and noise_events == 8
        and graph_options == FROZEN_GRAPH_OPTIONS
        and int(config.get("graph", {}).get("edge_delay_ticks", 1)) == 1
        and int(config.get("graph", {}).get("query_readout_latency_ticks", 2)) == 2
    )

    target: Path | None = None
    raw_handle: Any | None = None
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        raw_handle = (target / "episodes.jsonl").open("w", encoding="utf-8", newline="\n")
    write_audit_sample = bool(config.get("output", {}).get("write_audit_sample", False))

    results: list[dict[str, Any]] = []
    manifests: dict[str, Any] = {}
    graph_manifests: dict[str, Any] = {}
    started = perf_counter()
    try:
        for seed in selected_seeds:
            complete_graph_manifest = _graph_manifest(seed, graph_options)
            graph_manifest_name = f"graph_manifests/seed-{seed}.json"
            if target is not None:
                graph_target = target / graph_manifest_name
                graph_target.parent.mkdir(parents=True, exist_ok=True)
                graph_target.write_text(
                    json.dumps(
                        complete_graph_manifest,
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            graph_manifests[str(seed)] = {
                "path": graph_manifest_name if target is not None else None,
                "manifest_sha256": complete_graph_manifest["manifest_sha256"],
                "topology_sha256": complete_graph_manifest["topology_sha256"],
                "initial_weights_sha256": complete_graph_manifest[
                    "initial_weights_sha256"
                ],
                "edge_count": len(complete_graph_manifest["edges"]),
            }
            pair = generate_stream_pair(
                seed,
                episode_count=episode_count,
                noise_events=noise_events,
                required_role=SeedRole.DEVELOPMENT,
            )
            manifests[str(seed)] = pair.manifest.to_dict()
            initial_graphs: dict[str, tuple[str, str]] = {}
            for method in selected_methods:
                if method == "rand":
                    row = _run_random_method(
                        seed=seed, stream=pair.eval, raw_handle=raw_handle
                    )
                else:
                    audit_path = None
                    if target is not None and write_audit_sample:
                        audit_path = (
                            target
                            / "mechanism_audit_samples"
                            / f"seed-{seed}-{method}-train-episode-0.jsonl"
                        )
                    row = _run_learned_method(
                        seed=seed,
                        method=method,
                        pair=pair,
                        graph_options=graph_options,
                        raw_handle=raw_handle,
                        audit_sample_path=audit_path,
                    )
                    initial_graphs[method] = (
                        row["graph"]["topology_before"],
                        row["graph"]["initial_weights"],
                    )
                results.append(row)
            if (
                {"ccf_v0", "ccf_no_trace"}.issubset(initial_graphs)
                and initial_graphs["ccf_v0"] != initial_graphs["ccf_no_trace"]
            ):
                raise RuntimeError("paired learned methods did not share initialization")
    finally:
        if raw_handle is not None:
            raw_handle.close()

    elapsed = perf_counter() - started
    source_provenance = _source_provenance()
    hardware = _hardware_record()
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "mechanism": "CCF-v0",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "status": "development_only",
        "confirmatory_executed": False,
        "verdict": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "canonical_development_run": canonical_development,
        "seeds": list(selected_seeds),
        "methods": list(selected_methods),
        "effective_config": effective,
        "effective_config_sha256": config_hash,
        "frozen_development_protocol_sha256": frozen_protocol_sha256,
        "source_provenance": source_provenance,
        "data_manifests": manifests,
        "graph_manifests": graph_manifests,
        "seed_results": results,
        "aggregate": _aggregate_results(selected_seeds, results, selected_methods),
        "integrity": {
            "all_topologies_unchanged": all(
                row["graph"] is None or row["graph"]["topology_unchanged"]
                for row in results
            ),
            "all_evaluations_weight_frozen": all(
                row["graph"] is None
                or row["graph"]["evaluation_weights_unchanged"]
                for row in results
            ),
            "all_finite": all(
                row["stability"]["nonfinite_values"] == 0 for row in results
            ),
            "confirmatory_seed_count": 0,
        },
        "archive": {
            "episode_log": "episodes.jsonl" if target is not None else None,
            "mechanism_event_log": (
                "mechanism_audit_samples/" if write_audit_sample and target is not None else None
            ),
            "mechanism_event_log_status": (
                "first_training_episode_per_learned_method_and_seed"
                if write_audit_sample and target is not None
                else "disabled"
            ),
            "protocol_archive_complete": False,
            "artifact_manifest": "artifact_manifest.json" if target is not None else None,
        },
        "runtime": {
            "seconds": elapsed,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "hardware": hardware,
            "completed_utc": datetime.now(UTC).isoformat(),
        },
    }

    if target is not None:
        (target / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (target / "config.snapshot.json").write_text(
            json.dumps(effective, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (target / "data_manifests.json").write_text(
            json.dumps(manifests, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (target / "provenance.json").write_text(
            json.dumps(source_provenance, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        artifact_paths = [
            path
            for path in sorted(target.rglob("*"))
            if path.is_file() and path.name != "artifact_manifest.json"
        ]
        artifact_manifest = {
            "schema_version": "experiment-000-artifact-manifest-v1",
            "implementation_revision": IMPLEMENTATION_REVISION,
            "files": [
                {
                    "path": path.relative_to(target).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
                for path in artifact_paths
            ],
        }
        artifact_manifest["bundle_sha256"] = _sha256_json(
            artifact_manifest["files"]
        )
        (target / "artifact_manifest.json").write_text(
            json.dumps(
                artifact_manifest, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the development-only recurrent Experiment-000 gate."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_000_development.toml"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="development diagnostic override; must be positive and even",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="development seed subset; confirmatory seeds are rejected",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    output = args.output
    if output is None:
        configured = config.get("output", {}).get("directory")
        output = Path(configured) if configured else Path("runs/experiment_000_development")
    report = run_experiment000_development(
        config,
        output,
        seeds=args.seeds,
        episodes_per_split=args.episodes,
    )
    concise = {
        "status": report["status"],
        "verdict": report["verdict"],
        "canonical_development_run": report["canonical_development_run"],
        "seeds": report["seeds"],
        "episodes_per_split": report["effective_config"]["data"]["episodes_per_split"],
        "aggregate": report["aggregate"],
        "integrity": report["integrity"],
        "output": str(output.resolve()),
    }
    print(json.dumps(concise, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "SUPPORTED_METHODS",
    "forward_episode",
    "load_config",
    "run_experiment000_development",
]
