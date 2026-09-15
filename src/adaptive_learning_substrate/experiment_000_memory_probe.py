"""Paired-counterfactual cue-retention probe for Experiment 000.

The probe never trains the graph.  For every independently generated noise
stream it runs the frozen recurrent graph twice, once with ``CUE(0)`` and once
with ``CUE(1)``.  The two runs therefore differ only in the cue.  The feature
vector is the ordered vector of hidden messages that reaches the forced query
output; an absent message is represented by zero.

Only development seeds are accepted.  In particular, the confirmatory seed
partition is rejected before an output path is created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .experiment000_data import (
    DEVELOPMENT_SEEDS,
    Experiment000Episode,
    Experiment000Stream,
    SeedRole,
    generate_stream_pair,
    require_seed_role,
)
from .experiment_000 import FROZEN_GRAPH_OPTIONS, forward_episode
from .recurrent import QueryResult, RecurrentEventGraph, build_experiment_000_graph

MEMORY_PROBE_SCHEMA_VERSION = "experiment-000-memory-probe-v1"
DEFAULT_PAIRS_PER_SPLIT = 100
DEFAULT_RIDGE_ALPHA = 1e-3
DEFAULT_OUTPUT_PATH = Path("artifacts/experiment_000/memory_probe/report.json")
_RATIO_EPSILON = 1e-15


@dataclass(frozen=True, slots=True)
class _PairedSplit:
    """The two cue interventions collected for one data split."""

    cue_zero_features: np.ndarray
    cue_one_features: np.ndarray
    cue_zero_outputs: np.ndarray
    cue_one_outputs: np.ndarray

    @property
    def pair_count(self) -> int:
        return int(self.cue_zero_features.shape[0])


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


def _array_bundle_sha256(*arrays: np.ndarray) -> str:
    """Hash array shapes and canonical little-endian float64 values."""

    hasher = hashlib.sha256()
    for array in arrays:
        normalized = np.asarray(array, dtype="<f8", order="C")
        hasher.update(str(normalized.shape).encode("ascii"))
        hasher.update(b"\0")
        hasher.update(normalized.tobytes(order="C"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _validate_probe_arguments(
    *, seeds: Sequence[int], pairs_per_split: int, ridge_alpha: float
) -> tuple[int, ...]:
    resolved = tuple(seeds)
    if not resolved:
        raise ValueError("at least one development seed is required")
    if len(set(resolved)) != len(resolved):
        raise ValueError("development seeds must be unique")
    for seed in resolved:
        require_seed_role(seed, SeedRole.DEVELOPMENT)
    if (
        isinstance(pairs_per_split, bool)
        or not isinstance(pairs_per_split, int)
        or pairs_per_split <= 0
        or pairs_per_split % 2
    ):
        raise ValueError("pairs_per_split must be a positive even integer")
    if not math.isfinite(ridge_alpha) or ridge_alpha <= 0.0:
        raise ValueError("ridge_alpha must be finite and positive")
    return resolved


def recurrent_spectral_radius(graph: RecurrentEventGraph) -> float:
    """Return the spectral radius of the hidden-to-hidden weight matrix."""

    hidden_index = {node: index for index, node in enumerate(graph.hidden_nodes)}
    matrix = np.zeros((len(hidden_index), len(hidden_index)), dtype=np.float64)
    weights = graph.weights
    for edge in graph.edges:
        if edge.kind != "recurrent":
            continue
        matrix[hidden_index[edge.destination], hidden_index[edge.source]] = weights[
            edge.edge_id
        ]
    eigenvalues = np.linalg.eigvals(matrix)
    radius = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    if not math.isfinite(radius):
        raise FloatingPointError("non-finite recurrent spectral radius")
    return radius


def _output_feature_edge_ids(graph: RecurrentEventGraph) -> tuple[str, ...]:
    return tuple(
        sorted(
            edge.edge_id
            for edge in graph.edges
            if edge.kind == "output" and edge.destination == graph.output_node
        )
    )


def extract_query_features(
    graph: RecurrentEventGraph, query: QueryResult
) -> tuple[tuple[str, float], ...]:
    """Return the exact ordered hidden messages available to the query readout.

    Coordinates are all frozen output edges in lexical edge-ID order.  When a
    source unit did not emit at the relevant tick, that structural coordinate
    is represented by zero.
    """

    feature_edge_ids = _output_feature_edge_ids(graph)
    root = graph.unit_events_by_id[query.event_id]
    trace_values = {trace.edge_id: trace.message_value for trace in root.edge_traces}
    unexpected = set(trace_values).difference(feature_edge_ids)
    if unexpected:
        raise RuntimeError(f"query root used unexpected feature edges: {sorted(unexpected)}")
    features = tuple(
        (edge_id, float(trace_values.get(edge_id, 0.0)))
        for edge_id in feature_edge_ids
    )
    reconstructed = math.fsum(graph.weights[edge_id] * value for edge_id, value in features)
    if not math.isclose(reconstructed, root.preactivation, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError("query features do not reconstruct the root preactivation")
    return features


def _counterfactual_episode(
    base: Experiment000Episode, *, cue: int
) -> Experiment000Episode:
    return replace(
        base,
        episode_id=f"{base.episode_id}-memory-probe-cue-{cue}",
        cue=cue,
        target=cue,
    )


def _run_counterfactual(
    graph: RecurrentEventGraph,
    base: Experiment000Episode,
    *,
    cue: int,
    feature_edge_ids: tuple[str, ...],
) -> tuple[np.ndarray, float]:
    episode = _counterfactual_episode(base, cue=cue)
    query = forward_episode(graph, episode)
    extracted = extract_query_features(graph, query)
    if tuple(edge_id for edge_id, _ in extracted) != feature_edge_ids:
        raise RuntimeError("query feature order changed during the probe")
    feature = np.asarray([value for _, value in extracted], dtype=np.float64)
    if not np.all(np.isfinite(feature)) or not math.isfinite(query.activation):
        raise FloatingPointError("non-finite value in paired counterfactual forward pass")
    return feature, float(query.activation)


def _collect_split(
    graph: RecurrentEventGraph,
    stream: Experiment000Stream,
    *,
    feature_edge_ids: tuple[str, ...],
) -> _PairedSplit:
    cue_zero_features: list[np.ndarray] = []
    cue_one_features: list[np.ndarray] = []
    cue_zero_outputs: list[float] = []
    cue_one_outputs: list[float] = []
    for base in stream.episodes:
        zero_feature, zero_output = _run_counterfactual(
            graph, base, cue=0, feature_edge_ids=feature_edge_ids
        )
        one_feature, one_output = _run_counterfactual(
            graph, base, cue=1, feature_edge_ids=feature_edge_ids
        )
        cue_zero_features.append(zero_feature)
        cue_one_features.append(one_feature)
        cue_zero_outputs.append(zero_output)
        cue_one_outputs.append(one_output)
    return _PairedSplit(
        cue_zero_features=np.stack(cue_zero_features),
        cue_one_features=np.stack(cue_one_features),
        cue_zero_outputs=np.asarray(cue_zero_outputs, dtype=np.float64),
        cue_one_outputs=np.asarray(cue_one_outputs, dtype=np.float64),
    )


def _finite_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= _RATIO_EPSILON:
        return None
    ratio = numerator / denominator
    if not math.isfinite(ratio):
        raise FloatingPointError("non-finite retention ratio")
    return float(ratio)


def paired_retention_metrics(split: _PairedSplit) -> dict[str, Any]:
    """Compute explicitly scaled cue and noise retention measurements.

    ``cue_feature_delta_rms`` is the RMS Euclidean coefficient of the bipolar
    cue, namely half the same-noise separation.  ``noise_feature_rms`` is the
    RMS displacement of the pair midpoint from the mean midpoint, so the cue
    intervention is removed before estimating between-noise variability.
    """

    feature_delta = 0.5 * (split.cue_one_features - split.cue_zero_features)
    pair_midpoint = 0.5 * (split.cue_one_features + split.cue_zero_features)
    centered_midpoint = pair_midpoint - np.mean(pair_midpoint, axis=0, keepdims=True)
    cue_feature_delta_rms = float(
        np.sqrt(np.mean(np.sum(np.square(feature_delta), axis=1)))
    )
    noise_feature_rms = float(
        np.sqrt(np.mean(np.sum(np.square(centered_midpoint), axis=1)))
    )

    output_delta = 0.5 * (split.cue_one_outputs - split.cue_zero_outputs)
    cue_output_delta_rms = float(np.sqrt(np.mean(np.square(output_delta))))
    all_outputs = np.concatenate((split.cue_zero_outputs, split.cue_one_outputs))
    output_sd = float(np.std(all_outputs, ddof=0))
    values = (
        cue_feature_delta_rms,
        noise_feature_rms,
        cue_output_delta_rms,
        output_sd,
    )
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("non-finite paired-retention measurement")
    return {
        "pairs": split.pair_count,
        "cue_feature_delta_rms": cue_feature_delta_rms,
        "noise_feature_rms": noise_feature_rms,
        "cue_to_noise_feature_ratio": _finite_ratio(
            cue_feature_delta_rms, noise_feature_rms
        ),
        "output_cue_delta_rms": cue_output_delta_rms,
        "output_sd": output_sd,
        "output_cue_delta_to_sd": _finite_ratio(cue_output_delta_rms, output_sd),
        "paired_features_sha256": _array_bundle_sha256(
            split.cue_zero_features,
            split.cue_one_features,
            split.cue_zero_outputs,
            split.cue_one_outputs,
        ),
    }


def _ridge_rows(split: _PairedSplit) -> tuple[np.ndarray, np.ndarray]:
    feature_count = split.cue_zero_features.shape[1]
    features = np.stack(
        (split.cue_zero_features, split.cue_one_features), axis=1
    ).reshape(-1, feature_count)
    targets = np.tile(np.asarray((-1.0, 1.0), dtype=np.float64), split.pair_count)
    return features, targets


def ridge_readout(
    train: _PairedSplit, evaluation: _PairedSplit, *, alpha: float
) -> dict[str, Any]:
    """Fit ridge only on train pairs and score the disjoint evaluation pairs."""

    train_x, train_y = _ridge_rows(train)
    eval_x, eval_y = _ridge_rows(evaluation)
    train_mean = np.mean(train_x, axis=0)
    train_scale = np.std(train_x, axis=0, ddof=0)
    active = train_scale > _RATIO_EPSILON
    safe_scale = np.where(active, train_scale, 1.0)
    normalized_train = (train_x - train_mean) / safe_scale
    normalized_eval = (eval_x - train_mean) / safe_scale

    sample_count = train_x.shape[0]
    gram = (normalized_train.T @ normalized_train) / sample_count
    gram += alpha * np.eye(gram.shape[0], dtype=np.float64)
    right_hand_side = (normalized_train.T @ train_y) / sample_count
    coefficients = np.linalg.solve(gram, right_hand_side)
    intercept = float(np.mean(train_y))
    train_scores = normalized_train @ coefficients + intercept
    eval_scores = normalized_eval @ coefficients + intercept
    train_predictions = np.where(train_scores >= 0.0, 1.0, -1.0)
    eval_predictions = np.where(eval_scores >= 0.0, 1.0, -1.0)
    train_accuracy = float(np.mean(train_predictions == train_y))
    eval_accuracy = float(np.mean(eval_predictions == eval_y))
    if not all(
        math.isfinite(value)
        for value in (train_accuracy, eval_accuracy, float(np.linalg.norm(coefficients)))
    ):
        raise FloatingPointError("non-finite ridge readout result")
    return {
        "alpha": float(alpha),
        "target_encoding": {"cue_0": -1.0, "cue_1": 1.0},
        "feature_standardization": "train_mean_and_population_sd_only",
        "active_feature_count": int(np.count_nonzero(active)),
        "feature_count": int(train_x.shape[1]),
        "train_examples": int(train_x.shape[0]),
        "eval_examples": int(eval_x.shape[0]),
        "train_accuracy": train_accuracy,
        "eval_accuracy": eval_accuracy,
        "coefficient_l2": float(np.linalg.norm(coefficients)),
    }


def _mean_optional(values: Iterable[float | None]) -> float | None:
    realized = [float(value) for value in values if value is not None]
    return float(np.mean(realized)) if realized else None


def _run_seed_probe(
    *, seed: int, pairs_per_split: int, ridge_alpha: float
) -> dict[str, Any]:
    require_seed_role(seed, SeedRole.DEVELOPMENT)
    pair = generate_stream_pair(
        seed,
        episode_count=pairs_per_split,
        noise_events=8,
        required_role=SeedRole.DEVELOPMENT,
    )
    graph = build_experiment_000_graph(
        seed,
        mode="full",
        event_log_enabled=False,
        **FROZEN_GRAPH_OPTIONS,
    )
    feature_edge_ids = _output_feature_edge_ids(graph)
    topology_before = graph.topology_hash()
    weights_before = graph.weights_hash()
    spectral_radius = recurrent_spectral_radius(graph)
    train = _collect_split(graph, pair.train, feature_edge_ids=feature_edge_ids)
    evaluation = _collect_split(graph, pair.eval, feature_edge_ids=feature_edge_ids)
    weights_after = graph.weights_hash()
    topology_after = graph.topology_hash()
    if weights_before != weights_after:
        raise RuntimeError("memory probe changed frozen graph weights")
    if topology_before != topology_after:
        raise RuntimeError("memory probe changed frozen graph topology")

    return {
        "seed": seed,
        "seed_role": SeedRole.DEVELOPMENT.value,
        "data_manifest": pair.manifest.to_dict(),
        "graph": {
            "hidden_units": len(graph.hidden_nodes),
            "recurrent_edge_count": sum(edge.kind == "recurrent" for edge in graph.edges),
            "feature_edge_ids": list(feature_edge_ids),
            "feature_dimension": len(feature_edge_ids),
            "spectral_radius": spectral_radius,
            "topology_sha256": topology_before,
            "topology_unchanged": topology_before == topology_after,
            "initial_weights_sha256": weights_before,
            "final_weights_sha256": weights_after,
            "weights_unchanged": weights_before == weights_after,
        },
        "train": paired_retention_metrics(train),
        "eval": paired_retention_metrics(evaluation),
        "ridge_readout": ridge_readout(train, evaluation, alpha=ridge_alpha),
    }


def run_memory_probe(
    *,
    seeds: Sequence[int] = DEVELOPMENT_SEEDS,
    pairs_per_split: int = DEFAULT_PAIRS_PER_SPLIT,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the deterministic development-only paired cue-retention probe."""

    resolved_seeds = _validate_probe_arguments(
        seeds=seeds,
        pairs_per_split=pairs_per_split,
        ridge_alpha=ridge_alpha,
    )
    seed_results = [
        _run_seed_probe(
            seed=seed,
            pairs_per_split=pairs_per_split,
            ridge_alpha=ridge_alpha,
        )
        for seed in resolved_seeds
    ]
    report: dict[str, Any] = {
        "schema_version": MEMORY_PROBE_SCHEMA_VERSION,
        "scope": "development_only_memory_probe",
        "confirmatory_executed": False,
        "seeds": list(resolved_seeds),
        "pairs_per_split": pairs_per_split,
        "counterfactual_forward_episodes_per_seed": 4 * pairs_per_split,
        "noise_events": 8,
        "graph_options": dict(FROZEN_GRAPH_OPTIONS),
        "feature_definition": (
            "ordered hidden message_value on each frozen output edge at the forced "
            "query event; absent messages are zero"
        ),
        "formulas": {
            "spectral_radius": "rho(W_rec)=max(abs(eigvals(W_rec)))",
            "cue_feature_delta_rms": (
                "sqrt(mean_k(||d_k||_2^2)), d_k=(z_k,1-z_k,0)/2"
            ),
            "noise_feature_rms": (
                "sqrt(mean_k(||m_k-mean_j(m_j)||_2^2)), "
                "m_k=(z_k,0+z_k,1)/2"
            ),
            "cue_to_noise_feature_ratio": (
                "cue_feature_delta_rms/noise_feature_rms"
            ),
            "output_cue_delta_to_sd": (
                "sqrt(mean_k(delta_k^2))/population_sd({o_k,0,o_k,1}), "
                "delta_k=(o_k,1-o_k,0)/2"
            ),
            "ridge": (
                "beta=(X_train_std^T X_train_std/n + alpha I)^-1 "
                "X_train_std^T y_train/n"
            ),
        },
        "ridge_alpha": float(ridge_alpha),
        "seed_results": seed_results,
        "aggregate": {
            "mean_spectral_radius": float(
                np.mean([row["graph"]["spectral_radius"] for row in seed_results])
            ),
            "mean_train_cue_to_noise_feature_ratio": _mean_optional(
                row["train"]["cue_to_noise_feature_ratio"] for row in seed_results
            ),
            "mean_eval_cue_to_noise_feature_ratio": _mean_optional(
                row["eval"]["cue_to_noise_feature_ratio"] for row in seed_results
            ),
            "mean_train_output_cue_delta_to_sd": _mean_optional(
                row["train"]["output_cue_delta_to_sd"] for row in seed_results
            ),
            "mean_eval_output_cue_delta_to_sd": _mean_optional(
                row["eval"]["output_cue_delta_to_sd"] for row in seed_results
            ),
            "mean_ridge_train_accuracy": float(
                np.mean(
                    [row["ridge_readout"]["train_accuracy"] for row in seed_results]
                )
            ),
            "mean_ridge_eval_accuracy": float(
                np.mean(
                    [row["ridge_readout"]["eval_accuracy"] for row in seed_results]
                )
            ),
            "all_weights_unchanged": all(
                row["graph"]["weights_unchanged"] for row in seed_results
            ),
            "all_topologies_unchanged": all(
                row["graph"]["topology_unchanged"] for row in seed_results
            ),
        },
    }
    report["deterministic_payload_sha256"] = _sha256_json(report)

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
        description=(
            "Measure cue retention in the frozen Experiment-000 recurrent graph "
            "using same-noise counterfactual cue flips."
        )
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEVELOPMENT_SEEDS),
        help="development seeds only (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--pairs-per-split",
        type=int,
        default=DEFAULT_PAIRS_PER_SPLIT,
        help="same-noise pairs in each train/eval split (default: 100)",
    )
    parser.add_argument("--ridge-alpha", type=float, default=DEFAULT_RIDGE_ALPHA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)
    report = run_memory_probe(
        seeds=tuple(args.seeds),
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
    "DEFAULT_RIDGE_ALPHA",
    "MEMORY_PROBE_SCHEMA_VERSION",
    "extract_query_features",
    "main",
    "paired_retention_metrics",
    "recurrent_spectral_radius",
    "ridge_readout",
    "run_memory_probe",
]
