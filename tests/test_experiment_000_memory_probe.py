import json
import math
from pathlib import Path

import numpy as np
import pytest

from adaptive_learning_substrate.experiment000_data import generate_stream_pair
from adaptive_learning_substrate.experiment_000 import forward_episode
from adaptive_learning_substrate.experiment_000_memory_probe import (
    _PairedSplit,
    extract_query_features,
    main,
    paired_retention_metrics,
    run_memory_probe,
)
from adaptive_learning_substrate.recurrent import build_experiment_000_graph


def test_paired_metric_formulas_on_known_arrays() -> None:
    paired = _PairedSplit(
        cue_zero_features=np.asarray(((0.0, 0.0), (2.0, 0.0))),
        cue_one_features=np.asarray(((1.0, 0.0), (3.0, 0.0))),
        cue_zero_outputs=np.asarray((0.0, 2.0)),
        cue_one_outputs=np.asarray((1.0, 3.0)),
    )

    metrics = paired_retention_metrics(paired)

    assert metrics["cue_feature_delta_rms"] == pytest.approx(0.5)
    assert metrics["noise_feature_rms"] == pytest.approx(1.0)
    assert metrics["cue_to_noise_feature_ratio"] == pytest.approx(0.5)
    assert metrics["output_cue_delta_rms"] == pytest.approx(0.5)
    assert metrics["output_sd"] == pytest.approx(math.sqrt(1.25))
    assert metrics["output_cue_delta_to_sd"] == pytest.approx(0.5 / math.sqrt(1.25))
    assert len(metrics["paired_features_sha256"]) == 64


def test_query_features_are_stable_and_reconstruct_root_preactivation() -> None:
    graph = build_experiment_000_graph(0, event_log_enabled=False)
    episode = generate_stream_pair(
        0, episode_count=2, required_role="development"
    ).train.episodes[0]
    query = forward_episode(graph, episode)

    features = extract_query_features(graph, query)

    assert len(features) == 16
    assert [edge_id for edge_id, _ in features] == sorted(
        edge_id for edge_id, _ in features
    )
    root = graph.unit_events_by_id[query.event_id]
    reconstructed = math.fsum(
        graph.weights[edge_id] * value for edge_id, value in features
    )
    assert reconstructed == pytest.approx(root.preactivation, abs=1e-15)


def test_memory_probe_is_deterministic_and_keeps_frozen_graph_unchanged() -> None:
    left = run_memory_probe(seeds=(0,), pairs_per_split=2)
    right = run_memory_probe(seeds=(0,), pairs_per_split=2)

    assert left == right
    assert left["confirmatory_executed"] is False
    assert left["seeds"] == [0]
    assert left["deterministic_payload_sha256"] == right["deterministic_payload_sha256"]
    result = left["seed_results"][0]
    assert result["seed_role"] == "development"
    assert result["graph"]["hidden_units"] == 64
    assert result["graph"]["recurrent_edge_count"] == 512
    assert result["graph"]["feature_dimension"] == 16
    assert 0.0 < result["graph"]["spectral_radius"] < 1.0
    assert result["graph"]["weights_unchanged"] is True
    assert result["graph"]["topology_unchanged"] is True
    assert result["graph"]["initial_weights_sha256"] == result["graph"]["final_weights_sha256"]
    assert result["data_manifest"]["train_sha256"] != result["data_manifest"]["eval_sha256"]
    assert result["train"]["pairs"] == 2
    assert result["eval"]["pairs"] == 2
    assert 0.0 <= result["ridge_readout"]["train_accuracy"] <= 1.0
    assert 0.0 <= result["ridge_readout"]["eval_accuracy"] <= 1.0
    assert left["aggregate"]["all_weights_unchanged"] is True
    assert left["aggregate"]["all_topologies_unchanged"] is True


def test_memory_probe_cli_writes_json(tmp_path: Path) -> None:
    target = tmp_path / "memory-probe.json"

    exit_status = main(
        [
            "--seeds",
            "0",
            "--pairs-per-split",
            "2",
            "--output",
            str(target),
        ]
    )

    assert exit_status == 0
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["schema_version"] == "experiment-000-memory-probe-v1"
    assert report["seeds"] == [0]
    assert report["pairs_per_split"] == 2
    assert report["confirmatory_executed"] is False


def test_memory_probe_rejects_confirmatory_seed_before_writing(tmp_path: Path) -> None:
    target = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match="seed 1000 is confirmatory"):
        run_memory_probe(seeds=(1000,), pairs_per_split=2, output_path=target)

    assert not target.exists()
