import json
from pathlib import Path

import pytest

from adaptive_learning_substrate.experiment000_data import generate_stream_pair
from adaptive_learning_substrate.experiment_000 import (
    forward_episode,
    load_config,
    run_experiment000_development,
)
from adaptive_learning_substrate.recurrent import build_experiment_000_graph


def _small_config() -> dict:
    return {
        "protocol": {
            "name": "experiment_000_delayed_chain",
            "version": "experiment-000-data-v1",
            "scope": "development_only",
            "confirmatory": False,
        },
        "execution": {
            "enabled": True,
            "required_seed_role": "development",
            "seeds": [0],
            "paired_methods": ["ccf_v0", "ccf_no_trace", "rand"],
        },
        "data": {"episodes_per_split": 4, "noise_events": 8},
        "graph": {
            "hidden_count": 8,
            "recurrent_in_degree": 2,
            "input_fan_out": 2,
            "output_fan_in": 4,
            "edge_delay_ticks": 1,
            "initial_weight_scale": 0.35,
        },
        "ccf": {
            "activation": "tanh",
            "float_type": "float64",
            "learning_rate": 0.02,
            "trace_decay": 0.97,
            "route_gain": 0.9,
            "trace_horizon": 32,
            "hop_limit": 16,
        },
    }


def test_forward_episode_has_real_events_and_two_tick_readout_latency() -> None:
    episode = generate_stream_pair(
        0, episode_count=2, required_role="development"
    ).train.episodes[0]
    graph = build_experiment_000_graph(0, event_log_enabled=False)

    query = forward_episode(graph, episode)

    assert query.tick == 11
    assert graph.ledger["ticks"] == 12
    assert query.prediction in (0, 1)
    assert graph.audit["queried_output_event_id"] == query.event_id
    source_nodes = [event.node for event in graph.unit_events if not event.edge_traces]
    assert "cue" in source_nodes
    assert source_nodes.count("noise") == 8
    assert "query" in source_nodes


def test_runner_rejects_confirmatory_seed_before_creating_output(tmp_path: Path) -> None:
    target = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="seed 1000 is confirmatory"):
        run_experiment000_development(_small_config(), target, seeds=(1000,))
    assert not target.exists()


def test_disabled_confirmatory_config_cannot_enter_development_runner() -> None:
    config = load_config("configs/experiment_000_confirmatory.toml")
    with pytest.raises(ValueError, match="development-only"):
        run_experiment000_development(config, seeds=(0,), episodes_per_split=2)


def test_small_paired_run_writes_auditable_report(tmp_path: Path) -> None:
    report = run_experiment000_development(_small_config(), tmp_path)

    assert report["status"] == "development_only"
    assert report["confirmatory_executed"] is False
    assert report["verdict"] == "DEVELOPMENT_DIAGNOSTIC_ONLY"
    assert report["canonical_development_run"] is False
    assert report["seeds"] == [0]
    assert len(report["seed_results"]) == 3
    assert report["integrity"] == {
        "all_topologies_unchanged": True,
        "all_evaluations_weight_frozen": True,
        "all_finite": True,
        "confirmatory_seed_count": 0,
    }

    learned = {
        row["method"]: row
        for row in report["seed_results"]
        if row["method"] != "rand"
    }
    assert learned["ccf_v0"]["graph"]["topology_before"] == learned["ccf_no_trace"]["graph"]["topology_before"]
    assert learned["ccf_v0"]["graph"]["initial_weights"] == learned["ccf_no_trace"]["graph"]["initial_weights"]
    for row in learned.values():
        assert row["train"]["episodes"] == 4
        assert row["eval"]["episodes"] == 4
        assert row["graph"]["evaluation_weights_unchanged"] is True
        assert row["ledger"]["total"]["episodes"] == 8
        assert row["ledger"]["total"]["ticks"] == 96
        assert row["ledger"]["eval_delta"]["weight_write_touches"] == 0
        assert "peak_pending_messages" not in row["ledger"]["eval_delta"]
        assert set(row["ledger"]["cumulative_high_water"]) == {
            "peak_pending_messages",
            "peak_retained_event_records",
        }
        assert row["graph"]["peak_max_abs_weight"] >= row["graph"]["final_max_abs_weight"]

    random_row = next(row for row in report["seed_results"] if row["method"] == "rand")
    assert random_row["eval"]["activation"] is None
    assert "activation_mean" not in random_row["eval"]
    assert report["implementation_revision"] == "C000.1"
    assert report["source_provenance"]["bundle_sha256"]
    assert report["graph_manifests"]["0"]["edge_count"] > 0

    raw = (tmp_path / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(raw) == 20  # 8 full + 8 no-trace + 4 random evaluation rows
    assert all(json.loads(line)["target"] in (0, 1) for line in raw)
    for name in (
        "report.json",
        "config.snapshot.json",
        "data_manifests.json",
        "provenance.json",
        "artifact_manifest.json",
        "graph_manifests/seed-0.json",
    ):
        assert (tmp_path / name).is_file()
        json.loads((tmp_path / name).read_text(encoding="utf-8"))


def _deterministic_projection(report: dict) -> dict:
    projected = json.loads(json.dumps(report))
    projected.pop("runtime")
    for row in projected["seed_results"]:
        row.pop("runtime_seconds")
    return projected


def test_same_development_run_is_deterministic(tmp_path: Path) -> None:
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left = run_experiment000_development(
        _small_config(), left_dir, methods=("ccf_v0", "rand")
    )
    right = run_experiment000_development(
        _small_config(), right_dir, methods=("ccf_v0", "rand")
    )

    assert _deterministic_projection(left) == _deterministic_projection(right)
    assert (left_dir / "episodes.jsonl").read_bytes() == (right_dir / "episodes.jsonl").read_bytes()


def test_edited_small_config_cannot_claim_canonical_development() -> None:
    config = load_config("configs/experiment_000_development.toml")
    config["data"]["episodes_per_split"] = 2
    report = run_experiment000_development(config)
    assert report["canonical_development_run"] is False
