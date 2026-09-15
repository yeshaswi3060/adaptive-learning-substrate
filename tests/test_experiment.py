import json

from adaptive_learning_substrate.experiment import run_experiment


def smoke_config(seed: int = 11) -> dict:
    return {
        "experiment": {
            "name": "test_smoke",
            "seed": seed,
            "learner": "ccf",
            "phase_a_episodes": 256,
            "phase_b_episodes": 128,
            "train_delay_min": 1,
            "train_delay_max": 3,
            "eval_delay": 3,
        },
        "graph": {"trace_decay": 0.94, "learning_rate": 0.16, "weight_clip": 4.0},
        "output": {"write_event_log": True},
    }


def test_end_to_end_run_is_deterministic_and_writes_audit_artifacts(tmp_path) -> None:
    first = run_experiment(smoke_config(), tmp_path / "first")
    second = run_experiment(smoke_config(), tmp_path / "second")
    assert first == second
    assert first["confirmatory"] is False
    assert first["implemented_claim"] == "deterministic fixed-DAG software plumbing only"
    assert first["composition"]["method"] == "external_two_pass_primitive_adapter"
    assert first["audit"]["topology_unchanged"] is True
    assert first["phase_a_after_b"]["accuracy"] >= 0.875
    assert first["phase_b_after_b"]["accuracy"] >= 0.875
    assert first["composition"]["accuracy"] >= 0.75
    assert first["forgetting"] <= 0.125
    for directory in (tmp_path / "first", tmp_path / "second"):
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        assert report["audit"]["event_count"] > 0
        assert (directory / "events.jsonl").stat().st_size > 0
        assert (directory / "config.snapshot.json").exists()


def test_different_seed_changes_final_weight_hash() -> None:
    first = run_experiment(smoke_config(seed=1))
    second = run_experiment(smoke_config(seed=2))
    assert first["audit"]["weights_sha256"] != second["audit"]["weights_sha256"]


def test_same_mask_eligibility_baseline_is_runnable_and_reported() -> None:
    ccf_config = smoke_config(seed=4)
    baseline_config = smoke_config(seed=4)
    baseline_config["experiment"]["learner"] = "eligibility"
    ccf = run_experiment(ccf_config)
    baseline = run_experiment(baseline_config)
    assert ccf["audit"]["topology_sha256"] == baseline["audit"]["topology_sha256"]
    assert ccf["learner"] == "ccf"
    assert baseline["learner"] == "eligibility"
    assert baseline["phase_a_after_b"]["accuracy"] >= 0.875


def test_preregistered_experiment_001_requires_future_recurrent_runner() -> None:
    config = smoke_config()
    config["status"] = {
        "implementation_ready": False,
        "required_runner": "future_64_unit_recurrent_event_graph",
    }
    import pytest

    with pytest.raises(NotImplementedError, match="future 64-unit recurrent"):
        run_experiment(config)
