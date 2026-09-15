"""Nonselecting tests for diversity, locked interventions, and exact replay."""
import copy
from collections import Counter

import pytest

from adaptive_learning_substrate import sequence_diversity as study
from adaptive_learning_substrate.sequence_core import (
    CpuSequence,
    canonical,
    read_record,
    write_record,
)


@pytest.fixture(scope="module")
def datasets():
    return {task: study.make_data(720001, task) for task in ("copy3", "recall4")}


@pytest.mark.parametrize("task", ["copy3", "recall4"])
def test_data_regenerates_and_excludes_both_novel_and_changed_cues(datasets, task, tmp_path):
    data = datasets[task]
    assert data == study.make_data(720001, task)
    path = tmp_path / "data.json"
    write_record(path, data)
    assert read_record(path) == data
    start = 5 if task == "copy3" else 10
    signature = lambda row: tuple(row["prompt"][:start] + row["prompt"][start + row["delay"]:])
    train = {signature(row) for row in data["all_seen"]}
    assert len(train) == 128
    for name in ("heldout0", "heldout16", "counterfactual16"):
        assert train.isdisjoint(signature(row) for row in data[name])
        assert all(study.memory.oracle(row["prompt"]) == row["targets"] for row in data[name])


@pytest.mark.parametrize("task", ["copy3", "recall4"])
def test_coupled_stream_has_equal_budget_noise_and_balanced_blocks(datasets, task):
    data = datasets[task]
    assert len(data["training_order"]) == len(data["training_noise"]) == 4096
    assert len({tuple(n) for n in data["training_noise"]}) == 4096
    for begin in range(0, 4096, 128):
        order = data["training_order"][begin:begin + 128]
        assert sorted(order) == list(range(128))
        assert Counter(i % 16 for i in order) == {i: 8 for i in range(16)}
    start = 5 if task == "copy3" else 10
    test_noise = {tuple(r["prompt"][start:start + 16]) for r in data["all_seen"] + data["heldout16"]}
    assert {tuple(n) for n in data["training_noise"]}.isdisjoint(test_noise)
    for count in (16, 128):
        assert study.stream_digest(data, count) == data["training_stream_sha256"][str(count)]
        examples = [study.training_episode(data, count, i) for i in range(128)]
        for row in examples:
            assert study.memory.oracle(row["prompt"]) == row["targets"]
        for position in range(len(examples[0]["targets"])):
            assert Counter(r["targets"][position] for r in examples) == {v: 16 for v in range(64, 72)}
    for i in (0, 5, 128, 4095):
        a, b = [study.training_episode(data, c, i) for c in (16, 128)]
        assert a["prompt"][start:start + 16] == b["prompt"][start:start + 16] == data["training_noise"][i]


@pytest.mark.parametrize("task", ["copy3", "recall4"])
def test_common_seen_and_novel_evaluations_are_identical_across_diversities(datasets, task):
    a, b = [study.evaluation_sets(datasets[task], c) for c in (16, 128)]
    for key in ("common_seen", "heldout0", "heldout16", "counterfactual16"):
        assert a[key] == b[key]
    assert len(a["all_seen"]) == 16 and len(b["all_seen"]) == 128
    assert a["all_seen"] == b["all_seen"][:16]
    for first, changed in zip(a["heldout16"], a["counterfactual16"], strict=True):
        start = 5 if task == "copy3" else 10
        assert first["prompt"][start:] == changed["prompt"][start:]
        assert first["targets"] != changed["targets"]


@pytest.mark.parametrize("count", [True, 8, 17, 256])
def test_invalid_diversity_is_rejected(datasets, count):
    with pytest.raises(ValueError):
        study.training_episode(datasets["copy3"], count, 0)
    with pytest.raises(ValueError):
        study.evaluation_sets(datasets["copy3"], count)


@pytest.mark.parametrize("index", [True, -1, 4096, 0.0])
def test_invalid_presentation_index_is_rejected(datasets, index):
    with pytest.raises(ValueError):
        study.training_episode(datasets["copy3"], 16, index)


def test_stream_binding_changes_when_any_noise_token_changes(datasets):
    data = copy.deepcopy(datasets["copy3"])
    data["training_noise"][0][0] = 128 + (data["training_noise"][0][0] - 128 + 1) % 16
    for count in (16, 128):
        assert study.stream_digest(data, count) != data["training_stream_sha256"][str(count)]


def test_paired_accuracy_requires_both_original_and_intervened_answers():
    good = {"free_predictions": [64], "targets": [64]}
    bad = {"free_predictions": [64], "targets": [65]}
    assert study.paired_accuracy({"rows": [good, good, bad, bad]}, {"rows": [good, bad, good, bad]}) == .25
    with pytest.raises(ValueError):
        study.paired_accuracy({"rows": []}, {"rows": []})
    with pytest.raises(ValueError):
        study.paired_accuracy({"rows": [good]}, {"rows": [good, good]})


def test_measure_preserves_parameters_and_data(datasets):
    model = CpuSequence(720002)
    data = datasets["copy3"]
    before, original_data = canonical(model.snapshot()), canonical(data)
    result = study.measure(model, data, 16)
    assert canonical(model.snapshot()) == before and canonical(data) == original_data
    assert result["paired_exact"]["normal"] <= result["scores"]["heldout16"]["normal"]["exact_accuracy"]


def test_small_complete_retraining_and_original_engine_replay(tmp_path, monkeypatch):
    spec = copy.deepcopy(study.SPEC)
    spec.update(model_seeds=[720003], tasks=["copy3"], cue_counts=[8, 16],
                heldout_examples=8, delay=2, checkpoint_episodes=[16, 32])
    monkeypatch.setattr(study, "SPEC", spec)
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "source_bindings", lambda: {"synthetic-source": "a" * 64})
    monkeypatch.setattr(study.memory, "memory_snapshot", lambda: {"available_system_ram_bytes": 1024**3})
    primary, rerun = tmp_path / "artifacts/primary", tmp_path / "artifacts/rerun"
    monkeypatch.setattr(study, "PROCESS_TOKEN", "synthetic-primary")
    a = study.run(primary)
    monkeypatch.setattr(study, "PROCESS_TOKEN", "synthetic-rerun")
    b = study.run(rerun)
    assert a["scientific_payload"] == b["scientific_payload"]
    monkeypatch.setattr(study, "PROCESS_TOKEN", "synthetic-verifier")
    result = study.verify(primary, rerun, tmp_path / "verification.json")
    assert result["status"] == "PASS" and result["original_engine_evaluation_episodes"] == 1408


def test_resource_guard_precedes_output_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study.memory, "memory_snapshot", lambda: {"available_system_ram_bytes": 1})
    target = tmp_path / "artifacts/absent"
    with pytest.raises(MemoryError):
        study.run(target)
    assert not target.exists()
