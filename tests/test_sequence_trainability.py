"""Non-experiment-seed controls for tiny-set fitting and paired-delay data."""
import copy

import numpy as np
import pytest

from adaptive_learning_substrate import sequence_trainability as study
from adaptive_learning_substrate.sequence_core import CpuSequence, canonical


@pytest.mark.parametrize("task", ["copy3", "recall4"])
def test_balanced_unique_disjoint_paired_examples(task):
    data = study.make_data(700101, task)
    assert data == study.make_data(700101, task)
    assert data != study.make_data(700102, task)
    train = {tuple(row["prompt"]) for row in data["train"]["0"]}
    test = {tuple(row["prompt"]) for row in data["test"]["0"]}
    assert len(train) == 16 and len(test) == 64 and train.isdisjoint(test)
    for split, count in (("train", 16), ("test", 64)):
        zero, delayed = data[split]["0"], data[split]["16"]
        target_array = np.array([row["targets"] for row in zero])
        for column in target_array.T:
            assert np.array_equal(np.bincount(column, minlength=256)[64:72], [count // 8] * 8)
        for a, b in zip(zero, delayed, strict=True):
            assert a["targets"] == b["targets"]
            assert [token for token in b["prompt"] if not 128 <= token <= 143] == a["prompt"]
            assert len(b["prompt"]) - len(a["prompt"]) == 16
            assert study.memory.oracle(b["prompt"]) == b["targets"]
    if task == "recall4":
        for key in range(32, 36):
            targets = [r["targets"][0] for r in data["train"]["0"] if r["prompt"][-1] == key]
            assert len(targets) == 4 and len(set(targets)) >= 2


def test_epoch_schedule_has_exact_counts_at_both_checkpoints():
    data = study.make_data(700103, "copy3")
    order = data["training_order"]
    assert len(order) == 2048
    for start in range(0, 2048, 16):
        assert sorted(order[start:start + 16]) == list(range(16))
    for budget in (512, 2048):
        assert np.array_equal(np.bincount(order[:budget]), [budget // 16] * 16)


@pytest.mark.parametrize("count", [0, 7, 17, 256, True, 8.0])
def test_invalid_balanced_sample_count_rejected(count):
    with pytest.raises(ValueError):
        study.balanced_examples(700104, "copy3", "test", count, set())


@pytest.mark.parametrize("exact,reset,fit,control", [(1., 0., True, True),
    (15/16, 0., False, True), (1., .75, True, False), (.25, .25, False, False)])
def test_training_fit_and_memory_control_are_distinct(exact, reset, fit, control):
    result = study.fitting_flags({"exact_accuracy": exact}, {"exact_accuracy": reset})
    assert result["fits_training_examples"] is fit
    assert result["training_memory_control_pass"] is control


@pytest.mark.parametrize("task", ["copy3", "recall4"])
@pytest.mark.parametrize("reset", [False, True])
def test_original_engine_matches_diagnostic_evaluator(task, reset):
    model = CpuSequence(700105)
    data = study.make_data(700105, task)
    for row in data["train"]["0"]:
        study.memory.train_local(model, row)
    before = model.parameter_digest()
    expected = study.memory.evaluate(model, data["test"]["16"][:8], reset=reset)
    assert study.original_evaluate(model, data["test"]["16"][:8], reset=reset) == expected
    assert model.parameter_digest() == before


def test_measure_is_readonly_for_model_and_dataset():
    model = CpuSequence(700106)
    data = study.make_data(700106, "copy3")
    state, before = canonical(model.snapshot()), copy.deepcopy(data)
    result = study.measure(model, data, 0)
    assert data == before and canonical(model.snapshot()) == state
    assert result["training"]["episodes"] == 16
    assert set(result["test"]) == {"0", "16"}


def test_small_end_to_end_retraining_and_original_engine_verification(tmp_path, monkeypatch):
    spec = copy.deepcopy(study.SPEC)
    spec.update(model_seeds=[700107], tasks=["copy3"], train_examples=8, test_examples=8,
                delays=[0, 2], checkpoint_episodes=[8, 16])
    monkeypatch.setattr(study, "SPEC", spec)
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "source_bindings", lambda: {"synthetic-source": "a" * 64})
    monkeypatch.setattr(study.memory, "memory_snapshot", lambda: {"available_system_ram_bytes": 1024**3})
    first, second = tmp_path / "artifacts/primary", tmp_path / "artifacts/rerun"
    monkeypatch.setattr(study, "PROCESS_TOKEN", "synthetic-primary")
    primary = study.run(first)
    monkeypatch.setattr(study, "PROCESS_TOKEN", "synthetic-rerun")
    rerun = study.run(second)
    assert primary["scientific_payload"] == rerun["scientific_payload"]
    monkeypatch.setattr(study, "PROCESS_TOKEN", "synthetic-verifier")
    result = study.verify(first, second, tmp_path / "verification.json")
    assert result["status"] == "PASS" and result["distinct_process_instances"] == 3
    assert result["original_engine_evaluation_episodes"] == 576
    with pytest.raises(FileExistsError):
        study.run(first)


def test_resource_guard_precedes_any_output(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study.memory, "memory_snapshot", lambda: {"available_system_ram_bytes": 1})
    output = tmp_path / "artifacts/run"
    with pytest.raises(MemoryError):
        study.run(output)
    assert not output.exists()
