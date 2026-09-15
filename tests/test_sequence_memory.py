"""Synthetic, nonselecting checks for the prospective memory experiment."""
import copy

import numpy as np
import pytest

from adaptive_learning_substrate import sequence_memory as memory
from adaptive_learning_substrate.sequence_core import (
    CORE,
    PARAMETERS,
    CpuSequence,
    canonical,
    views,
)


@pytest.mark.parametrize("task", ["copy3", "recall4"])
def test_memory_data_is_deterministic_disjoint_and_oracle_correct(task):
    first = memory.make_dataset(700001, task)
    assert first == memory.make_dataset(700001, task)
    assert first != memory.make_dataset(700002, task)
    assert len(first["train"]) == 512
    assert {row["delay"] for row in first["train"]} == {4, 16}
    assert set(first["test"]) == {"4", "16", "32", "64"}
    prompts = {tuple(row["prompt"]) for row in first["train"]}
    for rows in first["test"].values():
        assert len(rows) == 128
        for row in rows:
            assert memory.oracle(row["prompt"]) == row["targets"]
            assert tuple(row["prompt"]) not in prompts


@pytest.mark.parametrize("task", ["copy3", "recall4"])
def test_supervision_has_only_answer_labels_and_no_future_target_inputs(task):
    row = memory.episode(memory.rng_for(700003, task, "test", 16), task, 16)
    tokens, labels = memory.supervised_tokens(row)
    assert tokens[:len(row["prompt"])] == row["prompt"]
    assert labels[:len(row["prompt"]) - 1] == [-1] * (len(row["prompt"]) - 1)
    assert labels[len(row["prompt"]) - 1:] == row["targets"]
    assert tokens[len(row["prompt"]):] == row["targets"][:-1]


def test_independent_forward_agrees_with_original_engine_without_mutating_it():
    model = CpuSequence(700004)
    x, state = views(model._buffer), np.zeros(memory.H)
    for token in (list(range(256)) + [64, 65, 243, 32] * 4):
        state, p, _, _ = memory.forward_one(x, model._parents, state, token)
        expected = model.step(token)
        assert np.array_equal(p, expected)
        assert np.array_equal(state, views(model._buffer)["s"])


@pytest.mark.parametrize("name", PARAMETERS)
def test_full_temporal_gradient_matches_finite_differences(name):
    model = CpuSequence(700005)
    tokens, labels = [65, 130, 32, 65, 243], [-1, -1, 64, -1, 66]
    before = model._buffer.copy()
    _, gradient, _ = memory.sequence_loss_gradient(before, model._parents, tokens, labels)
    index = (7, 65) if name in ("w", "g") else (7, 0) if name == "r" else (64, 7) if name == "o" else (64,) if name == "bo" else (7,)
    losses = []
    for sign in (-1, 1):
        candidate = before.copy()
        views(candidate)[name][index] += sign * 1e-5
        loss, _, _ = memory.sequence_loss_gradient(candidate, model._parents, tokens, labels)
        losses.append(loss)
    numeric = (losses[1] - losses[0]) / 2e-5
    assert gradient[name][index] == pytest.approx(numeric, abs=2e-9, rel=2e-4)
    assert np.array_equal(model._buffer, before)


def test_exact_gradient_follows_recurrent_cross_unit_path():
    model = CpuSequence(700006)
    x = views(model._buffer)
    child, slot = 4, 0
    parent = model._parents[child, slot]
    x["o"].fill(0)
    x["o"][64, child] = 0.8
    x["r"].fill(0)
    x["r"][child, slot] = 0.7
    tokens, labels = [65, 130, 243], [-1, -1, 64]
    _, gradient, _ = memory.sequence_loss_gradient(model._buffer, model._parents, tokens, labels)
    assert abs(gradient["w"][parent, 65]) > 1e-8
    without = model._buffer.copy()
    views(without)["r"][child, slot] = 0
    _, disconnected, _ = memory.sequence_loss_gradient(without, model._parents, tokens, labels)
    assert disconnected["w"][parent, 65] == 0


@pytest.mark.parametrize("tokens,labels", [([], []), ([1], [-1]), ([256], [64]),
    ([1], [True]), ([1], [64, 65]), ([1] * 257, [64] * 257)])
def test_invalid_gradient_input_does_not_mutate_model(tokens, labels):
    model = CpuSequence(700007)
    before = model._buffer.copy()
    with pytest.raises(ValueError):
        memory.sequence_loss_gradient(model._buffer, model._parents, tokens, labels)
    assert np.array_equal(model._buffer, before)


def test_gradient_descent_is_a_descent_direction_for_same_architecture():
    model = CpuSequence(700008)
    row = memory.episode(memory.rng_for(700008, "copy3", "test", 4), "copy3", 4)
    tokens, labels = memory.supervised_tokens(row)
    loss, gradient, _ = memory.sequence_loss_gradient(model._buffer, model._parents, tokens, labels)
    changed = model._buffer.copy()
    for name in PARAMETERS:
        views(changed)[name] -= 0.01 * gradient[name]
    after, _, _ = memory.sequence_loss_gradient(changed, model._parents, tokens, labels)
    assert after < loss


@pytest.mark.parametrize("mode", ["local", "readout", "bptt"])
def test_all_training_modes_preserve_topology_feedback_and_target_budget(mode):
    model = CpuSequence(700009, core_learning=mode != "readout")
    before = {name: array.copy() for name, array in views(model._buffer).items()}
    parents = model._parents.copy()
    row = memory.episode(memory.rng_for(700009, "copy3", "test", 4), "copy3", 4)
    if mode == "bptt":
        trainer = memory.ExactTrainer(model)
        trainer.train(row)
        assert trainer.iteration == 1
    else:
        memory.train_local(model, row)
    after = views(model._buffer)
    assert model.updates == 3 and model.steps == len(row["prompt"]) + 2
    assert any(not np.array_equal(before[name], after[name]) for name in CORE) == (mode != "readout")
    assert np.array_equal(parents, model._parents)
    assert np.array_equal(before["b"], after["b"])
    assert not model.pending


def test_free_running_evaluation_never_uses_targets_and_reset_removes_context():
    model = CpuSequence(700010)
    first = memory.episode(memory.rng_for(700010, "copy3", "test", 4), "copy3", 4)
    second = copy.deepcopy(first)
    second["prompt"][1:4] = [70, 71, 70]
    second["targets"] = [70, 71, 70]
    before = canonical(model.snapshot())
    a, p = memory.predict_answer(model, first["prompt"], 3)
    changed_labels = copy.deepcopy(first)
    changed_labels["targets"] = [0, 0, 0]
    assert memory.evaluate(model, [first])["rows"][0]["free_predictions"] == a
    assert memory.evaluate(model, [changed_labels])["rows"][0]["free_predictions"] == a
    assert memory.predict_answer(model, first["prompt"], 3, reset_before_query=True) == memory.predict_answer(model, second["prompt"], 3, reset_before_query=True)
    assert not np.array_equal(p, memory.predict_answer(model, second["prompt"], 3)[1])
    assert canonical(model.snapshot()) == before


def test_teacher_labels_do_not_change_preceding_probabilities():
    model = CpuSequence(700011)
    tokens = [64, 65, 130, 243]
    _, _, first = memory.sequence_loss_gradient(model._buffer, model._parents, tokens, [-1, -1, -1, 64])
    _, _, second = memory.sequence_loss_gradient(model._buffer, model._parents, tokens, [-1, -1, -1, 71])
    assert np.array_equal(first, second)


def test_exact_checkpoint_runs_in_original_cpu_engine(tmp_path):
    model = CpuSequence(700012)
    row = memory.episode(memory.rng_for(700012, "recall4", "test", 4), "recall4", 4)
    memory.ExactTrainer(model).train(row)
    path = tmp_path / "exact.json"
    model.save(path)
    original = CpuSequence()
    original.restore(path)
    for token in row["prompt"]:
        expected = original.step(token)
    _, probabilities = memory.predict_answer(model, row["prompt"], 1)
    assert np.array_equal(probabilities[0], expected)


def test_diagnostic_never_passes_empty_or_missing_conditions():
    assert memory.diagnostic_pass([]) is False


def test_evidence_metrics_recompute_and_reject_invalid_probabilities():
    rows = [{"targets": [64, 65, 66], "free_predictions": [64, 65, 70],
             "teacher_target_probabilities": [0.5, 0.25, 0.125]}]
    metric = memory.summarize_evidence(rows, reset=False)
    assert metric["token_accuracy"] == pytest.approx(2 / 3)
    assert metric["exact_accuracy"] == 0
    assert metric["teacher_nll"] == pytest.approx(-np.log([0.5, 0.25, 0.125]).mean())
    rows[0]["teacher_target_probabilities"][0] = float("nan")
    with pytest.raises(ValueError, match="probability"):
        memory.summarize_evidence(rows, reset=False)


def test_small_synthetic_end_to_end_retraining_and_independent_evaluator(tmp_path, monkeypatch):
    spec = copy.deepcopy(memory.SPEC)
    spec.update(model_seeds=[700013], train_episodes=8, eval_episodes_per_delay=2,
                train_delays=[1, 2], eval_delays=[1, 2])
    monkeypatch.setattr(memory, "SPEC", spec)
    monkeypatch.setattr(memory, "ROOT", tmp_path)
    monkeypatch.setattr(memory, "source_bindings", lambda: {"synthetic-source": "a" * 64})
    monkeypatch.setattr(memory, "memory_snapshot", lambda: {"available_system_ram_bytes": 1024**3})
    first, second = tmp_path / "artifacts/first", tmp_path / "artifacts/second"
    monkeypatch.setattr(memory, "PROCESS_TOKEN", "synthetic-primary")
    primary = memory.run(first)
    monkeypatch.setattr(memory, "PROCESS_TOKEN", "synthetic-rerun")
    rerun = memory.run(second)
    assert primary["scientific_payload"] == rerun["scientific_payload"]
    monkeypatch.setattr(memory, "PROCESS_TOKEN", "synthetic-verifier")
    result = memory.verify(first, second, tmp_path / "verified.json")
    assert result["status"] == "PASS"
    assert result["independent_evaluation_episodes"] == 48
    assert result["maximum_probability_error"] == 0
    assert result["distinct_process_instance_count"] == 3
    with pytest.raises(FileExistsError):
        memory.run(first)


def test_resource_failure_precedes_experiment_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "ROOT", tmp_path)
    monkeypatch.setattr(memory, "memory_snapshot", lambda: {"available_system_ram_bytes": 1})
    with pytest.raises(MemoryError):
        memory.run(tmp_path / "artifacts/run")
    assert not (tmp_path / "artifacts/run").exists()
