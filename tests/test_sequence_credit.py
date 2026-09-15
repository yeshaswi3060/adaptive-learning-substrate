"""Nonselecting mathematical, intervention, chronology, and replay tests."""
import copy

import numpy as np
import pytest

from adaptive_learning_substrate import sequence_credit as study
from adaptive_learning_substrate.sequence_core import (
    CORE,
    PARAMETERS,
    CpuSequence,
    canonical,
    views,
)


@pytest.mark.parametrize("name", PARAMETERS)
def test_diagonal_gradient_is_exact_without_cross_unit_jacobian(name):
    model = CpuSequence(710001)
    views(model._buffer)["r"].fill(0)
    tokens, labels = [65, 130, 32, 65, 243], [-1, -1, 64, -1, 66]
    before = model._buffer.copy()
    loss, gradient = study.streaming_gradient(before, model._parents, tokens, labels, "diagonal_adam")
    exact_loss, exact, _ = study.memory.sequence_loss_gradient(before, model._parents, tokens, labels)
    assert loss == exact_loss
    np.testing.assert_allclose(gradient[name], exact[name], atol=1e-14, rtol=1e-12)
    index = (7, 65) if name in ("w", "g") else (7, 0) if name == "r" else (64, 7) if name == "o" else (64,) if name == "bo" else (7,)
    losses = []
    for sign in (-1, 1):
        changed = before.copy()
        views(changed)[name][index] += sign * 1e-5
        losses.append(study.memory.sequence_loss_gradient(changed, model._parents, tokens, labels)[0])
    assert gradient[name][index] == pytest.approx((losses[1] - losses[0]) / 2e-5, abs=2e-9, rel=2e-4)
    assert np.array_equal(before, model._buffer)


def test_diagonal_explicitly_omits_a_cross_unit_temporal_path():
    model = CpuSequence(710002)
    x = views(model._buffer)
    child, slot = 4, 0
    parent = model._parents[child, slot]
    x["o"].fill(0)
    x["o"][64, child] = .8
    x["r"].fill(0)
    x["r"][child, slot] = .7
    tokens, labels = [65, 130, 243], [-1, -1, 64]
    loss, gradient = study.streaming_gradient(model._buffer, model._parents, tokens, labels, "diagonal_adam")
    exact_loss, exact, _ = study.memory.sequence_loss_gradient(model._buffer, model._parents, tokens, labels)
    assert loss == exact_loss and gradient["w"][parent, 65] == 0
    assert abs(exact["w"][parent, 65]) > 1e-8


def test_feedback_and_diagonal_differ_only_by_feedback_matrix():
    model = CpuSequence(710003)
    x = views(model._buffer)
    x["b"][:] = x["o"].T
    first = study.streaming_gradient(model._buffer, model._parents, [65, 130, 243], [-1, -1, 64], "feedback_adam")
    second = study.streaming_gradient(model._buffer, model._parents, [65, 130, 243], [-1, -1, 64], "diagonal_adam")
    assert first[0] == second[0]
    for name in PARAMETERS:
        np.testing.assert_allclose(first[1][name], second[1][name], atol=1e-14, rtol=1e-12)


@pytest.mark.parametrize("mode", ["readout_adam", "feedback_adam", "diagonal_adam", "bptt_adam"])
def test_adam_modes_preserve_architecture_feedback_and_counter_contract(mode):
    model = CpuSequence(710004, core_learning=mode != "readout_adam")
    before = {k: a.copy() for k, a in views(model._buffer).items()}
    parents = model._parents.copy()
    row = study.memory.episode(study.rng_for(710004, "copy3", "test"), "copy3", 2)
    trainer = study.CreditTrainer(model, mode)
    trainer.train(row)
    assert trainer.iteration == 1 and model.updates == 3 and model.steps == len(row["prompt"]) + 2
    assert not model.pending and np.array_equal(parents, model._parents)
    assert np.array_equal(before["b"], views(model._buffer)["b"])
    assert any(not np.array_equal(before[k], views(model._buffer)[k]) for k in CORE) == (mode != "readout_adam")
    assert all(np.max(np.abs(views(model._buffer)[k])) <= 3 for k in PARAMETERS)


def test_shared_bptt_adam_matches_original_optimizer_bitwise():
    a, b = CpuSequence(710005), CpuSequence(710005)
    old, new = study.memory.ExactTrainer(a), study.CreditTrainer(b, "bptt_adam")
    rng = study.rng_for(710005, "copy3", "optimizer")
    for _ in range(6):
        row = study.memory.episode(rng, "copy3", 2)
        assert old.train(row) == new.train(row)
        assert canonical(a.snapshot()) == canonical(b.snapshot())
        for k in PARAMETERS:
            assert np.array_equal(old.m[k], new.m[k]) and np.array_equal(old.v[k], new.v[k])


@pytest.mark.parametrize("tokens,labels", [([], []), ([1], [-1]), ([256], [64]),
                                         ([1], [True]), ([1], [64, 65]), ([1] * 257, [64] * 257)])
def test_invalid_streaming_input_preserves_weights(tokens, labels):
    model = CpuSequence(710006)
    before = model._buffer.copy()
    with pytest.raises(ValueError):
        study.streaming_gradient(model._buffer, model._parents, tokens, labels, "diagonal_adam")
    assert np.array_equal(before, model._buffer)


@pytest.mark.parametrize("task", ["copy3", "recall4"])
def test_data_pairs_cues_interventions_and_never_reuses_fresh_noise(task):
    data = study.make_data(710007, task)
    assert data == study.make_data(710007, task)
    assert len(data["fresh_train"]) == 2048
    start = 5 if task == "copy3" else 10
    noise = lambda row: tuple(row["prompt"][start:start + 16])
    base_noise = {noise(r) for r in data["anchor"]}
    fresh_noise = {noise(r) for r in data["fresh_train"]}
    test_noise = {noise(r) for k in ("fresh_noise", "heldout16") for r in data["evaluation"][k]}
    assert len(fresh_noise) == 2048 and fresh_noise.isdisjoint(base_noise | test_noise)
    assert base_noise.isdisjoint(test_noise)
    for j, row in enumerate(data["fresh_train"]):
        base = data["anchor"][data["training_order"][j]]
        assert row["targets"] == base["targets"] and row["prompt"][:start] == base["prompt"][:start]
        assert row["prompt"][start + 16:] == base["prompt"][start + 16:]
    for i in range(0, 2048, 16):
        assert sorted(data["training_order"][i:i + 16]) == list(range(16))
    for before, changed in zip(data["anchor"], data["evaluation"]["changed_cue"], strict=True):
        assert noise(before) == noise(changed) and before["targets"] != changed["targets"]
        assert study.memory.oracle(changed["prompt"]) == changed["targets"]
    train_cues = {tuple(r["prompt"][:start] + r["prompt"][start + 16:]) for r in data["anchor"]}
    assert train_cues.isdisjoint(tuple(r["prompt"]) for r in data["evaluation"]["heldout0"])


def test_new_trainer_checkpoint_replays_in_original_engine(tmp_path):
    model = CpuSequence(710008)
    row = study.memory.episode(study.rng_for(710008, "copy3", "test"), "copy3", 2)
    study.CreditTrainer(model, "diagonal_adam").train(row)
    checkpoint = tmp_path / "model.json"
    model.save(checkpoint)
    original = CpuSequence()
    original.restore(checkpoint)
    for reset in (False, True):
        assert study.memory.evaluate(model, [row], reset=reset) == study.fitting.original_evaluate(original, [row], reset=reset)


def test_end_to_end_fresh_retraining_and_independent_engine(tmp_path, monkeypatch):
    spec = copy.deepcopy(study.SPEC)
    spec.update(model_seeds=[710009], tasks=["copy3"], train_examples=8, heldout_examples=8,
                fresh_noise_replicates=1, delay=2, checkpoint_episodes=[8, 16])
    monkeypatch.setattr(study, "SPEC", spec)
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "source_bindings", lambda: {"synthetic-source": "a" * 64})
    monkeypatch.setattr(study.memory, "memory_snapshot", lambda: {"available_system_ram_bytes": 1024**3})
    primary, rerun = tmp_path / "artifacts/primary", tmp_path / "artifacts/rerun"
    monkeypatch.setattr(study, "PROCESS_TOKEN", "synthetic-primary")
    first = study.run(primary)
    monkeypatch.setattr(study, "PROCESS_TOKEN", "synthetic-rerun")
    second = study.run(rerun)
    assert first["scientific_payload"] == second["scientific_payload"]
    monkeypatch.setattr(study, "PROCESS_TOKEN", "synthetic-verifier")
    result = study.verify(primary, rerun, tmp_path / "verification.json")
    assert result["status"] == "PASS" and result["original_engine_evaluation_episodes"] == 1600


def test_resource_guard_runs_before_output_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study.memory, "memory_snapshot", lambda: {"available_system_ram_bytes": 1})
    target = tmp_path / "artifacts/never_created"
    with pytest.raises(MemoryError):
        study.run(target)
    assert not target.exists()
