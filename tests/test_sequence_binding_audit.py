"""Nonselecting synthetic checks for the frozen-state diagnostic."""
import copy

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from adaptive_learning_substrate import sequence_binding_audit as study
from adaptive_learning_substrate.sequence_core import (
    CpuSequence,
    H,
    canonical,
    read_record,
    write_record,
)


@pytest.fixture(scope="module")
def datasets():
    return {task: study.diversity.make_data(730001, task) for task in ("copy3", "recall4")}


@pytest.mark.parametrize("task", ["copy3", "recall4"])
def test_boundaries_full_values_and_stateful_reference_match(datasets, task):
    data = datasets[task]
    rows = data["heldout16"][:8] + data["counterfactual16"][:8]
    model = CpuSequence(730002)
    model.step(99)
    before = canonical(model.snapshot())
    prompts = [row["prompt"] for row in rows]
    a = study.extract_states(model, task, prompts)
    b = study.extract_states(model, task, prompts, original=True)
    for stage in study.SPEC["stages"]:
        np.testing.assert_array_equal(a[stage], b[stage])
        assert a[stage].shape == (16, H)
    for row in rows:
        full = study.full_values(task, row["prompt"])
        answer = full if task == "copy3" else [full[row["prompt"][-1] - 32]]
        assert answer == row["targets"]
    assert canonical(model.snapshot()) == before


@pytest.mark.parametrize("task", ["copy3", "recall4"])
def test_cue_state_precedes_noise_and_query(datasets, task):
    row = datasets[task]["heldout16"][0]
    prompt = row["prompt"]
    altered = prompt[:]
    start, end = study.prompt_layout(task, prompt)
    altered[start:end] = [128 + (v - 128 + 1) % 16 for v in altered[start:end]]
    if task == "recall4":
        altered[-1] = 32 + (altered[-1] - 32 + 1) % 4
    model = CpuSequence(730003)
    features = study.extract_states(model, task, [prompt, altered])
    np.testing.assert_array_equal(features["cue_end"][0], features["cue_end"][1])
    assert not np.array_equal(features["delay_end"][0], features["delay_end"][1])


def test_zero_delay_boundaries_both_recorded(datasets):
    row = datasets["copy3"]["heldout0"][0]
    features = study.extract_states(CpuSequence(730004), "copy3", [row["prompt"]])
    np.testing.assert_array_equal(features["cue_end"], features["delay_end"])


@pytest.mark.parametrize("edit", ["key", "marker", "value", "noise", "bool"])
def test_invalid_prompts_rejected_before_forward(datasets, edit):
    prompt = datasets["recall4"]["heldout16"][0]["prompt"][:]
    if edit == "key":
        prompt[1] = prompt[3]
    elif edit == "marker":
        prompt[9] = 0
    elif edit == "value":
        prompt[2] = 100
    elif edit == "noise":
        prompt[10] = 64
    else:
        prompt[10] = True
    with pytest.raises(ValueError):
        study.extract_states(CpuSequence(), "recall4", [prompt])


def test_ridge_matches_independent_augmented_least_squares():
    rng = np.random.default_rng(730005)
    x = rng.normal(size=(80, H))
    labels = rng.integers(64, 72, size=(80, 3))
    y = np.eye(8)[labels - 64].reshape(80, -1)
    with threadpool_limits(limits=1):
        probe = study.fit_probe(x, labels)
        design = np.column_stack([x, np.ones(len(x))])
        regularizer = np.zeros((H, H + 1))
        regularizer[:, :H] = np.sqrt(len(x) * study.SPEC["ridge_alpha"]) * np.eye(H)
        fitted = np.linalg.lstsq(np.vstack([design, regularizer]),
                                np.vstack([y, np.zeros((H, 24))]), rcond=None)[0]
    np.testing.assert_allclose(probe["weights"], fitted[:-1], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(probe["bias"], fitted[-1], rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("invalid", ["nan", "float_labels", "label_range", "shape", "empty"])
def test_invalid_probe_training_arrays(invalid):
    x, y = np.zeros((16, H)), np.full((16, 3), 64)
    if invalid == "nan":
        x[0, 0] = np.nan
    elif invalid == "float_labels":
        y = y.astype(float)
    elif invalid == "label_range":
        y[0, 0] = 72
    elif invalid == "shape":
        x = x[:, :-1]
    else:
        x, y = x[:0], y[:0]
    with pytest.raises(ValueError):
        study.fit_probe(x, y)


def test_controls_preserve_marginals_and_zero_state_is_intercept_only():
    labels = np.tile(np.arange(64, 72)[:, None], (2, 3))
    order = study.permutation(730006, "copy3", len(labels))
    np.testing.assert_array_equal(order, study.permutation(730006, "copy3", len(labels)))
    assert not np.array_equal(order, np.arange(len(labels)))
    np.testing.assert_array_equal(np.sort(labels, axis=0), np.sort(labels[order], axis=0))
    probe = study.fit_probe(np.zeros((len(labels), H)), labels)
    assert not np.any(probe["weights"])
    np.testing.assert_array_equal(study.predict_probe(probe, np.zeros((10, H))), np.full((10, 3), 64))


@pytest.mark.parametrize("task", ["copy3", "recall4"])
def test_probe_test_separation_and_counterfactual_pairing(datasets, task):
    data = copy.deepcopy(datasets[task])
    assert len(study.audit_sets(data)["fit"]) == 128
    data["all_seen"][0] = data["counterfactual16"][0]
    with pytest.raises(ValueError, match="leaked"):
        study.audit_sets(data)
    data = copy.deepcopy(datasets[task])
    data["counterfactual16"].reverse()
    with pytest.raises(ValueError, match="pairing"):
        study.audit_sets(data)


def test_read_only_audit_replays_and_shows_paired_scores(datasets):
    data = datasets["recall4"]
    model = CpuSequence(730007)
    before, frozen_data = canonical(model.snapshot()), canonical(data)
    with threadpool_limits(limits=1):
        a = study.audit_checkpoint(model, data)
        b = study.audit_checkpoint(model, data, original=True)
    assert a == b
    assert canonical(model.snapshot()) == before and canonical(data) == frozen_data
    for stage in a["stages"].values():
        for control in stage.values():
            pair = control["paired_answer_exact"]
            assert pair <= control["scores"]["heldout"]["answer_exact_accuracy"]
            assert pair <= control["scores"]["counterfactual"]["answer_exact_accuracy"]


def test_predicates_require_controls_and_pair_consistency():
    stages = {s: {c: {"scores": {"heldout": {"full_value_token_accuracy": .9 if c == "aligned" else .125,
                                           "answer_token_accuracy": .9 if c == "aligned" else .125}},
                      "paired_answer_exact": .75}
                  for c in study.SPEC["controls"]} for s in study.SPEC["stages"]}
    native = {"heldout": {"normal": {"token_accuracy": .5}}}
    assert study.predicates(stages, native) == {"encoding_accessible": True, "delay_drop": False,
                                              "query_drop": False, "diagnostic_decoder_gap": True}
    stages["delay_end"]["aligned"]["scores"]["heldout"]["full_value_token_accuracy"] = .5
    stages["query_end"]["aligned"]["paired_answer_exact"] = .1
    flags = study.predicates(stages, native)
    assert flags["delay_drop"] and not flags["diagnostic_decoder_gap"]


def test_complete_synthetic_audit_replay_and_tamper_rejection(tmp_path, monkeypatch):
    old = study.diversity
    spec = copy.deepcopy(old.SPEC)
    spec.update(model_seeds=[730008], tasks=["copy3"], cue_counts=[8, 16],
                modes=["readout_adam", "feedback_adam"], heldout_examples=8,
                delay=2, checkpoint_episodes=[16, 32])
    monkeypatch.setattr(old, "SPEC", spec)
    monkeypatch.setattr(old, "ROOT", tmp_path)
    monkeypatch.setattr(study, "ROOT", tmp_path)
    anchor = tmp_path / "source.txt"
    anchor.write_text("synthetic source")
    sources = {str(anchor): study.memory.file_digest(anchor)}
    monkeypatch.setattr(old, "source_bindings", lambda: sources)
    monkeypatch.setattr(study, "source_bindings", lambda: sources)
    monkeypatch.setattr(study.memory, "memory_snapshot", lambda: {"available_system_ram_bytes": 1024**3})
    upstream = tmp_path / "artifacts/historical/primary"
    report = old.run(upstream)
    audit_spec = copy.deepcopy(study.SPEC)
    audit_spec.update(cue_count=16, source_payload_sha256=report["scientific_payload_sha256"])
    monkeypatch.setattr(study, "SPEC", audit_spec)
    write_record(upstream.parent / "INDEPENDENT_VERIFICATION.json",
                 {"status": "PASS", "scientific_payload_sha256": report["scientific_payload_sha256"],
                  "distinct_process_instances": 3, "fresh_training_exact": True,
                  "original_engine_results_exact": True})
    primary, rerun = tmp_path / "artifacts/audit-primary", tmp_path / "artifacts/audit-rerun"
    monkeypatch.setattr(study, "PROCESS_TOKEN", "synthetic-primary")
    a = study.run(upstream, primary)
    monkeypatch.setattr(study, "PROCESS_TOKEN", "synthetic-rerun")
    b = study.run(upstream, rerun)
    assert a["scientific_payload"] == b["scientific_payload"]
    monkeypatch.setattr(study, "PROCESS_TOKEN", "synthetic-verifier")
    result = study.verify(primary, rerun, tmp_path / "verification.json")
    assert result["status"] == "PASS" and result["checkpoints"] == 4
    assert result["original_engine_state_episodes"] == 128
    assert read_record(tmp_path / "verification.json") == result
    monkeypatch.setattr(study, "PROCESS_TOKEN", "synthetic-primary")
    with pytest.raises(RuntimeError, match="distinct processes"):
        study.verify(primary, rerun, tmp_path / "duplicate.json")
    monkeypatch.setattr(study, "PROCESS_TOKEN", "synthetic-verifier")
    damaged = primary / a["scientific_payload"]["results"][0]["file"]
    damaged.write_bytes(damaged.read_bytes() + b" ")
    with pytest.raises(ValueError, match="sidecar mismatch"):
        study.verify(primary, rerun, tmp_path / "tampered.json")
    assert not (tmp_path / "tampered.json").exists()
    anchor.write_text("source drift")
    with pytest.raises(RuntimeError, match="input changed"):
        study.load_source(upstream)


def test_resource_guard_precedes_output_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study.memory, "memory_snapshot", lambda: {"available_system_ram_bytes": 1})
    target = tmp_path / "artifacts/absent"
    with pytest.raises(MemoryError):
        study.run(tmp_path / "not_read", target)
    assert not target.exists()
