"""Small independent CPU tests; hardware parity is run by the frozen experiment."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from adaptive_learning_substrate.sequence_core import (
    CONFIG,
    CORE,
    PARAMETERS,
    SIZE,
    CpuSequence,
    H,
    canonical,
    initial,
    views,
    write_record,
)


@pytest.mark.parametrize("token", [-1, 256, True, 1., None, "a", np.int64(1)])
def test_bad_byte_is_rejected_without_mutation(token):
    model = CpuSequence()
    before = canonical(model.snapshot())
    with pytest.raises(ValueError, match="integer byte"):
        model.step(token)
    assert canonical(model.snapshot()) == before


def test_all_bytes_have_normalized_predictions_and_bounded_state():
    model = CpuSequence()
    original = model.parameter_digest()
    for token in range(256):
        p = model.step(token)
        assert p.shape == (256,) and np.isfinite(p).all() and np.all(p >= 0)
        assert p.sum() == pytest.approx(1)
    assert model.parameter_digest() == original
    x = views(model._export()[0])
    assert np.max(abs(x["s"])) <= 1
    assert all(np.max(abs(x["e_"+key])) <= 4 for key in CORE)


def test_prediction_before_target_and_explicit_cancel():
    model = CpuSequence()
    with pytest.raises(RuntimeError, match="pending"):
        model.learn(1)
    model.step(1, training=True)
    before = canonical(model.snapshot())
    for action in (lambda: model.step(2), model.reset, lambda: model.learn(300)):
        with pytest.raises((ValueError, RuntimeError)):
            action()
        assert canonical(model.snapshot()) == before
    model.learn(2)
    with pytest.raises(RuntimeError):
        model.learn(2)
    model.step(3, training=True)
    model.cancel_prediction()
    model.reset()
    assert not model.pending and model.updates == 1


def test_chunks_do_not_reset_causal_state_and_order_matters():
    whole, chunks, other = CpuSequence(), CpuSequence(), CpuSequence()
    data = b"a byte sequence with several chunks"
    a = [whole.step(token) for token in data]
    b = [chunks.step(token) for chunk in (data[:3], data[3:11], data[11:]) for token in chunk]
    assert np.array_equal(a, b)
    for token in reversed(data):
        other.step(token)
    assert not np.allclose(whole.predict_next(), other.predict_next(), atol=1e-12, rtol=0)


def test_prefix_predictions_do_not_depend_on_future_suffix():
    first, second = CpuSequence(), CpuSequence()
    prefix = b"same past"
    a = [first.step(token) for token in prefix]
    b = [second.step(token) for token in prefix]
    for token in b"different future":
        second.step(token)
    assert np.array_equal(a, b)
    assert np.array_equal(a[-1], first.predict_next())


@pytest.mark.parametrize("learning", [True, False])
def test_core_plasticity_switch_and_readonly_evaluation(learning):
    model = CpuSequence(core_learning=learning)
    before = {key: value.copy() for key, value in views(model._export()[0]).items()}
    for token, target in zip(b"sequence", b"equence!", strict=True):
        model.step(token, training=True)
        model.learn(target)
    after = views(model._export()[0])
    assert not np.array_equal(before["o"], after["o"])
    assert any(not np.array_equal(before[key], after[key]) for key in CORE) == learning
    assert np.array_equal(before["b"], after["b"])
    digest = model.parameter_digest()
    for token in b"evaluation":
        model.step(token)
    assert model.parameter_digest() == digest


@pytest.mark.parametrize("name", CORE)
def test_one_step_local_eligibility_matches_finite_difference(name):
    model = CpuSequence()
    x = views(model._buffer)
    x["s"][:] = np.linspace(-.3, .3, H)
    base, parents = model._export()
    row, token = 7, 65
    index = (row, token) if name in ("w", "g") else (row, 0) if name == "r" else (row,)
    model.step(token)
    actual = views(model._export()[0])["e_"+name][index]
    values = []
    for sign in (-1, 1):
        candidate = CpuSequence()
        candidate._install(base, parents)
        views(candidate._buffer)[name][index] += sign*1e-6
        candidate.step(token)
        values.append(views(candidate._export()[0])["s"][row])
    assert actual == pytest.approx((values[1]-values[0])/2e-6, abs=2e-10, rel=1e-7)


def test_checkpoint_restores_pending_learning_and_exact_continuation(tmp_path):
    first, second = CpuSequence(), CpuSequence(123)
    for token in b"abc":
        first.step(token)
    first.step(100, training=True)
    path = tmp_path / "model.json"
    first.save(path)
    second.restore(path)
    first.learn(101)
    second.learn(101)
    assert canonical(first.snapshot()) == canonical(second.snapshot())
    assert np.array_equal(first.step(102), second.step(102))


@pytest.mark.parametrize("corruption", ["sidecar", "config", "self_edge", "state", "feedback", "chronology"])
def test_bad_checkpoint_does_not_modify_live_model(tmp_path, corruption):
    model = CpuSequence()
    snapshot = model.snapshot()
    if corruption == "config":
        snapshot["config"]["hidden"] += 1
    elif corruption == "self_edge":
        snapshot["parents"][0][0] = 0
    elif corruption in ("state", "feedback"):
        from adaptive_learning_substrate.sequence_core import OFFSETS
        snapshot["buffer"][OFFSETS["s" if corruption == "state" else "b"]] = 100
    elif corruption == "chronology":
        snapshot["updates"] = 10
    path = tmp_path / "bad.json"
    write_record(path, snapshot)
    if corruption == "sidecar":
        path.write_text("{}")
    before = canonical(model.snapshot())
    with pytest.raises(ValueError):
        model.restore(path)
    assert canonical(model.snapshot()) == before
    assert CONFIG["hidden"] == H


def test_checkpoint_sidecar_matches_project_binding_schema(tmp_path):
    path = tmp_path / "model.json"
    CpuSequence().save(path)
    binding = json.loads(path.with_suffix(".json.sha256").read_text())
    assert binding == {"algorithm": "sha256", "report_file": "model.json", "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(FileExistsError):
        CpuSequence().save(path)


def test_reset_preserves_weights_and_returns_uniform_initial_state():
    model = CpuSequence()
    model.step(1, training=True)
    model.learn(2)
    digest = model.parameter_digest()
    model.reset()
    assert model.parameter_digest() == digest
    x = views(model._export()[0])
    assert not x["s"].any() and not any(x["e_"+key].any() for key in CORE)
    assert np.array_equal(model.predict_next(), np.full(256, 1/256))


def test_layout_is_bounded_and_closed_model_rejects_work():
    assert initial(0)[0].size == SIZE and SIZE*8 < 1024**2
    model = CpuSequence()
    model.close()
    for action in (lambda: model.step(1), model.reset, model.snapshot, model.predict_next):
        with pytest.raises(RuntimeError, match="closed"):
            action()


def test_no_torch_import_for_sequence_implementation():
    result = subprocess.run([sys.executable, "-B", "-c", "import sys; import adaptive_learning_substrate.sequence_cuda; assert 'torch' not in sys.modules; print('NO_TORCH_IMPORT')"], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "NO_TORCH_IMPORT"


def test_initial_parameter_bounds_and_sparse_no_self_topology():
    buffer, parents = initial(15000)
    assert all(np.max(abs(views(buffer)[key])) <= 3 for key in PARAMETERS)
    assert all(i not in row and len(set(row)) == 4 for i, row in enumerate(parents))
