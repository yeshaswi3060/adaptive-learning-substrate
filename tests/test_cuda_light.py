"""CPU-only contracts for lightweight CUDA. Real GPU checks use the frozen probe."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from adaptive_learning_substrate.cuda_light import LightForward, LightHead, _check
from adaptive_learning_substrate.gpu_light_experiment import (
    SEEDS,
    admission_gates,
    read,
    run,
    write,
)


def test_light_import_does_not_load_torch():
    result = subprocess.run([sys.executable, "-B", "-c", "import sys; import adaptive_learning_substrate.cuda_light; assert 'torch' not in sys.modules; print('NO_TORCH_IMPORT')"], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "NO_TORCH_IMPORT"


@pytest.mark.parametrize("code", [1, 2, 999])
def test_driver_errors_fail_closed(code):
    with pytest.raises(RuntimeError, match=str(code)):
        _check(code, "test")


def test_zero_driver_status_accepted():
    assert _check(0, "test") is None


@pytest.mark.parametrize("cues,noise", [([], []), ([True], [[]]), ([1], [[0]]), ([1]*33, [[]]*33)])
def test_input_rejected_before_driver_access(cues, noise):
    gpu = LightForward(None, None)
    with pytest.raises(ValueError):
        gpu.run(cues, noise)


def test_tail_rejects_wrong_length_and_missing_cue():
    gpu = LightForward(None, None)
    for cues, noise in [([1], [[]]), ([None], [[1]*256])]:
        with pytest.raises(ValueError, match="tail requires"):
            gpu.run(cues, noise, tail=True)


def test_head_invalid_target_does_not_touch_driver():
    head = object.__new__(LightHead)
    head.pending = None
    for target in (0, True, 1., None):
        with pytest.raises(ValueError, match="bipolar"):
            head.reveal_target(0, target)
    with pytest.raises(RuntimeError, match="pending"):
        head.reveal_target(0, 1)


def test_records_are_exclusive_and_sidecar_validated(tmp_path):
    path = tmp_path / "record.json"
    write(path, {"pass": True})
    assert set(json.loads(path.with_suffix(".json.sha256").read_text())) == {"algorithm", "report_file", "report_sha256"}
    assert read(path) == {"pass": True}
    with pytest.raises(FileExistsError):
        write(path, {"pass": False})
    path.write_text('{"pass":false}')
    with pytest.raises(ValueError, match="sidecar"):
        read(path)


def test_nonfinite_evidence_rejected_before_file_creation(tmp_path):
    path = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        write(path, {"metric": float("nan")})
    assert not path.exists()


def test_official_output_namespace_rejected():
    with pytest.raises(ValueError, match="new immediate child"):
        run(Path("artifacts/experiment_000/test"), "probe", [])


@pytest.mark.parametrize("seeds", [(), SEEDS[:-1], tuple(reversed(SEEDS)), tuple(range(90, 95))])
def test_admission_rejects_wrong_seeds_before_gate_calculation(seeds):
    with pytest.raises(ValueError, match="five ordered seeds"):
        admission_gates([{"seed": seed} for seed in seeds])


def test_admission_thresholds_match_unchanged_reference():
    from adaptive_learning_substrate import experiment_000_lwoh_l1_execution_v3 as ref
    for signal in (0., .1, .5, 1., 3.):
        metrics = {"C": signal, "R": ref._guarded_ratio(signal, .1), "O": ref._guarded_ratio(signal, 1.)}
        ratio = ref._guarded_ratio(1., 1.)
        split = {"candidate_metrics": metrics, "native_metrics": {"C": .2, "R": ref._guarded_ratio(.2, .3)},
                 "split_event_ratio": ratio, "split_forward_touch_ratio": ratio,
                 "activity": [{"candidate_event_ratio": ratio, "candidate_forward_touch_ratio": ratio}]}
        rows = [{"seed": seed, "train": split, "eval": split, "ridge_candidate": {"eval_accuracy": .75},
                 "tail": {"pass": True}, "all_invariants_pass": True} for seed in SEEDS]
        development = admission_gates(rows)
        # Synthetic unit-test rows only; scientific evidence never relabels seeds.
        expected = ref.recompute_admission_gates([{**row, "seed": seed} for row, seed in zip(rows, ref.ADMISSION_SEEDS, strict=True)])
        assert {**development["gates"], "A10": development["gates"]["A10_local_only"]}.items() >= expected["gates"].items()
        assert development["readiness"] == 21
