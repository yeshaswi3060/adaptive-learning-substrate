from __future__ import annotations

import copy
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from adaptive_learning_substrate import experiment_000_lwoh_l1_execution_v3 as v3
from adaptive_learning_substrate.experiment_000 import FROZEN_GRAPH_OPTIONS
from adaptive_learning_substrate.recurrent import build_experiment_000_graph

SCRATCH_MASTER_SEEDS = (9090, 9091)


def _synthetic_admission_row(
    seed: int,
    *,
    candidate_c: float = 1.0,
    candidate_n: float = 1.0,
    native_c: float = 0.1,
    native_n: float = 1.0,
    output_cue: float = 1.0,
    output_sd: float = 1.0,
    ridge_accuracy: float = 0.9,
    invariant_pass: bool = True,
    tail_pass: bool = True,
) -> dict[str, Any]:
    """Build a metric-only fixture; it never executes or reads an assigned seed."""

    candidate_metrics = {
        "C": candidate_c,
        "N": candidate_n,
        "R": v3._guarded_ratio(candidate_c, candidate_n),
        "O": v3._guarded_ratio(output_cue, output_sd),
    }
    native_metrics = {
        "C": native_c,
        "N": native_n,
        "R": v3._guarded_ratio(native_c, native_n),
        "O": v3._guarded_ratio(native_c, native_n),
    }
    activity = [
        {
            "candidate_event_ratio": 1.0,
            "candidate_forward_touch_ratio": 1.0,
        }
    ]
    split = {
        "candidate_metrics": candidate_metrics,
        "native_metrics": native_metrics,
        "split_event_ratio": 1.0,
        "split_forward_touch_ratio": 1.0,
        "activity": activity,
    }
    return {
        "seed": seed,
        "train": copy.deepcopy(split),
        "eval": copy.deepcopy(split),
        "ridge_candidate": {"eval_accuracy": ridge_accuracy},
        "ridge_native": {"eval_accuracy": 0.5},
        "tail": {"pass": tail_pass},
        "all_invariants_pass": invariant_pass,
    }


def _synthetic_admission_rows(**overrides: Any) -> list[dict[str, Any]]:
    # These are inert row identifiers required by the gate recomputer. No RNG,
    # graph, artifact, or assigned-seed result is generated or loaded here.
    return [
        _synthetic_admission_row(seed, **overrides) for seed in v3.ADMISSION_SEEDS
    ]


def test_canonical_object_hash_excludes_file_newline() -> None:
    value = {"z": 1, "a": "café"}
    object_bytes = b'{"a":"caf\\u00e9","z":1}'

    assert v3._canonical_object_bytes(value) == object_bytes
    assert v3._canonical_bytes(value) == object_bytes + b"\n"
    assert v3._sha256_json(value) == hashlib.sha256(object_bytes).hexdigest()
    assert v3._sha256_json(value) != hashlib.sha256(object_bytes + b"\n").hexdigest()

    with pytest.raises(ValueError, match="compliant|range|NaN"):
        v3._canonical_bytes({"bad": float("nan")})


def test_write_record_is_canonical_self_hashed_and_sidecar_bound(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    record = {
        "schema_version": "synthetic-v1",
        "status": "PASS",
        "value": 7,
    }

    final = v3._write_record(path, record, self_field="payload_sha256")
    raw = path.read_bytes()
    sidecar_raw = path.with_suffix(".sha256").read_bytes()
    sidecar = json.loads(sidecar_raw)

    assert raw == v3._canonical_bytes(final)
    assert raw.endswith(b"\n")
    assert final["payload_sha256"] == v3._self_hash(final, "payload_sha256")
    assert sidecar == {
        "algorithm": "sha256",
        "report_file": "record.json",
        "report_sha256": hashlib.sha256(raw).hexdigest(),
    }
    assert sidecar_raw == v3._canonical_bytes(sidecar)
    assert v3._verify_record(
        path, schema="synthetic-v1", self_field="payload_sha256"
    ) == final

    assert v3._write_record(path, record, self_field="payload_sha256") == final


@pytest.mark.parametrize("tamper", ["report", "sidecar", "extra-sidecar-key"])
def test_record_verifier_fails_closed_on_tampering(
    tmp_path: Path, tamper: str
) -> None:
    path = tmp_path / "record.json"
    record = {"schema_version": "synthetic-v1", "status": "PASS"}
    final = v3._write_record(path, record, self_field="payload_sha256")

    if tamper == "report":
        changed = dict(final)
        changed["status"] = "ALTERED"
        path.write_bytes(v3._canonical_bytes(changed))
    else:
        sidecar_path = path.with_suffix(".sha256")
        sidecar = json.loads(sidecar_path.read_bytes())
        if tamper == "sidecar":
            sidecar["report_sha256"] = "0" * 64
        else:
            sidecar["unexpected"] = True
        sidecar_path.write_bytes(v3._canonical_bytes(sidecar))

    with pytest.raises(RuntimeError, match="self-hash|sidecar"):
        v3._verify_record(
            path, schema="synthetic-v1", self_field="payload_sha256"
        )


def test_v1_rng_namespace_known_vectors_and_disjointness() -> None:
    assert v3.SCIENTIFIC_RNG_NAMESPACE == "experiment-000-lwoh-l1-v1"
    assert v3._derive_seed(9090, "d4", "train", "cue") == (
        55741833488850947981006596922386973496
    )
    assert v3._derive_seed(9090, "d4", "train", "noise") == (
        154148289980798510087582736813048858193
    )
    assert v3._derive_seed(9091, "d16", "test", "random_prediction") == (
        6007819953503050446198913289901811316
    )

    realized = {
        v3._derive_seed(master, cell, split, purpose)
        for master in SCRATCH_MASTER_SEEDS
        for cell in ("d4", "d8", "d16")
        for split in ("train", "eval")
        for purpose in (
            "cue",
            "noise",
            "sample_identity",
            "independent_label",
            "random_prediction",
        )
    }
    assert len(realized) == 2 * 3 * 2 * 5


def test_cell_random_predictions_are_balanced_in_an_independent_pcg64_stream() -> None:
    cell = v3.generate_cell(
        master_seed=9090,
        cell="d4",
        split="train",
        count=64,
        noise_events=4,
        cue_visible=True,
    )
    duplicate = v3.generate_cell(
        master_seed=9090,
        cell="d4",
        split="train",
        count=64,
        noise_events=4,
        cue_visible=True,
    )
    expected = v3._balanced_bipolar(
        64,
        np.random.Generator(
            np.random.PCG64(cell.namespace_seeds["random_prediction"])
        ),
    )

    assert cell == duplicate
    assert cell.random_predictions == tuple(int(value) for value in expected)
    assert sum(value == 1 for value in cell.random_predictions) == 32
    assert sum(item.target == 1 for item in cell.supervision) == 32
    assert sum(value == 1 for value in cell.independent_labels) == 32
    assert cell.random_predictions != tuple(item.target for item in cell.supervision)
    assert cell.random_predictions != cell.independent_labels


def test_online_head_requires_prediction_before_target_reveal() -> None:
    head = v3.OnlineHead()
    feature = (0.2, -0.3, 0.4, -0.1, 0.5, -0.6, 0.7, -0.8)

    pending = head.predict_for_update(feature)
    assert pending.sequence == 0
    assert head.chronology_ledger == {
        "predictions": 1,
        "updates": 0,
        "pending_prediction": True,
    }
    with pytest.raises(RuntimeError, match="previous prediction"):
        head.predict_for_update(feature)

    wrong = v3.PendingPrediction(
        sequence=1,
        feature=pending.feature,
        score=pending.score,
        prediction=pending.prediction,
    )
    with pytest.raises(RuntimeError, match="pending prediction"):
        head.reveal_target(wrong, 1)
    assert head.chronology_ledger["pending_prediction"] is True

    delta = head.reveal_target(pending, 1)
    assert delta.shape == (8,)
    assert np.any(delta != 0.0)
    assert head.chronology_ledger == {
        "predictions": 1,
        "updates": 1,
        "pending_prediction": False,
    }
    with pytest.raises(RuntimeError, match="pending prediction"):
        head.reveal_target(pending, 1)


def test_learning_training_loop_uses_pending_prediction_chronology() -> None:
    source = inspect.getsource(v3._learning_seed)

    assert source.count("predict_for_update(") >= 3
    assert source.count("reveal_target(") >= 3
    assert "lwoh_head.update(" not in source
    assert "latest_head.update(" not in source
    assert "independent_head.update(" not in source


def test_exact_zero_guards_do_not_use_an_epsilon() -> None:
    finite = v3._guarded_ratio(1.0e-300, 1.0e-300)
    positive_over_zero = v3._guarded_ratio(1.0, 0.0)
    zero_over_zero = v3._guarded_ratio(0.0, 0.0)

    assert finite == {
        "numerator": 1.0e-300,
        "denominator": 1.0e-300,
        "status": "finite",
        "value": 1.0,
    }
    assert positive_over_zero == {
        "numerator": 1.0,
        "denominator": 0.0,
        "status": "positive_over_zero",
        "value": None,
    }
    assert zero_over_zero == {
        "numerator": 0.0,
        "denominator": 0.0,
        "status": "zero_over_zero",
        "value": None,
    }
    assert v3._ratio_at_least(positive_over_zero, 1.0e300) is True
    assert v3._ratio_at_least(zero_over_zero, 0.0) is False
    assert v3._ratio_multiple_at_least(
        positive_over_zero, v3._guarded_ratio(1.0, 2.0), 3.0
    ) is True
    assert v3._ratio_multiple_at_least(
        zero_over_zero, v3._guarded_ratio(1.0, 2.0), 0.0
    ) is False


def test_admission_gates_use_median_of_per_seed_guarded_ratios() -> None:
    # Per-seed ratios are [.2, .2, 5, 5, .25], whose median is .25.
    # The incorrect ratio-of-medians is .5/.5 == 1 and would pass A03.
    numerators = (0.1, 0.1, 0.5, 0.5, 0.5)
    denominators = (0.5, 0.5, 0.1, 0.1, 2.0)
    rows = [
        _synthetic_admission_row(
            seed,
            candidate_c=numerator,
            candidate_n=denominator,
            native_c=0.01,
            native_n=1.0,
        )
        for seed, numerator, denominator in zip(
            v3.ADMISSION_SEEDS, numerators, denominators, strict=True
        )
    ]

    decision = v3.recompute_admission_gates(rows)

    assert decision["gates"]["A03"] is False
    assert "A03" in decision["failed_gate_ids"]


def test_admission_output_ratio_obeys_positive_and_zero_over_zero_guards() -> None:
    positive = v3.recompute_admission_gates(
        _synthetic_admission_rows(output_cue=1.0, output_sd=0.0)
    )
    undefined = v3.recompute_admission_gates(
        _synthetic_admission_rows(output_cue=0.0, output_sd=0.0)
    )

    assert positive["gates"]["A07"] is True
    assert undefined["gates"]["A07"] is False
    assert "A07" in undefined["failed_gate_ids"]


def test_synthetic_admission_pass_and_sorted_fail_terminal() -> None:
    passed = v3.recompute_admission_gates(_synthetic_admission_rows())

    assert passed["all_admission_gates_pass"] is True
    assert passed["failed_gate_ids"] == []
    assert passed["terminal"] == "LWOH_ADMITTED:write_once_ttl32"
    assert all(passed["gates"].values())

    failed_rows = _synthetic_admission_rows(invariant_pass=False, tail_pass=False)
    failed = v3.recompute_admission_gates(failed_rows)
    assert failed["all_admission_gates_pass"] is False
    assert failed["failed_gate_ids"] == ["A09", "A10"]
    assert failed["terminal"] == "LWOH_ADMISSION_FAIL:A09,A10"


def test_observer_is_write_once_ttl32_and_native_shadow_equivalent() -> None:
    graph = v3.LWOHObserverGraph.from_seed(9090)
    shadow = build_experiment_000_graph(
        seed=9090,
        mode="full",
        event_log_enabled=False,
        **FROZEN_GRAPH_OPTIONS,
    )
    topology = graph.topology_hash()
    weights = graph.weights_hash()
    sample = v3.VisibleSample(
        sample_id="scratch:9090:native-shadow",
        cue=1,
        noise=(1, -1, 1, -1),
    )

    observation = v3.run_visible_sample(graph, sample, shadow=shadow)

    assert observation.native_equivalent is True
    assert graph.topology_hash() == shadow.topology_hash() == topology
    assert graph.weights_hash() == shadow.weights_hash() == weights
    assert graph.observer_ledger["lwoh_writes"] == 8
    assert graph.observer_ledger["latest_writes"] == graph.observer_ledger["kernel_calls"]
    assert len(set(graph._lwoh_kappa.values())) == 1
    write_tick = next(iter(graph._lwoh_kappa.values()))
    assert graph.observer_features("lwoh", tick=write_tick + 32) == observation.lwoh
    assert graph.observer_features("lwoh", tick=write_tick + 33) == (0.0,) * 8


def test_native_shadow_identity_mismatch_is_detected() -> None:
    graph = v3.LWOHObserverGraph.from_seed(9091)
    shadow = build_experiment_000_graph(
        seed=9090,
        mode="full",
        event_log_enabled=False,
        **FROZEN_GRAPH_OPTIONS,
    )
    sample = v3.VisibleSample(
        sample_id="scratch:9091:tampered-shadow",
        cue=-1,
        noise=(-1, 1, -1, 1),
    )

    with pytest.raises(RuntimeError, match="native execution"):
        v3.run_visible_sample(graph, sample, shadow=shadow)


def test_deterministic_payload_excludes_execution_phase_and_process_identity() -> None:
    scientific = {"seeds": [9090, 9091], "metric": {"value": 0.75}}
    primary = v3._finalize_report(
        {
            "schema_version": "synthetic-report-v1",
            "phase": "primary",
            "output_path": "primary.json",
            "worker_pid": 101,
            "worker_nonce": "101:one",
            "coordinator_pid": 201,
            "coordinator_nonce": "201:one",
            "created_at_unix_ns": 1,
            "runtime_seconds": 10.0,
            "scientific": scientific,
        }
    )
    rerun = v3._finalize_report(
        {
            "schema_version": "synthetic-report-v1",
            "phase": "rerun",
            "output_path": "rerun.json",
            "worker_pid": 102,
            "worker_nonce": "102:two",
            "coordinator_pid": 202,
            "coordinator_nonce": "202:two",
            "created_at_unix_ns": 2,
            "runtime_seconds": 20.0,
            "scientific": copy.deepcopy(scientific),
        }
    )

    assert primary["deterministic_payload_sha256"] == rerun[
        "deterministic_payload_sha256"
    ]
    changed = copy.deepcopy(rerun)
    changed["scientific"]["metric"]["value"] = 0.76
    changed.pop("deterministic_payload_sha256")
    changed = v3._finalize_report(changed)
    assert changed["deterministic_payload_sha256"] != primary[
        "deterministic_payload_sha256"
    ]


def test_official_admission_shape_rejects_scratch_before_worker_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def forbidden_worker(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        raise AssertionError("worker must not start")

    monkeypatch.setattr(v3, "_admission_seed", forbidden_worker)
    with pytest.raises(ValueError, match="official admission"):
        v3.run_admission(
            seeds=(9090,), workers=1, pairs=1, phase="admission"
        )
    assert called is False


def test_activation_and_freeze_fail_before_publication_when_prerequisite_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing_lineage(_root: Path) -> dict[str, Any]:
        raise RuntimeError("synthetic missing lineage")

    monkeypatch.setattr(v3, "_verify_lineage", missing_lineage)
    with pytest.raises(RuntimeError, match="missing lineage"):
        v3.commit_activation(tmp_path)

    assert not (tmp_path / v3.ACTIVATION_PATH).exists()
    assert not (tmp_path / v3.PHASE_LEDGER_PATH).exists()
    with pytest.raises((FileNotFoundError, RuntimeError)):
        v3.commit_pre_metric_freeze(tmp_path)
    assert not (tmp_path / v3.FREEZE_PATH).exists()


def test_phase_receipts_are_ordered_write_once_and_artifact_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v3, "PHASE_LEDGER_PATH", Path("synthetic/PHASE_LEDGER.json"))
    monkeypatch.setattr(v3, "PHASE_RECEIPT_DIR", Path("synthetic/phases"))
    activation_artifact = Path("synthetic/activation-artifact.json")
    freeze_artifact = Path("synthetic/freeze-artifact.json")
    (tmp_path / activation_artifact).parent.mkdir(parents=True)
    (tmp_path / activation_artifact).write_bytes(b"activation\n")
    (tmp_path / freeze_artifact).write_bytes(b"freeze\n")
    v3._write_static_phase_ledger(tmp_path)

    with pytest.raises(RuntimeError, match="missing predecessor"):
        v3._record_phase(tmp_path, "freeze", freeze_artifact)

    first = v3._record_phase(tmp_path, "activation", activation_artifact)
    second = v3._record_phase(tmp_path, "freeze", freeze_artifact)
    assert first["predecessor_receipt_sha256"] is None
    first_path = tmp_path / v3._phase_receipt_path(0, "activation")
    assert second["predecessor_receipt_sha256"] == hashlib.sha256(
        first_path.read_bytes()
    ).hexdigest()
    assert v3._verify_phase_chain(tmp_path, "freeze") is True

    # The protocol requires byte-identical retries, not rejection of retries.
    receipt_path = tmp_path / v3._phase_receipt_path(1, "freeze")
    before = receipt_path.read_bytes()
    assert v3._record_phase(tmp_path, "freeze", freeze_artifact) == second
    assert receipt_path.read_bytes() == before
    receipt_path.with_suffix(".sha256").unlink()
    assert v3._record_phase(tmp_path, "freeze", freeze_artifact) == second
    assert receipt_path.read_bytes() == before
    assert receipt_path.with_suffix(".sha256").is_file()

    (tmp_path / freeze_artifact).write_bytes(b"changed freeze\n")
    with pytest.raises(RuntimeError, match="existing phase receipt differs"):
        v3._record_phase(tmp_path, "freeze", freeze_artifact)
    assert receipt_path.read_bytes() == before
    (tmp_path / freeze_artifact).write_bytes(b"freeze\n")

    (tmp_path / activation_artifact).write_bytes(b"tampered\n")
    with pytest.raises(RuntimeError, match="artifact changed"):
        v3._verify_phase_chain(tmp_path, "freeze")


def test_deep_verifier_rejects_absent_and_partial_chains(tmp_path: Path) -> None:
    validator = getattr(v3, "validate_official_v3_chain", None)
    assert callable(validator), "V3 runner must expose validate_official_v3_chain"

    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        validator(project_root=tmp_path)

    partial = tmp_path / v3.FREEZE_PATH
    partial.parent.mkdir(parents=True)
    v3._write_record(
        partial,
        {
            "schema_version": "experiment-000-lwoh-l1-v3-pre-metric-freeze-v1",
            "status": "FROZEN_PRE_METRIC",
            "source_manifest_sha256": "0" * 64,
        },
        self_field="freeze_payload_sha256",
    )
    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        validator(project_root=tmp_path)


def test_pagefile_parser_reads_sample_before_typeperf_footer() -> None:
    output = (
        '"(PDH-CSV 4.0)","\\\\HOST\\Paging File(_Total)\\% Usage"\n'
        '"09/05/2026 00:39:05.609","9.450093"\n'
        '\nExiting, please wait...\nThe command completed successfully.\n'
    )
    assert v3._parse_pagefile_counter(output) == pytest.approx(9.450093)


@pytest.mark.parametrize("sample", ["nan", "inf", "-1", "101", "bad"])
def test_pagefile_parser_rejects_invalid_percentage(sample: str) -> None:
    output = f'"time","\\Paging File(_Total)\\% Usage"\n"now","{sample}"\n'
    with pytest.raises(OSError, match="percentage"):
        v3._parse_pagefile_counter(output)


@pytest.mark.parametrize("output", [
    '', '"now","9.4"\n',
    '"time","\\Memory\\% Committed Bytes In Use"\n"now","9.4"\n',
    '"time","\\Paging File(_Total)\\% Usage"\n',
    '"time","\\Paging File(_Total)\\% Usage"\n"now","9"\n"later","10"\n',
])
def test_pagefile_parser_rejects_missing_wrong_or_multiple_samples(output: str) -> None:
    with pytest.raises(OSError):
        v3._parse_pagefile_counter(output)


@pytest.mark.parametrize("pairs", [1, 2, 7, 9])
def test_scratch_cli_rejects_unregistered_pairs_before_dispatch(
    pairs: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("scratch dispatch must not happen")

    monkeypatch.setattr(v3, "commit_scratch_smoke", forbidden)
    with pytest.raises(ValueError, match="registered command"):
        v3.main([
            "--phase", "scratch", "--seeds", "9090", "9091", "--workers", "2",
            "--pairs", str(pairs), "--output", v3.SCRATCH_REPORT_PATH.as_posix(),
        ])


@pytest.mark.parametrize("kind", ["admission", "learning"])
def test_official_public_api_rejects_missing_authority_before_pool(
    kind: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("pool creation must not happen")

    monkeypatch.setattr(v3, "ProcessPoolExecutor", forbidden)
    with pytest.raises(RuntimeError, match="execution authority"):
        if kind == "admission":
            v3.run_admission(seeds=v3.ADMISSION_SEEDS, workers=5, pairs=100, phase=kind)
        else:
            v3.run_learning(seeds=v3.LEARNING_SEEDS, workers=5, sample_scale=1.0, phase=kind)


@pytest.mark.parametrize("kind", ["admission", "learning"])
def test_official_wrapper_passes_authority_only_after_prerequisites(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(v3, "_verify_phase_chain", lambda *a: calls.append("phase"))
    monkeypatch.setattr(v3, "_source_freeze", lambda *a: (
        calls.append("freeze") or {"source_manifest_sha256": "a" * 64}
    ))
    monkeypatch.setattr(v3, "_host_resource_preflight", lambda *a: (
        calls.append("host") or {"pass": True}
    ))
    monkeypatch.setenv("ALS_V3_SOURCE_MANIFEST_SHA256", "synthetic")

    def sentinel(**kwargs: Any) -> None:
        assert calls == ["phase", "freeze", "host"]
        assert kwargs["_authorization"] is v3._OFFICIAL_EXECUTION_AUTHORITY
        raise RuntimeError("SYNTHETIC_DISPATCH_VERIFIED_NO_WORKERS")

    monkeypatch.setattr(v3, f"run_{kind}", sentinel)
    runner = getattr(v3, f"_run_official_{kind}")
    output = v3.ADMISSION_PRIMARY_PATH if kind == "admission" else v3.LEARNING_PRIMARY_PATH
    with pytest.raises(RuntimeError, match="SYNTHETIC_DISPATCH_VERIFIED"):
        runner(tmp_path, phase=kind, output=output)


def test_readiness_chain_mapping_matches_global_validator_contract(tmp_path: Path) -> None:
    from adaptive_learning_substrate.artifact_validation import LWOH_V3_STAGE_CONTRACT

    for count, admitted in ((3, False), (5, True)):
        expected = {}
        for name, *_ in LWOH_V3_STAGE_CONTRACT[:count]:
            relative = v3.ARTIFACT_DIR / name
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            expected[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert v3._readiness_chain_artifacts(tmp_path, admission_pass=admitted) == expected


@pytest.mark.parametrize("count", [1, 20, 37])
def test_pytest_receipt_accepts_variable_real_pass_counts(tmp_path: Path, count: int) -> None:
    path = tmp_path / "tests.xml"
    path.write_text(
        f'<testsuites><testsuite tests="{count}" failures="0" errors="0" skipped="0">'
        + '<testcase name="synthetic"/>' * count + '</testsuite></testsuites>'
    )
    assert v3._verify_pytest_junit(path) == count


@pytest.mark.parametrize("body", [
    '<testsuite tests="0"/>',
    '<testsuite tests="1" failures="1"><testcase><failure/></testcase></testsuite>',
    '<testsuite tests="1" errors="1"><testcase><error/></testcase></testsuite>',
    '<testsuite tests="1" skipped="1"><testcase><skipped/></testcase></testsuite>',
    '<testsuite tests="2"><testcase/></testsuite>',
])
def test_pytest_receipt_rejects_empty_failed_skipped_or_wrong_counts(
    tmp_path: Path, body: str,
) -> None:
    path = tmp_path / "tests.xml"
    path.write_text(f'<testsuites>{body}</testsuites>')
    with pytest.raises(RuntimeError):
        v3._verify_pytest_junit(path)


def test_synthetic_latch_and_online_head_learn_delayed_cue_without_native_changes() -> None:
    # A constructed micro-control, not generated benchmark streams or gate evidence.
    graph = v3.LWOHObserverGraph.from_seed(9090)
    shadow = build_experiment_000_graph(
        seed=9090, mode="full", event_log_enabled=False, **FROZEN_GRAPH_OPTIONS,
    )
    topology, weights = graph.topology_hash(), graph.weights_hash()
    head = v3.OnlineHead()
    for index in range(32):
        cue = 1 if index % 2 else -1
        observation = v3.run_visible_sample(
            graph, v3.VisibleSample(f"synthetic:train:{index}", cue, (1, -1) * 4),
            shadow=shadow,
        )
        pending = head.predict_for_update(observation.lwoh)
        head.reveal_target(pending, cue)
    for cue in (-1, 1):
        for noise in ((-1,) * 8, (1,) * 16):
            observation = v3.run_visible_sample(
                graph, v3.VisibleSample(f"synthetic:eval:{cue}:{len(noise)}", cue, noise),
                shadow=shadow,
            )
            assert observation.native_equivalent is True
            _, prediction = head.predict(observation.lwoh)
            assert prediction == cue
    assert graph.topology_hash() == topology
    assert graph.weights_hash() == weights
    assert head.chronology_ledger["updates"] == 32


# Replay regressions use synthetic rows and stub every assigned-seed simulator.
def _replay_identity(pid):
    return {
        "process_id": pid,
        "process_instance_token": f"process-{pid}",
        "process_started_utc": "2026-09-05T00:00:00+00:00",
        "run_started_utc": "2026-09-05T00:01:00+00:00",
        "run_instance_token": f"run-{pid}",
        "start_nonce": f"nonce-{pid}",
        "executable_sha256": "e" * 64,
        "source_manifest_sha256": "f" * 64,
        "blas_state": {"all_loaded_pools_single_threaded": True},
    }


def _install_replay_fixture(monkeypatch, kind):
    seeds = v3.ADMISSION_SEEDS if kind == "admission" else v3.LEARNING_SEEDS
    decision = {
        "terminal": "LWOH_ADMISSION_FAIL" if kind == "admission" else "LWOH_LEARNING_FAIL",
        "failed_gate_ids": ["SYNTHETIC"],
        "all_admission_gates_pass": False,
    }
    reports = []
    for index in range(2):
        base = 100 + index * 100
        rows = [{"seed": seed, "value": 1.0, "worker_pid": base + 1 + i % 5,
                 "process_identity": _replay_identity(base + 1 + i % 5)}
                for i, seed in enumerate(seeds)]
        report = {
            "schema_version": f"experiment-000-lwoh-l1-v3-{kind}-report-v1",
            "protocol_version": v3.PROTOCOL_VERSION,
            "scientific_rng_namespace": v3.SCIENTIFIC_RNG_NAMESPACE,
            "phase": kind + ("-rerun" if index else ""),
            "status": decision["terminal"], "decision": copy.deepcopy(decision),
            "seeds": list(seeds), "workers": 5, "condition_ids": list(v3.CONDITION_IDS),
            "seed_results": rows,
            "nondeterministic_provenance": {"coordinator_identity": _replay_identity(base)},
        }
        report["pairs_per_split" if kind == "admission" else "sample_scale"] = 100 if kind == "admission" else 1.0
        reports.append(v3._finalize_report(report))
    monkeypatch.setattr(v3, "_verify_phase_chain", lambda *args: True)
    monkeypatch.setattr(v3, "_source_freeze", lambda *args: {"source_manifest_sha256": "f" * 64})
    monkeypatch.setattr(v3, "_host_resource_preflight", lambda *args: {"passed": True})
    monkeypatch.setattr(v3, "_process_identity", lambda *args: _replay_identity(999))
    monkeypatch.setattr(v3, "_verify_raw_report", lambda root, path, **kw: reports[int("rerun" in path.name)])
    monkeypatch.setattr(v3, "_file_identity", lambda root, path: {"sha256": "a" * 64})
    monkeypatch.setattr(v3, f"_validate_{kind}_raw_evidence", lambda *args: None)
    monkeypatch.setattr(v3, f"recompute_{kind}_gates", lambda rows: copy.deepcopy(decision))
    monkeypatch.setattr(v3, "_record_phase", lambda *args: None)
    published = []
    monkeypatch.setattr(v3, "_write_record", lambda path, record, **kw: published.append(record) or record)
    return reports, published


@pytest.mark.parametrize("kind", ["admission", "learning"])
def test_identical_fabricated_reports_are_not_replay(tmp_path, monkeypatch, kind):
    _, published = _install_replay_fixture(monkeypatch, kind)
    regenerated = []

    def seed_function(seed, **kwargs):
        regenerated.append(seed)
        return {"seed": seed, "value": 0.0}

    monkeypatch.setattr(v3, f"_{kind}_seed", seed_function)
    verify = v3.verify_admission_runs if kind == "admission" else v3.verify_learning_runs
    with pytest.raises(RuntimeError, match="replay"):
        verify(tmp_path)
    assert regenerated
    assert not published


@pytest.mark.parametrize("kind", ["admission", "learning"])
def test_stage_replay_regenerates_every_seed_and_persists_measured_evidence(tmp_path, monkeypatch, kind):
    reports, published = _install_replay_fixture(monkeypatch, kind)
    before = copy.deepcopy(reports)
    calls = []

    def regenerate(seed, **kwargs):
        calls.append((seed, kwargs))
        return {"seed": seed, "value": 1.0}

    monkeypatch.setattr(v3, f"_{kind}_seed", regenerate)
    verifier = v3.verify_admission_runs if kind == "admission" else v3.verify_learning_runs
    result = verifier(tmp_path)
    seeds = v3.ADMISSION_SEEDS if kind == "admission" else v3.LEARNING_SEEDS
    expected_args = {"pairs": 100} if kind == "admission" else {"sample_scale": 1.0}
    assert calls == [(seed, expected_args) for seed in seeds]
    assert result["replay_verified"] is True
    assert result["replay_evidence"]["verifier_process_identity"] == _replay_identity(999)
    assert [row["seed"] for row in result["replay_evidence"]["seed_rows"]] == list(seeds)
    assert len(published) == 1 and reports == before


@pytest.mark.parametrize("kind", ["admission", "learning"])
@pytest.mark.parametrize("fault", ["missing", "duplicate", "reordered", "boolean", "namespace", "condition", "samples", "status"])
def test_replay_registry_rejects_corruption_before_simulation(tmp_path, monkeypatch, kind, fault):
    reports, _ = _install_replay_fixture(monkeypatch, kind)
    report = reports[0]
    if fault == "missing":
        report["seed_results"].pop()
    elif fault == "duplicate":
        report["seed_results"][-1] = copy.deepcopy(report["seed_results"][0])
    elif fault == "reordered":
        report["seed_results"].reverse()
    elif fault == "boolean":
        report["seed_results"][0]["seed"] = True
    elif fault == "namespace":
        report["scientific_rng_namespace"] = "changed"
    elif fault == "condition":
        report["condition_ids"].pop()
    elif fault == "status":
        report["status"] = "fabricated"
    else:
        report["pairs_per_split" if kind == "admission" else "sample_scale"] = True
    monkeypatch.setattr(v3, f"_{kind}_seed", lambda *a, **kw: pytest.fail("simulator reached"))
    with pytest.raises((RuntimeError, TypeError)):
        v3._verify_fresh_replay(tmp_path, reports, kind=kind, source_sha256="f" * 64)


@pytest.mark.parametrize("kind", ["admission", "learning"])
@pytest.mark.parametrize("fault", ["resource", "source", "raw", "same_verifier", "cross_role", "executable", "row_identity"])
def test_replay_prerequisites_precede_scientific_operations(tmp_path, monkeypatch, kind, fault):
    reports, _ = _install_replay_fixture(monkeypatch, kind)

    def reject(*args, **kwargs):
        raise RuntimeError("synthetic rejected prerequisite")

    if fault == "resource":
        monkeypatch.setattr(v3, "_host_resource_preflight", reject)
    elif fault == "source":
        monkeypatch.setattr(v3, "_source_checkpoint", reject)
    elif fault == "raw":
        monkeypatch.setattr(v3, f"_validate_{kind}_raw_evidence", reject)
    elif fault == "same_verifier":
        monkeypatch.setattr(v3, "_process_identity", lambda *args: _replay_identity(100))
    elif fault == "cross_role":
        reports[1]["nondeterministic_provenance"]["coordinator_identity"] = _replay_identity(101)
    elif fault == "executable":
        reports[0]["seed_results"][0]["process_identity"]["executable_sha256"] = "0" * 64
    else:
        reports[0]["seed_results"][0].pop("process_identity")
    monkeypatch.setattr(v3, f"_{kind}_seed", lambda *a, **kw: pytest.fail("simulator reached"))
    with pytest.raises((RuntimeError, TypeError)):
        v3._verify_fresh_replay(tmp_path, reports, kind=kind, source_sha256="f" * 64)


@pytest.mark.parametrize("fault", ["input", "source", "phase"])
def test_replay_rechecks_prerequisites_after_regeneration(tmp_path, monkeypatch, fault):
    reports, _ = _install_replay_fixture(monkeypatch, "admission")
    ran = []

    def regenerate(seed, **kw):
        ran.append(seed)
        return {"seed": seed, "value": 1.0}

    def checkpoint(*args, **kwargs):
        if len(ran) == 5:
            raise RuntimeError("synthetic changed prerequisite")
        return "f" * 64

    monkeypatch.setattr(v3, "_admission_seed", regenerate)
    if fault == "input":
        monkeypatch.setattr(v3, "_file_identity", lambda *args: {"sha256": str(len(ran) == 5)})
    else:
        monkeypatch.setattr(v3, "_source_checkpoint" if fault == "source" else "_verify_phase_chain", checkpoint)
    with pytest.raises(RuntimeError):
        v3._verify_fresh_replay(tmp_path, reports, kind="admission", source_sha256="f" * 64)
    assert ran == list(v3.ADMISSION_SEEDS)


@pytest.mark.parametrize("fault", ["rows", "verified", "same_process", "source", "terminal"])
def test_replay_attestation_is_bound_to_regenerated_rows(monkeypatch, tmp_path, fault):
    reports, _ = _install_replay_fixture(monkeypatch, "admission")
    realized = {"verified": True, "source_manifest_sha256": "f" * 64,
                "verifier_process_identity": _replay_identity(999), "seed_rows": [{"seed": 90, "deterministic_row_sha256": "a" * 64}]}
    archived = copy.deepcopy(realized)
    archived["verifier_process_identity"] = _replay_identity(888)
    stage = {"replay_verified": True, "replay_evidence": archived,
             "status": "LWOH_ADMISSION_FAIL", "terminal": reports[0]["decision"]["terminal"],
             "failed_gate_ids": reports[0]["decision"]["failed_gate_ids"]}
    v3._validate_replay_attestation(stage, reports, realized)
    if fault == "rows":
        archived["seed_rows"] = []
    elif fault == "verified":
        archived["verified"] = False
    elif fault == "same_process":
        archived["verifier_process_identity"] = _replay_identity(999)
    elif fault == "terminal":
        stage["status"] = "LWOH_ADMISSION_PASS"
    else:
        archived["source_manifest_sha256"] = "0" * 64
    with pytest.raises(RuntimeError):
        v3._validate_replay_attestation(stage, reports, realized)


def test_deep_verifier_calls_real_replay_helper_instead_of_payload_equality():
    import ast

    source = inspect.getsource(v3.validate_official_v3_chain)
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "_verify_fresh_replay"]
    assert {next(keyword.value.value for keyword in call.keywords if keyword.arg == "kind") for call in calls} == {"admission", "learning"}
    assert '"replay_verified": deterministic_equal' not in source


@pytest.mark.parametrize("matches", [True, False])
def test_deep_admission_failure_branch_replays_before_returning(tmp_path, monkeypatch, matches):
    config = v3._load_config(Path(__file__).resolve().parents[1])
    reports, _ = _install_replay_fixture(monkeypatch, "admission")
    replay = {
        "verified": True, "source_manifest_sha256": "f" * 64,
        "verifier_process_identity": _replay_identity(888),
        "seed_rows": [{"seed": seed, "deterministic_row_sha256": v3._sha256_json({"seed": seed, "value": 1.0})}
                      for seed in v3.ADMISSION_SEEDS],
    }
    stage = {
        "status": "LWOH_ADMISSION_FAIL", "terminal": reports[0]["decision"]["terminal"],
        "failed_gate_ids": reports[0]["decision"]["failed_gate_ids"],
        "deterministic_payload_sha256": reports[0]["deterministic_payload_sha256"],
        "replay_verified": True, "replay_evidence": replay,
    }
    readiness = {
        "schema_version": "experiment-000-lwoh-l1-v3-readiness-v1",
        "status": "LWOH_ADMISSION_FAIL", "readiness_before": 21, "readiness_after": 21,
        "partial_score": False, "admission_seeds": list(v3.ADMISSION_SEEDS),
        "learning_seeds": list(v3.LEARNING_SEEDS), "delays": list(v3.DELAYS),
        "baselines": list(v3.BASELINES), "deterministic_admission": True,
        "deterministic_learning": False, "all_admission_gates_pass": False,
        "all_learning_gates_pass": False, "all_robustness_gates_pass": False,
        "all_integrity_gates_pass": False, "lineage_handoff_path": str(v3.LINEAGE_PATH),
        "lineage_handoff_sha256": "a" * 64, "verification_payload_sha256": "a" * 64,
        "chain_artifacts": v3._readiness_chain_artifacts(tmp_path, admission_pass=False),
    }
    monkeypatch.setattr(v3, "_load_config", lambda *args: config)
    monkeypatch.setattr(v3, "_verify_record", lambda path, **kw: readiness if path.name == v3.READINESS_PATH.name else stage)
    regenerated = []

    def regenerate(seed, **kwargs):
        regenerated.append(seed)
        return {"seed": seed, "value": 1.0 if matches else 0.0}

    monkeypatch.setattr(v3, "_admission_seed", regenerate)
    monkeypatch.setattr(v3, "_learning_seed", lambda *a, **kw: pytest.fail("learning after admission failure"))
    if matches:
        result = v3.validate_official_v3_chain(tmp_path)
        assert set(result) == set(config["v3_deep_verifier"]["exact_result_keys"])
        assert result["replay_verified"] is True and result["readiness_after"] == 21
        assert result["raw_learning_primary_valid"] is False
        assert regenerated == list(v3.ADMISSION_SEEDS)
    else:
        with pytest.raises(RuntimeError, match="replay differs"):
            v3.validate_official_v3_chain(tmp_path)
        assert regenerated == [v3.ADMISSION_SEEDS[0]]


@pytest.mark.parametrize("kind", ["admission", "learning"])
def test_real_scratch_rows_reproduce_and_nonmetric_tampering_is_detected(kind):
    with v3.threadpool_limits(limits=1):
        generate = (lambda: v3._admission_seed(9090, pairs=1)) if kind == "admission" else (lambda: v3._learning_seed(9090, sample_scale=1 / 256))
        first, second = generate(), generate()
    validator = v3._validate_admission_raw_evidence if kind == "admission" else v3._validate_learning_raw_evidence
    validator({"seed_results": [first]})
    assert len(v3._compare_replayed_seed(first, second, kind=kind)) == 64
    corrupted = copy.deepcopy(first)
    if kind == "admission":
        corrupted["train"]["paired_noise"][0][0] += 0.001
    else:
        corrupted["training"]["observation_and_chronology_ledger"][0]["chronology_before_target"]["lwoh_head"]["updates"] = 1
    with pytest.raises(RuntimeError, match="replay differs"):
        v3._compare_replayed_seed(corrupted, second, kind=kind)


# Raw phase-boundary recovery regression controls.
BOUNDARY_PHASE_CASES = ("scratch", "admission", "admission-rerun", "learning", "learning-rerun")


def _boundary_fixture_identity(pid):
    return {
        "process_id": pid, "process_instance_token": f"instance-{pid}",
        "process_started_utc": "2026-09-06T00:00:00+00:00",
        "run_started_utc": "2026-09-06T00:01:00+00:00",
        "run_instance_token": f"run-{pid}", "start_nonce": f"nonce-{pid}",
        "executable_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
        "source_manifest_sha256": "f" * 64,
        "blas_state": {"all_loaded_pools_single_threaded": True},
    }


def _boundary_fixture(root, monkeypatch, phase, *, create=True):
    kind = "learning" if phase.startswith("learning") else "admission"
    scratch = phase == "scratch"
    relative = {
        "scratch": v3.SCRATCH_REPORT_PATH,
        "admission": v3.ADMISSION_PRIMARY_PATH,
        "admission-rerun": v3.ADMISSION_RERUN_PATH,
        "learning": v3.LEARNING_PRIMARY_PATH,
        "learning-rerun": v3.LEARNING_RERUN_PATH,
    }[phase]
    ledger_phase = "scratch_smoke" if scratch else kind + ("_rerun" if phase.endswith("rerun") else "_primary")
    predecessor = v3.PHASES[v3.PHASES.index(ledger_phase) - 1]
    seeds = v3.SCRATCH_SEEDS if scratch else v3.ADMISSION_SEEDS if kind == "admission" else v3.LEARNING_SEEDS
    terminal = "SCRATCH_NONSELECTING" if scratch else "LWOH_" + kind.upper() + "_FAIL"
    decision = {"terminal": terminal, "synthetic": True}
    monkeypatch.setattr(v3, "_source_freeze", lambda *a: {"source_manifest_sha256": "f" * 64})
    monkeypatch.setattr(v3, "_validate_admission_raw_evidence", lambda *a: None)
    monkeypatch.setattr(v3, "_validate_learning_raw_evidence", lambda *a: None)
    monkeypatch.setattr(v3, f"recompute_{kind}_gates", lambda *a: copy.deepcopy(decision))

    def forbidden(*args, **kwargs):
        raise AssertionError("BOUNDARY_RETRY_RECOMPUTED_OR_STARTED_HOST_PREFLIGHT")

    monkeypatch.setattr(v3, f"run_{kind}", forbidden)
    monkeypatch.setattr(v3, "_host_resource_preflight", forbidden)
    if create:
        v3._write_static_phase_ledger(root)
        for index, previous in enumerate(v3.PHASES[:v3.PHASES.index(ledger_phase)]):
            artifact = Path(f"synthetic/previous-{index}.json")
            v3._write_record(root / artifact, {"schema_version": "synthetic", "value": index}, self_field="payload_sha256")
            v3._record_phase(root, previous, artifact)
    report = v3._finalize_report({
        "schema_version": f"experiment-000-lwoh-l1-v3-{kind}-report-v1",
        "protocol_version": v3.PROTOCOL_VERSION,
        "scientific_rng_namespace": v3.SCIENTIFIC_RNG_NAMESPACE,
        "phase": phase, "status": terminal, "decision": copy.deepcopy(decision),
        "seeds": list(seeds), "workers": 2 if scratch else 5,
        "condition_ids": list(v3.CONDITION_IDS),
        "pairs_per_split": 8 if scratch else 100, "sample_scale": 1.0,
        "seed_results": [
            {"seed": seed, "all_invariants_pass": True, "worker_pid": 101 + i % 5,
             "process_identity": _boundary_fixture_identity(101 + i % 5)}
            for i, seed in enumerate(seeds)
        ],
        "nondeterministic_provenance": {
            "coordinator_identity": _boundary_fixture_identity(100), "runtime_seconds": 1.25,
            "host_preflight": {
                "pass": True, "ac_power": True,
                "available_memory_bytes": 8 * 1024**3,
                "free_artifact_volume_bytes": 50 * 1024**3,
                "pagefile_used_fraction": 0.1, "cpu_mean_percent": 20.0,
                "cpu_sample_seconds": 60.0,
            },
        },
    })
    if scratch:
        run = lambda: v3.commit_scratch_smoke(root)
    else:
        run = lambda: getattr(v3, f"_run_official_{kind}")(root, phase=phase, output=relative)
    return relative, ledger_phase, predecessor, report, run


@pytest.mark.parametrize("phase", BOUNDARY_PHASE_CASES)
@pytest.mark.parametrize("state", ("orphan-report", "pair", "orphan-receipt", "complete"))
def test_boundary_recovery_never_recomputes(tmp_path, monkeypatch, phase, state):
    relative, ledger, _, report, run = _boundary_fixture(tmp_path, monkeypatch, phase)
    path = tmp_path / relative
    if state == "orphan-report":
        original = v3._atomic_publish

        def fail_sidecar(target, data):
            if target == path.with_suffix(".sha256"):
                raise OSError("INJECTED_SECOND_LEG_FAILURE")
            original(target, data)

        with monkeypatch.context() as fault:
            fault.setattr(v3, "_atomic_publish", fail_sidecar)
            with pytest.raises(OSError, match="SECOND_LEG"):
                v3._write_record(path, report, self_field="report_payload_sha256")
    else:
        v3._write_record(path, report, self_field="report_payload_sha256")
    receipt = tmp_path / v3._phase_receipt_path(v3.PHASES.index(ledger), ledger)
    if state in {"orphan-receipt", "complete"}:
        v3._record_phase(tmp_path, ledger, relative)
        if state == "orphan-receipt":
            receipt.with_suffix(".sha256").unlink()
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    final = run()
    assert final == v3._verify_raw_report(tmp_path, relative, schema=report["schema_version"])
    assert v3._verify_phase_chain(tmp_path, ledger)
    assert receipt.with_suffix(".sha256").exists()
    for p, (data, mtime) in before.items():
        assert p.read_bytes() == data
        assert p.stat().st_mtime_ns == mtime
    assert run() == final


def test_boundary_publication_rejects_different_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(v3, "_source_freeze", lambda *a: {"source_manifest_sha256": "f" * 64})
    relative = Path("synthetic/raw.json")
    report = v3._finalize_report({"schema_version": "synthetic", "value": 1,
                                 "nondeterministic_provenance": {"pid": 1}})
    v3._publish_raw_report(tmp_path, relative, report)
    changed = copy.deepcopy(report)
    changed["nondeterministic_provenance"]["pid"] = 2
    with pytest.raises(FileExistsError, match="overwrite"):
        v3._publish_raw_report(tmp_path, relative, changed)


@pytest.mark.parametrize("fault", (
    "self-hash", "deterministic-hash", "phase", "workers", "source", "executable",
    "worker-pid", "collision", "host-pass", "host-memory", "host-pagefile", "host-cpu",
    "host-duration", "decision", "raw-evidence", "sidecar", "orphan-sidecar", "source-drift",
    "predecessor-sidecar", "prerequisite-race", "report-race",
))
def test_boundary_recovery_rejects_corruption_before_receipt(tmp_path, monkeypatch, fault):
    relative, ledger, predecessor, report, run = _boundary_fixture(tmp_path, monkeypatch, "admission")
    path = tmp_path / relative
    if fault == "phase":
        report["phase"] = "admission-rerun"
    if fault == "workers":
        report["workers"] = 4
    if fault in {"source", "executable"}:
        report["seed_results"][0]["process_identity"][fault + "_sha256" if fault == "executable" else "source_manifest_sha256"] = "0" * 64
    if fault == "worker-pid":
        report["seed_results"][0]["worker_pid"] = 999
    if fault == "collision":
        report["nondeterministic_provenance"]["coordinator_identity"] = copy.deepcopy(report["seed_results"][0]["process_identity"])
    host_changes = {"host-pass": ("pass", False), "host-memory": ("available_memory_bytes", 1),
                    "host-pagefile": ("pagefile_used_fraction", 0.11), "host-cpu": ("cpu_mean_percent", 21),
                    "host-duration": ("cpu_sample_seconds", 1)}
    if fault in host_changes:
        key, value = host_changes[fault]
        report["nondeterministic_provenance"]["host_preflight"][key] = value
    if fault == "decision":
        report["decision"]["synthetic"] = False
    report = v3._finalize_report(report)
    if fault == "deterministic-hash":
        report["deterministic_payload_sha256"] = "0" * 64
    v3._write_record(path, report, self_field="report_payload_sha256")
    if fault == "self-hash":
        data = v3._read_json(path)
        data["report_payload_sha256"] = "0" * 64
        path.write_bytes(v3._canonical_bytes(data))
    elif fault == "sidecar":
        path.with_suffix(".sha256").write_text("{}\n")
    elif fault == "orphan-sidecar":
        path.unlink()
    else:
        path.with_suffix(".sha256").unlink()
    if fault == "raw-evidence":
        def reject_raw(*a):
            raise RuntimeError("invalid raw evidence")
        monkeypatch.setattr(v3, "_validate_admission_raw_evidence", reject_raw)
    if fault == "source-drift":
        def reject_source(*a, **k):
            raise RuntimeError("source drift")
        monkeypatch.setattr(v3, "_source_checkpoint", reject_source)
    if fault == "predecessor-sidecar":
        previous = tmp_path / v3._phase_receipt_path(v3.PHASES.index(predecessor), predecessor)
        previous.with_suffix(".sha256").unlink()
    if fault in {"prerequisite-race", "report-race"}:
        original_validate = v3._validate_boundary_report

        def mutate(*a, **k):
            original_validate(*a, **k)
            if fault == "report-race":
                data = v3._read_json(path)
                data["nondeterministic_provenance"]["runtime_seconds"] = 3
                path.write_bytes(v3._canonical_bytes(data))
            else:
                (tmp_path / "synthetic/previous-0.sha256").write_text("{}\n")

        monkeypatch.setattr(v3, "_validate_boundary_report", mutate)
    with pytest.raises((RuntimeError, FileNotFoundError, FileExistsError)):
        run()
    assert not (tmp_path / v3._phase_receipt_path(v3.PHASES.index(ledger), ledger)).exists()
    if fault not in {"sidecar", "orphan-sidecar", "self-hash"}:
        assert not path.with_suffix(".sha256").exists()


@pytest.mark.parametrize("phase", BOUNDARY_PHASE_CASES)
def test_boundary_absent_report_restarts_from_initial_state(tmp_path, monkeypatch, phase):
    relative, _, _, _, run = _boundary_fixture(tmp_path, monkeypatch, phase)
    with pytest.raises(AssertionError, match="RECOMPUTED_OR_STARTED"):
        run()
    assert not (tmp_path / relative).exists()


@pytest.mark.parametrize("field,value", (
    ("seeds", [9090]), ("workers", True), ("pairs_per_split", 1),
    ("condition_ids", []), ("decision", []), ("nondeterministic_provenance", []),
    ("protocol_version", "foreign"), ("scientific_rng_namespace", "foreign"),
    ("seed_results", []), ("status", "LWOH_ADMISSION_PASS"),
))
def test_boundary_scratch_rejects_unregistered_report(tmp_path, monkeypatch, field, value):
    relative, ledger, _, report, run = _boundary_fixture(tmp_path, monkeypatch, "scratch")
    report[field] = value
    report = v3._finalize_report(report)
    path = tmp_path / relative
    v3._write_record(path, report, self_field="report_payload_sha256")
    path.with_suffix(".sha256").unlink()
    with pytest.raises(RuntimeError):
        run()
    assert not path.with_suffix(".sha256").exists()
    assert not (tmp_path / v3._phase_receipt_path(v3.PHASES.index(ledger), ledger)).exists()


@pytest.mark.parametrize("leg", ("sidecar", "receipt"))
@pytest.mark.parametrize("change", ("source", "prerequisite", "report-sidecar"))
def test_boundary_rechecks_after_publication(tmp_path, monkeypatch, leg, change):
    relative, ledger, _, report, run = _boundary_fixture(tmp_path, monkeypatch, "admission")
    path = tmp_path / relative
    v3._write_record(path, report, self_field="report_payload_sha256")
    path.with_suffix(".sha256").unlink()
    target = path.with_suffix(".sha256") if leg == "sidecar" else (
        tmp_path / v3._phase_receipt_path(v3.PHASES.index(ledger), ledger)
    ).with_suffix(".sha256")
    original = v3._atomic_publish

    def publish_then_change(destination, data):
        original(destination, data)
        if destination != target:
            return
        if change == "source":
            def reject(*a):
                raise RuntimeError("source changed after publication")
            monkeypatch.setattr(v3, "_source_freeze", reject)
        elif change == "prerequisite":
            (tmp_path / "synthetic/previous-0.sha256").write_text("{}\n")
        else:
            path.with_suffix(".sha256").write_text("{}\n")

    monkeypatch.setattr(v3, "_atomic_publish", publish_then_change)
    with pytest.raises(RuntimeError):
        run()
