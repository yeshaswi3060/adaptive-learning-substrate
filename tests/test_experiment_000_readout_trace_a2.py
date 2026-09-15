"""Fast pre-smoke tests for the frozen readout-trace A2 procedure.

No test in this module evaluates an official seed (70--74).  Numerical checks
use analytic micrographs or scratch custom seeds below the reserved
confirmatory partition.
"""

from __future__ import annotations

import copy
import math
from dataclasses import replace
from typing import Any, Self

import numpy as np
import pytest

import adaptive_learning_substrate.experiment_000_readout_trace_a2 as a2
from adaptive_learning_substrate.events import EventLog
from adaptive_learning_substrate.experiment000_data import (
    SeedRole,
    generate_stream_pair,
)
from adaptive_learning_substrate.experiment_000 import (
    FROZEN_GRAPH_OPTIONS,
    forward_episode,
)
from adaptive_learning_substrate.experiment_000_memory_probe import (
    _output_feature_edge_ids,
    extract_query_features,
)
from adaptive_learning_substrate.experiment_000_readout_trace_a2 import (
    FROZEN_CONFIG_SHA256,
    FROZEN_PROTOCOL_SHA256,
    REGISTERED_RETENTIONS,
    REGISTERED_SEEDS,
    _aggregate_conditions,
    _candidate_trace_readout,
    _ordered_seed_results,
    _structural_mask_sha256,
    _validate_arguments,
    condition_name,
    frozen_source_integrity,
    main,
)
from adaptive_learning_substrate.frozen_history import (
    validate_frozen_recurrent_snapshot,
)
from adaptive_learning_substrate.readout_trace_recurrent import (
    ReadoutTraceRecurrentEventGraph,
    clone_with_readout_trace,
)
from adaptive_learning_substrate.recurrent import (
    RecurrentEdge,
    RecurrentEventGraph,
    build_experiment_000_graph,
)


def _micrograph(
    retention: float,
    *,
    hidden_count: int = 2,
    emit_threshold: float = 1e-3,
    input_weight: float = 1.0,
) -> ReadoutTraceRecurrentEventGraph:
    hidden = tuple(f"h{index}" for index in range(hidden_count))
    edges = (
        RecurrentEdge("input:x->h0", "x", "h0", "input"),
        RecurrentEdge("output:h0->output", "h0", "output", "output"),
    )
    return ReadoutTraceRecurrentEventGraph(
        trace_retention=retention,
        input_nodes=("x", "query"),
        hidden_nodes=hidden,
        output_node="output",
        edges=edges,
        initial_weights={"input:x->h0": input_weight, "output:h0->output": 0.7},
        emit_threshold=emit_threshold,
        event_log=EventLog(enabled=False),
    )


def _native_micrograph(
    *, hidden_count: int = 2, emit_threshold: float = 1e-3
) -> RecurrentEventGraph:
    candidate = _micrograph(
        0.5, hidden_count=hidden_count, emit_threshold=emit_threshold
    )
    return RecurrentEventGraph(
        input_nodes=candidate.input_nodes,
        hidden_nodes=candidate.hidden_nodes,
        output_node=candidate.output_node,
        edges=candidate.edges,
        initial_weights=candidate.weights,
        emit_threshold=emit_threshold,
        event_log=EventLog(enabled=False),
    )


def _update_touch_tuple(graph: ReadoutTraceRecurrentEventGraph) -> tuple[int, ...]:
    ledger = graph.trace_ledger
    return tuple(
        ledger[name]
        for name in (
            "local_trace_read_touches",
            "local_trace_decay_touches",
            "local_trace_write_touches",
        )
    )


def _scratch_episode(seed: int, cue: int):
    pair = generate_stream_pair(
        seed,
        episode_count=2,
        noise_events=8,
        required_role=SeedRole.CUSTOM,
    )
    base = pair.train.episodes[0]
    return replace(base, episode_id=f"{base.episode_id}-a2-test-{cue}", cue=cue, target=cue)


def test_frozen_a2_bindings_and_registered_identifiers_are_exact() -> None:
    integrity = frozen_source_integrity()
    assert FROZEN_PROTOCOL_SHA256 == (
        "9623c23efb586dc89351e2c19887cc25886db42f42a600247feb0c38dcb11484"
    )
    assert FROZEN_CONFIG_SHA256 == (
        "45e2f1450d57a62a9a62a8dd6ce8770da8751de4533f8338cd11acd95cbb3327"
    )
    assert all(
        integrity[name]
        for name in (
            "ccf_document_matches",
            "protocol_matches",
            "config_matches",
        )
    )
    assert integrity["recurrent_source_matches"] is False
    assert integrity["credit_method_bundle_matches"] is False
    assert validate_frozen_recurrent_snapshot()["sha256"] == (
        integrity["expected_recurrent_source_sha256"]
    )
    assert REGISTERED_SEEDS == (70, 71, 72, 73, 74)
    assert REGISTERED_RETENTIONS == (0.25, 0.50, 0.75, 0.90, 0.95)
    assert [condition_name(None), *(condition_name(x) for x in REGISTERED_RETENTIONS)] == [
        "native",
        "rho_0_25",
        "rho_0_50",
        "rho_0_75",
        "rho_0_90",
        "rho_0_95",
    ]


@pytest.mark.parametrize("sign", (-1.0, 1.0))
def test_signed_ema_one_tick_and_multitick_gaps(sign: float) -> None:
    rho = 0.5
    graph = _micrograph(rho)
    graph.begin_episode(f"signed-{sign}")

    graph.step({"x": sign * 0.2})
    graph.step({"x": sign * 0.1})
    first = (1.0 - rho) * math.tanh(sign * 0.2)
    assert graph.trace_snapshot["h0"] == pytest.approx((first, 1))

    graph.step({})
    second = rho * first + (1.0 - rho) * math.tanh(sign * 0.1)
    value, tick = graph.trace_snapshot["h0"]
    assert tick == 2
    assert value == pytest.approx(second, rel=1e-14, abs=1e-14)

    graph.step({})  # output-only processing; h0 remains inactive
    graph.step({"x": sign * 0.3})
    graph.step({})
    third = (rho**3) * second + (1.0 - rho) * math.tanh(sign * 0.3)
    value, tick = graph.trace_snapshot["h0"]
    assert tick == 5
    assert value == pytest.approx(third, rel=1e-14, abs=1e-14)
    assert _update_touch_tuple(graph) == (3, 3, 3)


def test_sentinel_first_update_and_exact_64_positive_zero_reset() -> None:
    native = build_experiment_000_graph(
        900, mode="full", event_log_enabled=False, **FROZEN_GRAPH_OPTIONS
    )
    graph = clone_with_readout_trace(native, 0.75, event_log_enabled=False)
    graph.begin_episode("reset-64")
    snapshot = graph.trace_snapshot
    assert len(snapshot) == 64
    assert all(value == 0.0 and tick is None for value, tick in snapshot.values())
    assert all(math.copysign(1.0, value) == 1.0 for value, _ in snapshot.values())
    assert graph.trace_ledger["local_trace_reset_touches"] == 64

    graph.step({"cue": 0.2})
    graph.step({})
    record = graph.trace_activation_records[-1]
    assert record.tick == 1
    assert record.retained_trace == 0.0
    evaluated = len(graph.trace_activation_records)
    assert evaluated > 0
    assert _update_touch_tuple(graph) == (evaluated, evaluated, evaluated)
    assert sum(tick == 1 for _, tick in graph.trace_snapshot.values()) == evaluated
    assert sum(tick is None for _, tick in graph.trace_snapshot.values()) == 64 - evaluated

    graph.begin_episode("reset-again")
    reset_snapshot = graph.trace_snapshot
    assert all(
        value == 0.0
        and math.copysign(1.0, value) == 1.0
        and tick is None
        for value, tick in reset_snapshot.values()
    )
    assert graph.trace_ledger["local_trace_reset_touches"] == 128


def test_processed_nonemitting_activation_still_updates_trace() -> None:
    graph = _micrograph(0.5, emit_threshold=2.0)
    graph.begin_episode("nonemit")
    graph.step({"x": 0.2})
    result = graph.step({})
    assert not any(":h0:" in event_id for event_id in result.emitted_event_ids)
    assert graph.trace_snapshot["h0"][0] == pytest.approx(0.5 * math.tanh(0.2))
    assert graph.trace_activation_records[-1].emitted is False
    assert _update_touch_tuple(graph) == (1, 1, 1)


def test_blank_inactivity_neither_updates_nor_touches_trace() -> None:
    graph = _micrograph(0.5)
    graph.begin_episode("blank")
    graph.step({"x": 0.2})
    graph.step({})
    before_snapshot = graph.trace_snapshot
    before_touches = _update_touch_tuple(graph)
    graph.step({})  # only the previously scheduled output can process
    assert graph.trace_snapshot == before_snapshot
    assert _update_touch_tuple(graph) == before_touches


def test_effective_trace_observation_is_nonmutating_and_separately_counted() -> None:
    graph = _micrograph(0.5)
    graph.begin_episode("observe")
    graph.step({"x": 0.2})
    graph.step({})
    before_snapshot = graph.trace_snapshot
    native_ledger = graph.ledger
    before_trace_ledger = graph.trace_ledger
    observed = graph.effective_trace(5)
    assert observed["h0"] == pytest.approx(
        (0.5**4) * (0.5 * math.tanh(0.2)), rel=1e-14, abs=1e-14
    )
    assert graph.trace_snapshot == before_snapshot
    assert graph.ledger == native_ledger
    after = graph.trace_ledger
    assert (
        after["local_trace_observation_touches"]
        - before_trace_ledger["local_trace_observation_touches"]
        == len(graph.hidden_nodes)
    )
    assert _update_touch_tuple(graph) == (1, 1, 1)


def test_rho_zero_is_step_exact_and_owns_no_trace_state() -> None:
    native = _native_micrograph()
    zero = clone_with_readout_trace(native, 0.0, event_log_enabled=False)
    native.begin_episode("rho-zero")
    zero.begin_episode("rho-zero")
    for inputs in ({"x": -0.2}, {"x": 0.1}, {}, {}):
        assert native.step(inputs) == zero.step(inputs)
        assert native.unit_events == zero.unit_events
        assert native.ledger == zero.ledger
    native_query = native.query()
    zero_query = zero.query()
    assert native_query == zero_query
    assert extract_query_features(native, native_query) == extract_query_features(
        zero, zero_query
    )
    assert native.unit_events == zero.unit_events
    assert native.ledger == zero.ledger
    assert native.audit == zero.audit
    assert zero.trace_snapshot == {}
    assert zero.effective_trace() == {node: 0.0 for node in zero.hidden_nodes}
    assert all(zero.trace_ledger[name] == 0 for name in a2._TRACE_TOUCH_FIELDS)


@pytest.mark.parametrize("retention", REGISTERED_RETENTIONS)
def test_nonzero_trace_preserves_source_hidden_and_output_dynamics(
    retention: float,
) -> None:
    native = _native_micrograph()
    candidate = clone_with_readout_trace(native, retention, event_log_enabled=False)
    native.begin_episode("same")
    candidate.begin_episode("same")
    for inputs in ({"x": 0.2}, {"x": -0.1}, {}, {}, {"query": 1.0}, {}, {}):
        assert native.step(inputs) == candidate.step(inputs)
        assert native.unit_events == candidate.unit_events
        assert native.ledger == candidate.ledger
    assert candidate.trace_ledger["local_trace_write_touches"] > 0


def test_structural_hash_detects_plastic_only_change() -> None:
    graph = _micrograph(0.5)
    changed_edges = (replace(graph.edges[0], plastic=False), *graph.edges[1:])
    changed = ReadoutTraceRecurrentEventGraph(
        trace_retention=0.5,
        input_nodes=graph.input_nodes,
        hidden_nodes=graph.hidden_nodes,
        output_node=graph.output_node,
        edges=changed_edges,
        initial_weights=graph.weights,
        event_log=EventLog(enabled=False),
    )
    assert graph.topology_hash() == changed.topology_hash()
    assert _structural_mask_sha256(graph) != _structural_mask_sha256(changed)


def test_candidate_output_is_exactly_reconstructed_from_ordered_16d_view() -> None:
    native = build_experiment_000_graph(
        900, mode="full", event_log_enabled=False, **FROZEN_GRAPH_OPTIONS
    )
    graph = clone_with_readout_trace(native, 0.75, event_log_enabled=False)
    query = forward_episode(graph, _scratch_episode(900, 1))
    edge_ids = _output_feature_edge_ids(graph)
    snapshot = graph.trace_snapshot
    observation_before = graph.trace_ledger["local_trace_observation_touches"]
    features, trace64, preactivation, output = _candidate_trace_readout(
        graph, query.tick, edge_ids
    )
    expected_effective = {
        node: (
            0.0
            if tick is None
            else (graph.trace_retention ** (query.tick - tick)) * value
        )
        for node, (value, tick) in snapshot.items()
    }
    source_by_edge = {
        edge.edge_id: edge.source
        for edge in graph.edges
        if edge.kind == "output" and edge.destination == graph.output_node
    }
    expected_features = [
        (edge_id, expected_effective[source_by_edge[edge_id]]) for edge_id in edge_ids
    ]
    weighted = math.fsum(graph.weights[edge_id] * value for edge_id, value in features)
    assert tuple(edge_id for edge_id, _ in features) == edge_ids
    assert len(features) == 16
    assert features == expected_features
    assert trace64.shape == (64,)
    np.testing.assert_array_equal(
        trace64,
        np.asarray([expected_effective[node] for node in graph.hidden_nodes]),
    )
    assert preactivation == weighted
    assert output == math.tanh(weighted)
    assert (
        graph.trace_ledger["local_trace_observation_touches"] - observation_before
        == 64
    )
    assert graph.trace_snapshot == snapshot


@pytest.mark.parametrize(
    "bad_seeds",
    (
        (70.0, 71, 72, 73, 74),
        (np.int64(70), 71, 72, 73, 74),
        (True, 71, 72, 73, 74),
    ),
)
def test_seed_validation_rejects_non_exact_builtin_ints(
    bad_seeds: tuple[object, ...],
) -> None:
    with pytest.raises(TypeError, match="exact built-in integers"):
        _validate_arguments(
            seeds=bad_seeds,  # type: ignore[arg-type]
            retentions=REGISTERED_RETENTIONS,
            pairs_per_split=2,
            ridge_alpha=1e-3,
            workers=5,
        )


def test_seed_validation_requires_the_exact_registered_order() -> None:
    for bad_seeds in (
        (71, 70, 72, 73, 74),
        (70, 71, 72, 73),
        (70, 71, 72, 73, 74, 75),
    ):
        with pytest.raises(ValueError, match="exact ordered seed list"):
            _validate_arguments(
                seeds=bad_seeds,
                retentions=REGISTERED_RETENTIONS,
                pairs_per_split=2,
                ridge_alpha=1e-3,
                workers=5,
            )


def test_argument_and_cli_guards_are_explicit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(ValueError, match="five CPU workers"):
        _validate_arguments(
            seeds=REGISTERED_SEEDS,
            retentions=REGISTERED_RETENTIONS,
            pairs_per_split=2,
            ridge_alpha=1e-3,
            workers=1,
        )
    with pytest.raises(TypeError, match="retentions"):
        _validate_arguments(
            seeds=REGISTERED_SEEDS,
            retentions=(True, 0.5, 0.75, 0.9, 0.95),
            pairs_per_split=2,
            ridge_alpha=1e-3,
            workers=5,
        )
    with pytest.raises(SystemExit) as error:
        main([])
    captured = capsys.readouterr()
    assert error.value.code == 2
    for flag in ("--seeds", "--retentions", "--workers", "--output"):
        assert flag in captured.err


def _synthetic_seed_results(*, one_null_ratio_per_condition: bool = False):
    results = []
    for seed_index, seed in enumerate(range(900, 905)):
        rows = []
        for retention in (None, *REGISTERED_RETENTIONS):
            native = retention is None
            ratio = 1.0 if native else 10.0
            multiple = 1.0 if native else 10.0
            if one_null_ratio_per_condition and not native and seed_index == 0:
                ratio = None
                multiple = None
            activity_split = {
                "total_emitted_event_ratio": 1.0,
                "maximum_matched_episode_event_count_ratio": 1.0,
                "forward_edge_touch_ratio": 1.0,
                "maximum_matched_episode_forward_edge_touch_ratio": 1.0,
            }
            rows.append(
                {
                    "condition": condition_name(retention),
                    "target_retention": retention,
                    "eval": {
                        "cue_to_noise_feature_ratio": ratio,
                        "cue_feature_delta_rms": 1.0 if native else 10.0,
                        "noise_feature_rms": 1.0,
                        "output_cue_delta_to_sd": 1.0,
                    },
                    "ridge_readout": {"train_accuracy": 1.0, "eval_accuracy": 1.0},
                    "activity": {
                        "combined": {
                            "mean_emitted_unit_events_per_forward_episode": 1.0,
                            "hidden_emission_density": 0.01,
                        }
                    },
                    "blank_tail_stability": {
                        "all_tail_state_wake_gates_pass": True,
                        "hidden_emission_density": 0.0,
                    },
                    "invariants": {
                        "synthetic": True,
                        "recurrent_activity_equals_native": True,
                    },
                    "paired_vs_native": {
                        "eval_cue_to_noise_feature_ratio_multiple": multiple,
                        "eval_absolute_cue_feature_delta_rms_multiple": multiple,
                        "ridge_eval_accuracy_difference": 0.0,
                        "eval_output_cue_delta_to_sd_difference": 0.0,
                        "mean_event_count_inflation": 1.0,
                        "activity_by_split": {
                            "train": activity_split,
                            "eval": activity_split,
                        },
                    },
                    "seed_gate": {
                        "criteria": {"every_split_activity_guard": True}
                    },
                }
            )
        results.append({"seed": seed, "condition_results": rows})
    return results


def test_smoke_is_mathematically_ineligible_even_when_every_threshold_is_high() -> None:
    aggregates, selection = _aggregate_conditions(
        _synthetic_seed_results(),
        REGISTERED_RETENTIONS,
        pairs_per_split=2,
    )
    assert all(
        row["gate"]["eligible"] is False and row["gate"]["pass"] is False
        for row in aggregates[1:]
    )
    assert selection["passing_conditions"] == []
    assert selection["selected_condition"] is None
    assert selection["any_target_passed"] is False
    assert selection["final_status"] == "NO_SELECTION"


def test_null_ratios_fail_every_all_defined_aggregate_gate() -> None:
    aggregates, selection = _aggregate_conditions(
        _synthetic_seed_results(one_null_ratio_per_condition=True),
        REGISTERED_RETENTIONS,
        pairs_per_split=100,
    )
    for row in aggregates[1:]:
        criteria = row["gate"]["criteria"]
        assert criteria["median_eval_cue_ratio_at_least_0_50"] is False
        assert criteria["median_eval_cue_ratio_at_least_3x_median_native"] is False
        assert (
            criteria["at_least_4_of_5_seeds_cue_ratio_at_least_3x_native"]
            is False
        )
        assert row["gate"]["pass"] is False
    assert selection["passing_conditions"] == []
    assert selection["selected_condition"] is None


def test_five_worker_path_preserves_order_with_an_inline_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}
    context = object()

    def fake_job(job: tuple[int, tuple[float, ...], int, float]) -> dict[str, object]:
        return {"seed": job[0], "job": job}

    class InlineExecutor:
        def __init__(
            self,
            *,
            max_workers: int,
            mp_context: object,
            max_tasks_per_child: int,
        ) -> None:
            seen["max_workers"] = max_workers
            seen["context"] = mp_context
            seen["max_tasks_per_child"] = max_tasks_per_child

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def map(
            self,
            function: Any,
            jobs: Any,
            *,
            chunksize: int,
        ) -> list[dict[str, object]]:
            materialized = tuple(jobs)
            seen["function"] = function
            seen["jobs"] = materialized
            seen["chunksize"] = chunksize
            return [function(job) for job in materialized]

    monkeypatch.setattr(a2, "_run_seed_job", fake_job)
    monkeypatch.setattr(a2, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(
        a2.multiprocessing,
        "get_context",
        lambda method: seen.setdefault("start_method", method) and context,
    )
    seeds = (74, 70, 72)
    retentions = (0.25, 0.95)

    serial = _ordered_seed_results(
        seeds=seeds,
        retentions=retentions,
        pairs_per_split=2,
        ridge_alpha=1e-3,
        workers=1,
    )
    parallel = _ordered_seed_results(
        seeds=seeds,
        retentions=retentions,
        pairs_per_split=2,
        ridge_alpha=1e-3,
        workers=5,
    )

    expected_jobs = tuple((seed, retentions, 2, 1e-3) for seed in seeds)
    assert parallel == serial
    assert a2._sha256_json(parallel) == a2._sha256_json(serial)
    assert [result["seed"] for result in parallel] == list(seeds)
    assert seen == {
        "start_method": "spawn",
        "max_workers": 5,
        "max_tasks_per_child": 1,
        "context": context,
        "function": fake_job,
        "jobs": expected_jobs,
        "chunksize": 1,
    }


def test_actual_scratch_one_worker_and_five_worker_payloads_are_identical() -> None:
    kwargs = {
        "seeds": (900, 901),
        "retentions": (0.25,),
        "pairs_per_split": 2,
        "ridge_alpha": 1e-3,
    }
    serial = _ordered_seed_results(**kwargs, workers=1)
    parallel = _ordered_seed_results(**kwargs, workers=5)
    assert len(
        {
            row["_nondeterministic_worker_backend"]["process_instance_token"]
            for row in parallel
        }
    ) == len(kwargs["seeds"])

    def deterministic(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                key: value
                for key, value in row.items()
                if key != "_nondeterministic_worker_backend"
            }
            for row in rows
        ]

    assert deterministic(serial) == deterministic(parallel)
    assert a2._sha256_json(deterministic(serial)) == a2._sha256_json(
        deterministic(parallel)
    )


def _fake_freeze_info(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "record": {
            "source_manifest": manifest,
            "source_manifest_sha256": manifest["bundle_sha256"],
            "pre_freeze_verification": {"file_sha256": "a" * 64},
        },
        "file_sha256": "b" * 64,
    }


def _install_guard_only_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> tuple[Any, Any, Any, Any]:
    manifest = a2._source_provenance()
    smoke = tmp_path / "smoke_pairs_2.json"
    verification = tmp_path / "SMOKE_VERIFICATION.json"
    primary = tmp_path / "primary.json"
    rerun = tmp_path / "rerun.json"
    monkeypatch.setattr(a2, "DEFAULT_SMOKE_REPORT_PATH", smoke)
    monkeypatch.setattr(a2, "SMOKE_VERIFICATION_PATH", verification)
    monkeypatch.setattr(a2, "PRIMARY_REPORT_PATH", primary)
    monkeypatch.setattr(a2, "RERUN_REPORT_PATH", rerun)
    monkeypatch.setattr(a2, "_source_provenance", lambda: manifest)
    monkeypatch.setattr(
        a2,
        "_validate_freeze_record",
        lambda *args, **kwargs: _fake_freeze_info(manifest),
    )
    monkeypatch.setattr(a2, "_validate_phase_sequence", lambda *_args: {})
    monkeypatch.setattr(
        a2,
        "run_zero_retention_equivalence",
        lambda: pytest.fail("a prerequisite guard exposed official task metrics"),
    )
    monkeypatch.setattr(
        a2,
        "_ordered_seed_results",
        lambda **_kwargs: pytest.fail("a prerequisite guard started official workers"),
    )
    return smoke, verification, primary, rerun


def _install_real_freeze_and_phase_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> dict[str, Any]:
    """Create a fully self-consistent synthetic prerequisite chain, without task data."""

    real_validate_pre_freeze = a2._validate_pre_freeze_verification
    real_validate_freeze = a2._validate_freeze_record
    paths = {
        "pre_freeze": tmp_path / "PRE_FREEZE_VERIFICATION.json",
        "freeze": tmp_path / "FREEZE_RECORD.json",
        "phase": tmp_path / "PHASE_SEQUENCE.json",
        "smoke": tmp_path / "smoke_pairs_2.json",
        "verification": tmp_path / "SMOKE_VERIFICATION.json",
        "primary": tmp_path / "primary.json",
        "rerun": tmp_path / "rerun.json",
        "determinism": tmp_path / "DETERMINISM_VERIFICATION.json",
    }
    monkeypatch.setattr(a2, "PRE_FREEZE_VERIFICATION_PATH", paths["pre_freeze"])
    monkeypatch.setattr(a2, "FREEZE_RECORD_PATH", paths["freeze"])
    monkeypatch.setattr(a2, "PHASE_SEQUENCE_PATH", paths["phase"])
    monkeypatch.setattr(a2, "DEFAULT_SMOKE_REPORT_PATH", paths["smoke"])
    monkeypatch.setattr(a2, "SMOKE_VERIFICATION_PATH", paths["verification"])
    monkeypatch.setattr(a2, "PRIMARY_REPORT_PATH", paths["primary"])
    monkeypatch.setattr(a2, "RERUN_REPORT_PATH", paths["rerun"])
    monkeypatch.setattr(a2, "DETERMINISM_VERIFICATION_PATH", paths["determinism"])
    phase_artifacts = {
        "tests": (paths["pre_freeze"],),
        "freeze": (paths["freeze"],),
        "persisted_smoke_and_sidecar": (
            paths["smoke"],
            paths["smoke"].with_suffix(".sha256"),
        ),
        "machine_smoke_verification": (paths["verification"],),
        "full": (paths["primary"], paths["primary"].with_suffix(".sha256")),
        "rerun": (paths["rerun"], paths["rerun"].with_suffix(".sha256")),
        "determinism_verification": (paths["determinism"],),
    }
    monkeypatch.setattr(a2, "_PHASE_ARTIFACT_PATHS", phase_artifacts)

    manifest = {"files": {}, "bundle_sha256": "c" * 64}
    monkeypatch.setattr(a2, "_source_provenance", lambda: manifest)
    monkeypatch.setattr(
        a2,
        "run_zero_retention_equivalence",
        lambda: pytest.fail("a prerequisite guard exposed official task metrics"),
    )
    monkeypatch.setattr(
        a2,
        "_ordered_seed_results",
        lambda **_kwargs: pytest.fail("a prerequisite guard started official workers"),
    )

    checks = [
        {"name": name, "command": "command", "input": "input", "literal_output": "ok", "exit_status": 0}
        for name in ("focused_analytic_tests", "ruff", "complete_pytest")
    ]
    pre_freeze: dict[str, Any] = {
        "schema_version": "experiment-000-readout-trace-a2-pre-freeze-v1",
        "status": "PASS",
        "all_exit_status_zero": True,
        "tested_source_manifest_sha256": manifest["bundle_sha256"],
        "checks": checks,
    }
    pre_freeze["verification_payload_sha256"] = a2._record_self_hash(
        pre_freeze, "verification_payload_sha256"
    )
    a2._write_json(paths["pre_freeze"], pre_freeze)

    with a2.threadpool_limits(limits=1, user_api="blas"):
        blas_runtime = a2._single_thread_blas_state()
    artifact_paths = {
        "pre_freeze_verification": str(paths["pre_freeze"]).replace("\\", "/"),
        "freeze": str(paths["freeze"]).replace("\\", "/"),
        "phase_sequence": str(paths["phase"]).replace("\\", "/"),
        "smoke": str(paths["smoke"]).replace("\\", "/"),
        "smoke_verification": str(paths["verification"]).replace("\\", "/"),
        "primary": str(paths["primary"]).replace("\\", "/"),
        "rerun": str(paths["rerun"]).replace("\\", "/"),
        "determinism": str(paths["determinism"]).replace("\\", "/"),
    }
    freeze: dict[str, Any] = {
        "schema_version": "experiment-000-readout-trace-a2-freeze-v1",
        "protocol_name": a2.PROTOCOL_NAME,
        "protocol_version": a2.PROTOCOL_VERSION,
        "protocol_sha256": a2.FROZEN_PROTOCOL_SHA256,
        "config_sha256": a2.FROZEN_CONFIG_SHA256,
        "created_utc": "2026-09-02T20:00:00+00:00",
        "created_local": "2026-09-03T01:30:00+05:30",
        "local_timezone": "UTC+05:30",
        "execution_enabled": True,
        "workers": a2.ORDERED_CPU_WORKERS,
        "backend": {"engine": "numpy_cpu", "float_type": "float64", "gpu": False},
        "blas_runtime": blas_runtime,
        "thread_environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "environment": {
            "implementation": a2.platform.python_implementation(),
            "python": a2.platform.python_version(),
            "numpy": a2.np.__version__,
            "threadpoolctl": a2.threadpoolctl.__version__,
            "platform": a2.platform.platform(),
        },
        "source_manifest": manifest,
        "source_manifest_sha256": manifest["bundle_sha256"],
        "pre_freeze_verification": {
            "path": artifact_paths["pre_freeze_verification"],
            "file_sha256": a2._file_sha256(paths["pre_freeze"]),
            "verification_payload_sha256": pre_freeze[
                "verification_payload_sha256"
            ],
        },
        "prior_v1a": {
            "classification": "INVALID_PROCEDURE_NONSELECTING",
            "observed_deterministic_payload_sha256": (
                "ab5d009ea3afdb8dced17919b255da01e7f8c309af9b2e371eb3b2963c5053cb"
            ),
            "observed_report_file_sha256": (
                "33bdf52cfdac5f30f797a5a75b034a8734f60729e42336fc91eff44ebe3b2d5a"
            ),
        },
        "phase_order": list(a2._PHASE_ORDER),
        "artifact_paths": artifact_paths,
    }
    freeze["freeze_payload_sha256"] = a2._record_self_hash(
        freeze, "freeze_payload_sha256"
    )
    a2._write_json(paths["freeze"], freeze)

    def identity(path: Any) -> dict[str, Any]:
        return {
            "path": str(path).replace("\\", "/"),
            "sha256": a2._file_sha256(path),
            "bytes": path.stat().st_size,
        }

    phase: dict[str, Any] = {
        "schema_version": "experiment-000-readout-trace-a2-phases-v1",
        "protocol_version": a2.PROTOCOL_VERSION,
        "phase_order": list(a2._PHASE_ORDER),
        "phases": [
            {"name": "tests", "artifacts": [identity(paths["pre_freeze"])]},
            {"name": "freeze", "artifacts": [identity(paths["freeze"])]},
        ],
    }
    phase["phase_sequence_payload_sha256"] = a2._record_self_hash(
        phase, "phase_sequence_payload_sha256"
    )
    a2._write_json(paths["phase"], phase)

    def validate_pre_freeze(current_manifest: Any) -> dict[str, Any]:
        try:
            return real_validate_pre_freeze(
                current_manifest, path=paths["pre_freeze"]
            )
        except ValueError as exc:
            # pytest's tmp_path is outside the repository; the real validator
            # has already completed every schema/hash check before its final
            # project-relative presentation path conversion.
            if "is not in the subpath" not in str(exc):
                raise
            return {
                "record": pre_freeze,
                "file_sha256": a2._file_sha256(paths["pre_freeze"]),
                "path": str(paths["pre_freeze"]).replace("\\", "/"),
            }

    def validate_freeze(
        *_args: Any, current_manifest: Any = None, **_kwargs: Any
    ) -> dict[str, Any]:
        return real_validate_freeze(
            paths["freeze"], current_manifest=current_manifest
        )

    monkeypatch.setattr(a2, "_validate_pre_freeze_verification", validate_pre_freeze)
    monkeypatch.setattr(a2, "_validate_freeze_record", validate_freeze)
    return {"manifest": manifest, "pre_freeze_record": pre_freeze, "freeze_record": freeze, "phase_record": phase, **paths}


@pytest.mark.parametrize("blocking_artifact", ("freeze", "phase"))
def test_pre_freeze_writer_preserves_phase_bound_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    blocking_artifact: str,
) -> None:
    pre_freeze = tmp_path / "PRE_FREEZE_VERIFICATION.json"
    freeze = tmp_path / "FREEZE_RECORD.json"
    phase = tmp_path / "PHASE_SEQUENCE.json"
    pre_freeze.write_bytes(b"pre-freeze-sentinel")
    blocker = freeze if blocking_artifact == "freeze" else phase
    blocker.write_bytes(f"{blocking_artifact}-sentinel".encode())
    before = {path: path.read_bytes() for path in (pre_freeze, blocker)}
    monkeypatch.setattr(a2, "PRE_FREEZE_VERIFICATION_PATH", pre_freeze)
    monkeypatch.setattr(a2, "FREEZE_RECORD_PATH", freeze)
    monkeypatch.setattr(a2, "PHASE_SEQUENCE_PATH", phase)
    checks = [
        {"name": name, "command": "c", "input": "i", "literal_output": "ok", "exit_status": 0}
        for name in ("focused_analytic_tests", "ruff", "complete_pytest")
    ]
    with pytest.raises(RuntimeError, match="immutable"):
        a2.create_pre_freeze_verification(checks, output_path=pre_freeze)
    assert {path: path.read_bytes() for path in before} == before
    assert (phase if blocking_artifact == "freeze" else freeze).exists() is False


@pytest.mark.parametrize("blocking_artifact", ("freeze", "phase"))
def test_freeze_writer_refuses_before_mutating_existing_phase_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    blocking_artifact: str,
) -> None:
    freeze = tmp_path / "FREEZE_RECORD.json"
    phase = tmp_path / "PHASE_SEQUENCE.json"
    blocker = freeze if blocking_artifact == "freeze" else phase
    blocker.write_bytes(f"{blocking_artifact}-sentinel".encode())
    before = blocker.read_bytes()
    monkeypatch.setattr(a2, "FREEZE_RECORD_PATH", freeze)
    monkeypatch.setattr(a2, "PHASE_SEQUENCE_PATH", phase)
    with pytest.raises(RuntimeError, match="immutable"):
        a2.create_freeze_record(output_path=freeze)
    assert blocker.read_bytes() == before
    if blocking_artifact == "phase":
        assert not freeze.exists()


@pytest.mark.parametrize(
    "failure_case",
    ("both_absent", "report_only", "sidecar_only", "sidecar_hash", "schema"),
)
def test_full_refuses_absent_or_tampered_smoke_before_any_registered_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    failure_case: str,
) -> None:
    smoke, verification, primary, _ = _install_guard_only_prerequisites(
        monkeypatch, tmp_path
    )
    if failure_case in {"sidecar_hash", "schema"}:
        a2._write_report_and_sidecar(smoke, {})
    elif failure_case == "report_only":
        smoke.write_text("{}\n", encoding="utf-8")
    elif failure_case == "sidecar_only":
        smoke.with_suffix(".sha256").write_text("{}\n", encoding="utf-8")
    if failure_case == "sidecar_hash":
        smoke.with_suffix(".sha256").write_text(
            '{"algorithm":"sha256","report_file":"smoke_pairs_2.json",'
            '"report_sha256":"wrong"}\n',
            encoding="utf-8",
        )
    with pytest.raises(RuntimeError):
        a2.run_readout_trace_experiment(
            pairs_per_split=100,
            workers=5,
            output_path=primary,
            smoke_report_path=smoke,
            smoke_verification_path=verification,
        )
    assert not primary.exists()


@pytest.mark.parametrize(
    "failure_case",
    ("both_absent", "report_only", "sidecar_only", "sidecar_hash", "schema"),
)
def test_rerun_refuses_absent_or_tampered_primary_before_any_registered_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    failure_case: str,
) -> None:
    smoke, verification, primary, rerun = _install_guard_only_prerequisites(
        monkeypatch, tmp_path
    )
    # Satisfy every earlier smoke prerequisite so this test reaches the
    # primary-report guard rather than passing on an unrelated missing-smoke
    # failure.
    a2._write_report_and_sidecar(smoke, {})
    verification.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        a2,
        "_validate_smoke_verification",
        lambda **_kwargs: {"verification_payload_sha256": "c" * 64},
    )
    if failure_case in {"sidecar_hash", "schema"}:
        a2._write_report_and_sidecar(primary, {})
    elif failure_case == "report_only":
        primary.write_text("{}\n", encoding="utf-8")
    elif failure_case == "sidecar_only":
        primary.with_suffix(".sha256").write_text("{}\n", encoding="utf-8")
    if failure_case == "sidecar_hash":
        primary.with_suffix(".sha256").write_text(
            '{"algorithm":"sha256","report_file":"primary.json",'
            '"report_sha256":"wrong"}\n',
            encoding="utf-8",
        )
    with pytest.raises(RuntimeError):
        a2.run_readout_trace_experiment(
            pairs_per_split=100,
            workers=5,
            output_path=rerun,
            smoke_report_path=smoke,
            smoke_verification_path=verification,
        )
    assert not rerun.exists()


@pytest.mark.parametrize(
    "failure_case",
    (
        "absent",
        "schema",
        "status",
        "all_checks",
        "self_hash",
        "smoke_report",
        "smoke_sidecar",
        "smoke_payload",
        "freeze",
        "source",
    ),
)
def test_full_refuses_missing_or_tampered_machine_smoke_verification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    failure_case: str,
) -> None:
    smoke, verification, primary, _ = _install_guard_only_prerequisites(
        monkeypatch, tmp_path
    )
    a2._write_report_and_sidecar(smoke, {})
    validated_smoke = {
        "report": {"deterministic_payload_sha256": "d" * 64},
        "report_file_sha256": "e" * 64,
        "sidecar_file_sha256": "f" * 64,
    }
    monkeypatch.setattr(a2, "_validate_smoke_report", lambda *_args, **_kwargs: validated_smoke)
    if failure_case != "absent":
        record = {
            "schema_version": "experiment-000-readout-trace-a2-smoke-verification-v1",
            "status": "PASS",
            "all_checks_pass": True,
            "smoke_report_sha256": validated_smoke["report_file_sha256"],
            "smoke_sidecar_sha256": validated_smoke["sidecar_file_sha256"],
            "smoke_deterministic_payload_sha256": validated_smoke["report"][
                "deterministic_payload_sha256"
            ],
            "freeze_record_sha256": "b" * 64,
            "source_manifest_sha256": a2._source_provenance()["bundle_sha256"],
        }
        if failure_case == "self_hash":
            record["verification_payload_sha256"] = "0" * 64
        else:
            mutation = {
                "schema": ("schema_version", "tampered"),
                "status": ("status", "FAIL"),
                "all_checks": ("all_checks_pass", False),
                "smoke_report": ("smoke_report_sha256", "0" * 64),
                "smoke_sidecar": ("smoke_sidecar_sha256", "0" * 64),
                "smoke_payload": ("smoke_deterministic_payload_sha256", "0" * 64),
                "freeze": ("freeze_record_sha256", "0" * 64),
                "source": ("source_manifest_sha256", "0" * 64),
            }[failure_case]
            record[mutation[0]] = mutation[1]
            record["verification_payload_sha256"] = a2._record_self_hash(
                record, "verification_payload_sha256"
            )
        a2._write_json(verification, record)
    with pytest.raises(RuntimeError):
        a2.run_readout_trace_experiment(
            pairs_per_split=100,
            workers=5,
            output_path=primary,
            smoke_report_path=smoke,
            smoke_verification_path=verification,
        )
    assert not primary.exists()


@pytest.mark.parametrize("failure_site", ("source", "freeze", "phase"))
def test_full_refuses_earlier_prerequisite_failure_before_registered_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    failure_site: str,
) -> None:
    smoke, verification, primary, _ = _install_guard_only_prerequisites(
        monkeypatch, tmp_path
    )
    if failure_site == "source":
        monkeypatch.setattr(
            a2,
            "frozen_source_integrity",
            lambda: {
                "recurrent_source_matches": True,
                "credit_method_bundle_matches": True,
                "ccf_document_matches": True,
                "protocol_matches": True,
                "config_matches": False,
            },
        )
    elif failure_site == "freeze":
        def reject_freeze(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("freeze prerequisite invalid")

        monkeypatch.setattr(a2, "_validate_freeze_record", reject_freeze)
    else:
        def reject_phase(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("phase prerequisite invalid")

        monkeypatch.setattr(a2, "_validate_phase_sequence", reject_phase)
    with pytest.raises(RuntimeError):
        a2.run_readout_trace_experiment(
            pairs_per_split=100,
            workers=5,
            output_path=primary,
            smoke_report_path=smoke,
            smoke_verification_path=verification,
        )
    assert not primary.exists()


@pytest.mark.parametrize(
    "failure_case",
    (
        "freeze_absent",
        "freeze_self_hash",
        "freeze_source_binding",
        "pre_freeze_absent",
        "pre_freeze_self_hash",
        "pre_freeze_source_binding",
        "phase_absent",
        "phase_self_hash",
        "phase_artifact_hash",
    ),
)
def test_full_refuses_real_freeze_pre_freeze_and_phase_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    failure_case: str,
) -> None:
    chain = _install_real_freeze_and_phase_prerequisites(monkeypatch, tmp_path)
    if failure_case == "freeze_absent":
        chain["freeze"].unlink()
    elif failure_case == "freeze_self_hash":
        bad = copy.deepcopy(chain["freeze_record"])
        bad["freeze_payload_sha256"] = "0" * 64
        a2._write_json(chain["freeze"], bad)
    elif failure_case == "freeze_source_binding":
        bad = copy.deepcopy(chain["freeze_record"])
        bad["source_manifest_sha256"] = "0" * 64
        bad["freeze_payload_sha256"] = a2._record_self_hash(
            bad, "freeze_payload_sha256"
        )
        a2._write_json(chain["freeze"], bad)
    elif failure_case == "pre_freeze_absent":
        chain["pre_freeze"].unlink()
    elif failure_case in {"pre_freeze_self_hash", "pre_freeze_source_binding"}:
        bad = copy.deepcopy(chain["pre_freeze_record"])
        if failure_case == "pre_freeze_self_hash":
            bad["verification_payload_sha256"] = "0" * 64
        else:
            bad["tested_source_manifest_sha256"] = "0" * 64
            bad["verification_payload_sha256"] = a2._record_self_hash(
                bad, "verification_payload_sha256"
            )
        a2._write_json(chain["pre_freeze"], bad)
    elif failure_case == "phase_absent":
        chain["phase"].unlink()
    else:
        bad = copy.deepcopy(chain["phase_record"])
        if failure_case == "phase_self_hash":
            bad["phase_sequence_payload_sha256"] = "0" * 64
        else:
            bad["phases"][0]["artifacts"][0]["sha256"] = "0" * 64
            bad["phase_sequence_payload_sha256"] = a2._record_self_hash(
                bad, "phase_sequence_payload_sha256"
            )
        a2._write_json(chain["phase"], bad)
    with pytest.raises(RuntimeError):
        a2.run_readout_trace_experiment(
            pairs_per_split=100,
            workers=5,
            output_path=chain["primary"],
            smoke_report_path=chain["smoke"],
            smoke_verification_path=chain["verification"],
        )
    assert not chain["primary"].exists()


def test_determinism_verifier_rejects_schema_tamper_before_registered_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _, _, primary, rerun = _install_guard_only_prerequisites(monkeypatch, tmp_path)
    output = tmp_path / "DETERMINISM_VERIFICATION.json"
    monkeypatch.setattr(a2, "DETERMINISM_VERIFICATION_PATH", output)
    a2._write_report_and_sidecar(primary, {"schema_version": "tampered"})
    a2._write_report_and_sidecar(rerun, {"schema_version": "tampered"})
    result = a2.verify_deterministic_full_runs(primary, rerun, output_path=output)
    assert result["status"] == "INVALID_PROCEDURE_NONSELECTING"
    assert result["procedure_valid"] is False
    assert result["valid_terminal_status"] is False
    assert result["failure_type"] == "RuntimeError"
    assert "frozen metadata" in result["failure_message"]
    assert output.is_file()


@pytest.mark.parametrize(
    "malformed_failure",
    (
        KeyError("missing raw field"),
        IndexError("malformed row index"),
        AttributeError("malformed nested object"),
        ArithmeticError("malformed numeric evidence"),
    ),
)
def test_determinism_verifier_normalizes_deep_malformed_report_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    malformed_failure: Exception,
) -> None:
    primary = tmp_path / "primary.json"
    rerun = tmp_path / "rerun.json"
    output = tmp_path / "DETERMINISM_VERIFICATION.json"
    monkeypatch.setattr(a2, "PRIMARY_REPORT_PATH", primary)
    monkeypatch.setattr(a2, "RERUN_REPORT_PATH", rerun)
    monkeypatch.setattr(a2, "DETERMINISM_VERIFICATION_PATH", output)

    def reject_malformed(*_args: Any, **_kwargs: Any) -> None:
        raise malformed_failure

    monkeypatch.setattr(
        a2, "_verify_deterministic_full_runs_validated", reject_malformed
    )
    result = a2.verify_deterministic_full_runs(primary, rerun, output_path=output)
    assert result["status"] == "INVALID_PROCEDURE_NONSELECTING"
    assert result["procedure_valid"] is False
    assert result["valid_terminal_status"] is False
    assert result["failure_type"] == type(malformed_failure).__name__
    assert output.is_file()


def test_scratch_smoke_report_is_generated_and_hardened_validator_accepts_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    scratch_seeds = (900,)
    scratch_retentions = (0.25,)
    output = tmp_path / "scratch_smoke.json"
    manifest = {"files": {}, "bundle_sha256": "c" * 64}
    source_integrity = {
        "recurrent_source_matches": True,
        "credit_method_bundle_matches": True,
        "ccf_document_matches": True,
        "protocol_matches": True,
        "config_matches": True,
    }
    zero_equivalence = {"pass": True, "scratch_only": True}
    monkeypatch.setattr(a2, "REGISTERED_SEEDS", scratch_seeds)
    monkeypatch.setattr(a2, "REGISTERED_RETENTIONS", scratch_retentions)
    monkeypatch.setattr(a2, "ORDERED_CPU_WORKERS", 1)
    monkeypatch.setattr(
        a2,
        "_validate_arguments",
        lambda **_kwargs: (scratch_seeds, scratch_retentions),
    )
    monkeypatch.setattr(a2, "DEFAULT_SMOKE_REPORT_PATH", output)
    monkeypatch.setattr(a2, "frozen_source_integrity", lambda: source_integrity)
    monkeypatch.setattr(a2, "_source_provenance", lambda: manifest)
    monkeypatch.setattr(
        a2,
        "_validate_freeze_record",
        lambda *args, **kwargs: _fake_freeze_info(manifest),
    )
    monkeypatch.setattr(a2, "_validate_phase_sequence", lambda *_args: {})
    monkeypatch.setattr(a2, "_record_phase", lambda *_args: {})
    monkeypatch.setattr(
        a2, "run_zero_retention_equivalence", lambda: zero_equivalence
    )

    def ordered_serially(**kwargs: Any) -> list[dict[str, Any]]:
        return [
            a2._run_seed_job(
                (
                    seed,
                    tuple(kwargs["retentions"]),
                    kwargs["pairs_per_split"],
                    kwargs["ridge_alpha"],
                )
            )
            for seed in kwargs["seeds"]
        ]

    monkeypatch.setattr(a2, "_ordered_seed_results", ordered_serially)
    report = a2.run_readout_trace_experiment(
        seeds=scratch_seeds,
        retentions=scratch_retentions,
        pairs_per_split=2,
        ridge_alpha=1e-3,
        workers=1,
        output_path=output,
    )
    assert report["status"] == "NO_SELECTION"
    assert report["selection"]["selected_condition"] is None
    assert output.is_file()
    assert output.with_suffix(".sha256").is_file()
    freeze_info = _fake_freeze_info(manifest)
    a2._validate_report_mapping(
        report,
        expected_kind="smoke",
        freeze_info=freeze_info,
        current_manifest=manifest,
    )

    bad_self_hash = copy.deepcopy(report)
    bad_self_hash["deterministic_payload_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="deterministic self-hash is invalid"):
        a2._validate_report_mapping(
            bad_self_hash,
            expected_kind="smoke",
            freeze_info=freeze_info,
            current_manifest=manifest,
        )

    # A selecting feature must be an exact output-edge source projection of the
    # same persisted 64-D trace snapshot.  Re-self-hash a trace-only mutation to
    # prove the hardened report validator, rather than only the outer hash,
    # rejects a self-consistent cross-view forgery.
    tampered = copy.deepcopy(report)
    candidate = tampered["seed_results"][0]["condition_results"][1]
    row = candidate["pair_rows"]["train"][0]
    expected_graph = build_experiment_000_graph(
        scratch_seeds[0], mode="full", event_log_enabled=False, **FROZEN_GRAPH_OPTIONS
    )
    first_edge_id = candidate["graph"]["feature_edge_ids"][0]
    first_source = next(
        edge.source for edge in expected_graph.edges if edge.edge_id == first_edge_id
    )
    trace_index = expected_graph.hidden_nodes.index(first_source)
    trace64 = list(row["cue_0_trace64_vector"])
    original = float(trace64[trace_index])
    trace64[trace_index] = float(np.nextafter(original, 0.0 if original else 1.0))
    row["cue_0_trace64_vector"] = trace64
    row["cue_0_trace64"] = list(trace64)
    trace_hash = a2._array_bundle_sha256(np.asarray(trace64, dtype=np.float64))
    row["cue_0_trace64_vector_sha256"] = trace_hash
    row["cue_0_trace64_sha256"] = trace_hash
    tampered["deterministic_payload_sha256"] = a2._recompute_report_payload_hash(
        tampered
    )
    with pytest.raises(
        RuntimeError,
        match="selecting 16-D features are not the output-edge source view",
    ):
        a2._validate_report_mapping(
            tampered,
            expected_kind="smoke",
            freeze_info=freeze_info,
            current_manifest=manifest,
        )

    valid_dynamics = report["seed_results"][0]["condition_results"][0][
        "pair_rows"
    ]["train"][0]["cue_0_recurrent_dynamics"]
    a2._validate_recurrent_dynamics_payload(valid_dynamics, expected_graph)

    wrong_schema = copy.deepcopy(valid_dynamics)
    wrong_schema["unexpected"] = None
    with pytest.raises(RuntimeError, match="payload schema"):
        a2._validate_recurrent_dynamics_payload(wrong_schema, expected_graph)

    wrong_event_index = copy.deepcopy(valid_dynamics)
    wrong_event_index["events"][0]["event_index"] = 1
    with pytest.raises(RuntimeError, match="event_index"):
        a2._validate_recurrent_dynamics_payload(wrong_event_index, expected_graph)

    wrong_parent = copy.deepcopy(valid_dynamics)
    traced_event = next(
        event for event in wrong_parent["events"] if event["edge_traces"]
    )
    traced_event["edge_traces"][0]["parent_event_index"] = traced_event[
        "event_index"
    ]
    with pytest.raises(RuntimeError, match="parent is not ancestral"):
        a2._validate_recurrent_dynamics_payload(wrong_parent, expected_graph)

    wrong_ledger = copy.deepcopy(valid_dynamics)
    wrong_ledger["ledger_delta"]["emitted_unit_events"] += 1
    with pytest.raises(RuntimeError, match="ledger_delta is inconsistent"):
        a2._validate_recurrent_dynamics_payload(wrong_ledger, expected_graph)

    # This is the original coherent-forgery proof: change the query prediction
    # to a string in every native/candidate transcript, then rebuild every local
    # dynamics hash, pair bundle, native comparison, and the report self-hash.
    forged = copy.deepcopy(report)
    for seed_result in forged["seed_results"]:
        conditions = seed_result["condition_results"]
        for condition in conditions:
            for split in ("train", "eval"):
                for row in condition["pair_rows"][split]:
                    for cue in (0, 1):
                        dynamics = row[f"cue_{cue}_recurrent_dynamics"]
                        dynamics["query_prediction"] = "FORGED_NON_PREDICTION"
                        row[f"cue_{cue}_recurrent_dynamics_sha256"] = a2._sha256_json(
                            dynamics
                        )
            condition["pair_rows"]["bundle_sha256"] = a2._sha256_json(
                [condition["pair_rows"]["train"], condition["pair_rows"]["eval"]]
            )
        native = conditions[0]
        for candidate_condition in conditions[1:]:
            a2._attach_native_comparison(
                candidate_condition, native, gate_eligible=False
            )
    forged["deterministic_payload_sha256"] = a2._recompute_report_payload_hash(
        forged
    )
    with pytest.raises(RuntimeError, match="query_prediction must be an integer"):
        a2._validate_report_mapping(
            forged,
            expected_kind="smoke",
            freeze_info=freeze_info,
            current_manifest=manifest,
        )
