"""Prospective lightweight CUDA development probe/admission; never official V3."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import experiment_000_lwoh_l1_execution_v3 as ref
from .cuda_light import Driver, LightForward, LightHead, default_compiler
from .gpu_lwoh import Topology
from .gpu_probe import _canonical, _close, _cpu_trace, _expect_error, _hash, host_memory

SEEDS = tuple(range(13000, 13005))
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "docs/GPU_LIGHT_V1.md"


def sources() -> dict[str, str]:
    paths = sorted((ROOT / "src/adaptive_learning_substrate").rglob("*.py"))
    paths += [ROOT / PROTOCOL, ROOT / "pyproject.toml"]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def write(path: Path, data: Any) -> str:
    content = _canonical(data) + b"\n"
    with path.open("xb") as stream:
        stream.write(content)
    digest = hashlib.sha256(content).hexdigest()
    with path.with_suffix(path.suffix + ".sha256").open("x") as stream:
        stream.write(_canonical({"algorithm": "sha256", "report_file": path.name,
                                 "report_sha256": digest}).decode() + "\n")
    return digest


def read(path: Path) -> Any:
    raw = path.read_bytes()
    binding = json.loads(path.with_suffix(path.suffix + ".sha256").read_text())
    expected = {"algorithm": "sha256", "report_file": path.name,
                "report_sha256": hashlib.sha256(raw).hexdigest()}
    if binding != expected:
        raise ValueError(f"sidecar mismatch: {path}")
    return json.loads(raw)


def headroom() -> None:
    if host_memory()["available_system_ram_bytes"] < 128 * 1024**2:
        raise MemoryError("available RAM below 128 MiB at batch boundary")


def compare_batch(gpu: LightForward, graph: Any, cues: list[Any], noise: list[list[int]]) -> dict[str, Any]:
    headroom()
    actual = gpu.run(cues, noise, trace=True)
    maximum = 0.0
    expected_rows = []
    for index, (cue, sequence) in enumerate(zip(cues, noise, strict=True)):
        expected = _cpu_trace(graph, cue, sequence)
        expected_rows.append(expected)
        for tick, frame in enumerate(expected["frames"]):
            maximum = max(maximum, _close(actual["frames"][index, tick, :65], frame["activations"], "tick activation"))
            if actual["frames"][index, tick, 65:].astype(bool).tolist() != frame["emitted"]:
                raise AssertionError("tick emission mask mismatch")
        for field in ("output", "lwoh", "latest"):
            maximum = max(maximum, _close(actual[field][index], expected[field], field))
        for field in ("events", "touches", "evaluations", "observer_calls", "writes"):
            if actual[field][index] != expected[field]:
                raise AssertionError(f"count mismatch: {field}")
        if (actual["output"][index] >= 0) != (expected["output"] >= 0):
            raise AssertionError("native categorical prediction mismatch")
        single = gpu.run([cue], [sequence])
        for field in ("output", "lwoh", "latest", "events", "touches"):
            _close(single[field][0], actual[field][index], "single/batch")
    return {"cues": cues, "noise": noise, "max_error": maximum,
            "gpu": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in actual.items()},
            "cpu": expected_rows}


def check_tail(gpu: LightForward, seed: int) -> dict[str, Any]:
    headroom()
    actual = gpu.run([-1, 1], [[1]*256]*2, trace=True, tail=True)
    expected = ref._tail_assay(seed)
    graph = ref.LWOHObserverGraph.from_seed(seed)
    for index, cue in enumerate((-1, 1)):
        graph.begin_episode("light-tail")
        for tick in range(260):
            graph._advance({"cue": float(cue)} if tick == 0 else {"query": 1.0} if tick == 257 else {}, force_output=tick == 259)
            _close(actual["frames"][index, tick, :65], [graph._activations[node] for node in graph.hidden_nodes + (graph.output_node,)], "tail activation")
            masks = [any(event.node == node and event.step == tick for event in graph.unit_events) for node in graph.nodes]
            if actual["frames"][index, tick, 65:].astype(bool).tolist() != masks:
                raise AssertionError("tail emission mask mismatch")
        row = expected["rows"][index]
        _close(actual["lwoh"][index], row["wake_feature"], "tail wake feature")
        if actual["tail_emissions"][index] != row["final_window_hidden_emissions"] or actual["expired_abs_max"][index] != row["effective_state_abs_max"]:
            raise AssertionError("tail silent/expiry mismatch")
    weights = np.asarray(graph.cue_edge_weights)
    outputs = np.tanh(actual["lwoh"] @ weights)
    feature_half = float(np.linalg.norm(actual["lwoh"][1] - actual["lwoh"][0]) / 2)
    output_half = float(abs(outputs[1] - outputs[0]) / 2)
    passed = bool(np.all(actual["tail_emissions"] == 0) and np.all(actual["expired_abs_max"] == 0)
                  and feature_half <= .001 and output_half <= .001)
    if passed != expected["pass"]:
        raise AssertionError("tail decision mismatch")
    return {"pass": passed, "gpu_hidden_emissions": actual["tail_emissions"].tolist(),
            "gpu_expired_abs_max": actual["expired_abs_max"].tolist(), "gpu_wake_features": actual["lwoh"].tolist(),
            "gpu_wake_outputs": outputs.tolist(), "wake_feature_half": feature_half, "wake_output_half": output_half,
            "cpu": expected, "cpu_gpu_equivalent": True}


def probe(driver: Driver, output: Path, manifest: dict[str, str]) -> dict[str, Any]:
    commitments, errors = {}, []
    for seed in (12000, 12001):
        graph = ref.LWOHObserverGraph.from_seed(seed)
        original = graph.topology_hash(), graph.weights_hash()
        gpu = LightForward(driver, Topology.from_reference(graph))
        for delay in (0, 4, 8, 16, 32, 40):
            cues = [-1, 1, None, -1, 1, None]
            noise = [[1 if (i+row) % 3 else -1 for i in range(delay)] for row in range(6)]
            data = compare_batch(gpu, graph, cues, noise)
            name = f"forward_{seed}_{delay}.json"
            commitments[name] = write(output / name, data)
            errors.append(data["max_error"])
        head, cpu = LightHead(driver), ref.OnlineHead()
        training = []
        try:
            for index in range(32):
                cue = -1 if index % 2 == 0 else 1
                noise = [1 if (i+index) % 3 else -1 for i in range(8)]
                feature = gpu.run([cue], [noise])["lwoh"][0]
                expected = _cpu_trace(graph, cue, noise)
                pending = cpu.predict_for_update(expected["lwoh"])
                token, score, prediction = head.predict_for_update(feature)
                _close(score, pending.score, "head score")
                if prediction != pending.prediction:
                    raise AssertionError("head prediction mismatch")
                if index == 0:
                    _expect_error(head.predict_for_update, RuntimeError, feature)
                    _expect_error(head.reveal_target, RuntimeError, token+1, cue)
                    _expect_error(head.reveal_target, ValueError, token, 0)
                delta = head.reveal_target(token, cue)
                _close(delta, cpu.reveal_target(pending, cue), "head delta")
                _close(head.theta, cpu.theta, "head theta")
                _expect_error(head.reveal_target, RuntimeError, token, cue)
                training.append({"cue": cue, "noise": noise, "feature": feature.tolist(), "score": score,
                                 "prediction": prediction, "delta": delta.tolist(), "theta": head.theta.tolist()})
            name = f"training_{seed}.json"
            commitments[name] = write(output / name, training)
        finally:
            head.close()
        name = f"tail_{seed}.json"
        commitments[name] = write(output / name, check_tail(gpu, seed))
        gpu.run([1]*32, [[1]*256]*32)
        baseline = sum(driver.allocations.values())
        for _ in range(128):
            headroom()
            gpu.run([-1, 1], [[1]*8]*2)
        if sum(driver.allocations.values()) != baseline:
            raise AssertionError("unreleased CUDA allocations")
        if (graph.topology_hash(), graph.weights_hash()) != original or sources() != manifest:
            raise AssertionError("source/graph drift")
    return {"terminal": "LIGHT_CUDA_PROBE_PASS", "raw_files": commitments, "forward_cases": 72,
            "training_updates": 64, "max_forward_error": max(errors), "allocation_growth_bytes": 0,
            "official_experiment": False, "readiness": 21}


def admission_gates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if tuple(row["seed"] for row in rows) != SEEDS:
        raise ValueError("development admission requires all five ordered seeds")
    candidate = [row["eval"]["candidate_metrics"] for row in rows]
    native = [row["eval"]["native_metrics"] for row in rows]
    cr, nr = ref._median_guarded_ratio(candidate, "R"), ref._median_guarded_ratio(native, "R")
    co = ref._median_guarded_ratio(candidate, "O")
    gates = {
        "A01": ref._median(x["C"] for x in candidate) >= 3*ref._median(x["C"] for x in native),
        "A02": sum(x["C"] >= 3*y["C"] for x, y in zip(candidate, native, strict=True)) >= 4,
        "A03": ref._ratio_at_least(cr, .5), "A04": ref._ratio_multiple_at_least(cr, nr, 3.),
        "A05": sum(ref._ratio_multiple_at_least(x["R"], y["R"], 3.) for x, y in zip(candidate, native, strict=True)) >= 4,
        "A06": ref._median(row["ridge_candidate"]["eval_accuracy"] for row in rows) >= .7,
        "A07": co.get("status") != "zero_over_zero" and ref._ratio_at_least(co, .5),
        "A08": all(ref._observed_ratio_at_most(split["split_event_ratio"], 1.1)
                   and ref._observed_ratio_at_most(split["split_forward_touch_ratio"], 1.1)
                   and all(ref._observed_ratio_at_most(a["candidate_event_ratio"], 1.15)
                           and ref._observed_ratio_at_most(a["candidate_forward_touch_ratio"], 1.15) for a in split["activity"])
                   for row in rows for split in (row["train"], row["eval"])),
        "A09": all(row["tail"]["pass"] for row in rows),
        "A10_local_only": all(row["all_invariants_pass"] for row in rows),
    }
    failed = [name for name, passed in gates.items() if not passed]
    return {"gates": gates, "failed": failed, "numerical_admission_pass": not failed,
            "median_ridge_accuracy": ref._median(row["ridge_candidate"]["eval_accuracy"] for row in rows),
            "candidate_R": cr, "native_R": nr, "candidate_O": co,
            "formal_integrity_attestation": "NOT_PERFORMED", "readiness": 21}


def admission(driver: Driver, output: Path, manifest: dict[str, str]) -> dict[str, Any]:
    summaries, commitments = [], {}
    for seed in SEEDS:
        headroom()
        graph = ref.LWOHObserverGraph.from_seed(seed)
        shadow = ref.build_experiment_000_graph(seed=seed, mode="full", event_log_enabled=False, **ref.FROZEN_GRAPH_OPTIONS)
        gpu = LightForward(driver, Topology.from_reference(graph))
        original = graph.topology_hash(), graph.weights_hash()
        result: dict[str, Any] = {"seed": seed}
        for split in ("train", "eval"):
            pairs = ref._paired_rows(seed, split, 100, 8)
            cpu = ref._collect_admission_split(graph, shadow, pairs)
            flat = [sample for pair in pairs for sample in pair]
            batches = []
            for start in range(0, len(flat), 32):
                headroom()
                batch = flat[start:start+32]
                actual = gpu.run([sample.cue for sample in batch], [list(sample.noise) for sample in batch])
                batches.append(actual)
            lwoh = np.concatenate([batch["lwoh"] for batch in batches])
            latest = np.concatenate([batch["latest"] for batch in batches])
            outputs = np.concatenate([batch["output"] for batch in batches])
            structural = np.tanh(lwoh @ np.asarray(graph.cue_edge_weights))
            arrays = {"lwoh_minus": lwoh[::2], "lwoh_plus": lwoh[1::2],
                      "latest_minus": latest[::2], "latest_plus": latest[1::2],
                      "native_minus": outputs[::2], "native_plus": outputs[1::2],
                      "structural_minus": structural[::2], "structural_plus": structural[1::2]}
            maximum = max(_close(value, cpu["arrays"][key], key) for key, value in arrays.items())
            events = np.concatenate([batch["events"] for batch in batches])
            touches = np.concatenate([batch["touches"] for batch in batches])
            for index, activity in enumerate(cpu["activity"]):
                if events[index] != activity["native_events"] or touches[index] != activity["native_forward_edge_touches"]:
                    raise AssertionError("GPU/native activity mismatch")
            result[split] = {**cpu, "arrays": {key: value.tolist() for key, value in arrays.items()},
                             "candidate_metrics": ref._paired_metrics(arrays["lwoh_minus"], arrays["lwoh_plus"], arrays["structural_minus"], arrays["structural_plus"]),
                             "native_metrics": ref._paired_metrics(arrays["latest_minus"], arrays["latest_plus"], arrays["native_minus"], arrays["native_plus"]),
                             "cpu_gpu_max_error": maximum, "gpu_events": events.tolist(), "gpu_touches": touches.tolist()}
            name = f"cpu_{seed}_{split}.json"
            commitments[name] = write(output / name, cpu)
        train, evaluation = result["train"]["arrays"], result["eval"]["arrays"]
        def interleave(split: dict[str, Any], kind: str) -> np.ndarray:
            return np.stack((split[kind+"_minus"], split[kind+"_plus"]), axis=1).reshape(-1, 8)
        y = np.tile((-1., 1.), 100)
        result["ridge_candidate"] = ref._ridge_fit(interleave(train, "lwoh"), y, interleave(evaluation, "lwoh"), y)
        result["ridge_native"] = ref._ridge_fit(interleave(train, "latest"), y, interleave(evaluation, "latest"), y)
        result["tail"] = check_tail(gpu, seed)
        result["all_invariants_pass"] = ((graph.topology_hash(), graph.weights_hash()) == original
                                         and len(graph.credit_packets) == 0 and sources() == manifest
                                         and all(a["native_equivalent"] for split in (result["train"], result["eval"]) for a in split["activity"]))
        result["graph_hashes"] = original
        name = f"seed_{seed}.json"
        commitments[name] = write(output / name, result)
        summaries.append(result)
        print(f"ADMISSION_SEED_COMPLETE {seed}", flush=True)
    return {"terminal": "DEVELOPMENT_ADMISSION_SCREEN", "decision": admission_gates(summaries),
            "raw_files": commitments, "episodes": 2000, "official_experiment": False,
            "learning_streams_generated": False, "readiness": 21}


def run(output: Path, phase: str, prerequisites: list[Path]) -> dict[str, Any]:
    output = output.resolve()
    parent = ROOT / "artifacts/gpu_light_v1_2026-09-05"
    if output.parent != parent or output.exists():
        raise ValueError("use a new immediate child of artifacts/gpu_light_v1_2026-09-05")
    if phase not in ("probe", "admission"):
        raise ValueError("unknown phase")
    manifest = sources()
    compiler = default_compiler()
    compiler_files = {}
    for path in (compiler, compiler.parent / "nvrtc-builtins64_129.dll"):
        with path.open("rb") as stream:
            compiler_files[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    compiler_hash = _hash(compiler_files)
    if phase == "admission":
        if len(prerequisites) != 2 or prerequisites[0].resolve() == prerequisites[1].resolve():
            raise ValueError("admission requires two distinct probe reports")
        reports = [read(path) for path in prerequisites]
        for report, path in zip(reports, prerequisites, strict=True):
            if report["payload"]["terminal"] != "LIGHT_CUDA_PROBE_PASS" or report["sources"] != manifest:
                raise ValueError("probe prerequisite failed or source drifted")
            if report["payload_sha256"] != _hash(report["payload"]) or report["compiler_sha256"] != compiler_hash:
                raise ValueError("prerequisite payload/compiler mismatch")
            for name, digest in report["payload"]["raw_files"].items():
                if Path(name).name != name or hashlib.sha256((path.parent / name).read_bytes()).hexdigest() != digest:
                    raise ValueError("prerequisite raw evidence mismatch")
                read(path.parent / name)
        if reports[0]["payload_sha256"] != reports[1]["payload_sha256"] or reports[0]["pid"] == reports[1]["pid"]:
            raise ValueError("probe rerun is not matching and fresh-process")
    initial = host_memory()
    if initial["available_system_ram_bytes"] < 512 * 1024**2:
        raise MemoryError("light CUDA startup requires 512 MiB available host RAM")
    output.mkdir()
    write(output / "PROSPECTIVE_FREEZE.json", {"sources": manifest, "compiler_sha256": compiler_hash,
          "phase": phase, "seeds": [12000, 12001] if phase == "probe" else list(SEEDS), "official": False})
    with (output / "PROTOCOL.md").open("xb") as stream:
        stream.write((ROOT / PROTOCOL).read_bytes())
    started = time.perf_counter()
    driver = Driver(compiler)
    try:
        headroom()
        initialized = host_memory()
        payload = probe(driver, output, manifest) if phase == "probe" else admission(driver, output, manifest)
        if sources() != manifest:
            raise RuntimeError("source drift before publication")
        result = {"payload": payload, "payload_sha256": _hash(payload), "sources": manifest,
                  "compiler_sha256": compiler_hash, "ptx_sha256": driver.ptx_sha256,
                  "pid": os.getpid(), "elapsed_seconds": time.perf_counter()-started,
                  "host_initial": initial, "host_initialized": initialized, "host_final": host_memory(),
                  "device_explicit_peak_bytes": driver.peak_allocated, "architecture": driver.architecture}
    finally:
        driver.close()
    if driver.allocations or driver.context:
        raise AssertionError("CUDA resources not released")
    result["context_released"] = True
    write(output / "report.json", result)
    print(payload["terminal"], result["payload_sha256"], flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("probe", "admission"), default="probe")
    parser.add_argument("--prerequisite", type=Path, action="append", default=[])
    args = parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run(args.output, args.phase, args.prerequisite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
