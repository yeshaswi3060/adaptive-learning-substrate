"""Run the prospective, nonselecting GPU VRAM V1 integration probe."""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .gpu_lwoh import (
    ATOL,
    PROBE_SEEDS,
    RTOL,
    TENSOR_BUDGET_BYTES,
    CudaForward,
    CudaOnlineHead,
    Topology,
)


def host_memory() -> dict[str, int]:
    """Windows working-set, private commit and available physical RAM (not VRAM)."""
    if os.name != "nt":
        raise RuntimeError("GPU V1 host-memory accounting is validated on Windows only")

    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in (
                "total", "available", "total_pagefile", "available_pagefile",
                "total_virtual", "available_virtual", "extended",
            )
        ]

    class Process(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in (
                "peak_working_set", "working_set", "peak_paged", "paged", "peak_nonpaged",
                "nonpaged", "commit", "peak_commit", "private",
            )
        ]

    memory, process = Memory(), Process()
    memory.length, process.cb = ctypes.sizeof(memory), ctypes.sizeof(process)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise OSError("GlobalMemoryStatusEx failed")
    current = ctypes.windll.kernel32.GetCurrentProcess
    current.restype = ctypes.c_void_p
    get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = (ctypes.c_void_p, ctypes.POINTER(Process), ctypes.c_ulong)
    get_info.restype = ctypes.c_bool
    if not get_info(current(), ctypes.byref(process), process.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return {"available_system_ram_bytes": int(memory.available),
            "process_working_set_bytes": int(process.working_set),
            "process_peak_working_set_bytes": int(process.peak_working_set),
            "process_private_commit_bytes": int(process.private),
            "process_peak_commit_bytes": int(process.peak_commit)}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sources() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    names = (
        "src/adaptive_learning_substrate/gpu_lwoh.py",
        "src/adaptive_learning_substrate/gpu_probe.py",
        "src/adaptive_learning_substrate/recurrent.py",
        "src/adaptive_learning_substrate/experiment_000_lwoh_l1_execution_v3.py",
        "docs/GPU_VRAM_PROBE_V1.md",
    )
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def _cpu_trace(graph: Any, cue: int | None, noise: list[int]) -> dict[str, Any]:
    before = graph.ledger
    graph.begin_episode("gpu-probe-synthetic")
    frames = []
    inputs = [{} if cue is None else {"cue": float(cue)}]
    inputs += [{"noise": float(value)} for value in noise]
    inputs += [{"query": 1.0}, {}, {}]
    for tick, row in enumerate(inputs):
        graph._advance(row, force_output=tick == len(inputs) - 1)
        frames.append({
            "tick": tick,
            "activations": [graph._activations[node] for node in graph.hidden_nodes + (graph.output_node,)],
            "emitted": [any(event.node == node and event.step == tick for event in graph.unit_events)
                        for node in graph.nodes],
        })
    after = graph.ledger
    return {"tick": len(inputs) - 1, "frames": frames,
            "output": graph._activations[graph.output_node],
            "lwoh": graph.observer_features("lwoh"), "latest": graph.observer_features("latest"),
            "events": after["emitted_unit_events"] - before["emitted_unit_events"],
            "touches": after["forward_edge_touches"] - before["forward_edge_touches"],
            "evaluations": after["activation_evaluations"] - before["activation_evaluations"],
            "observer_calls": graph.observer_ledger["kernel_calls"],
            "writes": graph.observer_ledger["lwoh_writes"]}


def _close(actual: Any, expected: Any, label: str) -> float:
    import numpy as np
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.allclose(a, b, atol=ATOL, rtol=RTOL):
        raise AssertionError(f"CPU/GPU mismatch: {label}")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def _expect_error(call: Any, error: type[Exception], *args: Any) -> None:
    try:
        call(*args)
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def run_probe(output: Path) -> dict[str, Any]:
    output = output.resolve()
    root = Path(__file__).resolve().parents[2]
    relative = output.relative_to(root)
    if len(relative.parts) < 3 or relative.parts[0] != "artifacts" or not relative.parts[1].startswith("gpu_"):
        raise ValueError("use a new artifacts/gpu_*/NAME diagnostic directory")
    if output.exists():
        raise FileExistsError("probe output already exists; evidence is never overwritten")
    initial_host = host_memory()
    if initial_host["available_system_ram_bytes"] < 1024**3:
        raise MemoryError("less than 1 GiB startup system RAM is available for the measured CUDA runtime")
    sources = _sources()
    # Persist the protocol/source binding before loading CUDA or observing outputs.
    output.mkdir(parents=True)
    (output / "PROSPECTIVE_FREEZE.json").write_bytes(_canonical({
        "sources": sources, "seeds": PROBE_SEEDS, "atol": ATOL, "rtol": RTOL,
        "selecting": False, "official_experiment": False,
    }) + b"\n")
    started = time.perf_counter()
    import torch as t

    from .experiment_000_lwoh_l1_execution_v3 import LWOHObserverGraph, OnlineHead
    if not t.cuda.is_available():
        raise RuntimeError("CUDA unavailable; probe will not fall back to CPU")
    t.set_num_threads(1)
    t.use_deterministic_algorithms(True)
    t.backends.cuda.matmul.fp32_precision = "ieee"
    t.backends.cudnn.conv.fp32_precision = "ieee"
    free, total = t.cuda.mem_get_info(0)
    t.cuda.set_per_process_memory_fraction(TENSOR_BUDGET_BYTES / total, 0)
    t.cuda.reset_peak_memory_stats(0)
    loaded_host = host_memory()
    if loaded_host["available_system_ram_bytes"] < 128 * 1024**2:
        raise MemoryError("system RAM headroom fell below 128 MiB after CUDA initialization")
    deterministic: dict[str, Any] = {"seeds": list(PROBE_SEEDS), "forward": [], "training": [],
                                     "allocation_bounded": True, "negative_cases_pass": True}
    samples_checked = 0
    max_error = 0.0
    with (output / "checks.jsonl").open("x", encoding="utf-8") as log:
        for seed in PROBE_SEEDS:
            graph = LWOHObserverGraph.from_seed(seed)
            original = (graph.topology_hash(), graph.weights_hash())
            gpu = CudaForward(Topology.from_reference(graph))
            gpu_weights = gpu.weights.clone()
            for delay in (0, 4, 8, 16, 32, 40):
                if host_memory()["available_system_ram_bytes"] < 128 * 1024**2:
                    raise MemoryError("system RAM headroom fell below 128 MiB during the probe")
                cues = [-1, 1, None, -1, 1, None]
                noise = [[1 if (i + row) % 3 else -1 for i in range(delay)] for row in range(6)]
                actual = gpu.run(cues, noise, trace=True)
                for index, (cue, sequence) in enumerate(zip(cues, noise, strict=True)):
                    expected = _cpu_trace(graph, cue, sequence)
                    errors = []
                    for frame, target in zip(actual["frames"], expected["frames"], strict=True):
                        errors.append(_close(frame["activations"][index], target["activations"], "tick activations"))
                        if frame["emitted"][index] != target["emitted"]:
                            raise AssertionError("CPU/GPU emission mask mismatch")
                    for field in ("output", "lwoh", "latest"):
                        errors.append(_close(actual[field][index].cpu().tolist(), expected[field], field))
                    for field in ("events", "touches", "evaluations", "observer_calls", "writes"):
                        if int(actual[field][index]) != expected[field]:
                            raise AssertionError(f"CPU/GPU count mismatch: {field}")
                    if bool(actual["output"][index] >= 0) != (expected["output"] >= 0):
                        raise AssertionError("native categorical prediction mismatch")
                    single = gpu.run([cue], [sequence])
                    for field in ("output", "lwoh", "latest"):
                        errors.append(_close(single[field][0].cpu().tolist(), actual[field][index].cpu().tolist(), "batch/single"))
                    if delay >= 32 and cue is not None and not bool((actual["lwoh"][index] == 0).all()):
                        raise AssertionError("expired latch was not zero")
                    row = {"seed": seed, "delay": delay, "row": index, "max_error": max(errors),
                           "output": float(actual["output"][index]), "pass": True}
                    max_error = max(max_error, row["max_error"])
                    deterministic["forward"].append(row)
                    log.write(_canonical(row).decode() + "\n")
                    log.flush()
                    samples_checked += 1
                repeated = gpu.run(cues, noise)
                if not all(t.equal(actual[key], repeated[key]) for key in ("output", "lwoh", "latest", "events", "touches")):
                    raise AssertionError("identical GPU inputs did not reproduce exactly")
                del actual, repeated, single
            head, cpu_head = CudaOnlineHead(), OnlineHead()
            train_errors = []
            for index in range(32):
                cue = -1 if index % 2 == 0 else 1
                noise = [1 if (i + index) % 3 else -1 for i in range(8)]
                feature = gpu.run([cue], [noise])["lwoh"][0]
                expected = _cpu_trace(graph, cue, noise)
                cpu_pending = cpu_head.predict_for_update(expected["lwoh"])
                token, score, prediction = head.predict_for_update(feature)
                train_errors.append(_close(float(score), cpu_pending.score, "training score"))
                if int(prediction) != cpu_pending.prediction:
                    raise AssertionError("training prediction mismatch")
                if index == 0:
                    _expect_error(head.predict_for_update, RuntimeError, feature)
                    _expect_error(head.reveal_target, RuntimeError, token + 1, cue)
                    _expect_error(head.reveal_target, ValueError, token, 0)
                delta = head.reveal_target(token, cue)
                expected_delta = cpu_head.reveal_target(cpu_pending, cue)
                train_errors.append(_close(delta.cpu().tolist(), expected_delta, "local update"))
                train_errors.append(_close(head.theta.cpu().tolist(), cpu_head.theta, "head weights"))
                _expect_error(head.reveal_target, RuntimeError, token, cue)
            eval_rows = []
            for delay in (4, 8, 16):
                for cue in (-1, 1, None):
                    noise = [-1] * delay
                    feature = gpu.run([cue], [noise])["lwoh"][0]
                    score, prediction = head.predict(feature)
                    expected = _cpu_trace(graph, cue, noise)
                    cpu_score, cpu_prediction = cpu_head.predict(expected["lwoh"])
                    train_errors.append(_close(float(score), cpu_score, "evaluation score"))
                    if int(prediction) != cpu_prediction:
                        raise AssertionError("evaluation prediction mismatch")
                    if cue is not None and int(prediction) != cue:
                        raise AssertionError("constructed delayed-cue control did not learn")
                    eval_rows.append({"delay": delay, "cue": cue, "prediction": int(prediction)})
            if head.theta.requires_grad or head.theta.grad is not None or head.updates != 32:
                raise AssertionError("head autograd/chronology mismatch")
            if (graph.topology_hash(), graph.weights_hash()) != original or not t.equal(gpu.weights, gpu_weights):
                raise AssertionError("frozen recurrent weights or topology changed")
            training = {"seed": seed, "updates": head.updates, "max_error": max(train_errors),
                        "theta": head.theta.cpu().tolist(), "evaluation": eval_rows, "pass": True}
            deterministic["training"].append(training)
            max_error = max(max_error, training["max_error"])
            log.write(_canonical(training).decode() + "\n")
            # Allocation does not scale with the number of processed episodes.
            del feature, delta, score, prediction, head, gpu_weights
            gc.collect()
            t.cuda.synchronize()
            warm = gpu.run([-1, 1], [[1, -1] * 8] * 2)
            del warm
            t.cuda.synchronize()
            allocated = t.cuda.memory_allocated()
            for _ in range(128):
                item = gpu.run([-1, 1], [[1, -1] * 8] * 2)
                del item
            t.cuda.synchronize()
            growth = t.cuda.memory_allocated() - allocated
            if growth > 1024**2:
                raise MemoryError("CUDA tensors accumulate across repeated batches")
            capacity = gpu.run([-1, 1] * 16, [[1, -1] * 128] * 32)
            if tuple(capacity["lwoh"].shape) != (32, 8) or not bool((capacity["lwoh"] == 0).all()):
                raise AssertionError("maximum supported batch/sequence failed")
            if not bool(t.isfinite(capacity["output"]).all()):
                raise AssertionError("maximum supported batch produced nonfinite output")
            del capacity
            _expect_error(gpu.run, ValueError, [], [])
            _expect_error(gpu.run, ValueError, [1] * 33, [[]] * 33)
            _expect_error(gpu.run, ValueError, [1], [[1] * 257])
            _expect_error(gpu.run, ValueError, [True], [[]])
            del gpu, graph
    t.cuda.synchronize()
    final_host = host_memory()
    final_free, _ = t.cuda.mem_get_info(0)
    if _sources() != sources:
        raise RuntimeError("source files changed during GPU validation")
    if t.cuda.max_memory_reserved() > TENSOR_BUDGET_BYTES:
        raise MemoryError("GPU tensor reserve exceeded the frozen probe budget")
    report = {
        "status": "GPU_PROBE_PASS", "official_experiment": False, "selecting": False,
        "readiness_changed": False, "forward_cases": samples_checked, "max_absolute_error": max_error,
        "deterministic": deterministic, "deterministic_sha256": _hash(deterministic), "sources": sources,
        "runtime": {"torch": t.__version__, "cuda": t.version.cuda, "gpu": t.cuda.get_device_name(0),
                    "seconds": time.perf_counter() - started, "pid": os.getpid()},
        "memory": {"before_torch": initial_host, "after_cuda_init": loaded_host, "at_end": final_host,
                   "device_free_before_bytes": free, "device_free_after_bytes": final_free,
                   "device_total_bytes": total, "tensor_peak_allocated_bytes": t.cuda.max_memory_allocated(),
                   "tensor_peak_reserved_bytes": t.cuda.max_memory_reserved(),
                   "tensor_allocator_budget_bytes": TENSOR_BUDGET_BYTES},
    }
    raw = _canonical(report) + b"\n"
    with (output / "report.json").open("xb") as handle:
        handle.write(raw)
    if (output / "report.json").read_bytes() != raw:
        raise RuntimeError("GPU report changed on reopen")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = run_probe(args.output)
    except (RuntimeError, ValueError, OSError, AssertionError, MemoryError, ImportError) as exc:
        print(f"GPU_PROBE_FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"{report['status']} cases={report['forward_cases']} sha256={report['deterministic_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
