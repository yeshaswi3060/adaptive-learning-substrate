"""Frozen small byte-model development experiment and independent CPU replay."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from .cuda_light import default_compiler
from .gpu_probe import host_memory
from .sequence_core import (
    CORE,
    PARAMETERS,
    SIZE,
    CpuSequence,
    canonical,
    initial,
    read_record,
    views,
    write_record,
)
from .sequence_cuda import GpuSequence, SequenceDriver

ROOT = Path(__file__).resolve().parents[2]
HOME = ROOT / "artifacts/sequence_core_v1_2026-09-05"
SEEDS = (15000, 15001, 15002)
DATA = (("docs/CCF_V0.md", 2048), ("docs/EXPERIMENT_001.md", 2048),
        ("docs/SEQUENCE_ARCHITECTURE_TARGET.md", 1024))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bindings() -> dict[str, str]:
    paths = sorted((ROOT / "src/adaptive_learning_substrate").rglob("*.py"))
    paths += [ROOT / "docs/SEQUENCE_CORE_V1.md", ROOT / "pyproject.toml"]
    paths += [ROOT / path for path, _ in DATA]
    paths += [default_compiler(), default_compiler().parent / "nvrtc-builtins64_129.dll"]
    return {str(path.resolve()): digest(path) for path in paths}


def headroom() -> None:
    if host_memory()["available_system_ram_bytes"] < 128*1024**2:
        raise MemoryError("less than 128 MiB available system RAM at sequence boundary")


def close(actual: Any, expected: Any, label: str) -> float:
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.allclose(a, b, atol=1e-10, rtol=1e-10):
        raise AssertionError("CPU/GPU mismatch: "+label)
    return float(np.max(np.abs(a-b))) if a.size else 0.


def payload_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def parity(driver: SequenceDriver, output: Path) -> dict[str, Any]:
    logs, checkpoints = [], {}
    for plastic in (False, True):
        cpu, gpu = CpuSequence(core_learning=plastic), GpuSequence(driver, core_learning=plastic)
        try:
            for index in range(320):
                token = index if index < 256 else (index*17) % 256
                training = index >= 256
                a, b = gpu.step(token, training=training), cpu.step(token, training=training)
                error = close(a, b, "byte probabilities")
                gap = float(np.sort(b)[-1]-np.sort(b)[-2])
                if gap > 1e-10 and int(a.argmax()) != int(b.argmax()):
                    raise AssertionError("byte argmax mismatch away from ties")
                if training:
                    target = (index*7+3) % 256
                    gpu.learn(target)
                    cpu.learn(target)
                ga, _ = gpu._export()
                ca, _ = cpu._export()
                error = max(error, close(ga, ca, "complete state/traces/parameters"))
                logs.append({"core_learning": plastic, "index": index, "token": token,
                             "training": training, "max_error": error,
                             "gpu_buffer_sha256": hashlib.sha256(ga.tobytes()).hexdigest(), "near_tie": gap <= 1e-10})
            gpu.step(99, training=True)
            name = f"gpu_pending_{int(plastic)}.json"
            checkpoints[name] = gpu.save(output / name)
            cpu.restore(output / name)
            gpu.learn(100)
            cpu.learn(100)
            close(gpu._export()[0], cpu._export()[0], "pending checkpoint CPU restore")
            name = f"cpu_checkpoint_{int(plastic)}.json"
            checkpoints[name] = cpu.save(output / name)
            gpu.restore(output / name)
            for token in b"cross backend continuation":
                close(gpu.step(token), cpu.step(token), "CPU checkpoint GPU restore")
            unchanged = gpu.parameter_digest()
            before = canonical(gpu.snapshot())
            for invalid in (-1, 256, True, None):
                try:
                    gpu.step(invalid)
                except ValueError:
                    pass
                else:
                    raise AssertionError("invalid GPU byte accepted")
                if canonical(gpu.snapshot()) != before:
                    raise AssertionError("invalid GPU input mutated state")
            gpu.reset()
            if gpu.parameter_digest() != unchanged:
                raise AssertionError("GPU reset changed weights")
            whole = [gpu.step(token).tolist() for token in b"chunk independence"]
            gpu.reset()
            chunks = [gpu.step(token).tolist() for part in (b"chunk ", b"independence") for token in part]
            if whole != chunks or gpu.parameter_digest() != unchanged:
                raise AssertionError("GPU chunk/evaluation invariance failed")
            allocated = sum(driver.allocations.values())
            for index in range(4096):
                if index % 128 == 0:
                    headroom()
                gpu.step(index % 256)
            if sum(driver.allocations.values()) != allocated or gpu.parameter_digest() != unchanged:
                raise AssertionError("GPU persistent buffer/evaluation bound failed")
        finally:
            gpu.close()
            cpu.close()
        try:
            gpu.step(1)
        except RuntimeError:
            pass
        else:
            raise AssertionError("closed GPU model accepted tokens")
    name = "parity_records.json"
    checkpoints[name] = write_record(output / name, logs)
    return {"pass": True, "token_steps_compared": 640, "updates_compared": 128,
            "max_error": max(row["max_error"] for row in logs), "near_ties": sum(row["near_tie"] for row in logs),
            "long_stream_tokens_each_mode": 4096, "device_growth_bytes": 0, "raw_files": checkpoints}


def evaluate(model: CpuSequence, data: bytes) -> dict[str, Any]:
    model.reset()
    before = model.parameter_digest()
    updates = model.updates
    probabilities, predictions, gaps = [], [], []
    for index in range(len(data)-1):
        if isinstance(model, GpuSequence) and index % 128 == 0:
            headroom()
        p = model.step(data[index])
        probabilities.append(float(p[data[index+1]]))
        predictions.append(int(p.argmax()))
        ordered = np.sort(p)
        gaps.append(float(ordered[-1]-ordered[-2]))
    nll = -float(np.log(probabilities).mean())
    if model.parameter_digest() != before or model.updates != updates:
        raise AssertionError("evaluation changed weights/update count")
    return {"nll_nats_per_byte": nll, "byte_perplexity": math.exp(nll),
            "byte_accuracy": float(np.mean(np.asarray(predictions) == np.frombuffer(data[1:], dtype=np.uint8))),
            "target_probabilities": probabilities, "predicted_bytes": predictions, "top_two_margins": gaps,
            "evaluation_updates": 0, "weight_digest": before}


def train(model: CpuSequence, documents: list[bytes], *, guarded: bool = True) -> dict[str, Any]:
    nll, count = 0., 0
    for data in documents:
        model.reset()
        for index in range(len(data)-1):
            if guarded and index % 128 == 0:
                headroom()
            p = model.step(data[index], training=True)
            target = data[index+1]  # Read only after the prediction has been emitted.
            nll -= math.log(float(p[target]))
            model.learn(target)
            count += 1
    return {"updates": count, "preupdate_nll_nats_per_byte": nll/count}


def bigram(documents: list[bytes], evaluation: bytes) -> dict[str, Any]:
    counts = np.ones((256, 256), dtype=np.int64)
    for data in documents:
        for index in range(len(data)-1):
            counts[data[index], data[index+1]] += 1
    p = counts/counts.sum(axis=1, keepdims=True)
    probabilities = [float(p[a, b]) for a, b in itertools.pairwise(evaluation)]
    nll = -float(np.log(probabilities).mean())
    return {"alpha": 1, "nll_nats_per_byte": nll, "byte_perplexity": math.exp(nll),
            "byte_accuracy": float(np.mean([int(p[a].argmax()) == b for a, b in itertools.pairwise(evaluation)]))}


def run(output: Path) -> None:
    output = output.resolve()
    if output.parent != HOME or output.exists():
        raise ValueError("use a fresh immediate child of the sequence transaction directory")
    manifest = bindings()
    initial_memory = host_memory()
    if initial_memory["available_system_ram_bytes"] < 512*1024**2:
        raise MemoryError("sequence CUDA startup needs 512 MiB available host RAM")
    data = [(ROOT / path).read_bytes()[:count] for path, count in DATA]
    if any(len(value) != count for value, (_, count) in zip(data, DATA, strict=True)):
        raise ValueError("frozen data file is shorter than the byte budget")
    output.mkdir()
    data_files = {f"data_{i}.bin": {"source": DATA[i][0], "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
                  for i, value in enumerate(data)}
    write_record(output / "PROSPECTIVE_FREEZE.json", {"bindings": manifest, "seeds": SEEDS, "data": data_files,
                 "python": platform.python_version(), "numpy": np.__version__, "official_experiment": False})
    for index, value in enumerate(data):
        with (output / f"data_{index}.bin").open("xb") as stream:
            stream.write(value)
    driver = SequenceDriver(default_compiler())
    started = time.perf_counter()
    try:
        headroom()
        checked = parity(driver, output)
        print("SEQUENCE_CPU_GPU_PARITY_PASS", checked["max_error"], flush=True)
        results, resources, raw_files = [], [], dict(checked["raw_files"])
        for seed in SEEDS:
            modes = {}
            for plastic, name in ((True, "local_learning"), (False, "readout_only")):
                headroom()
                model = GpuSequence(driver, seed, core_learning=plastic)
                try:
                    untrained = evaluate(model, data[2])
                    begin = time.perf_counter()
                    training = train(model, data[:2])
                    train_seconds = time.perf_counter()-begin
                    filename = f"trained_{seed}_{name}.json"
                    raw_files[filename] = model.save(output / filename)
                    parameters = views(model._export()[0])
                    originals = views(initial(seed)[0])
                    changed = {key: not np.array_equal(parameters[key], originals[key]) for key in PARAMETERS}
                    if not all(changed[key] == plastic for key in CORE) or not changed["o"] or not changed["bo"]:
                        raise AssertionError("plasticity/readout ablation does not match registered updates")
                    begin = time.perf_counter()
                    evaluation = evaluate(model, data[2])
                    eval_seconds = time.perf_counter()-begin
                    modes[name] = {"untrained": untrained, "training": training, "evaluation": evaluation,
                                   "changed_parameter_groups": changed, "checkpoint": filename}
                    resources.append({"seed": seed, "condition": name, "train_seconds": train_seconds,
                                      "eval_seconds": eval_seconds, "host": host_memory()})
                finally:
                    model.close()
                print("BYTE_MODEL_COMPLETE", seed, name, evaluation["nll_nats_per_byte"], flush=True)
            if bindings() != manifest:
                raise RuntimeError("source/data/compiler drift at seed boundary")
            filename = f"metrics_{seed}.json"
            raw_files[filename] = write_record(output / filename, modes)
            results.append({"seed": seed, "conditions": {name: {"untrained_nll": row["untrained"]["nll_nats_per_byte"],
                            **{key: row["evaluation"][key] for key in ("nll_nats_per_byte", "byte_perplexity", "byte_accuracy")}}
                            for name, row in modes.items()}})
        baseline = bigram(data[:2], data[2])
        uniform = math.log(256)
        candidate = [row["conditions"]["local_learning"] for row in results]
        learning_gate = all(row["nll_nats_per_byte"] < min(row["untrained_nll"], uniform) for row in candidate)
        train_ngrams = {value[i:i+8] for value in data[:2] for i in range(len(value)-7)}
        test_ngrams = [data[2][i:i+8] for i in range(len(data[2])-7)]
        payload = {"terminal": "SEQUENCE_CORE_DEVELOPMENT_RESULT", "engineering": checked, "data": data_files,
                   "seed_results": results, "bigram": baseline, "uniform_nll": uniform,
                   "descriptive_learning_gate": learning_gate,
                   "beats_bigram_all_seeds": all(row["nll_nats_per_byte"] < baseline["nll_nats_per_byte"] for row in candidate),
                   "beats_readout_only_all_seeds": all(row["conditions"]["local_learning"]["nll_nats_per_byte"] < row["conditions"]["readout_only"]["nll_nats_per_byte"] for row in results),
                   "test_8gram_fraction_present_in_training": sum(x in train_ngrams for x in test_ngrams)/len(test_ngrams),
                   "raw_files": raw_files, "state_buffer_bytes": SIZE*8, "official_readiness": 21,
                   "transformer_comparison": "NOT_RUN", "general_language_quality_claim": False}
        if driver.allocations or bindings() != manifest:
            raise RuntimeError("unreleased models or source drift before publication")
        report = {"payload": payload, "payload_sha256": payload_hash(payload), "bindings": manifest, "pid": os.getpid(),
                  "ptx_sha256": driver.ptx_sha256, "host_initial": initial_memory, "host_final": host_memory(),
                  "device_explicit_peak_bytes": driver.peak_allocated, "timings": resources,
                  "elapsed_seconds": time.perf_counter()-started}
    finally:
        driver.close()
    report["context_released"] = not driver.context and not driver.allocations
    write_record(output / "report.json", report)
    print("SEQUENCE_CORE_COMPLETE", report["payload_sha256"], flush=True)


def verify(first: Path, second: Path, output: Path) -> None:
    reports = [read_record(path / "report.json") for path in (first, second)]
    if reports[0]["pid"] == reports[1]["pid"] or reports[0]["payload_sha256"] != reports[1]["payload_sha256"]:
        raise AssertionError("sequence rerun is not matching and fresh-process")
    for path, report in zip((first, second), reports, strict=True):
        if report["bindings"] != bindings() or report["payload_sha256"] != payload_hash(report["payload"]) or not report["context_released"]:
            raise AssertionError("source/payload/context evidence mismatch")
        freeze = read_record(path / "PROSPECTIVE_FREEZE.json")
        if freeze["bindings"] != report["bindings"] or freeze["data"] != report["payload"]["data"]:
            raise AssertionError("prospective freeze mismatch")
        for name, expected in report["payload"]["raw_files"].items():
            if Path(name).name != name or digest(path / name) != expected:
                raise AssertionError("raw sequence evidence digest mismatch")
            read_record(path / name)
        for name, row in report["payload"]["data"].items():
            if Path(name).name != name or digest(path / name) != row["sha256"] or (path/name).stat().st_size != row["bytes"]:
                raise AssertionError("frozen input bytes mismatch")
    data = [(first / f"data_{index}.bin").read_bytes() for index in range(3)]
    checks, maximum = [], 0.
    for seed in SEEDS:
        modes = read_record(first / f"metrics_{seed}.json")
        for plastic, name in ((True, "local_learning"), (False, "readout_only")):
            cpu = CpuSequence(seed, core_learning=plastic)
            untrained = evaluate(cpu, data[2])
            training = train(cpu, data[:2], guarded=False)
            checkpoint = read_record(first / modes[name]["checkpoint"])
            maximum = max(maximum, close(cpu._export()[0], checkpoint["buffer"], "full independent training replay"))
            if cpu.updates != checkpoint["updates"] or cpu.steps != checkpoint["steps"]:
                raise AssertionError("training replay chronology mismatch")
            result = evaluate(cpu, data[2])
            maximum = max(maximum, close(result["target_probabilities"], modes[name]["evaluation"]["target_probabilities"], "held-out byte probabilities"))
            for index, (a, b) in enumerate(zip(result["predicted_bytes"], modes[name]["evaluation"]["predicted_bytes"], strict=True)):
                if a != b and modes[name]["evaluation"]["top_two_margins"][index] > 1e-10:
                    raise AssertionError("held-out byte argmax differs away from tie")
            close(training["preupdate_nll_nats_per_byte"], modes[name]["training"]["preupdate_nll_nats_per_byte"], "pre-update losses")
            close(untrained["nll_nats_per_byte"], modes[name]["untrained"]["nll_nats_per_byte"], "untrained loss")
            close(result["nll_nats_per_byte"], modes[name]["evaluation"]["nll_nats_per_byte"], "held-out NLL")
            checks.append({"seed": seed, "condition": name, "updates_replayed": cpu.updates, "nll": result["nll_nats_per_byte"]})
            cpu.close()
            print("CPU_TRAINING_REPLAY_PASS", seed, name, flush=True)
    if bigram(data[:2], data[2]) != reports[0]["payload"]["bigram"]:
        raise AssertionError("bigram recomputation differs")
    write_record(output, {"status": "SEQUENCE_CPU_REPLAY_PASS", "payload_sha256": reports[0]["payload_sha256"],
                          "checks": checks, "max_error": maximum, "official_readiness": 21})
    print("SEQUENCE_CPU_REPLAY_PASS", maximum, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary", type=Path)
    parser.add_argument("--rerun", type=Path)
    args = parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        if args.mode == "run":
            run(args.output)
        else:
            if args.primary is None or args.rerun is None:
                parser.error("verify needs --primary and --rerun")
            verify(args.primary, args.rerun, args.output)


if __name__ == "__main__":
    main()
