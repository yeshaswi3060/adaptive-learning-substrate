"""Frozen tiny-set fitting and zero-delay controls for the existing recurrence."""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from . import sequence_memory as memory
from .sequence_core import CONFIG, CpuSequence, read_record, write_record

ROOT = Path(__file__).resolve().parents[2]
PROCESS_TOKEN = hashlib.sha256(os.urandom(32)).hexdigest()
SPEC = {
    "version": "sequence-trainability-v1", "model_seeds": [17000, 17001],
    "tasks": ["copy3", "recall4"], "modes": ["local", "readout", "bptt"],
    "delays": [0, 16], "train_examples": 16, "test_examples": 64,
    "checkpoint_episodes": [512, 2048], "training_exact_fit_minimum": 0.95,
    "training_exact_reset_advantage_minimum": 0.50,
    "minimum_available_ram_bytes": 128 * 1024**2,
    "official_readiness_effect": 0,
}


def rng_for(seed: int, task: str, purpose: str) -> np.random.Generator:
    key = int.from_bytes(hashlib.sha256(f"trainability-v1|{seed}|{task}|{purpose}".encode()).digest()[:16], "little")
    return np.random.Generator(np.random.PCG64(key))


def balanced_examples(seed: int, task: str, split: str, count: int,
                      excluded: set[tuple[int, ...]]) -> list[dict[str, Any]]:
    """Each block balances answer values; delay variants share cue/query bytes."""
    if task not in ("copy3", "recall4") or type(count) is not int or not 8 <= count <= 128 or count % 8:
        raise ValueError("balanced examples require a known task and positive multiple of eight")
    rng = rng_for(seed, task, split)
    seen, examples = set(excluded), []
    for _ in range(count // 8):
        for _attempt in range(4096):
            values = np.stack([rng.permutation(np.arange(64, 72)) for _ in range(3)], axis=1).tolist()
            queries = rng.permutation([32, 33, 34, 35] * 2).tolist()
            block = []
            for index in range(8):
                if task == "copy3":
                    prefix, suffix, targets = [240] + values[index] + [242], [243], values[index]
                else:
                    keys = rng.permutation([32, 33, 34, 35]).tolist()
                    table = dict(zip(keys, rng.integers(64, 72, 4).tolist(), strict=True))
                    target = values[index][0]
                    table[queries[index]] = target
                    prefix = [241] + [item for pair in table.items() for item in pair] + [242]
                    suffix, targets = [243, queries[index]], [target]
                signature = tuple(prefix + suffix)
                block.append({"prefix": prefix, "suffix": suffix, "targets": targets,
                              "signature": signature})
            signatures = {row["signature"] for row in block}
            if len(signatures) == 8 and signatures.isdisjoint(seen):
                seen.update(signatures)
                examples.extend(block)
                break
        else:
            raise RuntimeError("balanced disjoint generator exhausted its fixed retry bound")
    return examples


def make_data(seed: int, task: str) -> dict[str, Any]:
    train = balanced_examples(seed, task, "train", SPEC["train_examples"], set())
    test = balanced_examples(seed, task, "test", SPEC["test_examples"], {row["signature"] for row in train})
    output: dict[str, Any] = {"seed": seed, "task": task}
    for split, examples in (("train", train), ("test", test)):
        noise_rng = rng_for(seed, task, split + "-noise")
        noise = noise_rng.integers(128, 144, (len(examples), max(SPEC["delays"]))).tolist()
        output[split] = {
            str(delay): [{"task": task, "delay": delay,
                          "prompt": row["prefix"] + noise[index][:delay] + row["suffix"],
                          "targets": row["targets"]}
                         for index, row in enumerate(examples)]
            for delay in SPEC["delays"]
        }
    order_rng = rng_for(seed, task, "training-order")
    maximum = max(SPEC["checkpoint_episodes"])
    if maximum % SPEC["train_examples"]:
        raise ValueError("training budget must contain complete balanced epochs")
    output["training_order"] = [int(i) for _ in range(maximum // SPEC["train_examples"])
                                for i in order_rng.permutation(SPEC["train_examples"])]
    for split in ("train", "test"):
        for rows in output[split].values():
            if any(memory.oracle(row["prompt"]) != row["targets"] for row in rows):
                raise AssertionError("trainability generator oracle differs")
    output["zero_delay_train_test_overlap"] = 0
    output["paired_delays"] = True
    return output


def source_bindings() -> dict[str, str]:
    paths = sorted((ROOT / "src/adaptive_learning_substrate").glob("*.py"))
    paths += [ROOT / name for name in (
        "docs/SEQUENCE_TRAINABILITY_V1.md", "docs/SEQUENCE_MEMORY_V1.md", "docs/SEQUENCE_CORE_V1.md",
        "tests/test_sequence_trainability.py", "tests/test_sequence_memory.py", "tests/test_sequence_core.py",
        "pyproject.toml", "requirements-dev.lock",
    )]
    paths.append(Path(sys.executable))
    return {str(path.resolve()): memory.file_digest(path) for path in paths}


def stable_sources(expected: dict[str, str]) -> None:
    if source_bindings() != expected:
        raise RuntimeError("trainability frozen sources changed")


def fitting_flags(normal: dict[str, Any], reset: dict[str, Any]) -> dict[str, Any]:
    return {"fits_training_examples": normal["exact_accuracy"] >= SPEC["training_exact_fit_minimum"],
            "training_exact_reset_advantage": normal["exact_accuracy"] - reset["exact_accuracy"],
            "training_memory_control_pass": normal["exact_accuracy"] - reset["exact_accuracy"]
            >= SPEC["training_exact_reset_advantage_minimum"]}


def measure(model: CpuSequence, dataset: dict[str, Any], train_delay: int) -> dict[str, Any]:
    training = memory.evaluate(model, dataset["train"][str(train_delay)])
    training_reset = memory.evaluate(model, dataset["train"][str(train_delay)], reset=True)
    test = {delay: {"normal": memory.evaluate(model, rows),
                    "reset": memory.evaluate(model, rows, reset=True)}
            for delay, rows in dataset["test"].items()}
    return {"training": training, "training_reset": training_reset, "test": test,
            "fitting": fitting_flags(training, training_reset)}


def cell_name(seed: int, task: str, delay: int, mode: str) -> str:
    return f"{seed}_{task}_delay{delay}_{mode}"


def fit_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for mode in SPEC["modes"]:
        for delay in SPEC["delays"]:
            for budget in SPEC["checkpoint_episodes"]:
                snapshots = [snapshot for row in results if row["mode"] == mode and row["train_delay"] == delay
                             for snapshot in row["snapshots"] if snapshot["episodes"] == budget]
                summary[f"{mode}_delay{delay}_episodes{budget}"] = {
                    "cells": len(snapshots),
                    "fit_cells": sum(s["metrics"]["fitting"]["fits_training_examples"] for s in snapshots),
                    "training_memory_control_pass_cells": sum(s["metrics"]["fitting"]["training_memory_control_pass"] for s in snapshots),
                }
    return summary


def run(output: Path) -> dict[str, Any]:
    output = output.resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        raise ValueError("trainability output must be under project artifacts")
    host = memory.headroom()
    bindings = source_bindings()
    output.mkdir(parents=True, exist_ok=False)
    data_files = {}
    for seed in SPEC["model_seeds"]:
        for task in SPEC["tasks"]:
            name = f"data_{seed}_{task}.json"
            data_files[name] = write_record(output / name, make_data(seed, task))
    freeze = {"schema_version": "sequence-trainability-freeze-v1", "spec": SPEC, "model": CONFIG,
              "sources": bindings, "data_files": data_files, "host_before": host,
              "environment": {"python": sys.version, "numpy": np.__version__},
              "process": {"pid": os.getpid(), "instance_token": PROCESS_TOKEN},
              "created_unix_ns": time.time_ns()}
    stable_sources(bindings)
    write_record(output / "FREEZE.json", freeze)
    stable_sources(bindings)
    freeze_sha256 = memory.file_digest(output / "FREEZE.json")

    def checkpoint_inputs() -> None:
        stable_sources(bindings)
        if read_record(output / "FREEZE.json") != freeze:
            raise RuntimeError("trainability prospective freeze changed")
        for name, expected in data_files.items():
            read_record(output / name)
            if memory.file_digest(output / name) != expected:
                raise RuntimeError("trainability frozen data changed at publication boundary")

    results, timings = [], []
    with threadpool_limits(limits=1):
        for seed in SPEC["model_seeds"]:
            for task in SPEC["tasks"]:
                name = f"data_{seed}_{task}.json"
                dataset = read_record(output / name)
                if memory.file_digest(output / name) != data_files[name]:
                    raise RuntimeError("trainability frozen dataset changed")
                for delay in SPEC["delays"]:
                    for mode in SPEC["modes"]:
                        checkpoint_inputs()
                        memory.headroom()
                        model = CpuSequence(seed, core_learning=mode != "readout")
                        trainer = memory.ExactTrainer(model) if mode == "bptt" else None
                        cell = cell_name(seed, task, delay, mode)
                        row = {"seed": seed, "task": task, "train_delay": delay, "mode": mode,
                               "initial_parameter_sha256": model.parameter_digest(), "snapshots": []}
                        losses, start = [], time.perf_counter()
                        try:
                            for episode_index, example_index in enumerate(dataset["training_order"], 1):
                                if (episode_index - 1) % 32 == 0:
                                    memory.headroom()
                                example = dataset["train"][str(delay)][example_index]
                                loss = trainer.train(example) if trainer else memory.train_local(model, example)
                                losses.append(loss)
                                if episode_index in SPEC["checkpoint_episodes"]:
                                    checkpoint_inputs()
                                    checkpoint = f"model_{cell}_episodes{episode_index}.json"
                                    metrics = measure(model, dataset, delay)
                                    model.reset()
                                    checkpoint_sha256 = model.save(output / checkpoint)
                                    snapshot = {"episodes": episode_index, "training_input_tokens": model.steps,
                                                "supervised_targets": model.updates,
                                                "optimizer_steps": trainer.iteration if trainer else model.updates,
                                                "train_losses": losses.copy(), "metrics": metrics,
                                                "parameter_sha256": model.parameter_digest(),
                                                "checkpoint": checkpoint, "checkpoint_sha256": checkpoint_sha256}
                                    row["snapshots"].append(snapshot)
                                    checkpoint_inputs()
                                    print(f"TRAINABILITY_CHECKPOINT cell={cell} episodes={episode_index} "
                                          f"training_exact={metrics['training']['exact_accuracy']:.4f}", flush=True)
                            results.append(row)
                            checkpoint_inputs()
                            write_record(output / f"result_{cell}.json", row)
                            checkpoint_inputs()
                            timings.append({"cell": cell, "seconds": time.perf_counter() - start,
                                            "memory": memory.memory_snapshot()})
                        finally:
                            model.close()
    payload = {"spec_sha256": memory.digest(SPEC), "data_files": data_files,
               "results": results, "fit_summary": fit_summary(results), "official_readiness_effect": 0}
    report = {"schema_version": "sequence-trainability-report-v1", "scientific_payload": payload,
              "scientific_payload_sha256": memory.digest(payload), "timings": timings,
              "freeze_sha256": freeze_sha256}
    checkpoint_inputs()
    write_record(output / "REPORT.json", report)
    checkpoint_inputs()
    print("TRAINABILITY_MATRIX_COMPLETE payload=" + memory.digest(payload), flush=True)
    return report


def original_evaluate(model: CpuSequence, examples: list[dict[str, Any]], *, reset: bool) -> dict[str, Any]:
    """Replay with the original stateful engine, not the diagnostic forward path."""
    before = model.parameter_digest()
    evidence = []
    for row in examples:
        model.reset()
        for token in row["prompt"][:-1]:
            model.step(token)
        if reset:
            model.reset()
        token, predictions = row["prompt"][-1], []
        for _ in row["targets"]:
            token = int(model.step(token).argmax())
            predictions.append(token)
        model.reset()
        for token in row["prompt"][:-1]:
            model.step(token)
        if reset:
            model.reset()
        token, probabilities = row["prompt"][-1], []
        for target in row["targets"]:
            p = model.step(token)
            probabilities.append(float(p[target]))
            token = target
        evidence.append({"prompt_sha256": memory.digest(row["prompt"]), "targets": row["targets"],
                         "free_predictions": predictions, "teacher_target_probabilities": probabilities})
    if model.parameter_digest() != before:
        raise AssertionError("original-engine evaluation changed parameters")
    return memory.summarize_evidence(evidence, reset=reset)


def verify(primary: Path, rerun: Path, output: Path) -> dict[str, Any]:
    evidence_files = {path: memory.file_digest(path) for folder in (primary, rerun)
                      for path in folder.iterdir() if path.is_file()}
    reports = [read_record(folder / "REPORT.json") for folder in (primary, rerun)]
    sources, tokens = source_bindings(), {PROCESS_TOKEN}
    expected_cells = {(seed, task, delay, mode) for seed in SPEC["model_seeds"]
                      for task in SPEC["tasks"] for delay in SPEC["delays"] for mode in SPEC["modes"]}
    data_by_key = {}
    for folder, report in zip((primary, rerun), reports, strict=True):
        freeze = read_record(folder / "FREEZE.json")
        if freeze["spec"] != SPEC or freeze["model"] != CONFIG or freeze["sources"] != sources:
            raise RuntimeError("trainability source/spec/model binding differs")
        token = freeze["process"]["instance_token"]
        if token in tokens:
            raise RuntimeError("trainability process instances overlap")
        tokens.add(token)
        if report["freeze_sha256"] != memory.file_digest(folder / "FREEZE.json"):
            raise RuntimeError("trainability freeze hash differs")
        payload = report["scientific_payload"]
        if (memory.digest(payload) != report["scientific_payload_sha256"]
                or payload != reports[0]["scientific_payload"]
                or payload["spec_sha256"] != memory.digest(SPEC)
                or payload["data_files"] != freeze["data_files"]):
            raise RuntimeError("trainability fresh replay/payload differs")
        expected_data = {f"data_{seed}_{task}.json" for seed in SPEC["model_seeds"] for task in SPEC["tasks"]}
        if set(freeze["data_files"]) != expected_data:
            raise RuntimeError("trainability data registry differs")
        for name, expected in freeze["data_files"].items():
            data = read_record(folder / name)
            if memory.file_digest(folder / name) != expected or data != make_data(data["seed"], data["task"]):
                raise RuntimeError("trainability regenerated dataset differs")
            data_by_key[data["seed"], data["task"]] = data
        rows = payload["results"]
        if len(rows) != len(expected_cells) or {(r["seed"], r["task"], r["train_delay"], r["mode"]) for r in rows} != expected_cells:
            raise RuntimeError("trainability condition registry differs")
        if payload["fit_summary"] != fit_summary(rows):
            raise RuntimeError("trainability fitting summary differs")
        for row in rows:
            cell = cell_name(row["seed"], row["task"], row["train_delay"], row["mode"])
            if read_record(folder / f"result_{cell}.json") != row:
                raise RuntimeError("trainability per-cell report differs")
            if [s["episodes"] for s in row["snapshots"]] != SPEC["checkpoint_episodes"]:
                raise RuntimeError("trainability checkpoint registry differs")
            for snapshot in row["snapshots"]:
                if memory.file_digest(folder / snapshot["checkpoint"]) != snapshot["checkpoint_sha256"]:
                    raise RuntimeError("trainability checkpoint hash differs")
                model = CpuSequence()
                model.restore(folder / snapshot["checkpoint"])
                if model.parameter_digest() != snapshot["parameter_sha256"]:
                    raise RuntimeError("trainability checkpoint parameter binding differs")
                model.close()
    checked = 0
    with threadpool_limits(limits=1):
        for row in reports[0]["scientific_payload"]["results"]:
            data = data_by_key[row["seed"], row["task"]]
            for snapshot in row["snapshots"]:
                model = CpuSequence()
                model.restore(primary / snapshot["checkpoint"])
                expected = snapshot["metrics"]
                for reset in (False, True):
                    memory.headroom()
                    actual = original_evaluate(model, data["train"][str(row["train_delay"])], reset=reset)
                    if actual != expected["training_reset" if reset else "training"]:
                        raise AssertionError("trainability independent training-set evaluation differs")
                    checked += len(data["train"][str(row["train_delay"])])
                    for delay in SPEC["delays"]:
                        actual = original_evaluate(model, data["test"][str(delay)], reset=reset)
                        if actual != expected["test"][str(delay)]["reset" if reset else "normal"]:
                            raise AssertionError("trainability independent held-out evaluation differs")
                        checked += len(data["test"][str(delay)])
                if fitting_flags(expected["training"], expected["training_reset"]) != expected["fitting"]:
                    raise RuntimeError("trainability fitting flag differs")
                model.close()
    stable_sources(sources)
    if {path: memory.file_digest(path) for folder in (primary, rerun)
            for path in folder.iterdir() if path.is_file()} != evidence_files:
        raise RuntimeError("trainability evidence files changed across verification")
    result = {"schema_version": "sequence-trainability-verification-v1", "status": "PASS",
              "scientific_payload_sha256": reports[0]["scientific_payload_sha256"],
              "distinct_process_instances": len(tokens), "fresh_training_exact": True,
              "original_engine_evaluation_episodes": checked, "original_engine_results_exact": True,
              "official_readiness_effect": 0}
    write_record(output, result)
    print(f"TRAINABILITY_VERIFICATION_PASS episodes={checked} exact=True", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--output", type=Path, required=True)
    verifier = commands.add_parser("verify")
    verifier.add_argument("--primary", type=Path, required=True)
    verifier.add_argument("--rerun", type=Path, required=True)
    verifier.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args.output)
    else:
        verify(args.primary, args.rerun, args.output)


if __name__ == "__main__":
    main()
