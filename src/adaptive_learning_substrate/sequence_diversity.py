"""Prospective matched-budget cue-diversity study of unchanged learning rules."""
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

from . import sequence_credit as credit
from . import sequence_memory as memory
from . import sequence_trainability as fitting
from .sequence_core import CONFIG, CpuSequence, canonical, read_record, write_record

ROOT = Path(__file__).resolve().parents[2]
PROCESS_TOKEN = hashlib.sha256(os.urandom(32)).hexdigest()
SPEC = {
    "version": "sequence-diversity-v1", "model_seeds": [19000, 19001],
    "tasks": ["copy3", "recall4"], "cue_counts": [16, 128],
    "modes": ["readout_adam", "feedback_adam", "diagonal_adam", "bptt_adam"],
    "heldout_examples": 64, "delay": 16, "checkpoint_episodes": [2048, 4096],
    "seen_fit_exact_minimum": 0.95, "heldout_token_minimum": 0.75,
    "heldout_copy_exact_minimum": 0.50, "heldout_reset_token_advantage_minimum": 0.15,
    "counterfactual_exact_minimum": 0.75, "paired_exact_minimum": 0.50,
    "paired_reset_advantage_minimum": 0.15,
    "minimum_available_ram_bytes": 128 * 1024**2, "official_readiness_effect": 0,
}


def rng_for(seed: int, task: str, purpose: str) -> np.random.Generator:
    key = int.from_bytes(hashlib.sha256(f"diversity-v1|{seed}|{task}|{purpose}".encode()).digest()[:16], "little")
    return np.random.Generator(np.random.PCG64(key))


def episode(cue: dict[str, Any], noise: list[int], task: str) -> dict[str, Any]:
    return {"task": task, "delay": len(noise), "prompt": cue["prefix"] + noise + cue["suffix"],
            "targets": cue["targets"][:]}


def intervene(row: dict[str, Any]) -> dict[str, Any]:
    prompt = row["prompt"][:]
    indices = range(1, 4) if row["task"] == "copy3" else [next(i + 1 for i in range(1, 9, 2) if prompt[i] == prompt[-1])]
    for i in indices:
        prompt[i] = 64 + (prompt[i] - 64 + 1) % 8
    return {**row, "prompt": prompt, "targets": memory.oracle(prompt)}


def training_episode(data: dict[str, Any], count: int, index: int) -> dict[str, Any]:
    if type(count) is not int or count not in SPEC["cue_counts"]:
        raise ValueError("unregistered cue count")
    if type(index) is not int or not 0 <= index < len(data["training_order"]):
        raise ValueError("training presentation index out of range")
    cue = data["cue_pool"][data["training_order"][index] % count]
    return episode(cue, data["training_noise"][index], data["task"])


def stream_digest(data: dict[str, Any], count: int) -> str:
    digest = hashlib.sha256()
    for i in range(len(data["training_order"])):
        digest.update(canonical(training_episode(data, count, i)) + b"\n")
    return digest.hexdigest()


def make_data(seed: int, task: str) -> dict[str, Any]:
    heldout = fitting.balanced_examples(seed, task, "diversity-heldout", SPEC["heldout_examples"], set())
    zero = [episode(cue, [], task) for cue in heldout]
    excluded = {tuple(row["prompt"]) for row in zero + [intervene(row) for row in zero]}
    maximum = max(SPEC["cue_counts"])
    pool = fitting.balanced_examples(seed, task, "diversity-train", maximum, excluded)
    pool = [{k: row[k][:] for k in ("prefix", "suffix", "targets")} for row in pool]
    budget = max(SPEC["checkpoint_episodes"])
    if budget % maximum or any(maximum % c for c in SPEC["cue_counts"]):
        raise ValueError("diversity budgets require balanced common presentation blocks")
    order_rng = rng_for(seed, task, "order")
    order = [int(i) for _ in range(budget // maximum) for i in order_rng.permutation(maximum)]
    noise_rng, used = rng_for(seed, task, "noise"), set()

    def noise() -> list[int]:
        for _ in range(4096):
            value = tuple(noise_rng.integers(128, 144, SPEC["delay"]).tolist())
            if value not in used:
                used.add(value)
                return list(value)
        raise RuntimeError("unique distractor generator exhausted")

    train_noise = [noise() for _ in order]
    seen = [episode(cue, noise(), task) for cue in pool]
    novel = [episode(cue, noise(), task) for cue in heldout]
    changed = [intervene(row) for row in novel]
    data = {"seed": seed, "task": task, "cue_pool": pool, "training_order": order,
            "training_noise": train_noise, "all_seen": seen,
            "heldout0": zero, "heldout16": novel, "counterfactual16": changed}
    data["training_stream_sha256"] = {str(c): stream_digest(data, c) for c in SPEC["cue_counts"]}
    for row in seen + zero + novel + changed:
        if memory.oracle(row["prompt"]) != row["targets"]:
            raise AssertionError("diversity task oracle differs")
    return data


def evaluation_sets(data: dict[str, Any], count: int) -> dict[str, list[dict[str, Any]]]:
    if type(count) is not int or count not in SPEC["cue_counts"]:
        raise ValueError("unregistered evaluation cue count")
    return {"common_seen": data["all_seen"][:min(SPEC["cue_counts"])], "all_seen": data["all_seen"][:count],
            **{k: data[k] for k in ("heldout0", "heldout16", "counterfactual16")}}


def paired_accuracy(first: dict[str, Any], changed: dict[str, Any]) -> float:
    a, b = first["rows"], changed["rows"]
    if not a or len(a) != len(b):
        raise ValueError("paired evaluation requires equal nonempty evidence")
    return sum(x["free_predictions"] == x["targets"] and y["free_predictions"] == y["targets"]
               for x, y in zip(a, b, strict=True)) / len(a)


def flags(scores: dict[str, Any], paired: dict[str, float], task: str) -> dict[str, bool]:
    normal, reset = scores["heldout16"]["normal"], scores["heldout16"]["reset"]
    return {"common_seen_fit": scores["common_seen"]["normal"]["exact_accuracy"] >= SPEC["seen_fit_exact_minimum"],
            "all_seen_fit": scores["all_seen"]["normal"]["exact_accuracy"] >= SPEC["seen_fit_exact_minimum"],
            "heldout_memory": normal["token_accuracy"] >= SPEC["heldout_token_minimum"]
            and normal["token_accuracy"] - reset["token_accuracy"] >= SPEC["heldout_reset_token_advantage_minimum"]
            and (task != "copy3" or normal["exact_accuracy"] >= SPEC["heldout_copy_exact_minimum"]),
            "counterfactual_response": scores["counterfactual16"]["normal"]["exact_accuracy"] >= SPEC["counterfactual_exact_minimum"]
            and paired["normal"] >= SPEC["paired_exact_minimum"]
            and paired["normal"] - paired["reset"] >= SPEC["paired_reset_advantage_minimum"]}


def measure(model: CpuSequence, data: dict[str, Any], count: int, *, original: bool = False) -> dict[str, Any]:
    evaluator = fitting.original_evaluate if original else memory.evaluate
    scores = {}
    for name, rows in evaluation_sets(data, count).items():
        memory.headroom()
        scores[name] = {"normal": evaluator(model, rows, reset=False), "reset": evaluator(model, rows, reset=True)}
    paired = {control: paired_accuracy(scores["heldout16"][control], scores["counterfactual16"][control])
              for control in ("normal", "reset")}
    return {"scores": scores, "paired_exact": paired, "flags": flags(scores, paired, data["task"])}


def source_bindings() -> dict[str, str]:
    paths = sorted((ROOT / "src/adaptive_learning_substrate").glob("*.py"))
    paths += [ROOT / name for name in ("docs/SEQUENCE_DIVERSITY_V1.md", "docs/SEQUENCE_CREDIT_V1.md",
              "docs/SEQUENCE_TRAINABILITY_V1.md", "docs/SEQUENCE_MEMORY_V1.md", "docs/SEQUENCE_CORE_V1.md",
              "tests/test_sequence_diversity.py", "tests/test_sequence_credit.py", "tests/test_sequence_trainability.py",
              "tests/test_sequence_memory.py", "tests/test_sequence_core.py", "pyproject.toml", "requirements-dev.lock")]
    paths.append(Path(sys.executable))
    return {str(path.resolve()): memory.file_digest(path) for path in paths}


def stable_sources(expected: dict[str, str]) -> None:
    if source_bindings() != expected:
        raise RuntimeError("diversity frozen sources changed")


def cell_name(seed: int, task: str, diversity: int, mode: str) -> str:
    return f"{seed}_{task}_cues{diversity}_{mode}"


def run(output: Path) -> dict[str, Any]:
    output = output.resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        raise ValueError("diversity output must be under project artifacts")
    host, sources = memory.headroom(), source_bindings()
    output.mkdir(parents=True, exist_ok=False)
    data_files = {f"data_{s}_{t}.json": write_record(output / f"data_{s}_{t}.json", make_data(s, t))
                  for s in SPEC["model_seeds"] for t in SPEC["tasks"]}
    freeze = {"schema_version": "sequence-diversity-freeze-v1", "spec": SPEC, "model": CONFIG,
              "learning_spec": credit.SPEC,
              "sources": sources, "data_files": data_files, "host_before": host,
              "environment": {"python": sys.version, "numpy": np.__version__},
              "process": {"pid": os.getpid(), "instance_token": PROCESS_TOKEN}, "created_unix_ns": time.time_ns()}
    stable_sources(sources)
    freeze_sha = write_record(output / "FREEZE.json", freeze)

    def checkpoint_inputs() -> None:
        stable_sources(sources)
        if read_record(output / "FREEZE.json") != freeze:
            raise RuntimeError("diversity prospective freeze changed")
        for name, sha in data_files.items():
            read_record(output / name)
            if memory.file_digest(output / name) != sha:
                raise RuntimeError("diversity frozen data changed")

    results, timings = [], []
    with threadpool_limits(limits=1):
        for seed in SPEC["model_seeds"]:
            for task in SPEC["tasks"]:
                data = read_record(output / f"data_{seed}_{task}.json")
                for diversity in SPEC["cue_counts"]:
                    for mode in SPEC["modes"]:
                        checkpoint_inputs()
                        model = CpuSequence(seed, core_learning=mode != "readout_adam")
                        trainer = credit.CreditTrainer(model, mode)
                        cell = cell_name(seed, task, diversity, mode)
                        row = {"seed": seed, "task": task, "diversity": diversity, "mode": mode,
                               "initial_parameter_sha256": model.parameter_digest(), "snapshots": []}
                        start, losses = time.perf_counter(), []
                        try:
                            for i, index in enumerate(data["training_order"], 1):
                                if (i - 1) % 32 == 0:
                                    memory.headroom()
                                example = training_episode(data, diversity, i - 1)
                                losses.append(trainer.train(example))
                                if i in SPEC["checkpoint_episodes"]:
                                    checkpoint_inputs()
                                    metrics = measure(model, data, diversity)
                                    model.reset()
                                    checkpoint = f"model_{cell}_episodes{i}.json"
                                    snapshot = {"episodes": i, "training_input_tokens": model.steps,
                                                "supervised_targets": model.updates,
                                                "optimizer_steps": trainer.iteration,
                                                "train_losses": losses.copy(), "metrics": metrics,
                                                "parameter_sha256": model.parameter_digest(), "checkpoint": checkpoint,
                                                "checkpoint_sha256": model.save(output / checkpoint)}
                                    row["snapshots"].append(snapshot)
                                    checkpoint_inputs()
                                    print(f"DIVERSITY_CHECKPOINT cell={cell} episodes={i} "
                                          f"seen_exact={metrics['scores']['all_seen']['normal']['exact_accuracy']:.4f} "
                                          f"heldout_exact={metrics['scores']['heldout16']['normal']['exact_accuracy']:.4f}", flush=True)
                            results.append(row)
                            write_record(output / f"result_{cell}.json", row)
                            checkpoint_inputs()
                            timings.append({"cell": cell, "seconds": time.perf_counter() - start,
                                            "memory": memory.memory_snapshot()})
                        finally:
                            model.close()
    payload = {"spec_sha256": memory.digest(SPEC), "data_files": data_files,
               "results": results, "official_readiness_effect": 0}
    report = {"schema_version": "sequence-diversity-report-v1", "scientific_payload": payload,
              "scientific_payload_sha256": memory.digest(payload), "timings": timings, "freeze_sha256": freeze_sha}
    checkpoint_inputs()
    write_record(output / "REPORT.json", report)
    checkpoint_inputs()
    print("DIVERSITY_MATRIX_COMPLETE payload=" + memory.digest(payload), flush=True)
    return report


def verify(primary: Path, rerun: Path, output: Path) -> dict[str, Any]:
    evidence_files = {p: memory.file_digest(p) for folder in (primary, rerun) for p in folder.iterdir() if p.is_file()}
    reports = [read_record(folder / "REPORT.json") for folder in (primary, rerun)]
    sources, tokens, datasets = source_bindings(), {PROCESS_TOKEN}, {}
    expected_cells = {(s, t, r, m) for s in SPEC["model_seeds"] for t in SPEC["tasks"]
                      for r in SPEC["cue_counts"] for m in SPEC["modes"]}
    for folder, report in zip((primary, rerun), reports, strict=True):
        freeze = read_record(folder / "FREEZE.json")
        if (freeze["spec"] != SPEC or freeze["model"] != CONFIG or freeze["sources"] != sources
                or freeze["learning_spec"] != credit.SPEC):
            raise RuntimeError("diversity source/spec/model binding differs")
        token = freeze["process"]["instance_token"]
        if token in tokens:
            raise RuntimeError("diversity process instances overlap")
        tokens.add(token)
        if report["freeze_sha256"] != memory.file_digest(folder / "FREEZE.json"):
            raise RuntimeError("diversity freeze hash differs")
        payload = report["scientific_payload"]
        if (memory.digest(payload) != report["scientific_payload_sha256"] or payload != reports[0]["scientific_payload"]
                or payload["spec_sha256"] != memory.digest(SPEC) or payload["data_files"] != freeze["data_files"]):
            raise RuntimeError("diversity fresh replay/payload differs")
        if set(freeze["data_files"]) != {f"data_{s}_{t}.json" for s in SPEC["model_seeds"] for t in SPEC["tasks"]}:
            raise RuntimeError("diversity data registry differs")
        for name, sha in freeze["data_files"].items():
            data = read_record(folder / name)
            if memory.file_digest(folder / name) != sha or data != make_data(data["seed"], data["task"]):
                raise RuntimeError("diversity regenerated dataset differs")
            datasets[data["seed"], data["task"]] = data
        rows = payload["results"]
        if len(rows) != len(expected_cells) or {(r["seed"], r["task"], r["diversity"], r["mode"]) for r in rows} != expected_cells:
            raise RuntimeError("diversity condition registry differs")
        for row in rows:
            cell = cell_name(row["seed"], row["task"], row["diversity"], row["mode"])
            if read_record(folder / f"result_{cell}.json") != row:
                raise RuntimeError("diversity per-cell report differs")
            if [s["episodes"] for s in row["snapshots"]] != SPEC["checkpoint_episodes"]:
                raise RuntimeError("diversity checkpoint registry differs")
            for snapshot in row["snapshots"]:
                if memory.file_digest(folder / snapshot["checkpoint"]) != snapshot["checkpoint_sha256"]:
                    raise RuntimeError("diversity checkpoint hash differs")
                model = CpuSequence()
                model.restore(folder / snapshot["checkpoint"])
                if model.parameter_digest() != snapshot["parameter_sha256"]:
                    raise RuntimeError("diversity checkpoint parameter binding differs")
                model.close()
    checked = 0
    with threadpool_limits(limits=1):
        for row in reports[0]["scientific_payload"]["results"]:
            data = datasets[row["seed"], row["task"]]
            for snapshot in row["snapshots"]:
                model = CpuSequence()
                model.restore(primary / snapshot["checkpoint"])
                if measure(model, data, row["diversity"], original=True) != snapshot["metrics"]:
                    raise AssertionError("diversity original-engine scores/flags differ")
                checked += 2 * sum(len(rows) for rows in evaluation_sets(data, row["diversity"]).values())
                model.close()
    stable_sources(sources)
    if {p: memory.file_digest(p) for folder in (primary, rerun) for p in folder.iterdir() if p.is_file()} != evidence_files:
        raise RuntimeError("diversity evidence changed during verification")
    result = {"schema_version": "sequence-diversity-verification-v1", "status": "PASS",
              "scientific_payload_sha256": reports[0]["scientific_payload_sha256"],
              "distinct_process_instances": len(tokens), "fresh_training_exact": True,
              "original_engine_evaluation_episodes": checked, "original_engine_results_exact": True,
              "official_readiness_effect": 0}
    write_record(output, result)
    print(f"DIVERSITY_VERIFICATION_PASS episodes={checked} exact=True", flush=True)
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
