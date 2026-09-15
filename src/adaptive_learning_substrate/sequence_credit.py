"""Output-aligned local eligibility, matched-Adam controls, and cue interventions.

The diagonal rule streams eligibility without a reverse-time tape. It omits
cross-unit temporal credit and is not exact BPTT for the connected recurrence.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from . import sequence_memory as memory
from . import sequence_trainability as fitting
from .sequence_core import (
    CONFIG,
    CORE,
    PARAMETERS,
    CpuSequence,
    H,
    read_record,
    token_check,
    views,
    write_record,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESS_TOKEN = hashlib.sha256(os.urandom(32)).hexdigest()
SPEC = {
    "version": "sequence-credit-v1", "model_seeds": [18000, 18001],
    "tasks": ["copy3", "recall4"], "regimes": ["fixed", "fresh"],
    "modes": ["local_sgd", "readout_adam", "feedback_adam", "diagonal_adam", "bptt_adam"],
    "train_examples": 16, "heldout_examples": 64, "fresh_noise_replicates": 4,
    "delay": 16, "checkpoint_episodes": [512, 2048],
    "anchor_fit_exact_minimum": 0.95, "noise_transfer_exact_minimum": 0.75,
    "noise_transfer_reset_exact_advantage_minimum": 0.50, "changed_cue_exact_minimum": 0.75,
    "heldout_token_minimum": 0.75, "heldout_copy_exact_minimum": 0.50,
    "heldout_reset_token_advantage_minimum": 0.15,
    "adam_rate": 0.003, "adam_beta1": 0.9, "adam_beta2": 0.999,
    "adam_epsilon": 1e-8, "gradient_norm_clip": 1.0, "weight_clip": 3.0,
    "minimum_available_ram_bytes": 128 * 1024**2, "official_readiness_effect": 0,
}


def streaming_gradient(buffer: np.ndarray, parents: np.ndarray, tokens: list[int],
                       labels: list[int], mode: str) -> tuple[float, dict[str, np.ndarray]]:
    """Frozen-episode gradient approximation; memory use does not grow with time."""
    if mode not in ("feedback_adam", "diagonal_adam", "readout_adam"):
        raise ValueError("unknown streaming credit mode")
    if not tokens or len(tokens) != len(labels) or len(tokens) > 256:
        raise ValueError("invalid bounded supervised sequence")
    for token in tokens:
        token_check(token)
    for target in labels:
        if target != -1:
            token_check(target)
        elif type(target) is not int:
            raise ValueError("invalid ignored target")
    count = sum(target != -1 for target in labels)
    if count == 0:
        raise ValueError("sequence has no supervised prediction")
    x = views(buffer)
    gradient = {name: np.zeros_like(x[name]) for name in PARAMETERS}
    traces = {name: np.zeros_like(x[name]) for name in CORE} if mode != "readout_adam" else {}
    state, loss = np.zeros(H), 0.0
    for token, target in zip(tokens, labels, strict=True):
        previous = state
        state, probability, c, g = memory.forward_one(x, parents, previous, token)
        if traces:
            dc, dg = (1 - g) * (1 - c * c), (previous - c) * g * (1 - g)
            diagonal = g + dg * x["a"]  # CpuSequence forbids self-parent edges.
            for name, trace in traces.items():
                trace *= diagonal.reshape((H,) + (1,) * (trace.ndim - 1))
            traces["w"][:, token] += dc
            traces["g"][:, token] += dg
            traces["r"] += dc[:, None] * previous[parents]
            traces["a"] += dg * previous
            traces["bc"] += dc
            traces["bg"] += dg
        if target != -1:
            loss -= math.log(max(float(probability[target]), 1e-300)) / count
            error = probability.copy()
            error[target] -= 1
            error /= count
            gradient["o"] += np.outer(error, state)
            gradient["bo"] += error
            if traces:
                signal = x["b"] @ error if mode == "feedback_adam" else x["o"].T @ error
                for name, trace in traces.items():
                    gradient[name] += signal.reshape((H,) + (1,) * (trace.ndim - 1)) * trace
    if not math.isfinite(loss) or any(not np.isfinite(g).all() for g in gradient.values()):
        raise FloatingPointError("nonfinite streaming credit result")
    return loss, gradient


class CreditTrainer(memory.ExactTrainer):
    """Same Adam, clipping, loss reduction and update timing in all Adam modes."""

    def __init__(self, model: CpuSequence, mode: str):
        if mode not in SPEC["modes"] or mode == "local_sgd":
            raise ValueError("CreditTrainer requires a registered Adam mode")
        super().__init__(model)
        self.mode = mode

    def train(self, row: dict[str, Any]) -> float:
        model = self.model
        model._open()
        if model.pending:
            raise RuntimeError("credit training requires no pending prediction")
        tokens, labels = memory.supervised_tokens(row)
        if self.mode == "bptt_adam":
            loss, gradients, _ = memory.sequence_loss_gradient(model._buffer, model._parents, tokens, labels)
        else:
            loss, gradients = streaming_gradient(model._buffer, model._parents, tokens, labels, self.mode)
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
        scale = min(1.0, SPEC["gradient_norm_clip"] / max(norm, 1e-300))
        self.iteration += 1
        x = views(model._buffer)
        for name, gradient in gradients.items():
            gradient = gradient * scale
            self.m[name] = .9 * self.m[name] + .1 * gradient
            self.v[name] = .999 * self.v[name] + .001 * gradient * gradient
            delta = SPEC["adam_rate"] * (self.m[name] / (1 - .9**self.iteration)) / (
                np.sqrt(self.v[name] / (1 - .999**self.iteration)) + SPEC["adam_epsilon"])
            x[name][:] = np.clip(x[name] - delta, -3, 3)
        model.steps += len(tokens)
        model.updates += len(row["targets"])
        model.reset()
        return loss


def rng_for(seed: int, task: str, purpose: str) -> np.random.Generator:
    key = int.from_bytes(hashlib.sha256(f"credit-v1|{seed}|{task}|{purpose}".encode()).digest()[:16], "little")
    return np.random.Generator(np.random.PCG64(key))


def make_data(seed: int, task: str) -> dict[str, Any]:
    train = fitting.balanced_examples(seed, task, "credit-train", SPEC["train_examples"], set())
    heldout = fitting.balanced_examples(seed, task, "credit-test", SPEC["heldout_examples"],
                                        {r["signature"] for r in train})
    rng, used = rng_for(seed, task, "noise"), set()

    def noise() -> list[int]:
        for _ in range(4096):
            value = tuple(rng.integers(128, 144, SPEC["delay"]).tolist())
            if value not in used:
                used.add(value)
                return list(value)
        raise RuntimeError("unique distractor generator exhausted")

    def episode(row: dict[str, Any], distractors: list[int]) -> dict[str, Any]:
        return {"task": task, "delay": len(distractors),
                "prompt": row["prefix"] + distractors + row["suffix"], "targets": row["targets"][:]}

    anchor = [episode(r, noise()) for r in train]
    order_rng = rng_for(seed, task, "order")
    maximum = max(SPEC["checkpoint_episodes"])
    if maximum % len(train):
        raise ValueError("credit budgets require complete balanced epochs")
    order = [int(i) for _ in range(maximum // len(train)) for i in order_rng.permutation(len(train))]
    fresh_train = [episode(train[index], noise()) for index in order]
    fresh_test = [episode(r, noise()) for r in train for _ in range(SPEC["fresh_noise_replicates"])]
    changed = []
    for row in anchor:
        prompt = row["prompt"][:]
        indices = range(1, 4) if task == "copy3" else [next(i + 1 for i in range(1, 9, 2) if prompt[i] == prompt[-1])]
        for index in indices:
            prompt[index] = 64 + (prompt[index] - 64 + 1) % 8
        changed.append({**row, "prompt": prompt, "targets": memory.oracle(prompt)})
    evaluation = {"anchor_cues": anchor, "fresh_noise": fresh_test, "changed_cue": changed,
                  "heldout0": [episode(r, []) for r in heldout],
                  "heldout16": [episode(r, noise()) for r in heldout]}
    for rows in [fresh_train, *evaluation.values()]:
        if any(memory.oracle(row["prompt"]) != row["targets"] for row in rows):
            raise AssertionError("credit task oracle differs")
    return {"seed": seed, "task": task, "training_order": order,
            "anchor": anchor, "fresh_train": fresh_train, "evaluation": evaluation}


def flags(metrics: dict[str, Any], task: str) -> dict[str, bool]:
    anchor, noise, changed, heldout = [metrics[k] for k in ("anchor_cues", "fresh_noise", "changed_cue", "heldout16")]
    return {
        "anchor_fit": anchor["normal"]["exact_accuracy"] >= SPEC["anchor_fit_exact_minimum"],
        "noise_transfer": noise["normal"]["exact_accuracy"] >= SPEC["noise_transfer_exact_minimum"]
        and noise["normal"]["exact_accuracy"] - noise["reset"]["exact_accuracy"] >= SPEC["noise_transfer_reset_exact_advantage_minimum"],
        "changed_cue_response": changed["normal"]["exact_accuracy"] >= SPEC["changed_cue_exact_minimum"],
        "heldout_memory": heldout["normal"]["token_accuracy"] >= SPEC["heldout_token_minimum"]
        and heldout["normal"]["token_accuracy"] - heldout["reset"]["token_accuracy"] >= SPEC["heldout_reset_token_advantage_minimum"]
        and (task != "copy3" or heldout["normal"]["exact_accuracy"] >= SPEC["heldout_copy_exact_minimum"]),
    }


def measure(model: CpuSequence, data: dict[str, Any], *, original: bool = False) -> dict[str, Any]:
    evaluator = fitting.original_evaluate if original else memory.evaluate
    metrics = {}
    for name, rows in data["evaluation"].items():
        memory.headroom()
        metrics[name] = {"normal": evaluator(model, rows, reset=False), "reset": evaluator(model, rows, reset=True)}
    return {"scores": metrics, "flags": flags(metrics, data["task"])}


def source_bindings() -> dict[str, str]:
    paths = sorted((ROOT / "src/adaptive_learning_substrate").glob("*.py"))
    paths += [ROOT / name for name in ("docs/SEQUENCE_CREDIT_V1.md", "docs/SEQUENCE_TRAINABILITY_V1.md",
              "docs/SEQUENCE_MEMORY_V1.md", "docs/SEQUENCE_CORE_V1.md", "tests/test_sequence_credit.py",
              "tests/test_sequence_trainability.py", "tests/test_sequence_memory.py", "tests/test_sequence_core.py",
              "pyproject.toml", "requirements-dev.lock")]
    paths.append(Path(sys.executable))
    return {str(path.resolve()): memory.file_digest(path) for path in paths}


def stable_sources(expected: dict[str, str]) -> None:
    if source_bindings() != expected:
        raise RuntimeError("credit frozen sources changed")


def cell_name(seed: int, task: str, regime: str, mode: str) -> str:
    return f"{seed}_{task}_{regime}_{mode}"


def run(output: Path) -> dict[str, Any]:
    output = output.resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        raise ValueError("credit output must be under project artifacts")
    host, sources = memory.headroom(), source_bindings()
    output.mkdir(parents=True, exist_ok=False)
    data_files = {f"data_{s}_{t}.json": write_record(output / f"data_{s}_{t}.json", make_data(s, t))
                  for s in SPEC["model_seeds"] for t in SPEC["tasks"]}
    freeze = {"schema_version": "sequence-credit-freeze-v1", "spec": SPEC, "model": CONFIG,
              "sources": sources, "data_files": data_files, "host_before": host,
              "environment": {"python": sys.version, "numpy": np.__version__},
              "process": {"pid": os.getpid(), "instance_token": PROCESS_TOKEN}, "created_unix_ns": time.time_ns()}
    stable_sources(sources)
    freeze_sha = write_record(output / "FREEZE.json", freeze)

    def checkpoint_inputs() -> None:
        stable_sources(sources)
        if read_record(output / "FREEZE.json") != freeze:
            raise RuntimeError("credit prospective freeze changed")
        for name, sha in data_files.items():
            read_record(output / name)
            if memory.file_digest(output / name) != sha:
                raise RuntimeError("credit frozen data changed")

    results, timings = [], []
    with threadpool_limits(limits=1):
        for seed in SPEC["model_seeds"]:
            for task in SPEC["tasks"]:
                data = read_record(output / f"data_{seed}_{task}.json")
                for regime in SPEC["regimes"]:
                    for mode in SPEC["modes"]:
                        checkpoint_inputs()
                        model = CpuSequence(seed, core_learning=mode != "readout_adam")
                        trainer = None if mode == "local_sgd" else CreditTrainer(model, mode)
                        cell = cell_name(seed, task, regime, mode)
                        row = {"seed": seed, "task": task, "regime": regime, "mode": mode,
                               "initial_parameter_sha256": model.parameter_digest(), "snapshots": []}
                        start, losses = time.perf_counter(), []
                        try:
                            for i, index in enumerate(data["training_order"], 1):
                                if (i - 1) % 32 == 0:
                                    memory.headroom()
                                example = data["anchor"][index] if regime == "fixed" else data["fresh_train"][i - 1]
                                losses.append(trainer.train(example) if trainer else memory.train_local(model, example))
                                if i in SPEC["checkpoint_episodes"]:
                                    checkpoint_inputs()
                                    metrics = measure(model, data)
                                    model.reset()
                                    checkpoint = f"model_{cell}_episodes{i}.json"
                                    snapshot = {"episodes": i, "training_input_tokens": model.steps,
                                                "supervised_targets": model.updates,
                                                "optimizer_steps": trainer.iteration if trainer else model.updates,
                                                "train_losses": losses.copy(), "metrics": metrics,
                                                "parameter_sha256": model.parameter_digest(), "checkpoint": checkpoint,
                                                "checkpoint_sha256": model.save(output / checkpoint)}
                                    row["snapshots"].append(snapshot)
                                    checkpoint_inputs()
                                    print(f"CREDIT_CHECKPOINT cell={cell} episodes={i} "
                                          f"anchor_exact={metrics['scores']['anchor_cues']['normal']['exact_accuracy']:.4f} "
                                          f"fresh_exact={metrics['scores']['fresh_noise']['normal']['exact_accuracy']:.4f}", flush=True)
                            results.append(row)
                            write_record(output / f"result_{cell}.json", row)
                            checkpoint_inputs()
                            timings.append({"cell": cell, "seconds": time.perf_counter() - start,
                                            "memory": memory.memory_snapshot()})
                        finally:
                            model.close()
    payload = {"spec_sha256": memory.digest(SPEC), "data_files": data_files,
               "results": results, "official_readiness_effect": 0}
    report = {"schema_version": "sequence-credit-report-v1", "scientific_payload": payload,
              "scientific_payload_sha256": memory.digest(payload), "timings": timings, "freeze_sha256": freeze_sha}
    checkpoint_inputs()
    write_record(output / "REPORT.json", report)
    checkpoint_inputs()
    print("CREDIT_MATRIX_COMPLETE payload=" + memory.digest(payload), flush=True)
    return report


def verify(primary: Path, rerun: Path, output: Path) -> dict[str, Any]:
    evidence_files = {p: memory.file_digest(p) for folder in (primary, rerun) for p in folder.iterdir() if p.is_file()}
    reports = [read_record(folder / "REPORT.json") for folder in (primary, rerun)]
    sources, tokens, datasets = source_bindings(), {PROCESS_TOKEN}, {}
    expected_cells = {(s, t, r, m) for s in SPEC["model_seeds"] for t in SPEC["tasks"]
                      for r in SPEC["regimes"] for m in SPEC["modes"]}
    for folder, report in zip((primary, rerun), reports, strict=True):
        freeze = read_record(folder / "FREEZE.json")
        if freeze["spec"] != SPEC or freeze["model"] != CONFIG or freeze["sources"] != sources:
            raise RuntimeError("credit source/spec/model binding differs")
        token = freeze["process"]["instance_token"]
        if token in tokens:
            raise RuntimeError("credit process instances overlap")
        tokens.add(token)
        if report["freeze_sha256"] != memory.file_digest(folder / "FREEZE.json"):
            raise RuntimeError("credit freeze hash differs")
        payload = report["scientific_payload"]
        if (memory.digest(payload) != report["scientific_payload_sha256"] or payload != reports[0]["scientific_payload"]
                or payload["spec_sha256"] != memory.digest(SPEC) or payload["data_files"] != freeze["data_files"]):
            raise RuntimeError("credit fresh replay/payload differs")
        if set(freeze["data_files"]) != {f"data_{s}_{t}.json" for s in SPEC["model_seeds"] for t in SPEC["tasks"]}:
            raise RuntimeError("credit data registry differs")
        for name, sha in freeze["data_files"].items():
            data = read_record(folder / name)
            if memory.file_digest(folder / name) != sha or data != make_data(data["seed"], data["task"]):
                raise RuntimeError("credit regenerated dataset differs")
            datasets[data["seed"], data["task"]] = data
        rows = payload["results"]
        if len(rows) != len(expected_cells) or {(r["seed"], r["task"], r["regime"], r["mode"]) for r in rows} != expected_cells:
            raise RuntimeError("credit condition registry differs")
        for row in rows:
            cell = cell_name(row["seed"], row["task"], row["regime"], row["mode"])
            if read_record(folder / f"result_{cell}.json") != row:
                raise RuntimeError("credit per-cell report differs")
            if [s["episodes"] for s in row["snapshots"]] != SPEC["checkpoint_episodes"]:
                raise RuntimeError("credit checkpoint registry differs")
            for snapshot in row["snapshots"]:
                if memory.file_digest(folder / snapshot["checkpoint"]) != snapshot["checkpoint_sha256"]:
                    raise RuntimeError("credit checkpoint hash differs")
                model = CpuSequence()
                model.restore(folder / snapshot["checkpoint"])
                if model.parameter_digest() != snapshot["parameter_sha256"]:
                    raise RuntimeError("credit checkpoint parameter binding differs")
                model.close()
    checked = 0
    with threadpool_limits(limits=1):
        for row in reports[0]["scientific_payload"]["results"]:
            data = datasets[row["seed"], row["task"]]
            for snapshot in row["snapshots"]:
                model = CpuSequence()
                model.restore(primary / snapshot["checkpoint"])
                if measure(model, data, original=True) != snapshot["metrics"]:
                    raise AssertionError("credit original-engine scores/flags differ")
                checked += 2 * sum(len(rows) for rows in data["evaluation"].values())
                model.close()
    stable_sources(sources)
    if {p: memory.file_digest(p) for folder in (primary, rerun) for p in folder.iterdir() if p.is_file()} != evidence_files:
        raise RuntimeError("credit evidence changed during verification")
    result = {"schema_version": "sequence-credit-verification-v1", "status": "PASS",
              "scientific_payload_sha256": reports[0]["scientific_payload_sha256"],
              "distinct_process_instances": len(tokens), "fresh_training_exact": True,
              "original_engine_evaluation_episodes": checked, "original_engine_results_exact": True,
              "official_readiness_effect": 0}
    write_record(output, result)
    print(f"CREDIT_VERIFICATION_PASS episodes={checked} exact=True", flush=True)
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
