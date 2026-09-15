"""Prospective multi-token memory diagnostics and exact-gradient reference.

The original recurrent architecture/local learning rule is unchanged. BPTT is a
comparison condition, not the candidate's learning rule or a novelty claim.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from .sequence_core import (
    CONFIG,
    PARAMETERS,
    CpuSequence,
    H,
    V,
    canonical,
    read_record,
    token_check,
    views,
    write_record,
)

PROCESS_TOKEN = hashlib.sha256(os.urandom(32)).hexdigest()
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "docs/SEQUENCE_MEMORY_V1.md"
SPEC = {
    "version": "sequence-memory-diagnostic-v1", "model_seeds": [16000, 16001, 16002],
    "tasks": ["copy3", "recall4"], "modes": ["local", "readout", "bptt"],
    "train_episodes": 512, "eval_episodes_per_delay": 128,
    "train_delays": [4, 16], "eval_delays": [4, 16, 32, 64],
    "copy_length": 3, "recall_pairs": 4, "value_alphabet": list(range(64, 72)),
    "noise_alphabet": list(range(128, 144)), "key_alphabet": list(range(32, 36)),
    "bptt_adam_rate": 0.003, "bptt_beta1": 0.9, "bptt_beta2": 0.999,
    "bptt_epsilon": 1e-8, "bptt_gradient_norm_clip": 1.0,
    "weight_clip": 3.0, "minimum_available_ram_bytes": 128 * 1024**2,
    "diagnostic_token_accuracy_minimum": 0.75,
    "diagnostic_copy_exact_minimum": 0.50,
    "diagnostic_reset_accuracy_advantage_minimum": 0.15,
    "official_readiness_effect": 0,
}


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rng_for(seed: int, task: str, split: str, delay: int) -> np.random.Generator:
    namespace = f"sequence-memory-v1|{seed}|{task}|{split}|{delay}".encode()
    key = int.from_bytes(hashlib.sha256(namespace).digest()[:16], "little")
    return np.random.Generator(np.random.PCG64(key))


def episode(rng: np.random.Generator, task: str, delay: int) -> dict[str, Any]:
    if task not in SPEC["tasks"] or type(delay) is not int or not 0 <= delay <= 128:
        raise ValueError("invalid memory task/delay")
    noise = rng.choice(SPEC["noise_alphabet"], delay).tolist()
    if task == "copy3":
        values = rng.choice(SPEC["value_alphabet"], 3).tolist()
        prompt, targets = [240] + values + [242] + noise + [243], values
    else:
        keys = rng.permutation(SPEC["key_alphabet"]).tolist()
        values = rng.choice(SPEC["value_alphabet"], 4).tolist()
        query = int(rng.integers(4))
        table = [item for pair in zip(keys, values, strict=True) for item in pair]
        prompt, targets = [241] + table + [242] + noise + [243, keys[query]], [values[query]]
    return {"task": task, "delay": delay, "prompt": prompt, "targets": targets}


def oracle(prompt: list[int]) -> list[int]:
    """Task-generator control only; never used as a trainable model feature."""
    if prompt[0] == 240:
        return prompt[1:4]
    if prompt[0] == 241:
        table = dict(zip(prompt[1:9:2], prompt[2:9:2], strict=True))
        return [table[prompt[-1]]]
    raise ValueError("unknown task prompt")


def make_dataset(seed: int, task: str) -> dict[str, Any]:
    generators = {d: rng_for(seed, task, "train", d) for d in SPEC["train_delays"]}
    train = [episode(generators[d], task, d) for i in range(SPEC["train_episodes"])
             for d in [SPEC["train_delays"][i % len(SPEC["train_delays"])]]]
    test = {str(d): [episode(rng, task, d) for _ in range(SPEC["eval_episodes_per_delay"])]
            for d in SPEC["eval_delays"] for rng in [rng_for(seed, task, "test", d)]}
    train_prompts = {tuple(row["prompt"]) for row in train}
    if any(tuple(row["prompt"]) in train_prompts for rows in test.values() for row in rows):
        raise AssertionError("train/test prompt overlap")
    if any(oracle(row["prompt"]) != row["targets"] for row in train + [r for rs in test.values() for r in rs]):
        raise AssertionError("task oracle failed")
    return {"seed": seed, "task": task, "train": train, "test": test,
            "train_test_prompt_overlap": 0, "oracle_exact_accuracy": 1.0}


def supervised_tokens(row: dict[str, Any]) -> tuple[list[int], list[int]]:
    # Targets enter later input positions only after their own prediction.
    tokens = row["prompt"] + row["targets"][:-1]
    labels = [-1] * (len(row["prompt"]) - 1) + row["targets"]
    return tokens, labels


def forward_one(x: dict[str, np.ndarray], parents: np.ndarray, previous: np.ndarray,
                token: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    token_check(token)
    c = np.tanh(x["w"][:, token] + np.sum(x["r"] * previous[parents], axis=1) + x["bc"])
    g = 1 / (1 + np.exp(-(x["g"][:, token] + x["a"] * previous + x["bg"])))
    state = g * previous + (1 - g) * c
    logits = x["o"] @ state + x["bo"]
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return state, p, c, g


def sequence_loss_gradient(buffer: np.ndarray, parents: np.ndarray, tokens: list[int],
                           labels: list[int]) -> tuple[float, dict[str, np.ndarray], list[np.ndarray]]:
    """Exact finite-episode BPTT, including every cross-unit recurrent path.

    All weights are held constant during this episode. Only answer positions
    contribute cross-entropy, averaged over the revealed answer tokens.
    """
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
    if not count:
        raise ValueError("sequence has no supervised prediction")
    x, tape, probabilities = views(buffer), [], []
    state, loss = np.zeros(H), 0.0
    for token, target in zip(tokens, labels, strict=True):
        previous = state
        state, p, c, g = forward_one(x, parents, previous, token)
        tape.append((previous, state, p, c, g))
        probabilities.append(p)
        if target != -1:
            loss -= math.log(max(float(p[target]), 1e-300)) / count
    gradient = {name: np.zeros_like(x[name]) for name in PARAMETERS}
    carry = np.zeros(H)
    for index in range(len(tokens) - 1, -1, -1):
        previous, state, p, c, g = tape[index]
        token, target = tokens[index], labels[index]
        ds = carry.copy()
        if target != -1:
            error = p.copy()
            error[target] -= 1
            error /= count
            gradient["o"] += np.outer(error, state)
            gradient["bo"] += error
            ds += x["o"].T @ error
        dc = ds * (1 - g) * (1 - c * c)
        dg = ds * (previous - c) * g * (1 - g)
        gradient["w"][:, token] += dc
        gradient["g"][:, token] += dg
        gradient["r"] += dc[:, None] * previous[parents]
        gradient["a"] += dg * previous
        gradient["bc"] += dc
        gradient["bg"] += dg
        carry = ds * g + dg * x["a"]
        np.add.at(carry, parents.ravel(), (dc[:, None] * x["r"]).ravel())
    if not math.isfinite(loss) or any(not np.isfinite(g).all() for g in gradient.values()):
        raise FloatingPointError("nonfinite exact-gradient result")
    return loss, gradient, probabilities


class ExactTrainer:
    def __init__(self, model: CpuSequence):
        self.model = model
        x = views(model._buffer)
        self.m = {name: np.zeros_like(x[name]) for name in PARAMETERS}
        self.v = {name: np.zeros_like(x[name]) for name in PARAMETERS}
        self.iteration = 0

    def train(self, row: dict[str, Any]) -> float:
        model = self.model
        model._open()
        if model.pending:
            raise RuntimeError("exact training requires no pending prediction")
        tokens, labels = supervised_tokens(row)
        loss, gradients, _ = sequence_loss_gradient(model._buffer, model._parents, tokens, labels)
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
        scale = min(1.0, SPEC["bptt_gradient_norm_clip"] / max(norm, 1e-300))
        self.iteration += 1
        x = views(model._buffer)
        for name, gradient in gradients.items():
            gradient = gradient * scale
            self.m[name] = .9 * self.m[name] + .1 * gradient
            self.v[name] = .999 * self.v[name] + .001 * gradient * gradient
            delta = SPEC["bptt_adam_rate"] * (self.m[name] / (1 - .9**self.iteration)) / (
                np.sqrt(self.v[name] / (1 - .999**self.iteration)) + SPEC["bptt_epsilon"])
            x[name][:] = np.clip(x[name] - delta, -3, 3)
        model.steps += len(tokens)
        model.updates += len(row["targets"])
        model.reset()
        return loss


def train_local(model: CpuSequence, row: dict[str, Any]) -> float:
    model.reset()
    tokens, labels = supervised_tokens(row)
    loss = 0.0
    for token, target in zip(tokens, labels, strict=True):
        p = model.step(token, training=target != -1)
        if target != -1:
            loss -= math.log(max(float(p[target]), 1e-300))
            model.learn(target)
    return loss / len(row["targets"])


def predict_answer(model: CpuSequence, prompt: list[int], length: int,
                   *, reset_before_query: bool = False, teacher: list[int] | None = None) -> tuple[list[int], list[list[float]]]:
    """Free-running by default: evaluation targets are not an input argument."""
    if type(length) is not int or length < 1 or not prompt:
        raise ValueError("invalid prediction prompt/length")
    x, state = views(model._buffer), np.zeros(H)
    for token in prompt[:-1]:
        state, _, _, _ = forward_one(x, model._parents, state, token)
    if reset_before_query:
        state.fill(0)
    current, predictions, probabilities = prompt[-1], [], []
    for index in range(length):
        state, p, _, _ = forward_one(x, model._parents, state, current)
        predictions.append(int(p.argmax()))
        probabilities.append(p.tolist())
        current = teacher[index] if teacher is not None else predictions[-1]
    return predictions, probabilities


def summarize_evidence(evidence: list[dict[str, Any]], *, reset: bool) -> dict[str, Any]:
    correct = exact = count = 0
    loss = 0.0
    for row in evidence:
        predictions, targets = row["free_predictions"], row["targets"]
        probabilities = row["teacher_target_probabilities"]
        if not targets or len(predictions) != len(targets) or len(probabilities) != len(targets):
            raise ValueError("incomplete evaluation evidence")
        if any(not math.isfinite(p) or not 0 < p <= 1 for p in probabilities):
            raise ValueError("invalid evaluation probability")
        correct += sum(a == b for a, b in zip(predictions, targets, strict=True))
        exact += predictions == targets
        count += len(targets)
        loss -= sum(math.log(max(p, 1e-300)) for p in probabilities)
    return {"token_accuracy": correct / count, "exact_accuracy": exact / len(evidence),
            "teacher_nll": loss / count, "episodes": len(evidence), "targets": count,
            "reset_before_query": reset, "rows": evidence}


def evaluate(model: CpuSequence, rows: list[dict[str, Any]], *, reset: bool = False) -> dict[str, Any]:
    before = model.parameter_digest()
    evidence = []
    for row in rows:
        length = len(row["targets"])
        predicted, _ = predict_answer(model, row["prompt"], length, reset_before_query=reset)
        _, teacher_p = predict_answer(model, row["prompt"], length, reset_before_query=reset, teacher=row["targets"])
        target_p = [p[target] for p, target in zip(teacher_p, row["targets"], strict=True)]
        evidence.append({"prompt_sha256": digest(row["prompt"]), "targets": row["targets"],
                         "free_predictions": predicted, "teacher_target_probabilities": target_p})
    if model.parameter_digest() != before:
        raise AssertionError("evaluation changed model parameters")
    return summarize_evidence(evidence, reset=reset)


def diagnostic_pass(results: list[dict[str, Any]]) -> bool:
    local = [row for row in results if row["mode"] == "local"]
    expected = {(seed, task) for seed in SPEC["model_seeds"] for task in SPEC["tasks"]}
    if len(local) != len(expected) or {(r["seed"], r["task"]) for r in local} != expected:
        return False
    return all(
        row["normal"][str(delay)]["token_accuracy"] >= SPEC["diagnostic_token_accuracy_minimum"]
        and (row["task"] != "copy3" or row["normal"][str(delay)]["exact_accuracy"] >= SPEC["diagnostic_copy_exact_minimum"])
        and row["normal"][str(delay)]["token_accuracy"] - row["reset"][str(delay)]["token_accuracy"]
        >= SPEC["diagnostic_reset_accuracy_advantage_minimum"]
        for row in local for delay in SPEC["eval_delays"]
    )


def bigram_results(dataset: dict[str, Any]) -> dict[str, Any]:
    counts = np.ones((V, V), dtype=np.float64)
    for row in dataset["train"]:
        tokens, labels = supervised_tokens(row)
        for token, target in zip(tokens, labels, strict=True):
            if target != -1:
                counts[token, target] += 1
    p = counts / counts.sum(axis=1, keepdims=True)
    output = {}
    for delay, rows in dataset["test"].items():
        correct = exact = total = 0
        loss, raw = 0., []
        for row in rows:
            current, predictions = row["prompt"][-1], []
            for _ in row["targets"]:
                current = int(p[current].argmax())
                predictions.append(current)
            current = row["prompt"][-1]
            target_p = []
            for target in row["targets"]:
                target_p.append(float(p[current, target]))
                current = target
            correct += sum(a == b for a, b in zip(predictions, row["targets"], strict=True))
            exact += predictions == row["targets"]
            total += len(predictions)
            loss -= sum(math.log(value) for value in target_p)
            raw.append({"free_predictions": predictions, "target_probabilities": target_p})
        output[delay] = {"token_accuracy": correct / total, "exact_accuracy": exact / len(rows),
                         "teacher_nll": loss / total, "rows": raw}
    return output


def memory_snapshot() -> dict[str, int]:
    from .gpu_probe import host_memory
    return host_memory()


def headroom() -> dict[str, int]:
    observed = memory_snapshot()
    if observed["available_system_ram_bytes"] < SPEC["minimum_available_ram_bytes"]:
        raise MemoryError("less than registered 128 MiB available RAM at memory-task boundary")
    return observed


def source_bindings() -> dict[str, str]:
    paths = sorted((ROOT / "src/adaptive_learning_substrate").glob("*.py"))
    paths += [ROOT / PROTOCOL, ROOT / "tests/test_sequence_memory.py",
              ROOT / "pyproject.toml", ROOT / "requirements-dev.lock", Path(sys.executable)]
    return {str(path.resolve()): file_digest(path) for path in paths}


def stable_sources(expected: dict[str, str]) -> None:
    if source_bindings() != expected:
        raise RuntimeError("prospective memory-experiment source binding changed")


def run(output: Path) -> dict[str, Any]:
    output = output.resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        raise ValueError("memory experiment output must be under project artifacts")
    host = headroom()
    output.mkdir(parents=True, exist_ok=False)
    bindings = source_bindings()
    data_files = {}
    # All datasets and the complete prospective freeze precede every metric.
    for seed in SPEC["model_seeds"]:
        for task in SPEC["tasks"]:
            name = f"data_{seed}_{task}.json"
            write_record(output / name, make_dataset(seed, task))
            data_files[name] = file_digest(output / name)
    freeze = {"schema_version": "sequence-memory-freeze-v1", "spec": SPEC, "model": CONFIG,
              "sources": bindings, "data_files": data_files,
              "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
              "created_unix_ns": time.time_ns(), "host_before": host,
              "process": {"pid": os.getpid(), "instance_token": PROCESS_TOKEN}}
    stable_sources(bindings)
    write_record(output / "FREEZE.json", freeze)
    stable_sources(bindings)
    results, timing = [], []
    with threadpool_limits(limits=1):
        for seed in SPEC["model_seeds"]:
            for task in SPEC["tasks"]:
                name = f"data_{seed}_{task}.json"
                if file_digest(output / name) != data_files[name]:
                    raise RuntimeError("frozen dataset changed")
                dataset = read_record(output / name)
                bigram = bigram_results(dataset)
                for mode in SPEC["modes"]:
                    headroom()
                    stable_sources(bindings)
                    model = CpuSequence(seed, core_learning=mode != "readout")
                    before = model.parameter_digest()
                    trainer = ExactTrainer(model) if mode == "bptt" else None
                    losses, start = [], time.perf_counter()
                    try:
                        for index, row in enumerate(dataset["train"]):
                            if index % 32 == 0:
                                headroom()
                            losses.append(trainer.train(row) if trainer else train_local(model, row))
                        training_seconds = time.perf_counter() - start
                        trained_digest = model.parameter_digest()
                        normal, reset = {}, {}
                        for delay, rows in dataset["test"].items():
                            headroom()
                            normal[delay] = evaluate(model, rows)
                            reset[delay] = evaluate(model, rows, reset=True)
                        model.reset()
                        checkpoint = f"model_{seed}_{task}_{mode}.json"
                        model.save(output / checkpoint)
                        result = {"seed": seed, "task": task, "mode": mode,
                                  "train_losses": losses, "training_input_tokens": model.steps,
                                  "supervised_target_count": model.updates,
                                  "optimizer_steps": trainer.iteration if trainer else model.updates,
                                  "initial_parameter_sha256": before, "trained_parameter_sha256": trained_digest,
                                  "checkpoint": checkpoint, "checkpoint_sha256": file_digest(output / checkpoint),
                                  "normal": normal, "reset": reset, "bigram": bigram}
                        results.append(result)
                        timing.append({"seed": seed, "task": task, "mode": mode,
                                       "training_seconds": training_seconds,
                                       "total_seconds": time.perf_counter() - start,
                                       "memory": memory_snapshot()})
                        stable_sources(bindings)
                        write_record(output / f"result_{seed}_{task}_{mode}.json", result)
                        stable_sources(bindings)
                        print(f"MEMORY_CELL_COMPLETE seed={seed} task={task} mode={mode}", flush=True)
                    finally:
                        model.close()
    passed = diagnostic_pass(results)
    payload = {"spec_sha256": digest(SPEC), "data_files": data_files, "results": results,
               "local_memory_gate_pass": passed, "official_readiness_effect": 0}
    report = {"schema_version": "sequence-memory-report-v1", "scientific_payload": payload,
              "scientific_payload_sha256": digest(payload), "timing_and_memory": timing,
              "freeze_sha256": file_digest(output / "FREEZE.json")}
    stable_sources(bindings)
    write_record(output / "REPORT.json", report)
    stable_sources(bindings)
    print("MEMORY_DIAGNOSTIC_" + ("PASS" if passed else "FAIL") + " payload=" + digest(payload), flush=True)
    return report


def verify(primary: Path, rerun: Path, output: Path) -> dict[str, Any]:
    """Fresh retraining happened in rerun; this independently reopens all outputs."""
    first, second = read_record(primary / "REPORT.json"), read_record(rerun / "REPORT.json")
    current = source_bindings()
    process_tokens = {PROCESS_TOKEN}
    for folder, report in ((primary, first), (rerun, second)):
        freeze = read_record(folder / "FREEZE.json")
        if freeze["sources"] != current or freeze["spec"] != SPEC or report["freeze_sha256"] != file_digest(folder / "FREEZE.json"):
            raise RuntimeError("memory verification source/spec/freeze differs")
        token = freeze["process"]["instance_token"]
        if token in process_tokens:
            raise RuntimeError("memory primary/rerun/verifier process instances overlap")
        process_tokens.add(token)
        payload = report["scientific_payload"]
        expected_cells = {(seed, task, mode) for seed in SPEC["model_seeds"]
                          for task in SPEC["tasks"] for mode in SPEC["modes"]}
        if (len(payload["results"]) != len(expected_cells)
                or {(r["seed"], r["task"], r["mode"]) for r in payload["results"]} != expected_cells
                or payload["local_memory_gate_pass"] != diagnostic_pass(payload["results"])):
            raise RuntimeError("memory condition registry or gate differs")
        for result in payload["results"]:
            saved = read_record(folder / f"result_{result['seed']}_{result['task']}_{result['mode']}.json")
            if saved != result or file_digest(folder / result["checkpoint"]) != result["checkpoint_sha256"]:
                raise RuntimeError("memory result/checkpoint binding differs")
            checkpoint_model = CpuSequence()
            checkpoint_model.restore(folder / result["checkpoint"])
            if checkpoint_model.parameter_digest() != result["trained_parameter_sha256"]:
                raise RuntimeError("checkpoint parameters differ")
            checkpoint_model.close()
        if digest(payload) != report["scientific_payload_sha256"]:
            raise RuntimeError("memory scientific payload hash differs")
        if payload != first["scientific_payload"]:
            raise RuntimeError("fresh memory retraining differs")
        for name, expected in freeze["data_files"].items():
            dataset = read_record(folder / name)
            if file_digest(folder / name) != expected or dataset != make_dataset(dataset["seed"], dataset["task"]):
                raise RuntimeError("memory data regeneration differs")
    # Score the original CpuSequence implementation, not forward_one, from
    # saved parameters; this provides a separate evaluator path for predictions.
    checked = 0
    max_error = 0.0
    with threadpool_limits(limits=1):
        for result in first["scientific_payload"]["results"]:
            model = CpuSequence()
            checkpoint = primary / result["checkpoint"]
            if file_digest(checkpoint) != result["checkpoint_sha256"]:
                raise RuntimeError("checkpoint identity differs")
            model.restore(checkpoint)
            dataset = read_record(primary / f"data_{result['seed']}_{result['task']}.json")
            if bigram_results(dataset) != result["bigram"]:
                raise RuntimeError("memory bigram control differs")
            for delay, rows in dataset["test"].items():
                for reset in (False, True):
                    saved = result["reset" if reset else "normal"][delay]["rows"]
                    if summarize_evidence(saved, reset=reset) != result["reset" if reset else "normal"][delay]:
                        raise RuntimeError("memory aggregate differs from raw predictions")
                    for row, expected in zip(rows, saved, strict=True):
                        if expected["prompt_sha256"] != digest(row["prompt"]) or expected["targets"] != row["targets"]:
                            raise RuntimeError("memory row is not bound to its regenerated prompt/target")
                        if checked % 128 == 0:
                            headroom()
                        model.reset()
                        for token in row["prompt"][:-1]:
                            model.step(token)
                        if reset:
                            model.reset()
                        current_token, predictions = row["prompt"][-1], []
                        for _ in row["targets"]:
                            p = model.step(current_token)
                            current_token = int(p.argmax())
                            predictions.append(current_token)
                        if predictions != expected["free_predictions"]:
                            raise AssertionError("independent free-running prediction differs")
                        model.reset()
                        for token in row["prompt"][:-1]:
                            model.step(token)
                        if reset:
                            model.reset()
                        current_token = row["prompt"][-1]
                        for target, expected_p in zip(row["targets"], expected["teacher_target_probabilities"], strict=True):
                            p = model.step(current_token)
                            max_error = max(max_error, abs(float(p[target]) - expected_p))
                            current_token = target
                        checked += 1
            if model.parameter_digest() != result["trained_parameter_sha256"]:
                raise AssertionError("independent evaluation changed parameters")
            model.close()
    if max_error > 1e-12:
        raise AssertionError("independent probability replay differs")
    stable_sources(current)
    result = {"schema_version": "sequence-memory-verification-v1", "status": "PASS",
              "scientific_payload_sha256": first["scientific_payload_sha256"],
              "fresh_retraining_exact": True, "independent_evaluation_episodes": checked,
              "maximum_probability_error": max_error, "official_readiness_effect": 0,
              "distinct_process_instance_count": len(process_tokens)}
    write_record(output, result)
    print(f"MEMORY_VERIFICATION_PASS episodes={checked} max_error={max_error:.17g}", flush=True)
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
