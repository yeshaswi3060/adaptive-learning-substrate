"""Frozen checkpoint probes; privileged diagnostic heads never update the model."""
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

from . import sequence_diversity as diversity
from . import sequence_memory as memory
from . import sequence_trainability as fitting
from .sequence_core import (
    CONFIG,
    CpuSequence,
    H,
    canonical,
    read_record,
    token_check,
    views,
    write_record,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESS_TOKEN = hashlib.sha256(os.urandom(32)).hexdigest()
SPEC = {
    "version": "sequence-binding-audit-v1",
    "source_payload_sha256": "b3b6ca62ebe784e50a51a4d8fde0f974c406d73910e535c4a5079cf30d97af2e",
    "cue_count": 128, "stages": ["cue_end", "delay_end", "query_end"],
    "controls": ["aligned", "shuffled", "zero"], "ridge_alpha": 0.001,
    "accessible_minimum": 0.75, "difference_minimum": 0.15, "paired_minimum": 0.50,
    "official_readiness_effect": 0,
}


def prompt_layout(task: str, prompt: list[int]) -> tuple[int, int]:
    """Validate structure without reading or accepting answer labels."""
    if task not in ("copy3", "recall4") or not isinstance(prompt, list):
        raise ValueError("unknown task or invalid prompt")
    for token in prompt:
        token_check(token)
    cue, suffix = (5, 1) if task == "copy3" else (10, 2)
    if len(prompt) < cue + suffix or prompt[0] != (240 if task == "copy3" else 241):
        raise ValueError("invalid cue prefix")
    if prompt[cue - 1] != 242 or prompt[-suffix] != 243:
        raise ValueError("invalid cue/query marker")
    values = prompt[1:4] if task == "copy3" else prompt[2:9:2]
    if any(not 64 <= value < 72 for value in values):
        raise ValueError("invalid cue value")
    if task == "recall4" and (sorted(prompt[1:9:2]) != [32, 33, 34, 35]
                              or prompt[-1] not in (32, 33, 34, 35)):
        raise ValueError("invalid recall keys")
    if any(not 128 <= token < 144 for token in prompt[cue:-suffix]):
        raise ValueError("invalid distractor")
    return cue, len(prompt) - suffix


def full_values(task: str, prompt: list[int]) -> list[int]:
    prompt_layout(task, prompt)
    if task == "copy3":
        return prompt[1:4]
    table = dict(zip(prompt[1:9:2], prompt[2:9:2], strict=True))
    return [table[key] for key in (32, 33, 34, 35)]


def extract_states(model: CpuSequence, task: str, prompts: list[list[int]], *,
                   original: bool = False) -> dict[str, np.ndarray]:
    """No targets, learning calls, future tokens or caller-state mutations."""
    model._open()
    if model.pending or not prompts:
        raise ValueError("state audit needs nonempty prompts and no pending target")
    result: dict[str, list[list[float]]] = {name: [] for name in SPEC["stages"]}
    engine = CpuSequence(model.seed) if original else None
    if engine:
        engine._install(*model._export())
    x = views(model._buffer)
    try:
        for prompt in prompts:
            cue, delay = prompt_layout(task, prompt)
            boundaries = {"cue_end": cue, "delay_end": delay, "query_end": len(prompt)}
            state = np.zeros(H)
            if engine:
                engine.reset()
            for i, token in enumerate(prompt, 1):
                if engine:
                    engine.step(token)
                    state = views(engine._buffer)["s"].copy()
                else:
                    state, _, _, _ = memory.forward_one(x, model._parents, state, token)
                for name, boundary in boundaries.items():
                    if i == boundary:
                        result[name].append(state.tolist())
    finally:
        if engine:
            engine.close()
    return {name: np.asarray(rows, dtype=np.float64) for name, rows in result.items()}


def fit_probe(states: np.ndarray, labels: np.ndarray) -> dict[str, list]:
    states, labels = np.asarray(states), np.asarray(labels)
    if (states.ndim != 2 or not states.shape[0] or states.shape[1] != H
            or not np.isfinite(states).all() or labels.ndim != 2
            or labels.shape[0] != states.shape[0] or labels.shape[1] not in (3, 4)
            or labels.dtype.kind not in "iu" or np.any(labels < 64) or np.any(labels >= 72)):
        raise ValueError("invalid probe training arrays")
    targets = np.eye(8)[labels - 64].reshape(len(labels), -1)
    mx, my = states.mean(axis=0), targets.mean(axis=0)
    centered = states - mx
    weights = np.linalg.solve(centered.T @ centered / len(states) + SPEC["ridge_alpha"] * np.eye(H),
                              centered.T @ (targets - my) / len(states))
    bias = my - mx @ weights
    if not np.isfinite(weights).all() or not np.isfinite(bias).all():
        raise FloatingPointError("nonfinite probe coefficients")
    return {"weights": weights.tolist(), "bias": bias.tolist()}


def predict_probe(probe: dict[str, list], states: np.ndarray) -> np.ndarray:
    scores = states @ np.asarray(probe["weights"]) + np.asarray(probe["bias"])
    return 64 + scores.reshape(len(states), -1, 8).argmax(axis=2)


def permutation(seed: int, task: str, count: int) -> np.ndarray:
    key = hashlib.sha256(f"binding-audit-v1|{seed}|{task}|shuffled".encode()).digest()[:16]
    return np.random.Generator(np.random.PCG64(int.from_bytes(key, "little"))).permutation(count)


def score_probe(predictions: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([full_values(row["task"], row["prompt"]) for row in rows])
    if not rows or predictions.shape != labels.shape:
        raise ValueError("probe prediction/label shape differs")
    answers = [prediction.tolist() if row["task"] == "copy3" else [int(prediction[row["prompt"][-1] - 32])]
               for prediction, row in zip(predictions, rows, strict=True)]
    targets = [row["targets"] for row in rows]
    correct = [a == t for a, t in zip(answers, targets, strict=True)]
    return {"full_value_token_accuracy": float(np.mean(predictions == labels)),
            "full_value_exact_accuracy": float(np.mean(np.all(predictions == labels, axis=1))),
            "answer_token_accuracy": float(np.mean(np.asarray(answers) == np.asarray(targets))),
            "answer_exact_accuracy": sum(correct) / len(rows), "answer_correct": correct,
            "full_predictions": predictions.tolist(), "answer_predictions": answers}


def audit_sets(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    sets = {"fit": data["all_seen"][:SPEC["cue_count"]],
            "heldout": data["heldout16"], "counterfactual": data["counterfactual16"]}
    if len(sets["fit"]) != SPEC["cue_count"] or not sets["heldout"]:
        raise ValueError("incomplete audit data")

    def signature(row: dict[str, Any]) -> tuple[int, ...]:
        cue, end = prompt_layout(row["task"], row["prompt"])
        if row["delay"] != end - cue or memory.oracle(row["prompt"]) != row["targets"]:
            raise ValueError("audit oracle or delay differs")
        return tuple(row["prompt"][:cue] + row["prompt"][end:])

    train = {signature(row) for row in sets["fit"]}
    if len(train) != len(sets["fit"]):
        raise ValueError("duplicate probe-training cue")
    for name in ("heldout", "counterfactual"):
        if not train.isdisjoint(signature(row) for row in sets[name]):
            raise ValueError("novel/intervened cue leaked into probe training")
    if len(sets["heldout"]) != len(sets["counterfactual"]):
        raise ValueError("incomplete counterfactual pairs")
    if [diversity.intervene(row) for row in sets["heldout"]] != sets["counterfactual"]:
        raise ValueError("counterfactual pairing differs")
    return sets


def predicates(stages: dict[str, Any], native: dict[str, Any]) -> dict[str, bool]:
    def value(stage: str, control: str, metric: str) -> float:
        return stages[stage][control]["scores"]["heldout"][metric]
    full, answer = "full_value_token_accuracy", "answer_token_accuracy"
    minimum, gap = SPEC["accessible_minimum"], SPEC["difference_minimum"]
    cue, delay, query = [value(s, "aligned", full) for s in SPEC["stages"]]
    accessible = cue >= minimum and all(cue - value("cue_end", c, full) >= gap for c in ("shuffled", "zero"))
    query_answer = value("query_end", "aligned", answer)
    return {"encoding_accessible": accessible, "delay_drop": accessible and cue - delay >= gap,
            "query_drop": delay >= minimum and delay - query >= gap,
            "diagnostic_decoder_gap": query_answer >= minimum
            and query_answer - native["heldout"]["normal"]["token_accuracy"] >= gap
            and all(query_answer - value("query_end", c, answer) >= gap for c in ("shuffled", "zero"))
            and stages["query_end"]["aligned"]["paired_answer_exact"] >= SPEC["paired_minimum"]}


def audit_checkpoint(model: CpuSequence, data: dict[str, Any], *, original: bool = False) -> dict[str, Any]:
    before = canonical(model.snapshot())
    sets = audit_sets(data)
    features = {name: extract_states(model, data["task"], [r["prompt"] for r in rows], original=original)
                for name, rows in sets.items()}
    labels = np.asarray([full_values(data["task"], row["prompt"]) for row in sets["fit"]])
    order = permutation(data["seed"], data["task"], len(labels))
    stages = {}
    for stage in SPEC["stages"]:
        stages[stage] = {}
        for control in SPEC["controls"]:
            train = features["fit"][stage]
            probe = fit_probe(np.zeros_like(train) if control == "zero" else train,
                              labels[order] if control == "shuffled" else labels)
            scores = {}
            for name, rows in sets.items():
                states = features[name][stage]
                prediction = predict_probe(probe, np.zeros_like(states) if control == "zero" else states)
                scores[name] = score_probe(prediction, rows)
            paired = np.mean(np.asarray(scores["heldout"]["answer_correct"])
                             & np.asarray(scores["counterfactual"]["answer_correct"]))
            stages[stage][control] = {"probe": probe, "scores": scores, "paired_answer_exact": float(paired)}
    evaluator = fitting.original_evaluate if original else memory.evaluate
    # The original evaluator has counters/state side effects; keep them in a clone.
    clone = CpuSequence(model.seed)
    clone._install(*model._export())
    try:
        native = {name: {control: evaluator(clone, rows, reset=control == "reset")
                         for control in ("normal", "reset")}
                  for name, rows in sets.items() if name != "fit"}
    finally:
        clone.close()
    if canonical(model.snapshot()) != before:
        raise AssertionError("checkpoint audit changed model state")
    return {"states": {name: {s: x.tolist() for s, x in f.items()} for name, f in features.items()},
            "full_labels": {name: [full_values(data["task"], r["prompt"]) for r in rows] for name, rows in sets.items()},
            "permutation": order.tolist(), "stages": stages, "native": native,
            "predicates": predicates(stages, native)}


def source_bindings() -> dict[str, str]:
    paths = sorted((ROOT / "src/adaptive_learning_substrate").glob("*.py"))
    paths += [ROOT / p for p in ("docs/SEQUENCE_BINDING_AUDIT_V1.md", "tests/test_sequence_binding_audit.py",
                                "pyproject.toml", "requirements-dev.lock")]
    paths.append(Path(sys.executable))
    return {str(path.resolve()): memory.file_digest(path) for path in paths}


def check_files(bindings: dict[str, str]) -> None:
    for name, expected in bindings.items():
        if memory.file_digest(Path(name)) != expected:
            raise RuntimeError("binding-audit input changed: " + name)


def load_source(source: Path) -> tuple[list[tuple[dict, dict]], dict, dict[str, str]]:
    """Admit the pinned historical evidence, not arbitrary edited checkpoint rows."""
    files: dict[str, str] = {}

    def record(path: Path) -> Any:
        value = read_record(path)
        for bound in (path, path.with_suffix(path.suffix + ".sha256")):
            files[str(bound.resolve())] = memory.file_digest(bound)
        return value

    report, freeze = record(source / "REPORT.json"), record(source / "FREEZE.json")
    receipt = record(source.parent / "INDEPENDENT_VERIFICATION.json")
    payload = report["scientific_payload"]
    if (memory.digest(payload) != SPEC["source_payload_sha256"]
            or report["scientific_payload_sha256"] != SPEC["source_payload_sha256"]
            or report["freeze_sha256"] != memory.file_digest(source / "FREEZE.json")
            or freeze["spec"] != diversity.SPEC or freeze["model"] != CONFIG
            or freeze["learning_spec"] != diversity.credit.SPEC
            or payload["spec_sha256"] != memory.digest(diversity.SPEC)
            or payload["data_files"] != freeze["data_files"]
            or receipt["status"] != "PASS" or receipt["scientific_payload_sha256"] != SPEC["source_payload_sha256"]
            or receipt["distinct_process_instances"] != 3 or not receipt["fresh_training_exact"]
            or not receipt["original_engine_results_exact"]):
        raise RuntimeError("historical diversity evidence binding differs")
    check_files(freeze["sources"])
    files.update(freeze["sources"])
    datasets = {}
    expected_data = {f"data_{s}_{t}.json" for s in diversity.SPEC["model_seeds"] for t in diversity.SPEC["tasks"]}
    if set(payload["data_files"]) != expected_data:
        raise RuntimeError("historical dataset registry differs")
    for name, sha in payload["data_files"].items():
        data = record(source / name)
        if memory.file_digest(source / name) != sha or data != diversity.make_data(data["seed"], data["task"]):
            raise RuntimeError("historical data regeneration differs")
        audit_sets(data)
        datasets[data["seed"], data["task"]] = data
    expected = {(s, t, SPEC["cue_count"], m) for s in diversity.SPEC["model_seeds"]
                for t in diversity.SPEC["tasks"] for m in diversity.SPEC["modes"]}
    selected = [r for r in payload["results"] if r["diversity"] == SPEC["cue_count"]]
    if len(selected) != len(expected) or {(r["seed"], r["task"], r["diversity"], r["mode"]) for r in selected} != expected:
        raise RuntimeError("historical condition registry differs")
    cells = []
    for row in selected:
        name = diversity.cell_name(row["seed"], row["task"], row["diversity"], row["mode"])
        if record(source / f"result_{name}.json") != row:
            raise RuntimeError("historical cell differs")
        if [s["episodes"] for s in row["snapshots"]] != diversity.SPEC["checkpoint_episodes"]:
            raise RuntimeError("historical checkpoint registry differs")
        for snapshot in row["snapshots"]:
            expected_name = f"model_{name}_episodes{snapshot['episodes']}.json"
            if snapshot["checkpoint"] != expected_name:
                raise RuntimeError("historical checkpoint name differs")
            record(source / expected_name)
            if memory.file_digest(source / expected_name) != snapshot["checkpoint_sha256"]:
                raise RuntimeError("historical checkpoint hash differs")
            cells.append((row, snapshot))
    check_files(files)
    return cells, datasets, files


def checked_result(source: Path, row: dict, snapshot: dict, data: dict, *, original: bool = False) -> dict:
    model = CpuSequence()
    try:
        model.restore(source / snapshot["checkpoint"])
        if model.parameter_digest() != snapshot["parameter_sha256"]:
            raise RuntimeError("historical parameter digest differs")
        result = audit_checkpoint(model, data, original=original)
    finally:
        model.close()
    for name, old in (("heldout", "heldout16"), ("counterfactual", "counterfactual16")):
        if result["native"][name] != snapshot["metrics"]["scores"][old]:
            raise AssertionError("native checkpoint replay differs from historical evidence")
    return {"seed": row["seed"], "task": row["task"], "mode": row["mode"],
            "episodes": snapshot["episodes"], "checkpoint": snapshot["checkpoint"],
            "checkpoint_sha256": snapshot["checkpoint_sha256"], "audit": result}


def run(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output == source:
        raise ValueError("audit output must be new project artifacts")
    host = memory.headroom()
    cells, datasets, inputs = load_source(source)
    sources = source_bindings()
    output.mkdir(parents=True, exist_ok=False)
    freeze = {"spec": SPEC, "source": str(source), "inputs": inputs, "sources": sources,
              "process": {"pid": os.getpid(), "instance_token": PROCESS_TOKEN},
              "created_unix_ns": time.time_ns(), "host_before": host,
              "environment": {"python": sys.version, "numpy": np.__version__}}
    freeze_sha = write_record(output / "FREEZE.json", freeze)

    def stable() -> None:
        if source_bindings() != sources or read_record(output / "FREEZE.json") != freeze:
            raise RuntimeError("audit prospective freeze changed")
        check_files(inputs)

    results, timings = [], []
    with threadpool_limits(limits=1):
        for row, snapshot in cells:
            memory.headroom()
            stable()
            start = time.perf_counter()
            result = checked_result(source, row, snapshot, datasets[row["seed"], row["task"]])
            name = snapshot["checkpoint"].replace("model_", "audit_", 1)
            sha = write_record(output / name, result)
            if read_record(output / name) != result:
                raise AssertionError("audit record reopen differs")
            results.append({"file": name, "sha256": sha})
            timings.append({"file": name, "seconds": time.perf_counter() - start, "memory": memory.memory_snapshot()})
            stable()
            print("BINDING_AUDIT_CHECKPOINT " + name, flush=True)
    payload = {"spec": SPEC, "results": results, "model_updates": 0, "official_readiness_effect": 0}
    report = {"scientific_payload": payload, "scientific_payload_sha256": memory.digest(payload),
              "freeze_sha256": freeze_sha, "timings": timings}
    stable()
    write_record(output / "REPORT.json", report)
    if read_record(output / "REPORT.json") != report:
        raise AssertionError("audit report reopen differs")
    stable()
    print("BINDING_AUDIT_COMPLETE payload=" + report["scientific_payload_sha256"], flush=True)
    return report


def verify(primary: Path, rerun: Path, output: Path) -> dict:
    evidence = {str(p.resolve()): memory.file_digest(p) for folder in (primary, rerun)
                for p in folder.iterdir() if p.is_file()}
    reports = [read_record(folder / "REPORT.json") for folder in (primary, rerun)]
    sources, tokens = source_bindings(), {PROCESS_TOKEN}
    source = Path(read_record(primary / "FREEZE.json")["source"])
    cells, datasets, inputs = load_source(source)
    expected_names = [s["checkpoint"].replace("model_", "audit_", 1) for _, s in cells]
    for folder, report in zip((primary, rerun), reports, strict=True):
        freeze = read_record(folder / "FREEZE.json")
        if (freeze["sources"] != sources or freeze["inputs"] != inputs or freeze["spec"] != SPEC
                or freeze["source"] != str(source) or report["freeze_sha256"] != memory.file_digest(folder / "FREEZE.json")):
            raise RuntimeError("audit verification freeze/input binding differs")
        token = freeze["process"]["instance_token"]
        if token in tokens:
            raise RuntimeError("audit verification requires distinct processes")
        tokens.add(token)
        payload = report["scientific_payload"]
        if (payload != reports[0]["scientific_payload"] or memory.digest(payload) != report["scientific_payload_sha256"]
                or payload["spec"] != SPEC or payload["model_updates"] != 0 or payload["official_readiness_effect"] != 0
                or [r["file"] for r in payload["results"]] != expected_names):
            raise RuntimeError("audit deterministic payload/registry differs")
        for record in payload["results"]:
            read_record(folder / record["file"])
            if memory.file_digest(folder / record["file"]) != record["sha256"]:
                raise RuntimeError("audit record digest differs")
    checked = 0
    with threadpool_limits(limits=1):
        for (row, snapshot), name in zip(cells, expected_names, strict=True):
            memory.headroom()
            data = datasets[row["seed"], row["task"]]
            expected = checked_result(source, row, snapshot, data, original=True)
            if expected != read_record(primary / name):
                raise AssertionError("original-engine states/probes/scores/predicates differ")
            checked += sum(len(rows) for rows in audit_sets(data).values())
    check_files(inputs)
    check_files(evidence)
    if source_bindings() != sources:
        raise RuntimeError("audit sources changed during verification")
    result = {"status": "PASS", "scientific_payload_sha256": reports[0]["scientific_payload_sha256"],
              "distinct_process_instances": len(tokens), "checkpoints": len(cells),
              "original_engine_state_episodes": checked, "exact_states_probes_scores": True,
              "model_updates": 0, "official_readiness_effect": 0}
    write_record(output, result)
    if read_record(output) != result:
        raise AssertionError("audit verification reopen differs")
    print(f"BINDING_AUDIT_VERIFICATION_PASS checkpoints={len(cells)} state_episodes={checked} exact=True", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--source", type=Path, required=True)
    runner.add_argument("--output", type=Path, required=True)
    verifier = commands.add_parser("verify")
    verifier.add_argument("--primary", type=Path, required=True)
    verifier.add_argument("--rerun", type=Path, required=True)
    verifier.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args.source, args.output)
    else:
        verify(args.primary, args.rerun, args.output)


if __name__ == "__main__":
    main()
