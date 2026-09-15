"""Deterministic construction, training, evaluation, and reporting."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .environments import Bits3, MicroProgram3BitEnvironment, Operation
from .events import EventLog
from .graph import Edge, FixedSparseEventGraph
from .learning import CausalCreditFlow, EligibilityTraceBaseline
from .metrics import MetricsTracker, PredictionRecord, forgetting

LearnerName = Literal["ccf", "eligibility"]


def build_microprogram_graph(
    *,
    seed: int,
    trace_decay: float = 0.97,
    weight_clip: float = 3.0,
    route_gain: float = 0.9,
    max_update: float = 0.05,
    trace_horizon: int = 32,
    log_events: bool = True,
) -> FixedSparseEventGraph:
    """Build one explicit immutable topology; only scalar edge weights can change."""

    rng = np.random.default_rng(seed)
    inputs = MicroProgram3BitEnvironment.input_nodes()
    groups = MicroProgram3BitEnvironment.output_groups()
    outputs = tuple(node for nodes in groups.values() for node in nodes)
    edges = [
        Edge(
            edge_id=f"e:{source}->{destination}",
            source=source,
            destination=destination,
            weight=float(rng.uniform(-1e-6, 1e-6)),
        )
        for source in inputs
        for destination in outputs
    ]
    return FixedSparseEventGraph(
        input_nodes=inputs,
        output_groups=groups,
        edges=edges,
        trace_decay=trace_decay,
        weight_clip=weight_clip,
        route_gain=route_gain,
        max_update=max_update,
        trace_horizon=trace_horizon,
        event_log=EventLog(enabled=log_events),
    )


def _make_learner(name: LearnerName, graph: FixedSparseEventGraph, learning_rate: float):
    if name == "ccf":
        return CausalCreditFlow(graph, learning_rate=learning_rate)
    if name == "eligibility":
        return EligibilityTraceBaseline(graph, learning_rate=learning_rate)
    raise ValueError(f"unknown learner: {name}")


def train_phase(
    *,
    graph: FixedSparseEventGraph,
    learner: CausalCreditFlow | EligibilityTraceBaseline,
    environment: MicroProgram3BitEnvironment,
    operations: Sequence[Operation],
    episodes: int,
    delay_min: int,
    delay_max: int,
    phase: str,
    metrics: MetricsTracker,
) -> None:
    for episode_index in range(episodes):
        episode = environment.sample(operations=operations, delay_min=delay_min, delay_max=delay_max)
        graph.begin_episode(f"{phase}:{episode_index}")
        result = graph.forward(environment.encode(episode.inputs, episode.operation))
        predicted = environment.decode(result.selections)
        # Distractors do not activate task edges, but each consumes one time step.
        graph.elapse(episode.delay)
        updates = learner.learn(result, environment.target_nodes(episode.target))
        metrics.add(
            PredictionRecord(
                phase=phase,
                task=episode.operation,
                predicted=predicted,
                target=episode.target,
                delay=episode.delay,
                update_count=len(updates),
            )
        )


def predict(
    graph: FixedSparseEventGraph,
    environment: MicroProgram3BitEnvironment,
    bits: Bits3,
    operation: Operation,
    *,
    episode_id: str,
    delay: int = 0,
) -> Bits3:
    graph.begin_episode(episode_id)
    result = graph.forward(environment.encode(bits, operation))
    graph.elapse(delay)
    return environment.decode(result.selections)


def evaluate_primitives(
    graph: FixedSparseEventGraph,
    environment: MicroProgram3BitEnvironment,
    operations: Sequence[Operation],
    *,
    phase: str,
    delay: int,
) -> dict[str, Any]:
    per_task: dict[str, float] = {}
    all_correct = 0
    total = 0
    for operation in operations:
        task_correct = 0
        for index, bits in enumerate(environment.all_inputs()):
            predicted = predict(
                graph,
                environment,
                bits,
                operation,
                episode_id=f"eval:{phase}:{operation}:{index}",
                delay=delay,
            )
            target = environment.apply(bits, operation)
            task_correct += int(predicted == target)
        per_task[operation] = task_correct / len(environment.all_inputs())
        all_correct += task_correct
        total += len(environment.all_inputs())
    return {"accuracy": all_correct / total if total else 0.0, "per_task": per_task, "cases": total}


def evaluate_compositions(
    graph: FixedSparseEventGraph,
    environment: MicroProgram3BitEnvironment,
    *,
    delay: int,
) -> dict[str, Any]:
    per_program: dict[str, float] = {}
    all_correct = 0
    total = 0
    for first in environment.OPERATIONS:
        for second in environment.OPERATIONS:
            correct = 0
            name = f"{first}->{second}"
            for index, bits in enumerate(environment.all_inputs()):
                intermediate = predict(
                    graph,
                    environment,
                    bits,
                    first,
                    episode_id=f"compose:{name}:{index}:0",
                    delay=delay,
                )
                predicted = predict(
                    graph,
                    environment,
                    intermediate,
                    second,
                    episode_id=f"compose:{name}:{index}:1",
                    delay=delay,
                )
                target = environment.compose(bits, (first, second))
                correct += int(predicted == target)
            per_program[name] = correct / len(environment.all_inputs())
            all_correct += correct
            total += len(environment.all_inputs())
    return {"accuracy": all_correct / total, "per_program": per_program, "cases": total}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_experiment(config: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, Any]:
    status = config.get("status", {})
    if status and not bool(status.get("implementation_ready", True)):
        raise NotImplementedError(
            "this preregistered Experiment 001 config requires the future 64-unit recurrent runner"
        )
    exp = config["experiment"]
    graph_config = config.get("graph", {})
    seed = int(exp.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    environment = MicroProgram3BitEnvironment(seed=seed)
    graph = build_microprogram_graph(
        seed=seed,
        trace_decay=float(graph_config.get("trace_decay", 0.97)),
        weight_clip=float(graph_config.get("weight_clip", 3.0)),
        route_gain=float(graph_config.get("route_gain", 0.9)),
        max_update=float(graph_config.get("max_update", 0.05)),
        trace_horizon=int(graph_config.get("trace_horizon", 32)),
        log_events=bool(config.get("output", {}).get("write_event_log", True)),
    )
    learning_rate = float(graph_config.get("learning_rate", 0.01))
    learner_name: LearnerName = exp.get("learner", "ccf")
    learner = _make_learner(learner_name, graph, learning_rate)
    metrics = MetricsTracker()
    phase_a_ops: tuple[Operation, ...] = tuple(exp.get("phase_a_operations", ["FLIP0", "ROTL"]))  # type: ignore[assignment]
    phase_b_ops: tuple[Operation, ...] = tuple(exp.get("phase_b_operations", ["NOT"]))  # type: ignore[assignment]
    delay_min = int(exp.get("train_delay_min", 1))
    delay_max = int(exp.get("train_delay_max", 3))
    eval_delay = int(exp.get("eval_delay", delay_max))

    topology_before = graph.topology_signature()
    train_phase(
        graph=graph,
        learner=learner,
        environment=environment,
        operations=phase_a_ops,
        episodes=int(exp.get("phase_a_episodes", 96)),
        delay_min=delay_min,
        delay_max=delay_max,
        phase="A",
        metrics=metrics,
    )
    before_b = evaluate_primitives(graph, environment, phase_a_ops, phase="before_B", delay=eval_delay)
    train_phase(
        graph=graph,
        learner=learner,
        environment=environment,
        operations=phase_b_ops,
        episodes=int(exp.get("phase_b_episodes", 48)),
        delay_min=delay_min,
        delay_max=delay_max,
        phase="B",
        metrics=metrics,
    )
    retained_a = evaluate_primitives(graph, environment, phase_a_ops, phase="retained_A", delay=eval_delay)
    learned_b = evaluate_primitives(graph, environment, phase_b_ops, phase="learned_B", delay=eval_delay)
    compositions = evaluate_compositions(graph, environment, delay=eval_delay)
    topology_after = graph.topology_signature()
    report: dict[str, Any] = {
        "scope": "minimal_stage1_scaffold_experiment_000_not_confirmatory_experiment_001",
        "confirmatory": False,
        "implemented_claim": "deterministic fixed-DAG software plumbing only",
        "experiment": str(exp.get("name", "experiment_001")),
        "seed": seed,
        "learner": learner_name,
        "phase_a_before_b": before_b,
        "phase_a_after_b": retained_a,
        "phase_b_after_b": learned_b,
        "forgetting": forgetting(float(before_b["accuracy"]), float(retained_a["accuracy"])),
        "composition": {
            **compositions,
            "method": "external_two_pass_primitive_adapter",
        },
        "training": {
            "phase_a": metrics.summary(phase="A"),
            "phase_b": metrics.summary(phase="B"),
        },
        "audit": {
            "topology_unchanged": topology_before == topology_after,
            "topology_sha256": _json_hash(topology_after),
            "weights_sha256": _json_hash(graph.edge_weights()),
            "event_count": len(graph.event_log.records),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (target / "config.snapshot.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if graph.event_log.enabled:
            graph.event_log.write_jsonl(target / "events.jsonl")
    return report
