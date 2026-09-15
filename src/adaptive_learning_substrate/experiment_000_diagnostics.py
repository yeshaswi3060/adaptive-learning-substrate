"""Small falsification diagnostics for the recurrent Experiment-000 runner."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

from .events import EventLog
from .experiment000_data import SeedRole, generate_stream_pair, require_seed_role
from .experiment_000 import forward_episode
from .recurrent import RecurrentEdge, RecurrentEventGraph

CHAIN_SCHEMA_VERSION = "experiment-000-chain-positive-control-v1"
WRONG_EDGE_ID = "chain:h04->h05"


def build_delay_chain_positive_control(
    *, seed: int, mode: str, learning_rate: float = 0.03
) -> RecurrentEventGraph:
    """Build an 11-edge cue path whose sole plastic edge starts inverted.

    Eight noise events advance time but have no graph edges.  The output edge is
    frozen, so CCF-NO-TRACE cannot solve the task by changing only the readout;
    full pathway routing must reach and repair the earlier wrong-sign edge.
    ``seed`` is recorded and guarded even though this intentionally hand-built
    positive-control mask contains no random topology choices.
    """

    require_seed_role(seed, SeedRole.DEVELOPMENT)
    hidden = tuple(f"h{index:02d}" for index in range(10))
    edges: list[RecurrentEdge] = [
        RecurrentEdge("chain:cue->h00", "cue", "h00", "input", plastic=False)
    ]
    for index in range(9):
        edge_id = f"chain:h{index:02d}->h{index + 1:02d}"
        edges.append(
            RecurrentEdge(
                edge_id,
                f"h{index:02d}",
                f"h{index + 1:02d}",
                "recurrent",
                plastic=edge_id == WRONG_EDGE_ID,
            )
        )
    edges.append(
        RecurrentEdge("chain:h09->output", "h09", "output", "output", plastic=False)
    )
    weights = {edge.edge_id: 0.9 for edge in edges}
    weights[WRONG_EDGE_ID] = -0.9
    return RecurrentEventGraph(
        input_nodes=("cue", "noise", "query"),
        hidden_nodes=hidden,
        output_node="output",
        edges=edges,
        initial_weights=weights,
        mode=mode,
        learning_rate=learning_rate,
        trace_decay=1.0,
        route_gain=1.0,
        emit_threshold=0.0,
        credit_minimum=0.0,
        trace_horizon=32,
        hop_limit=16,
        max_update=0.1,
        weight_clip=3.0,
        event_log=EventLog(enabled=False),
    )


def _run_mode(*, seed: int, mode: str, pair: Any) -> dict[str, Any]:
    graph = build_delay_chain_positive_control(seed=seed, mode=mode)
    topology_before = graph.topology_hash()
    weights_before = graph.weights_hash()
    wrong_before = graph.weights[WRONG_EDGE_ID]
    train_correct = 0
    started = perf_counter()
    for episode in pair.train.episodes:
        query = forward_episode(graph, episode)
        prediction = query.prediction
        target = episode.target
        train_correct += int(prediction == target)
        graph.apply_supervised_credit(target)

    wrong_after_train = graph.weights[WRONG_EDGE_ID]
    weights_after_train = graph.weights_hash()
    eval_correct = 0
    for episode in pair.eval.episodes:
        query = forward_episode(graph, episode)
        prediction = query.prediction
        target = episode.target
        eval_correct += int(prediction == target)
    weights_after_eval = graph.weights_hash()
    if weights_after_train != weights_after_eval:
        raise RuntimeError("positive-control evaluation changed weights")
    if topology_before != graph.topology_hash():
        raise RuntimeError("positive-control topology changed")
    if not all(math.isfinite(weight) for weight in graph.weights.values()):
        raise FloatingPointError("positive control produced a non-finite weight")
    count = pair.train.episode_count
    return {
        "mode": mode,
        "train_prequential_accuracy": train_correct / count,
        "eval_accuracy": eval_correct / pair.eval.episode_count,
        "wrong_edge_id": WRONG_EDGE_ID,
        "wrong_edge_initial_weight": wrong_before,
        "wrong_edge_trained_weight": wrong_after_train,
        "topology_sha256": topology_before,
        "topology_unchanged": topology_before == graph.topology_hash(),
        "initial_weights_sha256": weights_before,
        "trained_weights_sha256": weights_after_train,
        "evaluation_weights_unchanged": weights_after_train == weights_after_eval,
        "ledger": graph.ledger,
        "runtime_seconds": perf_counter() - started,
    }


def run_delay_chain_positive_control(
    *,
    seed: int = 0,
    episode_count: int = 100,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run full vs no-trace on the clean 11-edge delayed pathway."""

    require_seed_role(seed, SeedRole.DEVELOPMENT)
    if episode_count <= 0 or episode_count % 2:
        raise ValueError("episode_count must be a positive even integer")
    pair = generate_stream_pair(
        seed,
        episode_count=episode_count,
        noise_events=8,
        required_role=SeedRole.DEVELOPMENT,
    )
    full = _run_mode(seed=seed, mode="full", pair=pair)
    no_trace = _run_mode(seed=seed, mode="no_trace", pair=pair)
    passed = (
        full["eval_accuracy"] == 1.0
        and full["wrong_edge_trained_weight"] > 0.0
        and no_trace["eval_accuracy"] == 0.0
        and no_trace["wrong_edge_trained_weight"] == -0.9
    )
    report = {
        "schema_version": CHAIN_SCHEMA_VERSION,
        "scope": "development_positive_control",
        "confirmatory_executed": False,
        "seed": seed,
        "episodes_per_split": episode_count,
        "data_manifest": pair.manifest.to_dict(),
        "construction": {
            "hidden_units": 10,
            "edge_count": 11,
            "edge_delay_ticks": 1,
            "query_event_tick": 9,
            "forced_output_tick": 11,
            "plastic_edges": [WRONG_EDGE_ID],
            "learning_rate": 0.03,
            "trace_decay": 1.0,
            "route_gain": 1.0,
        },
        "full": full,
        "no_trace": no_trace,
        "pass": passed,
        "interpretation": (
            "full CCF can repair an earlier edge through an 11-edge timestamped path; "
            "failure on the random recurrent graph is therefore not a total failure "
            "of delayed parent-event routing"
            if passed
            else "the delayed pathway positive control failed"
        ),
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Experiment-000 clean delayed-chain positive control."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiment_000/chain_positive_control.json"),
    )
    args = parser.parse_args(argv)
    report = run_delay_chain_positive_control(
        seed=args.seed, episode_count=args.episodes, output_path=args.output
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHAIN_SCHEMA_VERSION",
    "WRONG_EDGE_ID",
    "build_delay_chain_positive_control",
    "run_delay_chain_positive_control",
]
