"""Bounded CUDA forward/observer/head prototype; not the official CCF runner.

Importing this module does not import PyTorch or initialize CUDA. The CPU
reference remains the authority; this adapter never mutates its graph.
"""
from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

MAX_BATCH = 32
MAX_DELAY = 256
TENSOR_BUDGET_BYTES = 256 * 1024**2
PROBE_SEEDS = (12000, 12001)
ATOL = RTOL = 1e-10


def validate_batch(cues: Sequence[int | None], noise: Sequence[Sequence[int]]) -> int:
    if not 1 <= len(cues) <= MAX_BATCH or len(noise) != len(cues):
        raise ValueError("batch must contain 1..32 matched cue/noise rows")
    delay = len(noise[0])
    if delay > MAX_DELAY or any(len(row) != delay for row in noise):
        raise ValueError("noise rows must have equal length no greater than 256")
    if any(cue is not None and (type(cue) is not int or cue not in (-1, 1)) for cue in cues):
        raise ValueError("cues must be built-in bipolar integers or None")
    if any(type(value) is not int or value not in (-1, 1) for row in noise for value in row):
        raise ValueError("noise must contain built-in bipolar integers")
    return delay


@dataclass(frozen=True)
class Topology:
    nodes: tuple[str, ...]
    inputs: tuple[str, ...]
    targets: tuple[str, ...]
    sources: tuple[tuple[int, ...], ...]
    weights: tuple[tuple[float, ...], ...]
    selected: tuple[int, ...]
    threshold: float
    epsilon: float
    topology_sha256: str
    weights_sha256: str

    @classmethod
    def from_reference(cls, graph: Any) -> Topology:
        nodes = tuple(graph.nodes)
        inputs = tuple(graph.input_nodes)
        if inputs != ("cue", "noise", "query") or len(graph.hidden_nodes) != 64:
            raise ValueError("GPU V1 requires the registered 3-input/64-hidden topology")
        if any(edge.delay_ticks != 1 for edge in graph.edges):
            raise ValueError("only one-tick edges are supported")
        targets = tuple(graph.hidden_nodes) + (graph.output_node,)
        groups = [sorted((e for e in graph.edges if e.destination == node), key=lambda e: e.edge_id)
                  for node in targets]
        width = max(len(group) for group in groups)
        if width > 32:
            raise ValueError("GPU V1 supports at most 32 incoming edges per unit")
        weights = graph.weights
        if any(not math.isfinite(value) for value in weights.values()):
            raise ValueError("edge weights must be finite")
        selected = tuple(nodes.index(node) - len(inputs) for node in graph.selected_nodes)
        return cls(
            nodes, inputs, targets,
            tuple(tuple(nodes.index(e.source) for e in group) + (len(nodes),) * (width - len(group))
                  for group in groups),
            tuple(tuple(weights[e.edge_id] for e in group) + (0.0,) * (width - len(group))
                  for group in groups),
            selected, graph.emit_threshold, graph.epsilon_message,
            graph.topology_hash(), graph.weights_hash(),
        )


def cuda_runtime() -> Any:
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback is enabled")
    return torch


class CudaForward:
    """All numeric recurrence and observer buffers live on CUDA, bounded per batch.

    Padded adjacency gathering replaces Python message objects. Presence masks
    preserve zero-valued emitted messages and distinguish them from no message.
    No event history is retained unless the small validation trace is requested.
    """

    def __init__(self, topology: Topology) -> None:
        self.t = t = cuda_runtime()
        self.spec = topology
        self.device = t.device("cuda:0")
        free, _ = t.cuda.mem_get_info(self.device)
        if free < TENSOR_BUDGET_BYTES:
            raise MemoryError("CUDA headroom below the 256 MiB probe reserve")
        with t.inference_mode():
            self.sources = t.tensor(topology.sources, dtype=t.int64, device=self.device)
            self.weights = t.tensor(topology.weights, dtype=t.float64, device=self.device)
            self.selected = t.tensor(topology.selected, dtype=t.int64, device=self.device)
        self.node_count = len(topology.nodes)

    def run(self, cues: Sequence[int | None], noise: Sequence[Sequence[int]], *,
            trace: bool = False) -> dict[str, Any]:
        delay = validate_batch(cues, noise)  # Validate before any new device allocation.
        t, batch = self.t, len(cues)
        with t.inference_mode():
            source_input = t.zeros((batch, delay + 4, 3), dtype=t.float64, device=self.device)
            source_input[:, 0, 0] = t.tensor([0 if c is None else c for c in cues], device=self.device)
            if delay:
                source_input[:, 1:delay + 1, 1] = t.tensor(noise, dtype=t.float64, device=self.device)
            source_input[:, delay + 1, 2] = 1.0
            activation = t.zeros((batch, self.node_count), dtype=t.float64, device=self.device)
            last_emitted = t.zeros_like(activation)
            previous_values = t.zeros((batch, self.node_count + 1), dtype=t.float64, device=self.device)
            previous_mask = t.zeros_like(previous_values, dtype=t.bool)
            memory = t.zeros((batch, 8), dtype=t.float64, device=self.device)
            latest = t.zeros_like(memory)
            written = t.zeros_like(memory, dtype=t.bool)
            latest_written = t.zeros_like(written)
            memory_tick = t.zeros_like(memory, dtype=t.int64)
            latest_tick = t.zeros_like(memory_tick)
            events = t.zeros(batch, dtype=t.int64, device=self.device)
            touches = t.zeros_like(events)
            evaluations = t.zeros_like(events)
            observer_calls = t.zeros_like(events)
            frames: list[dict[str, Any]] = []
            for tick in range(delay + 4):
                arrived = previous_mask[:, self.sources]
                values = previous_values[:, self.sources]
                contributions = t.where(arrived, values * self.weights, 0.0)
                total = contributions.sum(dim=-1)
                evaluated = arrived.any(dim=-1)
                forced = tick == delay + 3
                if forced:
                    evaluated[:, -1] = True
                updated = t.tanh(total)
                activation[:, 3:] = t.where(evaluated, updated, activation[:, 3:])
                emitted = evaluated & ((updated - last_emitted[:, 3:]).abs() >= self.spec.threshold)
                if forced:
                    emitted[:, -1] = True
                last_emitted[:, 3:] = t.where(emitted, updated, last_emitted[:, 3:])
                selected_eval = evaluated[:, self.selected]
                selected_values = activation[:, 3:][:, self.selected]
                first = selected_eval & ~written
                memory = t.where(first, selected_values, memory)
                memory_tick = t.where(first, tick, memory_tick)
                written |= first
                latest = t.where(selected_eval, selected_values, latest)
                latest_tick = t.where(selected_eval, tick, latest_tick)
                latest_written |= selected_eval
                external = source_input[:, tick]
                external_emitted = external.abs() >= self.spec.epsilon
                activation[:, :3] = t.where(external_emitted, external, activation[:, :3])
                last_emitted[:, :3] = t.where(external_emitted, external, last_emitted[:, :3])
                previous_values.zero_()
                previous_mask.zero_()
                previous_values[:, :3] = external
                previous_mask[:, :3] = external_emitted
                previous_values[:, 3:-1] = updated
                previous_mask[:, 3:-1] = emitted
                events += emitted.sum(dim=1) + external_emitted.sum(dim=1)
                touches += arrived.sum(dim=(1, 2))
                evaluations += evaluated.sum(dim=1)
                observer_calls += selected_eval.sum(dim=1)
                if trace:
                    frames.append({"tick": tick, "activations": activation[:, 3:].cpu().tolist(),
                                   "emitted": previous_mask[:, :-1].cpu().tolist()})
            final_tick = delay + 3
            memory = t.where(written & (final_tick - memory_tick <= 32), memory, 0.0)
            latest = t.where(latest_written & (final_tick - latest_tick <= 32), latest, 0.0)
            if not bool(t.isfinite(activation).all() & t.isfinite(memory).all() & t.isfinite(latest).all()):
                raise FloatingPointError("GPU forward produced a non-finite value")
            return {"tick": final_tick, "output": activation[:, -1], "lwoh": memory,
                    "latest": latest, "events": events, "touches": touches,
                    "evaluations": evaluations, "observer_calls": observer_calls,
                    "writes": written.sum(dim=1), "frames": frames}


class CudaOnlineHead:
    """Local eight-coordinate update, no autograd, explicit prediction-before-target."""

    def __init__(self) -> None:
        self.t = t = cuda_runtime()
        with t.inference_mode():
            self.theta = t.zeros(8, dtype=t.float64, device="cuda:0")
        self._pending: tuple[int, Any, Any] | None = None
        self.predictions = self.updates = 0

    def _feature(self, feature: Any) -> None:
        t = self.t
        if not isinstance(feature, t.Tensor):
            raise TypeError("feature must be a CUDA tensor")
        if feature.device != self.theta.device or feature.dtype != t.float64 or tuple(feature.shape) != (8,):
            raise ValueError("feature must be eight float64 values on the head's CUDA device")
        if not bool(t.isfinite(feature).all()):
            raise ValueError("feature must be finite")

    def predict(self, feature: Any) -> tuple[Any, Any]:
        self._feature(feature)
        with self.t.inference_mode():
            score = self.t.tanh((self.theta * feature).sum())
            return score, self.t.where(score >= 0.0, 1, -1)

    def predict_for_update(self, feature: Any) -> tuple[int, Any, Any]:
        if self._pending is not None:
            raise RuntimeError("consume the pending target before another training prediction")
        score, prediction = self.predict(feature)
        with self.t.inference_mode():
            self._pending = (self.predictions, feature.clone(), score.clone())
        token = self.predictions
        self.predictions += 1
        return token, score, prediction

    def reveal_target(self, token: int, target: int) -> Any:
        if type(target) is not int or target not in (-1, 1):
            raise ValueError("target must be a built-in bipolar integer")
        if type(token) is not int or self._pending is None or token != self._pending[0]:
            raise RuntimeError("target does not match the pending prediction")
        _, feature, score = self._pending
        t = self.t
        with t.inference_mode():
            normalizer = (feature * feature).sum().clamp(min=1.0)
            scalar = 0.5 * (target - score) * (1.0 - score * score) / normalizer
            delta = (scalar * feature).clamp(-0.05, 0.05)
            staged = (self.theta + delta).clamp(-3.0, 3.0)
            if not bool(t.isfinite(staged).all()):
                raise FloatingPointError("GPU head update produced non-finite weights")
            self.theta.copy_(staged)
        self._pending = None
        self.updates += 1
        return delta
