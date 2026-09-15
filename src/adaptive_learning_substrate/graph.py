"""An explicit fixed sparse event graph with local traces and local updates."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from .events import (
    CreditEvent,
    CreditPacket,
    EdgeTrace,
    EventLog,
    ForwardEvent,
    UnitEvent,
    UpdateEvent,
)


def _finite_float(value: object, *, name: str) -> float:
    """Normalize one scalar while rejecting values that cannot safely enter state."""

    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite scalar")
    return normalized


@dataclass(slots=True)
class Edge:
    """One directed, plastic connection and its local state."""

    edge_id: str
    source: str
    destination: str
    weight: float
    plastic: bool = True
    participation_trace: float = 0.0
    omission_trace: float = 0.0
    weight_secant_trace: float = 0.0
    source_secant_trace: float = 0.0
    active_this_event: bool = False
    last_event_id: str | None = None
    last_trace_step: int = -1


@dataclass(frozen=True, slots=True)
class ForwardResult:
    selections: dict[str, str]
    logits: dict[str, float]
    activations: dict[str, float]
    unit_event_ids: dict[str, str]


class FixedSparseEventGraph:
    """A deterministic DAG represented as an edge list rather than dense layers.

    Every active incoming edge stores its source participation and the local
    leave-one-message-out effect on the destination. Credit is routed only through
    these stored local traces; there is no gradient tape or reverse differentiation.

    This class is the minimal acyclic Stage-1 scaffold. The preregistered recurrent
    runner requires a timestamp-addressed event store and is intentionally gated.
    """

    def __init__(
        self,
        *,
        input_nodes: Iterable[str],
        hidden_nodes: Iterable[str] = (),
        output_groups: Mapping[str, Iterable[str]],
        edges: Iterable[Edge],
        trace_decay: float = 0.97,
        activation_threshold: float = 1e-12,
        weight_clip: float = 3.0,
        route_gain: float = 0.9,
        epsilon: float = 1e-12,
        max_update: float = 0.05,
        trace_horizon: int = 32,
        event_log: EventLog | None = None,
    ) -> None:
        trace_decay = _finite_float(trace_decay, name="trace_decay")
        activation_threshold = _finite_float(
            activation_threshold, name="activation_threshold"
        )
        weight_clip = _finite_float(weight_clip, name="weight_clip")
        route_gain = _finite_float(route_gain, name="route_gain")
        epsilon = _finite_float(epsilon, name="epsilon")
        max_update = _finite_float(max_update, name="max_update")
        trace_horizon_scalar = _finite_float(trace_horizon, name="trace_horizon")
        if not 0.0 <= trace_decay <= 1.0:
            raise ValueError("trace_decay must be in [0, 1]")
        if activation_threshold < 0.0:
            raise ValueError("activation_threshold must be non-negative")
        if weight_clip <= 0.0:
            raise ValueError("weight_clip must be positive")
        if not 0.0 <= route_gain <= 1.0:
            raise ValueError("route_gain must be in [0, 1]")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if max_update <= 0.0 or trace_horizon_scalar < 1:
            raise ValueError("max_update and trace_horizon must be positive")
        normalized_input_nodes = tuple(input_nodes)
        normalized_hidden_nodes = tuple(hidden_nodes)
        normalized_output_groups = {
            name: tuple(nodes) for name, nodes in output_groups.items()
        }
        normalized_output_nodes = tuple(
            node for nodes in normalized_output_groups.values() for node in nodes
        )
        all_nodes = normalized_input_nodes + normalized_hidden_nodes + normalized_output_nodes
        if len(set(all_nodes)) != len(all_nodes):
            raise ValueError("node names must be unique")
        normalized_edges = list(edges)
        if len({edge.edge_id for edge in normalized_edges}) != len(normalized_edges):
            raise ValueError("edge ids must be unique")
        if any(
            edge.source not in all_nodes or edge.destination not in all_nodes
            for edge in normalized_edges
        ):
            raise ValueError("every edge endpoint must name a graph node")
        if any(edge.destination in normalized_input_nodes for edge in normalized_edges):
            raise ValueError("input nodes cannot have incoming edges")
        for edge in normalized_edges:
            edge.weight = _finite_float(
                edge.weight, name=f"initial weight {edge.edge_id!r}"
            )

        self.input_nodes = normalized_input_nodes
        self.hidden_nodes = normalized_hidden_nodes
        self.output_groups = normalized_output_groups
        self.output_nodes = normalized_output_nodes
        self.nodes = all_nodes
        self.edges = normalized_edges
        self.trace_decay = trace_decay
        self.activation_threshold = activation_threshold
        self.weight_clip = weight_clip
        self.route_gain = route_gain
        self.epsilon = epsilon
        self.max_update = max_update
        self.trace_horizon = int(trace_horizon_scalar)
        self.event_log = event_log if event_log is not None else EventLog()
        self._incoming: dict[str, list[Edge]] = {node: [] for node in all_nodes}
        self._outgoing: dict[str, list[Edge]] = {node: [] for node in all_nodes}
        for edge in self.edges:
            self._incoming[edge.destination].append(edge)
            self._outgoing[edge.source].append(edge)
        self._topological_order = self._topological_sort()
        self._active_episode: str | None = None
        self._step = 0
        self._last_activations = {node: 0.0 for node in all_nodes}
        self._unit_events: list[UnitEvent] = []
        self._latest_event_by_node: dict[str, UnitEvent] = {}

    def _topological_sort(self) -> tuple[str, ...]:
        indegree = {node: len(self._incoming[node]) for node in self.nodes}
        ready = [node for node in self.nodes if indegree[node] == 0]
        result: list[str] = []
        while ready:
            node = ready.pop(0)
            result.append(node)
            for edge in self._outgoing[node]:
                indegree[edge.destination] -= 1
                if indegree[edge.destination] == 0:
                    ready.append(edge.destination)
        if len(result) != len(self.nodes):
            raise ValueError("fixed event graph must be acyclic")
        return tuple(result)

    def begin_episode(self, episode_id: str) -> None:
        self._active_episode = str(episode_id)
        self._step = 0
        self._last_activations = {node: 0.0 for node in self.nodes}
        self._latest_event_by_node = {}
        for edge in self.edges:
            edge.participation_trace = 0.0
            edge.omission_trace = 0.0
            edge.weight_secant_trace = 0.0
            edge.source_secant_trace = 0.0
            edge.active_this_event = False
            edge.last_event_id = None
            edge.last_trace_step = -1

    def _require_episode(self) -> str:
        if self._active_episode is None:
            raise RuntimeError("call begin_episode() before forwarding events")
        return self._active_episode

    def _decay_traces(self, steps: int = 1) -> None:
        factor = self.trace_decay**steps
        for edge in self.edges:
            edge.participation_trace *= factor
            edge.omission_trace *= factor
            edge.weight_secant_trace *= factor
            edge.source_secant_trace *= factor

    def elapse(self, steps: int = 1) -> None:
        """Advance silent time, decaying stored traces without a forward event."""

        self._require_episode()
        if steps < 0:
            raise ValueError("steps must be non-negative")
        self._decay_traces(steps)
        self._step += steps
        for edge in self.edges:
            if edge.last_trace_step >= 0 and self._step - edge.last_trace_step > self.trace_horizon:
                edge.participation_trace = 0.0
                edge.omission_trace = 0.0
                edge.weight_secant_trace = 0.0
                edge.source_secant_trace = 0.0
                edge.active_this_event = False

    def forward(self, inputs: Mapping[str, float]) -> ForwardResult:
        episode_id = self._require_episode()
        unknown = set(inputs).difference(self.input_nodes)
        if unknown:
            raise KeyError(f"unknown input nodes: {sorted(unknown)}")
        normalized_inputs = {
            node: _finite_float(value, name="input activities")
            for node, value in inputs.items()
        }
        runtime_scalars = {
            "trace_decay": self.trace_decay,
            "activation_threshold": self.activation_threshold,
            "epsilon": self.epsilon,
        }
        if not all(math.isfinite(value) for value in runtime_scalars.values()):
            raise FloatingPointError("forward configuration must remain finite")
        if self.epsilon <= 0.0:
            raise FloatingPointError("epsilon must remain positive")
        if not all(math.isfinite(edge.weight) for edge in self.edges):
            raise FloatingPointError("all current edge weights must be finite")
        if not all(
            math.isfinite(value)
            for edge in self.edges
            for value in (
                edge.participation_trace,
                edge.omission_trace,
                edge.weight_secant_trace,
                edge.source_secant_trace,
            )
        ):
            raise FloatingPointError("all current edge traces must be finite")

        # Build the complete event transaction off to the side.  Finite inputs
        # and weights can still overflow when multiplied, and a later DAG node
        # can fail after earlier nodes have produced valid events.  No episode
        # state, trace, or audit record is changed until every derived scalar has
        # been checked.
        factor = self.trace_decay
        staged_edge_state: dict[
            str, tuple[float, float, float, float, bool, str | None, int]
        ] = {}
        for edge in self.edges:
            decayed = (
                edge.participation_trace * factor,
                edge.omission_trace * factor,
                edge.weight_secant_trace * factor,
                edge.source_secant_trace * factor,
            )
            if not all(math.isfinite(value) for value in decayed):
                raise FloatingPointError("a non-finite decayed trace was produced")
            staged_edge_state[edge.edge_id] = (
                *decayed,
                False,
                edge.last_event_id,
                edge.last_trace_step,
            )

        base_step = self._step + 1
        staged_step = base_step
        activations = {node: 0.0 for node in self.nodes}
        unit_event_ids: dict[str, str] = {}
        staged_unit_events: list[UnitEvent] = []
        staged_audit_events: list[UnitEvent | ForwardEvent] = []
        staged_latest = dict(self._latest_event_by_node)
        for node, value in normalized_inputs.items():
            activations[node] = value
            if abs(value) >= self.activation_threshold:
                event_id = f"{episode_id}:{base_step}:{node}"
                event = UnitEvent(
                    event_id, episode_id, node, base_step, value, value, False, ()
                )
                staged_unit_events.append(event)
                staged_latest[node] = event
                staged_audit_events.append(event)
                unit_event_ids[node] = event_id

        logits: dict[str, float] = {}
        for node in self._topological_order:
            if node in self.input_nodes:
                continue
            contributions = [
                edge.weight * activations[edge.source]
                for edge in self._incoming[node]
            ]
            if not all(math.isfinite(value) for value in contributions):
                raise FloatingPointError("a non-finite forward contribution was produced")
            try:
                total = math.fsum(contributions)
            except OverflowError as exc:
                raise FloatingPointError(
                    "a non-finite forward preactivation was produced"
                ) from exc
            if not math.isfinite(total):
                raise FloatingPointError("a non-finite forward preactivation was produced")
            logits[node] = total
            value = math.tanh(total)
            if not math.isfinite(value):
                raise FloatingPointError("a non-finite forward activation was produced")
            activations[node] = (
                value if abs(value) >= self.activation_threshold else 0.0
            )
            parent_steps = [
                staged_latest[edge.source].step
                for edge in self._incoming[node]
                if abs(activations[edge.source]) >= self.activation_threshold
                and edge.source in staged_latest
            ]
            event_step = (max(parent_steps) + 1) if parent_steps else (base_step + 1)
            traces: list[EdgeTrace] = []
            for edge, contribution in zip(
                self._incoming[node], contributions, strict=True
            ):
                source_activity = activations[edge.source]
                if abs(source_activity) < self.activation_threshold:
                    continue
                omitted_total = total - contribution
                if not math.isfinite(omitted_total):
                    raise FloatingPointError(
                        "a non-finite omission preactivation was produced"
                    )
                omission_effect = activations[node] - math.tanh(omitted_total)
                weight_secant = (
                    omission_effect / edge.weight
                    if abs(edge.weight) >= self.epsilon
                    else 0.0
                )
                source_secant = (
                    omission_effect / source_activity
                    if abs(source_activity) >= self.epsilon
                    else 0.0
                )
                if not all(
                    math.isfinite(scalar)
                    for scalar in (omission_effect, weight_secant, source_secant)
                ):
                    raise FloatingPointError("a non-finite forward trace was produced")
                parent = unit_event_ids.get(
                    edge.source,
                    f"{episode_id}:{staged_step}:{edge.source}:unemitted",
                )
                trace = EdgeTrace(
                    edge_id=edge.edge_id,
                    parent_event_id=parent,
                    message_value=source_activity,
                    omission_effect=omission_effect,
                    weight_secant=weight_secant,
                    source_secant=source_secant,
                    created_step=event_step,
                )
                traces.append(trace)
                (
                    participation_trace,
                    omission_trace,
                    weight_secant_trace,
                    source_secant_trace,
                    _,
                    _,
                    _,
                ) = staged_edge_state[edge.edge_id]
                next_traces = (
                    participation_trace + source_activity,
                    omission_trace + omission_effect,
                    weight_secant_trace + weight_secant,
                    source_secant_trace + source_secant,
                )
                if not all(math.isfinite(scalar) for scalar in next_traces):
                    raise FloatingPointError("a non-finite accumulated trace was produced")
                staged_edge_state[edge.edge_id] = (
                    *next_traces,
                    True,
                    parent,
                    event_step,
                )
                staged_audit_events.append(
                    ForwardEvent(
                        episode_id=episode_id,
                        step=event_step,
                        edge_id=edge.edge_id,
                        source=edge.source,
                        destination=edge.destination,
                        source_activity=source_activity,
                        destination_activity=activations[node],
                        contribution=contribution,
                        trace_kind="participation",
                        participation_trace=next_traces[0],
                        omission_trace=next_traces[1],
                    )
                )
            event_id = f"{episode_id}:{event_step}:{node}"
            event = UnitEvent(
                event_id=event_id,
                episode_id=episode_id,
                node=node,
                step=event_step,
                preactivation=total,
                activation=activations[node],
                forced_output=node in self.output_nodes,
                edge_traces=tuple(traces),
            )
            staged_unit_events.append(event)
            staged_latest[node] = event
            staged_audit_events.append(event)
            unit_event_ids[node] = event_id
            staged_step = max(staged_step, event_step)

        selections: dict[str, str] = {}
        for group, nodes in self.output_groups.items():
            # tuple order is the deterministic tie breaker.
            selected = max(nodes, key=lambda node: (activations.get(node, 0.0), -nodes.index(node)))
            selections[group] = selected

        for edge in self.edges:
            (
                edge.participation_trace,
                edge.omission_trace,
                edge.weight_secant_trace,
                edge.source_secant_trace,
                edge.active_this_event,
                edge.last_event_id,
                edge.last_trace_step,
            ) = staged_edge_state[edge.edge_id]
        self._step = staged_step
        self._unit_events.extend(staged_unit_events)
        self._latest_event_by_node = staged_latest
        self._last_activations = activations
        self.event_log.extend(staged_audit_events)
        return ForwardResult(
            selections=selections,
            logits=logits,
            activations=activations,
            unit_event_ids=unit_event_ids,
        )

    def apply_credit(self, credits: Mapping[str, float], *, learning_rate: float) -> tuple[UpdateEvent, ...]:
        """Apply and locally route credit through the last causal traces.

        Each local share is proportional to the absolute leave-one-message-out
        omission effect.  Its update uses the locally stored weight secant.  The
        upstream route uses only the same trace's source secant sign.
        """

        episode_id = self._require_episode()
        learning_rate = _finite_float(learning_rate, name="learning_rate")
        if learning_rate < 0.0:
            raise ValueError("learning_rate must be non-negative")
        unknown = set(credits).difference(self.nodes)
        if unknown:
            raise KeyError(f"unknown credited nodes: {sorted(unknown)}")
        normalized_credits = {
            node: _finite_float(signal, name="credit signals")
            for node, signal in credits.items()
        }
        runtime_scalars = {
            "route_gain": self.route_gain,
            "epsilon": self.epsilon,
            "max_update": self.max_update,
            "weight_clip": self.weight_clip,
        }
        if not all(math.isfinite(value) for value in runtime_scalars.values()):
            raise FloatingPointError("credit configuration must remain finite")
        if self.epsilon <= 0.0:
            raise FloatingPointError("epsilon must remain positive")
        if not all(math.isfinite(edge.weight) for edge in self.edges):
            raise FloatingPointError("all current edge weights must be finite")

        node_credit = {node: 0.0 for node in self.nodes}
        staged_events: list[CreditEvent | CreditPacket | UpdateEvent] = []
        for node, signal in normalized_credits.items():
            node_credit[node] += signal
            staged_events.append(CreditEvent(episode_id, self._step, node, signal, "outcome"))
            event = self._latest_event_by_node.get(node)
            if event is not None:
                staged_events.append(CreditPacket(event.event_id, signal, self._step, 0))

        updates: list[UpdateEvent] = []
        staged_weights: list[tuple[Edge, float]] = []
        for node in reversed(self._topological_order):
            signal = node_credit[node]
            if not math.isfinite(signal):
                raise FloatingPointError("a non-finite routed credit was produced")
            if abs(signal) < 1e-15:
                continue
            incoming = self._incoming[node]
            if not incoming:
                continue
            local = [edge for edge in incoming if edge.active_this_event]
            if not all(
                math.isfinite(value)
                for edge in local
                for value in (
                    edge.omission_trace,
                    edge.weight_secant_trace,
                    edge.source_secant_trace,
                )
            ):
                raise FloatingPointError("active credit traces must be finite")
            normalizer = self.epsilon + math.fsum(abs(edge.omission_trace) for edge in local)
            if not math.isfinite(normalizer) or normalizer <= 0.0:
                raise FloatingPointError("the credit normalizer must be finite and positive")

            for edge in local:
                omission = edge.omission_trace
                if abs(omission) < self.epsilon:
                    continue
                old_weight = edge.weight
                share = abs(omission) / normalizer
                edge_credit = signal * share
                raw_delta = float(
                    np.clip(
                        learning_rate * edge_credit * edge.weight_secant_trace,
                        -self.max_update,
                        self.max_update,
                    )
                )
                new_weight = float(np.clip(old_weight + raw_delta, -self.weight_clip, self.weight_clip))
                if not all(
                    math.isfinite(value)
                    for value in (share, edge_credit, raw_delta, new_weight)
                ):
                    raise FloatingPointError("a non-finite weight candidate was produced")
                delta = new_weight - old_weight
                if edge.plastic:
                    staged_weights.append((edge, new_weight))
                else:
                    delta = 0.0
                    new_weight = old_weight
                update = UpdateEvent(
                    episode_id=episode_id,
                    step=self._step,
                    edge_id=edge.edge_id,
                    destination_credit=edge_credit,
                    trace_kind="participation",
                    trace_value=edge.weight_secant_trace,
                    delta=delta,
                    old_weight=old_weight,
                    new_weight=new_weight,
                )
                updates.append(update)
                staged_events.append(update)

                direction = 1.0 if edge.source_secant_trace >= 0.0 else -1.0
                routed = self.route_gain * edge_credit * direction
                node_credit[edge.source] += routed
                if not math.isfinite(routed) or not math.isfinite(node_credit[edge.source]):
                    raise FloatingPointError("a non-finite routed credit was produced")
                staged_events.append(
                    CreditEvent(episode_id, self._step, edge.source, routed, "routed")
                )

        for edge, new_weight in staged_weights:
            edge.weight = new_weight
        self.event_log.extend(staged_events)
        return tuple(updates)

    def apply_eligibility_credit(
        self, credits: Mapping[str, float], *, learning_rate: float
    ) -> tuple[UpdateEvent, ...]:
        """Same-mask eligibility-trace baseline without causal share routing."""

        episode_id = self._require_episode()
        learning_rate = _finite_float(learning_rate, name="learning_rate")
        if learning_rate < 0.0:
            raise ValueError("learning_rate must be non-negative")
        unknown = set(credits).difference(self.nodes)
        if unknown:
            raise KeyError(f"unknown credited nodes: {sorted(unknown)}")
        normalized_credits = {
            node: _finite_float(signal, name="credit signals")
            for node, signal in credits.items()
        }
        runtime_scalars = (self.epsilon, self.max_update, self.weight_clip)
        if not all(math.isfinite(value) for value in runtime_scalars):
            raise FloatingPointError("eligibility configuration must remain finite")
        if self.epsilon <= 0.0:
            raise FloatingPointError("epsilon must remain positive")
        if not all(math.isfinite(edge.weight) for edge in self.edges):
            raise FloatingPointError("all current edge weights must be finite")

        updates: list[UpdateEvent] = []
        staged_events: list[CreditEvent | UpdateEvent] = []
        staged_weights: list[tuple[Edge, float]] = []
        for node, signal in normalized_credits.items():
            staged_events.append(CreditEvent(episode_id, self._step, node, signal, "outcome"))
            for edge in self._incoming[node]:
                if not math.isfinite(edge.weight_secant_trace):
                    raise FloatingPointError("active eligibility traces must be finite")
                if not edge.active_this_event or abs(edge.weight_secant_trace) < self.epsilon:
                    continue
                old_weight = edge.weight
                raw_delta = float(
                    np.clip(
                        learning_rate * signal * edge.weight_secant_trace,
                        -self.max_update,
                        self.max_update,
                    )
                )
                new_weight = float(np.clip(old_weight + raw_delta, -self.weight_clip, self.weight_clip))
                if not all(math.isfinite(value) for value in (raw_delta, new_weight)):
                    raise FloatingPointError("a non-finite weight candidate was produced")
                delta = new_weight - old_weight
                if edge.plastic:
                    staged_weights.append((edge, new_weight))
                else:
                    delta = 0.0
                    new_weight = old_weight
                update = UpdateEvent(
                    episode_id,
                    self._step,
                    edge.edge_id,
                    signal,
                    "participation",
                    edge.weight_secant_trace,
                    delta,
                    old_weight,
                    new_weight,
                )
                updates.append(update)
                staged_events.append(update)

        for edge, new_weight in staged_weights:
            edge.weight = new_weight
        self.event_log.extend(staged_events)
        return tuple(updates)

    def edge_weights(self) -> dict[str, float]:
        return {edge.edge_id: edge.weight for edge in self.edges}

    def topology_signature(self) -> tuple[tuple[str, str, str], ...]:
        return tuple((edge.edge_id, edge.source, edge.destination) for edge in self.edges)

    @property
    def unit_events(self) -> tuple[UnitEvent, ...]:
        return tuple(self._unit_events)

    def grow_edge(self, *_: object, **__: object) -> None:
        raise NotImplementedError("topology growth is intentionally deferred until fixed-graph tests pass")

    def prune_edge(self, *_: object, **__: object) -> None:
        raise NotImplementedError("topology pruning is intentionally deferred until fixed-graph tests pass")
