"""Timestamped recurrent event graph for the delayed-credit smoke test.

This module is deliberately independent of an automatic-differentiation runtime.
The structural graph may contain cycles, but every edge has a one-tick delay.  A
realised :class:`UnitEvent` therefore only points to parent events from an earlier
tick, making the *event ancestry* acyclic even when the structural mask is not.

The implementation follows the frozen CCF-v0 vocabulary in ``docs/CCF_V0.md``:

* an emitted unit stores immutable, per-message ``EdgeTrace`` records;
* the trace contains a leave-one-message-out effect and two local secants;
* terminal credit is addressed to one exact output-event identifier;
* reconvergent packets are aggregated before events are processed newest-first;
* updates and routes use only the immutable event record and the local edge.

It is a research simulator, not a claim that the candidate rule is novel or that
the smoke-test task has been solved.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from .events import (
    CreditEvent,
    CreditPacket,
    EdgeTrace,
    EventLog,
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


@dataclass(frozen=True, slots=True)
class RecurrentEdge:
    """One immutable member of the frozen structural mask.

    Weights live in a separate private mapping so callers cannot accidentally
    change an endpoint while training.  ``delay_ticks`` is fixed to one for the
    first recurrent experiment.
    """

    edge_id: str
    source: str
    destination: str
    kind: str
    delay_ticks: int = 1
    plastic: bool = True


@dataclass(frozen=True, slots=True)
class EventMessage:
    """A scalar message scheduled by one exact parent event."""

    parent_event_id: str
    edge_id: str
    value: float
    arrival_tick: int


@dataclass(frozen=True, slots=True)
class RecurrentStepResult:
    """Read-only summary returned by :meth:`RecurrentEventGraph.step`."""

    tick: int
    inputs: tuple[tuple[str, float], ...]
    emitted_event_ids: tuple[str, ...]
    activations: tuple[tuple[str, float], ...]
    scheduled_message_count: int


@dataclass(frozen=True, slots=True)
class QueryResult:
    """The forced output event produced by :meth:`RecurrentEventGraph.query`."""

    tick: int
    event_id: str
    activation: float
    prediction: int


_LEDGER_FIELDS = (
    "episodes",
    "ticks",
    "source_events",
    "arrived_messages",
    "forward_edge_touches",
    "activation_evaluations",
    "omission_evaluations",
    "emitted_unit_events",
    "credit_event_touches",
    "credit_edge_touches",
    "weight_write_touches",
    "clipped_updates",
    "expired_event_records",
    "credit_stop_expired",
    "credit_stop_hop_limit",
    "credit_stop_small",
    "credit_stop_source",
    "credit_stop_zero_omission",
    "credit_stop_missing_parent",
    "peak_retained_event_records",
    "peak_pending_messages",
    "nonfinite_values",
)


class RecurrentEventGraph:
    """A deterministic sparse recurrent graph with timestamped local CCF traces.

    Use :func:`build_experiment_000_graph` rather than constructing this class by
    hand in ordinary experiments.  The constructor remains public so unit tests
    can exercise small masks.
    """

    def __init__(
        self,
        *,
        input_nodes: Iterable[str],
        hidden_nodes: Iterable[str],
        output_node: str,
        edges: Iterable[RecurrentEdge],
        initial_weights: Mapping[str, float],
        mode: str = "full",
        learning_rate: float = 0.01,
        trace_decay: float = 0.97,
        route_gain: float = 0.90,
        emit_threshold: float = 1e-3,
        epsilon_weight: float = 1e-12,
        epsilon_message: float = 1e-12,
        epsilon_route: float = 1e-12,
        credit_limit: float = 1.0,
        credit_minimum: float = 1e-8,
        trace_horizon: int = 32,
        hop_limit: int = 16,
        max_update: float = 0.05,
        weight_clip: float = 3.0,
        event_log: EventLog | None = None,
    ) -> None:
        normalized_mode = mode.lower().replace("-", "_")
        if normalized_mode in {"ccf", "ccf_v0"}:
            normalized_mode = "full"
        if normalized_mode in {"ccf_no_trace", "notrace"}:
            normalized_mode = "no_trace"
        if normalized_mode not in {"full", "no_trace"}:
            raise ValueError("mode must be 'full' or 'no_trace'")
        learning_rate = _finite_float(learning_rate, name="learning_rate")
        trace_decay = _finite_float(trace_decay, name="trace_decay")
        route_gain = _finite_float(route_gain, name="route_gain")
        emit_threshold = _finite_float(emit_threshold, name="emit_threshold")
        epsilon_weight = _finite_float(epsilon_weight, name="epsilon_weight")
        epsilon_message = _finite_float(epsilon_message, name="epsilon_message")
        epsilon_route = _finite_float(epsilon_route, name="epsilon_route")
        credit_limit = _finite_float(credit_limit, name="credit_limit")
        credit_minimum = _finite_float(credit_minimum, name="credit_minimum")
        trace_horizon_scalar = _finite_float(trace_horizon, name="trace_horizon")
        hop_limit_scalar = _finite_float(hop_limit, name="hop_limit")
        max_update = _finite_float(max_update, name="max_update")
        weight_clip = _finite_float(weight_clip, name="weight_clip")
        if not 0.0 <= trace_decay <= 1.0:
            raise ValueError("trace_decay must be in [0, 1]")
        if not 0.0 <= route_gain <= 1.0:
            raise ValueError("route_gain must be in [0, 1]")
        if learning_rate < 0.0:
            raise ValueError("learning_rate must be non-negative")
        if emit_threshold < 0.0:
            raise ValueError("emit_threshold must be non-negative")
        if trace_horizon_scalar < 1 or hop_limit_scalar < 1:
            raise ValueError("trace_horizon and hop_limit must be positive")
        if max_update <= 0.0 or weight_clip <= 0.0 or credit_limit <= 0.0:
            raise ValueError("clip limits must be positive")
        if min(epsilon_weight, epsilon_message, epsilon_route) <= 0.0:
            raise ValueError("epsilon values must be positive")
        if credit_minimum < 0.0:
            raise ValueError("credit_minimum must be non-negative")

        self.input_nodes = tuple(input_nodes)
        self.hidden_nodes = tuple(hidden_nodes)
        self.output_node = str(output_node)
        self.nodes = self.input_nodes + self.hidden_nodes + (self.output_node,)
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("all node names must be unique")

        self.edges = tuple(edges)
        if len({edge.edge_id for edge in self.edges}) != len(self.edges):
            raise ValueError("edge identifiers must be unique")
        node_set = set(self.nodes)
        if any(edge.source not in node_set or edge.destination not in node_set for edge in self.edges):
            raise ValueError("every edge endpoint must name a graph node")
        if any(edge.delay_ticks != 1 for edge in self.edges):
            raise ValueError("Experiment 000 requires every edge delay to equal one tick")
        if any(edge.destination in self.input_nodes for edge in self.edges):
            raise ValueError("input/source nodes cannot have incoming graph edges")
        if any(edge.source == self.output_node for edge in self.edges):
            raise ValueError("the scalar output node must be a sink")
        if any(edge.source == edge.destination for edge in self.edges):
            raise ValueError("recurrent self-edges are excluded")

        supplied = {str(key): float(value) for key, value in initial_weights.items()}
        expected_ids = {edge.edge_id for edge in self.edges}
        if set(supplied) != expected_ids:
            missing = sorted(expected_ids.difference(supplied))
            extra = sorted(set(supplied).difference(expected_ids))
            raise ValueError(f"weights must match the edge mask; missing={missing}, extra={extra}")
        if not all(math.isfinite(value) for value in supplied.values()):
            raise ValueError("all initial weights must be finite")

        self.mode = normalized_mode
        self.learning_rate = learning_rate
        self.trace_decay = trace_decay
        self.route_gain = route_gain
        self.emit_threshold = emit_threshold
        self.epsilon_weight = epsilon_weight
        self.epsilon_message = epsilon_message
        self.epsilon_route = epsilon_route
        self.credit_limit = credit_limit
        self.credit_minimum = credit_minimum
        self.trace_horizon = int(trace_horizon_scalar)
        self.hop_limit = int(hop_limit_scalar)
        self.max_update = max_update
        self.weight_clip = weight_clip
        self.event_log = event_log if event_log is not None else EventLog()

        self._weights = supplied
        self._edge_by_id = {edge.edge_id: edge for edge in self.edges}
        self._incoming: dict[str, tuple[RecurrentEdge, ...]] = {}
        self._outgoing: dict[str, tuple[RecurrentEdge, ...]] = {}
        for node in self.nodes:
            self._incoming[node] = tuple(
                sorted((edge for edge in self.edges if edge.destination == node), key=lambda edge: edge.edge_id)
            )
            self._outgoing[node] = tuple(
                sorted((edge for edge in self.edges if edge.source == node), key=lambda edge: edge.edge_id)
            )

        self._initial_topology_hash = self.topology_hash()
        self._ledger = {field: 0 for field in _LEDGER_FIELDS}
        self._episode_serial = 0
        self._active_episode: str | None = None
        self._tick = -1
        self._event_sequence = {node: 0 for node in self.nodes}
        self._pending: dict[int, list[EventMessage]] = {}
        self._event_store: dict[str, UnitEvent] = {}
        self._event_history: list[UnitEvent] = []
        self._credit_packets: list[CreditPacket] = []
        self._last_emitted = {node: 0.0 for node in self.nodes}
        self._activations = {node: 0.0 for node in self.nodes}
        self._latest_output_event_id: str | None = None
        self._latest_query: QueryResult | None = None

    # ------------------------------------------------------------------
    # Episode and forward-event handling

    def begin_episode(self, episode_id: str | int | None = None) -> str:
        """Clear episode-local dynamics and traces while preserving learned weights."""

        self._episode_serial += 1
        resolved = str(episode_id) if episode_id is not None else f"episode-{self._episode_serial}"
        self._active_episode = resolved
        self._tick = -1
        self._event_sequence = {node: 0 for node in self.nodes}
        self._pending = {}
        self._event_store = {}
        self._event_history = []
        self._credit_packets = []
        self._last_emitted = {node: 0.0 for node in self.nodes}
        self._activations = {node: 0.0 for node in self.nodes}
        self._latest_output_event_id = None
        self._latest_query = None
        self._ledger["episodes"] += 1
        return resolved

    def _require_episode(self) -> str:
        if self._active_episode is None:
            raise RuntimeError("call begin_episode() before step(), query(), or credit")
        return self._active_episode

    def _next_event_id(self, node: str) -> str:
        episode = self._require_episode()
        sequence = self._event_sequence[node]
        self._event_sequence[node] += 1
        return f"{episode}:{self._tick}:{node}:{sequence}"

    def _record_event(self, event: UnitEvent) -> None:
        self._event_store[event.event_id] = event
        self._event_history.append(event)
        self.event_log.append(event)
        self._ledger["emitted_unit_events"] += 1
        self._ledger["peak_retained_event_records"] = max(
            self._ledger["peak_retained_event_records"], len(self._event_store)
        )

    def _schedule_from(self, source: str, value: float, parent_event_id: str) -> int:
        count = 0
        for edge in self._outgoing[source]:
            message = EventMessage(
                parent_event_id=parent_event_id,
                edge_id=edge.edge_id,
                value=float(value),
                arrival_tick=self._tick + edge.delay_ticks,
            )
            self._pending.setdefault(message.arrival_tick, []).append(message)
            count += 1
        pending_count = sum(len(messages) for messages in self._pending.values())
        self._ledger["peak_pending_messages"] = max(
            self._ledger["peak_pending_messages"], pending_count
        )
        return count

    def _source_event(self, node: str, value: float) -> tuple[UnitEvent, int]:
        event = UnitEvent(
            event_id=self._next_event_id(node),
            episode_id=self._require_episode(),
            node=node,
            step=self._tick,
            preactivation=value,
            activation=value,
            forced_output=False,
            edge_traces=(),
        )
        self._activations[node] = value
        self._last_emitted[node] = value
        self._record_event(event)
        self._ledger["source_events"] += 1
        return event, self._schedule_from(node, value, event.event_id)

    def _process_unit(
        self,
        node: str,
        messages: tuple[EventMessage, ...],
        *,
        forced_output: bool,
    ) -> tuple[UnitEvent | None, int]:
        contributions: list[float] = []
        for message in messages:
            weight = self._weights[message.edge_id]
            contributions.append(weight * message.value)
        self._ledger["arrived_messages"] += len(messages)
        self._ledger["forward_edge_touches"] += len(messages)
        total = math.fsum(contributions)
        activation = math.tanh(total)
        self._ledger["activation_evaluations"] += 1
        if not math.isfinite(total) or not math.isfinite(activation):
            self._ledger["nonfinite_values"] += 1
            raise FloatingPointError(f"non-finite recurrent state at node {node!r}, tick {self._tick}")
        self._activations[node] = activation

        should_emit = forced_output or abs(activation - self._last_emitted[node]) >= self.emit_threshold
        if not should_emit:
            return None, 0

        traces: list[EdgeTrace] = []
        for message in messages:
            edge = self._edge_by_id[message.edge_id]
            weight = self._weights[edge.edge_id]
            omission_activation = math.tanh(total - weight * message.value)
            omission_effect = activation - omission_activation
            weight_secant = (
                omission_effect / weight if abs(weight) >= self.epsilon_weight else 0.0
            )
            source_secant = (
                omission_effect / message.value
                if abs(message.value) >= self.epsilon_message
                else 0.0
            )
            self._ledger["omission_evaluations"] += 1
            traces.append(
                EdgeTrace(
                    edge_id=edge.edge_id,
                    parent_event_id=message.parent_event_id,
                    message_value=message.value,
                    omission_effect=omission_effect,
                    weight_secant=weight_secant,
                    source_secant=source_secant,
                    created_step=self._tick,
                )
            )

        event = UnitEvent(
            event_id=self._next_event_id(node),
            episode_id=self._require_episode(),
            node=node,
            step=self._tick,
            preactivation=total,
            activation=activation,
            forced_output=forced_output,
            edge_traces=tuple(traces),
        )
        self._last_emitted[node] = activation
        self._record_event(event)
        if node == self.output_node:
            self._latest_output_event_id = event.event_id
        return event, self._schedule_from(node, activation, event.event_id)

    def _expire_old_events(self) -> None:
        expired = [
            event_id
            for event_id, event in self._event_store.items()
            if self._tick - event.step > self.trace_horizon
        ]
        for event_id in expired:
            del self._event_store[event_id]
        self._ledger["expired_event_records"] += len(expired)

    def _advance(
        self,
        inputs: Mapping[str, float],
        *,
        force_output: bool = False,
    ) -> RecurrentStepResult:
        self._require_episode()
        unknown = set(inputs).difference(self.input_nodes)
        if unknown:
            raise KeyError(f"unknown input channels: {sorted(unknown)}")
        normalized_inputs: dict[str, float] = {}
        for node, raw_value in inputs.items():
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("input values must be finite")
            normalized_inputs[node] = value

        self._tick += 1
        self._ledger["ticks"] += 1
        emitted: list[str] = []
        scheduled_count = 0

        arrivals = sorted(
            self._pending.pop(self._tick, []),
            key=lambda message: (self._edge_by_id[message.edge_id].destination, message.edge_id),
        )
        by_destination: dict[str, list[EventMessage]] = {}
        for message in arrivals:
            by_destination.setdefault(self._edge_by_id[message.edge_id].destination, []).append(message)

        # All arrived messages came from tick-1.  Process them without allowing
        # any just-emitted message to arrive in this same tick.
        for node in self.hidden_nodes + (self.output_node,):
            messages = tuple(by_destination.get(node, ()))
            if not messages and not (force_output and node == self.output_node):
                continue
            event, scheduled = self._process_unit(
                node,
                messages,
                forced_output=force_output and node == self.output_node,
            )
            scheduled_count += scheduled
            if event is not None:
                emitted.append(event.event_id)

        # External source events are also timestamped at this tick, but their
        # graph messages are delayed until tick+1 like every other edge.
        for node in self.input_nodes:
            if node not in normalized_inputs:
                continue
            value = normalized_inputs[node]
            if abs(value) < self.epsilon_message:
                continue
            event, scheduled = self._source_event(node, value)
            scheduled_count += scheduled
            emitted.append(event.event_id)

        self._expire_old_events()
        return RecurrentStepResult(
            tick=self._tick,
            inputs=tuple((node, normalized_inputs[node]) for node in self.input_nodes if node in normalized_inputs),
            emitted_event_ids=tuple(emitted),
            activations=tuple((node, self._activations[node]) for node in self.hidden_nodes + (self.output_node,)),
            scheduled_message_count=scheduled_count,
        )

    def step(self, inputs: Mapping[str, float]) -> RecurrentStepResult:
        """Advance one real tick using only messages whose one-tick delay elapsed."""

        return self._advance(inputs)

    def query(self) -> QueryResult:
        """Inject ``QUERY`` and return the forced output two graph delays later.

        The default builder connects each source to hidden units and hidden units
        to the output.  Consequently a query source event needs one tick to reach
        hidden units and one more tick to reach the output.  The output event at
        that second arrival tick is forced and becomes the sole root for terminal
        supervised credit.
        """

        if "query" not in self.input_nodes:
            raise RuntimeError("this graph has no 'query' input channel")
        self._advance({"query": 1.0})
        self._advance({})
        self._advance({}, force_output=True)
        if self._latest_output_event_id is None:  # pragma: no cover - defensive invariant
            raise RuntimeError("query did not create a forced output event")
        event = self._event_store[self._latest_output_event_id]
        result = QueryResult(
            tick=event.step,
            event_id=event.event_id,
            activation=event.activation,
            prediction=1 if event.activation >= 0.0 else 0,
        )
        self._latest_query = result
        return result

    @property
    def predict(self) -> int:
        """Binary prediction from the most recent forced query event."""

        if self._latest_query is None:
            raise RuntimeError("call query() before reading predict")
        return self._latest_query.prediction

    # ------------------------------------------------------------------
    # Delayed causal credit

    @staticmethod
    def _credit_target(target: float | bool) -> float:
        value = float(target)
        if not math.isfinite(value):
            raise ValueError("supervised target must be finite")
        if value == 0.0:
            return -1.0
        if value == 1.0:
            return 1.0
        if -1.0 <= value <= 1.0:
            return value
        raise ValueError("target must be a bit or a bipolar scalar in [-1, 1]")

    @staticmethod
    def _sign(value: float) -> float:
        return 1.0 if value > 0.0 else (-1.0 if value < 0.0 else 0.0)

    def _record_packet(self, packet: CreditPacket) -> None:
        self._credit_packets.append(packet)
        self.event_log.append(packet)

    def apply_supervised_credit(self, target: float | bool) -> tuple[UpdateEvent, ...]:
        """Apply a terminal activation residual through exact parent-event IDs.

        In ``no_trace`` mode, the forced output event remains available so its
        immediate incoming weights can update, while every earlier event record is
        removed before routing.  This is the preregistered CCF-NO-TRACE ablation.
        """

        episode = self._require_episode()
        if self._latest_query is None:
            raise RuntimeError("terminal credit is allowed only after query()")
        target_activation = self._credit_target(target)

        runtime_scalars = {
            "learning_rate": self.learning_rate,
            "trace_decay": self.trace_decay,
            "route_gain": self.route_gain,
            "epsilon_route": self.epsilon_route,
            "credit_limit": self.credit_limit,
            "credit_minimum": self.credit_minimum,
            "max_update": self.max_update,
            "weight_clip": self.weight_clip,
        }
        if not all(math.isfinite(value) for value in runtime_scalars.values()):
            raise FloatingPointError("credit configuration must remain finite")
        if self.learning_rate < 0.0:
            raise FloatingPointError("learning_rate must remain non-negative")
        if self.epsilon_route <= 0.0:
            raise FloatingPointError("epsilon_route must remain positive")
        if not all(math.isfinite(value) for value in self._weights.values()):
            raise FloatingPointError("all current edge weights must be finite")
        if not math.isfinite(self._latest_query.activation):
            raise FloatingPointError("the queried activation must be finite")

        query_event_id = self._latest_query.event_id
        root_event = self._event_store.get(query_event_id)
        root_signal = float(
            np.clip(
                target_activation - self._latest_query.activation,
                -self.credit_limit,
                self.credit_limit,
            )
        )
        if not math.isfinite(root_signal):
            raise FloatingPointError("a non-finite root credit was produced")

        staged_events: list[CreditEvent | CreditPacket | UpdateEvent] = [
            CreditEvent(episode, self._tick, self.output_node, root_signal, "outcome")
        ]
        root_packet = CreditPacket(query_event_id, root_signal, self._tick, 0)
        staged_events.append(root_packet)
        staged_packets = [root_packet]
        ledger_delta = {field: 0 for field in _LEDGER_FIELDS}

        def increment(field: str, amount: int = 1) -> None:
            ledger_delta[field] += amount

        # An outcome may itself be delayed.  Expired records are never rebuilt
        # from current activations or searched for via the structural graph.
        if root_event is None:
            increment("credit_event_touches")
            increment("credit_stop_expired")
            self._credit_packets.extend(staged_packets)
            for field, amount in ledger_delta.items():
                self._ledger[field] += amount
            self.event_log.extend(staged_events)
            return ()

        staged_event_store = self._event_store
        if self.mode == "no_trace":
            removed = len(self._event_store) - 1
            staged_event_store = {root_event.event_id: root_event}
            increment("expired_event_records", max(removed, 0))

        aggregated: dict[str, float] = {root_event.event_id: root_signal}
        hops: dict[str, int] = {root_event.event_id: 0}
        event_order = sorted(
            staged_event_store.values(),
            key=lambda event: (event.step, event.event_id),
            reverse=True,
        )
        updates: list[UpdateEvent] = []
        staged_weights = dict(self._weights)

        for event in event_order:
            if event.event_id not in aggregated:
                continue
            signal = float(
                np.clip(aggregated[event.event_id], -self.credit_limit, self.credit_limit)
            )
            if not math.isfinite(signal):
                raise FloatingPointError("a non-finite routed credit was produced")
            hop_count = hops[event.event_id]
            increment("credit_event_touches")

            if self._tick - event.step > self.trace_horizon:
                increment("credit_stop_expired")
                continue
            if hop_count >= self.hop_limit:
                increment("credit_stop_hop_limit")
                continue
            if abs(signal) < self.credit_minimum:
                increment("credit_stop_small")
                continue
            if not event.edge_traces:
                increment("credit_stop_source")
                continue
            if not all(
                math.isfinite(value)
                for trace in event.edge_traces
                for value in (
                    trace.message_value,
                    trace.omission_effect,
                    trace.weight_secant,
                    trace.source_secant,
                )
            ):
                raise FloatingPointError("credit traces must remain finite")

            denominator = self.epsilon_route + math.fsum(
                abs(trace.omission_effect) for trace in event.edge_traces
            )
            if not math.isfinite(denominator) or denominator <= 0.0:
                raise FloatingPointError("the credit normalizer must be finite and positive")
            if denominator <= self.epsilon_route:
                increment("credit_stop_zero_omission")
                continue

            for trace in sorted(event.edge_traces, key=lambda item: item.edge_id):
                if abs(trace.omission_effect) < self.epsilon_route:
                    continue
                increment("credit_edge_touches")
                share = abs(trace.omission_effect) / denominator
                edge_credit = signal * share
                edge = self._edge_by_id[trace.edge_id]
                old_weight = staged_weights[edge.edge_id]
                age = self._tick - event.step
                proposal = (
                    self.learning_rate
                    * (self.trace_decay**age)
                    * edge_credit
                    * trace.weight_secant
                )
                if not all(math.isfinite(value) for value in (share, edge_credit, proposal)):
                    raise FloatingPointError("a non-finite weight proposal was produced")
                bounded_delta = float(np.clip(proposal, -self.max_update, self.max_update))
                if bounded_delta != proposal:
                    increment("clipped_updates")
                candidate = float(
                    np.clip(
                        old_weight + bounded_delta,
                        -self.weight_clip,
                        self.weight_clip,
                    )
                )
                if not all(math.isfinite(value) for value in (bounded_delta, candidate)):
                    raise FloatingPointError("a non-finite weight candidate was produced")
                if candidate != old_weight + bounded_delta:
                    increment("clipped_updates")
                new_weight = candidate if edge.plastic else old_weight
                delta = new_weight - old_weight
                if edge.plastic:
                    staged_weights[edge.edge_id] = new_weight
                    increment("weight_write_touches")
                update = UpdateEvent(
                    episode_id=episode,
                    step=self._tick,
                    edge_id=edge.edge_id,
                    destination_credit=edge_credit,
                    trace_kind="omission",
                    trace_value=trace.weight_secant,
                    delta=delta,
                    old_weight=old_weight,
                    new_weight=new_weight,
                )
                updates.append(update)
                staged_events.append(update)

                routed = self.route_gain * edge_credit * self._sign(trace.source_secant)
                if not math.isfinite(routed):
                    raise FloatingPointError("a non-finite routed credit was produced")
                # Do not threshold an individual route here.  Several small
                # packets can address the same parent event; CCF-v0 requires
                # their signed sum to be formed first and applies c_min only
                # when that parent event is processed (Section 7.1).  Exact
                # zeros are the sole no-op because they cannot affect a sum.
                if routed == 0.0:
                    continue
                parent = staged_event_store.get(trace.parent_event_id)
                if parent is None:
                    increment("credit_stop_missing_parent")
                    continue
                # The strict inequality is the core event-ancestry invariant.
                if parent.step >= event.step:
                    raise RuntimeError("causal trace points to a non-earlier parent event")
                aggregate = aggregated.get(parent.event_id, 0.0) + routed
                if not math.isfinite(aggregate):
                    raise FloatingPointError("a non-finite routed credit was produced")
                aggregated[parent.event_id] = aggregate
                next_hop = hop_count + 1
                hops[parent.event_id] = min(hops.get(parent.event_id, next_hop), next_hop)
                staged_events.append(
                    CreditEvent(episode, self._tick, parent.node, routed, "routed")
                )
                packet = CreditPacket(parent.event_id, routed, self._tick, next_hop)
                staged_packets.append(packet)
                staged_events.append(packet)

        if not all(math.isfinite(value) for value in staged_weights.values()):
            raise FloatingPointError("a non-finite weight was produced by delayed credit")

        self._weights = staged_weights
        if self.mode == "no_trace":
            self._event_store = staged_event_store
        self._credit_packets.extend(staged_packets)
        for field, amount in ledger_delta.items():
            self._ledger[field] += amount
        self.event_log.extend(staged_events)
        return tuple(updates)

    # ------------------------------------------------------------------
    # Reproducibility, inspection, and compute accounting

    @property
    def weights(self) -> dict[str, float]:
        """A defensive copy of the mutable weight state."""

        return dict(self._weights)

    def edge_weights(self) -> dict[str, float]:
        """Compatibility spelling shared with the Stage-1 fixed-DAG scaffold."""

        return self.weights

    def topology_signature(self) -> tuple[tuple[str, str, str, int, str], ...]:
        return tuple(
            (edge.edge_id, edge.source, edge.destination, edge.delay_ticks, edge.kind)
            for edge in self.edges
        )

    def topology_hash(self) -> str:
        payload = json.dumps(
            self.topology_signature(), sort_keys=False, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest().upper()

    @property
    def topology_sha256(self) -> str:
        return self.topology_hash()

    def weights_hash(self) -> str:
        payload = json.dumps(
            sorted(self._weights.items()), separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest().upper()

    @property
    def ledger(self) -> dict[str, int]:
        return dict(self._ledger)

    def ledger_snapshot(self) -> dict[str, int]:
        return self.ledger

    @property
    def unit_events(self) -> tuple[UnitEvent, ...]:
        return tuple(self._event_history)

    @property
    def unit_events_by_id(self) -> dict[str, UnitEvent]:
        """Current-episode event history keyed by immutable event identifier."""

        return {event.event_id: event for event in self._event_history}

    @property
    def edges_by_id(self) -> dict[str, RecurrentEdge]:
        """A defensive lookup for inspecting the frozen mask."""

        return dict(self._edge_by_id)

    @property
    def credit_packets(self) -> tuple[CreditPacket, ...]:
        return tuple(self._credit_packets)

    @property
    def audit_records(self) -> tuple[dict[str, object], ...]:
        return self.event_log.records

    @property
    def audit(self) -> dict[str, object]:
        return {
            "episode_id": self._active_episode,
            "tick": self._tick,
            "mode": self.mode,
            "topology_sha256": self.topology_hash(),
            "topology_unchanged": self.topology_hash() == self._initial_topology_hash,
            "weights_sha256": self.weights_hash(),
            "retained_event_count": len(self._event_store),
            "latest_output_event_id": self._latest_output_event_id,
            "queried_output_event_id": None if self._latest_query is None else self._latest_query.event_id,
            "nonfinite_values": self._ledger["nonfinite_values"],
            "ledger": self.ledger,
        }


def _repair_outgoing_coverage(
    recurrent_sources: dict[str, list[str]],
    hidden_nodes: tuple[str, ...],
    output_sources: tuple[str, ...],
) -> None:
    """Repair orphan sources without changing any destination's in-degree."""

    outgoing_count = {node: int(node in output_sources) for node in hidden_nodes}
    for sources in recurrent_sources.values():
        for source in sources:
            outgoing_count[source] += 1

    for orphan in tuple(node for node in hidden_nodes if outgoing_count[node] == 0):
        repaired = False
        for destination in hidden_nodes:
            sources = recurrent_sources[destination]
            if destination == orphan or orphan in sources:
                continue
            for index, displaced in enumerate(tuple(sources)):
                if outgoing_count[displaced] <= 1:
                    continue
                sources[index] = orphan
                outgoing_count[orphan] += 1
                outgoing_count[displaced] -= 1
                sources.sort()
                repaired = True
                break
            if repaired:
                break
        if not repaired:  # pragma: no cover - impossible for valid default degrees
            raise RuntimeError(f"could not repair outgoing coverage for {orphan}")


def build_experiment_000_graph(
    seed: int,
    *,
    mode: str = "full",
    hidden_count: int = 64,
    recurrent_in_degree: int = 8,
    input_fan_out: int = 8,
    output_fan_in: int = 16,
    input_nodes: Iterable[str] = ("cue", "noise", "query"),
    output_node: str = "output",
    initial_weight_scale: float = 0.35,
    event_log_enabled: bool = True,
    **graph_options: object,
) -> RecurrentEventGraph:
    """Build the seeded cue/noise/query graph frozen for Experiment 000.

    Connectivity and initial weights use separate deterministic random streams.
    Every hidden unit has exactly ``recurrent_in_degree`` recurrent inputs, every
    input channel projects to exactly ``input_fan_out`` hidden units, and the
    output receives exactly ``output_fan_in`` hidden inputs.
    """

    if hidden_count < 2:
        raise ValueError("hidden_count must be at least two")
    if not 1 <= recurrent_in_degree < hidden_count:
        raise ValueError("recurrent_in_degree must be in [1, hidden_count)")
    if not 1 <= input_fan_out <= hidden_count:
        raise ValueError("input_fan_out must be in [1, hidden_count]")
    if not 1 <= output_fan_in <= hidden_count:
        raise ValueError("output_fan_in must be in [1, hidden_count]")
    if initial_weight_scale <= 0.0:
        raise ValueError("initial_weight_scale must be positive")

    channels = tuple(str(node) for node in input_nodes)
    if len(set(channels)) != len(channels):
        raise ValueError("input channel names must be unique")
    for required in ("cue", "noise", "query"):
        if required not in channels:
            raise ValueError(f"Experiment 000 requires the {required!r} channel")

    hidden = tuple(f"h{index:03d}" for index in range(hidden_count))
    if output_node in channels or output_node in hidden:
        raise ValueError("output node name must be unique")

    mask_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xCCF, 0]))
    weight_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xCCF, 1]))

    input_targets: dict[str, tuple[str, ...]] = {}
    for channel in channels:
        chosen = mask_rng.choice(hidden_count, size=input_fan_out, replace=False)
        input_targets[channel] = tuple(sorted(hidden[int(index)] for index in chosen))

    recurrent_sources: dict[str, list[str]] = {}
    for destination_index, destination in enumerate(hidden):
        candidates = np.delete(np.arange(hidden_count), destination_index)
        chosen = mask_rng.choice(candidates, size=recurrent_in_degree, replace=False)
        recurrent_sources[destination] = sorted(hidden[int(index)] for index in chosen)

    output_indices = mask_rng.choice(hidden_count, size=output_fan_in, replace=False)
    output_sources = tuple(sorted(hidden[int(index)] for index in output_indices))
    _repair_outgoing_coverage(recurrent_sources, hidden, output_sources)

    edges: list[RecurrentEdge] = []
    for channel in channels:
        for destination in input_targets[channel]:
            edges.append(
                RecurrentEdge(
                    edge_id=f"input:{channel}->{destination}",
                    source=channel,
                    destination=destination,
                    kind="input",
                )
            )
    for destination in hidden:
        for source in recurrent_sources[destination]:
            edges.append(
                RecurrentEdge(
                    edge_id=f"recurrent:{source}->{destination}",
                    source=source,
                    destination=destination,
                    kind="recurrent",
                )
            )
    for source in output_sources:
        edges.append(
            RecurrentEdge(
                edge_id=f"output:{source}->{output_node}",
                source=source,
                destination=output_node,
                kind="output",
            )
        )

    weights: dict[str, float] = {}
    for edge in edges:
        if edge.kind == "recurrent":
            fan_in = recurrent_in_degree
        elif edge.kind == "output":
            fan_in = output_fan_in
        else:
            fan_in = max(1, len(channels))
        value = float(weight_rng.normal(0.0, initial_weight_scale / math.sqrt(fan_in)))
        if abs(value) < 1e-6:
            value = math.copysign(1e-6, value if value != 0.0 else 1.0)
        weights[edge.edge_id] = value

    return RecurrentEventGraph(
        input_nodes=channels,
        hidden_nodes=hidden,
        output_node=output_node,
        edges=tuple(edges),
        initial_weights=weights,
        mode=mode,
        event_log=EventLog(enabled=event_log_enabled),
        **graph_options,
    )


# Two explicit aliases make the builder easy to discover without changing the
# package root while the Experiment-000 runner is being developed in parallel.
build_seeded_recurrent_graph = build_experiment_000_graph
build_recurrent_event_graph = build_experiment_000_graph


__all__ = [
    "CreditPacket",
    "EdgeTrace",
    "EventMessage",
    "QueryResult",
    "RecurrentEdge",
    "RecurrentEventGraph",
    "RecurrentStepResult",
    "UnitEvent",
    "build_experiment_000_graph",
    "build_recurrent_event_graph",
    "build_seeded_recurrent_graph",
]
