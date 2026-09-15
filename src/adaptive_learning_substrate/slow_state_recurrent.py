"""Experiment-only lazy local-state dynamics for the slow-state preregistration.

The production/native recurrent simulator is a frozen external control.  This
module subclasses it without adding a field, branch, option, or counter to
``recurrent.py``.  A zero-retention instance delegates every unit evaluation to
the native implementation; nonzero instances replace hidden-unit evaluation
only and keep source/output behavior, graph edges, weights, and CCF traces in
the native representation.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any

from .events import EdgeTrace, EventLog, UnitEvent
from .recurrent import EventMessage, RecurrentEventGraph


@dataclass(frozen=True, slots=True)
class SlowStateActivationRecord:
    """One hidden activation evaluation, including below-emission evaluations."""

    node: str
    tick: int
    incoming_drive: float
    retained_state: float
    preactivation: float
    activation: float
    emitted: bool


_CLONED_HYPERPARAMETERS: tuple[str, ...] = (
    "mode",
    "learning_rate",
    "trace_decay",
    "route_gain",
    "emit_threshold",
    "epsilon_weight",
    "epsilon_message",
    "epsilon_route",
    "credit_limit",
    "credit_minimum",
    "trace_horizon",
    "hop_limit",
    "max_update",
    "weight_clip",
)


class SlowStateRecurrentEventGraph(RecurrentEventGraph):
    """A frozen-retention hidden-state intervention around the native graph."""

    def __init__(self, *, local_state_retention: float, **native_kwargs: Any) -> None:
        if (
            isinstance(local_state_retention, bool)
            or not math.isfinite(float(local_state_retention))
            or not 0.0 <= float(local_state_retention) <= 1.0
        ):
            raise ValueError("local_state_retention must be a finite number in [0, 1]")
        super().__init__(**native_kwargs)
        self._slow_state_retention = float(local_state_retention)
        self._slow_hidden_nodes = frozenset(self.hidden_nodes)
        self._slow_states = (
            {node: 0.0 for node in self.hidden_nodes}
            if self._slow_state_retention > 0.0
            else {}
        )
        self._slow_state_ticks: dict[str, int | None] = (
            {node: None for node in self.hidden_nodes}
            if self._slow_state_retention > 0.0
            else {}
        )
        self._slow_activation_history: list[SlowStateActivationRecord] = []
        self._slow_ledger = {
            "hidden_activation_evaluations": 0,
            "local_state_read_touches": 0,
            "local_state_decay_touches": 0,
            "local_state_write_touches": 0,
            "local_state_reset_touches": 0,
            "local_state_observation_touches": 0,
        }
        self._slow_reset_count = 0
        self._slow_resets_valid = True

    @property
    def local_state_retention(self) -> float:
        """The construction-time retention coefficient; no setter is exposed."""

        return self._slow_state_retention

    def begin_episode(self, episode_id: str | int | None = None) -> str:
        resolved = super().begin_episode(episode_id)
        # Administrative reset is deliberately separate from forward touches.
        self._slow_activation_history = []
        if self._slow_state_retention > 0.0:
            self._slow_states = {node: 0.0 for node in self.hidden_nodes}
            self._slow_state_ticks = {node: None for node in self.hidden_nodes}
            self._slow_ledger["local_state_reset_touches"] += len(
                self.hidden_nodes
            )
            self._slow_reset_count += 1
            self._slow_resets_valid = self._slow_resets_valid and all(
                value == 0.0 and math.copysign(1.0, value) == 1.0
                for value in self._slow_states.values()
            ) and all(value is None for value in self._slow_state_ticks.values())
        return resolved

    def _native_fast_path_with_diagnostics(
        self,
        node: str,
        messages: tuple[EventMessage, ...],
        *,
        forced_output: bool,
    ) -> tuple[UnitEvent | None, int]:
        """Delegate λ=0 exactly to native, then append noncausal diagnostics."""

        incoming_drive = math.fsum(
            self._weights[message.edge_id] * message.value for message in messages
        )
        activation = math.tanh(incoming_drive)
        result = super()._process_unit(
            node, messages, forced_output=forced_output
        )
        emitted = result[0] is not None
        self._slow_ledger["hidden_activation_evaluations"] += 1
        self._slow_activation_history.append(
            SlowStateActivationRecord(
                node=node,
                tick=self._tick,
                incoming_drive=incoming_drive,
                retained_state=0.0,
                preactivation=incoming_drive,
                activation=activation,
                emitted=emitted,
            )
        )
        return result

    def _process_unit(
        self,
        node: str,
        messages: tuple[EventMessage, ...],
        *,
        forced_output: bool,
    ) -> tuple[UnitEvent | None, int]:
        # Sources never call this method.  Output behavior always remains the
        # exact inherited implementation.  λ=0 hidden behavior also delegates
        # exactly, with zero forward state touches and no state write.
        if node not in self._slow_hidden_nodes:
            return super()._process_unit(
                node, messages, forced_output=forced_output
            )
        if self._slow_state_retention == 0.0:
            return self._native_fast_path_with_diagnostics(
                node, messages, forced_output=forced_output
            )
        if not messages:
            raise RuntimeError("slow local state requires a nonempty hidden message batch")

        contributions: list[float] = []
        for message in messages:
            weight = self._weights[message.edge_id]
            contributions.append(weight * message.value)
        self._ledger["arrived_messages"] += len(messages)
        self._ledger["forward_edge_touches"] += len(messages)
        incoming_drive = math.fsum(contributions)

        self._slow_ledger["local_state_read_touches"] += 1
        previous_tick = self._slow_state_ticks[node]
        if previous_tick is None:
            retained_state = 0.0
        else:
            gap = self._tick - previous_tick
            if gap <= 0:
                raise RuntimeError("local-state processing gap must be positive")
            retained_state = float(
                (self._slow_state_retention**gap) * self._slow_states[node]
            )
        self._slow_ledger["local_state_decay_touches"] += 1
        total = incoming_drive + retained_state
        activation = math.tanh(total)
        self._slow_ledger["hidden_activation_evaluations"] += 1

        self._ledger["activation_evaluations"] += 1
        if not all(
            math.isfinite(value)
            for value in (incoming_drive, retained_state, total, activation)
        ):
            self._ledger["nonfinite_values"] += 1
            raise FloatingPointError(
                f"non-finite slow local state at node {node!r}, tick {self._tick}"
            )
        self._slow_states[node] = activation
        self._slow_state_ticks[node] = self._tick
        self._slow_ledger["local_state_write_touches"] += 1
        self._activations[node] = activation
        should_emit = (
            forced_output
            or abs(activation - self._last_emitted[node]) >= self.emit_threshold
        )
        self._slow_activation_history.append(
            SlowStateActivationRecord(
                node=node,
                tick=self._tick,
                incoming_drive=incoming_drive,
                retained_state=retained_state,
                preactivation=total,
                activation=activation,
                emitted=should_emit,
            )
        )
        if not should_emit:
            return None, 0

        traces: list[EdgeTrace] = []
        for message in messages:
            edge = self._edge_by_id[message.edge_id]
            weight = self._weights[edge.edge_id]
            # The retained term is already in ``total`` and is held fixed when
            # exactly one incident message is omitted.  No local-state trace is
            # fabricated.
            omission_activation = math.tanh(total - weight * message.value)
            omission_effect = activation - omission_activation
            weight_secant = (
                omission_effect / weight
                if abs(weight) >= self.epsilon_weight
                else 0.0
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
        return event, self._schedule_from(node, activation, event.event_id)

    @property
    def slow_state_ledger(self) -> dict[str, int]:
        return dict(self._slow_ledger)

    @property
    def local_state_snapshot(self) -> dict[str, tuple[float, int | None]]:
        if self._slow_state_retention == 0.0:
            return {}
        return {
            node: (self._slow_states[node], self._slow_state_ticks[node])
            for node in self.hidden_nodes
        }

    @property
    def slow_activation_records(self) -> tuple[SlowStateActivationRecord, ...]:
        return tuple(self._slow_activation_history)

    @property
    def local_state_reset_count(self) -> int:
        return self._slow_reset_count

    @property
    def local_state_resets_valid(self) -> bool:
        return self._slow_resets_valid

    def effective_local_states(
        self, observation_tick: int | None = None
    ) -> dict[str, float]:
        """Return a non-mutating lazy snapshot and count its 64 diagnostic reads."""

        self._require_episode()
        tick = self._tick if observation_tick is None else int(observation_tick)
        if tick < self._tick:
            raise ValueError("observation_tick cannot precede the current graph tick")
        if self._slow_state_retention == 0.0:
            return {node: 0.0 for node in self.hidden_nodes}
        before_states = tuple(self._slow_states.items())
        before_ticks = tuple(self._slow_state_ticks.items())
        before_pending = {
            key: tuple(value) for key, value in self._pending.items()
        }
        result: dict[str, float] = {}
        for node in self.hidden_nodes:
            previous_tick = self._slow_state_ticks[node]
            if previous_tick is None or self._slow_state_retention == 0.0:
                result[node] = 0.0
            else:
                gap = tick - previous_tick
                if gap < 0:
                    raise RuntimeError("local-state timestamp exceeds observation tick")
                result[node] = float(
                    (self._slow_state_retention**gap) * self._slow_states[node]
                )
        self._slow_ledger["local_state_observation_touches"] += len(self.hidden_nodes)
        if (
            tuple(self._slow_states.items()) != before_states
            or tuple(self._slow_state_ticks.items()) != before_ticks
            or {key: tuple(value) for key, value in self._pending.items()}
            != before_pending
        ):
            raise RuntimeError("effective-state observation mutated graph dynamics")
        return result


def clone_with_local_state_retention(
    graph: RecurrentEventGraph,
    retention: float,
    *,
    event_log_enabled: bool | None = None,
) -> SlowStateRecurrentEventGraph:
    """Clone a native graph exactly, adding only experiment-local hidden state."""

    if isinstance(retention, bool):
        raise TypeError("retention must be a finite number in [0, 1]")
    value = float(retention)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("retention must be a finite number in [0, 1]")
    log_enabled = (
        graph.event_log.enabled
        if event_log_enabled is None
        else bool(event_log_enabled)
    )
    clone = SlowStateRecurrentEventGraph(
        local_state_retention=value,
        input_nodes=graph.input_nodes,
        hidden_nodes=graph.hidden_nodes,
        output_node=graph.output_node,
        edges=graph.edges,
        initial_weights=graph.weights,
        event_log=EventLog(enabled=log_enabled),
        **{field: getattr(graph, field) for field in _CLONED_HYPERPARAMETERS},
    )
    if clone.topology_signature() != graph.topology_signature():
        raise RuntimeError("slow-state clone changed the explicit edge topology")
    clone_weights = clone.weights
    native_weights = graph.weights
    if set(clone_weights) != set(native_weights) or any(
        struct.pack("<d", clone_weights[edge_id])
        != struct.pack("<d", native_weights[edge_id])
        for edge_id in native_weights
    ):
        raise RuntimeError("slow-state clone changed a graph weight")
    for field in _CLONED_HYPERPARAMETERS:
        if getattr(clone, field) != getattr(graph, field):
            raise RuntimeError(f"slow-state clone failed to preserve {field}")
    return clone


__all__ = [
    "SlowStateActivationRecord",
    "SlowStateRecurrentEventGraph",
    "clone_with_local_state_retention",
]
