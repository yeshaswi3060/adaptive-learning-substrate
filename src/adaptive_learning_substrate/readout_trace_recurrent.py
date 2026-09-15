"""Experiment-only readout-decoupled local-trace dynamics.

This module supports the preregistered ``readout-trace-v1`` protocol
(``docs/EXPERIMENT_000_READOUT_TRACE_PROTOCOL.md``). The native recurrent
simulator in ``recurrent.py`` is a frozen external control: this subclass adds
**no** field, branch, option, or counter to it and does **not** change any
activation, message, emission decision, recurrent value, edge trace, or event
that the native graph produces.

Alongside the unchanged native forward pass, every hidden unit keeps one
bounded passive exponential-moving-average trace of its own activations:

    m_i  <-  rho ** (t - kappa_i) * m_i  +  (1 - rho) * a_i(t)
    kappa_i  <-  t

updated on exactly the ticks the native graph already evaluates unit ``i``. The
trace is write-only from the forward pass and read-only at ``QUERY``: it never
enters a pre-activation, an emission test, an edge, or a scheduled message, and
it never fabricates a causal trace. ``rho = 0`` recovers native graph dynamics
bit-for-bit and performs zero trace touches; the decayed-trace readout is not
claimed to equal the native instantaneous-message readout.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any

from .events import EventLog, UnitEvent
from .recurrent import EventMessage, RecurrentEventGraph

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

_TRACE_BOUND_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class ReadoutTraceActivationRecord:
    """One native hidden activation evaluation and the trace it produced.

    ``activation`` is the native value (unchanged). ``retained_trace`` is
    ``rho ** gap * m_i`` before this evaluation's contribution; ``trace_after``
    is ``m_i`` after the bounded-EMA update.
    """

    node: str
    tick: int
    activation: float
    retained_trace: float
    trace_after: float
    emitted: bool


class ReadoutTraceRecurrentEventGraph(RecurrentEventGraph):
    """Native recurrent dynamics plus a passive readout-only per-unit trace."""

    def __init__(self, *, trace_retention: float, **native_kwargs: Any) -> None:
        if (
            isinstance(trace_retention, bool)
            or not math.isfinite(float(trace_retention))
            or not 0.0 <= float(trace_retention) <= 1.0
        ):
            raise ValueError("trace_retention must be a finite number in [0, 1]")
        super().__init__(**native_kwargs)
        self._trace_rho = float(trace_retention)
        self._trace_hidden_nodes = frozenset(self.hidden_nodes)
        self._active = self._trace_rho > 0.0
        self._trace: dict[str, float] = (
            {node: 0.0 for node in self.hidden_nodes} if self._active else {}
        )
        self._trace_tick: dict[str, int | None] = (
            {node: None for node in self.hidden_nodes} if self._active else {}
        )
        self._trace_history: list[ReadoutTraceActivationRecord] = []
        self._trace_ledger = {
            "hidden_activation_evaluations": 0,
            "local_trace_read_touches": 0,
            "local_trace_decay_touches": 0,
            "local_trace_write_touches": 0,
            "local_trace_reset_touches": 0,
            "local_trace_observation_touches": 0,
        }
        self._trace_reset_count = 0
        self._trace_resets_valid = True
        self._trace_bound_ok = True

    @property
    def trace_retention(self) -> float:
        """The construction-time retention coefficient; no setter is exposed."""

        return self._trace_rho

    # ------------------------------------------------------------------
    # Episode handling: administrative reset, counted separately.

    def begin_episode(self, episode_id: str | int | None = None) -> str:
        resolved = super().begin_episode(episode_id)
        self._trace_history = []
        if self._active:
            self._trace = {node: 0.0 for node in self.hidden_nodes}
            self._trace_tick = {node: None for node in self.hidden_nodes}
            self._trace_ledger["local_trace_reset_touches"] += len(self.hidden_nodes)
            self._trace_reset_count += 1
            self._trace_resets_valid = (
                self._trace_resets_valid
                and all(
                    value == 0.0 and math.copysign(1.0, value) == 1.0
                    for value in self._trace.values()
                )
                and all(value is None for value in self._trace_tick.values())
            )
        return resolved

    # ------------------------------------------------------------------
    # Forward pass: native result first, then the passive trace update.

    def _process_unit(
        self,
        node: str,
        messages: tuple[EventMessage, ...],
        *,
        forced_output: bool,
    ) -> tuple[UnitEvent | None, int]:
        result = super()._process_unit(node, messages, forced_output=forced_output)
        if node not in self._trace_hidden_nodes:
            return result
        # The native pass has stored the freshly computed activation.
        activation = self._activations[node]
        emitted = result[0] is not None
        # Every hidden evaluation is recorded for saturation diagnostics; this
        # is a read of the native activation, not a trace touch.
        self._trace_ledger["hidden_activation_evaluations"] += 1
        if not self._active:
            # rho == 0: native fast path, zero trace touches.
            self._trace_history.append(
                ReadoutTraceActivationRecord(
                    node=node,
                    tick=self._tick,
                    activation=float(activation),
                    retained_trace=0.0,
                    trace_after=0.0,
                    emitted=emitted,
                )
            )
            return result
        if not messages:
            raise RuntimeError(
                "readout trace expects a nonempty hidden message batch on evaluation"
            )
        self._trace_ledger["local_trace_read_touches"] += 1
        previous_tick = self._trace_tick[node]
        if previous_tick is None:
            retained = 0.0
        else:
            gap = self._tick - previous_tick
            if gap <= 0:
                raise RuntimeError("readout-trace processing gap must be positive")
            retained = float((self._trace_rho**gap) * self._trace[node])
        self._trace_ledger["local_trace_decay_touches"] += 1
        updated = retained + (1.0 - self._trace_rho) * float(activation)
        if not math.isfinite(updated):
            self._ledger["nonfinite_values"] += 1
            raise FloatingPointError(
                f"non-finite readout trace at node {node!r}, tick {self._tick}"
            )
        if abs(updated) > 1.0 + _TRACE_BOUND_TOLERANCE:
            self._trace_bound_ok = False
            raise FloatingPointError(
                f"readout trace left [-1, 1] at node {node!r}: {updated}"
            )
        self._trace[node] = updated
        self._trace_tick[node] = self._tick
        self._trace_ledger["local_trace_write_touches"] += 1
        self._trace_history.append(
            ReadoutTraceActivationRecord(
                node=node,
                tick=self._tick,
                activation=float(activation),
                retained_trace=retained,
                trace_after=updated,
                emitted=emitted,
            )
        )
        return result

    # ------------------------------------------------------------------
    # Read-only trace access for the QUERY readout and the blank-tail assay.

    def effective_trace(self, observation_tick: int | None = None) -> dict[str, float]:
        """Return ``rho ** (T - kappa_i) * m_i`` for every hidden unit, non-mutating.

        Records 64 ``local_trace_observation_touches`` for a nonzero-rho
        candidate; native and the rho=0 fast path return the conceptual all-zero
        vector and add no touches.
        """

        self._require_episode()
        tick = self._tick if observation_tick is None else int(observation_tick)
        if tick < self._tick:
            raise ValueError("observation_tick cannot precede the current graph tick")
        if not self._active:
            return {node: 0.0 for node in self.hidden_nodes}
        before_trace = tuple(self._trace.items())
        before_ticks = tuple(self._trace_tick.items())
        before_pending = {key: tuple(value) for key, value in self._pending.items()}
        result: dict[str, float] = {}
        for node in self.hidden_nodes:
            previous_tick = self._trace_tick[node]
            if previous_tick is None:
                result[node] = 0.0
            else:
                gap = tick - previous_tick
                if gap < 0:
                    raise RuntimeError("trace timestamp exceeds observation tick")
                result[node] = float((self._trace_rho**gap) * self._trace[node])
        self._trace_ledger["local_trace_observation_touches"] += len(self.hidden_nodes)
        if (
            tuple(self._trace.items()) != before_trace
            or tuple(self._trace_tick.items()) != before_ticks
            or {key: tuple(value) for key, value in self._pending.items()}
            != before_pending
        ):
            raise RuntimeError("effective-trace observation mutated graph dynamics")
        return result

    def query_trace_features(
        self, feature_edge_ids: tuple[str, ...], observation_tick: int
    ) -> tuple[tuple[str, float], ...]:
        """Ordered ``z_e = rho ** (t_query - kappa_j(e)) * m_j(e)`` per output edge.

        ``feature_edge_ids`` must be the frozen output edges in lexical order.
        A source unit that never fired contributes ``0.0``. Reads are counted as
        observation touches and never mutate the trace.
        """

        effective = self.effective_trace(observation_tick)
        source_by_edge = {
            edge.edge_id: edge.source
            for edge in self.edges
            if edge.kind == "output" and edge.destination == self.output_node
        }
        missing = set(feature_edge_ids).difference(source_by_edge)
        if missing:
            raise RuntimeError(f"unknown output feature edges: {sorted(missing)}")
        features: list[tuple[str, float]] = []
        for edge_id in feature_edge_ids:
            source = source_by_edge[edge_id]
            value = effective.get(source, 0.0) if self._active else 0.0
            features.append((edge_id, float(value)))
        return tuple(features)

    # ------------------------------------------------------------------
    # Diagnostics.

    @property
    def trace_ledger(self) -> dict[str, int]:
        return dict(self._trace_ledger)

    @property
    def trace_snapshot(self) -> dict[str, tuple[float, int | None]]:
        if not self._active:
            return {}
        return {
            node: (self._trace[node], self._trace_tick[node])
            for node in self.hidden_nodes
        }

    @property
    def trace_activation_records(self) -> tuple[ReadoutTraceActivationRecord, ...]:
        return tuple(self._trace_history)

    @property
    def trace_reset_count(self) -> int:
        return self._trace_reset_count

    @property
    def trace_resets_valid(self) -> bool:
        return self._trace_resets_valid

    @property
    def trace_bound_ok(self) -> bool:
        return self._trace_bound_ok


def clone_with_readout_trace(
    graph: RecurrentEventGraph,
    trace_retention: float,
    *,
    event_log_enabled: bool | None = None,
) -> ReadoutTraceRecurrentEventGraph:
    """Clone a native graph exactly, adding only the experiment-local trace."""

    if isinstance(trace_retention, bool):
        raise TypeError("trace_retention must be a finite number in [0, 1]")
    value = float(trace_retention)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("trace_retention must be a finite number in [0, 1]")
    log_enabled = (
        graph.event_log.enabled
        if event_log_enabled is None
        else bool(event_log_enabled)
    )
    clone = ReadoutTraceRecurrentEventGraph(
        trace_retention=value,
        input_nodes=graph.input_nodes,
        hidden_nodes=graph.hidden_nodes,
        output_node=graph.output_node,
        edges=graph.edges,
        initial_weights=graph.weights,
        event_log=EventLog(enabled=log_enabled),
        **{field: getattr(graph, field) for field in _CLONED_HYPERPARAMETERS},
    )
    if clone.topology_signature() != graph.topology_signature():
        raise RuntimeError("readout-trace clone changed the explicit edge topology")
    clone_weights = clone.weights
    native_weights = graph.weights
    if set(clone_weights) != set(native_weights) or any(
        struct.pack("<d", clone_weights[edge_id])
        != struct.pack("<d", native_weights[edge_id])
        for edge_id in native_weights
    ):
        raise RuntimeError("readout-trace clone changed a graph weight")
    for field in _CLONED_HYPERPARAMETERS:
        if getattr(clone, field) != getattr(graph, field):
            raise RuntimeError(f"readout-trace clone failed to preserve {field}")
    return clone


__all__ = [
    "ReadoutTraceActivationRecord",
    "ReadoutTraceRecurrentEventGraph",
    "clone_with_readout_trace",
]
