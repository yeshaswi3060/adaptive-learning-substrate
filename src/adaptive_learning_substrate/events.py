"""Structured, JSON-serialisable audit events for the simulator."""

from __future__ import annotations

import json
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class EdgeTrace:
    """Immutable participation record for one nonzero arrived message.

    ``omission_effect`` is the local leave-one-message-out quantity
    ``a - tanh(u - w*x)``.  The two secants are local finite-difference
    sensitivities guarded against division by a near-zero weight or message.
    """

    edge_id: str
    parent_event_id: str
    message_value: float
    omission_effect: float
    weight_secant: float
    source_secant: float
    created_step: int


@dataclass(frozen=True, slots=True)
class UnitEvent:
    event_id: str
    episode_id: str
    node: str
    step: int
    preactivation: float
    activation: float
    forced_output: bool
    edge_traces: tuple[EdgeTrace, ...]


@dataclass(frozen=True, slots=True)
class CreditPacket:
    target_event_id: str
    signal: float
    created_step: int
    hops: int = 0


@dataclass(frozen=True, slots=True)
class ForwardEvent:
    episode_id: str
    step: int
    edge_id: str
    source: str
    destination: str
    source_activity: float
    destination_activity: float
    contribution: float
    trace_kind: Literal["participation", "omission"]
    participation_trace: float
    omission_trace: float


@dataclass(frozen=True, slots=True)
class CreditEvent:
    episode_id: str
    step: int
    node: str
    signal: float
    source: Literal["outcome", "routed"]


@dataclass(frozen=True, slots=True)
class UpdateEvent:
    episode_id: str
    step: int
    edge_id: str
    destination_credit: float
    trace_kind: Literal["participation", "omission"]
    trace_value: float
    delta: float
    old_weight: float
    new_weight: float


AuditEvent = ForwardEvent | CreditEvent | UpdateEvent | UnitEvent | CreditPacket


class EventLog:
    """Append-only in-memory log with deterministic JSONL output."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._records: list[dict[str, Any]] = []

    def append(self, event: AuditEvent) -> None:
        if not self.enabled:
            return
        record = asdict(event)
        record["event_type"] = type(event).__name__
        self._records.append(record)

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        # Never expose the dictionaries that back the append-only audit log.
        # ``asdict`` can contain nested mutable containers, so a shallow copy is
        # insufficient: callers must be free to transform a returned snapshot
        # without rewriting history held by this instance.
        return tuple(deepcopy(record) for record in self._records)

    def by_type(self, event_type: str) -> tuple[dict[str, Any], ...]:
        return tuple(
            deepcopy(record)
            for record in self._records
            if record["event_type"] == event_type
        )

    def extend(self, events: Iterable[AuditEvent]) -> None:
        for event in events:
            self.append(event)

    def write_jsonl(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            for record in self._records:
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
        return target
