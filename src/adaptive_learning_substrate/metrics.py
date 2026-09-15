"""Small metric helpers with no hidden aggregation state."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    phase: str
    task: str
    predicted: tuple[int, int, int]
    target: tuple[int, int, int]
    delay: int
    update_count: int = 0

    @property
    def exact(self) -> bool:
        return self.predicted == self.target

    @property
    def correct_bits(self) -> int:
        return sum(a == b for a, b in zip(self.predicted, self.target, strict=True))


class MetricsTracker:
    def __init__(self) -> None:
        self._records: list[PredictionRecord] = []

    def add(self, record: PredictionRecord) -> None:
        self._records.append(record)

    @property
    def records(self) -> tuple[PredictionRecord, ...]:
        return tuple(self._records)

    def summary(self, *, phase: str | None = None, task: str | None = None) -> dict[str, float | int]:
        selected = [
            record
            for record in self._records
            if (phase is None or record.phase == phase) and (task is None or record.task == task)
        ]
        if not selected:
            return {"episodes": 0, "exact_accuracy": 0.0, "bit_accuracy": 0.0, "mean_updates": 0.0}
        return {
            "episodes": len(selected),
            "exact_accuracy": sum(record.exact for record in selected) / len(selected),
            "bit_accuracy": sum(record.correct_bits for record in selected) / (3 * len(selected)),
            "mean_updates": sum(record.update_count for record in selected) / len(selected),
        }


def exact_accuracy(records: Iterable[PredictionRecord]) -> float:
    values = list(records)
    return sum(record.exact for record in values) / len(values) if values else 0.0


def forgetting(pre_accuracy: float, post_accuracy: float) -> float:
    return max(0.0, float(pre_accuracy) - float(post_accuracy))

