"""Local learners that consume delayed outcome events."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .events import UpdateEvent
from .graph import FixedSparseEventGraph, ForwardResult


@dataclass(slots=True)
class CausalCreditFlow:
    """CCF-v0: reinforce the target path and suppress a wrong emitted path."""

    graph: FixedSparseEventGraph
    learning_rate: float = 0.01
    positive_credit: float = 1.0
    negative_credit: float = 1.0

    def learn(
        self,
        result: ForwardResult,
        target_nodes: Mapping[str, str],
    ) -> tuple[UpdateEvent, ...]:
        credits: dict[str, float] = {}
        for group, correct_node in target_nodes.items():
            for node in self.graph.output_groups[group]:
                bipolar_target = self.positive_credit if node == correct_node else -self.negative_credit
                # Root credit is explicitly in the output activation coordinate.
                prediction_error = float(np.clip(bipolar_target - result.activations[node], -1.0, 1.0))
                credits[node] = prediction_error
        return self.graph.apply_credit(credits, learning_rate=self.learning_rate)


@dataclass(slots=True)
class EligibilityTraceBaseline:
    """Reward-modulated emitted-path eligibility baseline on the same graph.

    Unlike CCF-v0 it never credits an omitted alternative.  A correct emitted
    path receives +1; a wrong emitted path receives -1.
    """

    graph: FixedSparseEventGraph
    learning_rate: float = 0.01

    def learn(
        self,
        result: ForwardResult,
        target_nodes: Mapping[str, str],
    ) -> tuple[UpdateEvent, ...]:
        credits: dict[str, float] = {}
        for group, correct_node in target_nodes.items():
            for node in self.graph.output_groups[group]:
                target = 1.0 if node == correct_node else -1.0
                credits[node] = float(np.clip(target - result.activations[node], -1.0, 1.0))
        return self.graph.apply_eligibility_credit(credits, learning_rate=self.learning_rate)
