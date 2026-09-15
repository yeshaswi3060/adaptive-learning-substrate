"""Small deterministic environments for falsifiable Stage-1 experiments."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np

Operation = Literal["FLIP0", "ROTL", "NOT"]
Bits3 = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class ChainTransition:
    observation: dict[str, float]
    terminated: bool
    reward: float | None
    credit_ready: bool
    delay_remaining: int


class DelayedChainEnvironment:
    """A chain whose terminal success/failure arrives after silent delay ticks."""

    def __init__(
        self,
        length: int = 5,
        credit_delay: int = 8,
        correct_actions: Sequence[int] | None = None,
        seed: int = 0,
    ) -> None:
        if length < 1 or credit_delay < 0:
            raise ValueError("length must be positive and credit_delay non-negative")
        actions = tuple(correct_actions) if correct_actions is not None else tuple(i % 2 for i in range(length))
        if len(actions) != length or any(action not in (0, 1) for action in actions):
            raise ValueError("correct_actions must contain one binary action per state")
        self.length = length
        self.credit_delay = credit_delay
        self.correct_actions = actions
        self.rng = np.random.default_rng(seed)
        self.state = 0
        self.terminated = False
        self._pending_reward: float | None = None
        self._delay_remaining = 0

    @property
    def input_nodes(self) -> tuple[str, ...]:
        return tuple(f"state:{i}" for i in range(self.length))

    def observation(self) -> dict[str, float]:
        if self.terminated:
            return {}
        return {f"state:{self.state}": 1.0}

    def reset(self) -> dict[str, float]:
        self.state = 0
        self.terminated = False
        self._pending_reward = None
        self._delay_remaining = 0
        return self.observation()

    def step(self, action: int) -> ChainTransition:
        if self.terminated:
            raise RuntimeError("episode terminated; call tick() for delayed credit or reset()")
        if action not in (0, 1):
            raise ValueError("action must be 0 or 1")
        success = action == self.correct_actions[self.state]
        if not success:
            self.terminated = True
            self._pending_reward = -1.0
        elif self.state == self.length - 1:
            self.terminated = True
            self._pending_reward = 1.0
        else:
            self.state += 1
        if self.terminated:
            self._delay_remaining = self.credit_delay
            ready = self.credit_delay == 0
            reward = self._consume_if_ready() if ready else None
            return ChainTransition({}, True, reward, ready, self._delay_remaining)
        return ChainTransition(self.observation(), False, None, False, 0)

    def _consume_if_ready(self) -> float | None:
        reward = self._pending_reward
        self._pending_reward = None
        return reward

    def tick(self) -> ChainTransition:
        if not self.terminated or self._pending_reward is None:
            raise RuntimeError("no delayed outcome is pending")
        if self._delay_remaining > 0:
            self._delay_remaining -= 1
        ready = self._delay_remaining == 0
        reward = self._consume_if_ready() if ready else None
        # A deterministic distractor id makes delayed streams reproducible.
        distractor = int(self.rng.integers(0, 2**31 - 1))
        return ChainTransition(
            {f"distractor:{distractor % 64}": 1.0},
            True,
            reward,
            ready,
            self._delay_remaining,
        )


@dataclass(frozen=True, slots=True)
class MicroProgramEpisode:
    operation: Operation
    inputs: Bits3
    target: Bits3
    delay: int
    distractors: tuple[int, ...]


class MicroProgram3BitEnvironment:
    """Three-bit primitive operations with delayed target events."""

    OPERATIONS: tuple[Operation, ...] = ("FLIP0", "ROTL", "NOT")

    def __init__(self, seed: int = 0, distractor_alphabet: int = 64) -> None:
        if distractor_alphabet < 1:
            raise ValueError("distractor_alphabet must be positive")
        self.rng = np.random.default_rng(seed)
        self.distractor_alphabet = distractor_alphabet

    @staticmethod
    def validate_bits(bits: Sequence[int]) -> Bits3:
        value = tuple(int(bit) for bit in bits)
        if len(value) != 3 or any(bit not in (0, 1) for bit in value):
            raise ValueError("bits must be exactly three binary values ordered (x2, x1, x0)")
        return value  # type: ignore[return-value]

    @classmethod
    def apply(cls, bits: Sequence[int], operation: Operation) -> Bits3:
        x2, x1, x0 = cls.validate_bits(bits)
        if operation == "FLIP0":
            return x2, x1, 1 - x0
        if operation == "ROTL":
            return x1, x0, x2
        if operation == "NOT":
            return 1 - x2, 1 - x1, 1 - x0
        raise ValueError(f"unknown operation: {operation}")

    @classmethod
    def compose(cls, bits: Sequence[int], operations: Iterable[Operation]) -> Bits3:
        result = cls.validate_bits(bits)
        for operation in operations:
            result = cls.apply(result, operation)
        return result

    @classmethod
    def all_inputs(cls) -> tuple[Bits3, ...]:
        return tuple(product((0, 1), repeat=3))  # type: ignore[return-value]

    @classmethod
    def input_nodes(cls) -> tuple[str, ...]:
        return tuple(
            f"feature:{operation}:x{position}={value}"
            for operation in cls.OPERATIONS
            for position in (2, 1, 0)
            for value in (0, 1)
        )

    @classmethod
    def encode(cls, bits: Sequence[int], operation: Operation) -> dict[str, float]:
        values = cls.validate_bits(bits)
        return {
            f"feature:{operation}:x{position}={value}": 1.0
            for position, value in zip((2, 1, 0), values, strict=True)
        }

    @staticmethod
    def output_groups() -> dict[str, tuple[str, str]]:
        return {f"bit{x}": (f"out:x{x}=0", f"out:x{x}=1") for x in (2, 1, 0)}

    @staticmethod
    def target_nodes(bits: Sequence[int]) -> dict[str, str]:
        x2, x1, x0 = MicroProgram3BitEnvironment.validate_bits(bits)
        return {
            "bit2": f"out:x2={x2}",
            "bit1": f"out:x1={x1}",
            "bit0": f"out:x0={x0}",
        }

    @staticmethod
    def decode(selections: dict[str, str]) -> Bits3:
        return tuple(int(selections[f"bit{x}"].rsplit("=", 1)[1]) for x in (2, 1, 0))  # type: ignore[return-value]

    def sample(
        self,
        *,
        operations: Sequence[Operation],
        delay_min: int,
        delay_max: int,
    ) -> MicroProgramEpisode:
        if not operations:
            raise ValueError("operations cannot be empty")
        if delay_min < 0 or delay_max < delay_min:
            raise ValueError("invalid delay interval")
        operation = operations[int(self.rng.integers(0, len(operations)))]
        inputs: Bits3 = tuple(int(x) for x in self.rng.integers(0, 2, size=3))  # type: ignore[assignment]
        delay = int(self.rng.integers(delay_min, delay_max + 1))
        distractors = tuple(int(x) for x in self.rng.integers(0, self.distractor_alphabet, size=delay))
        return MicroProgramEpisode(operation, inputs, self.apply(inputs, operation), delay, distractors)

    def balanced_cases(self, operations: Sequence[Operation]) -> Iterator[tuple[Operation, Bits3]]:
        for operation in operations:
            for bits in self.all_inputs():
                yield operation, bits

