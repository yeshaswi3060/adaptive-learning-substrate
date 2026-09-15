"""Deterministic, development-only data schedules for Experiment 001.

This module implements only the preregistered episode *descriptions*.  It does
not construct a learner, reveal targets before terminal feedback, or authorize
the reserved confirmatory seeds.  Keeping schedule construction independent of
method code makes the paired-stream integrity checks in ``docs/EXPERIMENT_001.md``
directly testable before the recurrent runner and baselines exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .environments import Bits3, MicroProgram3BitEnvironment, Operation

PROTOCOL_VERSION = "experiment-001-schedule-v1"
DEVELOPMENT_SEEDS = (0, 1, 2, 3, 4)
RESERVED_CONFIRMATORY_SEEDS = tuple(range(1000, 1020))
PHASE_A_OPERATIONS: tuple[Operation, ...] = ("FLIP0", "ROTL")
PHASE_B_OPERATIONS: tuple[Operation, ...] = ("NOT",)
PRIMITIVE_PROGRAMS: tuple[tuple[Operation, ...], ...] = tuple(
    (operation,) for operation in MicroProgram3BitEnvironment.OPERATIONS
)
COMPOSITION_PROGRAMS: tuple[tuple[Operation, ...], ...] = tuple(
    (first, second)
    for first in MicroProgram3BitEnvironment.OPERATIONS
    for second in MicroProgram3BitEnvironment.OPERATIONS
)
PAIRED_METHOD_ROLES = ("ccf_v0", "et_3f", "bptt", "bptt_cm")
PHASE_B_CHECKPOINTS = (0, 50, 100, 200, 400, 800, 1200, 1600, 2000)

SplitName = Literal[
    "phase_a_train",
    "phase_b_train",
    "primary_primitives",
    "primary_compositions",
    "secondary_primitives",
    "secondary_compositions",
]
VisibleEventKind = Literal["START", "OPERATION", "NOISE", "QUERY"]


@dataclass(frozen=True, slots=True)
class Experiment001VisibleEvent:
    """One learner-visible event; terminal targets are intentionally impossible."""

    kind: VisibleEventKind
    bits: Bits3 | None = None
    operation: Operation | None = None

    def __post_init__(self) -> None:
        if self.kind in {"START", "NOISE"}:
            if self.bits is None or self.operation is not None:
                raise ValueError(f"{self.kind} requires bits and forbids an operation")
            MicroProgram3BitEnvironment.validate_bits(self.bits)
        elif self.kind == "OPERATION":
            if self.bits is not None or self.operation not in MicroProgram3BitEnvironment.OPERATIONS:
                raise ValueError("OPERATION requires one registered operation and forbids bits")
        elif self.kind == "QUERY":
            if self.bits is not None or self.operation is not None:
                raise ValueError("QUERY carries no payload")
        else:  # pragma: no cover - protected by public typing plus runtime checks
            raise ValueError(f"unknown visible event kind: {self.kind!r}")

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"kind": self.kind}
        if self.bits is not None:
            value["bits"] = list(self.bits)
        if self.operation is not None:
            value["operation"] = self.operation
        return value


@dataclass(frozen=True, slots=True)
class ScheduleEpisode:
    """One immutable, method-agnostic episode description."""

    episode_id: str
    split: SplitName
    index: int
    stream_id: str
    stream_namespace: str
    inputs: Bits3
    operations: tuple[Operation, ...]
    delay: int
    noise: tuple[Bits3, ...]
    target: Bits3

    def visible_events(self) -> tuple[Experiment001VisibleEvent, ...]:
        """Return START -> ordered OPERATION(s) -> NOISE* -> QUERY, without target."""

        return (
            Experiment001VisibleEvent("START", bits=self.inputs),
            *(Experiment001VisibleEvent("OPERATION", operation=value) for value in self.operations),
            *(Experiment001VisibleEvent("NOISE", bits=value) for value in self.noise),
            Experiment001VisibleEvent("QUERY"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "delay": self.delay,
            "episode_id": self.episode_id,
            "index": self.index,
            "inputs": list(self.inputs),
            "noise": [list(bits) for bits in self.noise],
            "operations": list(self.operations),
            "split": self.split,
            "stream_id": self.stream_id,
            "stream_namespace": self.stream_namespace,
            "target": list(self.target),
            "visible_events": [event.as_dict() for event in self.visible_events()],
        }


@dataclass(frozen=True, slots=True)
class NamedSubstream:
    """Auditable derivation record for one independent RNG namespace."""

    name: str
    derivation_sha256: str
    bit_generator: str = "PCG64"

    def as_dict(self) -> dict[str, str]:
        return {
            "bit_generator": self.bit_generator,
            "derivation_sha256": self.derivation_sha256,
            "name": self.name,
        }


@dataclass(frozen=True, slots=True)
class Experiment001ScheduleBundle:
    """All frozen-size training and evaluation schedules for one dev seed."""

    master_seed: int
    phase_a_train: tuple[ScheduleEpisode, ...]
    phase_b_train: tuple[ScheduleEpisode, ...]
    primary_primitives: tuple[ScheduleEpisode, ...]
    primary_compositions: tuple[ScheduleEpisode, ...]
    secondary_primitives: tuple[ScheduleEpisode, ...]
    secondary_compositions: tuple[ScheduleEpisode, ...]
    named_substreams: tuple[NamedSubstream, ...]

    def splits(self) -> tuple[tuple[str, tuple[ScheduleEpisode, ...]], ...]:
        return (
            ("phase_a_train", self.phase_a_train),
            ("phase_b_train", self.phase_b_train),
            ("primary_primitives", self.primary_primitives),
            ("primary_compositions", self.primary_compositions),
            ("secondary_primitives", self.secondary_primitives),
            ("secondary_compositions", self.secondary_compositions),
        )

    def episodes(self) -> Iterator[ScheduleEpisode]:
        for _, episodes in self.splits():
            yield from episodes


def _validate_master_seed(master_seed: int) -> int:
    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise TypeError("master_seed must be an integer")
    if not 0 <= master_seed < 2**63:
        raise ValueError("master_seed must be in [0, 2**63)")
    return master_seed


def _derivation_digest(master_seed: int, name: str) -> bytes:
    seed = _validate_master_seed(master_seed)
    if not name or "\x00" in name:
        raise ValueError("substream name must be non-empty and contain no NUL")
    material = f"{PROTOCOL_VERSION}\x00{seed}\x00{name}".encode()
    return hashlib.sha256(material).digest()


def named_substream(master_seed: int, name: str) -> NamedSubstream:
    return NamedSubstream(name=name, derivation_sha256=_derivation_digest(master_seed, name).hex())


def _named_rng(master_seed: int, name: str) -> np.random.Generator:
    digest = _derivation_digest(master_seed, name)
    entropy = np.frombuffer(digest, dtype="<u4").astype(np.uint32).tolist()
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _stable_id(master_seed: int, namespace: str, index: int) -> str:
    payload = f"{PROTOCOL_VERSION}\x00{master_seed}\x00{namespace}\x00{index}".encode()
    return hashlib.sha256(payload).hexdigest()


def _noise_bits(rng: np.random.Generator, delay: int) -> tuple[Bits3, ...]:
    values = rng.integers(0, 2, size=(delay, 3), dtype=np.int8)
    return tuple(tuple(int(bit) for bit in row) for row in values)  # type: ignore[return-value]


def _make_episode(
    *,
    master_seed: int,
    split: SplitName,
    index: int,
    inputs: Bits3,
    operations: tuple[Operation, ...],
    delay: int,
    noise_rng: np.random.Generator,
) -> ScheduleEpisode:
    namespace = f"{split}/noise"
    program = "-then-".join(operations)
    return ScheduleEpisode(
        episode_id=f"{split}:{index:06d}:{program}",
        split=split,
        index=index,
        stream_id=_stable_id(master_seed, namespace, index),
        stream_namespace=namespace,
        inputs=MicroProgram3BitEnvironment.validate_bits(inputs),
        operations=operations,
        delay=delay,
        noise=_noise_bits(noise_rng, delay),
        target=MicroProgram3BitEnvironment.compose(inputs, operations),
    )


def build_training_schedules(
    master_seed: int,
    *,
    phase_a_repetitions_per_word: int = 250,
    phase_b_repetitions_per_word: int = 250,
    delay_min: int = 4,
    delay_max: int = 8,
) -> tuple[tuple[ScheduleEpisode, ...], tuple[ScheduleEpisode, ...]]:
    """Build exactly balanced, independently shuffled Phase-A and Phase-B data."""

    _validate_master_seed(master_seed)
    if phase_a_repetitions_per_word < 1 or phase_b_repetitions_per_word < 1:
        raise ValueError("repetitions per word must be positive")
    if delay_min < 0 or delay_max < delay_min:
        raise ValueError("invalid delay interval")

    def build_phase(
        *,
        split: Literal["phase_a_train", "phase_b_train"],
        operations: tuple[Operation, ...],
        repetitions: int,
    ) -> tuple[ScheduleEpisode, ...]:
        cases = [
            (operation, bits)
            for operation in operations
            for bits in MicroProgram3BitEnvironment.all_inputs()
            for _ in range(repetitions)
        ]
        order_rng = _named_rng(master_seed, f"{split}/order")
        delay_rng = _named_rng(master_seed, f"{split}/delay")
        noise_rng = _named_rng(master_seed, f"{split}/noise")
        order = order_rng.permutation(len(cases))
        episodes: list[ScheduleEpisode] = []
        for index, case_index in enumerate(order):
            operation, bits = cases[int(case_index)]
            delay = int(delay_rng.integers(delay_min, delay_max + 1))
            episodes.append(
                _make_episode(
                    master_seed=master_seed,
                    split=split,
                    index=index,
                    inputs=bits,
                    operations=(operation,),
                    delay=delay,
                    noise_rng=noise_rng,
                )
            )
        return tuple(episodes)

    return (
        build_phase(
            split="phase_a_train",
            operations=PHASE_A_OPERATIONS,
            repetitions=phase_a_repetitions_per_word,
        ),
        build_phase(
            split="phase_b_train",
            operations=PHASE_B_OPERATIONS,
            repetitions=phase_b_repetitions_per_word,
        ),
    )


def build_evaluation_schedule(
    master_seed: int,
    *,
    split: Literal[
        "primary_primitives",
        "primary_compositions",
        "secondary_primitives",
        "secondary_compositions",
    ],
    programs: Sequence[tuple[Operation, ...]],
    delay: int,
    streams_per_word: int = 64,
) -> tuple[ScheduleEpisode, ...]:
    """Build fixed-delay evaluation cases without learner or method state."""

    _validate_master_seed(master_seed)
    if delay < 0 or streams_per_word < 1:
        raise ValueError("delay must be non-negative and streams_per_word positive")
    allowed = set(MicroProgram3BitEnvironment.OPERATIONS)
    normalized: list[tuple[Operation, ...]] = []
    for program in programs:
        value = tuple(program)
        if not value or len(value) > 2 or any(operation not in allowed for operation in value):
            raise ValueError(f"invalid evaluation program: {program!r}")
        normalized.append(value)  # type: ignore[arg-type]
    if len(set(normalized)) != len(normalized):
        raise ValueError("evaluation programs must be unique")

    noise_rng = _named_rng(master_seed, f"{split}/noise")
    episodes: list[ScheduleEpisode] = []
    index = 0
    for program in normalized:
        for bits in MicroProgram3BitEnvironment.all_inputs():
            for _ in range(streams_per_word):
                episodes.append(
                    _make_episode(
                        master_seed=master_seed,
                        split=split,
                        index=index,
                        inputs=bits,
                        operations=program,
                        delay=delay,
                        noise_rng=noise_rng,
                    )
                )
                index += 1
    return tuple(episodes)


def build_development_bundle(master_seed: int) -> Experiment001ScheduleBundle:
    """Materialize the frozen protocol sizes for one registered dev seed only."""

    if master_seed in RESERVED_CONFIRMATORY_SEEDS:
        raise RuntimeError("reserved confirmatory schedules remain blocked")
    if master_seed not in DEVELOPMENT_SEEDS:
        raise ValueError(f"development seed must be one of {DEVELOPMENT_SEEDS}")

    phase_a, phase_b = build_training_schedules(master_seed)
    primary_primitives = build_evaluation_schedule(
        master_seed,
        split="primary_primitives",
        programs=PRIMITIVE_PROGRAMS,
        delay=8,
    )
    primary_compositions = build_evaluation_schedule(
        master_seed,
        split="primary_compositions",
        programs=COMPOSITION_PROGRAMS,
        delay=8,
    )
    secondary_primitives = build_evaluation_schedule(
        master_seed,
        split="secondary_primitives",
        programs=PRIMITIVE_PROGRAMS,
        delay=16,
    )
    secondary_compositions = build_evaluation_schedule(
        master_seed,
        split="secondary_compositions",
        programs=COMPOSITION_PROGRAMS,
        delay=16,
    )
    substream_names = (
        "graph_mask",
        "weight_initialization",
        "phase_a_train/order",
        "phase_a_train/delay",
        "phase_a_train/noise",
        "phase_b_train/order",
        "phase_b_train/delay",
        "phase_b_train/noise",
        "primary_primitives/noise",
        "primary_compositions/noise",
        "secondary_primitives/noise",
        "secondary_compositions/noise",
        "ablation_shuffle",
        "rand_actions",
    )
    return Experiment001ScheduleBundle(
        master_seed=master_seed,
        phase_a_train=phase_a,
        phase_b_train=phase_b,
        primary_primitives=primary_primitives,
        primary_compositions=primary_compositions,
        secondary_primitives=secondary_primitives,
        secondary_compositions=secondary_compositions,
        named_substreams=tuple(named_substream(master_seed, name) for name in substream_names),
    )


def canonical_episode_bytes(episode: ScheduleEpisode) -> bytes:
    return json.dumps(
        episode.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def schedule_sha256(episodes: Iterable[ScheduleEpisode]) -> str:
    digest = hashlib.sha256()
    for episode in episodes:
        digest.update(canonical_episode_bytes(episode))
        digest.update(b"\n")
    return digest.hexdigest()


def paired_method_schedule_hashes(
    episodes: Sequence[ScheduleEpisode],
    methods: Sequence[str] = PAIRED_METHOD_ROLES,
) -> dict[str, str]:
    """Return the same method-agnostic stream hash for every paired method."""

    if not methods or len(set(methods)) != len(methods) or any(not method for method in methods):
        raise ValueError("methods must be unique non-empty names")
    digest = schedule_sha256(episodes)
    return {method: digest for method in methods}


def _check_episode(
    episode: ScheduleEpisode,
    *,
    expected_split: str,
    expected_delay: int | None,
    errors: list[str],
) -> None:
    if episode.split != expected_split:
        errors.append(f"{episode.episode_id}: split mismatch")
    if episode.delay != len(episode.noise):
        errors.append(f"{episode.episode_id}: noise length differs from delay")
    if expected_delay is not None and episode.delay != expected_delay:
        errors.append(f"{episode.episode_id}: fixed evaluation delay mismatch")
    if any(bit not in (0, 1) for noise in episode.noise for bit in noise):
        errors.append(f"{episode.episode_id}: non-binary noise")
    if episode.target != MicroProgram3BitEnvironment.compose(episode.inputs, episode.operations):
        errors.append(f"{episode.episode_id}: target does not match left-to-right composition")
    events = episode.visible_events()
    if events[0].kind != "START" or events[-1].kind != "QUERY":
        errors.append(f"{episode.episode_id}: visible event boundaries are invalid")
    visible_operations = tuple(
        event.operation for event in events if event.kind == "OPERATION"
    )
    if visible_operations != episode.operations:
        errors.append(f"{episode.episode_id}: visible operation order differs")
    if any("target" in event.as_dict() for event in events):
        errors.append(f"{episode.episode_id}: terminal target leaked into visible events")


def validate_development_bundle(bundle: Experiment001ScheduleBundle) -> dict[str, Any]:
    """Validate the protocol's balance, disjointness, and paired-stream invariants."""

    errors: list[str] = []
    if bundle.master_seed not in DEVELOPMENT_SEEDS:
        errors.append("master seed is not a registered development seed")

    expected_counts = {
        "phase_a_train": 4000,
        "phase_b_train": 2000,
        "primary_primitives": 3 * 8 * 64,
        "primary_compositions": 9 * 8 * 64,
        "secondary_primitives": 3 * 8 * 64,
        "secondary_compositions": 9 * 8 * 64,
    }
    actual_counts = {name: len(episodes) for name, episodes in bundle.splits()}
    if actual_counts != expected_counts:
        errors.append(f"split counts differ: {actual_counts!r}")

    phase_a_balance = Counter((episode.operations, episode.inputs) for episode in bundle.phase_a_train)
    expected_phase_a = {
        ((operation,), bits): 250
        for operation in PHASE_A_OPERATIONS
        for bits in MicroProgram3BitEnvironment.all_inputs()
    }
    if dict(phase_a_balance) != expected_phase_a:
        errors.append("Phase A is not exactly 250 occurrences per operation/word")

    phase_b_balance = Counter((episode.operations, episode.inputs) for episode in bundle.phase_b_train)
    expected_phase_b = {
        ((operation,), bits): 250
        for operation in PHASE_B_OPERATIONS
        for bits in MicroProgram3BitEnvironment.all_inputs()
    }
    if dict(phase_b_balance) != expected_phase_b:
        errors.append("Phase B is not exactly 250 occurrences per operation/word")

    if any(not 4 <= episode.delay <= 8 for episode in bundle.phase_a_train + bundle.phase_b_train):
        errors.append("a training delay is outside [4, 8]")

    for name, episodes in bundle.splits():
        expected_delay = 8 if name.startswith("primary_") else 16 if name.startswith("secondary_") else None
        for episode in episodes:
            _check_episode(
                episode,
                expected_split=name,
                expected_delay=expected_delay,
                errors=errors,
            )

    for name, episodes, programs in (
        ("primary_primitives", bundle.primary_primitives, PRIMITIVE_PROGRAMS),
        ("primary_compositions", bundle.primary_compositions, COMPOSITION_PROGRAMS),
        ("secondary_primitives", bundle.secondary_primitives, PRIMITIVE_PROGRAMS),
        ("secondary_compositions", bundle.secondary_compositions, COMPOSITION_PROGRAMS),
    ):
        balance = Counter((episode.operations, episode.inputs) for episode in episodes)
        expected = {
            (program, bits): 64
            for program in programs
            for bits in MicroProgram3BitEnvironment.all_inputs()
        }
        if dict(balance) != expected:
            errors.append(f"{name} is not exactly 64 streams per program/word")

    episode_ids = [episode.episode_id for episode in bundle.episodes()]
    stream_ids = [episode.stream_id for episode in bundle.episodes()]
    if len(episode_ids) != len(set(episode_ids)):
        errors.append("episode identifiers are not globally unique")
    if len(stream_ids) != len(set(stream_ids)):
        errors.append("RNG stream identifiers are not globally disjoint")

    substream_names = [record.name for record in bundle.named_substreams]
    substream_hashes = [record.derivation_sha256 for record in bundle.named_substreams]
    if len(substream_names) != len(set(substream_names)):
        errors.append("named RNG substreams are duplicated")
    if len(substream_hashes) != len(set(substream_hashes)):
        errors.append("named RNG substream derivations collided")

    hashes = {name: schedule_sha256(episodes) for name, episodes in bundle.splits()}
    paired_hashes = paired_method_schedule_hashes(tuple(bundle.episodes()))
    if len(set(paired_hashes.values())) != 1:
        errors.append("paired methods do not share one byte-identical schedule hash")

    return {
        "counts": actual_counts,
        "errors": errors,
        "paired_method_schedule_sha256": paired_hashes,
        "split_sha256": hashes,
        "status": "PASS" if not errors else "FAIL",
    }


def validate_protocol_config(path: str | Path) -> dict[str, Any]:
    """Check that the existing blocked config matches the frozen schedule sizes."""

    config_path = Path(path)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    status = config.get("status", {})
    experiment = config.get("experiment", {})
    expected = {
        "implementation_ready": False,
        "phase_a_episodes": 4000,
        "phase_a_operations": list(PHASE_A_OPERATIONS),
        "phase_b_episodes": 2000,
        "phase_b_operations": list(PHASE_B_OPERATIONS),
        "train_delay_min": 4,
        "train_delay_max": 8,
        "eval_delay": 8,
        "secondary_eval_delay": 16,
    }
    observed = {
        "implementation_ready": status.get("implementation_ready"),
        "phase_a_episodes": experiment.get("phase_a_episodes"),
        "phase_a_operations": experiment.get("phase_a_operations"),
        "phase_b_episodes": experiment.get("phase_b_episodes"),
        "phase_b_operations": experiment.get("phase_b_operations"),
        "train_delay_min": experiment.get("train_delay_min"),
        "train_delay_max": experiment.get("train_delay_max"),
        "eval_delay": experiment.get("eval_delay"),
        "secondary_eval_delay": experiment.get("secondary_eval_delay"),
    }
    errors = [
        f"{key}: expected {value!r}, observed {observed[key]!r}"
        for key, value in expected.items()
        if observed[key] != value
    ]
    return {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "errors": errors,
        "observed": observed,
        "status": "PASS" if not errors else "FAIL",
    }


def build_manifest(
    bundle: Experiment001ScheduleBundle,
    *,
    config_path: str | Path,
) -> dict[str, Any]:
    validation = validate_development_bundle(bundle)
    config_validation = validate_protocol_config(config_path)
    status = "PASS" if validation["status"] == config_validation["status"] == "PASS" else "FAIL"
    return {
        "authorization": {
            "confirmatory": False,
            "runner_implemented": False,
            "scope": "development_schedule_only",
            "targets_terminal_only": True,
        },
        "config_validation": config_validation,
        "master_seed": bundle.master_seed,
        "named_substreams": [record.as_dict() for record in bundle.named_substreams],
        "operation_token_schema": list(MicroProgram3BitEnvironment.OPERATIONS),
        "phase_b_checkpoints": list(PHASE_B_CHECKPOINTS),
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "validation": validation,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an Experiment 001 development schedule manifest.")
    parser.add_argument("--seed", type=int, default=0, help="registered development seed (0..4)")
    parser.add_argument("--config", default="configs/experiment_001.toml")
    parser.add_argument("--output", type=Path, help="optional manifest path; stdout otherwise")
    args = parser.parse_args(argv)

    bundle = build_development_bundle(args.seed)
    manifest = build_manifest(bundle, config_path=args.config)
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())


__all__ = [
    "COMPOSITION_PROGRAMS",
    "DEVELOPMENT_SEEDS",
    "PAIRED_METHOD_ROLES",
    "PHASE_A_OPERATIONS",
    "PHASE_B_CHECKPOINTS",
    "PHASE_B_OPERATIONS",
    "PRIMITIVE_PROGRAMS",
    "PROTOCOL_VERSION",
    "RESERVED_CONFIRMATORY_SEEDS",
    "Experiment001ScheduleBundle",
    "Experiment001VisibleEvent",
    "NamedSubstream",
    "ScheduleEpisode",
    "build_development_bundle",
    "build_evaluation_schedule",
    "build_manifest",
    "build_training_schedules",
    "canonical_episode_bytes",
    "main",
    "named_substream",
    "paired_method_schedule_hashes",
    "schedule_sha256",
    "validate_development_bundle",
    "validate_protocol_config",
]
