"""Deterministic data protocol for the Experiment-000 delayed-cue gate.

An episode exposes only this event sequence to a learner::

    t=0       CUE(c)
    t=1..8    NOISE(bit)
    t=9       QUERY

The terminal target is stored separately and equals ``c``.  Consequently a
runner can request a prediction after ``QUERY`` and only then deliver the
target.  Training and evaluation use independently derived, named random
substreams.  Their episode/noise-stream identities are also disjoint.

There are only 256 possible visible eight-bit noise words.  A 2,000-episode
split must therefore repeat some visible words.  "Disjoint streams" means
disjoint RNG namespaces and sample identities, not the impossible requirement
that the two splits contain disjoint sets of eight-bit values.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Literal

import numpy as np

PROTOCOL_VERSION = "experiment-000-data-v1"
DEFAULT_EPISODES_PER_SPLIT = 2_000
DEFAULT_NOISE_EVENTS = 8
DEVELOPMENT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
CONFIRMATORY_SEEDS: tuple[int, ...] = tuple(range(1_000, 1_020))
ANALYSIS_SEED = 20_260_901

Split = Literal["train", "eval"]
TokenKind = Literal["CUE", "NOISE", "QUERY"]


class SeedRole(str, Enum):
    """Pre-registered use of a master seed."""

    DEVELOPMENT = "development"
    CONFIRMATORY = "confirmatory"
    CUSTOM = "custom"


def classify_seed(master_seed: int) -> SeedRole:
    """Return the pre-registered role of ``master_seed``.

    Unknown non-negative seeds are useful for unit tests and diagnostics and
    are labelled ``CUSTOM``.  Call :func:`require_seed_role` when a run must be
    restricted to one of the pre-registered partitions.
    """

    _validate_master_seed(master_seed)
    if master_seed in DEVELOPMENT_SEEDS:
        return SeedRole.DEVELOPMENT
    if master_seed in CONFIRMATORY_SEEDS:
        return SeedRole.CONFIRMATORY
    return SeedRole.CUSTOM


def require_seed_role(master_seed: int, role: SeedRole | str) -> None:
    """Reject accidental use of a seed outside the requested partition."""

    expected = SeedRole(role)
    actual = classify_seed(master_seed)
    if expected is SeedRole.CUSTOM:
        if actual is not SeedRole.CUSTOM:
            raise ValueError(
                f"seed {master_seed} is reserved for {actual.value}, not custom use"
            )
        return
    if actual is not expected:
        raise ValueError(
            f"seed {master_seed} is {actual.value}; required role is {expected.value}"
        )


@dataclass(frozen=True, slots=True)
class Experiment000Event:
    """One learner-visible event.

    ``value`` is binary for ``CUE`` and ``NOISE`` and is ``None`` for
    ``QUERY``.  There is deliberately no learner-visible target event here.
    """

    time: int
    kind: TokenKind
    value: int | None


@dataclass(frozen=True, slots=True)
class Experiment000Episode:
    """Immutable delayed-cue episode description."""

    episode_id: str
    noise_stream_id: str
    split: Split
    master_seed: int
    index: int
    cue: int
    noise: tuple[int, ...]
    target: int

    @property
    def events(self) -> tuple[Experiment000Event, ...]:
        """Return exactly ``CUE``, the noise events, and ``QUERY`` in order."""

        return (
            Experiment000Event(time=0, kind="CUE", value=self.cue),
            *(
                Experiment000Event(time=time, kind="NOISE", value=bit)
                for time, bit in enumerate(self.noise, start=1)
            ),
            Experiment000Event(time=len(self.noise) + 1, kind="QUERY", value=None),
        )

    def canonical_record(self) -> dict[str, object]:
        """Return the stable, JSON-ready representation used for hashing."""

        return {
            "cue": self.cue,
            "episode_id": self.episode_id,
            "index": self.index,
            "master_seed": self.master_seed,
            "noise": list(self.noise),
            "noise_stream_id": self.noise_stream_id,
            "split": self.split,
            "target": self.target,
        }


@dataclass(frozen=True, slots=True)
class SubstreamManifest:
    """Auditable named RNG substreams for one train/evaluation split."""

    split: Split
    namespace: str
    cue_order_seed: int
    noise_seed: int
    identity_seed: int
    bit_generator: str = "numpy.random.PCG64"


@dataclass(frozen=True, slots=True)
class Experiment000Stream:
    """One complete, immutable train or evaluation stream."""

    split: Split
    master_seed: int
    seed_role: SeedRole
    noise_events: int
    substreams: SubstreamManifest
    episodes: tuple[Experiment000Episode, ...]

    @property
    def sha256(self) -> str:
        return stream_sha256(self)

    @property
    def episode_count(self) -> int:
        return len(self.episodes)


@dataclass(frozen=True, slots=True)
class Experiment000Manifest:
    """Hash and seed record for a paired train/evaluation stream."""

    protocol_version: str
    master_seed: int
    seed_role: str
    episodes_per_split: int
    noise_events: int
    train_sha256: str
    eval_sha256: str
    pair_sha256: str
    train_substreams: SubstreamManifest
    eval_substreams: SubstreamManifest

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":") if indent is None else None, indent=indent
        )


@dataclass(frozen=True, slots=True)
class StreamValidation:
    """Concise successful validation result."""

    split: Split
    episode_count: int
    cue_zeros: int
    cue_ones: int
    noise_events: int
    sha256: str


@dataclass(frozen=True, slots=True)
class StreamPair:
    """Convenience bundle returned by :func:`generate_stream_pair`."""

    train: Experiment000Stream
    eval: Experiment000Stream
    manifest: Experiment000Manifest


def _validate_master_seed(master_seed: int) -> None:
    if isinstance(master_seed, bool) or not isinstance(master_seed, int) or master_seed < 0:
        raise ValueError("master_seed must be a non-negative integer")


def _validate_generation_arguments(
    *, master_seed: int, split: str, episode_count: int, noise_events: int
) -> None:
    _validate_master_seed(master_seed)
    if split not in ("train", "eval"):
        raise ValueError("split must be 'train' or 'eval'")
    if isinstance(episode_count, bool) or not isinstance(episode_count, int) or episode_count <= 0:
        raise ValueError("episode_count must be a positive integer")
    if episode_count % 2:
        raise ValueError("episode_count must be even for exact cue balance")
    if isinstance(noise_events, bool) or not isinstance(noise_events, int) or noise_events <= 0:
        raise ValueError("noise_events must be a positive integer")


def _derive_seed(master_seed: int, label: str) -> int:
    """Derive a portable 128-bit seed from a master seed and a named label."""

    material = f"{PROTOCOL_VERSION}|{master_seed}|{label}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:16], "big")


def derive_substream_manifest(master_seed: int, split: Split) -> SubstreamManifest:
    """Create independent named substream seeds for one split."""

    _validate_generation_arguments(
        master_seed=master_seed, split=split, episode_count=2, noise_events=1
    )
    namespace = f"{PROTOCOL_VERSION}/seed-{master_seed}/{split}"
    return SubstreamManifest(
        split=split,
        namespace=namespace,
        cue_order_seed=_derive_seed(master_seed, f"{split}/cue-order"),
        noise_seed=_derive_seed(master_seed, f"{split}/noise"),
        identity_seed=_derive_seed(master_seed, f"{split}/identity"),
    )


def _identity(prefix: str, identity_seed: int, index: int) -> str:
    material = f"{PROTOCOL_VERSION}|{prefix}|{identity_seed}|{index}".encode()
    return hashlib.sha256(material).hexdigest()[:24]


def _canonical_episodes(
    *,
    master_seed: int,
    split: Split,
    episode_count: int,
    noise_events: int,
    substreams: SubstreamManifest,
) -> tuple[Experiment000Episode, ...]:
    """Reconstruct canonical episodes from the declared RNG provenance."""

    cues = np.concatenate(
        (
            np.zeros(episode_count // 2, dtype=np.uint8),
            np.ones(episode_count // 2, dtype=np.uint8),
        )
    )
    order_rng = np.random.Generator(np.random.PCG64(substreams.cue_order_seed))
    cues = cues[order_rng.permutation(episode_count)]

    noise_rng = np.random.Generator(np.random.PCG64(substreams.noise_seed))
    noise_matrix = noise_rng.integers(
        0, 2, size=(episode_count, noise_events), dtype=np.uint8
    )

    return tuple(
        Experiment000Episode(
            episode_id=f"exp000-{split}-{_identity('episode', substreams.identity_seed, index)}",
            noise_stream_id=f"noise-{split}-{_identity('noise', substreams.identity_seed, index)}",
            split=split,
            master_seed=master_seed,
            index=index,
            cue=int(cues[index]),
            noise=tuple(int(bit) for bit in noise_matrix[index]),
            target=int(cues[index]),
        )
        for index in range(episode_count)
    )


def generate_stream(
    master_seed: int,
    split: Split,
    *,
    episode_count: int = DEFAULT_EPISODES_PER_SPLIT,
    noise_events: int = DEFAULT_NOISE_EVENTS,
) -> Experiment000Stream:
    """Generate an exactly cue-balanced, deterministically shuffled stream.

    The noise matrix is produced by independent calls to PCG64's binary integer
    sampler.  Train and evaluation have different derived seeds even when they
    share the same master seed.
    """

    _validate_generation_arguments(
        master_seed=master_seed,
        split=split,
        episode_count=episode_count,
        noise_events=noise_events,
    )
    typed_split: Split = split
    substreams = derive_substream_manifest(master_seed, typed_split)
    episodes = _canonical_episodes(
        master_seed=master_seed,
        split=typed_split,
        episode_count=episode_count,
        noise_events=noise_events,
        substreams=substreams,
    )
    stream = Experiment000Stream(
        split=typed_split,
        master_seed=master_seed,
        seed_role=classify_seed(master_seed),
        noise_events=noise_events,
        substreams=substreams,
        episodes=episodes,
    )
    validate_stream(stream)
    return stream


def generate_stream_pair(
    master_seed: int,
    *,
    episode_count: int = DEFAULT_EPISODES_PER_SPLIT,
    noise_events: int = DEFAULT_NOISE_EVENTS,
    required_role: SeedRole | str | None = None,
) -> StreamPair:
    """Generate, validate, hash, and manifest paired train/evaluation data."""

    if required_role is not None:
        require_seed_role(master_seed, required_role)
    train = generate_stream(
        master_seed, "train", episode_count=episode_count, noise_events=noise_events
    )
    evaluation = generate_stream(
        master_seed, "eval", episode_count=episode_count, noise_events=noise_events
    )
    validate_stream_pair(train, evaluation)
    manifest = build_manifest(train, evaluation)
    return StreamPair(train=train, eval=evaluation, manifest=manifest)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def stream_sha256(
    stream: Experiment000Stream | Sequence[Experiment000Episode],
) -> str:
    """Hash episode descriptions with a stable canonical JSON encoding."""

    episodes = stream.episodes if isinstance(stream, Experiment000Stream) else tuple(stream)
    hasher = hashlib.sha256()
    hasher.update(f"{PROTOCOL_VERSION}\n".encode("ascii"))
    for episode in episodes:
        hasher.update(_canonical_json(episode.canonical_record()))
        hasher.update(b"\n")
    return hasher.hexdigest()


def pair_sha256(train: Experiment000Stream, evaluation: Experiment000Stream) -> str:
    """Hash the ordered train/evaluation hash pair."""

    payload = {
        "eval": stream_sha256(evaluation),
        "protocol_version": PROTOCOL_VERSION,
        "train": stream_sha256(train),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def build_manifest(
    train: Experiment000Stream, evaluation: Experiment000Stream
) -> Experiment000Manifest:
    """Validate a stream pair and create its reproducibility manifest."""

    validate_stream_pair(train, evaluation)
    return Experiment000Manifest(
        protocol_version=PROTOCOL_VERSION,
        master_seed=train.master_seed,
        seed_role=train.seed_role.value,
        episodes_per_split=train.episode_count,
        noise_events=train.noise_events,
        train_sha256=stream_sha256(train),
        eval_sha256=stream_sha256(evaluation),
        pair_sha256=pair_sha256(train, evaluation),
        train_substreams=train.substreams,
        eval_substreams=evaluation.substreams,
    )


def validate_episode(
    episode: Experiment000Episode, *, expected_noise_events: int = DEFAULT_NOISE_EVENTS
) -> None:
    """Validate timing, binary domains, and post-query target separation."""

    if episode.split not in ("train", "eval"):
        raise ValueError(f"episode {episode.episode_id}: invalid split")
    _validate_master_seed(episode.master_seed)
    if episode.index < 0:
        raise ValueError(f"episode {episode.episode_id}: negative index")
    if episode.cue not in (0, 1):
        raise ValueError(f"episode {episode.episode_id}: cue must be binary")
    if episode.target not in (0, 1) or episode.target != episode.cue:
        raise ValueError(f"episode {episode.episode_id}: target must equal cue")
    if len(episode.noise) != expected_noise_events:
        raise ValueError(
            f"episode {episode.episode_id}: expected {expected_noise_events} noise events, "
            f"got {len(episode.noise)}"
        )
    if any(bit not in (0, 1) for bit in episode.noise):
        raise ValueError(f"episode {episode.episode_id}: all noise values must be binary")

    events = episode.events
    expected_kinds: tuple[TokenKind, ...] = (
        "CUE",
        *("NOISE" for _ in range(expected_noise_events)),
        "QUERY",
    )
    if tuple(event.time for event in events) != tuple(range(expected_noise_events + 2)):
        raise ValueError(f"episode {episode.episode_id}: event times are not contiguous")
    if tuple(event.kind for event in events) != expected_kinds:
        raise ValueError(f"episode {episode.episode_id}: event order is invalid")
    if events[-1].value is not None:
        raise ValueError(f"episode {episode.episode_id}: QUERY must not expose the target")


def validate_balance(episodes: Iterable[Experiment000Episode]) -> tuple[int, int]:
    """Require exact 50/50 cue balance and return ``(zeros, ones)``."""

    values = tuple(episodes)
    if not values:
        raise ValueError("stream must contain at least one episode")
    zeros = sum(episode.cue == 0 for episode in values)
    ones = sum(episode.cue == 1 for episode in values)
    if zeros != ones:
        raise ValueError(f"cue imbalance: zeros={zeros}, ones={ones}")
    return zeros, ones


def validate_stream(stream: Experiment000Stream) -> StreamValidation:
    """Validate one stream's structure, balance, IDs, and seed manifest."""

    _validate_generation_arguments(
        master_seed=stream.master_seed,
        split=stream.split,
        episode_count=stream.episode_count,
        noise_events=stream.noise_events,
    )
    if stream.substreams.split != stream.split:
        raise ValueError("substream split does not match stream split")
    if stream.substreams != derive_substream_manifest(stream.master_seed, stream.split):
        raise ValueError("substream manifest does not match the master seed and split")
    if stream.seed_role is not classify_seed(stream.master_seed):
        raise ValueError("seed role does not match the master seed")
    if stream.episode_count % 2:
        raise ValueError("stream size must be even")

    episode_ids: set[str] = set()
    noise_ids: set[str] = set()
    for expected_index, episode in enumerate(stream.episodes):
        validate_episode(episode, expected_noise_events=stream.noise_events)
        if episode.split != stream.split or episode.master_seed != stream.master_seed:
            raise ValueError(f"episode {episode.episode_id}: stream metadata mismatch")
        if episode.index != expected_index:
            raise ValueError(
                f"episode {episode.episode_id}: index {episode.index} != {expected_index}"
            )
        if episode.episode_id in episode_ids:
            raise ValueError(f"duplicate episode id: {episode.episode_id}")
        if episode.noise_stream_id in noise_ids:
            raise ValueError(f"duplicate noise stream id: {episode.noise_stream_id}")
        episode_ids.add(episode.episode_id)
        noise_ids.add(episode.noise_stream_id)

    zeros, ones = validate_balance(stream.episodes)

    expected_episodes = _canonical_episodes(
        master_seed=stream.master_seed,
        split=stream.split,
        episode_count=stream.episode_count,
        noise_events=stream.noise_events,
        substreams=stream.substreams,
    )
    for expected_index, (episode, expected) in enumerate(
        zip(stream.episodes, expected_episodes, strict=True)
    ):
        actual_record = episode.canonical_record()
        expected_record = expected.canonical_record()
        mismatched_fields = tuple(
            field
            for field, expected_value in expected_record.items()
            # Compare the canonical JSON representation rather than Python
            # equality.  Python considers values such as ``True`` and ``1``
            # (or ``1.0`` and ``1``) equal, but those values produce different
            # canonical records and therefore different stream hashes.
            if _canonical_json(actual_record[field])
            != _canonical_json(expected_value)
        )
        if mismatched_fields:
            fields = ", ".join(mismatched_fields)
            identity_context = (
                "; identity leakage or tampering"
                if {"episode_id", "noise_stream_id"}.intersection(mismatched_fields)
                else ""
            )
            raise ValueError(
                f"episode {expected_index}: deterministic provenance mismatch in "
                f"{fields}{identity_context}"
            )

    return StreamValidation(
        split=stream.split,
        episode_count=stream.episode_count,
        cue_zeros=zeros,
        cue_ones=ones,
        noise_events=stream.noise_events,
        sha256=stream_sha256(stream),
    )


def validate_stream_pair(
    train: Experiment000Stream, evaluation: Experiment000Stream
) -> None:
    """Reject split leakage, mismatched settings, or inconsistent seeds."""

    validate_stream(train)
    validate_stream(evaluation)
    if train.split != "train" or evaluation.split != "eval":
        raise ValueError("pair must be ordered as train, eval")
    if train.master_seed != evaluation.master_seed:
        raise ValueError("train and eval must share the paired master seed")
    if train.seed_role is not evaluation.seed_role:
        raise ValueError("train and eval seed roles differ")
    if train.episode_count != evaluation.episode_count:
        raise ValueError("train and eval episode counts differ")
    if train.noise_events != evaluation.noise_events:
        raise ValueError("train and eval noise-event counts differ")
    if train.substreams.namespace == evaluation.substreams.namespace:
        raise ValueError("train/eval RNG namespaces overlap")
    if train.substreams.cue_order_seed == evaluation.substreams.cue_order_seed:
        raise ValueError("train/eval cue-order RNG seeds overlap")
    if train.substreams.noise_seed == evaluation.substreams.noise_seed:
        raise ValueError("train/eval noise RNG seeds overlap")
    if train.substreams.identity_seed == evaluation.substreams.identity_seed:
        raise ValueError("train/eval identity RNG seeds overlap")

    train_episode_ids = {episode.episode_id for episode in train.episodes}
    eval_episode_ids = {episode.episode_id for episode in evaluation.episodes}
    if train_episode_ids.intersection(eval_episode_ids):
        raise ValueError("train/eval episode identity leakage")
    train_noise_ids = {episode.noise_stream_id for episode in train.episodes}
    eval_noise_ids = {episode.noise_stream_id for episode in evaluation.episodes}
    if train_noise_ids.intersection(eval_noise_ids):
        raise ValueError("train/eval noise-stream identity leakage")
    if stream_sha256(train) == stream_sha256(evaluation):
        raise ValueError("train and eval stream hashes must differ")


__all__ = [
    "ANALYSIS_SEED",
    "CONFIRMATORY_SEEDS",
    "DEFAULT_EPISODES_PER_SPLIT",
    "DEFAULT_NOISE_EVENTS",
    "DEVELOPMENT_SEEDS",
    "PROTOCOL_VERSION",
    "Experiment000Episode",
    "Experiment000Event",
    "Experiment000Manifest",
    "Experiment000Stream",
    "SeedRole",
    "StreamPair",
    "StreamValidation",
    "SubstreamManifest",
    "build_manifest",
    "classify_seed",
    "derive_substream_manifest",
    "generate_stream",
    "generate_stream_pair",
    "pair_sha256",
    "require_seed_role",
    "stream_sha256",
    "validate_balance",
    "validate_episode",
    "validate_stream",
    "validate_stream_pair",
]
