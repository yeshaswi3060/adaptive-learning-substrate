import json
from dataclasses import replace

import pytest

from adaptive_learning_substrate import experiment000_data
from adaptive_learning_substrate.experiment000_data import (
    CONFIRMATORY_SEEDS,
    DEFAULT_EPISODES_PER_SPLIT,
    DEVELOPMENT_SEEDS,
    Experiment000Stream,
    SeedRole,
    classify_seed,
    generate_stream,
    generate_stream_pair,
    require_seed_role,
    stream_sha256,
    validate_stream,
    validate_stream_pair,
)


def test_default_episode_protocol_is_balanced_and_target_is_post_query() -> None:
    pair = generate_stream_pair(0, required_role=SeedRole.DEVELOPMENT)

    assert pair.train.episode_count == DEFAULT_EPISODES_PER_SPLIT == 2_000
    assert pair.eval.episode_count == 2_000
    for stream in (pair.train, pair.eval):
        validation = validate_stream(stream)
        assert validation.cue_zeros == validation.cue_ones == 1_000
        for episode in stream.episodes:
            assert episode.target == episode.cue
            assert len(episode.noise) == 8
            assert set(episode.noise) <= {0, 1}
            assert [event.kind for event in episode.events] == [
                "CUE",
                *(["NOISE"] * 8),
                "QUERY",
            ]
            assert [event.time for event in episode.events] == list(range(10))
            assert episode.events[-1].value is None


def test_generation_and_manifest_hashes_are_reproducible() -> None:
    first = generate_stream_pair(3, episode_count=40)
    second = generate_stream_pair(3, episode_count=40)

    assert first == second
    assert first.train.sha256 == second.train.sha256
    assert first.eval.sha256 == second.eval.sha256
    assert first.manifest.pair_sha256 == second.manifest.pair_sha256
    assert first.manifest.train_sha256 == stream_sha256(first.train.episodes)
    assert json.loads(first.manifest.to_json())["protocol_version"] == "experiment-000-data-v1"


def test_train_eval_substreams_and_sample_identities_are_disjoint() -> None:
    pair = generate_stream_pair(4, episode_count=512)
    validate_stream_pair(pair.train, pair.eval)

    assert pair.train.substreams.namespace != pair.eval.substreams.namespace
    assert pair.train.substreams.cue_order_seed != pair.eval.substreams.cue_order_seed
    assert pair.train.substreams.noise_seed != pair.eval.substreams.noise_seed
    assert pair.train.sha256 != pair.eval.sha256
    assert not ({x.episode_id for x in pair.train.episodes} & {x.episode_id for x in pair.eval.episodes})
    assert not (
        {x.noise_stream_id for x in pair.train.episodes}
        & {x.noise_stream_id for x in pair.eval.episodes}
    )


def test_different_master_seed_changes_both_split_hashes() -> None:
    first = generate_stream_pair(0, episode_count=100)
    second = generate_stream_pair(1, episode_count=100)
    assert first.train.sha256 != second.train.sha256
    assert first.eval.sha256 != second.eval.sha256


def test_development_and_reserved_confirmatory_seed_partitions_are_separate() -> None:
    assert set(DEVELOPMENT_SEEDS).isdisjoint(CONFIRMATORY_SEEDS)
    assert [classify_seed(seed) for seed in DEVELOPMENT_SEEDS] == [
        SeedRole.DEVELOPMENT
    ] * 5
    assert all(classify_seed(seed) is SeedRole.CONFIRMATORY for seed in CONFIRMATORY_SEEDS)
    require_seed_role(1_000, SeedRole.CONFIRMATORY)
    with pytest.raises(ValueError, match="required role is development"):
        require_seed_role(1_000, SeedRole.DEVELOPMENT)
    with pytest.raises(ValueError, match="required role is confirmatory"):
        generate_stream_pair(0, episode_count=20, required_role=SeedRole.CONFIRMATORY)


def test_validation_rejects_imbalance_target_leakage_and_split_identity_leakage() -> None:
    pair = generate_stream_pair(2, episode_count=20)

    bad_target_episode = replace(pair.train.episodes[0], target=1 - pair.train.episodes[0].cue)
    bad_target = replace(
        pair.train, episodes=(bad_target_episode, *pair.train.episodes[1:])
    )
    with pytest.raises(ValueError, match="target must equal cue"):
        validate_stream(bad_target)

    one_episode = pair.train.episodes[0]
    replacement_index = next(
        index for index, episode in enumerate(pair.train.episodes) if episode.cue != one_episode.cue
    )
    imbalanced_episode = replace(
        pair.train.episodes[replacement_index], cue=one_episode.cue, target=one_episode.cue
    )
    imbalanced_episodes = list(pair.train.episodes)
    imbalanced_episodes[replacement_index] = imbalanced_episode
    imbalanced = replace(pair.train, episodes=tuple(imbalanced_episodes))
    with pytest.raises(ValueError, match="cue imbalance"):
        validate_stream(imbalanced)

    leaked_eval = Experiment000Stream(
        split="eval",
        master_seed=pair.eval.master_seed,
        seed_role=pair.eval.seed_role,
        noise_events=pair.eval.noise_events,
        substreams=pair.eval.substreams,
        episodes=(
            replace(
                pair.eval.episodes[0],
                episode_id=pair.train.episodes[0].episode_id,
                noise_stream_id=pair.train.episodes[0].noise_stream_id,
            ),
            *pair.eval.episodes[1:],
        ),
    )
    with pytest.raises(ValueError, match="identity leakage"):
        validate_stream_pair(pair.train, leaked_eval)


def test_validation_rejects_noise_not_generated_by_declared_rng() -> None:
    stream = generate_stream(2, "train", episode_count=20)
    final = stream.episodes[-1]
    tampered_noise = (1 - final.noise[0], *final.noise[1:])
    tampered = replace(
        stream,
        episodes=(*stream.episodes[:-1], replace(final, noise=tampered_noise)),
    )

    assert stream_sha256(tampered) != stream_sha256(stream)
    with pytest.raises(
        ValueError, match=r"episode 19: deterministic provenance mismatch in noise"
    ):
        validate_stream(tampered)


def test_validation_compares_exact_canonical_value_types() -> None:
    stream = generate_stream(2, "train", episode_count=20)
    second = stream.episodes[1]
    type_tampered = replace(
        stream,
        episodes=(
            stream.episodes[0],
            replace(second, index=True),
            *stream.episodes[2:],
        ),
    )

    # ``True == 1`` in Python, but the canonical JSON records and hashes differ.
    assert stream_sha256(type_tampered) != stream_sha256(stream)
    with pytest.raises(
        ValueError, match=r"episode 1: deterministic provenance mismatch in index"
    ):
        validate_stream(type_tampered)


def test_validation_reconstructs_provenance_without_public_generation_reentry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = generate_stream(2, "train", episode_count=20)

    def fail_on_reentry(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"unexpected generation reentry: {args!r}, {kwargs!r}")

    monkeypatch.setattr(experiment000_data, "generate_stream", fail_on_reentry)
    validation = validate_stream(stream)

    assert validation.sha256 == stream_sha256(stream)


@pytest.mark.parametrize(
    ("episode_count", "noise_events", "message"),
    [(9, 8, "even"), (0, 8, "positive"), (10, 0, "positive")],
)
def test_invalid_generation_shapes_are_rejected(
    episode_count: int, noise_events: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        generate_stream(0, "train", episode_count=episode_count, noise_events=noise_events)
