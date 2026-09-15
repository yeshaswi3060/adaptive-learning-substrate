from dataclasses import replace

import pytest

from adaptive_learning_substrate.experiment000_data import (
    generate_stream,
    validate_stream,
)


def _replace_episode(stream, index, **changes):
    episodes = list(stream.episodes)
    episodes[index] = replace(episodes[index], **changes)
    return replace(stream, episodes=tuple(episodes))


@pytest.mark.parametrize("split", ["train", "eval"])
def test_valid_stream_matches_deterministic_provenance(split: str) -> None:
    stream = generate_stream(17, split, episode_count=40, noise_events=5)

    validation = validate_stream(stream)

    assert validation.split == split
    assert validation.episode_count == 40
    assert validation.sha256 == stream.sha256


def test_validation_rejects_balanced_cue_tampering() -> None:
    stream = generate_stream(17, "train", episode_count=40, noise_events=5)
    first = 0
    second = next(
        index
        for index, episode in enumerate(stream.episodes)
        if episode.cue != stream.episodes[first].cue
    )
    episodes = list(stream.episodes)
    episodes[first] = replace(
        episodes[first], cue=episodes[second].cue, target=episodes[second].cue
    )
    episodes[second] = replace(
        episodes[second], cue=stream.episodes[first].cue, target=stream.episodes[first].cue
    )
    tampered = replace(stream, episodes=tuple(episodes))

    with pytest.raises(ValueError, match="deterministic provenance mismatch in cue"):
        validate_stream(tampered)


def test_validation_rejects_binary_noise_tampering() -> None:
    stream = generate_stream(17, "train", episode_count=40, noise_events=5)
    episode = stream.episodes[7]
    noise = (1 - episode.noise[0], *episode.noise[1:])
    tampered = _replace_episode(stream, 7, noise=noise)

    with pytest.raises(ValueError, match="deterministic provenance mismatch in noise"):
        validate_stream(tampered)


def test_validation_rejects_fabricated_zero_length_noise_protocol() -> None:
    stream = generate_stream(17, "train", episode_count=40, noise_events=1)
    fabricated = replace(
        stream,
        noise_events=0,
        episodes=tuple(replace(episode, noise=()) for episode in stream.episodes),
    )

    with pytest.raises(ValueError, match="noise_events must be a positive integer"):
        validate_stream(fabricated)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("episode_id", "exp000-train-forged"),
        ("noise_stream_id", "noise-train-forged"),
    ],
)
def test_validation_rejects_tampered_derived_identifier(field: str, value: str) -> None:
    stream = generate_stream(17, "train", episode_count=40, noise_events=5)
    tampered = _replace_episode(stream, 11, **{field: value})

    with pytest.raises(ValueError, match=rf"deterministic provenance mismatch in {field}"):
        validate_stream(tampered)
