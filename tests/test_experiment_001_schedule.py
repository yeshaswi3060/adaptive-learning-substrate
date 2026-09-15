from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from adaptive_learning_substrate.environments import MicroProgram3BitEnvironment
from adaptive_learning_substrate.experiment_001_schedule import (
    COMPOSITION_PROGRAMS,
    PAIRED_METHOD_ROLES,
    PHASE_B_CHECKPOINTS,
    build_development_bundle,
    build_evaluation_schedule,
    build_manifest,
    build_training_schedules,
    main,
    paired_method_schedule_hashes,
    schedule_sha256,
    validate_development_bundle,
    validate_protocol_config,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def development_bundle():
    return build_development_bundle(0)


def test_training_schedule_is_exactly_balanced_and_deterministic() -> None:
    first_a, first_b = build_training_schedules(0)
    second_a, second_b = build_training_schedules(0)

    assert len(first_a) == 4000
    assert len(first_b) == 2000
    assert schedule_sha256(first_a) == schedule_sha256(second_a)
    assert schedule_sha256(first_b) == schedule_sha256(second_b)
    assert schedule_sha256(first_a) != schedule_sha256(build_training_schedules(1)[0])

    phase_a_counts = Counter((episode.operations, episode.inputs) for episode in first_a)
    phase_b_counts = Counter((episode.operations, episode.inputs) for episode in first_b)
    assert set(phase_a_counts.values()) == {250}
    assert len(phase_a_counts) == 2 * 8
    assert set(phase_b_counts.values()) == {250}
    assert len(phase_b_counts) == 8
    assert all(4 <= episode.delay <= 8 for episode in first_a + first_b)
    assert all(len(episode.noise) == episode.delay for episode in first_a + first_b)


def test_evaluation_has_64_streams_per_word_and_left_to_right_targets() -> None:
    episodes = build_evaluation_schedule(
        0,
        split="primary_compositions",
        programs=COMPOSITION_PROGRAMS,
        delay=8,
    )
    assert len(episodes) == 9 * 8 * 64
    counts = Counter((episode.operations, episode.inputs) for episode in episodes)
    assert set(counts.values()) == {64}
    assert len(counts) == 9 * 8

    bits = (0, 0, 0)
    forward = next(
        episode
        for episode in episodes
        if episode.operations == ("FLIP0", "ROTL") and episode.inputs == bits
    )
    assert forward.target == MicroProgram3BitEnvironment.compose(bits, ("FLIP0", "ROTL"))
    assert forward.target != MicroProgram3BitEnvironment.compose(bits, ("ROTL", "FLIP0"))
    visible = forward.visible_events()
    assert [event.kind for event in visible] == [
        "START",
        "OPERATION",
        "OPERATION",
        *("NOISE" for _ in range(8)),
        "QUERY",
    ]
    assert tuple(
        event.operation for event in visible if event.kind == "OPERATION"
    ) == ("FLIP0", "ROTL")
    assert all("target" not in event.as_dict() for event in visible)


def test_bundle_validates_disjoint_streams_and_paired_method_bytes(development_bundle) -> None:
    validation = validate_development_bundle(development_bundle)
    assert validation["status"] == "PASS"
    assert validation["errors"] == []
    assert validation["counts"] == {
        "phase_a_train": 4000,
        "phase_b_train": 2000,
        "primary_primitives": 1536,
        "primary_compositions": 4608,
        "secondary_primitives": 1536,
        "secondary_compositions": 4608,
    }

    episodes = tuple(development_bundle.episodes())
    assert len({episode.episode_id for episode in episodes}) == len(episodes)
    assert len({episode.stream_id for episode in episodes}) == len(episodes)
    hashes = paired_method_schedule_hashes(episodes)
    assert tuple(hashes) == PAIRED_METHOD_ROLES
    assert len(set(hashes.values())) == 1
    assert len({item.derivation_sha256 for item in development_bundle.named_substreams}) == len(
        development_bundle.named_substreams
    )
    seed_one = build_development_bundle(1)
    assert [item.derivation_sha256 for item in seed_one.named_substreams] != [
        item.derivation_sha256 for item in development_bundle.named_substreams
    ]


def test_protocol_config_remains_blocked_and_matches_schedule_constants() -> None:
    result = validate_protocol_config(ROOT / "configs" / "experiment_001.toml")
    assert result["status"] == "PASS"
    assert result["errors"] == []
    assert result["observed"]["implementation_ready"] is False


def test_manifest_and_cli_are_deterministic_development_only(
    development_bundle, tmp_path
) -> None:
    config = ROOT / "configs" / "experiment_001.toml"
    first = build_manifest(development_bundle, config_path=config)
    second = build_manifest(build_development_bundle(0), config_path=config)
    assert first == second
    assert first["status"] == "PASS"
    assert first["authorization"] == {
        "confirmatory": False,
        "runner_implemented": False,
        "scope": "development_schedule_only",
        "targets_terminal_only": True,
    }
    assert first["operation_token_schema"] == ["FLIP0", "ROTL", "NOT"]
    assert first["phase_b_checkpoints"] == list(PHASE_B_CHECKPOINTS)

    output = tmp_path / "manifest.json"
    assert main(["--seed", "0", "--config", str(config), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == first


def test_confirmatory_and_unregistered_seeds_are_not_materialized() -> None:
    with pytest.raises(RuntimeError, match="confirmatory schedules remain blocked"):
        build_development_bundle(1000)
    with pytest.raises(ValueError, match="development seed"):
        build_development_bundle(17)


def test_validator_rejects_terminal_target_corruption(development_bundle) -> None:
    original = development_bundle.phase_a_train[0]
    wrong_target = tuple(1 - bit for bit in original.target)
    corrupted_episode = replace(original, target=wrong_target)
    corrupted = replace(
        development_bundle,
        phase_a_train=(corrupted_episode, *development_bundle.phase_a_train[1:]),
    )
    result = validate_development_bundle(corrupted)
    assert result["status"] == "FAIL"
    assert any("target does not match left-to-right composition" in error for error in result["errors"])
