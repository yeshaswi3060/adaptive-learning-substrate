import json
import math
from pathlib import Path

import pytest

from adaptive_learning_substrate.events import EdgeTrace, UnitEvent
from adaptive_learning_substrate.experiment_000_alignment import (
    fixed_event_dag_output,
    fixed_event_dag_output_gradient,
    main,
    run_alignment_diagnostic,
)


def _trace(edge_id: str, parent_event_id: str, message: float, step: int) -> EdgeTrace:
    return EdgeTrace(
        edge_id=edge_id,
        parent_event_id=parent_event_id,
        message_value=message,
        omission_effect=0.0,
        weight_secant=0.0,
        source_secant=0.0,
        created_step=step,
    )


def test_fixed_event_gradient_accumulates_shared_weight_over_time() -> None:
    weight = 0.4
    source_value = 0.7
    first_activation = math.tanh(weight * source_value)
    output_activation = math.tanh(weight * first_activation)
    events = (
        UnitEvent("source", "episode", "input", 0, source_value, source_value, False, ()),
        UnitEvent(
            "first",
            "episode",
            "hidden",
            1,
            weight * source_value,
            first_activation,
            False,
            (_trace("shared", "source", source_value, 1),),
        ),
        UnitEvent(
            "output",
            "episode",
            "output",
            2,
            weight * first_activation,
            output_activation,
            True,
            (_trace("shared", "first", first_activation, 2),),
        ),
    )

    gradient = fixed_event_dag_output_gradient(
        events=events,
        weights={"shared": weight},
        edge_ids=("shared",),
        root_event_id="output",
    )["shared"]

    expected = (1.0 - output_activation**2) * (
        first_activation
        + weight * (1.0 - first_activation**2) * source_value
    )
    step = 1e-6
    plus = fixed_event_dag_output(
        events=events, weights={"shared": weight + step}, root_event_id="output"
    )
    minus = fixed_event_dag_output(
        events=events, weights={"shared": weight - step}, root_event_id="output"
    )
    numerical = (plus - minus) / (2.0 * step)
    assert gradient == pytest.approx(expected, rel=1e-13, abs=1e-13)
    assert gradient == pytest.approx(numerical, rel=1e-9, abs=1e-10)


def test_alignment_diagnostic_is_deterministic_and_validates_derivatives() -> None:
    left = run_alignment_diagnostic(
        seeds=(42,), episodes_per_seed=2, finite_difference_coordinates=32
    )
    right = run_alignment_diagnostic(
        seeds=(42,), episodes_per_seed=2, finite_difference_coordinates=32
    )

    assert left["deterministic_payload_sha256"] == right["deterministic_payload_sha256"]
    assert left["confirmatory_executed"] is False
    assert left["seeds"] == [42]
    assert left["integrity"] == {
        "all_custom_seeds": True,
        "confirmatory_seed_count": 0,
        "all_topologies_unchanged": True,
        "all_event_dags_immutable": True,
        "all_finite": True,
        "finite_difference_validation_pass": True,
    }
    seed = left["seed_results"][0]
    assert seed["seed_role"] == "custom"
    assert seed["graph"]["group_edge_counts"] == {
        "cue": 8,
        "noise": 8,
        "query": 8,
        "recurrent": 512,
        "output": 16,
    }
    assert seed["graph"]["topology_unchanged"] is True
    assert seed["graph"]["weights_changed_by_ccf"] is True
    for episode in seed["episodes"]:
        finite_difference = episode["finite_difference"]
        assert finite_difference["selected_coordinates"] == 32
        assert finite_difference["fixed_dag_within_tolerance"] == 32
        assert finite_difference["stable_coordinates"] >= 26
        assert finite_difference["dynamic_stable_within_tolerance"] == finite_difference[
            "stable_coordinates"
        ]
        assert finite_difference["fixed_dag_max_absolute_error"] < 1e-7
        assert finite_difference["validation_pass"] is True
        assert episode["event_dag_unchanged_after_credit"] is True
        assert episode["replay"]["equal_norm_achieved"] is True
        global_alignment = episode["alignment"]["global"]
        assert global_alignment["actual_update_norm"] > 0.0
        assert global_alignment["exact_direction_norm"] > 0.0
        assert -1.0 <= global_alignment["cosine"] <= 1.0
        assert global_alignment["dot_product"] > 0.0
    assert left["aggregate"]["finite_difference"]["all_fixed_within_tolerance"] is True
    assert left["aggregate"]["finite_difference"]["validation_pass"] is True
    assert left["runtime"]["seconds"] > 0.0


def test_alignment_cli_writes_deterministic_json(tmp_path: Path) -> None:
    target = tmp_path / "alignment.json"

    status = main(
        [
            "--seeds",
            "42",
            "--episodes-per-seed",
            "2",
            "--finite-difference-coordinates",
            "32",
            "--output",
            str(target),
        ]
    )

    assert status == 0
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["schema_version"] == "experiment-000-fixed-event-alignment-v1"
    assert report["seeds"] == [42]
    assert report["episodes_per_seed"] == 2
    assert len(report["deterministic_payload_sha256"]) == 64


@pytest.mark.parametrize(
    ("seed", "message"),
    ((0, "reserved for development"), (1000, "reserved for confirmatory")),
)
def test_alignment_rejects_reserved_seed_before_writing(
    tmp_path: Path, seed: int, message: str
) -> None:
    target = tmp_path / f"must-not-exist-{seed}.json"

    with pytest.raises(ValueError, match=message):
        run_alignment_diagnostic(
            seeds=(seed,),
            episodes_per_seed=2,
            finite_difference_coordinates=32,
            output_path=target,
        )

    assert not target.exists()


def test_alignment_rejects_unregistered_custom_seed() -> None:
    with pytest.raises(ValueError, match="not one of the frozen alignment seeds"):
        run_alignment_diagnostic(seeds=(47,), episodes_per_seed=2)
