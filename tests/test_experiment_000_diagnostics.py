from pathlib import Path

import pytest

from adaptive_learning_substrate.experiment_000_diagnostics import (
    WRONG_EDGE_ID,
    build_delay_chain_positive_control,
    run_delay_chain_positive_control,
)


def test_positive_control_has_only_one_earlier_plastic_edge() -> None:
    graph = build_delay_chain_positive_control(seed=0, mode="full")
    plastic = [edge.edge_id for edge in graph.edges if edge.plastic]
    assert plastic == [WRONG_EDGE_ID]
    assert graph.weights[WRONG_EDGE_ID] == -0.9
    assert graph.edges_by_id["chain:h09->output"].plastic is False


def test_full_credit_passes_delayed_chain_while_no_trace_cannot(tmp_path: Path) -> None:
    output = tmp_path / "positive-control.json"
    report = run_delay_chain_positive_control(output_path=output)

    assert report["pass"] is True
    assert report["confirmatory_executed"] is False
    assert report["full"]["eval_accuracy"] == 1.0
    assert report["full"]["wrong_edge_trained_weight"] > 0.0
    assert report["no_trace"]["eval_accuracy"] == 0.0
    assert report["no_trace"]["wrong_edge_trained_weight"] == -0.9
    assert report["full"]["topology_unchanged"] is True
    assert report["full"]["evaluation_weights_unchanged"] is True
    assert output.is_file()


def test_positive_control_rejects_confirmatory_seed() -> None:
    with pytest.raises(ValueError, match="seed 1000 is confirmatory"):
        run_delay_chain_positive_control(seed=1000, episode_count=2)
