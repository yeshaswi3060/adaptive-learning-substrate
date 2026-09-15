import pytest

from adaptive_learning_substrate.environments import (
    DelayedChainEnvironment,
    MicroProgram3BitEnvironment,
)


def test_three_bit_operation_semantics_and_composition() -> None:
    env = MicroProgram3BitEnvironment(seed=7)
    assert env.apply((1, 0, 0), "FLIP0") == (1, 0, 1)
    assert env.apply((1, 0, 1), "ROTL") == (0, 1, 1)
    assert env.apply((1, 0, 1), "NOT") == (0, 1, 0)
    assert env.compose((1, 0, 0), ("FLIP0", "ROTL")) == (0, 1, 1)
    assert len(env.all_inputs()) == 8


def test_delayed_chain_withholds_terminal_credit_for_exact_delay() -> None:
    env = DelayedChainEnvironment(length=2, credit_delay=3, correct_actions=(0, 1), seed=5)
    assert env.reset() == {"state:0": 1.0}
    transition = env.step(0)
    assert not transition.terminated
    terminal = env.step(1)
    assert terminal.terminated and terminal.reward is None
    assert [env.tick().reward for _ in range(2)] == [None, None]
    outcome = env.tick()
    assert outcome.credit_ready and outcome.reward == 1.0
    with pytest.raises(RuntimeError, match="no delayed outcome"):
        env.tick()


def test_delayed_chain_reports_failure_after_delay() -> None:
    env = DelayedChainEnvironment(length=3, credit_delay=1, correct_actions=(0, 1, 0))
    env.reset()
    assert env.step(1).reward is None
    assert env.tick().reward == -1.0

