from adaptive_learning_substrate.events import EdgeTrace, EventLog, UnitEvent


def _event_with_nested_trace() -> UnitEvent:
    return UnitEvent(
        event_id="event-1",
        episode_id="episode-1",
        node="hidden:0",
        step=3,
        preactivation=0.5,
        activation=0.25,
        forced_output=False,
        edge_traces=(
            EdgeTrace(
                edge_id="edge-1",
                parent_event_id="event-0",
                message_value=0.5,
                omission_effect=0.1,
                weight_secant=0.2,
                source_secant=0.3,
                created_step=2,
            ),
        ),
    )


def test_records_returns_independent_deep_copied_snapshots() -> None:
    log = EventLog()
    log.append(_event_with_nested_trace())

    first = log.records
    second = log.records
    assert first[0] is not second[0]
    assert first[0]["edge_traces"] is not second[0]["edge_traces"]

    first[0]["activation"] = 999.0
    first[0]["edge_traces"][0]["message_value"] = 999.0

    assert second[0]["activation"] == 0.25
    assert second[0]["edge_traces"][0]["message_value"] == 0.5
    assert log.records == second


def test_by_type_returns_deep_copies_not_backing_records() -> None:
    log = EventLog()
    log.append(_event_with_nested_trace())

    selected = log.by_type("UnitEvent")
    selected[0]["node"] = "tampered"
    selected[0]["edge_traces"][0]["source_secant"] = 999.0

    fresh_selected = log.by_type("UnitEvent")
    assert fresh_selected[0]["node"] == "hidden:0"
    assert fresh_selected[0]["edge_traces"][0]["source_secant"] == 0.3
    assert log.by_type("CreditEvent") == ()
