from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from adaptive_learning_substrate import experiment_000_online_trace_learning as online
from adaptive_learning_substrate.events import EdgeTrace, EventLog, UnitEvent


def _nested_event() -> UnitEvent:
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


def test_event_log_snapshots_cannot_rewrite_history(tmp_path: Path) -> None:
    log = EventLog()
    log.append(_nested_event())

    first = log.records
    first[0]["activation"] = 999.0
    first[0]["edge_traces"][0]["message_value"] = 999.0
    selected = log.by_type("UnitEvent")
    selected[0]["node"] = "tampered"

    fresh = log.records
    assert fresh[0]["activation"] == 0.25
    assert fresh[0]["edge_traces"][0]["message_value"] == 0.5
    assert fresh[0]["node"] == "hidden:0"

    path = log.write_jsonl(tmp_path / "events.jsonl")
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted == json.loads(json.dumps(fresh[0]))


def test_atomic_writer_never_overwrites_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "evidence.json"
    online._atomic_write_json(target, {"writer": "first"})

    try:
        online._atomic_write_json(target, {"writer": "second"})
    except FileExistsError:
        pass
    else:  # pragma: no cover - assertion spelling keeps the failure explicit.
        raise AssertionError("second immutable-evidence write unexpectedly succeeded")

    assert json.loads(target.read_text(encoding="utf-8")) == {"writer": "first"}


def test_atomic_writer_has_exactly_one_process_winner(tmp_path: Path) -> None:
    target = tmp_path / "contended.json"
    gate = tmp_path / "go"
    source_root = Path(__file__).resolve().parents[1] / "src"
    workers: list[subprocess.Popen[str]] = []
    worker_code = """
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, sys.argv[4])
from adaptive_learning_substrate import experiment_000_online_trace_learning as online

target = Path(sys.argv[1])
gate = Path(sys.argv[2])
ready = Path(sys.argv[3])
writer = ready.name
ready.write_text('ready', encoding='utf-8')
while not gate.exists():
    time.sleep(0.005)
try:
    online._atomic_write_json(target, {'writer': writer})
except FileExistsError:
    print('LOST', flush=True)
    raise SystemExit(3)
print('WON', flush=True)
"""
    for index in range(4):
        ready = tmp_path / f"ready-{index}"
        workers.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    worker_code,
                    os.fspath(target),
                    os.fspath(gate),
                    os.fspath(ready),
                    os.fspath(source_root),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )

    deadline = time.monotonic() + 30.0
    while not all((tmp_path / f"ready-{index}").exists() for index in range(4)):
        if time.monotonic() >= deadline:
            for worker in workers:
                worker.kill()
            raise AssertionError("contention workers did not reach the publication gate")
        time.sleep(0.01)
    gate.write_text("go", encoding="utf-8")

    results = [worker.communicate(timeout=30.0) for worker in workers]
    return_codes = [worker.returncode for worker in workers]
    assert return_codes.count(0) == 1, results
    assert return_codes.count(3) == 3, results
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["writer"] in {f"ready-{index}" for index in range(4)}
    assert not tuple(tmp_path.glob(".*.tmp"))
