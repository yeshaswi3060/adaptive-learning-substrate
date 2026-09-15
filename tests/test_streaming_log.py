import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from adaptive_learning_substrate.events import (
    CreditEvent,
    CreditPacket,
    EdgeTrace,
    UnitEvent,
    UpdateEvent,
)
from adaptive_learning_substrate.recurrent import build_experiment_000_graph
from adaptive_learning_substrate.streaming_log import (
    StreamingEventLog,
    StreamingGzipEventLog,
)


def _canonical_jsonl(events: tuple[object, ...]) -> bytes:
    lines: list[bytes] = []
    for event in events:
        record = asdict(event)  # type: ignore[arg-type]
        record["event_type"] = type(event).__name__
        lines.append(
            json.dumps(
                record, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
            + b"\n"
        )
    return b"".join(lines)


def _sample_events() -> tuple[object, ...]:
    trace = EdgeTrace(
        edge_id="cue-to-hidden",
        parent_event_id="episode-7:cue:0:0",
        message_value=-0.25,
        omission_effect=-0.2,
        weight_secant=0.8,
        source_secant=0.5,
        created_step=1,
    )
    return (
        UnitEvent(
            event_id="episode-7:hidden:1:0",
            episode_id="episode-7",
            node="hidden",
            step=1,
            preactivation=-0.25,
            activation=-0.24491866240370913,
            forced_output=False,
            edge_traces=(trace,),
        ),
        CreditEvent("episode-7", 3, "output", 0.75, "outcome"),
        CreditPacket("episode-7:hidden:1:0", 0.4, 3, 1),
        UpdateEvent(
            episode_id="episode-7",
            step=3,
            edge_id="hidden-to-output",
            destination_credit=0.75,
            trace_kind="omission",
            trace_value=-0.2,
            delta=-0.015,
            old_weight=0.5,
            new_weight=0.485,
        ),
    )


def test_jsonl_and_gzip_bytes_are_deterministic_and_hash_verified(tmp_path) -> None:
    events = _sample_events()
    first_path = tmp_path / "first-name.jsonl.gz"
    second_path = tmp_path / "different-name.jsonl.gz"

    with StreamingEventLog(first_path) as first:
        first.extend(events)  # type: ignore[arg-type]
        assert first.compressed_sha256 is None
        assert first.record_count == len(events)
        assert first.records == ()
        assert not first_path.exists()

    with StreamingGzipEventLog(second_path) as second:
        second.extend(events)  # type: ignore[arg-type]

    expected = _canonical_jsonl(events)
    first_compressed = first_path.read_bytes()
    second_compressed = second_path.read_bytes()

    assert gzip.decompress(first_compressed) == expected
    assert gzip.decompress(second_compressed) == expected
    assert first_compressed == second_compressed
    assert first.uncompressed_sha256 == hashlib.sha256(expected).hexdigest()
    assert first.uncompressed_size_bytes == len(expected)
    assert first.compressed_sha256 == hashlib.sha256(first_compressed).hexdigest()
    assert second.compressed_sha256 == hashlib.sha256(second_compressed).hexdigest()
    assert first.compressed_path == first_path
    assert first.path == first_path
    assert first.closed is True
    assert first.published is True


def test_recurrent_graph_uses_streaming_log_without_in_memory_audit_records(
    tmp_path,
) -> None:
    output = tmp_path / "graph-events.jsonl.gz"
    with StreamingEventLog(output) as event_log:
        graph = build_experiment_000_graph(11)
        graph.event_log = event_log
        graph.begin_episode("streamed")
        graph.step({"cue": 1.0})
        for bit in (0, 1, 1, 0, 1, 0, 0, 1):
            graph.step({"noise": 1.0 if bit else -1.0})
        query = graph.query()
        graph.apply_supervised_credit(1 - query.prediction)

        assert graph.audit_records == ()
        assert event_log.records == ()
        assert event_log.record_count > 0
        assert not hasattr(event_log, "__dict__")
        assert not hasattr(event_log, "_records")

    records = [
        json.loads(line) for line in gzip.decompress(output.read_bytes()).splitlines()
    ]
    assert len(records) == event_log.record_count
    assert {record["event_type"] for record in records} >= {
        "UnitEvent",
        "CreditEvent",
        "CreditPacket",
        "UpdateEvent",
    }


def test_disabled_and_closed_lifecycle_is_explicit(tmp_path) -> None:
    disabled_path = tmp_path / "disabled.jsonl.gz"
    disabled = StreamingEventLog(disabled_path, enabled=False)
    disabled.extend(_sample_events())  # type: ignore[arg-type]
    disabled.close()
    disabled.close()
    assert disabled.record_count == 0
    assert disabled.records == ()
    assert disabled.uncompressed_sha256 == hashlib.sha256(b"").hexdigest()
    assert disabled.compressed_sha256 is None
    assert disabled.published is False
    assert not disabled_path.exists()

    enabled = StreamingEventLog(tmp_path / "closed.jsonl.gz")
    enabled.close()
    enabled.close()
    assert enabled.compressed_sha256 is not None
    with pytest.raises(RuntimeError, match="closed"):
        enabled.append(_sample_events()[0])  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="closed"):
        enabled.flush()
    with pytest.raises(RuntimeError, match="closed"):
        enabled.__enter__()


def test_existing_artifact_is_not_overwritten_by_default(tmp_path) -> None:
    output = tmp_path / "immutable.jsonl.gz"
    with StreamingEventLog(output) as first:
        first.append(_sample_events()[0])  # type: ignore[arg-type]
    original_bytes = output.read_bytes()
    original_hash = hashlib.sha256(original_bytes).hexdigest()

    contender = StreamingEventLog(output)
    contender.append(_sample_events()[1])  # type: ignore[arg-type]
    with pytest.raises(FileExistsError):
        contender.close()

    assert contender.closed is True
    assert contender.published is False
    assert contender.compressed_sha256 is None
    assert output.read_bytes() == original_bytes
    assert hashlib.sha256(output.read_bytes()).hexdigest() == original_hash
    assert list(tmp_path.glob(".stream-log-*.tmp")) == []


def test_close_race_has_one_complete_winner_and_cleans_loser(tmp_path) -> None:
    output = tmp_path / "contended.jsonl.gz"
    first = StreamingEventLog(output)
    second = StreamingEventLog(output)
    first.append(_sample_events()[0])  # type: ignore[arg-type]
    second.append(_sample_events()[1])  # type: ignore[arg-type]

    first.close()
    winner_bytes = output.read_bytes()
    assert gzip.decompress(winner_bytes) == _canonical_jsonl((_sample_events()[0],))

    with pytest.raises(FileExistsError):
        second.close()

    assert second.closed is True
    assert second.published is False
    assert second.compressed_sha256 is None
    assert output.read_bytes() == winner_bytes
    assert list(tmp_path.glob(".stream-log-*.tmp")) == []


def test_context_exception_aborts_publication_and_cleans_temporary_file(
    tmp_path,
) -> None:
    output = tmp_path / "failed.jsonl.gz"

    with (
        pytest.raises(RuntimeError, match="injected failure"),
        StreamingEventLog(output) as event_log,
    ):
        event_log.append(_sample_events()[0])  # type: ignore[arg-type]
        event_log.flush()
        assert not output.exists()
        raise RuntimeError("injected failure")

    assert event_log.closed is True
    assert event_log.published is False
    assert event_log.compressed_sha256 is None
    assert not output.exists()
    assert list(tmp_path.glob(".stream-log-*.tmp")) == []


def test_explicit_replace_preserves_atomic_complete_stream_behavior(tmp_path) -> None:
    output = tmp_path / "replaceable.jsonl.gz"
    with StreamingEventLog(output) as first:
        first.append(_sample_events()[0])  # type: ignore[arg-type]
    original_bytes = output.read_bytes()

    with StreamingEventLog(output, replace_existing=True) as replacement:
        replacement.append(_sample_events()[1])  # type: ignore[arg-type]
        replacement.flush()
        assert output.read_bytes() == original_bytes

    replacement_bytes = output.read_bytes()
    assert replacement_bytes != original_bytes
    assert gzip.decompress(replacement_bytes) == _canonical_jsonl(
        (_sample_events()[1],)
    )
    assert (
        replacement.compressed_sha256 == hashlib.sha256(replacement_bytes).hexdigest()
    )
    assert replacement.published is True
    assert list(tmp_path.glob(".stream-log-*.tmp")) == []


def test_zero_event_log_publishes_a_complete_empty_gzip(tmp_path) -> None:
    output = tmp_path / "empty.jsonl.gz"
    with StreamingEventLog(output) as event_log:
        assert not output.exists()

    compressed = output.read_bytes()
    assert gzip.decompress(compressed) == b""
    assert event_log.record_count == 0
    assert event_log.published is True
    assert event_log.compressed_sha256 == hashlib.sha256(compressed).hexdigest()


def test_write_failure_poison_prevents_later_publication(tmp_path, monkeypatch) -> None:
    output = tmp_path / "poisoned.jsonl.gz"
    event_log = StreamingEventLog(output)

    def fail_write(self, data):
        del self, data
        raise OSError("injected gzip write failure")

    monkeypatch.setattr(gzip.GzipFile, "write", fail_write)
    with pytest.raises(OSError, match="injected gzip write failure"):
        event_log.append(_sample_events()[0])  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="earlier I/O failure"):
        event_log.close()

    assert event_log.closed is True
    assert event_log.published is False
    assert event_log.compressed_sha256 is None
    assert not output.exists()
    assert list(tmp_path.glob(".stream-log-*.tmp")) == []


def test_constructor_failure_releases_temporary_artifact(tmp_path, monkeypatch) -> None:
    output = tmp_path / "constructor-failure.jsonl.gz"

    def fail_constructor(*args, **kwargs):
        del args, kwargs
        raise OSError("injected gzip constructor failure")

    monkeypatch.setattr(gzip, "GzipFile", fail_constructor)
    with pytest.raises(OSError, match="injected gzip constructor failure"):
        StreamingEventLog(output)

    assert not output.exists()
    assert list(tmp_path.glob(".stream-log-*.tmp")) == []


def test_posix_link_publication_branch_is_complete_and_no_replace(
    tmp_path, monkeypatch
) -> None:
    import adaptive_learning_substrate.streaming_log as streaming_module

    output = tmp_path / "posix-branch.jsonl.gz"
    first = StreamingEventLog(output)
    contender = StreamingEventLog(output)
    first.append(_sample_events()[0])  # type: ignore[arg-type]
    contender.append(_sample_events()[1])  # type: ignore[arg-type]

    monkeypatch.setattr(streaming_module.os, "name", "posix")
    first.close()
    first_bytes = output.read_bytes()
    assert first.published is True
    assert gzip.decompress(first_bytes) == _canonical_jsonl((_sample_events()[0],))

    with pytest.raises(FileExistsError):
        contender.close()

    assert output.read_bytes() == first_bytes
    assert contender.published is False
    assert list(tmp_path.glob(".stream-log-*.tmp")) == []


def test_two_process_publish_contention_has_exactly_one_complete_winner(
    tmp_path,
) -> None:
    output = tmp_path / "process-contended.jsonl.gz"
    ready_paths = [tmp_path / f"ready-{index}" for index in range(2)]
    release_path = tmp_path / "release"
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (str(project_root / "src"), environment.get("PYTHONPATH", "")),
        )
    )
    worker_code = r"""
from pathlib import Path
import sys
import time

from adaptive_learning_substrate.events import CreditEvent
from adaptive_learning_substrate.streaming_log import StreamingEventLog

target = Path(sys.argv[1])
ready = Path(sys.argv[2])
release = Path(sys.argv[3])
worker_id = sys.argv[4]
event_log = StreamingEventLog(target)
event_log.append(CreditEvent(worker_id, 1, "node", 1.0, "process-race"))
ready.write_text("ready", encoding="utf-8")
deadline = time.monotonic() + 30.0
while not release.exists():
    if time.monotonic() >= deadline:
        event_log.abort()
        print("RELEASE_TIMEOUT")
        raise SystemExit(19)
    time.sleep(0.01)
try:
    event_log.close()
except FileExistsError:
    print(f"COLLISION:{worker_id}")
    raise SystemExit(17)
print(f"PUBLISHED:{worker_id}")
"""
    workers = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                worker_code,
                str(output),
                str(ready_paths[index]),
                str(release_path),
                f"worker-{index}",
            ],
            cwd=project_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index in range(2)
    ]
    try:
        deadline = time.monotonic() + 30.0
        while not all(path.exists() for path in ready_paths):
            failed = [worker for worker in workers if worker.poll() is not None]
            assert not failed, [worker.communicate() for worker in failed]
            assert time.monotonic() < deadline, "workers did not reach publish barrier"
            time.sleep(0.01)
        release_path.write_text("release", encoding="utf-8")
        results = [worker.communicate(timeout=30.0) for worker in workers]
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait(timeout=10.0)

    return_codes = [worker.returncode for worker in workers]
    assert sorted(return_codes) == [0, 17], results
    assert sum("PUBLISHED:" in stdout for stdout, _ in results) == 1
    assert sum("COLLISION:" in stdout for stdout, _ in results) == 1
    records = [
        json.loads(line) for line in gzip.decompress(output.read_bytes()).splitlines()
    ]
    assert len(records) == 1
    assert records[0]["episode_id"] in {"worker-0", "worker-1"}
    assert records[0]["source"] == "process-race"
    assert list(tmp_path.glob(".stream-log-*.tmp")) == []
