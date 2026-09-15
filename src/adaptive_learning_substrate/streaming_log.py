"""Constant-memory, deterministic gzip audit logging.

``StreamingEventLog`` implements the small interface used by
``RecurrentEventGraph`` while deliberately making historical records
unavailable through ``records``.  Each event is converted to the same canonical
JSON object as :class:`adaptive_learning_substrate.events.EventLog` and is
written immediately as one UTF-8 JSONL record.

The gzip header is normalised (no source filename and an mtime of zero), so in
the same Python/zlib environment the same event sequence, compression level,
and flush schedule produce identical compressed bytes even when written to
differently named files.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any, BinaryIO, Self

from .events import AuditEvent


class StreamingEventLog:
    """Stage audit events in deterministic ``.jsonl.gz`` storage.

    Enabled logs are written to an exclusive same-directory temporary file.
    :meth:`close` publishes the complete gzip stream atomically and refuses to
    replace an existing artifact unless ``replace_existing=True`` was selected
    explicitly.  A context-manager exception aborts publication and removes the
    temporary file.

    ``compressed_sha256`` becomes available only after successful publication,
    when the gzip trailer has been written.  The uncompressed hash is updated
    incrementally and never requires retaining a record in memory.
    """

    __slots__ = (
        "_closed",
        "_compressed_sha256",
        "_failed",
        "_gzip_handle",
        "_path",
        "_published",
        "_raw_handle",
        "_record_count",
        "_replace_existing",
        "_temporary_path",
        "_uncompressed_hasher",
        "_uncompressed_size_bytes",
        "enabled",
    )

    def __init__(
        self,
        path: str | Path,
        enabled: bool = True,
        *,
        compresslevel: int = 6,
        replace_existing: bool = False,
    ) -> None:
        if not isinstance(compresslevel, int) or isinstance(compresslevel, bool):
            raise TypeError("compresslevel must be an integer")
        if not 0 <= compresslevel <= 9:
            raise ValueError("compresslevel must be in [0, 9]")
        if not isinstance(replace_existing, bool):
            raise TypeError("replace_existing must be a boolean")

        self.enabled = bool(enabled)
        requested_path = Path(os.path.abspath(os.fspath(path)))
        # Resolve the parent once without following the final name.  This pins a
        # symlinked parent for the transaction while a dangling incumbent at the
        # requested final name remains occupied for no-replace publication.
        self._path = (
            requested_path.parent.resolve() / requested_path.name
            if self.enabled
            else requested_path
        )
        self._temporary_path: Path | None = None
        self._raw_handle: BinaryIO | None = None
        self._gzip_handle: gzip.GzipFile | None = None
        self._record_count = 0
        self._uncompressed_hasher = hashlib.sha256()
        self._uncompressed_size_bytes = 0
        self._compressed_sha256: str | None = None
        self._replace_existing = replace_existing
        self._closed = False
        self._failed = False
        self._published = False

        if self.enabled:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".stream-log-",
                suffix=".tmp",
                dir=self._path.parent,
            )
            temporary_path = Path(temporary_name)
            raw_handle: BinaryIO | None = None
            try:
                raw_handle = os.fdopen(descriptor, "wb")
                gzip_handle = gzip.GzipFile(
                    filename="",
                    mode="wb",
                    compresslevel=compresslevel,
                    fileobj=raw_handle,
                    mtime=0,
                )
            except BaseException as error:
                try:
                    if raw_handle is None:
                        os.close(descriptor)
                    else:
                        raw_handle.close()
                except OSError as cleanup_error:
                    error.add_note(
                        f"temporary descriptor cleanup failed: {cleanup_error}"
                    )
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    error.add_note(f"temporary path cleanup failed: {cleanup_error}")
                raise
            assert raw_handle is not None
            self._temporary_path = temporary_path
            self._raw_handle = raw_handle
            self._gzip_handle = gzip_handle

    @property
    def compressed_path(self) -> Path:
        """Path of the gzip artifact (final after :meth:`close`)."""

        return self._path

    @property
    def path(self) -> Path:
        """Alias for :attr:`compressed_path`."""

        return self._path

    @property
    def record_count(self) -> int:
        """Number of records successfully written."""

        return self._record_count

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        """Return an empty tuple because records are never retained in memory."""

        return ()

    @property
    def uncompressed_sha256(self) -> str:
        """SHA-256 of the canonical JSONL bytes written so far."""

        return self._uncompressed_hasher.hexdigest()

    @property
    def compressed_sha256(self) -> str | None:
        """SHA-256 of the finished gzip file, or ``None`` before close."""

        return self._compressed_sha256

    @property
    def uncompressed_size_bytes(self) -> int:
        """Number of canonical JSONL bytes submitted to gzip."""

        return self._uncompressed_size_bytes

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def published(self) -> bool:
        """Whether a complete staged stream reached the final path."""

        return self._published

    def append(self, event: AuditEvent) -> None:
        """Serialise and immediately write one audit event."""

        if not self.enabled:
            return
        if self._closed:
            raise RuntimeError("streaming event log is closed")
        if self._failed:
            raise RuntimeError(
                "streaming event log is poisoned by an earlier I/O failure"
            )

        record = asdict(event)
        record["event_type"] = type(event).__name__
        line = (
            json.dumps(
                record, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
            + b"\n"
        )
        assert self._gzip_handle is not None
        try:
            written = self._gzip_handle.write(line)
            if written != len(line):
                raise OSError(
                    f"incomplete gzip write: accepted {written} of {len(line)} bytes"
                )
        except BaseException:
            self._failed = True
            raise
        self._uncompressed_hasher.update(line)
        self._uncompressed_size_bytes += len(line)
        self._record_count += 1

    def extend(self, events: Iterable[AuditEvent]) -> None:
        """Append every event from ``events`` in iteration order."""

        for event in events:
            self.append(event)

    def by_type(self, event_type: str) -> tuple[dict[str, Any], ...]:
        """Return no records; streaming mode intentionally has no query cache."""

        del event_type
        return ()

    def flush(self) -> None:
        """Flush staged bytes without publishing or finalising the gzip stream."""

        if not self.enabled:
            return
        if self._closed:
            raise RuntimeError("streaming event log is closed")
        if self._failed:
            raise RuntimeError(
                "streaming event log is poisoned by an earlier I/O failure"
            )
        assert self._gzip_handle is not None
        assert self._raw_handle is not None
        try:
            self._gzip_handle.flush()
            self._raw_handle.flush()
        except BaseException:
            self._failed = True
            raise

    def close(self) -> None:
        """Finish and atomically publish the gzip artifact.

        The default no-replace branch uses atomic rename on Windows and atomic
        hard-link creation on POSIX.  Both paths reside in the same directory,
        so a concurrent publisher either wins completely or receives
        :class:`FileExistsError`; it can never truncate or partially replace
        the winner.
        """

        if self._closed:
            return
        if self._failed:
            try:
                self.abort()
            finally:
                raise RuntimeError(
                    "streaming event log cannot publish after an earlier I/O failure"
                )
        self._closed = True
        if not self.enabled:
            return

        gzip_handle = self._gzip_handle
        raw_handle = self._raw_handle
        temporary_path = self._temporary_path
        self._gzip_handle = None
        self._raw_handle = None
        self._temporary_path = None
        try:
            self._finish_handles(gzip_handle, raw_handle, durable=True)

            assert temporary_path is not None
            compressed_hasher = hashlib.sha256()
            with temporary_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    compressed_hasher.update(chunk)

            if self._replace_existing:
                os.replace(temporary_path, self._path)
                temporary_path = None
            elif os.name == "nt":
                # Windows rename is atomic and rejects an existing destination.
                os.rename(temporary_path, self._path)
                temporary_path = None
            else:
                # POSIX rename replaces an incumbent, whereas link is an atomic
                # create-if-absent operation that also rejects dangling links.
                os.link(temporary_path, self._path)

            self._compressed_sha256 = compressed_hasher.hexdigest()
            self._published = True
        except BaseException as error:
            self._failed = True
            cleanup_error = self._discard_temporary(temporary_path)
            if cleanup_error is not None:
                error.add_note(f"temporary path cleanup failed: {cleanup_error}")
            raise
        else:
            cleanup_error = self._discard_temporary(temporary_path)
            if cleanup_error is not None:
                raise cleanup_error

    def abort(self) -> None:
        """Close the staged stream and discard it without publication."""

        if self._closed:
            return
        self._closed = True
        if not self.enabled:
            return

        gzip_handle = self._gzip_handle
        raw_handle = self._raw_handle
        temporary_path = self._temporary_path
        self._gzip_handle = None
        self._raw_handle = None
        self._temporary_path = None
        primary_error: BaseException | None = None
        try:
            self._finish_handles(gzip_handle, raw_handle, durable=False)
        except BaseException as error:  # noqa: BLE001 - preserve cleanup ordering
            primary_error = error
        cleanup_error = self._discard_temporary(temporary_path)
        if cleanup_error is not None:
            if primary_error is None:
                raise cleanup_error
            primary_error.add_note(f"temporary path cleanup failed: {cleanup_error}")
        if primary_error is not None:
            raise primary_error

    @staticmethod
    def _discard_temporary(temporary_path: Path | None) -> OSError | None:
        """Remove a staged path and return, rather than hide, a cleanup error."""

        if temporary_path is None:
            return None
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError as error:
            return error
        return None

    @staticmethod
    def _finish_handles(
        gzip_handle: gzip.GzipFile | None,
        raw_handle: BinaryIO | None,
        *,
        durable: bool,
    ) -> None:
        """Close both layers while retaining the first finalisation error."""

        primary_error: BaseException | None = None
        try:
            if gzip_handle is not None:
                gzip_handle.close()
        except BaseException as error:  # noqa: BLE001 - preserve first close failure
            primary_error = error

        raw_error: BaseException | None = None
        if raw_handle is not None:
            try:
                if durable and primary_error is None:
                    raw_handle.flush()
                    os.fsync(raw_handle.fileno())
            except BaseException as error:  # noqa: BLE001 - close still required
                raw_error = error
            finally:
                try:
                    raw_handle.close()
                except BaseException as error:  # noqa: BLE001 - retain first failure
                    if raw_error is None:
                        raw_error = error
        if primary_error is None:
            primary_error = raw_error

        if primary_error is not None:
            raise primary_error

    def __enter__(self) -> Self:
        if self._closed:
            raise RuntimeError("streaming event log is closed")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        del traceback
        if exc_type is None:
            self.close()
        else:
            try:
                self.abort()
            except BaseException as cleanup_error:  # noqa: BLE001
                # Preserve the exception from the protected body.  The final
                # path was never published, so a cleanup error cannot turn a
                # partial stream into accepted evidence.
                if isinstance(exc_value, BaseException):
                    exc_value.add_note(f"streaming-log cleanup failed: {cleanup_error}")
        return False


# The explicit gzip name is useful at call sites that may support other streaming
# encodings later, while ``StreamingEventLog`` remains the concise public name.
StreamingGzipEventLog = StreamingEventLog


__all__ = ["StreamingEventLog", "StreamingGzipEventLog"]
