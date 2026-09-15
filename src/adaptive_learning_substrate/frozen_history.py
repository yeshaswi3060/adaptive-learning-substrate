"""Packaged byte snapshots required to reproduce superseded frozen protocols."""

from __future__ import annotations

import hashlib
import re
from importlib.resources import files
from typing import Any

SHA256_RE = re.compile(r"[0-9a-f]{64}")
FROZEN_RECURRENT_SOURCE_SHA256 = (
    "b56216abd19f5d3aaf935b6e58831c54ccf078ce4e4c7ef80e006b04ca6231b3"
)
FROZEN_RECURRENT_SOURCE_RESOURCE = (
    "_frozen_sources/recurrent_b56216.snapshot"
)


def validate_frozen_recurrent_snapshot() -> dict[str, Any]:
    """Return the immutable legacy recurrent-source identity or fail closed."""

    resource = files("adaptive_learning_substrate").joinpath(
        FROZEN_RECURRENT_SOURCE_RESOURCE
    )
    payload = resource.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if (
        SHA256_RE.fullmatch(FROZEN_RECURRENT_SOURCE_SHA256) is None
        or observed != FROZEN_RECURRENT_SOURCE_SHA256
    ):
        raise RuntimeError(
            "packaged frozen recurrent-source snapshot differs from its registered hash"
        )
    return {
        "resource": FROZEN_RECURRENT_SOURCE_RESOURCE,
        "bytes": len(payload),
        "sha256": observed,
    }
