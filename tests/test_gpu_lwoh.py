"""CPU-only contract checks for the optional GPU prototype (no CUDA import)."""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from adaptive_learning_substrate import gpu_lwoh
from adaptive_learning_substrate.experiment_000_lwoh_l1_execution_v3 import (
    LWOHObserverGraph,
)
from adaptive_learning_substrate.gpu_probe import run_probe


@pytest.mark.parametrize("cues,noise,delay", [
    ([-1, 1, None], [[], [], []], 0),
    ([1], [[-1, 1] * 128], 256),
    ([1] * 32, [[1]] * 32, 1),
])
def test_gpu_batch_contract_accepts_supported_boundaries(cues, noise, delay):
    assert gpu_lwoh.validate_batch(cues, noise) == delay


@pytest.mark.parametrize("cues,noise", [
    ([], []), ([1], []), ([1] * 33, [[]] * 33), ([1], [[1] * 257]),
    ([True], [[]]), ([0], [[]]), ([1.0], [[]]), ([1], [[False]]),
    ([1], [[0]]), ([1], [[float("nan")]]), ([1, -1], [[], [1]]),
])
def test_gpu_batch_contract_rejects_invalid_input(cues, noise):
    with pytest.raises(ValueError):
        gpu_lwoh.validate_batch(cues, noise)


def test_exported_gpu_topology_preserves_source_weights_and_sorted_incoming_edges():
    graph = LWOHObserverGraph.from_seed(12000)
    before = graph.topology_hash(), graph.weights_hash()
    spec = gpu_lwoh.Topology.from_reference(graph)
    assert (spec.topology_sha256, spec.weights_sha256) == before
    assert len(spec.sources) == 65
    assert len(spec.selected) == 8
    for row, node in enumerate(spec.targets):
        edges = sorted((edge for edge in graph.edges if edge.destination == node), key=lambda e: e.edge_id)
        actual = [(spec.nodes[source], weight) for source, weight in zip(spec.sources[row], spec.weights[row], strict=True)
                  if source < len(spec.nodes)]
        assert actual == [(edge.source, graph.weights[edge.edge_id]) for edge in edges]
    assert (graph.topology_hash(), graph.weights_hash()) == before


def test_gpu_module_import_does_not_load_torch_or_initialize_cuda():
    result = subprocess.run([
        sys.executable, "-B", "-c",
        ("import sys; import adaptive_learning_substrate.gpu_lwoh; "
        "import adaptive_learning_substrate.gpu_probe; "
        "assert 'torch' not in sys.modules; print('LAZY_CUDA_IMPORT_PASS')"),
    ], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "LAZY_CUDA_IMPORT_PASS"


def test_missing_cuda_is_rejected_without_fallback(monkeypatch):
    class Cuda:
        @staticmethod
        def is_available():
            return False

    class Torch:
        cuda = Cuda()

    original = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", lambda name: Torch() if name == "torch" else original(name))
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        gpu_lwoh.cuda_runtime()


def test_gpu_probe_rejects_canonical_namespace_before_loading_runtime():
    root = Path(gpu_lwoh.__file__).resolve().parents[2]
    with pytest.raises(ValueError, match="diagnostic directory"):
        run_probe(root / "artifacts/experiment_000/not_a_gpu_probe")


def test_gpu_probe_rejects_existing_directory_before_loading_runtime():
    root = Path(gpu_lwoh.__file__).resolve().parents[2]
    # The path is checked before its contents; mock existence, never create evidence.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "exists", lambda _path: True)
        with pytest.raises(FileExistsError, match="never overwritten"):
            run_probe(root / "artifacts/gpu_synthetic/existing")
