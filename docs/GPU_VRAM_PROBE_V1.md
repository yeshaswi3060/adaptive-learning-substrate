# GPU-resident LWOH forward/head prototype — V1

This is a separate development backend, not a modification or activation of V3.
The CPU implementation, registered scientific thresholds and canonical status remain unchanged.
The backend ports fixed-weight recurrent forward execution, passive write-once/latest
observers, and the eight-coordinate local online readout. It does **not** port CCF
credit propagation, structural plasticity, or the official evidence archive.

## Prospective checks (fixed before GPU results)

- Graph seeds: 12000 and 12001; reserved V3/confirmatory seeds are not used.
- Float64 CUDA only; no autograd, attention, TF32, or CPU fallback.
- Same CPU-built topology and initial weights; one-tick edge delays, change-triggered
  emission, sticky inactive activations, forced QUERY output and TTL=32 semantics.
- Forward probe: both cue signs and an absent cue; deterministic distractor patterns;
  delays 0, 4, 8, 16, 32 and 40. Compare every tick's activations and emission mask,
  final native output, both observers, event and edge-touch counts.
- Continuous values: absolute and relative tolerance 1e-10. Discrete predictions,
  evaluation/emission counts and masks must agree exactly. A mismatch fails the run.
- Training probe: 32 alternating-sign synthetic examples with eight distractors;
  compare every pre-update prediction, score, local update and head parameters to
  the CPU reference. Evaluate both signs at delays 4, 8 and 16, and absent-cue controls.
  This is integration testing, not evidence of a general learning advantage.
- Repeat identical GPU inputs and the whole fresh-process probe. Deterministic
  payloads must agree; hardware, timing and process-memory fields are excluded.
- Check batched/single-example equivalence, expired memory, invalid input rejection,
  missing-CUDA rejection, invalid training chronology and unchanged frozen weights.
- Long-run allocation check: 128 repeated batches; tensor allocation must not grow
  by more than 1 MiB after warmup. Run-level batches contain at most 32 examples,
  sequences at most 256 distractors; input validation precedes GPU allocation.
- CUDA tensor-allocator limit: 256 MiB. Leave at least 256 MiB reported device
  headroom at startup. This limit does not include CUDA driver/context allocations.
- Windows startup available-RAM floor: 1 GiB, chosen after the initial runtime profile
  (about 1 GiB peak working set). The first exploratory probe used a 256 MiB startup
  check; its original freeze is retained. No scientific tolerance changed.
  Stop at a case boundary if available RAM falls below 128 MiB.
  Record host RAM before/after importing
  PyTorch and end-to-end peak process working set/private commit. This is a probe
  prerequisite, **not** a guarantee about competing applications or zero system RAM use.
- Stream per-case verification records to JSONL; do not retain a whole experiment's
  Python event history. Use one process, no DataLoader workers or pinned-memory queue.

Output goes only into a fresh diagnostic directory outside official experiment
namespaces. Do not infer a scientific pass or raise readiness from this probe.
Do not lower V3's resource check. A future GPU experiment needs its own frozen
execution/measurement contract and independent verification.

## Commands

Set `PYTHONPATH=src` and run:

```text
python -B -m adaptive_learning_substrate.gpu_probe --output artifacts/gpu_vram_v1_2026-09-05/primary_validated
python -B -m adaptive_learning_substrate.gpu_probe --output artifacts/gpu_vram_v1_2026-09-05/rerun_validated
```

PyTorch/CUDA is optional and loaded only in this backend. The existing NumPy-only
package and test workflow retain their existing dependency set.
