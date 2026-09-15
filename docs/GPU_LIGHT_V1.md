# Lightweight CUDA V1: prospective engineering and admission screen

This development version addresses host-memory overhead, not the research score.
The existing PyTorch backend, V3 protocol/guards/seeds and canonical readiness are
unchanged. No official admission or learning pass can be issued by this runner.

## Frozen before execution

- Load the installed NVIDIA driver and NVRTC DLL directly through ctypes. Do not
  import PyTorch, enable autograd, allocate pinned queues, spawn workers, or offload
  recurrence to host memory. Compile for the measured device capability with
  `--fmad=false`; use float64, no fast-math. One CUDA thread handles each episode.
- Keep the existing topology, threshold, emission, observation, TTL and head rule.
  Host arrays are bounded input/output transfers, CPU reference checks and summary
  statistics; this is not a claim of zero RAM use or an entirely GPU-only program.
- New experimental resource contract: at least 512 MiB available RAM before
  driver/compiler initialization, at least 128 MiB at each subsequent batch
  boundary, explicit device allocations at most 16 MiB, initial device headroom
  at least 256 MiB. These are conservative stop conditions, not an assurance about
  other applications. Record actual peak working set/private commit and explicit
  CUDA allocation; driver, compiler, JIT and context allocations are not included
  in the explicit device cap. Keep the original PyTorch 1 GiB startup guard intact.
- Before any GPU outputs: persist SHA-256 manifest of all package Python files,
  this protocol, pyproject, and CUDA compiler DLL plus a copy of this protocol.
  Check source identity after each seed and immediately before report publication.
- Engineering seeds 12000/12001: both signs and absent cue, delays 0/4/8/16/32/40;
  all tick activations within atol=rtol=1e-10, masks and discrete counts/predictions
  exact against the unchanged CPU reference. Compare single/batch execution.
  Test 32 sequential local-head updates per seed and invalid chronology.
  Test blank-tail 256 ticks (not distractors), final-window hidden silence, exact
  expiry, wake cue independence. Test maximum batch/delay and 128 repeated batches.
  Stream raw numerical evidence. Repeat the whole probe in a new process.
- Only after two matching engineering passes: admission development seeds
  13000..13004, 100 balanced cue pairs each in separate train/eval splits at D8.
  Use the existing PCG64 stream derivation and graph construction. Generate every
  CPU/native-shadow episode independently; compare every GPU output, feature and
  logical activity count, then recompute metrics and ridge from GPU features.
  Use unchanged numerical A01..A09 thresholds from V3, including exact tail assay.
  A10 in this diagnostic is the local numerical/invariant check **only**, not
  V3's complete source/provenance/phase/independent-audit attestation.
  Repeat admission in a new process; identical deterministic results are required.
- A numerical admission failure ends this version: no learning streams generated,
  no seed search, no threshold changes, no score increase. Admission success merely
  permits preparing a separately frozen learning/evidence stage. New learning seeds
  13005..13014 remain unopened in this stage. V3 seeds 90..104 remain unopened.
- All output is exclusive-create in `artifacts/gpu_light_v1_2026-09-05/NAME`.
  Archive raw arrays with project-compatible JSON SHA-256 sidecars
  (`algorithm`, `report_file`, `report_sha256`). Reports have a complete deterministic
  payload hash; only process/resource/time metadata is outside that payload.
  Independent regeneration/replay and the official score validator still need
  their own completed, audited evidence chain before readiness can change.

Implementation references: [NVIDIA Driver API](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html)
and [NVRTC](https://docs.nvidia.com/cuda/nvrtc/). These describe the module-loading
and runtime-compilation interfaces, not the validity of this project's results.

## Packaging correction before final release

The first development run wrote plain-text sidecars, which the project-wide
validator correctly rejected. Preserve those digests and the first-run source;
convert only the sidecar container to the existing project JSON binding schema.
The final runner also makes evaluation physically read-only for theta (the first
kernel stored numerically unchanged values). No numeric rule, sample, seed,
threshold or valid negative finding changes. Fresh final runs under a new source
freeze are an engineering regression reproduction, not seed selection or a new
chance to pass the failed scientific screen.
