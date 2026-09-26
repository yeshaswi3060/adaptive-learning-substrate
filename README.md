# Adaptive Learning Substrate

**New to the project?** Start with the illustrated
[project guide](docs/PROJECT_GUIDE.md) for the repository map, the research
question, what has failed, and the next experiment.

## Mission

Develop and experimentally validate a fundamentally new learning mechanism that can learn continuously from experience, create or reorganize its own computational structure, preserve earlier knowledge, and operate efficiently without Transformer attention or global end-to-end backpropagation.

This project is not intended to repackage an existing model or build another application around current AI. Its purpose is to investigate a new computational foundation for adaptable intelligence.

## What we aim to achieve

We aim to create a small working research system that:

1. Learns new reusable skills while it is operating.
2. Retains previously learned skills without complete retraining.
3. Assigns credit using a new local learning rule rather than global backpropagation.
4. Grows, removes, or reorganizes computational pathways as experience changes.
5. Uses sparse, event-driven computation instead of dense Transformer attention.
6. Combines learned procedures to solve unfamiliar problems.
7. Runs on accessible hardware before any attempt to scale it.

## Core research question

> Can a dynamically structured network learn useful global behavior through local causal credit signals, without Transformers, global backpropagation, or repeated full retraining?

## First technical objective

Design a candidate **Causal Credit Flow** rule. Every active unit will retain a short causal trace. When the system observes success, failure, or prediction error, a credit event will travel only through the pathways that participated in the outcome. Each unit will update from local state, local history, and the arriving credit signal.

The name is provisional. We will claim that the mechanism is new only after comparing its equations and behavior with existing research and patents.

## Success criteria

The first prototype succeeds only if reproducible experiments show that it can:

- learn a new task from a small number of experiences;
- retain old tasks after learning new ones;
- combine known operations in an unseen arrangement;
- operate without Transformer attention and end-to-end backpropagation;
- outperform appropriate compute-matched baselines on at least one important measure;
- explain which new mechanism caused the improvement through ablation tests.

## Research stages

### Stage 1 — Foundations and simulator

Implement a minimal environment containing locally connected computational units, delayed outcomes, changing rules, and complete visualization of the network's activity.

### Stage 2 — Learning-rule discovery

Formulate and test candidate local credit equations. Reject mechanisms that are unstable, reproduce existing algorithms, or fail simple falsification experiments.

### Stage 3 — Controlled evaluation

Evaluate rule induction, continual learning, compositional reasoning, delayed credit, and long-sequence memory. Compare against backpropagation, recurrent networks, Transformers, predictive coding, eligibility traces, and other relevant baselines.

### Stage 4 — Independent verification

Publish the equations, code, tests, negative results, and exact experiment configurations. Invite independent reproduction before making strong novelty or capability claims.

### Stage 5 — Scaling and application

If the mechanism survives controlled testing, extend it to perception, language, robotics, and scientific problem-solving.

## Humanity-focused applications

A successful continuously learning, efficient system could eventually support:

- private personalized education that adapts to each learner;
- affordable medical and scientific assistance in resource-limited regions;
- agricultural systems that learn local conditions;
- robots that acquire useful skills without massive centralized retraining;
- disaster-response tools that adapt when communication and cloud access are limited;
- energy-efficient intelligence running on ordinary devices.

## Principles

- Evidence before publicity.
- A precise mechanism before a large model.
- Falsifiable experiments before claims of novelty.
- Accessible computation before massive scaling.
- Human benefit, privacy, reliability, and independent verification from the beginning.

## Immediate milestone

Build a small prototype that learns several primitive operations, adds one new operation during use, retains its earlier abilities, and composes the operations in a configuration it never encountered during training.

That result would not be AGI. It would be the first evidence that the proposed learning principle deserves deeper research.

## Current implementation status

**Authoritative status (2026-09-04):** Gate A remains open and research
readiness remains **21/100**. The completed A3 passive-readout-trace experiment
reproduced deterministically with terminal status **`NO_SELECTION`**. Its
strongest candidate, `rho_0_95`, passed only **3 of 7** retention gates. The
current integrity-repaired source lineage is intentionally different from the
historical A3 and LWOH-L1 V2 source freezes, so those freezes must not be reused
for a new official run.

LWOH-L1 V2 is closed as an invalid procedure/source-drift lineage. Its
contingency evaluation fails closed as `INVALID_A3_DO_NOT_ACTIVATE_LWOH` with
action `REPAIR_A3_DO_NOT_ACTIVATE_LWOH`, or records `RuntimeError` when the
required terminal binding cannot be validated. No V2 activation, admission, or
online-learning result artifact exists. The next official memory experiment
must be a separately preregistered **post-A3/pre-LWOH V3**, frozen after the
integrity repairs and before any LWOH metric is observed.

Regenerate and verify the machine-derived current-status page with:

```powershell
python -m adaptive_learning_substrate.artifact_validation write-status
python -m adaptive_learning_substrate.artifact_validation verify
```

The generated [`docs/CANONICAL_STATUS.md`](docs/CANONICAL_STATUS.md), together
with the validator result and terminal artifacts, is authoritative over older
roadmap prose.

Stage 1 now has a **minimal, deterministic scaffold**. It is deliberately smaller
than the confirmatory experiment:

- [`docs/CCF_V0.md`](docs/CCF_V0.md) freezes the first mathematical learning rule;
- [`docs/EXPERIMENT_001.md`](docs/EXPERIMENT_001.md) preregisters Experiment 000 and Experiment 001;
- [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md) separates completed plumbing from pending research work;
- `src/adaptive_learning_substrate/` contains the fixed-graph simulator and local learners;
- `tests/` checks deterministic execution, delayed traces, locality, and fixed topology.

Run the verified scaffold from PowerShell:

```powershell
$env:PYTHONPATH='src'
python -m pytest -q -p no:cacheprovider
python -m adaptive_learning_substrate.cli --config configs/experiment_000.toml --output-dir artifacts/stage1/ccf
```

The resulting scaffold scores validate software plumbing only. They do **not**
establish novelty, superiority, or success on the preregistered 64-unit recurrent
Experiment 001.

### Development-only recurrent Experiment 000

A development diagnostic remains runnable on a fixed sparse graph with 64 hidden
units. Each episode presents `CUE`, eight actual `NOISE` events, and `QUERY`;
the target is delivered only after a two-tick readout produces a prediction.
The runner compares full CCF-v0, `CCF-NO-TRACE`, and deterministic uniform random
on paired streams. It uses NumPy float64 on the CPU and never uses automatic
differentiation or Transformer attention.

Run the verified 100-episode-per-split diagnostic from PowerShell:

```powershell
$env:PYTHONPATH='src'
python -m adaptive_learning_substrate.experiment_000 --config configs/experiment_000_development.toml --episodes 100 --output artifacts/experiment_000/development_100
```

Remove `--episodes 100` only when deliberately running the full 2,000-episode
development configuration. These are debugging results, not confirmatory
evidence. Confirmatory seeds 1000–1019 remain reserved and unexecuted;
`configs/experiment_000_confirmatory.toml` is intentionally disabled.

### Exact update-alignment diagnostic

The completed custom-seed diagnostic compares CCF-v0's actual weight change
with exact local descent on the timestamped event graph that really occurred:

```powershell
$env:PYTHONPATH='src'
python -m adaptive_learning_substrate.experiment_000_alignment `
  --seeds 42 43 44 45 46 `
  --episodes-per-seed 20 `
  --finite-difference-coordinates 32 `
  --output artifacts/experiment_000/alignment/custom_seeds_42_46_episodes_20.json
```

All 3,200 derivative checks passed and an identical full rerun reproduced the
same deterministic payload hash. Median CCF-to-exact cosine was 0.5167 and
every CCF update pointed downhill, but the equal-norm exact direction reduced
local loss about 2.07 times as much. Almost no exact sensitivity reached the
original cue, while CCF over-allocated update energy to query and noise edges.
The current diagnosis is therefore a combined forward-memory and
credit-allocation bottleneck—not successful continual learning. See
[`docs/EXPERIMENT_000_ALIGNMENT_RESULT.md`](docs/EXPERIMENT_000_ALIGNMENT_RESULT.md)
for the equations, complete metrics, and artifact hashes.

### Architecture-only recurrent-memory sweep

The next frozen diagnostic tested whether uniformly increasing only the
recurrent spectral radius could preserve the cue without unstable or excessive
activity. It used fresh custom seeds 50--54, 100 paired train streams and 100
paired evaluation streams per seed, four registered radii, a native control,
and no learning or credit calls:

```powershell
$env:PYTHONPATH='src'
python -m adaptive_learning_substrate.experiment_000_memory_sweep `
  --seeds 50 51 52 53 54 `
  --target-radii 0.60 0.80 0.95 1.05 `
  --pairs-per-split 100 `
  --ridge-alpha 0.001 `
  --output artifacts/experiment_000/memory_sweep/custom_seeds_50_54_pairs_100.json
```

The official result was **`NO_SELECTION`**, reproduced by a second full run
with deterministic payload SHA-256
`f6ba680f91917521da5bbfb89d7d1dbd102fe59149049d0ed0464bb3ece7f8e6`.
Larger radii amplified the cue, but none passed every cue/noise, activity, and
quiescence gate; radius 1.05 also kept firing late on four of five seeds. No
failed condition advances. See
[`docs/EXPERIMENT_000_MEMORY_SWEEP_RESULT.md`](docs/EXPERIMENT_000_MEMORY_SWEEP_RESULT.md).

### Architecture-only slow local-state diagnostic

The next frozen diagnostic kept every weight, edge, and the CCF equation fixed
and instead gave each hidden unit one fixed lazy exponential local memory of its
own last activation. It used fresh custom seeds 60--64, 100 paired train and 100
paired evaluation streams per seed, five retention coefficients, a native
control, and no learning or credit calls:

```powershell
$env:PYTHONPATH='src'
python -m adaptive_learning_substrate.experiment_000_slow_state `
  --seeds 60 61 62 63 64 `
  --retentions 0.25 0.50 0.75 0.90 0.95 `
  --pairs-per-split 100 `
  --ridge-alpha 0.001 `
  --output artifacts/experiment_000/slow_state/custom_seeds_60_64_pairs_100.json
```

The official result was again **`NO_SELECTION`**, reproduced by a second full
run with deterministic payload SHA-256
`0d95583e2e104a6c945fec0796051f9c198b4038982fa4a63d69d97989970e1e` and
confirmed by the frozen `verify_deterministic_full_runs` check. This time the
diagnosis is sharper: at retention 0.90 and 0.95 the cue became fully linearly
decodable at `QUERY` (fixed ridge readout 1.00 evaluation accuracy on all five
seeds, cue/noise ratio 0.75--0.86 versus 0.16 native). Only retention 0.95
cleared all seven preregistered retention gates; retention 0.90 failed the
output gate (`O = 0.415954 < 0.50`). Both failed stability and budget -- the
ungated leak self-drives the network to ~0.96 blank-tick
hidden emission density (native ~0.015) and 1.17--1.18x matched-episode event
activity (limit 1.15). The forward representation is not the sole bottleneck;
the next intervention must supply a bounded local state that is read at `QUERY`
without becoming a self-sustaining excitation source. See
[`docs/EXPERIMENT_000_SLOW_STATE_RESULT.md`](docs/EXPERIMENT_000_SLOW_STATE_RESULT.md).

### A3 passive-readout-trace result and closed V2 lineage

Gate A's third attempt, A3, completed on custom seeds 105--109 with a persisted
smoke test, full run, independent rerun, and terminal determinism verification.
The terminal artifact is
[`artifacts/experiment_000/readout_trace_a3/DETERMINISM_VERIFICATION.json`](artifacts/experiment_000/readout_trace_a3/DETERMINISM_VERIFICATION.json).
It records deterministic payload equality and final status **`NO_SELECTION`**.
The strongest candidate, `rho_0_95`, passed only 3 of the 7 registered retention
gates: median cue/noise ratio, its native multiple, and ridge accuracy. It failed
both absolute-cue gates, the within-seed cue-ratio count, and the output-ratio
gate. Stable passive memory was therefore too weak for selection.

The subsequent LWOH-L1 V2 contingency freeze is retained only as historical
procedure evidence. Integrity repairs changed shared source files, and V2 never
produced an activation receipt, admission run, or learning run. It is closed as
invalid rather than resumed. A newly versioned post-A3/pre-LWOH V3 must state its
equations, thresholds, seeds, failure branches, and source manifest prospectively,
then freeze them before observing any LWOH metric.

The evidence-based route to a top-class research system is tracked in
[`docs/RESEARCH_READINESS_90.md`](docs/RESEARCH_READINESS_90.md). The current
controlled-research score is 21/100; 90/100 is a future evidence gate, not a
claim based on code volume.
