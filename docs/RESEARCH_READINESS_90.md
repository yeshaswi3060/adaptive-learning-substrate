# Research Readiness: Evidence-Based Route from 21/100 to 60/100 and 90/100

**Audit date:** 2026-09-04 (updated after A3 `NO_SELECTION` and integrity repair)

**Current research-readiness score:** **21/100**  
**Current deployable-product readiness:** **approximately 2--3/100**  
**First major milestone:** **60/100 research readiness**, meaning an externally
reproducible, competitive research prototype.  
**Top-class research target:** **90/100 research readiness**, meaning a mature,
independently reproduced research system that survives multiple real task
families, strong baselines, structural ablations, robustness tests, and
efficiency audits. Neither label means a finished commercial product, a
market-ready service, or evidence of AGI.

## 1. What the score means

This is a conservative project-control rubric, not a scientific metric and not
a percentage probability of eventual success. It measures how much of the
project's central claim is supported by inspectable evidence.

Points are awarded only for completed evidence: an implemented mechanism, a
test, a frozen protocol, raw results, a deterministic rerun, a controlled
baseline, or an independent reproduction. A proposal, attractive metric, or
planned feature earns no result points. A negative experiment can improve
diagnostic knowledge and reproducibility, but it does not earn capability
points for a behavior that failed.

The separate product-readiness estimate is much lower because the repository
does not yet contain a task-capable model, production API, deployment and
monitoring system, user evaluation, security/privacy program, reliability
service levels, or cost/latency validation. Reaching 60/100 would make the
project a serious research prototype. Reaching 90/100 would make the research
case unusually complete, but it would still not automatically provide product
security, operations, regulation, support, or market adoption.

## 2. Evidence that exists now

The project has produced solid engineering and diagnostic evidence:

1. `docs/CCF_V0.md` specifies a local Causal Credit Flow rule, and the NumPy
   simulator represents timestamped unit events, local edge traces, delayed
   credit packets, fixed topology, and bounded local weight updates.
2. The positive control in `docs/EXPERIMENT_000_DEVELOPMENT_LOG.md` showed that
   CCF repaired the only plastic earlier edge on a clean 11-edge delayed path:
   the edge moved from `-0.9` to `0.12472518983448343`, full CCF reached `1.0`
   evaluation accuracy, and `CCF-NO-TRACE` remained at `0.0`.
3. The actual 64-hidden-unit delayed recurrent diagnostic did **not** learn:
   mean evaluation accuracy was `0.482` for CCF, `0.482` for no-trace, and
   `0.484` for random on the reduced development run. This is chance-level
   behavior, so Experiment 000 has not passed.
4. The exact fixed-event-DAG alignment diagnostic passed all `3,200` derivative
   checks and reproduced deterministic payload SHA-256
   `17573dd402b252c82c3c22286f6cb0898bb9cd50fd8d37b6cdf57ea38fa7ec58`.
   Median CCF-to-exact cosine was `0.516699176351386`; every measured CCF update
   pointed downhill, but an equal-norm exact direction reduced local loss about
   `2.06943` times as much. The evidence identifies both weak forward memory and
   inefficient credit allocation. See `docs/EXPERIMENT_000_ALIGNMENT_RESULT.md`.
5. The frozen architecture-only recurrent-radius sweep on custom seeds 50--54
   completed twice with identical deterministic payload SHA-256
   `f6ba680f91917521da5bbfb89d7d1dbd102fe59149049d0ed0464bb3ece7f8e6`.
   Its official result was exactly `NO_SELECTION`. The strongest radius,
   `rho_1_05`, improved median absolute cue signal by about `34.40x` but still
   missed the median cue/noise threshold (`0.46845 < 0.50`), achieved only
   three of five required within-seed ratio improvements, exceeded the matched
   activity limit, and failed the blank-tail stability gate. It must not be
   promoted as the architecture. The primary raw report is
   `artifacts/experiment_000/memory_sweep/custom_seeds_50_54_pairs_100.json`.

6. The frozen architecture-only slow local-state diagnostic on custom seeds
   60--64 completed twice with identical deterministic payload SHA-256
   `0d95583e2e104a6c945fec0796051f9c198b4038982fa4a63d69d97989970e1e`,
   verified by `verify_deterministic_full_runs`. Its official result was again
   exactly `NO_SELECTION`. An ungated additive exponential per-unit local state
   at retention 0.90 and 0.95 produced fixed ridge readout 1.00 evaluation
   accuracy on all five seeds. Only retention 0.95 cleared all seven retention
   gates (retention 0.90 failed the output gate: `O = 0.415954 < 0.50`). Both
   produced median absolute cue ~38--49x native and cue/noise ratio 0.75--0.86,
   but failed the quiescence gate
   (~0.96 blank-tick hidden emission density versus ~0.015 native) and the
   matched-episode activity budget (1.17--1.18x versus 1.15). The primary raw
   report is
   `artifacts/experiment_000/slow_state/custom_seeds_60_64_pairs_100.json`.

7. A3, the stable passive-readout-trace diagnostic on custom seeds 105--109,
   completed with a persisted smoke, full run, independent full rerun, and
   machine terminal verification. The deterministic payloads were identical and
   the official terminal result is **`NO_SELECTION`**. Its strongest candidate,
   `rho_0_95`, passed only **3 of 7** retention gates: median cue/noise ratio,
   the corresponding native multiple, and ridge evaluation accuracy. It failed
   both absolute-cue gates, the within-seed cue-ratio count, and the output-ratio
   gate. The terminal record is
   `artifacts/experiment_000/readout_trace_a3/DETERMINISM_VERIFICATION.json`.

8. Evidence-integrity repairs changed shared source files after the historical
   A3 and LWOH-L1 V2 freezes. The A3 result remains a completed historical
   negative result, but neither old source manifest governs the current code.
   LWOH-L1 V2 is closed as invalid procedure/source drift, with no activation,
   admission, or online-learning artifact. Its fail-closed contingency states
   are `INVALID_A3_DO_NOT_ACTIVATE_LWOH` /
   `REPAIR_A3_DO_NOT_ACTIVATE_LWOH`, or `RuntimeError` when terminal validation
   fails. A separate post-A3/pre-LWOH V3 must be preregistered and frozen before
   any new LWOH metric.

These results establish that the simulator, local causal routing on a clean
path, derivative instrumentation, and falsification workflow are real, and that
a per-unit forward memory can make the cue linearly decodable. They do
**not** establish continual learning, unseen composition, useful structural
plasticity, superiority to baselines, novelty, or real-world competitiveness.

## 3. Current scorecard

| Evidence category | Maximum | Current | Why the current points are justified |
|---|---:|---:|---|
| Mechanism specification and locality | 12 | **8** | Frozen CCF-v0 equations, local records and updates, locality tests, deterministic simulator, and clean-chain positive control exist. The rule is not yet validated on the recurrent task. |
| Delayed causal credit on a useful task | 10 | **2** | The clean-path positive control passes, but the 64-unit delayed recurrent diagnostic remains at chance and the confirmatory gate is closed. |
| Continual learning and retention | 10 | **1** | Task adapters and a frozen protocol exist; no sequential learn-new/retain-old result has passed. |
| Unseen compositional behavior | 8 | **1** | Primitive adapters and externally chained scaffold checks exist; the required single-episode unseen compositions have not passed. |
| Online structural plasticity | 10 | **0** | Growth, pruning, and rewiring are deliberately absent; topology is currently fixed. |
| Baselines and novelty boundary | 14 | **1** | A same-topology eligibility comparison scaffold and baseline protocol exist, but ET-3F, BPTT, BPTT-CM, full ablations, and a literature/patent boundary map are incomplete. |
| Reproducibility and provenance | 10 | **5** | Frozen protocols/configurations, seed partitions, hashes, raw JSON reports, invariants, and identical diagnostic reruns exist. A complete environment lock, canonical streaming archive, and clean-machine external reproduction remain pending. |
| Compute and energy efficiency | 8 | **1** | Edge-touch ledgers and accessible CPU execution exist. No passing compute-matched advantage or energy measurement exists; native activity is already dense. |
| Robustness and failure tolerance | 10 | **0** | Finiteness and blank-tail checks diagnose individual runs, but no systematic multi-condition robustness result has passed. |
| Packaging and external usability | 8 | **2** | A Python package, command-line runners, tests, configs, and documentation exist. There is no frozen release bundle or verified clean-install reproduction. |
| **Total** | **100** | **21** | **Early research scaffold with strong diagnostics, not a competitive system.** |

The total is intentionally not raised by the three completed `NO_SELECTION`
memory diagnostics. Their preserved negative results narrow the research
question, and A3's identical terminal rerun strengthens the historical record,
but none delivered a usable memory architecture. The later integrity repairs
also mean the current executable lineage needs a new prospective freeze. These
are diagnostic and evidence-quality advances, not capability points, so the
score remains **21/100**.

The machine-derived current view is `docs/CANONICAL_STATUS.md`. Regenerate it
with `python -m adaptive_learning_substrate.artifact_validation write-status`
and verify it with
`python -m adaptive_learning_substrate.artifact_validation verify`; a verified
terminal artifact and validator result override stale roadmap prose.

## 4. Exact scoring bands

The following bands prevent subjective score inflation. Intermediate points
may be awarded only when the corresponding evidence is complete.

### 4.1 Mechanism specification and locality--12 points

- `0`: no explicit rule;
- `3`: conceptual rule only;
- `6`: frozen equations and information-access contract;
- `8`: deterministic implementation, locality/invariant tests, and a positive
  causal-path control;
- `10`: the mechanism passes its delayed recurrent task and required ablations;
- `12`: theoretical boundary analysis plus independent empirical reproduction.

### 4.2 Delayed causal credit--10 points

- `0`: no delayed task;
- `2`: constructed clean-path positive control;
- `4`: recurrent development result materially above chance on multiple seeds;
- `6`: the preregistered Experiment-000 confirmatory gate passes;
- `8`: the result survives additional delays and compute-matched local/global
  references;
- `10`: independent reproduction on a separately implemented task/runner.

### 4.3 Continual learning and retention--10 points

- `0`: no sequential task protocol;
- `1`: task definitions/adapters and frozen protocol only;
- `4`: development prototype learns a new operation without replay and retains
  old operations;
- `6`: the preregistered multi-seed retention criteria pass;
- `8`: required ablations and learned baselines establish the mechanism effect;
- `10`: independent reproduction under a second task family.

### 4.4 Unseen composition--8 points

- `0`: no composition test;
- `1`: externally chained primitives or protocol only;
- `3`: development system solves unseen in-episode programs;
- `5`: the preregistered nine-program, multi-seed composition gate passes;
- `6`: a significant advantage or efficiency parity is shown against required
  learned baselines;
- `8`: independent reproduction on a held-out composition grammar.

### 4.5 Online structural plasticity--10 points

- `0`: fixed topology only;
- `2`: deterministic local growth/prune/rewire operations with invariant tests;
- `4`: online structural change improves or preserves a controlled task while
  retaining earlier skills;
- `6`: a frozen-structure ablation attributes the effect to reorganization;
- `8`: the mechanism stays bounded and useful across task orders and budgets;
- `10`: independent reproduction with an interpretable structural audit trail.

### 4.6 Baselines and novelty boundary--14 points

- `0`: no comparison;
- `1`: comparison scaffold or protocol only;
- `4`: ET-3F, BPTT, BPTT-CM, random, and required CCF ablations are implemented
  and verified for fairness;
- `7`: compute-matched experiments and an equation-by-equation primary
  literature/patent boundary map are complete;
- `10`: the preregistered comparative gate passes against both ET-3F and
  BPTT-CM;
- `12`: conclusions survive a strong alternative implementation or benchmark;
- `14`: external reviewers reproduce the comparison and find no unreported
  equivalent mechanism in the documented search boundary.

### 4.7 Reproducibility and provenance--10 points

- `0`: unverifiable output;
- `2`: deterministic seeds and ordinary tests;
- `5`: frozen protocols/configurations, hashes, raw reports, integrity checks,
  and identical reruns;
- `7`: complete environment lock, streaming canonical logs, one-command
  analysis, and machine-readable artifact manifest;
- `9`: clean-machine reproduction by a person/process not involved in the
  implementation;
- `10`: a second independent reproduction with matching verdict.

### 4.8 Compute and energy efficiency--8 points

- `0`: no measurement;
- `1`: compute ledger and accessible-hardware execution;
- `3`: complete edge-touch, memory, latency, wall-clock, and energy tables;
- `4`: capability is retained under a frozen compute budget against every
  required baseline;
- `6`: the preregistered compute- or experience-efficiency comparative gate
  passes;
- `8`: the result reproduces across CPU and GPU-class accessible hardware.

### 4.9 Robustness and failure tolerance--10 points

- `0`: no systematic robustness result;
- `2`: deterministic failure injection and multiple delay/noise levels;
- `4`: a frozen seed/task-order/perturbation stress matrix passes its confidence
  and stability criteria;
- `6`: recovery, non-finite, runaway-activity, and bounded-resource gates pass;
- `8`: held-out distribution shifts preserve the minimum useful behavior;
- `10`: independent stress testing reproduces the result.

### 4.10 Packaging and external usability--8 points

- `0`: cannot be run by another user;
- `2`: package, CLI, tests, configs, and basic documentation;
- `4`: pinned environment, schema, licenses, exact run scripts, and validated
  artifact bundle;
- `6`: a clean install reproduces the principal tables with one command;
- `8`: versioned archival release with citation metadata and external run log.

## 5. The non-negotiable definition of 60/100

The project receives a 60/100 research-readiness label only when:

1. the total score is at least `60`; **and**
2. every category reaches its floor below; **and**
3. the frozen `docs/EXPERIMENT_001.md` verdict is exactly
   `EXPERIMENT 001 PASS` (or a prospectively versioned successor with equally
   strict or stricter gates); **and**
4. a clean-machine reproduction from the frozen bundle returns the same
   qualitative verdict.

| Category | Required floor at 60 |
|---|---:|
| Mechanism specification and locality | 8/12 |
| Delayed causal credit | 6/10 |
| Continual learning and retention | 6/10 |
| Unseen composition | 5/8 |
| Online structural plasticity | 4/10 |
| Baselines and novelty boundary | 7/14 |
| Reproducibility and provenance | 7/10 |
| Compute and energy efficiency | 4/8 |
| Robustness and failure tolerance | 4/10 |
| Packaging and external usability | 4/8 |

The floors sum to `55`, so another five points must come from genuine strength
above the minimum. This prevents a polished package or one exceptional result
from hiding a missing core capability.

## 6. The non-negotiable definition of 90/100

The 90/100 target is not “a larger Experiment 001.” It requires the mechanism
to remain useful when tasks, implementations, hardware, and evaluators change.
The project receives a 90/100 research-readiness label only when:

1. the total score is at least `90`;
2. every category reaches the hard floor below;
3. every 60/100 condition and Gate A through Gate F has passed;
4. Gate G through Gate L below has passed without a post-result threshold
   change;
5. at least two independent reproducers have returned the same principal
   verdict, and at least one reproduction uses an independently written
   implementation of the learning rule or evaluator;
6. all capability wording is scoped to the tested task families, and the
   documented novelty boundary contains no unresolved equivalence that would
   make the central “new mechanism” wording false.

| Category | Hard floor at 90 |
|---|---:|
| Mechanism specification and locality | 11/12 |
| Delayed causal credit | 9/10 |
| Continual learning and retention | 9/10 |
| Unseen composition | 7/8 |
| Online structural plasticity | 8/10 |
| Baselines and novelty boundary | 12/14 |
| Reproducibility and provenance | 10/10 |
| Compute and energy efficiency | 7/8 |
| Robustness and failure tolerance | 9/10 |
| Packaging and external usability | 8/8 |
| **Required total of floors** | **90/100** |

Because these floors already total `90`, no category can compensate for a
deficit in another. For example, excellent accuracy cannot replace structural
plasticity, and beautiful packaging cannot replace independent reproduction.
A score above 90 would require additional independently verified strength in
the categories whose maxima exceed their floor.

## 7. Exact critical-path gates

All gates below are pass/fail. Later work does not start on confirmatory seeds
until the earlier gate passes. Proposed future thresholds are readiness gates,
not claims about results already obtained.

### Gate A--select a stable memory architecture

**Attempts so far, all three `NO_SELECTION`:**

1. *Uniform recurrent-radius scaling* (custom seeds 50--54, payload
   `f6ba680f...`). Stronger recurrence amplified the cue but never produced a
   clean enough cue/noise ratio, and `rho_1_05` fired persistently on four of
   five seeds. Rejected.
2. *Ungated additive exponential per-unit local state* (custom seeds 60--64,
   payload `0d95583e...`, verified rerun). New information: retention 0.90 and
   0.95 both reached ridge readout 1.00 on all five seeds, but only retention
   0.95 cleared **all seven retention gates**; retention 0.90 failed the output
   gate (`O = 0.415954 < 0.50`). Both had median absolute cue ~38--49x native
   and cue/noise ratio 0.75--0.86. Both failed stability and budget -- the
   ungated leak self-drives the network to
   ~0.96 blank-tick hidden emission density (native ~0.015) and 1.17--1.18x
   matched-episode activity. Rejected as written; see
   `docs/EXPERIMENT_000_SLOW_STATE_RESULT.md`.
3. *A3 passive readout-only trace* (custom seeds 105--109, deterministic full
   rerun). The candidate dynamics stayed native and stable, but the memory signal
   was insufficient. `rho_0_95` was strongest and passed only 3/7 retention
   gates; the exact terminal status was `NO_SELECTION`. Rejected.

Together the three attempts show a retention/stability tradeoff: strong additive
memory retains the cue but self-excites, while stable passive A3 memory is too
weak. The old A3 and LWOH-L1 V2 freezes are historical lineages because integrity
repairs changed current shared source bytes. V2 produced no activation,
admission, or learning artifacts and is closed as invalid procedure/source
drift.

The next architecture-only intervention must be a separately preregistered
**post-A3/pre-LWOH V3** with fresh custom seeds, CCF held fixed, and current
integrity-repaired source hashes. Its equations, thresholds, failures, tests,
smoke, and source manifest must be frozen before any LWOH metric. It must retain
the following admission requirements unless a prospectively justified protocol
registers a new scientific question:

- median evaluation absolute cue RMS at least `3x` native and at least four of
  five seeds individually at least `3x` native;
- median cue/noise ratio at least `0.50`, at least `3x` native, and at least
  four of five within-seed ratios at least `3x` native;
- median ridge evaluation accuracy at least `0.70`;
- median output cue-delta/SD at least `0.50`;
- total task activity at most `1.10x` native for every seed/split and maximum
  matched-episode activity at most `1.15x`;
- zero hidden emissions during blank ticks 193--256 for both cue polarities,
  every seed, and every candidate;
- all topology, frozen-weight, no-learning, finiteness, and deterministic-rerun
  checks pass.

If no candidate passes, report `NO_SELECTION` again; do not promote the closest
curve.

### Gate B--repair credit allocation without mixing interventions

Hold the Gate-A architecture fixed and version any changed credit equation as
CCF-v1. On a new exact-DAG alignment protocol and fresh custom seeds, require:

- `100%` of stratified finite-difference checks pass tolerance;
- median CCF-to-exact cosine at least `0.70`;
- at least `95%` of episodes have a positive CCF-to-exact dot product;
- mean CCF equal-norm loss reduction is at least `80%` of the mean exact
  equal-norm loss reduction;
- CCF's combined squared update fraction on noise and query inputs is no more
  than `2x` the exact direction's fraction (with a preregistered zero guard);
- two complete runs have identical deterministic payload hashes and use zero
  confirmatory seeds.

These thresholds deliberately require improvement beyond the current median
cosine `0.5167`, loss-effectiveness ratio about `0.483`, and strongly
misallocated noise/query update energy.

### Gate C--pass delayed recurrent learning

Run the existing Experiment-000 confirmatory gate only after A and B pass and
all code/configuration hashes are frozen. Across 20 confirmatory seeds, the
existing protocol requires all of:

- mean delay-8 accuracy at least `0.90`;
- at least `16/20` seeds at accuracy at least `0.85`;
- mean advantage over random at least `0.25`, with the multiplicity-adjusted
  95% paired confidence interval excluding zero;
- mean advantage over `CCF-NO-TRACE` at least `0.15`, with the adjusted 95%
  paired confidence interval excluding zero;
- unchanged graph masks and no non-finite run.

### Gate D--pass continual learning, composition, mechanism, and competition

Use the frozen Experiment-001 protocol. The core behavior gate requires:

- mean first-phase old-skill accuracy `A_A >= 0.95`;
- mean new-skill accuracy `A_new >= 0.90`;
- mean retained-old accuracy `A_ret >= 0.90`;
- mean forgetting `F <= 0.05`;
- mean unseen-composition accuracy `A_comp >= 0.80`;
- every one of nine composition programs has mean accuracy at least `0.65`;
- at least `15/20` seeds individually satisfy the protocol's five per-seed
  behavior bounds;
- all integrity checks pass.

The mechanism gate additionally requires bottleneck-score advantages of at
least `0.10` over `NO-TRACE`, `0.10` over `SHUFFLED-CAUSE`, and `0.05` over
`BROADCAST`, each with a positive Holm-adjusted 95% paired interval.

Against **each** of ET-3F and BPTT-CM, the comparative gate then requires one
preregistered important win: bottleneck score at least `0.05` higher, median
E90 at least `25%` lower with no meaningful score loss, or median edge touches
at least `2x` lower with no meaningful score loss. Only the exact final label
`EXPERIMENT 001 PASS` clears this gate.

### Gate E--demonstrate bounded structural plasticity

After fixed-topology credit passes, preregister one local structural rule and a
frozen-structure ablation. Require all of:

- growth, pruning, or rewiring decisions use only declared local state/history;
- every structural event has a reason code and before/after graph hash;
- no invalid topology, non-finite state, or budget violation across all seeds;
- after adding the new operation, old-skill forgetting remains at most `0.10`;
- compared with the frozen-structure ablation, either bottleneck score improves
  by at least `0.05` or median edge touches fall by at least `25%`, while the
  lower paired 95% confidence bound on score difference is no worse than
  `-0.02`;
- a deterministic replay reconstructs the same structural event sequence.

### Gate F--novelty boundary, robustness, and external reproduction

Before any novelty or real-world competitiveness claim:

- build an equation-by-equation boundary matrix against primary papers and
  relevant patents, recording local variables, timing, causal routing, update
  equation, structural rule, and experimentally distinguishable difference;
- publish overlaps and negative search results, and treat the name/mechanism as
  provisional wherever equivalence remains unresolved;
- freeze a stress matrix covering seed, task order, delay, distractor rate,
  initialization, and activity/compute budgets; define thresholds before runs;
- provide a pinned environment, one-command full analysis, raw streaming logs,
  artifact manifest, and exact expected verdict;
- obtain at least one clean-machine reproduction by a person or process that
  did not implement the tested mechanism.

### Gate G--generalize to three real task families

The 90 target requires evidence beyond the hand-designed micro-programs. Freeze
three task families before their final evaluations:

1. delayed symbolic programs and continual composition;
2. a public, nonstationary sequential data task with real recorded observations;
3. a public perception-control or embodied simulator task with delayed outcomes.

At least two families must use externally maintained public data or environments,
and at least one final evaluation must be hidden from the mechanism developers
until the release candidate is frozen. For every family require:

- a public task/version checksum, train/development/final split contract, primary
  score, chance level, and failure threshold;
- at least 20 paired seeds for stochastic simulators or five official disjoint
  data splits for recorded datasets;
- online acquisition of a new skill/regime without full replay, old-skill
  forgetting at most `0.05`, and no complete retraining;
- performance above the preregistered useful-task threshold and a normalized
  primary score no more than `0.02` below the strongest compute-matched learned
  baseline unless CCF is significantly better;
- across at least two of the three families, either normalized primary score is
  at least `0.05` higher, E90 is at least `25%` lower, or active edge touches are
  at least `2x` lower than every required compute-matched learned baseline, with
  multiplicity-adjusted paired evidence.

One family passing cannot substitute for another. Any task-specific adapter must
be frozen and included in the compute ledger.

### Gate H--pass a preregistered robustness and recovery matrix

Create at least 30 final stress cells spanning all three task families and the
following axes: seed/data split, task order, delay length, distractor/noise rate,
initialization, bounded observation corruption, compute budget, and one abrupt
distribution shift. Freeze the matrix before final runs. Require:

- at least `90%` of cells meet their preregistered useful-task threshold;
- the median normalized-score drop from the in-distribution reference is at
  most `0.05`, and the tenth-percentile drop is at most `0.15`;
- zero non-finite states, invalid structural transitions, hash/provenance
  failures, or unbounded post-input activity in every final cell;
- after an abrupt shift, the system regains at least `95%` of its own pre-shift
  score within the preregistered adaptation budget while old-task forgetting
  remains at most `0.10`;
- confidence intervals and all failed cells remain in the published analysis.

No average may hide a failed safety/integrity invariant.

### Gate I--demonstrate sparse, accessible-hardware efficiency

Measure full training-plus-adaptation and evaluation on a CPU-only accessible
machine and on the project's RTX 5060-class machine. For CCF, ET-3F, BPTT-CM,
and the strongest capability-matched baseline, publish wall time, examples,
active unit-ticks, active edge touches, peak RAM/VRAM, serialized model size,
and device energy using one frozen measurement procedure. Require:

- CCF's primary score is non-inferior within `0.02` on every claimed efficiency
  comparison;
- median active hidden unit-tick density is at most `0.25` and its per-episode
  95th percentile is at most `0.40` on every claimed task family;
- on at least two task families, CCF uses at least `2x` fewer active edge
  touches and at least `1.5x` less measured adaptation energy than the strongest
  non-inferior learned baseline;
- CPU and GPU executions return the same categorical verdict, with numerical
  tolerance frozen before the cross-backend run;
- all compilation, preprocessing, adapter, and failed-run costs are included.

This gate is intentionally incompatible with calling the current hidden density
near `0.82--0.89` “sparse.”

### Gate J--show structural plasticity across task families

Extend Gate E from one controlled task to every Gate-G family. Require:

- a fixed maximum unit/edge budget and an emergency no-growth ceiling before
  each final run;
- local reason codes, parent evidence, before/after hashes, and deterministic
  replay for `100%` of structural events;
- final graph size between `0.5x` and `2.0x` its initial size unless a narrower
  task-specific range was preregistered;
- zero structural invariant failures in Gate-H stress runs;
- versus a frozen-structure ablation, at least two task families show either a
  normalized-score advantage of `0.05` or an active-edge-touch reduction of
  `25%`, while every family's paired lower 95% confidence bound on score
  difference is no worse than `-0.02`;
- old-skill forgetting after reorganization remains at most `0.05`.

### Gate K--close the novelty boundary under independent challenge

Run a documented systematic search over primary research and relevant patents,
using saved queries, dates, inclusion/exclusion rules, citation chaining, and a
versioned candidate ledger. The final boundary matrix must compare equations,
state variables, eligibility/causal traces, feedback access, temporal routing,
credit propagation, structural plasticity, and falsifying experiments. Require:

- every central mechanism component maps to its closest known antecedent;
- at least two technically qualified reviewers with no authorship of the tested
  code independently attempt to find an equivalent earlier mechanism;
- one reviewer is explicitly asked to construct the strongest equivalence or
  “this is only X” argument, and the response is published;
- searches are refreshed within 90 days of the archival release;
- claims are narrowed wherever evidence supports only a new combination,
  implementation, or empirical behavior;
- discovery of an equation-and-information-contract equivalent prior mechanism
  fails the “fundamentally new rule” gate until the claim is corrected.

This is a scientific novelty boundary, not a legal patentability or freedom-to-
operate opinion.

### Gate L--release and independently reproduce the complete system

Publish a versioned archival bundle containing tagged source, source archive
hash, pinned CPU/GPU environments, dependency lock, software bill of materials,
licenses, dataset/environment checksums, protocol/config hashes, raw streaming
logs, analysis notebooks/scripts, expected table hashes, failure cases, a
research card, and citation metadata. Require:

- one clean command rebuilds the environment and one clean command reproduces
  every principal table from raw artifacts;
- continuous integration runs unit/invariant tests on CPU and a scheduled
  deterministic parity suite on the supported GPU backend;
- two independent reproducers return the same pass/fail verdict on Gates C--J;
- at least one reproducer uses an independently written CCF implementation or
  independent evaluator rather than importing the project's learning module;
- reproduced primary metrics fall inside the preregistered statistical or
  numerical tolerance, and every discrepancy is published;
- the archival release is immutable and receives a persistent identifier.

Only after Gate L passes can reproducibility score `10/10` and packaging score
`8/8`.

## 8. Shortest defensible execution sequence

1. **Keep A3 closed as `NO_SELECTION`.** Preserve its full/rerun hashes,
   3-of-7 strongest-candidate result, and deterministic terminal verification.
2. **Keep LWOH-L1 V2 closed as invalid.** Preserve its pre-terminal record, but
   do not create activation, admission, or learning artifacts under that drifted
   lineage.
3. **Preregister and freeze post-A3/pre-LWOH V3.** Bind the integrity-repaired
   source, fresh seeds, equations, thresholds, failures, tests, and smoke before
   observing any LWOH metric. Advance only a Gate-A pass.
4. **Develop CCF-v1 allocation on the selected architecture.** Keep the memory
   mechanism fixed, and advance only a Gate-B pass.
5. **Run Experiment 000 development, then confirmation.** Do not touch reserved
   seeds until the implementation/config/environment bundle is frozen.
6. **Complete ET-3F, BPTT, BPTT-CM, RAND, and all CCF ablations.** Validate
   fairness and ledgers before Experiment 001.
7. **Run Experiment 001 once under the frozen protocol.** Accept its exact
   verdict; a failed gate creates a new version and fresh seeds rather than a
   changed threshold.
8. **Add structural plasticity only after fixed-topology learning works.** This
   keeps learning-rule, memory, and structure effects identifiable.
9. **Complete novelty mapping, stress testing, and release packaging.** Ask an
   independent reproducer to run the frozen bundle.
10. **Rescore from artifacts.** Award 60 only if the total, all category floors,
   `EXPERIMENT 001 PASS`, and clean-machine reproduction conditions are met.
11. **Freeze three task-family protocols.** Keep the successful micro-program
    study, add public nonstationary sequential data and delayed
    perception/control, and run Gate G without task-specific threshold tuning.
12. **Expand structural plasticity and robustness.** Run the Gate-H stress
    matrix and Gate-J frozen-structure comparisons across all three families.
13. **Build and audit the efficient GPU backend.** First prove CPU/GPU parity,
    then perform the complete Gate-I activity, compute, memory, latency, and
    energy comparison; the RTX 5060 is one supported accessible tier, not the
    only reported machine.
14. **Invite an independent novelty challenge.** Close Gate K, revise claims to
    their defensible scope, and freeze the release candidate.
15. **Release and reproduce.** Two independent reproducers, including one
    independent implementation/evaluator, must close Gate L.
16. **Rescore again from immutable evidence.** Award 90 only if every 90 floor
    and every Gate A--L condition is satisfied. A result of 89 remains 89.

The RTX 5060 can accelerate later batched baselines and larger sweeps, but more
hardware does not repair a failed causal mechanism. Current NumPy float64
diagnostics intentionally run on CPU; GPU use should begin when a tested
vectorized backend preserves the deterministic and fairness contracts.

## 9. Immediate decision

The next scientific task is **not** to scale the current model, resume LWOH-L1
V2, or declare 60% or 90% completion. Three architecture-only memory
interventions have returned `NO_SELECTION`. Strong additive memory retained the
cue but self-fired; stable passive A3 memory was too weak and its strongest
candidate passed only 3/7 gates. Readiness therefore stays **21/100**.

The shortest controlled step is a separately versioned **post-A3/pre-LWOH V3**.
It must be preregistered after the evidence-integrity repairs, bind the current
source lineage and fresh seeds, and freeze before any LWOH metric. Only a fully
passing Gate-A candidate advances to credit-allocation work (Gate B) and the
delayed recurrent learning gate (Gate C). LWOH-L1 V2 remains closed with no
activation/admission/learning artifacts; treating its drifted freeze as current
would create procedure volume, not evidence.
