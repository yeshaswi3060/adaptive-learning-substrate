# Stage 1 implementation status

## Authoritative current status -- 2026-09-04

- Research readiness remains **21/100**; Gate A has not selected a memory
  architecture.
- A3 completed with an identical deterministic rerun and terminal
  **`NO_SELECTION`**. Its strongest candidate, `rho_0_95`, passed **3/7**
  retention gates.
- Evidence-integrity repairs changed shared source bytes after the historical A3
  and LWOH-L1 V2 freezes. Those records remain historical evidence, but neither
  frozen lineage is a valid source manifest for a new official experiment.
- LWOH-L1 V2 is closed as invalid procedure/source drift. It has no activation,
  admission, or online-learning artifact. `evaluate_a3_contingency` fails closed
  with `INVALID_A3_DO_NOT_ACTIVATE_LWOH` /
  `REPAIR_A3_DO_NOT_ACTIVATE_LWOH`, and reports `RuntimeError` when the terminal
  binding cannot be validated.
- The next official Gate-A attempt is a separately preregistered
  **post-A3/pre-LWOH V3**, frozen after the integrity repairs and before any LWOH
  metric is observed.

The generated `docs/CANONICAL_STATUS.md` and
`python -m adaptive_learning_substrate.artifact_validation verify` provide the
machine-checked current-status view. This section overrides stale current-tense
wording below while preserving the older sections as experiment history.

## Completed in the minimal scaffold

- Deterministic Python/NumPy package and pytest suite.
- Explicit fixed directed graph represented by an edge list.
- Immutable `UnitEvent`, `EdgeTrace`, and `CreditPacket` records.
- Local leave-one-message-out omission effect:
  \[
  o_e=a_j-\tanh(u_j-w_ex_e).
  \]
- Locally guarded weight and source secants, trace decay, trace expiry, clipped
  weight updates, and fixed-topology assertions.
- CCF-v0 plumbing for a single realized acyclic event ancestry.
- Same-topology eligibility-trace comparison.
- Exact three-bit `FLIP0`, `ROTL`, and `NOT` task adapters.
- JSON reports and JSONL audit events.
- A seeded 64-hidden-unit fixed sparse recurrent graph with one-tick delayed
  edges and multiple timestamped activations of the same unit.
- Balanced and disjoint Experiment-000 streams containing `CUE`, eight real
  `NOISE` events, `QUERY`, and target delivery only after prediction.
- A development-only runner for seeds 0–4 comparing CCF-v0,
  `CCF-NO-TRACE`, and deterministic uniform random while recording stream,
  topology, weight, compute-ledger, and episode hashes/data.
- A reconvergence regression test proving that small routed packets aggregate
  at their shared parent before the event-level credit threshold is applied.
- An exact fixed-event-DAG update-alignment instrument that differentiates
  shared temporal weights without autograd, validates against stratified finite
  differences, and compares CCF with an equal-norm local descent direction.
- A preregistered architecture-only recurrent-radius sweep with paired
  counterfactual probes, ridge decoding, activity accounting, long blank-tail
  stability checks, strict selection gates, and deterministic report hashes.
- A second preregistered architecture-only diagnostic: an experiment-only
  subclass adding one fixed lazy exponential per-hidden-unit local state, with an
  exact `lambda=0` native-equivalence proof, analytic micrographs, the same
  paired retention/activity/quiescence gates, both-polarity 256-tick blank-tail
  and wake assays, and a verified identical deterministic rerun.
- The A3 passive-readout-trace diagnostic on custom seeds 105--109, including a
  persisted smoke, two full runs, and
  `artifacts/experiment_000/readout_trace_a3/DETERMINISM_VERIFICATION.json`.
  The two deterministic payloads matched, the terminal result was
  `NO_SELECTION`, and `rho_0_95` was strongest but passed only 3/7 retention
  gates.

## What the current scores mean

The runnable scaffold learns a small, direct operation-routing task. Its
two-operation score chains two separately evaluated primitive predictions through
an external adapter. This checks that learned primitive mappings can be called in
sequence; it is **not** the preregistered single-episode compositional task.

The scaffold therefore validates implementation properties such as deterministic
execution, local updates, delayed trace decay, and stable topology. It does not test
the full research hypothesis.

The recurrent reduced-budget diagnostic (100 train + 100 evaluation episodes
per development seed) produced mean accuracy 0.482 for full CCF, 0.482 for
no-trace, and 0.484 for random. That is chance-level behavior, not a gate pass.
A clean 11-edge positive control did pass: full CCF repaired the only plastic
earlier edge and reached 1.0, while no-trace remained at 0.0. Together, these
results point to distributed recurrent memory/credit competition as the next
research problem rather than a total failure of timestamped routing.

The completed 100-episode update-alignment diagnostic sharpened that result.
Every CCF update had a positive dot product with exact local descent, and median
cosine was 0.516699176351386, but an equal-norm exact step reduced fixed-DAG
loss about 2.07 times as much. Both directions assigned nearly zero norm to cue
edges; CCF additionally concentrated 88.4% of its squared update norm in query
and noise inputs while the exact direction concentrated 79.0% in output edges.
The evidence therefore supports a combined forward-memory and credit-allocation
bottleneck, not a reversed update rule. All 3,200 derivative checks passed and
two full runs reproduced the same deterministic payload hash.

The subsequent recurrent-radius sweep completed twice on fresh custom seeds
50--54. Its deterministic payload SHA-256 was
`f6ba680f91917521da5bbfb89d7d1dbd102fe59149049d0ed0464bb3ece7f8e6`
in both runs. The result was `NO_SELECTION`: stronger recurrence amplified the
cue but no candidate passed all cue/noise, activity, and quiescence gates.
Radius 1.05 additionally failed the late blank-tail gate on four of five seeds.
Uniform recurrent rescaling is therefore rejected as the memory repair.

The slow local-state diagnostic then completed twice on fresh custom seeds
60--64. Its deterministic payload SHA-256 was
`0d95583e2e104a6c945fec0796051f9c198b4038982fa4a63d69d97989970e1e`
in both runs, verified by the frozen `verify_deterministic_full_runs` check. The
result was again `NO_SELECTION`, but with a sharper diagnosis. An ungated
additive exponential per-unit local state at retention 0.90 and 0.95 reached
fixed ridge readout 1.00 evaluation accuracy on all five seeds, but only
retention 0.95 cleared all seven preregistered retention gates; retention 0.90
failed the output gate (`O = 0.415954 < 0.50`). Both had median absolute cue
separation ~38--49x native and cue/noise ratio 0.75--0.86. Both failed stability
and budget: the ungated leak self-drives the network into ~0.96 blank-tick hidden
emission density (native ~0.015) so it never goes quiet, and it pushes matched
per-episode event activity to 1.17--1.18x (limit 1.15). The forward
representation is therefore not the sole bottleneck; the next intervention must
supply a bounded local state that is read at `QUERY` without becoming a
self-sustaining excitation source. See
[`docs/EXPERIMENT_000_SLOW_STATE_RESULT.md`](EXPERIMENT_000_SLOW_STATE_RESULT.md).

## Deliberately pending

- The 20-seed Experiment 000 confirmatory gate.
- Frozen source, environment, configuration, baseline, and analysis hashes
  required before confirmatory execution.
- A compressed streaming mechanism log capable of archiving canonical runs
  without retaining millions of event records in memory.
- ET-3F with a frozen random feedback matrix, BPTT, and BPTT-CM.
- All preregistered ablations and paired-bootstrap confidence intervals.
- Growth, pruning, and rewiring; these remain blocked until fixed-topology credit passes.
- Any novelty or superiority claim.

## Next engineering gate

Gate A is still open after three completed `NO_SELECTION` results: uniform
recurrent-radius scaling, ungated additive local state, and A3's passive
readout-only trace. The first was unstable at its strongest setting, the second
retained the cue but self-excited, and the stable A3 trace was too weak. In A3,
`rho_0_95` passed only 3/7 retention gates despite deterministic equality across
the full run and rerun.

The historical A1/A2 procedure-invalid records remain preserved, and the valid
A3 terminal record is
`artifacts/experiment_000/readout_trace_a3/DETERMINISM_VERIFICATION.json`.
Subsequent integrity repairs intentionally changed the current source lineage,
so the A3 freeze describes the historical run rather than current executable
bytes.

LWOH-L1 V2 is also closed. Its pre-terminal contingency freeze predates the
current source lineage, no activation receipt was committed, and no admission or
learning artifacts were produced. The fail-closed contingency outcomes are
`INVALID_A3_DO_NOT_ACTIVATE_LWOH` with action
`REPAIR_A3_DO_NOT_ACTIVATE_LWOH`, or `RuntimeError` when terminal validation
cannot establish a registered branch. V2 must not be resumed or relabeled.

The next step is to write a separate post-A3/pre-LWOH V3 protocol with fresh
development seeds, explicit equations and thresholds, registered failure
conditions, integrity-repaired source hashes, and a new phase ledger. The V3
protocol, configuration, tests, smoke contract, and source manifest must be
frozen before any LWOH admission or learning metric is observed. Credit-equation
changes remain a separate CCF-v1 experiment, and confirmatory seeds remain
disabled.
