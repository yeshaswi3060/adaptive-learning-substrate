# Experiment 000 online trace learning protocol (TRACE-HEAD-L1)

**Frozen protocol:** `trace-head-l1-v1`  
**Purpose:** the first reproducible learning result against compute-matched and
non-learning baselines.  This is a custom-seed experiment rather than a
confirmatory or novelty claim.  Readiness can increase only after the complete
separate-process evidence gate passes.

## 1. Stop rule and prerequisites

Official execution is blocked until the compact A3 readout-trace procedure is
complete on the exact fresh custom seeds `105,106,107,108,109`.  The
implementation must verify the entire seven-stage A3 phase ledger, source freeze,
both full reports and their SHA-256 sidecars, the A3 determinism report,
canonical equality, independently verified execution freshness, and a terminal
status of the form `SELECTED:<condition>`.  The resource-invalid v1a2 procedure
cannot satisfy this prerequisite.  The selected retention is read from verified
A3 artifacts; it is never chosen by this learning run.

After A3 passes, the runner verifies the selected-trace head-alignment evidence:
`alignment_seeds_75_79_episodes_32.json`, its independent rerun, both SHA-256
sidecars, and `DETERMINISM_VERIFICATION.json` under
`artifacts/experiment_000/trace_head_alignment/`.  Both reports must use schema
`experiment-000-trace-head-l1-v1`, bind the exact selected A3 condition and
retention, pass their recorded integrity gates, and have canonically identical
deterministic payloads.  The verifier must establish a fresh process and end in
`TRACE_HEAD_ALIGNMENT_PASS`.  The older fixed-event alignment on seeds 42--46
is unrelated and is not an accepted prerequisite.  Failure stops before any
official seed is generated or official report path is created.

## 2. Frozen learner

The recurrent graph, topology, weights, emissions, and event stream remain
frozen.  A separate persistent head has exactly 16 float64 parameters and no
bias.  For a feature vector `z` and bipolar target `y`:

```text
q = tanh(theta^T z)
L = 0.5 * (y - q)^2
d = max(1, ||z||^2)
Delta theta = clip(0.5 * (y-q) * (1-q^2) * z / d, -0.05, +0.05)
theta = clip(theta + Delta theta, -3, +3)
```

`theta` starts at exact `+0.0`.  Coordinate and parameter clipping are
elementwise.  A score tie predicts `+1`.

The temporal contract is strict and audited once per example:

1. present cue and distractors (or an empty cue tick in the removal control),
2. present query and derive the 16-D feature vector,
3. make every prediction,
4. reveal the target,
5. update trainable online heads.

The target is unavailable to feature construction and prediction.  Evaluation
performs no parameter writes.  Each train and evaluation stream is traversed
once, with no replay.

## 3. Conditions and fairness

All conditions use the same ordered samples and predictions are paired within
seed.

- **`trace_head_v1`** — selected A3 passive trace features and true delayed
  training labels.
- **`native_head`** — the native instantaneous 16-D query messages with the
  same zero initialization, rule, clips, parameter count, feature count, and
  number of train updates.  It additionally executes a blind, faithful 64-state
  dummy trace path.  For every native hidden activation it repeats the exact
  node/tick-specific power-law decay, innovation, state/timestamp write, bound
  check, and history construction.  At query it repeats the 64-D effective
  snapshot, ordered 16-D feature gather, weighted dot, and `tanh`.  Its
  state/history/snapshot/output hashes and read, decay, write, reset,
  observation, and hidden-evaluation ledger must equal the candidate path
  exactly.  The dummy values never enter the native prediction, so this
  equalizes the complete extra representation mechanism without leaking the
  candidate representation.
- **`trace_independent_label`** — trace features but an independently generated,
  exactly balanced training-label permutation.  Its named PCG64 namespace is
  disjoint from data and all other baselines.
- **`frozen_trace_head`** — exact-zero trace head with no updates.
- **`rand`** — deterministic independent PCG64 predictions, with no fitted
  state.
- **`ridge_trace` / `ridge_native`** — diagnostic ridge fits using train
  features, train labels, train means, and train population standard deviations
  only.  Evaluation features are transformed and scored but never fitted.  The
  diagnostics do not select a condition.

For every example, the trace graph's native events and activations must equal a
literal native graph exactly.  A single non-mutating 64-D trace snapshot at the
query tick derives the ordered 16-D trace feature vector.  Graph weights and
topology never change.

## 4. Registered data cells

Official seeds are the exact ordered custom list `80,81,...,89`.  Exactly five
single-thread CPU workers are used and parent results are restored to seed
order.

| Cell | Distractors | Cue | Train | Eval |
|---|---:|---|---:|---:|
| `d8_clean` | 8 | visible | 512 | 512 |
| `d4_clean` | 4 | visible | 256 | 256 |
| `d16_clean` | 16 | visible | 256 | 256 |
| `d8_cue_removed` | 8 | empty tick | 256 | 256 |

Each registered seed deterministically derives a different named data root for
each cell while the frozen recurrent graph remains keyed by the registered
seed.  All streams are exactly balanced.  Identities and named RNG substreams
are disjoint across train/evaluation and across cells.  The frozen scratch smoke
uses custom seeds 900 and 901 with 2 train and 2 evaluation examples in each of
the four cells, and it must use a noncanonical scratch path.  Smoke is always
nonselecting.

## 5. Primary gates

All thresholds use evaluation accuracy and the seed as the independent unit.

1. Mean `trace_head_v1` accuracy on `d8_clean` is at least `0.75`.
2. At least 8 of 10 seeds reach `0.70` on `d8_clean`.
3. Mean paired advantage on `d8_clean` is at least `0.20` versus frozen,
   `0.10` versus native, and `0.15` versus independent-label.
4. Online trace accuracy is no more than `0.05` below the train-only trace
   ridge diagnostic.

The three predeclared D8 contrasts (trace minus frozen, native, and independent
label) additionally use all `2^10` exact one-sided sign flips.  Holm-adjusted
`p` must be strictly below `0.05` for every contrast.  A deterministic 20,000-
replicate seed bootstrap is computed for each mean paired difference; every
95% lower bound must be above zero.

## 6. Robustness and negative controls

- D4 mean trace accuracy is at least `0.75`; D16 mean is at least `0.65`.
- At least 8 of 10 seeds reach `0.60` in each of D4 and D16.
- The D16 mean drop from D8 is at most `0.15`.
- Aggregate trace accuracy in the D8 cue-removed cell is in `[0.45, 0.55]`.
- D8 independent-label and frozen-head aggregate accuracies are each in
  `[0.45, 0.55]`.

## 7. Integrity gates

Reports must establish disjoint sample IDs and RNG namespaces; one pass and no
replay; prediction before target reveal; zero evaluation writes; unchanged
recurrent weights, topology, and structural events; exact native/candidate
events and activations; finite values; bounded/reset traces; reconciled graph,
trace, and head ledgers; 16 parameters and 16 feature reads for both online
heads; exact five-worker ordered execution; and zero confirmatory seeds.

Before prerequisites or stream generation, the runner validates the external,
read-only, timestamped, self-hashed record
`configs/experiment_000_online_trace_learning.freeze.json`.  Its SHA-256
manifest covers this protocol, the complete TOML, implementation, tests, package
initializer, and the AST-derived transitive closure of every local imported
scientific source (including event definitions and the selected upstream A3 and
alignment implementations).  It also explicitly covers the A3 and alignment
protocols, complete configs, and tests, which are not Python imports.  Any
closure or byte drift stops the run.  The TOML
is validated as exact frozen bytes, not as a selected subset of fields, and the
freeze record plus its own file hash are bound into both full reports.  The
record is created exactly once only while official artifacts are absent and is
never regenerated after official work begins.  Its literal file SHA-256 is the
external preregistration root supplied in `TRACE_HEAD_L1_FREEZE_SHA256`; editing
and re-self-hashing the record cannot move that independent anchor.

Each cell persists canonical per-example evidence: registered episode and noise
identities, post-prediction target, feature/event/snapshot hashes, every
condition prediction, online activations, and theta transition hashes.  Full
report validation independently regenerates every registered stream and graph
observation, replays the frozen equation without calling the production cell
runner, refits both train-only ridge diagnostics, and reconstructs every metric,
head, ledger, transcript, aggregate, statistic, and gate.  Reported counts or
opaque hashes are never accepted as the source of a learning result.

Every report is canonical JSON plus a binding SHA-256 sidecar.  A primary and
independent rerun are archived, and the strict verifier revalidates both reports,
their sidecars, their prerequisite bindings, recomputes aggregates, statistics,
and gates, and requires canonical deterministic payload equality.  Coordinator
and five-worker PIDs, per-process tokens, process/job start times, self-hashes,
and coordinator-worker bindings must be internally valid within each archived
run and disjoint across the pair.  These archived fields are consistency
records, not independent attestations of historical origin.  The rerun binds
the immutable primary report and sidecar hashes.  Readiness instead relies on
verifier-observed process evidence: the terminal verifier launches two
simultaneous fresh subprocesses with unpredictable distinct nonces, observes
their operating-system PIDs, and makes each subprocess independently regenerate
and replay its bound full report.  Both nonce-bound reconstruction digests must
match for `D02_VERIFIER_ORCHESTRATED_REPLAYS` to pass.
The terminal verification also has a SHA-256 sidecar and rereads both input
reports, both sidecars, prerequisites, source manifest, and freeze record after
persistence before returning success.

If every statistical, performance, robustness, integrity, and determinism gate
passes, the terminal decision is `LEARNING_RESULT_PASS`.  That fully verified
result is the predeclared evidence gate that moves evidence-readiness from 21 to
25: +2 for delayed-credit learning and +2 for robustness.  Otherwise the
terminal decision is `LEARNING_RESULT_FAIL:<gate_ids>` and readiness remains
21.  No score changes before the separate-process deterministic verifier issues
the terminal result.

No official task metric is valid before this protocol, implementation, and
scratch-only contract tests are frozen.  Official evidence is intentionally not
executed as part of implementation.
