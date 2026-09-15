# Experiment 000 LWOH-L1 V3: Post-A3, Pre-Metric Protocol

**Status:** prospectively registered; no V3 admission or learning metric exists.
**Protocol version:** `experiment-000-lwoh-l1-v3`
**Claim boundary:** custom-seed development evidence only.

## V3 lineage and non-negotiable boundary

The independently verified A3 terminal is `NO_SELECTION`. The historical LWOH-L1 V2
pre-terminal freeze is not execution authority because its source manifest drifted after
the evidence-integrity repair. V2 produced no activation receipt, admission metric, or
learning metric. V3 is a separately preregistered post-A3/pre-LWOH lineage.

Seeds `90..94` (admission) and `95..104` (conditional learning) are explicitly
reaffirmed here before use because no LWOH metric has ever been produced from them.
Scratch seeds are exactly `9090,9091`. Historical A3 seeds `105..109` are forbidden
for V3, and confirmatory seeds `1000..1019` remain closed. No confirmatory seed is
authorized.

V3 changes only lineage, persistence, validation, and fail-closed execution controls.
Every scientific mechanism, equation, condition, sample size, Gate A threshold, learning
threshold, robustness threshold, statistical test, score rule, and claim limit in the
frozen V1 body below remains unchanged.

The authoritative repaired-lineage handoff is
`artifacts/evidence_integrity_repair_2026-09-04/PRE_METRIC_LINEAGE.json`, file SHA-256
`072f35110cc234abb12851cb67876c7a4e5f715cec8bc559195ddab73541d30b`, canonical
payload SHA-256 `3b54878cca73210218494499f1e68ce628e09d2c057c811fc766b437cc8a21d4`,
with adjacent sidecar file SHA-256
`83701e1aac04e28952479c9668bb02526f3fe9c8827cb2f2151ebe8e03c34215`.
It binds repaired-source manifest
`327234214bbaded98d98b9a7bf671eb263a008a18264f1b7319d47d2bf4c87f5`
and validator import closure
`dfe6d44343dc85790da1724c41fd41e7c4e22458faa44ca7d221a8eebfe52067`.

## Operative V3 scientific identity

The fourteen scientific configuration sections are byte-for-byte semantic copies of
the V1 TOML objects and are checked against their registered canonical-object hashes.
V3 adds only tables whose names begin with `v3_` plus persistence/source-control
tables; those additions cannot override a scientific value.

The scientific RNG namespace is exactly `experiment-000-lwoh-l1-v1`. Every scientific
stream seed is the first 128 bits of
`SHA256(experiment-000-lwoh-l1-v1|master_seed|cell|split|purpose)`. The V3 lineage
identifier is never an RNG operand. Any embedded reference below to `protocol_version`
in the seed equation denotes that exact frozen V1 namespace, not the V3 document
version.

The exact ordered condition identifiers are:

`lwoh_head`, `latest_head`, `lwoh_independent_label`, `frozen_lwoh_head`, `rand`,
`ridge_lwoh`, `ridge_latest`, `visible_cue_hold`.

No alias may replace or combine these identifiers in a raw report or the deep verifier.
The six readiness-only baseline aliases remain exactly `random`, `frozen_head`,
`latest_state`, `independent_label`, `ridge`, `visible_cue`, because the independent
global validator requires that compact readiness representation.

## Frozen V3 artifact contract

The canonical namespace is `artifacts/experiment_000/lwoh_l1_v3/`. Every JSON artifact
is canonical UTF-8 and has an adjacent binding `.sha256` sidecar. Object serialization
uses sorted keys, separators `(',', ':')`, ASCII escaping, and rejects NaN/infinity.
The JSON file appends exactly one LF byte. A record self-hash removes only its named
self-hash field and hashes the compact canonical object **without** that LF. Its sidecar
is itself compact canonical JSON plus one LF and contains exactly
`{"algorithm":"sha256","report_file":"<basename>.json","report_sha256":"<file sha256>"}`;
the sidecar report hash covers the complete report file **including** its LF.

The global chain uses these exact files:

1. `ACTIVATION_VERIFICATION.json` — schema
   `experiment-000-lwoh-l1-v3-activation-verification-v1`, status
   `ACTIVATED_A3_NO_SELECTION_V3`, and self field `verification_payload_sha256`;
   it binds A3 `NO_SELECTION`, V2 invalid/no-metrics status, repaired lineage, and
   untouched assigned seeds.
2. `PRE_METRIC_FREEZE.json` — freezes this protocol, configuration, runner, tests,
   transitive runtime dependencies, validator, repaired lineage, exact seed partitions,
   thresholds, paths, and the absence of all assigned-seed V3 metrics.
3. `FREEZE_VERIFICATION.json` — independently re-hashes and validates the freeze,
   source closure, prerequisites, seed absence, and fail-closed validator before tests
   or scratch work.
4. `ADMISSION_DETERMINISM_VERIFICATION.json` — accepts only two separate-process
   admission reports, valid sidecars, strict regeneration, identical deterministic
   payloads, exact A01--A10 recomputation, and an exact terminal.
5. `IMPLEMENTATION_SOURCE_FREEZE.json` — created only after exact admission pass and
   before learning; revalidates that the complete pre-metric source manifest is unchanged.
6. `LEARNING_DETERMINISM_VERIFICATION.json` — accepts only two separate-process learning
   reports, valid sidecars, strict regeneration, identical deterministic payloads, and
   recomputed primary/statistical/robustness/control/integrity gates.
7. `READINESS_VERIFICATION.json` — the only score-bearing terminal. Its schema is
   `experiment-000-lwoh-l1-v3-readiness-v1`; only exact
   `LWOH_LEARNING_PASS` with the complete valid chain sets readiness from 21 to 25.

Any missing, partial, raced, malformed, noncanonical, source-drifted, sidecar-mismatched,
wrong-seed, wrong-path, wrong-worker, wrong-order, or false-gate state remains 21.

Additional internally bound artifacts are `PHASE_LEDGER.json`, its hash-chained phase
receipts, `TEST_VERIFICATION.json`, the noncanonical scratch report/verification under
`artifacts/experiment_000/lwoh_l1_v3_scratch/`, primary and rerun
admission reports, and primary and rerun learning reports. The only allowed order is:

`activation -> freeze -> freeze_verification -> tests -> scratch_smoke ->
scratch_verification -> admission_primary -> admission_rerun ->
admission_verification -> implementation_source_revalidation -> learning_primary ->
learning_rerun -> learning_verification -> readiness_verification`.

An admission failure takes the sole shortcut
`admission_verification -> readiness_verification`, skipping steps 10--13. On this
branch `IMPLEMENTATION_SOURCE_FREEZE.json`, both learning reports, and
`LEARNING_DETERMINISM_VERIFICATION.json` (and their sidecars/receipts) must be absent.

Assigned-seed commands are blocked until the repaired-lineage handoff and global
V3-aware validator both verify. Learning is blocked unless admission returns exactly
`LWOH_ADMITTED:write_once_ttl32`. Any failed admission returns
`LWOH_ADMISSION_FAIL:<registered_gate_ids>`; any failed learning returns
`LWOH_LEARNING_FAIL:<registered_gate_ids>`. Neither earns partial points.

Verification `status` and the scientific `terminal` are separate fields. Admission
verification status is exactly `LWOH_ADMISSION_PASS` or `LWOH_ADMISSION_FAIL`; its
terminal is exactly `LWOH_ADMITTED:write_once_ttl32` or the admission-failure form.
Learning verification status is exactly `LWOH_LEARNING_PASS` or `LWOH_LEARNING_FAIL`;
its terminal is exactly `LWOH_LEARNING_PASS` or the learning-failure form. Failed gate
identifiers are emitted once in the exact registered family order below, never map,
locale, completion, or process order:

- admission: `A01,A02,A03,A04,A05,A06,A07,A08,A09,A10`;
- primary: `P01,P02,P03_frozen,P03_latest,P03_independent,P04`;
- statistics: `exact_assignments_1024,all_holm_adjusted_p_strictly_below_0_05,all_bootstrap_lower_bounds_positive`;
- robustness: `R01_d4,R01_d16,R02_d4,R02_d16,R03,R04,R05_independent,R05_frozen`;
- controls/integrity: `visible_cue_clean_exact_one,visible_cue_removed_exact_half,all_seed_integrity`.

Before any official worker is created, the runner must revalidate the complete freeze
and prerequisites and pass the registered host preflight: AC power, at least 8 GiB
available memory, at least 50 GiB free artifact-volume space, 60-second mean CPU no
greater than 20%, and pagefile use no greater than 10%. Each official primary and rerun
uses exactly five fresh single-threaded workers. Worker identity includes PID, a random
process-instance token, start nonce, run-start UTC, executable SHA-256, and source-
manifest SHA-256; the five identities are distinct, exclude the coordinator, and the
primary/rerun/coordinator/verifier identity sets are pairwise disjoint.

At every phase boundary publication is idempotent but immutable. If the canonical
report is absent, publish it exclusively. If it exists, continue only when its canonical
self-hash and bytes equal the exact would-be bytes. A missing sidecar may then be derived
only from that validated immutable report, and a missing receipt may be appended only
after report and sidecar validation. Existing canonical bytes are never recomputed,
replaced, or truncated. A crash inside a scientific phase restarts that phase from its
initial state in a fresh process. Standard output contains only terminal identifiers and
artifact hashes, never the raw report.

The post-admission `IMPLEMENTATION_SOURCE_FREEZE.json` is a revalidation, not a new
freeze: its complete source manifest and manifest hash must be byte-identical to those
in `PRE_METRIC_FREEZE.json`. The deep callback is exactly
`adaptive_learning_substrate.experiment_000_lwoh_l1_execution_v3.validate_official_v3_chain`.
It validates both raw admission reports and, when present, both raw learning reports;
regenerates streams and predictions; recomputes every gate from complete per-seed rows;
checks sidecars, phase receipts, source stability, exact condition/seed registries,
absence branches, replay, and process freshness; and returns only the exact key set
registered in the configuration. Extended deep-verifier facts never appear as extra
keys in `READINESS_VERIFICATION.json`.

The readiness record has exactly these nineteen keys and no others:
`schema_version`, `status`, `readiness_before`, `readiness_after`, `partial_score`,
`admission_seeds`, `learning_seeds`, `delays`, `baselines`,
`deterministic_admission`, `deterministic_learning`, `all_admission_gates_pass`,
`all_learning_gates_pass`, `all_robustness_gates_pass`,
`all_integrity_gates_pass`, `lineage_handoff_path`, `lineage_handoff_sha256`,
`chain_artifacts`, and `verification_payload_sha256`. Statistical, control, raw-row,
condition-ID, replay, and process-freshness details remain required in the stage records
and exact deep-verifier result but are not extra readiness keys.

The pre-metric source registry mechanically covers the runner's transitive local import
closure and the repaired validator closure. The latter is exactly the frozen recurrent
snapshot plus `artifact_validation.py`, `environments.py`, `events.py`,
`experiment000_data.py`, `experiment_000.py`, `experiment_000_memory_probe.py`,
`frozen_history.py`, `graph.py`, `learning.py`, `readout_trace_recurrent.py`, and
`recurrent.py`. The freeze additionally binds this protocol/configuration, runner/test,
V1 scientific source documents, dependency lock, CI workflow, relevant shared tests,
and lineage JSON/sidecar. Every prerequisite group is re-hashed before freeze, before
each official process, around each publication, and during independent terminal
verification; any absent, added, changed, or raced byte leaves readiness at 21.

## Frozen operational mathematics and observed integrity

The admission ratio aggregation is `median_of_per_seed_guarded_ratios`. Gate A07 uses
`per_seed_output_cue_delta_rms` as its numerator and `per_seed_output_sd` as its
denominator; the `zero_over_zero` decision is `fail`. Ridge uses
`Z_train_transpose_Z_train_over_n_plus_alpha_I` for its Gram equation,
`Z_train_transpose_y_train_over_n` for its right-hand side,
`mean_training_target` for its intercept, `exact_zero_scale_coordinate_inactive` for
zero-variance coordinates, and `numpy_linalg_solve` as its solver.

The exact sign-flip extreme tolerance is numerically zero. Bootstrap resampling uses
`seed` as the resample unit, shares one deterministic index matrix across contrasts,
uses the `percentile_95_two_sided_lower_endpoint` interval, takes lower quantile
0.025, and uses quantile method `linear`. These conventions clarify the already frozen
tests and thresholds; they do not tune or relax a scientific gate.

The runner persists the shared observer-kernel counters, per-condition compute
ledgers, parent and worker BLAS observations, peak-memory telemetry, five distinct
worker identities, complete edge-field commitments including `plastic`, native event
and activation equivalence, observed zero CCF calls, and prediction/target chronology.
Each integrity decision is recomputed from those observations rather than asserted.

---

# Frozen scientific body (V1 mechanisms and thresholds, operational V3 paths)
# Experiment 000 LWOH-L1 conditional learning protocol

**Protocol version:** `experiment-000-lwoh-l1-v3`  
**Status:** prospectively registered after verified A3 and before any V3 metric  
**Intervention:** local write-once, hard-expiring hidden-state latch plus an
online tanh head  
**Claim level:** custom-seed development evidence only

## 1. Post-A3 activation and immutable boundary

This protocol follows the completed A3 readout-trace procedure, whose valid,
independently verified deterministic terminal is exactly `NO_SELECTION`.
Activation binds the A3 freeze, phase ledger, primary report and sidecar,
separate-process rerun and sidecar, and terminal verification. The historical
V2 pre-terminal freeze is retained as negative provenance only: repaired core
sources no longer match its manifest, and V2 created no activation or metric.

V3 is prospectively registered against the repaired lineage. Seeds 90--104
remain untouched by any LWOH metric and are explicitly reaffirmed before use.
No A3 condition value, rank, near-miss, effect size, or per-seed metric selects
or alters this mechanism, horizon, seed, sample size, baseline, threshold, or
analysis. This document, its configuration, runner, tests, transitive sources,
validator, and repaired-lineage handoff must be hash-bound before any assigned-
seed V3 metric is generated.

The previous protocols, results, artifacts, and seed assignments remain
immutable. LWOH-L1 uses no development seeds 0--4, alignment seeds 42--46,
memory-sweep seeds 50--54, slow-state seeds 60--64, invalid A1 seeds 65--69,
A2 seeds 70--74, A2-selected-branch seeds 75--89, historical A3 seeds 105--109,
or confirmatory seeds 1000--1019.

LWOH-L1 tests one fixed mechanism; there is no candidate grid, tuning run, or
lowest-passing selection. A valid admission failure stops before learning
seeds are generated. A threshold or design change requires a new protocol
version and untouched new seeds.

## 2. Exact bounded local memory

For graph seed `s`, let `I_s` be the exactly eight hidden destinations of the
frozen `input:cue` edges, ordered lexically by hidden-node ID. Membership is
fixed by the graph topology before data generation and never depends on a
label, activation, or result. Let `v_i` be the frozen weight on
`input:cue->i`.

Each `i in I_s` owns episode-local state `(m_i, kappa_i, b_i)`, reset to exact
`(+0.0, 0, 0)`. Whenever the unchanged native graph actually evaluates hidden
unit `i` at tick `t` and produces native activation `a_i(t)`, use

```text
g_i(t) = 1 - b_i
m_i     = clip(g_i(t) * a_i(t) + (1-g_i(t)) * m_i, -1, +1)
kappa_i = g_i(t) * t + (1-g_i(t)) * kappa_i
b_i     = 1
```

Units not actually evaluated receive no touch. Thus each selected unit records
its first native activation and ignores later activations. In a clean episode,
the first evaluation of every selected unit is the one-tick response to the
learner-visible cue; no target is available then. The rule uses only a local
activation, local state, and event timestamp. It never reads the target.

The hard horizon is exactly `H = 32`. At a non-mutating observation tick `T`,

```text
z_i(T) = m_i  if b_i=1 and 0 <= T-kappa_i <= 32
         +0.0 otherwise
```

Thirty-two is the smallest power of two strictly larger than the maximum
official clean-query age, `D_max + 2 = 18` for `D_max=16`; it was fixed from
the event schedule, not a task metric. The logical state is exactly zero after
expiry. Stored implementation cache bytes may remain but are inaccessible and
must not count as effective state.

The eight-dimensional vector `z` is read only after the forced query output
exists. The latch never enters a hidden preactivation, emission test, edge,
message, `UnitEvent`, `EdgeTrace`, or recurrent update. It cannot create an
event or self-excitation. CCF-v0 is present but never called.

For the no-learning admission only, the fixed structural probe is

```text
p_struct = sum_{i in I_s} v_i * z_i
o_struct = tanh(p_struct)
```

It contains no learned parameter and is not used as the online learner or as a
superiority baseline. It verifies that the polarity held by the structural cue
recipients remains readable.

## 3. Online head

The persistent head has exactly eight float64 parameters and no bias. It is
initialized to exact positive zero. Given pre-target feature `z` and bipolar
target `y in {-1,+1}`:

```text
q           = tanh(theta^T z)
prediction  = +1 if q >= 0 else -1
L           = 0.5 * (y-q)^2
d           = max(1, ||z||^2)
Delta theta = clip(0.5 * (y-q) * (1-q^2) * z / d, -0.05, +0.05)
theta       = clip(theta + Delta theta, -3, +3)
```

Clipping is coordinatewise. Every training example follows exactly: present
cue and distractors; present query and form every feature; make every
condition's prediction; reveal the separately stored target; update trainable
heads. Each stream is traversed once with no replay. Evaluation performs no
parameter write.

Before an official run, synthetic and scratch-only tests must compare the
analytic descent direction with central finite differences on all eight
coordinates: every coordinate passes absolute error `<=1e-8` or relative error
`<=1e-6`, cosine is `>=0.999999`, and the analytic direction has positive dot
product with exact descent. These are implementation tests, not task metrics.

## 4. Conditions and compute fairness

All conditions consume the same ordered examples and are paired within seed.

1. `lwoh_head` uses `z` and true delayed training labels.
2. `latest_head` is the primary compute-matched comparator. It owns the same
   eight `(m,kappa,b)` records and calls the identical scalar kernel on every
   same native activation, but fixes `g_i(t)=1`, so it retains each selected
   unit's latest activation. It uses the identical head equation, initialization,
   clips, parameters, feature reads, predictions, and updates.
3. `lwoh_independent_label` uses LWOH features but an independently generated,
   exactly balanced training-label permutation.
4. `frozen_lwoh_head` has exact-zero parameters and performs no update.
5. `rand` emits deterministic independent PCG64 predictions.
6. `ridge_lwoh` and `ridge_latest` are diagnostics fitted only from D8 training
   features, labels, means, and population standard deviations, with
   `alpha=1e-3`; they never select a condition.
7. `visible_cue_hold` stores the one learner-visible cue bit and is a
   transparent non-learning task ceiling. With a removed cue it predicts `+1`.
   It is excluded from compute, efficiency, significance, and superiority
   comparisons.

`lwoh_head` and `latest_head` must have exactly equal native forward episodes,
event/activation identities, graph edge touches, observer-kernel calls, state
bytes, eight feature reads, eight parameters, head dot-product coordinates,
and training update coordinates. Both execute the same kernel arithmetic; the
baseline uses a fixed-one gate. No-op padding may be used only if recorded and
identical across all examples. Wall time and peak memory are reported but are
not substituted for exact logical ledgers.

## 5. Stage A: prospective stable-memory admission

Admission seeds are exactly `90,91,92,93,94`, in order, all with custom role.
Use delay 8, 100 base streams per split, and paired resets with cue `-1` and
`+1` under identical noise. Train data fits only the frozen ridge diagnostic;
there is no online update. Native control features are the latest activations
of the same eight `I_s` units at query, in the same order; its output is the
ordinary native forced output. Candidate features and output are `z` and
`o_struct`.

For paired features define `d_k=(z_{k,+}-z_{k,-})/2`, pair midpoint
`u_k=(z_{k,+}+z_{k,-})/2`, `C=sqrt(mean_k ||d_k||^2)`,
`N=sqrt(mean_k ||u_k-mean(u)||^2)`, and `R=C/N`. JSON never stores NaN or
infinity. The predeclared zero guard stores `(C,N,"positive_over_zero")` when
`C>0,N=0`; `R>=r` is then tested as `C>=r*N`, and candidate/native ratio tests
use cross multiplication. `C=N=0` fails every ratio gate.

The single candidate is admitted only if all frozen Gate-A requirements pass:

- `A01`: median evaluation `C` is at least `3x` native;
- `A02`: at least four of five seeds have `C >= 3*C_native`;
- `A03`: median evaluation `R >= 0.50`;
- `A04`: median evaluation `R` is at least `3x` native;
- `A05`: at least four of five seeds have `R >= 3*R_native`;
- `A06`: median train-only ridge evaluation accuracy is at least `0.70`;
- `A07`: median structural-output cue-delta/SD is at least `0.50`;
- `A08`: every seed/split has total event and forward-edge-touch ratios
  `<=1.10`, and every matched-episode ratio is `<=1.15`;
- `A09`: for both cue polarities on every seed, native hidden emissions are
  zero on blank ticks 193--256; effective LWOH state at tick 256 is exact zero;
  wake feature L2 half-difference and structural-output half-difference are
  each `<=1e-3`;
- `A10`: all finite, bound, reset, write-once, expiry, native-equivalence,
  topology, structural, frozen-weight, radius/norm, no-credit, data, ledger,
  source, phase, and provenance checks pass.

Run the complete admission twice in distinct processes and require valid
sidecars plus canonical deterministic-payload equality. Only the strict
verification status `LWOH_ADMISSION_PASS` together with terminal
`LWOH_ADMITTED:write_once_ttl32` permits Stage L. A failed gate produces status
`LWOH_ADMISSION_FAIL` and terminal
`LWOH_ADMISSION_FAIL:<registered_gate_ids>`, does not generate seeds 95--104,
and leaves readiness at 21.

## 6. Stage L: online learning and registered data

Learning seeds are exactly `95,96,97,98,99,100,101,102,103,104`, in order,
all with custom role and disjoint from admission. Exactly five single-threaded
CPU worker processes restore results to seed order. NumPy float64 is required;
GPU and autograd are disabled.

Each seed trains all trainable conditions exactly once on `d8_clean` with 512
balanced examples. The resulting heads are frozen and evaluated without
updates on independent cell/split namespaces:

| Cell | Distractors | Visible cue | Examples |
|---|---:|:---:|---:|
| `d8_clean_eval` | 8 | yes | 512 |
| `d4_clean_eval` | 4 | yes | 256 |
| `d16_clean_eval` | 16 | yes | 256 |
| `d8_cue_removed_eval` | 8 | no; exact empty cue tick | 256 |

For every `(master_seed,cell,split,purpose)`, derive a distinct PCG64 seed from
the first 128 bits of
`SHA256(experiment-000-lwoh-l1-v1|master_seed|cell|split|purpose)`. The literal
V1 namespace is a scientific constant; the V3 lineage identifier is never an RNG
operand. Episode and noise IDs
include cell and split. All streams are exactly balanced, and all train,
evaluation, cell, independent-label, random-prediction, and analysis namespaces
are disjoint. The analysis/bootstrap seed is exactly `2026090302`.

Scratch smoke uses only custom seeds `9090,9091`, at most eight balanced train
and eight balanced evaluation examples per cell, and a noncanonical scratch
path. Smoke is always nonselecting and may reveal only implementation faults.

## 7. Frozen learning gates

The seed is the independent unit. All thresholds are inclusive except the
stated p-value bound.

Primary D8 gates:

- `P01`: mean `lwoh_head` evaluation accuracy is at least `0.75`;
- `P02`: at least 8 of 10 seeds reach `0.70`;
- `P03`: mean paired advantage is at least `0.20` versus frozen, `0.10`
  versus latest, and `0.15` versus independent-label;
- `P04`: online LWOH accuracy is no more than `0.05` below `ridge_lwoh`.

For the three P03 contrasts, enumerate all `2^10` one-sided sign flips. Holm
adjustment across the three comparisons must give `p < 0.05` for every
contrast. A deterministic 20,000-replicate seed bootstrap uses analysis seed
`2026090302`; every 95% lower confidence bound on the mean paired difference
must be above zero.

Robustness and controls:

- `R01`: D4 mean LWOH accuracy is at least `0.75` and D16 mean is at least
  `0.65`;
- `R02`: at least 8 of 10 seeds reach `0.60` in each of D4 and D16;
- `R03`: the D16 mean drop from D8 is at most `0.15`;
- `R04`: aggregate LWOH accuracy with the D8 cue removed is in `[0.45,0.55]`;
- `R05`: aggregate D8 independent-label and frozen-head accuracies are each in
  `[0.45,0.55]`.

`visible_cue_hold` must be exactly 1.0 on every clean cell and exactly 0.5 on
the balanced cue-removed cell; this is a data/control integrity check, not a
capability comparison.

## 8. Integrity, persistence, and deterministic rerun

Every report must prove:

- prediction precedes the first target access for every condition and example;
- exactly one D8 training pass, no replay, and exactly 512 updates for each
  trainable head; final true-label heads differ from exact-zero initialization;
- evaluation performs zero writes and all theta hashes are unchanged across
  every evaluation cell;
- every memory write follows the exact local equation; at most one effective
  LWOH write occurs per selected unit; all values are finite and in bounds;
- hard expiry, episode reset, selected-node identity, lexical feature order,
  and feature dimension eight hold exactly;
- LWOH and latest conditions have exact compute-ledger equality and all native
  event, activation, topology, edge-field (including `plastic`), weight,
  recurrent-weight, spectral-radius, operator-norm, and structural hashes agree;
- zero CCF calls, credit packets, recurrent writes, topology changes, and
  confirmatory seeds;
- five unique fresh worker processes, ordered results, one BLAS thread in the
  parent and workers, CPU float64, and no GPU/autograd;
- cell/sample/RNG disjointness, exact balance, finite metrics, raw prediction
  and target hashes, and reconciled graph/memory/head ledgers;
- protocol, configuration, runner, tests, dependencies, and prerequisite
  artifacts match a pre-run freeze at run start, before persistence, and at run
  end.

Official reports are canonical JSON with binding SHA-256 sidecars. The learning
run is repeated from initial state in a separate process and writes a distinct
path. A strict verifier re-hashes sources and prerequisites, verifies both
sidecars and process freshness, regenerates every stream, recomputes predictions
where possible, recomputes all aggregates/statistics/gates from persisted raw
rows, and requires canonical deterministic-payload equality.

## 9. Required order and artifact paths

The operative sequence is:

1. activation verification;
2. pre-metric freeze;
3. independent freeze verification;
4. non-task unit, analytic, integration, and adversarial verifier tests;
5. nonselecting scratch smoke;
6. scratch verification;
7. admission primary;
8. admission rerun;
9. admission deterministic verification;
10. only after status `LWOH_ADMISSION_PASS` and terminal
    `LWOH_ADMITTED:write_once_ttl32`, byte-identical implementation-source
    revalidation;
11. learning primary;
12. learning rerun;
13. learning deterministic verification;
14. readiness verification.

If step 9 produces status `LWOH_ADMISSION_FAIL`, the only legal next step is step
14. Steps 10--13 and all of their artifacts, sidecars, and receipts remain absent.

Canonical directory: `artifacts/experiment_000/lwoh_l1_v3/`.

- `PRE_METRIC_FREEZE.json`
- `ACTIVATION_VERIFICATION.json`
- `FREEZE_VERIFICATION.json`
- `PHASE_LEDGER.json`
- `TEST_VERIFICATION.json`
- `artifacts/experiment_000/lwoh_l1_v3_scratch/smoke.json`
- `artifacts/experiment_000/lwoh_l1_v3_scratch/SMOKE_VERIFICATION.json`
- `admission_seeds_90_94_pairs_100.json` and `.sha256`
- `admission_seeds_90_94_pairs_100_rerun.json` and `.sha256`
- `ADMISSION_DETERMINISM_VERIFICATION.json`
- `IMPLEMENTATION_SOURCE_FREEZE.json` (post-admission unchanged-source revalidation)
- `learning_seeds_95_104.json` and `.sha256`
- `learning_seeds_95_104_rerun.json` and `.sha256`
- `LEARNING_DETERMINISM_VERIFICATION.json`
- `READINESS_VERIFICATION.json`

Every JSON in this list, the phase ledger, and every phase receipt has its own
adjacent canonical three-key sidecar. Raw admission reports use schema
`experiment-000-lwoh-l1-v3-admission-report-v1`; raw learning reports use schema
`experiment-000-lwoh-l1-v3-learning-report-v1`. Their deterministic scientific
payload excludes output path/name, primary/rerun label, coordinator/worker/verifier
identity, process times, runtime, peak memory, and publication metadata, while the
full report retains that provenance outside the deterministic payload.

Future canonical commands are fixed by the configuration. The runner must
reject wrong seeds, worker count, phase order, sample sizes, output paths, or
source hashes before generating official data.

## 10. Terminal decision and strict score rule

Only if every admission, primary, statistical, robustness, control, integrity,
source, phase, sidecar, and separate-process determinism gate passes may the
verifier return `LWOH_LEARNING_PASS`. Under the current readiness rubric, that
narrow development result moves the score from 21 to 25: delayed recurrent
development evidence rises from 2/10 to 4/10 (`+2`), and deterministic
delay/cue-removal robustness rises from 0/10 to 2/10 (`+2`). No other category
changes.

Any admission failure has verification status `LWOH_ADMISSION_FAIL` and scientific
terminal `LWOH_ADMISSION_FAIL:<registered_gate_ids>`. Any learning failure has
verification status `LWOH_LEARNING_FAIL` and scientific terminal
`LWOH_LEARNING_FAIL:<registered_gate_ids>`. The readiness status is the corresponding
unsuffixed verification status, and the score remains exactly 21. Passing only accuracy,
only admission, or only one run earns no partial score.

Even a pass establishes only that a task-specific, first-arrival bounded latch
can support an online learned readout on this synthetic delayed-cue task and its
registered shifts. It is not evidence that CCF-v0 learned, that recurrent
weights received correct credit, or that the project has continual learning,
composition, structural plasticity, novelty, energy advantage, real-world
competitiveness, or product readiness. It does not authorize confirmatory seeds
or Experiment 001; Gate B and the remaining critical path stay closed.
