# Experiment 000 LWOH-L1 conditional learning protocol

**Protocol version:** `experiment-000-lwoh-l1-v1`  
**Status:** prospectively frozen contingency before any official A2 task metric  
**Intervention:** local write-once, hard-expiring hidden-state latch plus an
online tanh head  
**Claim level:** custom-seed development evidence only

## 1. Conditional activation and immutable boundary

This protocol is a contingency, not an additional A2 candidate. It activates
only if the complete A2 readout-trace procedure ends in a valid, independently
verified, deterministic terminal decision exactly equal to `NO_SELECTION`.
Activation requires valid A2 primary and separate-process rerun reports, valid
SHA-256 sidecars, canonical deterministic-payload equality, an intact phase
ledger and source freeze, and a strict verifier result. An incomplete, invalid,
aborted, source-drifted, or non-deterministic A2 run does not activate LWOH-L1;
A2 must be repaired instead.

If A2 ends in `SELECTED:<condition>`, this contingency is permanently
`CANCELLED_A2_SELECTED`, seeds 90--104 remain unused by LWOH-L1, and the frozen
A2-selected alignment/learning route retains seeds 75--89. The LWOH-L1 design
may inspect only the verified A2 terminal decision and integrity evidence. No
A2 condition value, rank, near-miss, effect size, or per-seed metric may select
or alter this mechanism, horizon, seed, sample size, baseline, threshold, or
analysis. This document and its configuration must be hashed before any A2
official metric is produced.

The previous protocols, results, artifacts, and seed assignments remain
immutable. LWOH-L1 uses no development seeds 0--4, alignment seeds 42--46,
memory-sweep seeds 50--54, slow-state seeds 60--64, invalid A1 seeds 65--69,
A2 seeds 70--74, A2-selected-branch seeds 75--89, or confirmatory seeds
1000--1019.

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
terminal result `LWOH_ADMITTED:write_once_ttl32` permits Stage L. Any failed
gate returns `LWOH_ADMISSION_FAIL:<gate_ids>`, does not generate seeds 95--104,
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
`SHA256(protocol_version|master_seed|cell|split|purpose)`. Episode and noise IDs
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

The only allowed order is:

1. freeze this protocol/configuration and all later runner/test/dependency
   sources before any official metric;
2. non-task unit, analytic, integration, and adversarial verifier tests;
3. nonselecting scratch smoke and smoke verification;
4. admission primary, admission rerun, admission deterministic verification;
5. only after `LWOH_ADMITTED:write_once_ttl32`, learning primary, learning
   rerun, and independent deterministic verification.

Canonical directory: `artifacts/experiment_000/lwoh_l1/`.

- `CONTINGENCY_FREEZE_RECORD.json`
- `PHASE_LEDGER.json`
- `admission_seeds_90_94_pairs_100.json` and `.sha256`
- `admission_seeds_90_94_pairs_100_rerun.json` and `.sha256`
- `ADMISSION_DETERMINISM_VERIFICATION.json`
- `learning_seeds_95_104.json` and `.sha256`
- `learning_seeds_95_104_rerun.json` and `.sha256`
- `DETERMINISM_VERIFICATION.json`

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

Any failure returns `LWOH_ADMISSION_FAIL:<gate_ids>` or
`LWOH_LEARNING_FAIL:<gate_ids>`, and the score remains exactly 21. Passing only
accuracy, only admission, or only one run earns no partial score.

Even a pass establishes only that a task-specific, first-arrival bounded latch
can support an online learned readout on this synthetic delayed-cue task and its
registered shifts. It is not evidence that CCF-v0 learned, that recurrent
weights received correct credit, or that the project has continual learning,
composition, structural plasticity, novelty, energy advantage, real-world
competitiveness, or product readiness. It does not authorize confirmatory seeds
or Experiment 001; Gate B and the remaining critical path stay closed.
