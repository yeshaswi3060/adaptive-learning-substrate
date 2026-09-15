# Experiment 000 readout-decoupled local-trace invalid-procedure record

**Status:** DRAFT — `INVALID_PROCEDURE_NONSELECTING`
**Date:** 2026-09-03
**Protocol:** `readout-trace-v1a`
**Mechanism:** CCF-v0/C000.1 present but never called
**Claimed seeds:** custom seeds 65--69; not reusable for a valid rerun
**Claimed scale, not verified:** 100 train pairs plus 100 evaluation pairs per seed and condition;
400 counterfactual forward episodes per seed and condition; 12,000 task forward
episodes total

> **Procedure invalidation:** A primary report and matching sidecar arrived only
> after this procedure was invalidated; the required earlier smoke, rerun, and
> determinism artifacts remain absent. The frozen configuration remained
> `execution.enabled=false`, required analytic/trace tests were incomplete, and
> the graph-subclass source changed after the primary report completed. Every
> number below is retained only as **unverified historical/preflight material**.
> It is excluded from design, selection, readiness scoring, and scientific
> claims. See
> `artifacts/experiment_000/readout_trace/PROCEDURE_INVALIDATION_65_69.txt`.

## One-sentence conclusion

The invalid procedure produced a table claiming perfect ridge decoding at
retentions 0.75, 0.90, and 0.95 and recurrent activity identical to native,
while also claiming sub-threshold native-output cue signal. Missing artifacts
and broken execution order mean none of those claims are evidence and no
candidate can be selected or rejected from them.

This document is not a completed result. It records an invalid, nonselecting
procedure so its numbers cannot silently influence a fresh experiment.

## What changed and what did not

The intended design kept topology, every edge weight, delays, event rules, the
data protocol, and the recurrent forward/emission path identical to native. The
native module `src/adaptive_learning_substrate/recurrent.py` had recorded SHA-256
`b56216abd19f5d3aaf935b6e58831c54ccf078ce4e4c7ef80e006b04ca6231b3`; the CCF
method bundle and `docs/CCF_V0.md` had frozen hashes. The missing raw artifacts
prevent independent verification of the claimed zero credit, credit-packet,
and weight-write counts.

The single change was an experiment-only subclass
(`readout_trace_recurrent.py`). On each native hidden activation `a_i(t)` the
unit updates a bounded exponential moving average of its own output:

```
m_i  <-  rho ** (t - kappa_i) * m_i  +  (1 - rho) * a_i(t)          |m_i| <= 1
```

The trace is never added to a pre-activation, never tested against the emission
threshold, never placed on an edge. At `QUERY` (forced-output tick `t_out`) the
16-dimensional feature vector is the decayed source-unit trace on each frozen
output edge, `z_e = rho ** (t_out - kappa_{j(e)}) * m_{j(e)}`, and the
candidate forced output is `tanh(sum_e w_e z_e)`. Under protocol amendment A1,
the `rho = 0` fast path is required to recover the native graph and **native
readout** with zero trace touches; this is not a claim that a decayed-trace
readout at `rho = 0` equals the native readout. The missing test/run artifacts
do not verify the claimed 40 counterfactual episodes.

> **Simple explanation:** every hidden unit kept a small fading average of its
> own recent activity. That average was *only looked at* when the question was
> asked -- it never fed back into the network, so the network behaved exactly as
> the untouched original. We then asked whether that fading average, seen
> through the existing output wiring, still carried the early cue.

## Conditions and gates

Six conditions (`native`, `rho_0_25`, `rho_0_50`, `rho_0_75`, `rho_0_90`,
`rho_0_95`), the same grid and the same frozen pass gates as `slow-state-v1-a1`:
median absolute cue RMS `C` at least `3x` native and `>= 3x` for at least 4 of 5
seeds; median cue/noise ratio `R >= 0.50`, `>= 3x` native, and `>= 3x` for at
least 4 of 5 seeds; median ridge evaluation accuracy `>= 0.70`; median output
ratio `O >= 0.50`; total and matched-episode activity within `1.10x` / `1.15x`
native; zero hidden emissions in blank ticks 193--256; effective trace and wake
half-differences within `1e-4` / `1e-3`; every finiteness, tanh-range, topology,
weight, `|m_i| <= 1`, immutable-`rho`, and no-credit invariant.

## Unverified historical/preflight table (nonselecting)

The following values were written as medians across five seeds. No underlying
report survives, so they are not verified results. `C mult` / `R mult` were
described as medians of per-seed candidate/native ratios.

| Condition | Ridge eval | `C` (cue RMS) | `C mult` | Seeds `C>=3x` | `R` (cue/noise) | `R mult` | Seeds `R>=3x` | `O` (output) | Blank-tail density | Matched-ep. activity | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| `native` | 0.535 | 0.016252 | 1.00 | 0/5 | 0.1602 | 1.00 | 0/5 | 0.1309 | 0.01471 | 1.000 | control |
| `rho_0_25` | 0.540 | 0.004357 | 0.27 | 0/5 | 0.1460 | 0.91 | 0/5 | 0.1999 | 0.01471 | 1.000 | **fail** |
| `rho_0_50` | 0.610 | 0.004193 | 0.26 | 0/5 | 0.0886 | 0.55 | 0/5 | 0.1209 | 0.01471 | 1.000 | **fail** |
| `rho_0_75` | **1.000** | 0.005539 | 0.34 | 0/5 | 0.0982 | 0.61 | 0/5 | 0.1638 | 0.01471 | 1.000 | **fail** |
| `rho_0_90` | **1.000** | 0.010248 | 0.63 | 0/5 | 0.2364 | 1.48 | 1/5 | 0.3860 | 0.01471 | 1.000 | **fail** |
| `rho_0_95` | **1.000** | 0.008606 | 0.53 | 0/5 | 0.3094 | 1.93 | 1/5 | 0.4838 | 0.01471 | 1.000 | **fail** |

The invalid output claimed that every candidate's recurrent blank-tail hidden
emission density, total task activity, and maximum matched-episode activity
were identical to native (ratio 1.000). Even if reproduced, that would mean
identical **recurrent activity**, not zero cost: candidate traces require extra
local reads, decay, writes, resets, and observations. Per-episode equality is
not independently confirmed because the raw rows are absent.

### Which gates each candidate reached

- `rho_0_25`, `rho_0_50` -- the historical table marks every retention gate as
  failed and describes absolute and relative separation below native.
- `rho_0_75` -- the table reports ridge accuracy 1.00 (gate 6) but marks the
  absolute-cue gates (`C mult` 0.34), all cue/noise-ratio gates (`R` 0.098,
  `R mult` 0.61), and the output gate (`O` 0.164).
- `rho_0_90` -- the table reports ridge accuracy 1.00, `R mult` 1.48 (1 of 5
  seeds `>= 3x`), `O` reaches 0.386. Still fails absolute-cue, ratio, and output
  gates.
- `rho_0_95` -- the table labels this strongest and reports ridge accuracy 1.00. `R` 0.309
  (`R mult` 1.93, 1 of 5 seeds `>= 3x`), `O` **0.484** -- just short of the 0.50
  output gate. `C mult` 0.53. Fails gates 1--5 and 7.

The invalid output also claimed that every invariant passed and the native
control was valid. Those assertions require the missing raw reports and are not
accepted as evidence.

## Hypotheses suggested by the invalid output

The table suggests a possible contrast with the two valid Gate-A attempts, but
it does not establish a third result:

| | attempt 2 (valid additive local state result) | invalid readout-trace numbers |
|---|---|---|
| Ridge decodability | 1.00 at `rho >= 0.90` | 1.00 at `rho >= 0.75` |
| Absolute cue magnitude (`C mult`) | 38--49x native (**pass**) | 0.3--0.6x native (**fail**) |
| Cue/noise ratio (`R`) | 0.75--0.86 (**pass**) | 0.10--0.31 (**fail**) |
| Blank-tail quiescence | ~0.96 density (**fail**) | 0.0147 = native (**pass**) |
| Matched-episode activity | 1.17--1.18x (**fail**) | 1.000x (**pass**) |

Attempt 2 passed on signal and failed on stability. The invalid table suggests
a mirror-image hypothesis: a trace that never feeds the forward path may retain
native recurrent activity while under-driving the output edges. A valid fresh
run must test that hypothesis.

Two questions remain open:

1. **Does a passive local trace reliably carry the cue?** The invalid table
   reports ridge evaluation accuracy 1.00 at `rho >= 0.75`, but fresh seeds and
   persisted reports are required before treating this as recoverable memory.

2. **Does the bounded EMA under-drive the native output preactivation?** The
   `(1 - rho)` input weight plus roughly ten ticks of decay between the cue and
   the readout predicts smaller `z_e` than an instantaneous native message. The
   invalid table's `O = 0.484` at `rho_0_95` is a design clue, not evidence.

The frozen absolute-magnitude and cue/noise-ratio gates were written for the
memory-sweep and slow-state families, where a candidate could make an
**unstable, densely self-firing** state look like memory. A readout-decoupled
trace is intended not to change those recurrent dynamics, although it still
performs extra trace operations. A valid procedure must determine whether the
existing gates measure readout gain and memory appropriately; the invalid ridge
numbers do not answer that question.

The non-evidentiary hypothesis retained for future testing is:

> A passive readout-only per-unit trace may preserve native recurrent activity
> while adding trace-operation cost, and may need a separately frozen readout
> gain to reach the output-signal threshold. Fresh evidence is required.

## Procedure and artifact failure

The claimed execution order is not accepted. No persisted passing smoke exists
before the alleged full run, the frozen config still says
`execution.enabled=false`, the promised analytic micrograph and trace-contract
tests were incomplete, and the runner/subclass changed after the numbers were
written.

One primary report and its matching sidecar arrived at 01:53:56. Its internal
payload hash is self-consistent, but its status is provisional pending an
identical full rerun and it was launched under the already-invalid procedure.
It therefore supplies no evidence. The following required artifacts remain
absent:

- `artifacts/experiment_000/readout_trace/smoke_pairs_2.json` and sidecar;
- `artifacts/experiment_000/readout_trace/custom_seeds_65_69_pairs_100_rerun.json`
  and sidecar;
- `artifacts/experiment_000/readout_trace/DETERMINISM_VERIFICATION.json`.

Consequently, there is no valid two-run deterministic result. The late
primary's verifiable file and payload hashes are audit metadata recorded in
`PROCEDURE_INVALIDATION_65_69.txt`, not scientific evidence. The literal
placeholders in the former draft were never evidence. The pre-label draft is preserved by identity in
`PROCEDURE_INVALIDATION_65_69.txt` with SHA-256
`1D10FAD4B4D8C23E310197B33108E1E846A0CC12FDB7B5AF6B94C7EEF7E8B5A5`.
`FREEZE_RECORD.json` exists, but a freeze record alone does not establish that
the required procedure ran.

## Decision and next controlled step

**Decision: `INVALID_PROCEDURE_NONSELECTING`.** This is neither a valid pass nor
a valid `NO_SELECTION` experiment. It has no effect on design, gate thresholds,
readiness score, or candidate selection. The native architecture remains the
project state.

Gate A remains open after the two valid `NO_SELECTION` attempts: uniform
recurrent-radius scaling and ungated additive local state. A new readout-trace
test must use fresh registered custom seeds disjoint from 65--69, freeze an
execution-enabled config and implementation, complete the analytic and trace
contract tests, persist a passing smoke before either full run, and retain both
full reports, sidecars, and independent determinism verification. Any gain or
accumulation variant must be specified before that fresh run; the invalid table
must not choose it.
