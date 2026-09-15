# Experiment 000 architecture-only slow local-state result

**Status:** completed negative result; `NO_SELECTION`
**Date:** 2026-09-03
**Protocol:** `slow-state-v1-a1`
**Mechanism:** CCF-v0/C000.1 present but never called
**Seeds:** custom seeds 60--64; confirmatory seeds used: 0
**Scale:** 100 train pairs plus 100 evaluation pairs per seed and condition;
400 counterfactual forward episodes per seed and condition; 12,000 task forward
episodes total

## One-sentence conclusion

Giving every hidden unit a fixed, lazy, exponential local memory made the early
`CUE` strongly and linearly decodable at `QUERY` (ridge evaluation accuracy
reached 1.00 at retention 0.90 and 0.95), but the same ungated leak drove the
network into persistent self-firing that never settled and pushed matched
per-episode event activity past its budget, so the preregistered decision is
**`NO_SELECTION`**.

This is a useful, reproducible negative result. It is **not** an Experiment-000
learning pass, a CCF improvement, a novelty result, evidence of sparsity or
efficiency, or evidence of a market-ready system.

## What changed and what did not

Topology, every edge weight (input, recurrent, output), delays, event rules,
data protocol, and the CCF equation were left byte-identical to native. The
native recurrent module `src/adaptive_learning_substrate/recurrent.py` retained
SHA-256 `b56216abd19f5d3aaf935b6e58831c54ccf078ce4e4c7ef80e006b04ca6231b3`, and
the CCF method-source bundle and `docs/CCF_V0.md` retained their frozen hashes.
No learning occurred: all credit-event, credit-edge, credit-packet, and
weight-write counts were zero for every condition and probe.

The single architectural change was an experiment-only subclass
(`slow_state_recurrent.py`) in which each hidden unit `i` keeps two
episode-local scalars: its last processed activation `s_i in [-1, 1]` and the
tick `tau_i` at which that activation was computed. When a nonempty message
batch arrives at tick `t`, the unit uses

```
r_i(t) = lambda ** (t - tau_i) * s_i          (retained local state)
p_i(t) = u_i(t) + r_i(t)                       (u_i = usual weighted incoming drive)
a_i(t) = tanh(p_i(t))
s_i <- a_i(t),  tau_i <- t                     (local write)
```

`lambda = 0` recovers native activation exactly. Blank ticks trigger no state
read, decay, write, activation, or emission for an inactive unit; elapsed time
is represented by the one timestamp and an analytic decay. State is reset to
`+0.0` at every `begin_episode()` and never crosses an episode boundary. The
retained term creates no edge, message, or scheduled self-message, and no causal
trace was fabricated for it.

> **Simple explanation:** Each hidden unit was given a small private memory that
> fades by a fixed factor while the unit is idle and is consulted only when a
> real event reaches it. Nothing was taught. We compared five fixed fading
> speeds on the same paired examples and rejected any version that merely stayed
> active, saturated, fired more, or never went quiet after the input stopped.

## Conditions

| Condition | lambda | Half-life (ticks) | Ten-tick retention |
|---|---:|---:|---:|
| `native` | 0.00 | 0 | 0 |
| `lambda_0_25` | 0.25 | 0.500 | 0.00000095 |
| `lambda_0_50` | 0.50 | 1.000 | 0.00097656 |
| `lambda_0_75` | 0.75 | 2.409 | 0.05631351 |
| `lambda_0_90` | 0.90 | 6.579 | 0.34867844 |
| `lambda_0_95` | 0.95 | 13.513 | 0.59873694 |

## Exact frozen pass gates

A candidate could advance only if **every** rule below passed on the full
100-pair evaluation split (`C` = cue-feature RMS, `R` = cue/noise feature ratio,
`O` = output cue-delta / output SD, `A` = ridge evaluation accuracy; medians are
across the five seed-level values):

1. `median_s C_s,q  >=  3 * median_s C_s,native`;
2. at least 4 of 5 seeds with `C_s,q >= 3 C_s,native`;
3. `median_s R_s,q  >=  0.50`;
4. `median_s R_s,q  >=  3 * median_s R_s,native`;
5. at least 4 of 5 seeds with `R_s,q >= 3 R_s,native`;
6. `median_s A_s,q  >=  0.70`;
7. `median_s O_s,q  >=  0.50`;
8. every seed/split: total task event ratio `<= 1.10` and every matched-episode
   event-count ratio `<= 1.15`; the same limits on `forward_edge_touches`;
9. after each of `CUE(-1)` and `CUE(+1)` followed by 256 blank ticks, zero
   hidden emissions in ticks 193--256 for every seed and polarity, effective
   lazy state `max_i |s_i(256)| <= 1e-4`, and paired wake feature/output
   half-differences `<= 1e-3`;
10. every finite-value, tanh-range, topology, structural (`plastic`-inclusive),
    per-kind weight, spectral-radius/operator-norm, episode-reset,
    event-driven-touch, feature-identity, and no-credit invariant passes.

If several candidates passed, the rule selected the lowest numerical `lambda`.
If none passed, the required literal result was `NO_SELECTION`; promoting the
closest-looking candidate was forbidden.

## Full 100-pair result

Medians across the five seeds. `C mult` and `R mult` are the median of the
per-seed candidate/native ratios. `Act. guard` is gate 8; `Tail/wake` is gate 9.

| Condition | Median `C` | `C mult` | Seeds `C>=3x` | Median `R` | `R mult` | Seeds `R>=3x` | Ridge eval `A` | Median `O` | Blank-tail hidden density | Act. guard | Tail/wake | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| `native` | 0.016732 | 1.00 | 0/5 | 0.161556 | 1.00 | 0/5 | 0.595 | 0.170429 | 0.01465 | -- | pass | control |
| `lambda_0_25` | 0.014490 | 0.87 | 0/5 | 0.104669 | 0.65 | 0/5 | 0.675 | 0.117821 | 0.01984 | pass | pass | **fail** |
| `lambda_0_50` | 0.028994 | 1.73 | 0/5 | 0.114430 | 0.71 | 0/5 | 0.980 | 0.123020 | 0.03729 | pass | pass | **fail** |
| `lambda_0_75` | 0.223178 | 11.98 | 4/5 | 0.420253 | 2.69 | <4/5 | 1.000 | 0.240465 | 0.94257 | **fail** | **fail** | **fail** |
| `lambda_0_90` | 0.631405 | 37.74 | 5/5 | 0.751456 | 4.65 | 5/5 | 1.000 | 0.415954 | 0.96149 | **fail** | **fail** | **fail** |
| `lambda_0_95` | 0.813982 | 48.65 | 5/5 | 0.860618 | 5.33 | 5/5 | 1.000 | 0.549275 | 0.95935 | **fail** | **fail** | **fail** |

Task hidden emission density (hidden emissions / all hidden node-ticks on the
task) was already ~0.837 for native and stayed ~0.85--0.87 across candidates.
This experiment therefore supports no sparse-efficiency claim.

### Exact failed gates by candidate

- `lambda_0_25` -- fails gates 1--7. Retention is below native on the cue/noise
  ratio (`R mult` 0.65) and the absolute cue (`C mult` 0.87); the very short
  half-life discards the cue before `QUERY`. Activity and tail gates pass.
- `lambda_0_50` -- fails gates 1--5 and 7. Ridge accuracy alone reaches the
  threshold (0.98, gate 6), but absolute and relative cue separation remain far
  short (`C mult` 1.73 < 3, `R` 0.114 < 0.50, `R mult` 0.71 < 3, `O` 0.123).
  Activity and tail gates pass.
- `lambda_0_75` -- passes gates 1, 2, 6. Fails gate 3 (`R` 0.420 < 0.50), gate 4
  (`R mult` 2.69 < 3), gate 5, gate 7 (`O` 0.240), gate 8 (max matched-episode
  event ratio > 1.15), and gate 9 (blank-tail hidden density ~0.94; the network
  never goes quiet).
- `lambda_0_90` -- passes gates 1--6 (`C mult` 37.7, `R` 0.751, `R mult` 4.65,
  ridge 1.00, 5/5 on both threefold checks). Fails **gate 7** (`O` 0.416 <
  0.50), **gate 8** (max matched-episode event ratio 1.183 > 1.15; split-total
  ratio 1.063 was within limit), and **gate 9** (blank-tail hidden density
  ~0.961; persistent self-firing on every seed and polarity, effective state and
  wake checks fail).
- `lambda_0_95` -- passes gates 1--7 (`C mult` 48.6, `R` 0.861, `R mult` 5.33,
  ridge 1.00, `O` 0.549). Fails **gate 8** (max matched-episode event ratio
  1.166 > 1.15; split-total ratio 1.058 within limit) and **gate 9** (blank-tail
  hidden density ~0.959; the network never settles).

All frozen-state, finite-value, tanh-range, topology, structural, per-kind
weight, spectral-radius, operator-norm, episode-reset, event-driven-touch,
feature-identity, and no-credit invariants passed for every candidate. The
native control was valid. The global tail/effective-state/wake integrity field
is false only because candidates `lambda_0_75`--`lambda_0_95` failed that
candidate-specific stability assay.

## Scientific interpretation

The result is a clean signal-versus-stability tradeoff, and it is more
informative than the earlier recurrent-radius sweep.

1. **A per-unit local memory does carry the cue.** At `lambda_0_90` and
   `lambda_0_95`, a fixed linear ridge readout reached 1.00 evaluation accuracy
   on all five seeds, median absolute cue separation reached ~38--49x native,
   and cue/noise ratio reached 0.75--0.86 (native 0.16). Only `lambda_0_95`
   cleared all seven preregistered retention gates; `lambda_0_90` failed gate 7
   because `O = 0.415954 < 0.50`. The forward representation is not the only
   bottleneck.

2. **The specific mechanism is unusable as written for two independent
   reasons.** The retained term is added directly to the unit's pre-activation
   with no gate and no floor, so:
   - it re-drives the unit on subsequent ticks even with no new input, producing
     near-continuous autonomous firing (blank-tail hidden density ~0.96 versus
     native ~0.015) that fails the quiescence, effective-state, and wake gates;
   - it raises within-episode event activity above the strict matched-episode
     budget (1.15--1.18x versus the 1.15 limit) even though the loose
     split-total budget (1.10) was met.

3. **The two failures point at the same missing property:** the local state must
   inform the unit's readout *without* becoming a self-sustaining excitation
   source. Candidate designs that address this are gated / event-conditioned
   retention, a retained term with a hard dead-zone that collapses to exactly
   zero below a small floor, a leaky integrator that is read at `QUERY` but not
   fed back into the emission path, or a normalized (bounded-energy) state.

The supported conclusion is narrow:

> Under this graph, data protocol, and tested grid, an ungated additive
> exponential per-unit local state is not an acceptable memory repair, despite
> making the cue linearly decodable.

The result does **not** rule out local state timescales in general, gated event
memory, explicit bounded-decay variables, a readout-only memory path, or a
future CCF-v1 rule. CCF was deliberately inactive here.

## Reproducibility and deterministic rerun

Execution order followed Section 9 of the protocol: unit/integration tests
(22 passed) and analytic micrographs, then a two-pair smoke on all five seeds
and six conditions (all implementation/global/native integrity invariants
passed; `smoke_pairs_2.json`), then the 100-pair full run, then an independent
identical rerun to a second output path.

Both full runs produced deterministic payload SHA-256

`0d95583e2e104a6c945fec0796051f9c198b4038982fa4a63d69d97989970e1e`

The frozen determinism check (`verify_deterministic_full_runs`) recomputed each
report's payload hash from its own bytes, confirmed both self-hashes, confirmed
matching `status`, and returned terminal status `NO_SELECTION`. Its output is
saved at `artifacts/experiment_000/slow_state/DETERMINISM_VERIFICATION.json`.

The complete JSON reports are not byte-identical because machine/runtime
provenance is intentionally outside the deterministic payload. Recorded backend:
CPython 3.14.6, NumPy 2.5.1, float64, CPU, no autograd, no GPU. (The recorded
`runtime_seconds` for the primary run is not a clean wall-clock figure -- the
run spanned host sleep/resume cycles -- so it is reported only as
nondeterministic provenance and is not a benchmark.)

Artifacts:

- `artifacts/experiment_000/slow_state/custom_seeds_60_64_pairs_100.json`
  -- report SHA-256 `c1a9273e97da7b650a3b6daed7b1ef1df79f3ce4c3448159aec88e5609aa4f8d`;
- `artifacts/experiment_000/slow_state/custom_seeds_60_64_pairs_100_rerun.json`
  -- report SHA-256 `4ec4f74f0ed407793ea49c017f39b801722d4bb7c3b35b4602cdb032dbca2190`;
- `artifacts/experiment_000/slow_state/DETERMINISM_VERIFICATION.json`;
- runner source-provenance bundle SHA-256
  `6c10749c70b6ab0c59a08362c40033156ed4e69e7fac2f9cab1b440d70fce2d2`;
- frozen pre-run protocol bytes SHA-256
  `b33be9b4fffab70cf472720d17b04e710166cb0f719cf8e8d34595476d9cd74b`
  (this document is a separate post-result file, so the protocol file's hash is
  unchanged).

An earlier attempt on these seeds was invalidated before the freeze because
source changed mid-execution; see
`artifacts/experiment_000/slow_state/PREFREEZE_INVALIDATION.txt`. That attempt
was not used for selection and its numbers are not reported here.

## Decision and next controlled step

**Decision: `NO_SELECTION`.** The native architecture remains the project state.
No candidate -- including the strong-retention `lambda_0_90` and `lambda_0_95`
conditions -- may advance to an alignment or learning experiment.

Gate A of `docs/RESEARCH_READINESS_90.md` (select a stable memory architecture)
remains open after two attempts: uniform recurrent-radius scaling
(`NO_SELECTION`) and ungated additive local state (`NO_SELECTION`). The next
permitted step is a third separately preregistered architecture-only
intervention that keeps CCF fixed and adds a **gated or readout-decoupled**
bounded local state, evaluated on fresh custom seeds against the same frozen
retention, activity, quiescence, and determinism gates. It must pass every gate
before any learning claim.
