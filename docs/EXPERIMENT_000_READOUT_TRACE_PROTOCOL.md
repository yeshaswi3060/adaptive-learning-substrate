# Experiment 000 Architecture-Only Readout-Decoupled Local-Trace Protocol

**Status:** frozen before smoke or full run; A1 amended during implementation
before any task metric
**Protocol version:** `readout-trace-v1a`
**Freeze date:** 2026-09-03 (owner-approved: seeds 65--69, bounded EMA trace,
16-dimensional output-edge feature vector, reused `0.25..0.95` grid)
**A1 amendment:** 2026-09-03, feature contract and `rho = 0` equivalence scope
only; no equation, grid, seed, sample, gate, aggregation, or selection change;
no task metric observed
**Mechanism:** CCF-v0, implementation revision C000.1, present but never called
**Intervention:** one fixed passive per-hidden-unit activity trace, read **only**
at `QUERY`; the recurrent forward path is byte-identical to native
**Seed partition:** custom seeds 65--69 only; development (0--4), alignment
(42--46), memory-sweep (50--54), slow-state (60--64), and confirmatory
(1000--1019) seeds are forbidden

## 0. Why this experiment

Gate A of `docs/RESEARCH_READINESS_90.md` (select a stable memory architecture)
is open after two `NO_SELECTION` results:

1. uniform recurrent-radius scaling -- rejected; cue/noise ratio never clean
   enough, strongest radius fired persistently.
2. ungated additive per-unit local state (`slow-state-v1-a1`) -- rejected, but
   informative: at retention 0.90--0.95 the cue became **fully linearly
   decodable** (fixed ridge readout 1.00 evaluation accuracy on all five seeds,
   cue/noise ratio 0.75--0.86 versus 0.16 native), clearing every preregistered
   *retention* gate for the first time. It failed only on stability: the ungated
   leak, added straight into the pre-activation, re-drove units on later ticks
   and produced ~0.96 blank-tick hidden emission density (native ~0.015) plus
   1.17--1.18x matched-episode event activity (limit 1.15).

Attempt 2 shows a per-unit forward memory *can* carry the cue. The open question
is now narrow and specific:

> Can a **bounded local trace that never enters the recurrent/emission path** --
> so the network's spiking dynamics stay exactly native -- still expose the cue
> at `QUERY` strongly enough to clear the frozen retention gates?

If yes, Gate A closes with a mechanism whose stability is guaranteed by
construction. If no, we learn that the cue information the slow-state run
decoded existed only *because* the leak re-excited the network, which would
redirect Gate A toward gated event memory instead.

## 1. Question, claim boundary, and stop boundary

This is a **no-learning architecture-selection experiment**. A passing condition
demonstrates only that a fixed, passive, readout-only trace makes the early
`CUE` more linearly decodable at `QUERY` under identical intervening noise. It
cannot establish that CCF learns, that Experiment 000 is solved, that the design
is novel, or that it is sparse or efficient.

The native recurrent forward/credit module
`src/adaptive_learning_substrate/recurrent.py` must remain byte-identical to
SHA-256 `b56216abd19f5d3aaf935b6e58831c54ccf078ce4e4c7ef80e006b04ca6231b3`. The
CCF method-source bundle of `_credit_target`, `_sign`, `_record_packet`, and
`apply_supervised_credit` must retain SHA-256
`38378b7dc7d0b69644c9bd790a1f09d2ebfb9ec642da5136a2475f89f260607d` under the
byte-exact extraction defined in the slow-state protocol Section 1. The frozen
`docs/CCF_V0.md` SHA-256 is
`24907da979acef2ce95b6bb50ca52461d7b9d8b3bb6154fe871ae9000d7b7e9b`.

Terminal feedback never enters the graph, and the runner must never call
`apply_supervised_credit`. Candidate trace storage and extra audit counters live
in a new experiment-only subclass/wrapper module; the native class and builder
are used unchanged for the recurrent forward pass.

No exploratory pilot is authorized on seeds 65--69. A smoke run may expose only
software defects or invariant failures; its numerical task metrics must not
change any condition, threshold, aggregation, or sample size.

## 2. Frozen trace equation

The intervention applies to hidden units only, never to the three source units
or the scalar output unit. It does **not** change any activation, message,
emission decision, recurrent value, edge trace, or event that the native graph
produces. The native graph runs exactly as today.

Alongside the native forward pass, each hidden unit `i` keeps two episode-local
scalars: a passive trace `m_i in [-1, 1]` and the tick `kappa_i` at which it was
last updated. Before the unit has produced any activation, `m_i = +0.0` and
`kappa_i` is a sentinel.

Whenever the native forward pass computes a hidden activation `a_i(t) in [-1, 1]`
for unit `i` at tick `t` (that is, on exactly the ticks the native graph already
evaluates that unit), the trace is updated by a bounded exponential moving
average:

```
m_i  <-  rho ** (t - kappa_i) * m_i  +  (1 - rho) * a_i(t)
kappa_i  <-  t
```

For a positive processing gap `t - kappa_i >= 1`, the weights
`rho**(t-kappa_i)` and `(1-rho)` are each in `[0, 1]` and sum to at most `1`, so
`|m_i| <= max(|m_i_old|, |a_i(t)|) <= 1` always. `rho = 0` gives
`m_i = a_i(t)` on every processing tick. The implementation must not evaluate
`0 ** 0`; one-tick edges forbid same-tick recurrent arrival.

The trace is **write-only from the forward pass and read-only at `QUERY`**. It is
never added to a pre-activation, never compared against the emission threshold,
never placed on an edge, never scheduled as a message, and never used to
fabricate a causal trace. `begin_episode()` resets every `m_i` to `+0.0` and
every `kappa_i` to the sentinel before any input is accepted. State never
crosses an episode boundary.

### Lazy semantics

When unit `i` is not evaluated at tick `t` (native produced no activation for
it), the runner performs no trace read, decay, or write for `i`. Its
mathematically effective trace at any later observation tick `T` is

```
m_hat_i(T) = rho ** (T - kappa_i) * m_i .
```

For every nonzero-`rho` candidate, the first trace update after the episode
sentinel counts one logical read, one decay, and one write even though the
retained value is defined as zero. Resetting the 64 traces in `begin_episode()`
is administrative work recorded separately as 64 `local_trace_reset_touches`,
excluded from the forward-trace touch ledgers. Native and the `rho = 0`
equivalence fast path own no trace and report zero reset, observation, and
forward-trace touches.

## 3. Frozen conditions

Six conditions, in execution and report order:

1. `native` -- unmodified graph (identical to the `rho = 0` instance) with the
   native message readout;
2. `rho_0_25` -- `rho = 0.25`;
3. `rho_0_50` -- `rho = 0.50`;
4. `rho_0_75` -- `rho = 0.75`;
5. `rho_0_90` -- `rho = 0.90`;
6. `rho_0_95` -- `rho = 0.95`.

The grid is identical to `slow-state-v1-a1` so every condition can be compared
one-for-one against attempt 2. It must not be extended, interpolated, or
reordered after any smoke or full value becomes visible.

For every seed, all six conditions use exactly the native edge tuple, the native
float64 weight map, the native recurrent spectral radius, and the native
operator norm. The built-in `topology_hash()` and a full structural hash that
also includes each edge's `plastic` flag must both equal native before and after
every probe. Because this experiment does not touch the recurrent path, these
equalities are expected to be trivially true; a mismatch is an implementation
defect.

## 4. Frozen data and counterfactual pairing

Identical to `slow-state-v1-a1` Section 4, with the seed list changed:

- Official master seeds: exactly `65, 66, 67, 68, 69`, in that order, each
  required to have the `custom` role.
- Seeds 0--4, 42--46, 50--54, 60--64, and 1000--1019 are forbidden;
  confirmatory-seed count must be zero.
- Full run: 100 base noise streams in train and 100 independently generated base
  noise streams in evaluation, per seed. Smoke run: two base streams per split.
- Each base stream has eight bipolar noise events and is run twice from reset:
  once with `CUE(0)`, once with `CUE(1)`, identical noise.
- Train/evaluation RNG namespaces, identities, episode IDs, and noise IDs are
  disjoint. The train split only fits the fixed diagnostic ridge readout.
- A full seed/condition contains 400 counterfactual forward episodes; the
  official run contains 12,000 task forward episodes.

## 5. Frozen feature and output definition

**Amendment A1 (2026-09-03, during implementation, before any smoke or task
result).** The original v1 draft claimed `z_e` reduces to the native query-time
message at `rho = 0`. That is false: a decayed history term cannot equal an
instantaneous message when the source unit last fired more than one tick before
the readout. A1 corrects the feature contract and the `rho = 0` equivalence
scope. A1 changes no equation, grid value, seed, sample size, gate threshold,
aggregation, or selection rule. No readout-trace task metric had been observed.
Operative version: `readout-trace-v1a`.

### Native condition readout

The `native` control uses the **native readout unchanged**, exactly as
`slow-state-v1-a1`: the feature vector is the ordered vector of actual
hidden messages arriving on the frozen output edges at the forced-output event,
`0` where no message arrived, and the forced output is the native forced-output
activation. The `native` condition performs zero trace reads.

### Candidate condition readout

Let the 16 frozen output edges be `e_1..e_16` ordered by edge ID, with source
hidden unit `j(e)` and weight `w_e`. Let `t_out` be the forced-output tick. For a
candidate (`rho > 0`), the feature vector `z in R^16` is the source-unit trace
decayed to the readout tick:

```
z_e = rho ** (t_out - kappa_{j(e)}) * m_{j(e)}          (0.0 if j(e) never fired)
```

read without mutating `m` or `kappa` (counted as observation touches). The
candidate forced-output preactivation and activation used for every gate are
recomputed from these features:

```
p_output = sum_e ( w_e * z_e )
o        = tanh(p_output)
```

`C`, `R`, `O`, and ridge accuracy for a candidate are therefore computed in the
decayed-trace representation and compared against the same statistics computed
for `native` in its instantaneous-message representation -- the same
cross-representation comparison structure `slow-state-v1-a1` used between
modified-activation and native features. The gate question is: *does the
decayed-trace readout separate the cue markedly better than the native readout?*

### `rho = 0` graph-equivalence contract (scope corrected by A1)

The `rho = 0` instance must reproduce the **native graph** bit-for-bit: identical
step results, emitted event IDs (after episode-ID normalization), activations,
native query result, native-readout features, native forced output, and additive
legacy ledger deltas on the smoke examples, with **zero** trace read, decay,
write, reset, and observation touches (the `rho = 0` fast path owns no trace).
This is a check on the dynamics and the native readout; it is **not** a claim
that the decayed-trace readout equals the native readout.

**Diagnostic-only (never selects):** the runner also reports the decodability of
the full 64-dimensional trace vector `[m_hat_i(t_out)]_{i=1..64}` (its own ridge
fit and cue/noise ratio). This measures how much cue information the trace holds
in principle. It is recorded for interpretation and is excluded from every
selection gate.

## 6. Frozen measurements

Identical to `slow-state-v1-a1` Sections 6--7, evaluated on the 16-dim feature
vector `z` and scalar output `o` defined above:

- retained bipolar-cue RMS `C`, intervening-noise scale `N`, cue/noise ratio
  `R = C/N`, and output ratio `O` (output cue-delta / output SD), with the same
  `null` rules for degenerate denominators;
- a per-seed, per-condition fixed ridge readout (`alpha = 1e-3`, train-mean and
  population-SD standardization, inactive coordinates excluded exactly), applied
  to the disjoint evaluation split; report train and evaluation accuracy;
- per-episode deltas of `emitted_unit_events`, `forward_edge_touches`,
  `activation_evaluations`, hidden emitted events and density, plus the new
  `local_trace_read_touches`, `local_trace_decay_touches`,
  `local_trace_write_touches`, `local_trace_reset_touches`, and
  `local_trace_observation_touches`;
- hidden activation saturation fractions at 0.95, 0.99, 0.999 and maximum
  absolute preactivation (native values, unchanged; reported for continuity);
- the deterministic both-polarity 256-tick blank-tail and wake assay: inject one
  `cue` at tick 0, then 256 empty steps; record hidden emissions per tick, the
  vector hash, the last nonzero tick, and the tick 193--256 sum; at tick 256
  snapshot every `m_hat_i(256)` read-only; then read the condition's own readout
  (native-message readout for `native`, decayed-trace readout for a candidate)
  and compare the two cue polarities. Because a candidate trace decays over 256
  blank ticks, its effective-trace maximum and wake half-differences are expected
  to be far below the `1e-4` / `1e-3` gates.

Because the recurrent path is native, the task activity ledgers, hidden emission
counts, and blank-tail emission vectors are expected to equal the `native`
condition exactly for every candidate. The gates below still evaluate them; any
deviation means the trace leaked into the forward path and is an implementation
defect requiring a restart from smoke after correction.

## 7. Preregistered selection gates

Identical thresholds to `slow-state-v1-a1` Section 8. A candidate `q` passes only
if **all** hold (medians across the five seed-level values; `C_native`,
`R_native` are the native medians):

1. `median_s C_s,q  >=  3 * median_s C_s,native`;
2. at least 4 of 5 seeds with `C_s,q >= 3 C_s,native`;
3. `median_s R_s,q  >=  0.50`;
4. `median_s R_s,q  >=  3 * median_s R_s,native`;
5. at least 4 of 5 seeds with `R_s,q >= 3 R_s,native`;
6. `median_s A_s,q  >=  0.70`  (ridge evaluation accuracy);
7. `median_s O_s,q  >=  0.50`;
8. for every seed and split, total task event ratio `<= 1.10` and every
   matched-episode event-count ratio `<= 1.15`, and the same limits on
   `forward_edge_touches`;
9. after each of `CUE(-1)` and `CUE(+1)` followed by 256 blank ticks: zero
   hidden emissions in ticks 193--256 for every seed and polarity; effective
   trace `max_i |m_hat_i(256)| <= 1e-4`; paired wake feature and output
   half-differences each `<= 1e-3`;
10. every finite-value, tanh-range, topology, structural (`plastic`-inclusive),
    per-kind weight, spectral-radius, operator-norm, episode-reset,
    event-driven-touch, feature-identity, `|m_i| <= 1`, immutable-`rho`, and
    no-credit invariant passes.

If several candidates pass, select the **lowest numerical `rho`**. Native is a
control and is not selectable. Do not choose the best-looking failure,
interpolate, tune a threshold, inspect confirmatory seeds, or combine this grid
with any other intervention.

If no candidate passes, report exactly `NO_SELECTION`, retain the native
architecture, and record which of the retention gates (1--7) the strongest
candidate reached -- this determines whether Gate A next tries a gated event
memory or abandons per-unit local memory. A passing candidate may advance only
to a separately preregistered exact-alignment experiment on new custom seeds.

## 8. Execution, determinism, and rerun stop rules

Reject any seed list other than exactly `65,66,67,68,69`, any condition list or
order other than the six frozen conditions, and any pair count other than 2 or
100, before creating an output file. Reject omitted, duplicated, nonfinite,
out-of-order, or extra seed/retention values.

Execution order is fixed:

1. run unit/integration tests, including exact `rho = 0` equivalence and
   analytic micrographs (positive and negative trace over one-tick and
   multi-tick gaps; exact `rho**delta_t` decay; bounded `|m_i| <= 1`; no
   blank-tick update; episode reset and its separate 64-touch counter; a
   non-emitting processed activation still updates the trace; first-sentinel
   processing still records one read/decay/write; the native recurrent path,
   events, activations, and forced output are unchanged for every `rho`;
   effective-trace observation is non-mutating; the full structural hash changes
   if only a `plastic` flag changes);
2. run one two-pair smoke on all five seeds and all conditions;
3. only if smoke completes and every implementation/global/native integrity
   invariant passes, run the 100-pair full experiment;
4. independently rerun the identical full command to a second output path;
5. compare `deterministic_payload_sha256` exactly with
   `verify_deterministic_full_runs`.

If the two full payload hashes differ, report `NONDETERMINISTIC_INVALID`. If the
native control is invalid, report `INVALID_NATIVE_CONTROL`. Otherwise apply
Section 7 once and stop at `SELECTED:<condition>` or `NO_SELECTION`.

Use canonical JSON (sorted keys, compact separators, ASCII, `allow_nan=false`).
Hash arrays by shape plus contiguous little-endian float64 bytes. Runtime,
timestamps, absolute paths, and machine text stay outside the deterministic
payload. Across-seed quartiles use NumPy `quantile` linear interpolation at
probabilities 0.25, 0.50, 0.75.

## 9. Required report

As `slow-state-v1-a1` Section 10, with: the frozen native recurrent-source
whole-file hash proof; seed-role validation for 65--69 and forbidden-seed checks
for all five prior partitions; the `rho = 0` native-equivalence result including
forced-output equality; ordered output-edge IDs and raw paired array hashes; the
16-dim feature gates and the 64-dim diagnostic-only decodability; both-polarity
blank-tail vectors, effective-trace maxima, wake features/outputs, and every
tail/wake gate; candidate-minus-native seed rows with median/quartile/min/max
summaries; every individual gate, the passing-condition list, and exactly one
final status; the deterministic payload SHA-256 inside the report; and complete
report-file SHA-256 sidecars for the smoke, primary full, and rerun full
reports.

No value observed after this document is frozen may alter it.

## 10. Frozen choices (owner-approved 2026-09-03, before any run)

1. **Seeds:** exactly `65, 66, 67, 68, 69`, custom role.
2. **Trace form:** bounded EMA `m_i <- rho**(t-kappa_i) * m_i + (1-rho)*a_i(t)`,
   with the invariant `|m_i| <= 1`. The slow-state accumulate form is not used.
3. **Selecting feature vector:** the 16 output-edge source traces (Section 5),
   same dimension as the native readout, no new connectivity. The 64-dimensional
   per-unit trace vector is reported only as a non-selecting diagnostic.
4. **Retention grid:** `{0.25, 0.50, 0.75, 0.90, 0.95}`, identical to
   `slow-state-v1-a1`, for one-for-one comparison. A flat low end is an accepted,
   informative outcome.
