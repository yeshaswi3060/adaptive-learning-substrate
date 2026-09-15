# Experiment 000 Readout-Decoupled Local Trace: A2 Recovery Protocol

**Status:** prospective A2 recovery; frozen scientific design before any
seed-70--74 task metric  
**Protocol version:** `readout-trace-v1a2`  
**Mechanism:** CCF-v0/C000.1 is present but is never called  
**Intervention:** bounded passive per-hidden-unit EMA trace, read only by the
candidate output at `QUERY`; native recurrent dynamics remain unchanged  
**Official A2 seeds:** custom seeds 70--74 only  
**Backend:** NumPy CPU float64; five ordered seed workers; GPU disabled

## 0. Why A2 exists

The original `readout-trace-v1a` scientific design used custom seeds 65--69.
A full-size report on those seeds became visible before the protocol's required
persisted smoke report and machine smoke verification existed, while its frozen
configuration still contained `execution.enabled = false`. That execution did
not satisfy the frozen phase order and cannot select an architecture.

The exposed v1a report is classified exactly as:

`INVALID_PROCEDURE_NONSELECTING`

Its observed primary report had deterministic payload SHA-256
`ab5d009ea3afdb8dced17919b255da01e7f8c309af9b2e371eb3b2963c5053cb`
and report-file SHA-256
`33bdf52cfdac5f30f797a5a75b034a8734f60729e42336fc91eff44ebe3b2d5a`.
Those hashes identify the invalid record; they do not make it an official
result. Seeds 65--69 are permanently excluded from A2, and none of their task
metrics may pass a gate, choose a condition, change a threshold, or justify a
claim.

A2 is a prospective replication on untouched custom seeds 70--74. It preserves
the v1a equation, condition grid, native and candidate feature definitions,
sample sizes, measurements, numeric gates, aggregation, and lowest-passing
selection rule. Its changes are procedural only:

1. execution is explicitly enabled, but only after the implementation has
   passed non-task tests and a complete freeze record exists;
2. every phase has machine-verifiable artifact prerequisites;
3. a smoke run can never select a condition;
4. source hashes are checked at run start and run end;
5. five CPU workers execute independent seeds while report order remains fixed;
6. the downstream credit limitation is explicit.

No result from v1a is imported into A2 analysis.

## 1. Question and claim boundary

Does a fixed, bounded, readout-decoupled local trace expose the early cue at
`QUERY` strongly enough to pass the unchanged retention gates while preserving
the native recurrent event dynamics exactly?

This is a no-learning architecture-selection instrument. A pass would establish
only that the frozen passive trace/readout representation retains the cue under
this task and activity contract. It would not establish delayed learning,
continual learning, composition, CCF effectiveness, novelty, sparsity,
compute/energy superiority, or product readiness.

Terminal feedback never enters the graph. The runner must never call
`apply_supervised_credit`. All credit-event, credit-edge, credit-packet, and
weight-write counts must remain zero.

## 2. Frozen trace equation

For hidden unit `i`, retain episode-local scalar `m_i` and its last update tick
`kappa_i`. Before the first hidden activation in an episode:

```text
m_i = +0.0
kappa_i = sentinel
```

Whenever the unchanged native forward pass evaluates that hidden unit and
produces activation `a_i(t)`, a nonzero-retention candidate performs:

```text
m_i <- rho ** (t - kappa_i) * m_i + (1 - rho) * a_i(t)
kappa_i <- t
```

The retained term is defined as zero on the first update after the sentinel.
The first update still records one logical trace read, one decay, and one write.
There is no evaluation of `0 ** 0`.

For a read-only observation at tick `T`:

```text
m_hat_i(T) = rho ** (T - kappa_i) * m_i
```

and `m_hat_i(T) = 0.0` if the unit has never been evaluated. Because
`0 <= rho <= 1`, `|a_i| <= 1`, and the update coefficients sum to at most one,
every stored trace must satisfy `|m_i| <= 1`.

The trace:

- applies only to hidden units;
- is updated only after a real native hidden activation evaluation;
- receives no blank-tick update for an inactive unit;
- never enters a hidden preactivation or emission decision;
- never creates an edge, message, scheduled self-event, `UnitEvent`, or
  `EdgeTrace`;
- resets to positive zero at every `begin_episode()` and never crosses an
  episode boundary;
- is immutable except for the equation above and episode reset.

`rho = 0` uses the native fast path, owns no trace state, performs zero trace
touches, and must reproduce native graph dynamics and the native message readout
bit-for-bit.

## 3. Frozen conditions and data

Conditions, in exact execution and report order:

1. `native` -- unmodified graph and native instantaneous-message readout;
2. `rho_0_25` -- `rho = 0.25`;
3. `rho_0_50` -- `rho = 0.50`;
4. `rho_0_75` -- `rho = 0.75`;
5. `rho_0_90` -- `rho = 0.90`;
6. `rho_0_95` -- `rho = 0.95`.

The grid must not be extended, interpolated, reordered, or tuned.

Official seeds are exactly `70, 71, 72, 73, 74`, in that order, and each must
resolve to the `custom` role. The following partitions are forbidden:

- development seeds 0--4;
- alignment seeds 42--46;
- recurrent-radius seeds 50--54;
- slow-state seeds 60--64;
- invalid v1a readout-trace seeds 65--69;
- confirmatory seeds 1000--1019.

For every seed:

- full run: 100 base noise streams in `train` and 100 independently generated
  base noise streams in `eval`;
- smoke: two base streams per split;
- each base stream contains eight bipolar noise events;
- each base stream is run from reset with cue 0 and cue 1, using identical
  intervening noise inside the pair;
- every condition reuses the exact same paired stream manifests;
- train and evaluation RNG namespaces, sample identities, episode IDs, and
  noise IDs remain disjoint;
- train is used only to fit the fixed diagnostic ridge readout;
- target and terminal feedback remain hidden from the graph.

A full seed/condition has 400 forward episodes. The complete A2 full run has
12,000 forward episodes. These sample sizes are fixed.

## 4. Frozen graph, feature, and output definitions

Use the existing Experiment-000 graph: 64 hidden units, recurrent in-degree 8,
input fan-out 8, output fan-in 16, one-tick edges, two-tick query readout latency,
initial scale 0.35, `tanh`, float64, and emission threshold `1e-3`. Topology,
every edge field including `plastic`, every weight, spectral radius, operator
norm, delay, event rule, and native hyperparameter remain unchanged.

### 4.1 Native control

The native feature vector is the lexically ordered 16-vector of actual hidden
messages arriving on the frozen output edges at the forced-output event, with
zero for an edge that supplied no message. The native output is the ordinary
forced-output activation. Native performs zero trace reads.

### 4.2 Candidate trace readout

Let the 16 frozen output edges be `e_1..e_16` in lexical edge-ID order. Let
`j(e)` be the source hidden unit and `w_e` the unchanged edge weight. At forced
output tick `t_out`:

```text
z_e = rho ** (t_out - kappa_j(e)) * m_j(e)
p_output = sum_e w_e * z_e
o = tanh(p_output)
```

`z_e = 0.0` if `j(e)` was never evaluated. The read is non-mutating. Candidate
retention measurements use this 16-dimensional trace representation and the
candidate output `o`; native measurements use the native representation above.
The feature dimension and edge IDs are identical, but the temporal readout
semantics intentionally differ.

The complete 64-dimensional effective trace may be reported as a diagnostic
with its own ridge fit. It is excluded from every selection gate.

## 5. Frozen measurements

For paired features `z_k,0` and `z_k,1`, define:

```text
d_k = (z_k,1 - z_k,0) / 2
m_k = (z_k,1 + z_k,0) / 2
C = sqrt(mean_k ||d_k||_2^2)
N = sqrt(mean_k ||m_k - mean(m)||_2^2)
R = C / N
```

For candidate outputs `o_k,0`, `o_k,1`:

```text
delta_k = (o_k,1 - o_k,0) / 2
O = sqrt(mean_k delta_k^2) / SD({o_k,0, o_k,1})
```

A denominator at or below `1e-15` produces `null`, not infinity or NaN, and
fails its corresponding selection gate.

For every seed and condition, fit the unchanged ridge diagnostic on train
features only. Encode cue 0 as -1 and cue 1 as +1; standardize using train mean
and population standard deviation; set zero-variance coordinates to zero and
exclude them from the solve; use `alpha = 1e-3`; evaluate once on the disjoint
evaluation split.

Record raw paired features and outputs, `C`, `N`, `R`, `O`, ridge train/eval
accuracy, activations and saturation diagnostics, emitted unit/hidden events,
hidden density, activation evaluations, forward edge touches, and all trace
read/decay/write/reset/observation touches.

### 5.1 Blank-tail and wake assay

For each seed, condition, and cue polarity `-1,+1`:

1. reset the episode;
2. inject the cue at tick 0;
3. execute exactly 256 empty native steps;
4. store every hidden-emission count for ticks 1--256;
5. observe the effective trace at tick 256 without mutation;
6. issue the ordinary wake query and read the condition's own feature/output.

Report the complete hidden-count vector/hash, last nonzero tick, sum over ticks
193--256, effective-trace maximum, both wake feature vectors/outputs, their
paired half-differences, and all trace touches.

## 6. Unchanged selection gates

Only the full 100-pair result may select. For candidate `q`, all ten groups must
pass:

1. median evaluation `C_q >= 3 * median C_native`;
2. at least four of five seeds have `C_s,q >= 3 * C_s,native`;
3. median evaluation `R_q >= 0.50`;
4. median evaluation `R_q >= 3 * median R_native`;
5. at least four of five seeds have `R_s,q >= 3 * R_s,native`;
6. median ridge evaluation accuracy is at least `0.70`;
7. median `O` is at least `0.50`;
8. for every seed and split, total emitted-event ratio and total forward-edge-
   touch ratio are each at most `1.10`, while every matched-episode event ratio
   and forward-touch ratio is at most `1.15`;
9. both cue polarities on every seed have zero hidden emissions during blank
   ticks 193--256, effective trace maximum at tick 256 at most `1e-4`, wake
   feature half-difference L2 at most `1e-3`, and wake output half-difference
   absolute value at most `1e-3`;
10. every required implementation, finite-value, tanh/trace-bound, topology,
    structural, frozen-weight, radius/norm, episode-reset, feature-identity,
    event-driven-touch, recurrent-equivalence, no-credit, provenance, phase,
    and source-integrity invariant passes.

Medians are across the five seed-level values. The threefold checks also report
within-seed ratios and require four of five successes. Nulls are never dropped
and fail the relevant gate.

Native is a control and is not selectable. If several candidates pass, select
the lowest numerical `rho`. A smoke report is ineligible by construction:
`eligible = false`, `pass = false`, `passing_conditions = []`, and
`selected_condition = null`, even when its descriptive numbers happen to exceed
all thresholds. If no full candidate passes, report exactly `NO_SELECTION`.

## 7. Fixed execution-only parallelism

A2 uses exactly five CPU worker processes and no GPU. Each worker owns one seed,
one graph at a time, and its seed-namespaced deterministic streams. Within a
worker the condition order remains `native, 0.25, 0.50, 0.75, 0.90, 0.95`.
Workers share no graph, RNG, mutable accumulator, or output file.

The coordinator must return and serialize seed results in exact order
`70,71,72,73,74` regardless of completion order. Conditions, pairs, and rows
are likewise serialized in their frozen order before aggregation and hashing.
Parallel completion timing, process IDs, and wall-clock runtime are excluded
from the deterministic payload. Five-worker and one-worker execution must have
identical deterministic payloads in a non-official parity test; the official
smoke/full/rerun all use five workers.

Each worker uses NumPy float64 on CPU with numerical-library thread count one
(`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and
`NUMEXPR_NUM_THREADS=1`) to avoid nested oversubscription. `gpu_used` is frozen
to `false`.

Parallelism changes only execution scheduling. It may not change an equation,
stream, seed, condition, feature, result order, aggregation, gate, or hash.

## 8. Mandatory phase order and artifact prerequisites

The only valid order is:

`tests -> freeze -> persisted smoke + sidecar -> machine smoke verification -> full -> rerun -> determinism verification`

### Phase 1 -- tests

Before any seed-70--74 task metric, tests must cover:

- signed positive and negative EMA updates over one- and multi-tick gaps;
- exact first-sentinel read/decay/write accounting;
- a processed but non-emitting activation updating the trace;
- blank ticks producing no inactive-unit trace update;
- positive-zero episode reset, sentinel restoration, and exactly 64 reset
  touches for a nonzero candidate;
- `rho=0` bitwise native dynamics/readout with zero trace touches;
- effective-trace observation being non-mutating;
- full structural hash detecting a `plastic`-only change;
- exact reconstruction of candidate output from the ordered 16 features;
- recurrent events/activations/ledgers remaining native for every `rho`;
- smoke being mathematically unable to select;
- full/rerun refusal when any prerequisite artifact or hash is absent/wrong;
- deterministic ordered five-worker versus one-worker parity.

The complete unit/integration suite and configured lint/static checks must pass.
Tests may use analytic micrographs and non-official scratch seeds, never task
metrics from seeds 70--74.

### Phase 2 -- freeze

After tests pass and before smoke, create
`artifacts/experiment_000/readout_trace_a2/FREEZE_RECORD.json`. It must record:

- A2 protocol/config hashes;
- A2 runner, trace wrapper, A2 tests, CCF specification, native recurrent source,
  data generator, feature/metric helpers, and every imported scientific helper;
- a canonical source-manifest hash;
- exact environment/dependency identity;
- correct UTC timestamp plus local timezone;
- `execution_enabled = true`, worker count 5, CPU float64, GPU false;
- the invalid-v1a classification and report hashes above;
- exact phase order and all output paths.

Any scientific source or dependency change after this freeze invalidates A2 and
requires a new protocol version and fresh seeds.

### Phase 3 -- persisted smoke and sidecar

Run exactly two pairs per split on all five A2 seeds and all six conditions,
with five ordered workers. Persist:

- `artifacts/experiment_000/readout_trace_a2/smoke_pairs_2.json`;
- `artifacts/experiment_000/readout_trace_a2/smoke_pairs_2.sha256`.

The runner must validate arguments, execution enablement, freeze record, and
start-source hashes before creating the report. It must re-hash all frozen
sources at run end before writing. Smoke is descriptive and cannot select or
change any design field.

### Phase 4 -- machine smoke verification

Create `artifacts/experiment_000/readout_trace_a2/SMOKE_VERIFICATION.json` only
after independently verifying:

- report-file sidecar and deterministic self-hash;
- run kind `smoke`, seeds/order, conditions/order, two pairs/split, five workers,
  CPU float64, GPU false, and confirmatory count zero;
- `selected_condition = null`, empty passing list, and smoke-ineligible status;
- every native/global/source/topology/weight/no-learning/recurrent-equivalence
  invariant;
- exact start/end source-manifest equality to the freeze record;
- required raw rows, manifests, and array hashes.

The verification file contains its own canonical verification payload hash and
the verified smoke/sidecar hashes. A false or absent check blocks full execution.

### Phase 5 -- primary full run

The full runner must refuse before output creation unless the freeze record,
smoke report, smoke sidecar, and successful machine verification all exist and
match their bound hashes. Then run 100 pairs per split and persist:

- `artifacts/experiment_000/readout_trace_a2/custom_seeds_70_74_pairs_100.json`;
- `artifacts/experiment_000/readout_trace_a2/custom_seeds_70_74_pairs_100.sha256`.

The primary status is provisional until the independent rerun verifies.

### Phase 6 -- independent rerun

The rerun must refuse unless the verified primary report and sidecar exist and
the complete frozen source manifest still matches. Run the identical command,
data, worker count, and ordered serialization to a distinct path:

- `artifacts/experiment_000/readout_trace_a2/custom_seeds_70_74_pairs_100_rerun.json`;
- `artifacts/experiment_000/readout_trace_a2/custom_seeds_70_74_pairs_100_rerun.sha256`.

### Phase 7 -- deterministic verification

Independently recompute both report-file hashes and both deterministic payload
hashes, verify their sidecars, source start/end hashes, run metadata, status,
selection object, and canonical payload equality. Persist:

- `artifacts/experiment_000/readout_trace_a2/DETERMINISM_VERIFICATION.json`.

Only this verified result may receive `SELECTED:<condition>` or `NO_SELECTION`.
A mismatch returns `NONDETERMINISTIC_INVALID`; an invalid native control returns
`INVALID_NATIVE_CONTROL`; a phase/source/artifact mismatch returns
`INVALID_PROCEDURE_NONSELECTING`.

## 9. Source integrity and report requirements

Every smoke/full/rerun records complete start and end source manifests and their
canonical hashes. The two manifests must equal each other and the freeze record.
The check occurs again before the verifier issues a terminal result. A report
must also contain:

- protocol/config/source/environment hashes and seed-role validation;
- zero confirmatory-seed count;
- train/eval stream manifests, identities, and hashes;
- graph topology, full structural, complete/per-kind weight, radius, and norm
  comparisons;
- raw paired feature/output/activity/trace-touch rows and their hashes;
- all retention, output, ridge, activity, blank-tail, wake, and diagnostic-only
  metrics;
- every individual seed and aggregate gate;
- fixed worker count/backend and ordered-result proof;
- explicit invalid-v1a exclusion;
- deterministic payload SHA-256 inside the report and report-file SHA-256 only
  in the adjacent sidecar;
- runtime/machine text only as non-deterministic provenance.

Canonical JSON uses sorted keys, compact separators, ASCII escaping, and
`allow_nan=false`. Numerical arrays are hashed as shape plus contiguous
little-endian float64 bytes. Across-seed quartiles use NumPy linear interpolation
at probabilities 0.25, 0.50, 0.75.

## 10. Downstream credit contract

The candidate trace readout has no current causal `UnitEvent` or `EdgeTrace`
path. It is computed as a read-only diagnostic from passive hidden state outside
the native output event ancestry. Therefore:

- A2 selection is a **representation selection only**;
- a pass advances only to a separately preregistered exact-alignment instrument
  on fresh custom seeds that differentiates the actual trace-readout computation;
- a pass does not authorize CCF-v0 training with this readout;
- no ordinary CCF-v0 update may be interpreted as credit through the passive
  trace;
- any local causal ancestry, temporal trace-credit equation, or trainable
  trace-readout integration is a separately versioned CCF-v1 mechanism with its
  own protocol, tests, seeds, and ablations.

This boundary prevents a representation improvement from being mistaken for a
working local credit-assignment mechanism.

## 11. Stop rule

Do not alter this A2 equation, grid, feature representation, sample size,
numeric gate, aggregation, worker count, or decision rule after a seed-70--74
task metric becomes visible. A full `NO_SELECTION` keeps the native architecture.
A full selection advances only under the downstream credit contract above.

