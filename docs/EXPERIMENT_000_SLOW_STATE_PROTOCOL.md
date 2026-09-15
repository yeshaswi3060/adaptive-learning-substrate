# Experiment 000 Architecture-Only Slow Local State Protocol

**Status:** A1 preflight-audited and frozen before smoke or official run  
**Original draft freeze:** 2026-09-02, before implementation  
**A1 audit amendment:** 2026-09-02, during implementation and before any slow-state result  
**Mechanism:** CCF-v0, implementation revision C000.1, present but never called  
**Intervention:** one fixed, lazy exponential-retention coefficient on hidden-unit local state  
**Seed partition:** custom seeds 60--64 only; development, confirmatory, alignment, and memory-sweep seeds are forbidden

The recurrent-radius sweep returned `NO_SELECTION`: stronger recurrent weights
increased absolute cue signal, but no candidate passed the complete cue/noise,
activity, and quiescence gates. This experiment therefore leaves every edge,
weight, delay, and graph hyperparameter unchanged and tests a different
architectural hypothesis: a unit may retain its own last locally computed state
for a controlled time instead of asking recurrent circulation alone to carry
all memory.

> **Simple explanation:** Each hidden unit gets a small local memory. The memory
> fades by a fixed amount while the unit is idle and is consulted only when a
> real event reaches that unit. Nothing learns in this test. We compare several
> fixed fading speeds using the same paired examples and reject any version that
> merely stays active, saturates, or causes substantially more emitted events or
> edge traffic.

### Preflight amendment A1 -- independent protocol audit

After the initial protocol was drafted and implementation work began, but before
any slow-state smoke or full result was produced, an independent audit checked
the document and configuration for internal and reproducibility ambiguities. A1
clarifies: optional-null handling; smoke blocking versus candidate numerical
gates; reset and diagnostic touch accounting; a structural hash that includes
`plastic`; exact CCF method-source preservation; ridge ties/zero-variance
features; quantiles; feature-reconstruction tolerance; smoke sidecar output; and
required candidate micrograph tests. It also isolates candidate dynamics in a
new experiment module so the native simulator remains an external frozen
control. A1 does **not** change the equation,
condition grid, seeds, samples, numerical selection thresholds, activity limits,
tail/wake limits, aggregation intent, or selection rule. No slow-state numerical
result was observed while making A1. Version `slow-state-v1-a1` is the operative
freeze.

## 1. Question, claim boundary, and stop boundary

Can a bounded, event-triggered local state preserve the cue across eight
intervening noise events, while retaining the native graph's weights and
topology and satisfying the same strict activity and long-tail constraints?

This is a **no-learning architecture-selection experiment**. It cannot establish
that CCF learns, that the system solves Experiment 000, that the design is novel,
or that it is sparse or efficient. A passing condition demonstrates only that
the fixed forward dynamics expose a stronger, bounded cue signal at `QUERY`.
The complete native forward/credit module
`src/adaptive_learning_substrate/recurrent.py` must remain byte-identical to the
pre-intervention file, SHA-256
`b56216abd19f5d3aaf935b6e58831c54ccf078ce4e4c7ef80e006b04ca6231b3`.
Candidate dynamics must live in a new experiment-only module/subclass; native
must use the original class and builder. Terminal feedback never enters the
graph, and the runner must never call `apply_supervised_credit`. As a second,
narrow check, the exact source bundle of `_credit_target`, `_sign`,
`_record_packet`, and `apply_supervised_credit` must retain SHA-256
`38378b7dc7d0b69644c9bd790a1f09d2ebfb9ec642da5136a2475f89f260607d`,
using this byte-exact extraction: read UTF-8 text with universal newline
normalization; parse the class with `ast`; for each listed method in that order,
take `splitlines(keepends=True)` from the earliest decorator line (or definition
line) through `end_lineno` inclusive; join the four raw segments with the UTF-8
separator `\n---METHOD---\n`; then hash the bytes. A unit test must reproduce
the registered hash. The frozen `docs/CCF_V0.md` SHA-256 is
`24907da979acef2ce95b6bb50ca52461d7b9d8b3bb6154fe871ae9000d7b7e9b`.

The original draft preceded implementation; operative A1 was frozen during
implementation but before any slow-state task-data smoke/full execution or
numerical metric, and no slow-state task result had been observed. No
exploratory pilot is authorized on
seeds 60--64. A smoke run may expose only software defects or invariant
failures; its numerical task metrics must not change the conditions, thresholds,
aggregation, or sample size below.

## 2. Frozen local-state equation

The intervention applies to hidden units only. It does not apply to the three
source units or the scalar output unit. For hidden unit \(i\), let \(M_i(t)\) be
the nonempty batch of messages that arrives at tick \(t\), and define the usual
weighted incoming drive

\[
u_i(t)=\sum_{m\in M_i(t)} w_{e(m)}v_m.
\]

Implement this equation in an experiment-only subclass or wrapper. Do not add a
retention option, state field, ledger field, or branch to the frozen native
`RecurrentEventGraph`; the subclass owns all slow-state storage and extra audit
counters.

The unit stores two episode-local scalars: its last processed activation
\(s_i\in[-1,1]\) and the tick \(\tau_i\) at which that activation was computed.
Before the unit has processed an event, its stored state is exactly zero. For a
candidate retention coefficient \(\lambda\), processing a nonempty message batch
uses

\[
r_i(t)=\lambda^{t-\tau_i}s_i,
\qquad
p_i(t)=u_i(t)+r_i(t),
\qquad
a_i(t)=\tanh(p_i(t)),
\]

and then performs the local assignments

\[
s_i\leftarrow a_i(t),\qquad \tau_i\leftarrow t.
\]

If the unit has never processed a message, \(r_i(t)=0\). Every processing gap is
a positive integer, so \(\lambda=0\) gives \(r_i(t)=0\) and exactly recovers the
native activation \(a_i(t)=\tanh(u_i(t))\). The implementation must not evaluate
`0 ** 0`; same-tick recurrent arrival is already forbidden by one-tick edges.

The incoming drive is not multiplied by \(1-\lambda\). Doing that would change
the response to the first event after reset and would confound local timescale
with instantaneous input gain. Here every condition has the same first-event
drive; only the explicitly retained prior activation differs.

The retained term is a local state contribution, not a serialized graph edge. It
creates no `RecurrentEdge`, edge ID, message, or scheduled self-message.
Mathematically it **is** an explicit temporal self-state path, and that path is
the sole architecture change being tested. Edge-omission traces, although
learning is disabled here, must hold the retained term fixed:

\[
a_i^{(-e)}(t)=\tanh\!\left(r_i(t)+u_i(t)-w_ev_e\right),
\qquad
\Delta_{i,e}=a_i(t)-a_i^{(-e)}(t).
\]

No causal trace may be fabricated for the retained term in this experiment.
Whether and how credit should traverse local state is a later learning-rule
question and cannot be inferred from a retention pass.

### Lazy event-driven semantics

When \(M_i(t)\) is empty, the runner performs no state read, decay, write,
activation evaluation, or emission test for unit \(i\). Its mathematically
effective state at any later observation tick \(T\) is computed lazily as

\[
\hat s_i(T)=\lambda^{T-\tau_i}s_i.
\]

Thus elapsed time is represented by one local timestamp and an analytic decay,
not by a dense update of all 64 units on every blank tick. `begin_episode()` must
reset every \(s_i\) to `+0.0` and every \(\tau_i\) to a sentinel before any input
is accepted. State never crosses an episode boundary.

This is local because a unit reads only its own state, its own timestamp, its
arriving messages, and their incident weights. It is event-driven because an
inactive unit receives no update. It is not learning because \(\lambda\) is fixed
for the complete condition and no parameter changes from experience.

For every nonzero-\(\lambda\) candidate, the first hidden processing after the
episode sentinel counts one logical state read, one decay operation, and one
state write even though its retained value is defined as zero. Resetting the 64
candidate states in `begin_episode()` is administrative work recorded separately
as 64 `local_state_reset_touches`; it is excluded from the three forward-state
touch ledgers. Native and the \(\lambda=0\) equivalence fast path own no local
state and therefore report zero reset, observation, and forward-state touches.

## 3. Frozen conditions and theoretical coverage

The six conditions, in execution and report order, are:

1. `native` -- the unmodified graph, equivalent to \(\lambda=0\);
2. `lambda_0_25` -- \(\lambda=0.25\);
3. `lambda_0_50` -- \(\lambda=0.50\);
4. `lambda_0_75` -- \(\lambda=0.75\);
5. `lambda_0_90` -- \(\lambda=0.90\);
6. `lambda_0_95` -- \(\lambda=0.95\).

The grid was selected from decay geometry before observing any slow-state result.
For a unit that receives no intervening event, its half-life is
\(h(\lambda)=\log(0.5)/\log(\lambda)\), and its ten-tick retention is
\(\lambda^{10}\):

| Condition | Half-life (ticks) | Ten-tick retention |
|---|---:|---:|
| `lambda_0_25` | 0.500 | 0.00000095 |
| `lambda_0_50` | 1.000 | 0.00097656 |
| `lambda_0_75` | 2.409 | 0.05631351 |
| `lambda_0_90` | 6.579 | 0.34867844 |
| `lambda_0_95` | 13.513 | 0.59873694 |

These values bracket a very short state, intermediate states, and states that
remain material over the approximately ten ticks between first hidden cue
processing and forced output. The grid must not be extended, interpolated, or
reordered after smoke or full values become visible.

For each seed, all six conditions must have exactly the native edge tuple and
native float64 weight map. The candidate builder may clone the native graph but
must not redraw it. Input, recurrent, and output weights are bitwise identical;
the recurrent spectral radius and operator norm therefore remain identical to
native. All edge IDs, endpoints, kinds, delays, plasticity flags, source/output
names, graph options, and the recurrent zero mask must match native.

In this protocol, "topology unchanged" always means the explicit message-edge
topology represented by `RecurrentEdge` and its hash. It does not hide the new
internal temporal path: that path, its state, timestamp, and fixed \(\lambda\)
must be reported separately.

Because the built-in `topology_hash()` omits the edge `plastic` flag, the runner
must also compute a full structural hash. Canonically hash the ordered input
nodes, ordered hidden nodes, output node, and every edge in stored order as
`(edge_id, source, destination, kind, delay_ticks, plastic)`. Both the built-in
topology hash and this full structural hash must equal native.

Complete and per-kind weight maps use the same frozen byte framing as the prior
memory sweep. Sort selected edges by edge ID; initialize SHA-256 with ASCII
`str((row_count,))` and one NUL byte; then, for each row, append UTF-8 edge ID,
one NUL byte, exactly one contiguous little-endian float64 weight, and one newline
byte. Hash all kinds for the complete map and each of `input`, `recurrent`, and
`output` separately. Equality means digest equality and direct bitwise equality
of every framed float64 value.

## 4. Frozen data and counterfactual pairing

- Official master seeds: exactly `60, 61, 62, 63, 64`, in that order, each
  required to have the `custom` role.
- Seeds 0--4, 42--46, 50--54, and 1000--1019 are forbidden. Confirmatory-seed
  count must be zero.
- Full run: 100 base noise streams in the train split and 100 independently
  generated base noise streams in the evaluation split, per seed.
- Smoke run: two base streams per split. Smoke validates execution and
  invariants only and cannot select a condition.
- Each base stream contains eight bipolar noise events.
- Each base stream is run twice from reset state: once with `CUE(0)` and once
  with `CUE(1)`. The two runs use identical noise and differ only in cue.
- The same train/evaluation manifests and base streams are reused by all six
  conditions within a seed.
- Train and evaluation RNG namespaces, identities, episode IDs, and noise IDs
  remain disjoint.
- `QUERY` is learner-visible; its target and terminal feedback are not.

A full seed/condition therefore contains 400 counterfactual forward episodes:
two cues times 100 pairs times two splits. The official run contains 12,000
task forward episodes across five seeds and six conditions. The train split is
used only to fit a fixed diagnostic ridge readout; it never updates the graph.
Every selection gate uses the disjoint evaluation split.

## 5. Frozen graph and feature definition

Use 64 hidden units, recurrent in-degree 8, input fan-out 8, output fan-in 16,
one-tick edges, two-tick query readout latency, initialization scale 0.35,
`tanh`, float64, and emission threshold \(10^{-3}\). Preserve all other
Experiment-000/CCF-v0 values: learning rate 0.01, trace decay 0.97, route gain
0.90, credit limit 1.0, maximum update 0.05, weight bound 3.0, trace horizon 32,
hop limit 16, epsilons \(10^{-12}\), and credit minimum \(10^{-8}\). These
learning values are inert provenance.

For a query, define \(z\in\mathbb{R}^{16}\) as the hidden message value arriving
on each frozen output edge, ordered by edge ID. A missing output-edge message at
the relevant tick contributes zero. The weighted feature vector must reconstruct
the forced-output preactivation with

\[
|p_{output}-\sum_e w_ez_e|
\le10^{-12}+10^{-12}\max(|p_{output}|,|\sum_e w_ez_e|).
\]

The native condition must use the same public forward path as the current
simulator and must first verify the frozen whole-file SHA-256 above.
Independently, a zero-retention subclass-equivalence test must
compare native and \(\lambda=0\) step results, emitted event IDs after normalized
episode-ID substitution, activations, query result, features, and additive
legacy ledger deltas on the smoke examples. The \(\lambda=0\) implementation must
take an explicit native fast path and perform zero local-state touches. Every
numerical value must be bitwise equal and every discrete field identical before
an official full run is permitted.

## 6. Frozen retention measurements

For evaluation pair \(k\), let \(z_{k,0}\) and \(z_{k,1}\) be query features for
the two cue interventions under identical noise. Define

\[
d_k=\frac{z_{k,1}-z_{k,0}}{2},\qquad
m_k=\frac{z_{k,1}+z_{k,0}}{2},
\]

\[
C=\sqrt{\frac1K\sum_k\lVert d_k\rVert_2^2},\qquad
N=\sqrt{\frac1K\sum_k\lVert m_k-\bar m\rVert_2^2},
\qquad
\boxed{R=C/N}.
\]

For scalar query activations \(o_{k,0},o_{k,1}\), define

\[
\delta_k=\frac{o_{k,1}-o_{k,0}}2,
\qquad
\boxed{O=\frac{\sqrt{K^{-1}\sum_k\delta_k^2}}
{\operatorname{SD}(\{o_{k,0},o_{k,1}\}_{k=1}^K)}}.
\]

A denominator at or below \(10^{-15}\) produces `null`, not infinity or NaN;
`null` fails its selection gate. All realized numeric fields must be finite, but
these protocol-defined optional ratio fields may be `null`; such a `null` is not
by itself a structural/native-integrity failure. A required seed-level `null`
makes that candidate's corresponding aggregate gate false; required nulls are
never silently dropped to compute a passing median. Distribution summaries may
summarize the realized numeric subset only when they also report the exact null
count and mark the selection aggregate invalid.

Fit a separate ridge diagnostic for every seed and condition. Stack both cues
from each training pair, encode cue 0 as -1 and cue 1 as +1, standardize using
training means and population standard deviations only, and use
\(\alpha=10^{-3}\):

\[
\beta=\left(\frac{X^TX}{n}+\alpha I\right)^{-1}\frac{X^Ty}{n}.
\]

Apply the frozen readout to the disjoint evaluation split and report accuracy.
No cross-condition or evaluation information may enter the fit. A coordinate
whose training population SD is at or below \(10^{-15}\) is standardized to
exactly zero in train and evaluation, excluded from the solve, and assigned an
embedded coefficient of zero. A diagnostic ridge score exactly equal to zero is
classified as cue 1, matching the frozen `score >= 0` rule.

## 7. Activity, state, and stability measurements

For every task episode, record deltas of `emitted_unit_events`,
`forward_edge_touches`, `activation_evaluations`, the new
`hidden_activation_evaluations`, hidden emitted events, and the new
`local_state_read_touches`, `local_state_decay_touches`, and
`local_state_write_touches`. For candidate conditions the three local-state
touch counts must be equal to the number of hidden activation evaluations, and
must be zero for an inactive hidden unit on a blank tick. Native local-state
touch counts are zero. Administrative reset and read-only observation counters
are separate and are never added to these forward touch counts.

Let \(E_{s,q,x,k,c}\) be emitted-unit-event count for seed \(s\), condition
\(q\), split \(x\), pair \(k\), and cue \(c\), and let \(A_{s,q,x}\) be its
sum over cues and pairs. For every candidate, seed, and split require

\[
\frac{A_{s,q,x}}{A_{s,native,x}}\le1.10,
\qquad
\max_{k,c}\frac{E_{s,q,x,k,c}}{E_{s,native,x,k,c}}\le1.15.
\]

Apply the same 1.10 total and 1.15 maximum matched-episode limits to
`forward_edge_touches`. Every native denominator must be positive. These are
hard per-seed/per-split gates, not grand averages.

For the ordinary eleven hidden-processing ticks, report hidden emission density

\[
D^{hidden}_{s,q,x,k,c}=
\frac{\text{emitted hidden UnitEvents}}{64\times11}.
\]

Also report hidden activation saturation fractions at absolute thresholds 0.95,
0.99, and 0.999, maximum absolute preactivation, and state-touch counts. These
fractions use every hidden activation evaluation, including evaluations that do
not emit, as their denominator. They are diagnostics, not substitutes for the
preregistered selection gates. Because the native graph is already densely
active, no result from this experiment may be described as sparse or efficient.

### Deterministic both-polarity 256-tick blank-tail and wake assay

For every seed, condition, and cue polarity \(p\in\{-1,+1\}\), reset the graph,
inject only `graph.step({"cue": p})` at tick 0, and then call
`graph.step({})` exactly 256 times at ticks 1--256. Record the hidden-emission
count at every blank tick, the complete vector and deterministic hash, the last
nonzero tick, and

\[
T_{s,q,p}=\sum_{t=193}^{256}H_{s,q,p,t}.
\]

Every seed, condition, and polarity must satisfy \(T_{s,q,p}=0\). At tick 256,
compute every hidden unit's effective lazy state without changing it:

\[
B_{s,q,p}=\max_i|\hat s_i(256)|.
\]

This is a read-only diagnostic snapshot: it does not alter stored state,
timestamps, events, or pending messages, and its 64 reads are recorded only as
`local_state_observation_touches`. Snapshot reads and analytic decays are
excluded from the three forward-state touch ledgers. This 64-touch rule applies
only to nonzero-\(\lambda\) candidates. Native and the \(\lambda=0\) equivalence
fast path report the conceptual all-zero effective-state vector and zero
observation touches.

Require \(B_{s,q,p}\le10^{-4}\), one tenth of the emission threshold. Then,
without resetting that polarity trial, call `query()` and extract the ordinary
16 query features. Pair the two polarity trials and require both

\[
\frac12\lVert z_{wake,+1}-z_{wake,-1}\rVert_2\le10^{-3},
\qquad
\frac12|o_{wake,+1}-o_{wake,-1}|\le10^{-3}.
\]

The wake check prevents a dormant but large state from passing merely because
no blank-tick message arrived to trigger it. It is a decay/stability test, not a
task-retention metric.

Every condition must additionally satisfy the integrity invariants below:

1. all realized numeric weights, states, decay factors, preactivations,
   activations, features, outputs, ridge values, and metrics are finite;
   protocol-defined optional ratio `null`s are allowed and fail their gates;
2. every hidden state and activation lies in the mathematical `tanh` range;
3. the built-in topology hash and full structural hash including `plastic` equal
   native before and after every probe;
4. complete, recurrent, input, and output weight hashes equal native bit-for-bit
   before and after each split, tail trial, and complete condition;
5. recurrent spectral radius and operator 2-norm equal native within
   \(10^{-12}\) absolute and relative tolerance;
6. each episode begins with zero local state and no state crosses an episode;
7. the registered \(\lambda\) is immutable and no adaptive decay occurs;
8. no credit packet, credit-edge touch, or weight-write touch occurs;
9. the graph reports zero nonfinite values and all weights stay within 3.0;
10. feature edge IDs, dimension, and ordering are identical across conditions;
11. forward state touches occur only with a nonempty hidden message batch and
    equal the hidden activation-evaluation count in candidates; reset and
    observation touches equal their separately defined counts;
12. the frozen native `recurrent.py`, CCF method-source bundle, and CCF document
    hashes match.

Items 1--12 are implementation/global/native integrity invariants. A native or
global failure invalidates the run; a structural or accounting mismatch in any
candidate indicates an implementation defect and requires a complete restart
from smoke after correction. Candidate numerical activity, blank-tail,
effective-state, and wake thresholds are selection gates rather than software
integrity invariants: they reject that candidate only when evaluated on the full
run.

## 8. Preregistered selection rule

Only the full 100-pair evaluation result can select a candidate. For candidate
\(q\), let \(C_{s,q}\), \(R_{s,q}\), \(O_{s,q}\), and
\(\mathcal A_{s,q}\) denote evaluation cue-feature RMS, cue/noise ratio, output
ratio, and ridge accuracy for seed \(s\). A candidate passes only if **all** of
the following hold:

1. \(\operatorname{median}_s C_{s,q}\ge
   3\operatorname{median}_s C_{s,native}\);
2. at least four of five seeds satisfy \(C_{s,q}\ge3C_{s,native}\);
3. \(\operatorname{median}_s R_{s,q}\ge0.50\);
4. \(\operatorname{median}_s R_{s,q}\ge
   3\operatorname{median}_s R_{s,native}\);
5. at least four of five seeds satisfy \(R_{s,q}\ge3R_{s,native}\);
6. \(\operatorname{median}_s\mathcal A_{s,q}\ge0.70\);
7. \(\operatorname{median}_s O_{s,q}\ge0.50\);
8. every weight, topology, finite-state, episode-reset, event-driven-touch,
   CCF-hash, activity, and feature gate with split scope passes for every
   seed/split, and every blank-tail, effective-state, and wake gate passes for
   every seed/polarity.

For native multipliers, a native denominator at or below \(10^{-15}\) produces
`null` and fails the factor gate. Medians are taken across the five seed-level
metrics; task examples are never pooled across seeds for a gate.

If several candidates pass, select the **lowest numerical \(\lambda\)**. Native
is a control and is not selectable. Do not choose the best-looking failure,
interpolate, tune a threshold, inspect confirmatory seeds, or combine this grid
with recurrent-radius changes.

If no candidate passes, report exactly `NO_SELECTION`, retain the native
architecture, and stop this intervention. If one passes, report
`SELECTED:<condition>`; it may advance only to a separately preregistered exact
alignment experiment on new custom seeds. A retention pass does not authorize a
learning run until the local-state causal-trace semantics have also been
specified and falsified.

## 9. Execution, determinism, and rerun stop rules

Reject any seed list other than exactly `60,61,62,63,64`, any condition list or
order other than the six frozen conditions, and any pair count other than 2 or
100 before creating an output file.

Execution order is fixed:

1. run unit/integration tests, including exact \(\lambda=0\) equivalence and the
   nonzero-retention micrographs below;
2. run one two-pair smoke on all five seeds and all conditions;
3. if and only if smoke completes and all implementation/global/native integrity
   invariants pass, run the 100-pair full experiment; candidate numerical
   retention, activity, tail, effective-state, and wake failures in smoke are
   recorded but do not block, prune, or select a condition;
4. independently rerun the identical full command to a second output path;
5. compare deterministic payload SHA-256 values exactly.

Before smoke, deterministic analytic micrographs must verify: a positive and a
negative retained state over one-tick and multi-tick gaps; exact
\(\lambda^{\Delta t}\) decay; no blank-message state update, touch, or emission;
episode reset and its separate 64-touch counter; a processed activation that
does not emit still writes local state; first-sentinel processing still records
one read/decay/write; edge omission holds the retained term fixed; source and
output dynamics remain native; effective-state observation is non-mutating and
uses only its separate 64-touch counter; and the full structural hash changes if
only an edge's `plastic` flag changes. Numeric micrograph expectations use the
same \(10^{-12}\) absolute-plus-relative tolerance as feature reconstruction.

The implementation CLI contract is also frozen. With `PYTHONPATH=src`, use:

```text
python -m adaptive_learning_substrate.experiment_000_slow_state --seeds 60 61 62 63 64 --retentions 0.25 0.50 0.75 0.90 0.95 --pairs-per-split 2 --ridge-alpha 0.001 --output artifacts/experiment_000/slow_state/smoke_pairs_2.json
python -m adaptive_learning_substrate.experiment_000_slow_state --seeds 60 61 62 63 64 --retentions 0.25 0.50 0.75 0.90 0.95 --pairs-per-split 100 --ridge-alpha 0.001 --output artifacts/experiment_000/slow_state/custom_seeds_60_64_pairs_100.json
python -m adaptive_learning_substrate.experiment_000_slow_state --seeds 60 61 62 63 64 --retentions 0.25 0.50 0.75 0.90 0.95 --pairs-per-split 100 --ridge-alpha 0.001 --output artifacts/experiment_000/slow_state/custom_seeds_60_64_pairs_100_rerun.json
```

The runner must reject omitted, duplicated, nonfinite, out-of-order, or extra
seed/retention values rather than silently normalizing a different experiment.

Across-seed quartiles in the required summary use the NumPy `quantile` linear
interpolation rule at probabilities 0.25, 0.50, and 0.75. The median selection
value is the same 0.50 quantile.

Smoke metrics cannot select or amend the design. A software defect may be
corrected only with a documented code change and a complete restart from smoke;
numerical thresholds and conditions stay frozen. If the two full deterministic
payload hashes differ, report `NONDETERMINISTIC_INVALID` and do not select. If
the native control is invalid, report `INVALID_NATIVE_CONTROL`. Otherwise apply
Section 8 once and stop at `SELECTED:<condition>` or `NO_SELECTION`.

Use canonical JSON with sorted keys, compact separators, ASCII escaping, and
`allow_nan=false`. Hash arrays using shape plus contiguous little-endian float64
bytes. Runtime, timestamps, absolute paths, and machine-specific text remain
outside the deterministic payload.

## 10. Required report

The report must contain:

- protocol, configuration, implementation, data-protocol, recurrent-source,
  and memory-probe SHA-256 hashes;
- explicit proof that the native recurrent-source whole-file hash matches the
  pre-intervention control;
- exact seed-role validation, forbidden-seed checks, and zero confirmatory count;
- train/evaluation manifests, substreams, identities, and hashes;
- condition order, exact \(\lambda\), theoretical half-life, and ten-tick decay;
- topology, complete-weight, per-kind weight, spectral-radius, and operator-norm
  comparisons against native before and after each probe, including the full
  structural hash with `plastic`;
- zero-retention native-equivalence results;
- ordered feature edge IDs and raw paired array hashes;
- raw per-pair cue outputs, event counts, hidden counts, hidden density,
  forward-edge touches, hidden activation evaluations, state touches, and
  matched native ratios;
- train/evaluation retention metrics and ridge diagnostics;
- saturation fractions and maximum absolute preactivation;
- complete both-polarity blank-tail vectors and hashes, last-nonzero ticks,
  effective-state maxima, reset/observation touches, wake features, wake outputs,
  and every tail/wake gate;
- candidate-minus-native seed rows and median, quartile, minimum, and maximum
  summaries across seeds;
- every individual gate, passing-condition list, and exactly one final status;
- runtime/backend as nondeterministic provenance outside the payload;
- deterministic payload SHA-256 inside the report; and
- complete report-file SHA-256 in a sidecar manifest for the smoke, primary full,
  and rerun full reports, avoiding self-reference.

No value observed after this freeze may alter this protocol.
