# Experiment 000 Architecture-Only Memory-Strength Sweep

**Status:** completed official run; negative result (`NO_SELECTION`)  
**Original freeze:** 2026-09-02  
**Preflight amendment:** A1, 2026-09-02  
**Mechanism:** CCF-v0, implementation revision C000.1, present but never called  
**Intervention:** uniform rescaling of recurrent weights only  
**Seed partition:** custom seeds 50--54 only; development and confirmatory seeds
are forbidden

This experiment follows the fixed-event alignment diagnostic. That diagnostic
found that the current graph carries almost no useful cue sensitivity to the
query. The present sweep changes one architectural quantity while leaving the
topology, input and output weights, event rule, and CCF equation untouched.

> **Simple explanation:** We make only the recurrent loop weaker or stronger,
> show each version exactly the same noise, and ask whether the old cue survives
> until `QUERY`. No version is allowed to learn during this measurement.

## 1. Question and scope

Does uniformly increasing the native recurrent spectral radius preserve the
cue for nine intervening events without causing excessive event activity?

The experiment is a **no-learning architecture selection instrument**. It is
not an Experiment-000 mechanism pass, a learning result, a novelty result, or
a confirmation. The selected radius, if any, may advance only to a separately
preregistered alignment experiment on new custom seeds.

CCF-v0/C000.1 must remain byte-for-byte unchanged. The runner must never call
`apply_supervised_credit`, and terminal feedback must never enter the graph.
Every counterfactual forward pass starts with cleared episode-local state.

## Preflight amendment A1 -- disclosed exploratory audit pilot

After the original document was written but **before any official root smoke or
full run**, an independent exploratory audit pilot exercised two pairs and a
long blank tail on seeds 50--54. The following scratch values were seen:

| Condition | Mean two-pair cue/noise ratio | Mean cue RMS | Mean noise RMS |
|---|---:|---:|---:|
| native | 0.205 | 0.00463 | 0.0561 |
| rho_0_60 | 0.152 | 0.01139 | 0.1038 |
| rho_0_80 | 0.242 | 0.03243 | 0.1654 |
| rho_0_95 | 0.512 | 0.12208 | 0.2391 |
| rho_1_05 | 1.016 | 0.28565 | 0.3082 |

With `CUE(+1)` followed by 256 empty ticks, `rho_1_05` continued dense
autonomous activity through tick 256 on four of five seeds, producing roughly
4,070--4,096 hidden events in the last 64 ticks. In these scratch runs,
`rho_0_95`, `rho_0_80`, and `rho_0_60` became quiet by ticks 72, 24, and 14 or
earlier, respectively. Two-pair ridge training accuracy was 1.0 in every cell,
which demonstrates why that tiny fit is not evidence. Mean actual-task hidden
emission densities were already approximately 0.832 for native, 0.885 for
`rho_0_95`, and 0.889 for `rho_1_05`, out of 704 possible hidden node-ticks per
episode.

These scratch observations are **non-official**, cannot pass or select a
condition, and cannot substitute for the 100-pair result. They exposed flaws
in the original validity gates: a twofold average activity allowance could hide
autonomous firing, a ratio could rise without enough absolute cue growth, and
total events could obscure already-dense hidden activity. Amendment A1 adds a
both-polarity blank-tail assay, tightens matched task-activity limits, requires
a threefold increase in absolute cue RMS, and mandates hidden-density and edge-
touch reporting. The conditions, seeds, full sample size, and original
ratio/ridge/output thresholds are unchanged. All A1 rules below are frozen
after this disclosed pilot and before the official run.

## 2. Frozen conditions

For each seed, first construct the ordinary Experiment-000 graph. Let
\(W_s\) be its sparse hidden-to-hidden matrix and let

\[
\rho_s=\rho(W_s)=\max_i |\lambda_i(W_s)|.
\]

The five conditions, in execution and report order, are:

1. `native` -- no rescaling;
2. `rho_0_60` -- target \(r=0.60\);
3. `rho_0_80` -- target \(r=0.80\);
4. `rho_0_95` -- target \(r=0.95\);
5. `rho_1_05` -- target \(r=1.05\).

For a target condition \(r\), calculate the positive factor

\[
\gamma_{s,r}=\frac{r}{\rho_s}
\]

and create its weight map by

\[
w^{(r)}_e=
\begin{cases}
\gamma_{s,r}w^{(native)}_e,&e\text{ is recurrent},\\
w^{(native)}_e,&e\text{ is cue, noise, query, or output}.
\end{cases}
\]

Because \(\rho(\gamma W)=|\gamma|\rho(W)\), this operation targets the
requested radius without redrawing the graph. A native radius at or below
\(10^{-15}\) invalidates the complete official run. For each target, require

\[
|\rho(W_s^{(r)})-r|
\le 10^{-10}+10^{-10}\max(\rho(W_s^{(r)}),r).
\]

The rescaling must preserve the recurrent zero mask and signs. It must preserve
all nonrecurrent float64 values bit-for-bit, every edge ID, delay, endpoint,
edge kind, plasticity flag, and graph hyperparameter. Building a new graph
with a different general initialization scale is forbidden because that would
also alter input/output weights. Every scaled weight must remain finite and
must satisfy the frozen bound

\[
\max_e |w_e^{(r)}|\le3.0.
\]

A target violating this bound fails before its probe begins.

For secondary provenance, also report the recurrent operator 2-norm and the
preregistered nonnormality ratio (operator norm divided by spectral radius):

\[
\|W_s^{(q)}\|_2=\sigma_{max}(W_s^{(q)}),
\qquad
\nu_{s,q}=\frac{\|W_s^{(q)}\|_2}{\rho(W_s^{(q)})}.
\]

Use the same \(10^{-15}\) denominator guard. A sparse non-normal recurrent
matrix can transiently amplify a state even when \(\rho<1\), so neither radius
nor \(\nu\) is a stability pass. The both-polarity blank-tail behavior in
Section 6 is the hard stability gate.

The spectral radius is only the controlled knob; it is not assumed to be a
complete measure of nonlinear memory. The paired probe below decides whether
memory actually improved.

## 3. Data and counterfactual pairing

- Official master seeds: `50, 51, 52, 53, 54`, each required to have the
  `custom` role.
- Full run: 100 base noise streams in the train split and 100 independently
  generated base noise streams in the evaluation split, per seed.
- Smoke run: two base streams per split, used only to validate execution and
  invariants. Smoke metrics cannot select a condition or change a threshold.
- Each base stream contains eight bipolar noise events.
- Each base stream is run twice from reset state: once with `CUE(0)` and once
  with `CUE(1)`. The two runs have identical noise and differ only in cue.
- The exact same train/evaluation manifests and base streams are reused by all
  five conditions within a seed.
- Train and evaluation RNG namespaces and sample identities remain disjoint.
- `QUERY` is learner-visible; its target and terminal feedback are not.

Thus a full seed/condition contains 400 counterfactual forward episodes:
\(2\) cues times \(100\) pairs times \(2\) splits. Across five seeds and five
conditions, the official run contains 10,000 forward episodes.

The train split is used only to fit the fixed diagnostic ridge readout. It does
not update the graph. All selection gates use evaluation measurements.

## 4. Frozen graph and feature definition

Use 64 hidden units, recurrent in-degree 8, input fan-out 8, output fan-in 16,
one-tick edges, two-tick query readout latency, native initialization scale
0.35, `tanh`, float64, and emission threshold \(10^{-3}\). Retain all other
Experiment-000/CCF-v0 options: learning rate 0.01, trace decay 0.97, route gain
0.90, credit limit 1.0, maximum update 0.05, weight bound 3.0, trace horizon
32, hop limit 16, epsilons \(10^{-12}\), and credit minimum \(10^{-8}\).
The learning-related values are recorded provenance and remain inert.

For a query, define feature vector \(z\in\mathbb{R}^{16}\) as the hidden
message value arriving on each frozen output edge, ordered by edge ID. An
output edge that did not emit at the relevant tick contributes zero. The
weighted feature vector must reconstruct the forced-output preactivation to
float64 tolerance.

## 5. Retention measurements

For pair \(k\), condition \(q\), and split \(x\), let \(z_{k,0}\) and
\(z_{k,1}\) be query feature vectors under the two cue interventions. Define

\[
d_k=\frac{z_{k,1}-z_{k,0}}{2},\qquad
m_k=\frac{z_{k,1}+z_{k,0}}{2}.
\]

The factor one half makes \(d_k\) the coefficient of a bipolar cue. Define

\[
C=\sqrt{\frac1K\sum_k\|d_k\|_2^2},
\qquad
N=\sqrt{\frac1K\sum_k\|m_k-\bar m\|_2^2},
\]

\[
\boxed{R=\frac{C}{N}}.
\]

`R` is the cue-to-noise feature ratio. A larger value means the query features
change more with the old cue than with the intervening noise.

For query output activations \(o_{k,0},o_{k,1}\), define

\[
\delta_k=\frac{o_{k,1}-o_{k,0}}2,
\qquad
\boxed{O=\frac{\sqrt{K^{-1}\sum_k\delta_k^2}}
{\operatorname{SD}(\{o_{k,0},o_{k,1}\}_{k=1}^K)}}.
\]

Denominators at or below \(10^{-15}\) produce `null`, never infinity or NaN;
`null` fails selection.

Fit a ridge diagnostic separately for every seed and condition. Stack both
cues from each training pair, encode cue 0 as -1 and cue 1 as +1, standardize
with training means and population standard deviations only, and use
\(\alpha=10^{-3}\):

\[
\beta=\left(\frac{X^TX}{n}+\alpha I\right)^{-1}
\frac{X^Ty}{n}.
\]

Apply the frozen readout to the disjoint evaluation split and report accuracy.
No cross-condition or evaluation information may enter this fit.

## 6. Activity and stability measurements

For every task forward episode, record deltas of `emitted_unit_events` and
`forward_edge_touches`. Count emitted hidden events separately. Let
\(E_{s,q,x,k,c}\) be the total emitted-unit-event count for seed \(s\),
condition \(q\), split \(x\), pair \(k\), and cue \(c\), and let
\(A_{s,q,x}\) be its sum across both cues and every pair. Compare only exactly
matched episodes:

\[
Q_{s,q,x}=\frac{A_{s,q,x}}{A_{s,native,x}}.
\]

For every candidate, seed, and split require

\[
Q_{s,q,x}\le1.10
\]

and

\[
\max_{k,c}
\frac{E_{s,q,x,k,c}}{E_{s,native,x,k,c}}
\le1.15.
\]

Every native denominator must be positive. The first bound limits total task
work to 10% above native; the matched 15% bound prevents one episode burst
from being hidden by an acceptable total.

For the ordinary eleven hidden-processing ticks, also report per-episode
hidden emission density

\[
D^{hidden}_{s,q,x,k,c}=
\frac{\text{emitted hidden UnitEvents}}{64\times11}
=\frac{\text{emitted hidden UnitEvents}}{704}.
\]

Report this density and forward-edge touches; neither is pooled into the cue
metric. Because the disclosed pilot found native density near 0.832, even a
selected radius makes **no sparse-computation or efficiency claim**.

### Deterministic 256-tick blank-tail assay

Separately for every seed, condition, and cue polarity
\(p\in\{-1,+1\}\), reset the graph, inject exactly one cue at tick 0 with
`graph.step({"cue": p})`, provide no noise and no query, and then call
`graph.step({})` exactly 256 times at ticks 1 through 256. Both signs are run;
a sign is never inferred from the other.

At every blank tick \(t\), record

\[
H_{s,q,p,t}=\#\{\text{emitted UnitEvents on hidden nodes at tick }t\}.
\]

Report the complete 256-count vector, its deterministic hash, the last tick
whose count is nonzero (or `null`), and

\[
T_{s,q,p}=\sum_{t=193}^{256}H_{s,q,p,t}.
\]

Every seed, condition, and polarity must satisfy

\[
\boxed{T_{s,q,p}=0}.
\]

Do not average this gate across seeds or signs. One nonzero tail fails that
candidate; a native tail failure invalidates the official run. Topology and
complete weight hashes must remain unchanged across every blank-tail trial.

Every condition must additionally satisfy all of these invariants:

1. all weights, eigenvalues, preactivations, activations, features, outputs,
   ridge values, and reported metrics are finite;
2. every activation lies in the mathematical `tanh` range;
3. the topology hash is unchanged across all conditions for a seed;
4. nonrecurrent weight hashes equal native before and after the probe;
5. each condition's complete weight hash is identical before and after every
   split and at the end of the probe;
6. no credit packets or weight-write touches occur;
7. the graph reports zero nonfinite values;
8. feature edge IDs and dimension remain identical.
9. every scaled weight obeys the frozen absolute bound 3.0;
10. every blank-tail assay has zero hidden emissions in ticks 193--256.

Any invariant failure invalidates the candidate. A native invariant failure
invalidates the entire official run.

## 7. Preregistered selection rule

Only the full 100-pair evaluation result can select a candidate. For condition
\(q\), let \(C_{s,q}\), \(R_{s,q}\), \(O_{s,q}\), and
\(\mathcal A_{s,q}\) be
evaluation cue-feature-delta RMS, cue/noise ratio, output ratio, and ridge
accuracy for seed \(s\). A target passes only if **all** of the following hold:

1. \(\operatorname{median}_s C_{s,q}\ge
   3\operatorname{median}_s C_{s,native}\);
2. at least four of five seeds satisfy
   \(C_{s,q}\ge3C_{s,native}\);
3. \(\operatorname{median}_s R_{s,q}\ge0.50\);
4. \(\operatorname{median}_s R_{s,q}\ge
   3\operatorname{median}_s R_{s,native}\);
5. at least four of five seeds satisfy
   \(R_{s,q}\ge3R_{s,native}\);
6. \(\operatorname{median}_s \mathcal A_{s,q}\ge0.70\);
7. \(\operatorname{median}_s O_{s,q}\ge0.50\);
8. every finite, target-radius, weight-bound, frozen-state, topology, feature,
   task-activity, and blank-tail
   invariant in Sections 2 and 6 passes for every seed and split.

For the reported candidate/native multiplier fields for both \(C\) and \(R\),
a native denominator at or below \(10^{-15}\) produces `null` and fails the
corresponding factor gate. This guard prevents a zero native measurement from
making a nominal threefold comparison meaningless.

If several targets pass, select the **lowest numerical target radius**. Native
is a control and is not selectable. Do not interpolate between targets, tune a
threshold, prefer the best-looking metric, or inspect confirmatory seeds.

### Aggregation refinement frozen in the original preregistration

The original recommendation stated numeric thresholds without defining how
five seeds should be combined. Pooling all examples or using a mean could let
one unusually strong seed hide failure on others. Therefore the numeric gates
above use the median seed, the threefold improvement additionally requires
four-of-five within-seed paired successes, and the activity limit applies to
every seed and split rather than to a grand average. This aggregation rule was
set before the disclosed A1 pilot. A1 retains it and adds the separately
documented absolute-signal, activity, weight-bound, and blank-tail rules above.

### Full rejection behavior

If no target satisfies every gate, report exactly `NO_SELECTION`. Keep the
native architecture as the project state, do not advance the closest target,
and do not run alignment or learning with any swept target. The raw result may
motivate a separately preregistered memory intervention, such as a local state
timescale, but it cannot alter this protocol retrospectively.

If a target is selected, the result still demonstrates only cue retention and
bounded activity. Its next permitted step is the already defined exact
alignment instrument on new custom seeds that are disjoint from 50--54,
42--46, development seeds 0--4, and confirmatory seeds 1000--1019.

## 8. Determinism, hashes, and required output

Reject any official seed list other than exactly `50,51,52,53,54` before
creating an output file. Record confirmatory seed count as zero. Execute two
independent full reruns; selection is valid only if their deterministic payload
SHA-256 values match exactly.

Use canonical JSON with sorted keys, compact separators, ASCII escaping, and
`allow_nan=false`. Hash numerical array bundles using array shape plus
contiguous little-endian float64 bytes. Exclude runtime, timestamps, absolute
paths, and machine-specific text from the deterministic payload.

The report must contain:

- protocol, configuration, implementation, data-protocol, and relevant source
  file SHA-256 hashes;
- exact seed-role validation and zero confirmatory-seed count;
- train/evaluation stream manifests and hashes;
- condition order, target, native radius, scale factor, realized radius,
  radius-tolerance result, recurrent operator 2-norm, and operator-norm/radius
  ratio;
- topology, complete-weight, recurrent-weight, and nonrecurrent-weight hashes
  before and after each probe;
- ordered feature edge IDs and pair-array SHA-256 values;
- raw per-pair cue outputs, total/hidden event counts, hidden emission density,
  forward-edge touches, and matched activity ratios, plus
  per-seed/condition/split retention metrics;
- both-polarity 256-tick hidden-event vectors and hashes, last-nonzero ticks,
  and final-64 totals for every seed and condition;
- ridge train/evaluation metrics;
- paired candidate-minus-native rows and activity ratios;
- median, quartiles, minima, and maxima across seeds;
- every individual gate, the final passing target list, and exactly one final
  status: `SELECTED:<condition>` or `NO_SELECTION`;
- runtime/backend as nondeterministic provenance outside the hashed payload;
- deterministic payload SHA-256 inside the report;
- complete report-file SHA-256 in a separate sidecar manifest, avoiding the
  impossible self-reference of placing a file's own hash inside that file.

No selection threshold may change after an official smoke or full value becomes
visible. The disclosed A1 pilot remains nonselecting exploratory evidence.

## 9. Post-result record (not part of the preregistration)

> **Added after both official full reports existed on 2026-09-02.** Sections
> 1--8 above remain the frozen `memory-sweep-v1-a1` criteria that governed the
> run. This section records the outcome; it does not change, reinterpret, or
> relax any gate.

The first full run and its independent rerun both returned exactly
`NO_SELECTION`. No target condition passed every gate, no target radius was
selected, and confirmatory seed count remained zero. Their deterministic
payload SHA-256 values matched exactly:
`f6ba680f91917521da5bbfb89d7d1dbd102fe59149049d0ed0464bb3ece7f8e6`.

The complete report files and independently checked file hashes are:

- `artifacts/experiment_000/memory_sweep/custom_seeds_50_54_pairs_100.json`
  -- SHA-256
  `5dd993ba62770ce41ee84a8c8a42b80c642445a734fa4fddd537d2500d641a4c`;
- `artifacts/experiment_000/memory_sweep/custom_seeds_50_54_pairs_100_rerun.json`
  -- SHA-256
  `1be4397c94dc28d25d70f0a3b722d4e4530a2c86581010dfbd1b171b95b245c5`.

Both adjacent `.sha256` sidecars match the bytes of their named report. The
full interpretation, gate table, and simple mathematical explanation are in
`docs/EXPERIMENT_000_MEMORY_SWEEP_RESULT.md`.
