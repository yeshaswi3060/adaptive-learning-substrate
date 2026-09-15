# Experiment 000 architecture-only memory-strength sweep result

**Status:** completed negative result; `NO_SELECTION`  
**Date:** 2026-09-02  
**Protocol:** `memory-sweep-v1-a1`  
**Mechanism:** CCF-v0/C000.1 present but never called  
**Seeds:** custom seeds 50--54; confirmatory seeds used: 0  
**Scale:** 100 train pairs plus 100 evaluation pairs per seed and condition;
400 counterfactual forward episodes per seed and condition  

## One-sentence conclusion

Uniformly increasing recurrent spectral radius strengthened the old cue, but
no tested radius preserved a sufficiently clean cue while also satisfying the
activity and stability guards, so the preregistered decision is
**`NO_SELECTION`**.

This is a useful, reproducible negative result. It is not an Experiment-000
learning pass, a CCF improvement, a novelty result, or evidence of a
market-ready system.

## What changed and what did not

For each seed, the native recurrent weight matrix was \(W_s\), with spectral
radius

\[
\rho_s=\max_i |\lambda_i(W_s)|.
\]

For target radius \(r\in\{0.60,0.80,0.95,1.05\}\), only recurrent weights were
rescaled:

\[
\gamma_{s,r}=\frac{r}{\rho_s},
\qquad
W^{(r)}_s=\gamma_{s,r}W_s.
\]

Input weights, output weights, topology, event rules, data, and measurement
code stayed fixed. No learning occurred: all credit counts and all weight-write
counts were zero, and complete condition weights remained unchanged throughout
every probe.

> **Simple explanation:** The same network was tested with four strengths of
> its internal recurrent loop. We did not teach any version. We only asked
> whether information about an early `CUE` was still visible at `QUERY`, and
> whether the loop became noisy or kept firing after the input stopped.

## Measurements

Let \(h^+_k\) and \(h^-_k\) be the query-time hidden feature vectors for
matched pair \(k\), where the two episodes differ only in cue sign. With
\(K\) matched pairs, the retained bipolar-cue coefficient was measured as

\[
C=\sqrt{\frac{1}{K}\sum_{k=1}^{K}
\left\|\frac{h^+_k-h^-_k}{2}\right\|_2^2}.
\]

Writing \(m_k=(h^+_k+h^-_k)/2\) and
\(\bar m=K^{-1}\sum_k m_k\), the intervening-noise scale was

\[
N=\sqrt{\frac{1}{K}\sum_{k=1}^{K}\|m_k-\bar m\|_2^2}.
\]

The cue-to-noise ratio was

\[
R=\frac{C}{N}.
\]

The experiment also fitted a fixed ridge readout on the train split and
reported evaluation accuracy \(\mathcal A\). It separately measured the output
cue delta divided by output standard deviation, denoted \(O\). Larger values
mean that the old cue is easier to distinguish from irrelevant variation.

## Exact frozen pass gates

A target could advance only if **every** rule below passed:

1. median evaluation \(C\) was at least 3 times native median \(C\);
2. at least 4 of 5 seeds had \(C_{s,q}\ge3C_{s,native}\);
3. median evaluation \(R\ge0.50\);
4. median evaluation \(R\) was at least 3 times native median \(R\);
5. at least 4 of 5 seeds had \(R_{s,q}\ge3R_{s,native}\);
6. median ridge evaluation accuracy \(\mathcal A\ge0.70\);
7. median output ratio \(O\ge0.50\);
8. for every seed and split, total task event ratio was at most 1.10 and every
   matched episode event-count ratio was at most 1.15;
9. after each of `CUE(-1)` and `CUE(+1)` followed by 256 blank ticks, hidden
   emissions in ticks 193--256 were exactly zero for every seed;
10. every finite-value, registered-condition, target-radius, weight-bound,
    frozen-weight, topology, feature-identity, and no-credit invariant passed.

If several targets passed, the rule selected the lowest one. If none passed,
the required literal result was `NO_SELECTION`; choosing the closest-looking
target was forbidden.

## Full 100-pair result

All values below are medians across the five seeds except the explicitly
labelled counts and maxima. `C/native` and `R/native` are the ratio of the
candidate median to the native median.

| Condition | Median \(C\) | `C/native` | Seeds with \(C\ge3C_n\) | Median \(R\) | `R/native` | Seeds with \(R\ge3R_n\) | Ridge eval. | Median \(O\) | Hidden density | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| native | 0.00932056 | 1.000 | 0/5 | 0.146212 | 1.000 | 0/5 | 0.560 | 0.119019 | 0.823530 | control only |
| `rho_0_60` | 0.0162377 | 1.742 | 0/5 | 0.142516 | 0.975 | 0/5 | 0.600 | 0.0971363 | 0.866261 | fail |
| `rho_0_80` | 0.0340370 | 3.652 | 4/5 | 0.142507 | 0.975 | 0/5 | 0.915 | 0.144056 | 0.879716 | fail |
| `rho_0_95` | 0.138893 | 14.902 | 5/5 | 0.317838 | 2.174 | 2/5 | 0.995 | 0.391390 | 0.885408 | fail |
| `rho_1_05` | 0.320617 | 34.399 | 5/5 | 0.468447 | 3.204 | 3/5 | 1.000 | 0.573720 | 0.889581 | fail |

Hidden density is hidden emissions divided by all possible hidden node-ticks
on the task. Native was already dense at about 82.4%, rising to about 89.0% at
`rho_1_05`. Therefore this sweep supports no sparse-efficiency claim.

### Activity and blank-tail guards

| Condition | Maximum split total-event ratio | Maximum matched-episode ratio | Final 64 blank ticks quiet for all seeds and signs? |
|---|---:|---:|---|
| `rho_0_60` | 1.060707 | 1.160920 | yes |
| `rho_0_80` | 1.081220 | 1.180769 | yes |
| `rho_0_95` | 1.090656 | 1.217308 | yes |
| `rho_1_05` | 1.095162 | 1.226923 | **no** |

All four conditions stayed under the 1.10 split-total limit, but all four
exceeded the stricter 1.15 matched-episode limit at least once. At
`rho_1_05`, seed 50 became quiet by blank tick 133, while seeds 51--54 were
still active at tick 256 for both cue signs. Their final-64 hidden-event totals
per sign were 4,096, 4,070, 4,068, and 4,091 respectively, out of a maximum
4,096. This is near-continuous autonomous firing rather than controlled memory.

### Exact failed gates by target

- `rho_0_60` failed both absolute-cue threefold gates, all three ratio gates,
  ridge accuracy, output ratio, and the per-episode activity guard.
- `rho_0_80` passed the absolute-cue and ridge gates, but failed all three
  ratio gates, output ratio, and the per-episode activity guard.
- `rho_0_95` passed the absolute-cue, ridge, and blank-tail gates, but failed
  all three ratio gates, output ratio, and the per-episode activity guard.
- `rho_1_05` passed median absolute-cue growth, median threefold ratio growth,
  ridge accuracy, and output ratio. It still had median \(R=0.468447<0.50\),
  only 3 of 5 seeds reached threefold \(R\) rather than the required 4,
  matched-episode activity exceeded 1.15, and four seeds failed the blank-tail
  cessation test.

All frozen-state, finite-value, topology, weight, registered-condition,
target-radius, and no-update invariants passed. The native control was valid.
The global blank-tail integrity field is false only because `rho_1_05` failed
that candidate-specific stability assay.

## Scientific interpretation

The sweep reveals a real signal-versus-noise tradeoff rather than a hidden
success. Increasing recurrent radius made the absolute cue difference much
larger: `rho_0_95` and `rho_1_05` reached roughly 14.9 and 34.4 times the
native median. But the same stronger recurrence amplified unrelated variation
and event activity. Consequently, the clean cue/noise ratio grew too slowly,
and the strongest loop crossed into persistent autonomous firing on four
seeds.

The high ridge accuracies at radii 0.80--1.05 show that some cue information is
linearly decodable from the measured features. They do not override the frozen
requirements for absolute separation, seed robustness, bounded activity, and
cessation. Those independent gates prevent a classifier from making an
unstable, densely firing state look like useful memory.

The supported conclusion is narrow and valuable:

> Under this graph, data protocol, and tested range, uniform recurrent-weight
> scaling alone is not an acceptable memory repair.

The result does **not** rule out recurrent computation in general, local state
timescales, explicit decay variables, gated event memory, or a different
topology. It also says nothing yet about whether a future CCF-v1 rule learns
better, because CCF was deliberately inactive here.

## Reproducibility and deterministic rerun

The official full run and independent rerun both produced deterministic
payload SHA-256:

`f6ba680f91917521da5bbfb89d7d1dbd102fe59149049d0ed0464bb3ece7f8e6`

The complete JSON files are not byte-identical because machine/runtime
provenance is intentionally outside that deterministic payload. The numerical
payload, gate decisions, status, and selection are identical. Runtime was
180.7340942 seconds for the first run and 183.2904476 seconds for the rerun on
CPython 3.14.6 with NumPy 2.5.1, float64, CPU, and no autograd or GPU.

The two report sidecars were independently read and matched against freshly
computed SHA-256 values:

- `artifacts/experiment_000/memory_sweep/custom_seeds_50_54_pairs_100.json`
  -- `5dd993ba62770ce41ee84a8c8a42b80c642445a734fa4fddd537d2500d641a4c`;
- `artifacts/experiment_000/memory_sweep/custom_seeds_50_54_pairs_100_rerun.json`
  -- `1be4397c94dc28d25d70f0a3b722d4e4530a2c86581010dfbd1b171b95b245c5`.

The runner's source-provenance bundle SHA-256 was
`0eefc6466b33872df9bf2dbc3ee06b7d45e689e3b1be3236c8e32c477353bc48`.
It records the frozen pre-run protocol bytes as SHA-256
`4ad812ab303d98ee5449f7db865e453089b821dc284c5b4020bc6e0c00443d3b`.
The protocol document now contains a clearly labelled post-result status
appendix, so its present file hash is expected to differ from that frozen hash.
The annotated protocol's current SHA-256 is
`6ca239c6caa3669861deaa7273452e610f7f06de62bc93618536a2cd3183c496`.

## Decision and next controlled step

**Decision: `NO_SELECTION`.** The native architecture remains the project
state. No swept target may advance to alignment or learning, including the
best-looking `rho_1_05` condition.

The next permitted scientific step is a separately preregistered
architecture-only intervention that introduces a **local state timescale** (or
equivalent bounded event-memory state) while keeping the CCF equation fixed.
It must be evaluated on fresh custom seeds with explicit signal, noise,
activity, cessation, and determinism gates before any learning claim.
