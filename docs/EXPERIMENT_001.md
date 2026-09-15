# Experiment 001: Delayed Micro-Programs and Continual Composition

**Status:** preregistered protocol; results must not be added to this document  
**Scope:** fixed sparse topology; learning-rule test only  
**Required gate:** Experiment 000 must pass before the confirmatory Experiment 001 run

This document fixes what we will train, what we will measure, and what counts as success **before** looking at confirmatory results. If the mechanism, data stream, threshold, or analysis changes after a confirmatory run begins, the changed study receives a new experiment version and fresh confirmatory seeds.

> **Simple explanation:** We are writing the exam and its pass marks before the system takes the exam. That prevents us from moving the goalposts after seeing the scores.

---

## 1. Questions being tested

Experiment 001 asks whether Causal Credit Flow v0 (CCF-v0) can:

1. learn two small operations from delayed terminal error;
2. learn a third operation later without replaying examples of the first two;
3. retain the first two operations;
4. execute unseen two-operation programs by composing learned operations;
5. attribute delayed error to the pathways that actually participated;
6. do at least one important thing better than compute-matched learned baselines.

This experiment does **not** test graph growth or pruning. The sparse graph is generated before training and then frozen. That isolation is deliberate: topology changes must not hide a failure of the proposed credit rule.

---

## 2. Frozen task definitions

### 2.1 Bit and order convention

A word is

\[
x=(x_2,x_1,x_0), \qquad x_i\in\{0,1\}.
\]

`x0` is the rightmost, least-significant bit. The three primitive operations are

\[
\begin{aligned}
\operatorname{FLIP0}(x_2,x_1,x_0) &= (x_2,x_1,1-x_0),\\
\operatorname{ROTL}(x_2,x_1,x_0)  &= (x_1,x_0,x_2),\\
\operatorname{NOT}(x_2,x_1,x_0)   &= (1-x_2,1-x_1,1-x_0).
\end{aligned}
\]

An operation sequence is applied from left to right. For example,

\[
[\operatorname{FLIP0},\operatorname{ROTL}](x)
=\operatorname{ROTL}(\operatorname{FLIP0}(x)).
\]

> **Simple explanation:** `FLIP0` changes only the rightmost bit, `ROTL` moves every bit one place to the left in a circle, and `NOT` reverses all three bits. “Left to right” means do the first named operation first.

### 2.2 Episode format

Every episode begins from a reset recurrent state and uses this event sequence:

1. `START(x2,x1,x0)` presents the input word.
2. One operation token is presented in primitive-training episodes; two operation tokens are presented in composition tests.
3. The environment presents \(D\) independent distractor events `NOISE(n2,n1,n0)`, where every noise bit is sampled independently from Bernoulli\((0.5)\).
4. `QUERY` asks the network for a three-bit answer.
5. Only after the answer, the environment reveals the terminal target and error.

The target for a program \(P=[o_1,\ldots,o_m]\) is

\[
y=P(x)=o_m(\cdots o_2(o_1(x))\cdots).
\]

At `QUERY`, output unit \(k\) produces its forced `UnitEvent` activation
\(a_k\in[-1,1]\). A task bit \(y_k\in\{0,1\}\) is represented internally by
the bipolar target \(r_k=2y_k-1\), and the reported bit is

\[
\hat y_k=\mathbb{1}[a_k\geq 0].
\]

The bounded terminal activation credit available to a learning method is

\[
c_k=\operatorname{clip}(r_k-a_k,-1,1).
\]

No target, loss, correctness flag, or partial answer is supplied before `QUERY`. CCF-v0 injects this value as a `CreditPacket` at the corresponding forced output `UnitEvent`, then routes it through immutable local `EdgeTrace` records according to `docs/CCF_V0.md`. The eligibility-trace baseline receives the same terminal activation residual. BPTT uses only the terminal squared activation loss \(\tfrac12\sum_k(r_k-a_k)^2\); it receives no intermediate loss.

> **Simple explanation:** The system sees the input and instruction, then irrelevant noise, and only at the end learns whether each answer bit was too high or too low. It must connect that late correction to earlier useful activity.

### 2.3 Delay schedule

- Training: \(D\sim\operatorname{UniformInteger}\{4,5,6,7,8\}\).
- Primary evaluation: \(D=8\).
- Secondary delay-generalization evaluation: \(D=16\).
- A zero-delay evaluation is diagnostic only and cannot satisfy a pass criterion.

The token type has a separate channel, so noise payloads can never accidentally become operation or query tokens.

---

## 3. Fixed sparse network and fairness constraints

The benchmark network has 64 recurrent hidden units and three output units. For each confirmatory seed:

- every hidden unit receives exactly 8 recurrent hidden-to-hidden edges;
- every input channel projects to exactly 8 hidden units;
- every output unit receives exactly 16 hidden inputs;
- recurrent self-edges are excluded;
- every graph edge has a fixed one-tick delay;
- a deterministic connectivity repair performed **before initialization** ensures that every hidden unit has at least one outgoing recurrent or output route;
- the adjacency masks, unit count, and input/output channels never change during either phase.

All learned methods in a paired seed receive the same graph mask, initial forward weights, episode order, words, delays, distractors, evaluation cases, and parameter-count budget. Method-specific learning state, such as an eligibility trace, may differ. All operation input channels, including `NOT`, exist from initialization; the `NOT` channel is simply unused in Phase A. Recurrent activations and all episode-local `UnitEvent`, `EdgeTrace`, eligibility-trace, and `CreditPacket` state are cleared at every episode boundary; learned weights persist.

The precise CCF-v0 state equations, credit packets, local information boundary, clipping, and update rule are frozen in `docs/CCF_V0.md`. If that specification conflicts with this protocol, the run is blocked until a dated amendment states which definition governs; silent interpretation is forbidden.

> **Simple explanation:** Every learner gets the same small “brain” and the same experiences. We lock the wiring so this experiment measures the learning rule, not the ability to add more wiring.

### 3.1 Compute ledger

Every implementation must count, per episode:

1. active unit updates;
2. forward edge touches;
3. backward or credit-routing edge touches;
4. parameter-write edge touches;
5. peak bytes of learning state;
6. wall-clock time as a secondary hardware-dependent measure.

One edge touch means reading an edge and using it in one forward, backward/credit, or parameter-update calculation. Counts, not dense theoretical matrix sizes, are primary for sparse methods.

The episode-matched BPTT run sees all training episodes. A second `BPTT-CM` checkpoint is compute matched: for seed \(s\), BPTT training stops before its cumulative total edge touches would exceed the total used by CCF-v0 for that seed. Both BPTT results are reported; only `BPTT-CM` is used for a compute-matched superiority claim.

---

## 4. Experiment 000: delayed-chain smoke test

Experiment 000 is a required, smaller gate. It tests whether a terminal signal can train a dependency that began many events earlier.

### 4.1 Task

1. At \(t=0\), present a balanced binary cue \(c\in\{0,1\}\).
2. Present exactly eight independent binary noise events.
3. Present `QUERY`.
4. Target \(y=c\); reveal the terminal residual only after the prediction.

The train stream contains 2,000 episodes: exactly 1,000 per cue, shuffled within the seed. Evaluation contains 2,000 fresh episodes: 1,000 per cue with unseen distractor streams. The primary metric is binary exact accuracy at delay 8. Delay 16 is secondary. `QUERY` is learner-visible at event time 9; because every graph edge has a one-tick delay, the fixed readout adapter advances two additional propagation ticks and forces the scalar output at graph tick 11. The target remains unavailable until that forced output exists.

### 4.2 Gate criteria

Across the 20 confirmatory seeds, Experiment 000 passes only if all conditions hold:

1. mean CCF-v0 delay-8 accuracy is at least **0.90**;
2. at least **16 of 20** seeds reach accuracy at least **0.85**;
3. CCF-v0 exceeds the uniform-random baseline by at least **0.25** mean absolute accuracy and the multiplicity-adjusted 95% paired confidence interval excludes zero;
4. CCF-v0 exceeds `CCF-NO-TRACE` by at least **0.15** mean absolute accuracy and the adjusted 95% paired confidence interval excludes zero;
5. all graph masks remain unchanged and all 20 CCF runs finish without a non-finite state.

If any gate condition fails, the confirmatory Experiment 001 run does not start. Diagnosis may use development seeds, but any fix creates a new CCF version, a new protocol amendment, and unused confirmatory seeds.

> **Simple explanation:** Before asking the system to learn programs, we first check that it can remember one useful bit across eight pieces of noise and learn from a correction that arrives only at the end.

---

## 5. Experiment 001 training sequence

### 5.1 Phase A — learn the first two operations

Train for exactly 4,000 single-operation episodes:

- 2,000 `FLIP0` episodes;
- 2,000 `ROTL` episodes;
- within each operation, each of the eight input words occurs exactly 250 times;
- the resulting balanced multiset is shuffled once by the paired seed;
- delay and noise follow Section 2.3.

Save a read-only Phase-A checkpoint. Evaluate it without learning on the two primitives and, diagnostically, on the four length-two programs made from `FLIP0` and `ROTL`.

### 5.2 Phase B — add `NOT` without replay

Continue from the Phase-A checkpoint for exactly 2,000 single-operation `NOT` episodes. Each input word occurs exactly 250 times.

During Phase B:

- no `FLIP0` or `ROTL` training episode is allowed;
- no stored Phase-A activation, target, gradient, trace, or example may be replayed;
- no synthetic example of an old operation may be generated;
- the learner receives only the current `NOT` episode and its terminal error;
- the topology and unit count remain fixed.

After Phase B, evaluate without learning on all three primitives and all nine ordered length-two programs:

\[
\mathcal C=\{[a,b]:a,b\in\{\operatorname{FLIP0},\operatorname{ROTL},\operatorname{NOT}\}\}.
\]

None of the two-operation programs appears in training.

> **Simple explanation:** First the system learns two skills. Then it sees only the new third skill. We check whether the old skills survived and whether the system can chain any two skills even though it was trained only on one instruction at a time.

### 5.3 Evaluation cases

For each primitive or composition program, each evaluation set contains

\[
8\text{ input words}\times64\text{ independent distractor streams}=512\text{ episodes}.
\]

Model state and all episode-local learning state are reset between evaluation episodes. Weights do not update during evaluation. Primary sets use \(D=8\); separate secondary sets use \(D=16\). Evaluation random streams are disjoint from training and development streams.

Phase-B learning curves are measured after 0, 50, 100, 200, 400, 800, 1,200, 1,600, and 2,000 episodes. Evaluation targets from these checkpoints are never delivered to the learner.

---

## 6. Compared methods

### 6.1 Required methods

1. **CCF-v0:** the full frozen rule in `docs/CCF_V0.md`.
2. **ET-3F:** conventional local three-factor eligibility-trace learning on the same forward graph. For edge \(i\to j\),
   \[
   e_{ij}(t)=\lambda e_{ij}(t-1)+x_{ij}(t)\phi'(u_j(t)),
   \]
   and at terminal feedback,
   \[
   \Delta w_{ij}=\eta\,L_j\,e_{ij},\qquad
   L_j=\sum_k B_{jk}c_k,
   \]
   where \(B\) is a fixed seeded random feedback matrix. Its scale is normalized once at initialization and is not trained.
3. **BPTT:** same event-driven forward graph and terminal loss, unrolled through the complete episode, trained by global backpropagation through time. The exact Boolean emit/no-emit decisions realized on the forward pass are treated as constants during the backward pass (zero derivative through the threshold); gradients pass through the `tanh` values and realized active edges. No surrogate gradient is used. Report both episode-matched `BPTT` and edge-touch-matched `BPTT-CM`.
4. **RAND:** at `QUERY`, sample uniformly from the eight possible three-bit words; expected exact accuracy is \(1/8=0.125\). It receives no training.

ET-3F is the closest standard local delayed-credit comparison. BPTT is a global-credit reference, not evidence that the CCF locality constraint was met. RAND verifies that task generation and scoring do not accidentally leak the answer.

### 6.2 Hyperparameter fairness

Before confirmatory execution, each learned method may use at most 24 hyperparameter configurations on development seeds 0–4. Selection maximizes the development version of the bottleneck score in Section 8.4; a non-finite run scores zero. Search spaces, selected values, software revision, and configuration hashes must be written into the run manifest before any confirmatory result is inspected. Confirmatory seeds are never used for tuning or early stopping.

All methods receive the same maximum tuning count, data, forward graph, and initialization family. BPTT may use method-appropriate optimizer parameters; CCF and ET may use method-appropriate local-rule parameters.

---

## 7. Required CCF ablations

Every ablation uses the same paired seeds, graph masks, streams, episode counts, clipping rules, and tuning budget as full CCF-v0. Exactly one named mechanism changes.

1. **CCF-NO-TRACE:** clear every causal trace before terminal credit is applied. Current-query activity may remain, but earlier participation cannot be credited.
2. **CCF-BROADCAST:** replace pathway-selective credit routing with a normalized terminal signal broadcast to all eligible hidden units; preserve trace values and total credit norm.
3. **CCF-SHUFFLED-CAUSE:** before each terminal update, deterministically permute edge identities within each trace age. This preserves the number, age, and magnitude of trace entries while breaking which edge caused which event.
4. **CCF-POSITIVE-ONLY:** clamp negative credit components to zero while leaving positive credit unchanged.
5. **CCF-IMMEDIATE:** provide an oracle target immediately after each operation token. This is a positive control showing how much difficulty comes from delayed feedback; it is not eligible for a superiority claim because it solves an easier task.

> **Simple explanation:** An ablation removes or scrambles one part. If full CCF performs well but performs the same after its causal traces are removed or shuffled, then the proposed explanation is not supported.

---

## 8. Metrics and formulas

The unit of statistical inference is the **seed**, not an individual episode.

### 8.1 Exact accuracy

For \(N\) episodes,

\[
A=\frac{1}{N}\sum_{n=1}^{N}\mathbb{1}[\hat y^{(n)}=y^{(n)}].
\]

All three bits must be correct. Per-bit accuracy is reported only as a diagnostic.

### 8.2 Continual-learning quantities

For seed \(s\), define

\[
\begin{aligned}
A_A(s)&=\tfrac12(A_{\mathrm{FLIP0}}^{A}+A_{\mathrm{ROTL}}^{A}),\\
A_{new}(s)&=A_{\mathrm{NOT}}^{B},\\
A_{ret}(s)&=\tfrac12(A_{\mathrm{FLIP0}}^{B}+A_{\mathrm{ROTL}}^{B}),\\
F(s)&=\max(0,A_A(s)-A_{ret}(s)).
\end{aligned}
\]

Here, \(F\) is forgetting; smaller is better. Also report signed backward transfer

\[
BWT(s)=A_{ret}(s)-A_A(s).
\]

### 8.3 Composition

For the nine unseen ordered programs,

\[
A_{comp}(s)=\frac{1}{9}\sum_{P\in\mathcal C}A_P(s).
\]

Every program is also reported separately so a high average cannot conceal one failed composition.

### 8.4 Bottleneck score

The primary combined score is

\[
S(s)=\min\{A_A(s),A_{new}(s),A_{ret}(s),A_{comp}(s)\}.
\]

> **Simple explanation:** The score equals the weakest essential ability. Excellent new learning cannot hide catastrophic forgetting, and good memory cannot hide failed composition.

### 8.5 Sample and compute efficiency

- **E90:** first Phase-B checkpoint where `NOT` accuracy is at least 0.90 and remains at least 0.90 at the next checkpoint. If never reached, record `>2000` and treat it as right-censored.
- **Active fraction:** active hidden-unit updates divided by all possible hidden-unit updates.
- **Edge touches:** the separate forward, credit/backward, and write counts in Section 3.1.
- **Memory:** peak bytes of method-specific learning state, excluding shared model parameters.

### 8.6 Stability and integrity

Report non-finite values, clipped updates, maximum absolute weight, silent episodes, and exact pre/post graph-mask hashes. A non-finite confirmatory seed remains in analysis with all later accuracies set to zero; it is never silently removed.

---

## 9. Confirmatory seeds and randomization

- Development seeds: `0, 1, 2, 3, 4`.
- Confirmatory paired seeds: `1000, 1001, ..., 1019` (20 seeds).
- Analysis/bootstrap RNG seed: `20260901`.

Each master seed deterministically derives named substreams for graph mask, weight initialization, Phase-A order, Phase-B order, delays, distractors, evaluation, ablation shuffle, and random-baseline actions. A stored seed manifest must make the substreams auditable.

The word/operation counts are exactly balanced before seeded shuffling. All methods for seed \(s\) use the same applicable substreams. Runs may be scheduled in any wall-clock order, but result files are analyzed only after all required methods and seeds are complete.

---

## 10. Explicit verdict rules

All thresholds use unrounded seed-level values at primary delay \(D=8\).

### 10.1 Core behavior gate

CCF-v0 passes the core behavior gate only if all conditions hold across the 20 confirmatory seeds:

1. \(\operatorname{mean}(A_A)\geq\mathbf{0.95}\);
2. \(\operatorname{mean}(A_{new})\geq\mathbf{0.90}\);
3. \(\operatorname{mean}(A_{ret})\geq\mathbf{0.90}\);
4. \(\operatorname{mean}(F)\leq\mathbf{0.05}\);
5. \(\operatorname{mean}(A_{comp})\geq\mathbf{0.80}\);
6. the mean accuracy of **each** of the nine composition programs is at least **0.65**;
7. at least **15 of 20** seeds individually satisfy \(A_A\geq0.90\), \(A_{new}\geq0.85\), \(A_{ret}\geq0.85\), \(F\leq0.10\), and \(A_{comp}\geq0.70\);
8. every pre/post graph-mask hash matches and all protocol integrity checks pass.

### 10.2 Mechanism gate

Full CCF-v0 must beat its ablations in paired seed analysis:

- mean \(S_{CCF}-S_{NO\text{-}TRACE}\geq\mathbf{0.10}\);
- mean \(S_{CCF}-S_{SHUFFLED\text{-}CAUSE}\geq\mathbf{0.10}\);
- mean \(S_{CCF}-S_{BROADCAST}\geq\mathbf{0.05}\).

For every listed contrast, the Holm-adjusted 95% paired confidence interval must exclude zero in the positive direction. `POSITIVE-ONLY` and `IMMEDIATE` are mandatory diagnostics but have no fixed effect-size gate.

### 10.3 Comparative gate

Against **each** compute-matched learned baseline (`ET-3F` and `BPTT-CM`), CCF-v0 must satisfy at least one of these preregistered important wins, with Holm-adjusted paired evidence in the stated direction:

1. **Capability:** mean \(S\) is at least **0.05** higher; or
2. **Experience efficiency:** median E90 is at least **25% lower**, while the lower confidence bound for \(S_{CCF}-S_{baseline}\) is no worse than \(-0.02\); or
3. **Compute efficiency:** median total edge touches are at least **2.0× lower**, while the lower confidence bound for \(S_{CCF}-S_{baseline}\) is no worse than \(-0.02\).

The episode-matched BPTT result is always reported but does not replace `BPTT-CM` in this gate.

### 10.4 Final label

- **STOP — SMOKE FAIL:** Experiment 000 fails; Experiment 001 confirmatory execution is not run.
- **FAIL:** Experiment 001 core behavior gate fails.
- **BEHAVIOR ONLY:** core behavior passes but the mechanism gate fails.
- **MECHANISM-PROMISING:** core behavior and mechanism gates pass, but the comparative gate fails.
- **EXPERIMENT 001 PASS:** core behavior, mechanism, and comparative gates all pass.

No other wording counts as a pass. Secondary delay-16 results, attractive visualizations, or a single exceptional seed cannot override these labels.

---

## 11. Statistical analysis plan

1. Aggregate episode outcomes within each method/seed/task first.
2. Report the mean, median, standard deviation, and 95% bootstrap confidence interval across seeds.
3. Use 10,000 paired bootstrap resamples of the 20 seed indices for method differences. Use the fixed analysis seed in Section 9.
4. Use paired sign-flip permutation tests for confirmatory method contrasts; use 100,000 Monte Carlo sign flips when exact enumeration is not used.
5. Apply Holm correction at familywise \(\alpha=0.05\) to:
   - the two Experiment-000 gate contrasts;
   - the three mechanism-gate contrasts;
   - the ET-3F and BPTT-CM comparative contrasts for each claimed outcome.
6. Report raw and adjusted p-values, paired mean/median differences, confidence intervals, and the number of seeds favoring each method.
7. Do not discard seeds, episodes, or outliers. Corrupt result files are regenerated from the recorded configuration; algorithmic crashes remain failures as specified in Section 8.6.
8. Treat delay 16, bit accuracy, wall-clock time, Phase-A composition, and all unregistered slices as secondary or exploratory. Label them accordingly.

Because E90 is interval-measured at checkpoints and may be censored, compare it with paired checkpoint ranks and also report Kaplan-Meier time-to-90% curves. Do not replace the preregistered checkpoints with interpolated crossing times.

---

## 12. Required integrity checks

Before results are unblinded, automated tests must verify:

- all eight words map correctly under every primitive and all nine length-two programs;
- operation order is left to right;
- training and evaluation distractor streams are disjoint;
- Phase A is exactly balanced and contains only `FLIP0`/`ROTL` single-operation episodes;
- Phase B is exactly balanced and contains only `NOT` single-operation episodes;
- no evaluation call updates weights or persistent learning state;
- no Phase-A item enters the Phase-B learner input or any replay buffer;
- every paired method receives byte-identical episode descriptions;
- topology hashes remain constant;
- CCF/ET code paths do not invoke autodiff or BPTT;
- BPTT receives loss only at the terminal output;
- all operation channels exist before Phase A;
- the compute ledger reconciles with a hand-counted tiny run.

The experiment record must include source commit, environment lock hash, configuration hashes, seed manifest, hardware description, raw per-episode outputs, aggregate tables, and logs from these integrity tests.

---

## 13. Result-table templates

### 13.1 Main outcomes at delay 8

| Method | Phase-A old \(A_A\) | New `NOT` \(A_{new}\) | Retained old \(A_{ret}\) | Forgetting \(F\) | Composition \(A_{comp}\) | Bottleneck \(S\) | E90 | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| CCF-v0 | — | — | — | — | — | — | — | — |
| ET-3F | — | — | — | — | — | — | — | — |
| BPTT-CM | — | — | — | — | — | — | — | — |
| BPTT episode-matched | — | — | — | — | — | — | — | reference |
| RAND | — | — | — | — | — | — | n/a | reference |

Each numeric cell must contain `mean [95% CI]`; E90 also reports the censored count.

### 13.2 Primitive retention

| Method | `FLIP0` after A | `ROTL` after A | `FLIP0` after B | `ROTL` after B | `NOT` after B | BWT |
|---|---:|---:|---:|---:|---:|---:|
| CCF-v0 | — | — | — | — | — | — |
| ET-3F | — | — | — | — | — | — |
| BPTT-CM | — | — | — | — | — | — |

### 13.3 All unseen ordered compositions

| Method | F→F | F→R | F→N | R→F | R→R | R→N | N→F | N→R | N→N | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CCF-v0 | — | — | — | — | — | — | — | — | — | — |
| ET-3F | — | — | — | — | — | — | — | — | — | — |
| BPTT-CM | — | — | — | — | — | — | — | — | — | — |

`F`, `R`, and `N` abbreviate `FLIP0`, `ROTL`, and `NOT`. `F→R` means apply `FLIP0` first and `ROTL` second.

### 13.4 Mechanism ablations

| Variant | \(A_A\) | \(A_{new}\) | \(A_{ret}\) | \(A_{comp}\) | \(S\) | Paired ΔS vs full | Adjusted 95% CI | Adjusted p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CCF-v0 | — | — | — | — | — | n/a | n/a | n/a |
| CCF-NO-TRACE | — | — | — | — | — | — | — | — |
| CCF-BROADCAST | — | — | — | — | — | — | — | — |
| CCF-SHUFFLED-CAUSE | — | — | — | — | — | — | — | — |
| CCF-POSITIVE-ONLY | — | — | — | — | — | — | — | — |
| CCF-IMMEDIATE | — | — | — | — | — | — | — | — |

### 13.5 Compute and stability

| Method | Active-unit fraction | Forward touches | Credit/backward touches | Write touches | Peak learning-state bytes | Non-finite seeds | Mask hash match |
|---|---:|---:|---:|---:|---:|---:|---|
| CCF-v0 | — | — | — | — | — | — | — |
| ET-3F | — | — | — | — | — | — | — |
| BPTT-CM | — | — | — | — | — | — | — |
| BPTT episode-matched | — | — | — | — | — | — | — |

### 13.6 Experiment 000 gate

| Method | Delay-8 accuracy | Seeds ≥ 0.85 | Δ vs RAND | Δ vs NO-TRACE | Non-finite seeds | Gate |
|---|---:|---:|---:|---:|---:|---|
| CCF-v0 | — | —/20 | — | — | — | — |
| CCF-NO-TRACE | — | —/20 | — | n/a | — | reference |
| RAND | — | —/20 | n/a | — | 0 | reference |

---

## 14. Decision after the experiment

- A smoke failure sends us back to the delayed-credit equation and implementation tests.
- A core failure rejects CCF-v0 for this benchmark; graph growth is not added to rescue it.
- A mechanism failure means observed behavior cannot be attributed to the proposed causal trace/routing components.
- A comparative failure means the mechanism may work but has not yet earned a superiority claim.
- Only a full pass justifies the next controlled experiment with topology growth and pruning.

> **Simple explanation:** A negative result is useful. It tells us exactly whether the problem was basic delayed learning, memory, composition, the claimed causal mechanism, or competitiveness with standard methods.
