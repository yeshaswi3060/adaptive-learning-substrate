# Sequence architecture progress — 2026-09-05

## Objective

Build a competitive replacement for Transformer sequence modeling, not a wrapper.
This page records implementation progress separately from official research
readiness, which remains **21/100**.

## Implemented now

- A native 256-byte vocabulary sequence core, with 32 persistent gated units and
  four recurrent parents per unit; 25,056 trainable parameters.
- Trainable input/gate lookups, sparse recurrence, self-gating, biases and output
  decoder. Updates use a bounded local-eligibility/random-feedback approximation,
  not attention, autograd or a temporal global backward pass.
- CPU reference and direct CUDA implementation. No torch import, no growing
  key/value cache, no task-ID routing, no hard-coded first-cue memory.
- Stateful next-byte probabilities; prediction-before-target learning; explicit
  resets/cancellation; checked cross-backend checkpoints and continuation.
- Source/data freezes, three-seed development training, exact fresh-process
  repetition, and full independent CPU replay of the training and held-out data.

Sparse here describes recurrent connectivity. Every unit currently updates on
every token; event-sparse execution and autonomous topology changes are not built.
These equations combine established recurrent/feedback/eligibility ingredients;
no algorithmic novelty is established.

## Measured small-text result

Training used 4,096 total bytes across two project documents (4,094 next-byte
updates per model). A different 1,024-byte document supplied 1,023 held-out pairs.
This is an engineering corpus, not a representative language-model benchmark.

| Model | Held-out nats/byte (lower is better) |
|---|---:|
| Uniform bytes | 5.5452 |
| Frozen recurrent core, trained readout | 3.9227 (three-seed mean) |
| New locally trained recurrent core | 3.7215 (three-seed mean) |
| Laplace-smoothed bigram baseline | **3.4888** |

Local core learning improved probability loss over readout-only learning for
each seed, but **did not beat the bigram baseline**. Byte argmax accuracy also
does not establish an advantage. No Transformer or state-space baseline has
been run, and no useful long-range-memory or continual-retention claim follows.

The fresh GPU rerun matched exactly. Independent CPU replay reproduced all
24,564 training updates across six models, with maximum complete-buffer/held-out
probability discrepancy approximately 1.34e-15. Engineering parity covered
640 token steps and 128 updates, plus checkpoint/reset and constant-allocation
checks over 4,096 further tokens in each mode. Full suite: **590 passed**.

Peak observed GPU-run process working set was about **218 MiB RAM**. The fixed
model/state/eligibility buffer uses 401,152 bytes on GPU; indices add 512 bytes.
Reset/checkpoint staging temporarily uses a second copy. These counts exclude
driver/compiler/context costs. The startup guard remains 512 MiB available host
RAM, with runtime boundary checks at 128 MiB. Resource tests do not guarantee
zero issues under arbitrary competing application load.

## Next experiments toward the replacement objective

1. Delayed multi-token copy and keyed associative recall, with a state-reset
   ablation, to test whether recurrence supplies useful memory rather than only
   improved short-range byte statistics.
2. A gradient-trained reference of the same core to separate architecture capacity
   from local-credit limitations; this is a comparison, not a change of the eventual
   no-global-backprop objective.
3. A broader document-separated corpus and matched tiny Transformer/recurrent/
   state-space baselines with frozen data, tuning, compute and quality metrics.
4. Only then test domain adaptation/forgetting and structural changes.

The older architecture-target document was preserved because its exact bytes
are frozen held-out data for this experiment; this page updates implementation
status without changing that dataset or retroactively modifying the experiment.

## Commands and source

Run the documented experiment only into a new output directory; existing records
are never overwritten. The runner enforces source/resource checks and writes a
fresh prospective freeze before measuring results.

```powershell
$env:PYTHONPATH='C:/Users/Yash/Desktop/LifelongLearningProject/src'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
& C:/Users/Yash/miniconda3/python.exe -B -m adaptive_learning_substrate.sequence_experiment run --output C:/Users/Yash/Desktop/LifelongLearningProject/artifacts/sequence_core_v1_2026-09-05/new_run
```

Implementation: `src/adaptive_learning_substrate/sequence_core.py`,
`sequence_cuda.py`, and `sequence_experiment.py`. Protocol: `SEQUENCE_CORE_V1.md`.
Recorded results: `artifacts/sequence_core_v1_2026-09-05/`.


## Memory diagnostic update — 2026-09-06

The frozen sequence-memory V1 experiment is complete. **The current candidate
failed the multi-token-memory diagnostic.** This is a scientific negative result,
not another recovery-only repair and not an increase in readiness.

The experiment used three seeds, separate copy3 and recall4 tasks, 512 training
examples per seed/task, and 128 held-out examples at each delay 4, 16, 32, 64.
Training saw only delays 4/16. All conditions used the same unchanged 32-unit
recurrence and initialization; local/readout learning were compared with a new
finite-difference-checked exact BPTT/Adam reference. Matched data/target counts
do not imply matched optimizer steps or compute.

Means across all three seeds and four delays (all per-cell results retained):

| Task | Training | Token accuracy | Exact answer | NLL | Normal minus reset |
|---|---|---:|---:|---:|---:|
| copy3 | local | 12.85% | 0.13% | 2.7911 | +0.30 pp |
| copy3 | readout | 13.09% | 0.07% | 3.0109 | +0.50 pp |
| copy3 | bptt | 12.41% | 0.46% | 2.1083 | -0.24 pp |
| copy3 | bigram | 12.59% | 0.33% | 2.9208 | n/a |
| recall4 | local | 12.37% | 12.37% | 4.1539 | -0.07 pp |
| recall4 | readout | 12.63% | 12.63% | 4.2027 | +0.20 pp |
| recall4 | bptt | 13.35% | 13.35% | 2.1335 | +0.00 pp |
| recall4 | bigram | 10.74% | 10.74% | 3.1474 | n/a |


Eight-value random guessing gives expected token accuracy 12.5%. Copy-exact
guessing gives 0.1953%. The candidate's token accuracy was near that descriptive
reference, and resetting its recurrent state immediately before the query made
little average difference. Exact-gradient training substantially reduced NLL,
but also failed the fixed memory thresholds. Within this budget, switching
training rules alone did not establish useful memory. These results do not prove
that the recurrence lacks capacity or that more training would fix it.

Two complete fresh-process runs matched exactly across 18 trained models,
184,320 input-token steps and 18,432 supervised answer tokens per run.
The third process independently replayed 18,432 normal/reset evaluation
episodes through the original CpuSequence implementation, with maximum target-
probability difference 0. The maximum primary-run process peak
working set was 58.66 MiB. No GPU, torch, or worker pool was used.

Scientific payload SHA-256: `02786f2648ef7acd0ed669ef46088807bad66f8de81fc59c8a4484cc23f3e55b`.
Protocol: `docs/SEQUENCE_MEMORY_V1.md`. Complete evidence and per-seed/per-delay
results: `artifacts/sequence_memory_v1_2026-09-06/`.

**Next architecture work:** prospectively register tiny-set overfitting and
zero-delay controls, then a controlled training-budget/state-capacity comparison.
First establish whether the gradient reference can fit the task and preserve
cue information; use that result to target the local-credit or recurrence change.
Avoid changing the failed V1 thresholds or treating lower NLL as successful
memory. Broader text and matched Transformer/state-space comparisons remain
required. Official readiness stays **21/100**; official V3 evidence is absent.


## Tiny-set trainability update — 2026-09-06

**The unchanged recurrence can fit these tiny tasks with exact-gradient training,
but the current local-learning rule did not fit them, and held-out performance
remains poor.** This advances the diagnosis, not the Transformer-replacement claim
or the official readiness score.

The new prospective V1 matrix used two seeds, copy3/recall4, sixteen balanced
training examples, and separately trained zero-/sixteen-distractor models. All
24 models received 2,048 presentations (128 complete epochs), with a predeclared
checkpoint at 512. Sixty-four distinct held-out examples were scored at both
delays. All three learning modes used the existing 32-unit, four-parent recurrence
and original hyperparameters. BPTT uses Adam/episode updates; local/readout use
their original answer-token updates, so this does not isolate locality alone.

At 512 presentations no mode passed the fixed >=95% training-exact criterion.
At 2,048 presentations, **BPTT fit 7/8 seed/task/delay cells; local fit 0/8 and
readout fit 0/8**. BPTT's remaining cell achieved 15/16 (93.75%), below the
unchanged threshold. All eight BPTT cells passed the separate >=50 percentage
point training-exact advantage over reset. Checkpoints on one trajectory are
correlated; they are not independent experimental replicates.

Final-checkpoint means across the two seeds, with training and held-out scores
kept separate (exact-answer accuracy, not token accuracy):

| Task | Learning | Training distractors | Train exact | Train reset exact | Held-out exact, same delay | Fit seeds |
|---|---|---:|---:|---:|---:|---:|
| copy3 | local | 0 | 3.12% | 0.00% | 0.00% | 0/2 |
| copy3 | local | 16 | 15.62% | 0.00% | 0.00% | 0/2 |
| copy3 | readout | 0 | 3.12% | 3.12% | 0.00% | 0/2 |
| copy3 | readout | 16 | 15.62% | 0.00% | 0.00% | 0/2 |
| copy3 | bptt | 0 | 96.88% | 6.25% | 1.56% | 1/2 |
| copy3 | bptt | 16 | 100.00% | 3.12% | 0.00% | 2/2 |
| recall4 | local | 0 | 56.25% | 31.25% | 17.19% | 0/2 |
| recall4 | local | 16 | 59.38% | 31.25% | 12.50% | 0/2 |
| recall4 | readout | 0 | 56.25% | 31.25% | 18.75% | 0/2 |
| recall4 | readout | 16 | 65.62% | 31.25% | 14.84% | 0/2 |
| recall4 | bptt | 0 | 100.00% | 31.25% | 14.84% | 2/2 |
| recall4 | bptt | 16 | 100.00% | 31.25% | 10.16% | 2/2 |


BPTT's zero-distractor copy exact score fell from 96.88% training to 1.56%
held-out; its sixteen-distractor copy score fell from 100% to 0%. Same-delay
BPTT recall fell from 100% training to 14.84%/10.16% held-out (delays 0/16).
This is a large observed generalization gap on these development tasks, not a
competitive sequence-model result. The eight-value recall guessing reference
is 12.5%, and three-value copy guessing exact accuracy is 0.1953%; no statistical
significance or population claim is made from two seeds.

The reset result establishes dependence on prior state, not causal use of the
correct cue. In particular, each sixteen-distractor training prompt repeats the
same noise; the model may memorize cue or noise signatures. Perfect fitting
therefore does not establish the intended copy/retrieval algorithm. Zero
distractors still leaves recurrent cue/query steps. Recall reset scores can
exploit finite-sample key/label associations despite globally balanced answers.

Both complete fresh-process runs matched exactly: 24 models, 48 checkpoints,
49,152 episode presentations,
884,736 input-token steps, and
98,304 supervised answer targets per run. A third
distinct process reproduced all 13,824
training/test normal/reset evaluation episodes through the original CpuSequence
engine, including every free prediction and teacher-forced target probability.
The primary run's maximum process peak working set was 52.10 MiB;
no GPU, torch, or worker pool was used. This is not a Transformer resource
comparison. Scientific payload SHA-256:
`068496bd091d248bb6c94e5e4f92b8c35685f35eb36679cbbdea47ed00eb8310`.

**Next architecture decision:** before increasing state width, register
resampled-distractor and cue-intervention controls plus a controlled learning-rule
comparison that separates optimizer/update timing from temporal credit. Preserve
these results, including the 15/16 failure. The same-architecture fitting result
argues against treating these failures as proven insufficient state capacity;
the poor held-out results require generalization work, not more memorization.
Broader document-separated text and matched Transformer/recurrent/state-space
comparisons remain necessary for the replacement objective.

Protocol: `docs/SEQUENCE_TRAINABILITY_V1.md`. All 288 evaluation cells, complete
datasets, schedules, checkpoints and command evidence are retained under
`artifacts/sequence_trainability_v1_2026-09-06/`. Original memory V1 and official
V3 files are unchanged. Official readiness remains **21/100**, not 25%; the
official V3 evidence chain remains absent.


## Local-credit and distractor-control update — 2026-09-06

**A local-eligibility + Adam variant learned noise-robust recall of known cues,
but not reliable recall of novel or changed cues.** The explicitly proposed
output-aligned variant did not outperform the fixed-feedback variant. The
Transformer-replacement objective remains unachieved; readiness remains 21/100.

### Code delivered and the controlled experiment
Added `sequence_credit.streaming_gradient` and `CreditTrainer`, plus a frozen
experiment runner and independent verifier. The local eligibility computation
streams a diagonal state Jacobian forward without a reverse-time tape; it omits
cross-unit temporal paths. The output-aligned condition uses current output
weights transposed; the feedback condition uses the original fixed random
matrix. Both use unclipped eligibility/signals, mean answer-token gradients,
global gradient clipping and episode-level Adam. They are not the original
per-token local-SGD learning rule, and these bundled differences are disclosed.

The feedback-vs-output-aligned contrast holds other training settings constant,
but feedback magnitude as well as direction differs. The diagonal-vs-BPTT
contrast changes temporal credit while preserving optimizer/update timing and
recurrence. A readout-only Adam condition controls for learning only the head.
The old BPTT optimizer and new shared optimizer match bitwise in tests; the
diagonal gradient agrees with finite differences when cross-unit Jacobians
vanish and deliberately omits a tested cross-unit path when they do not.

The prospective matrix trained 40 models: two seeds x two tasks x fixed/fresh
noise x five learning modes. All saw the same sixteen cues and 2,048 cue
presentations in shuffled complete epochs; fresh-noise models received a new
independent sixteen-token distractor sequence every presentation. Both 512 and
2,048 checkpoints were fixed in advance. Anchor prompts are seen training
examples only for fixed-noise models, not for fresh-noise models.

### Results, not a selected best checkpoint
The table reports final-checkpoint exact-answer accuracy averaged across both
seeds. Each mean covers 32 anchors, 128 known-cue/new-noise examples, 32 paired
cue interventions, or 128 novel-cue examples. All earlier checkpoints, normal
and reset scores, token accuracy, NLL, and zero-delay held-out scores remain in
the complete 800-cell CSV/JSON evidence.

| Task | Noise training | Learning | Anchor exact | New-noise exact | Changed-cue exact | Novel-cue exact, delay16 |
|---|---|---|---:|---:|---:|---:|
| copy3 | fixed | local_sgd | 15.62% | 4.69% | 0.00% | 0.00% |
| copy3 | fixed | readout_adam | 15.62% | 3.91% | 0.00% | 0.00% |
| copy3 | fixed | feedback_adam | 100.00% | 0.78% | 0.00% | 0.78% |
| copy3 | fixed | diagonal_adam | 100.00% | 3.12% | 0.00% | 0.00% |
| copy3 | fixed | bptt_adam | 100.00% | 10.16% | 0.00% | 0.00% |
| copy3 | fresh | local_sgd | 3.12% | 3.12% | 0.00% | 0.00% |
| copy3 | fresh | readout_adam | 0.00% | 2.34% | 0.00% | 0.00% |
| copy3 | fresh | feedback_adam | 31.25% | 43.75% | 6.25% | 0.00% |
| copy3 | fresh | diagonal_adam | 31.25% | 21.88% | 0.00% | 0.00% |
| copy3 | fresh | bptt_adam | 28.12% | 39.06% | 0.00% | 0.00% |
| recall4 | fixed | local_sgd | 68.75% | 34.38% | 9.38% | 17.19% |
| recall4 | fixed | readout_adam | 81.25% | 35.94% | 3.12% | 21.09% |
| recall4 | fixed | feedback_adam | 100.00% | 33.59% | 0.00% | 16.41% |
| recall4 | fixed | diagonal_adam | 100.00% | 25.00% | 0.00% | 17.97% |
| recall4 | fixed | bptt_adam | 100.00% | 40.62% | 0.00% | 24.22% |
| recall4 | fresh | local_sgd | 40.62% | 42.19% | 3.12% | 20.31% |
| recall4 | fresh | readout_adam | 34.38% | 40.62% | 6.25% | 17.19% |
| recall4 | fresh | feedback_adam | 100.00% | 100.00% | 15.62% | 23.44% |
| recall4 | fresh | diagonal_adam | 65.62% | 65.62% | 12.50% | 17.19% |
| recall4 | fresh | bptt_adam | 75.00% | 75.78% | 9.38% | 19.53% |


### What improved
With fresh-noise training, **feedback_adam reached 100% known-cue recall with
new distractors in both seeds**, versus 42.19% for original local SGD and 40.62%
for readout-only Adam. Resetting state before the query reduced its score to
34.38% on average, a 65.62 percentage-point drop. Both seeds passed the registered
noise-transfer predicate, unlike the original local/readout conditions.

For copying known cues under new noise, the same variant reached 43.75%, versus
3.12% for original local SGD. It did not meet the 75% noise-transfer threshold.
These improvements concern known cues and a bundled learning/training change,
not a broadly generalizing sequence algorithm or an optimizer-only causal claim.

### What failed and what the controls reveal
- **Repeated-noise fitting was misleading.** The fixed-noise feedback, diagonal
  and BPTT variants fit every anchor in both tasks/seeds, yet their copy exact
  scores on new noise were only 0.78%, 3.12%, and 10.16%, respectively. All three
  scored 0% on changed-cue copying. This is evidence of nuisance-sensitive
  memorization, not proof of the precise internal mechanism.
- **Output alignment was not an improvement here.** Under fresh-noise training,
  diagonal_adam achieved 21.88% copy and 65.62% recall on known cues with new
  noise, below feedback_adam's 43.75% and 100%. BPTT achieved 39.06% and 75.78%.
  These are fixed-budget descriptive results from two seeds, not a general
  ranking of learning algorithms or a scale-controlled feedback comparison.
- **Novel/changed cues remain the bottleneck.** Fresh-noise feedback_adam scored
  23.44% on novel-cue recall at delay16 and 15.62% after changing a known cue's
  queried value. Its novel-cue copying exact score was 0% at both tested delays.
  Novel-cue recall with reset was 17.97%, leaving only a 5.47 pp state advantage.
  Global answer balancing does not eliminate finite-sample query-label shortcuts.
- **No final condition passed the full held-out-memory predicate or the
  changed-cue-response predicate.** Every failure is retained. Changed cues may
  be out of the training distribution; weak scores establish limited transfer,
  not universal incapacity. No threshold or seed was changed after results.

Two seeds and correlated checkpoints do not support significance, novelty,
competitive language-model quality or Transformer-efficiency claims. In
particular, a 100% known-cue score is not 100% task generalization or readiness.

### Verified execution and next architecture decision
Two complete fresh-process runs matched exactly. Each processed
81,920 episode presentations,
2,129,920 input-token steps and
163,840 supervised targets. A third distinct process
regenerated the datasets and reproduced every prediction/probability/flag for
35,840 evaluation episodes through
the original CpuSequence engine. Primary peak process working set was
66.73 MiB, with one CPU BLAS thread and no GPU, torch or worker pool.
This is not a resource comparison against a Transformer.

Scientific payload SHA-256: `6829914711c91d8466f7d4fa63df1ac767c772c17cf0a976738084902b763b9c`.
Protocol: `docs/SEQUENCE_CREDIT_V1.md`. Full evidence:
`artifacts/sequence_credit_v1_2026-09-06/`.

**Next:** register a matched-presentation cue-diversity study with independently
regenerated distractors and newly locked novel-cue/intervention evaluations.
Keep the fixed-feedback Adam variant as a development candidate, retain the
output-aligned and exact-gradient controls, and test whether more diverse cues
teach value binding rather than cue identity. Do not enlarge state or declare
the output-aligned rule superior on these results. If broader cue training still
fails, use that evidence to target the recurrence/binding mechanism. Broader text
and matched Transformer/recurrent/state-space comparisons remain required.

The original default learner, recurrence, earlier experiments and official V3
sources/protocols remain unchanged. The new variants are explicit opt-in research
conditions, not promoted defaults. Official readiness remains **21/100**, with
the official V3 evidence chain absent.


## Cue-diversity and locked-counterfactual update — 2026-09-06

**Training on more distinct cues improved genuinely held-out performance, not
just recognition of known cues.** At a fixed 4,096-presentation budget, increasing
the fixed-feedback candidate's training inventory from sixteen to 128 cues raised
novel-copy exact accuracy from **1.56% to 10.94%**, and novel-recall accuracy from
**19.53% to 37.50%**. Both seeds improved on each task. Reliable value binding and
the Transformer-replacement goal remain unachieved; official readiness is 21/100.

### What was implemented
Added `sequence_diversity.make_data`, `training_episode`, `run`, and `verify`.
The implementation creates nested cue inventories, coupled noise/order streams,
whole-stream hashes, leakage checks covering original AND intervened test cues,
and paired counterfactual scoring. It reuses the previous CreditTrainer and
CpuSequence unchanged; this is a data/budget experiment, not a new recurrence.

The frozen matrix trained 32 models: two seeds x two tasks x two cue counts x
four unchanged Adam learning modes. Checkpoints were fixed at 2,048 and 4,096
presentations. Each presentation received fresh sixteen-token noise; its noise
vector matched across cue-count/mode conditions. In each 128-presentation block,
the larger pool used a permutation of all 128 indices and the smaller pool used
those indices modulo16, keeping exposure budgets and label balance controlled.
Unique-cue count and repetition frequency necessarily changed together.

Both original and value-intervened novel cue signatures were excluded from the
entire training pool before metrics, even with distractors removed. The same
64 novel cues and 64 paired interventions were evaluated for every condition,
plus zero-delay novel cues, a common sixteen-cue audit, and all known cues with
new noise. Accuracy was free-running; NLL was separately teacher-forced. No
evaluation example updated weights or selected a checkpoint.

### Complete final-checkpoint results
Means across both seeds at the predeclared final budget. Novel, counterfactual,
and paired means each cover 128 examples/pairs. All-seen covers 32 versus 256
examples and is not a matched-population contrast; the common sixteen-cue scores
are separately retained. Common/all-seen overlap, as do some evaluation cues;
these measurements are not independent replicates. No novel or counterfactual cue overlaps training; the known-cue audits use
new distractors. Both checkpoints, all resets, NLL, token accuracy, and paired raw
scores are retained in the complete CSV/JSON evidence.

| Task | Training cues | Learning | All-seen exact | Novel exact, delay16 | Counterfactual exact | Both-pair exact |
|---|---:|---|---:|---:|---:|---:|
| copy3 | 16 | readout_adam | 3.12% | 0.00% | 0.00% | 0.00% |
| copy3 | 16 | feedback_adam | 90.62% | 1.56% | 0.78% | 0.00% |
| copy3 | 16 | diagonal_adam | 96.88% | 0.78% | 0.00% | 0.00% |
| copy3 | 16 | bptt_adam | 100.00% | 0.78% | 0.78% | 0.00% |
| copy3 | 128 | readout_adam | 1.17% | 0.00% | 0.00% | 0.00% |
| copy3 | 128 | feedback_adam | 23.44% | 10.94% | 8.59% | 3.91% |
| copy3 | 128 | diagonal_adam | 15.62% | 3.12% | 3.91% | 0.00% |
| copy3 | 128 | bptt_adam | 18.75% | 7.81% | 6.25% | 0.00% |
| recall4 | 16 | readout_adam | 40.62% | 15.62% | 13.28% | 0.00% |
| recall4 | 16 | feedback_adam | 100.00% | 19.53% | 27.34% | 8.59% |
| recall4 | 16 | diagonal_adam | 100.00% | 18.75% | 28.12% | 9.38% |
| recall4 | 16 | bptt_adam | 100.00% | 14.06% | 21.88% | 2.34% |
| recall4 | 128 | readout_adam | 21.09% | 17.97% | 13.28% | 0.00% |
| recall4 | 128 | feedback_adam | 60.94% | 37.50% | 35.94% | 10.94% |
| recall4 | 128 | diagonal_adam | 55.47% | 29.69% | 28.91% | 6.25% |
| recall4 | 128 | bptt_adam | 54.69% | 27.34% | 32.81% | 4.69% |


### What the controlled contrasts support
- **Diversity helped novel-cue transfer.** For feedback_adam, the final novel-copy
  gain was +9.38 percentage points and the novel-recall gain was +17.97 pp. At
  128 cues, recall with state reset was 12.50%, versus 37.50% normally. Copy token
  accuracy was 38.28% normally versus 12.50% with reset; copy exact was 10.94%.
  These are direct held-out improvements under matched presentations, not a
  readiness increase or a statistical-significance claim from two seeds.
- **The budget extension also mattered.** Within the 128-cue feedback condition,
  novel-copy exact rose from 0.78% at 2,048 presentations to 10.94% at 4,096;
  novel-recall rose from 27.34% to 37.50%. These checkpoints share a trajectory;
  the comparison is not a fresh independent experiment or a tuned stopping rule.
- **Memorization accuracy decreased as diversity increased.** The feedback model
  fit the sixteen known cues at 90.62% copy/100% recall, but all 128 known cues at
  only 23.44% copy/60.94% recall. On the common sixteen-cue audit its larger-pool
  scores were 40.62%/65.62%. This is an exposure/generalization tradeoff, with
  substantial underfitting still present, not proof that state capacity is the
  limiting factor.
- **The gain is not explained by training only the output head in this budget.**
  The 128-cue readout control scored 0% novel-copy exact and 17.97% novel recall.
  Output-aligned diagonal learning scored 3.12% and 29.69%; exact BPTT scored
  7.81% and 27.34%. Feedback_adam remained the development candidate, but these
  fixed-budget results do not establish a general ranking of algorithms.

### What remains unresolved
The 128-cue feedback candidate answered changed novel cues at 8.59% copy and
35.94% recall. Requiring BOTH a novel cue and its changed-value counterpart to
be correct reduced accuracy to **3.91% copy and 10.94% recall**, versus zero
under reset. That paired test exposes weak consistency despite better marginal
recall. Neither task satisfies the registered reliable-memory or counterfactual
predicates. No final model/seed condition passed either complete predicate.

The candidate still fails most novel answers. Training curves also remain far
from saturation/complete fitting. This evidence does not prove insufficient
capacity, isolate optimization from representation, establish language-model
quality, or demonstrate a Transformer replacement. No thresholds, seeds,
hyperparameters, architecture, or training inputs were changed after metrics.

### Verification and next architecture decision
Two complete fresh-process runs matched exactly. Each processed
131,072 episode presentations,
3,407,872 training input-token steps, and
262,144 supervised targets across 32 models and 64
checkpoints. A third distinct process regenerated datasets and complete stream
hashes and reproduced every score, prediction, target probability, paired score
and predicate in 35,840 evaluation
episodes through the original CpuSequence engine. The largest sampled per-cell
process peak-working-set reading was 70.54 MiB; final serialization
allocations are not a separately measured lifetime peak. Runs used one CPU BLAS
thread, no GPU, torch or worker pool. This is not a Transformer resource comparison.

Scientific payload SHA-256: `b3b6ca62ebe784e50a51a4d8fde0f974c406d73910e535c4a5079cf30d97af2e`.
Protocol: `docs/SEQUENCE_DIVERSITY_V1.md`. Complete evidence:
`artifacts/sequence_diversity_v1_2026-09-06/`.

**Next:** target retention/decoder/value-binding errors while retaining this
diverse-cue stream and the fixed-feedback development baseline. Register a
controlled training-dynamics versus representation comparison before changes:
the budget trend supports checking optimization, while the low paired score
requires stronger binding consistency. Preserve novel and counterfactual
separation, match budgets, and avoid adding untested features merely to improve
training-set scores. Broader text and matched Transformer/recurrent/state-space
baselines are still required before any replacement claim.

The original learner, all previous source/protocol files, and official V3 remain
unchanged. The new study is opt-in, and all reported checkpoints are retained.
Official readiness remains **21/100**, with official V3 evidence absent.


## Frozen checkpoint binding audit — 2026-09-08

The new opt-in `sequence_binding_audit` diagnostic locates accessible value
information without retraining or changing the historical models. Its protocol
was fixed before new probe metrics. All 32 diversity-V1 128-cue checkpoints were
inspected: two seeds, two tasks, four learning modes, and both 2,048/4,096 budgets.

**Copying shows a measurable retention loss.** At 4,096 presentations, the
fixed-feedback candidate's held-out full-value linear-probe accuracy fell from
**82.55% after the cue to 46.09% after distractors**, then reached 47.14% after the
query. Both seeds passed the predeclared encoding-accessible and delay-drop
predicates. Cue-end shuffled/zero controls scored 13.54%/12.50%; after noise they
scored 13.02%/12.50%. The corresponding 2,048-budget cue/noise scores were
78.65%/37.76%, so the existing training trend also remains relevant.

**Recall has a different diagnostic pattern.** The candidate's full-table
cue/noise/query probe scores were 38.67%/40.43%/37.70%. No recall checkpoint met
the 75% accessible-encoding predicate. This points toward an encoding/binding
investigation but does not establish absent information, a capacity limit or a
failure that retention alone would solve.

Across all 32 checkpoints, ten passed encoding_accessible and delay_drop (all
copying); zero passed query_drop or diagnostic_decoder_gap. Query-stage paired
probe-answer exact accuracy was only 1.56% for feedback copying and 11.72% for
feedback recall at the final budget. All modes, seeds, budgets, probe-training
scores and negative controls remain in the complete result records.

These are privileged eight-value, parallel-head diagnostic probes; recall uses
the externally known queried key to choose a head. They are not native 256-byte
model scores, matched decoder comparisons or a new trained capability. Novel
and intervened cue signatures stay outside both model and probe training. Weak
linear probes do not exclude nonlinear codes, and two seeds/correlated
checkpoints do not support a significance claim. The original model scores and
parameters are unchanged.

Two complete fresh-process audits matched exactly. A third distinct process
re-extracted all **8,192 state episodes** using CpuSequence.step, refit the
**288 stage/control probes**, and exactly reproduced all states, coefficients,
predictions, metrics and predicates across **32 checkpoints**. Native predictions
and teacher target probabilities also matched the prior diversity evidence.
Verification status: **PASS**. The primary timed checkpoint work took 15.45 seconds
(excluding initial admission and final report I/O); the largest sampled process
peak working set was 86.07 MiB. This is not a Transformer resource comparison.

Scientific payload SHA-256:
`51e70f65865f5779a394f4e10d89f2668b4acba1ee3a7f55c40eb8ea5e6aaa1b`.
Protocol: `docs/SEQUENCE_BINDING_AUDIT_V1.md`. Full results, controls and exact
execution evidence: `artifacts/sequence_binding_audit_v1_2026-09-08/RESULT_NOTE.md`,
`RESULT_SUMMARY.json`, and `INDEPENDENT_VERIFICATION.json` in that directory.

**Next:** prospectively register one bounded retention change against the
unchanged fixed-feedback copying baseline and an unchanged-recurrence
training-budget control, on fresh development seeds and locked diverse cues,
noise and counterfactuals. Investigate recall encoding/binding separately. No
architecture intervention or new candidate training ran in this audit. Broader
text and matched Transformer/recurrent/state-space comparisons remain required.
Official research readiness remains **21/100**; prior protocols, sources and
terminal artifacts are unchanged.
