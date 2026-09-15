# Sequence trainability V1 — prospective fitting controls, 2026-09-06

## Why this follows the failed memory experiment

Sequence-memory V1 failed its registered held-out memory thresholds. Neither the
unchanged locally trained core nor the same-recurrence exact-gradient reference
demonstrated reliable copy3/recall4 memory after 512 different training examples.
That failure does not identify whether the model fails to fit known examples,
retain information over distractors, generalize, or receive sufficient training.

This is a separate, prospectively specified follow-up. It preserves V1 results,
source snapshots and thresholds. No configuration below changes after metrics.
The goal remains a competitive non-Transformer sequence architecture; fitting
sixteen examples is a diagnostic, not that goal or a language-model result.

## Frozen design matrix

| Variable | Fixed values |
|---|---|
| Model seeds | 17000, 17001 |
| Tasks | Three-value copying; four-key associative recall |
| Architecture | Existing 32-unit gated recurrence, four fixed parents/unit, 256 output bytes |
| Conditions | Original local learning; original readout-only learning; original exact BPTT/Adam reference |
| Training examples | 16 unique examples per seed/task, repeated |
| Training distractors | 0 or 16, separate model instances |
| Checkpoints | After 512 and 2,048 episode presentations (32 and 128 epochs) |
| Held-out examples | 64 unique examples per seed/task, never used for learning |
| Evaluation distractors | Both 0 and 16 at every checkpoint |
| Controls | Reset before the final query; balanced labels; symbolic generator oracle |

The full matrix contains 24 trained models and 48 saved model checkpoints per
run. All models are trained through the full 2,048 presentations irrespective
of intermediate success or failure. There is no early stopping, seed selection,
hyperparameter search, width change, or post-result budget increase in this V1.
The 512/2,048 checkpoints share a training trajectory; they are not independent
replicates. Two seeds are limited development evidence, not external replication.

## Data, balancing, and chronology

Reuse the original task byte encodings: values 64..71; keys 32..35; independent
distractors 128..143; markers COPY=240, RECALL=241, DISTRACTOR=242, QUERY=243.
Noise is absent in the zero-delay condition; the ordinary cue/query markers
remain, so zero distractors does not mean a zero-length recurrent dependency.

Every block of eight examples balances the eight target values at each copy
position. Recall balances queried target values and queries each key twice per
block; all four keys have differing target values across examples. The labels
are therefore not IID samples, and class-frequency shortcuts are controlled,
not assumed absent. Key/value table order is randomized. Copy value triples and
complete recall prompt signatures are unique within and between train/test
splits, using a deterministic bounded rejection procedure before metrics.

Within a seed/task, delay-0 and delay-16 examples share the exact cue values,
key/value table, query, and target. Only independent distractor bytes are inserted
before the query. Held-out examples are disjoint from the sixteen training
prompts even after removing distractors. Each training epoch uses a fixed seeded
shuffle of all sixteen indices; the entire 2,048-item schedule is written before
metrics and is identical across learning conditions and training delays.

Prediction, target revelation, reset, teacher forcing, and free-running scoring
follow sequence-memory V1 unchanged. Supervision is only on answer tokens.
Primary token/exact-answer accuracy is free-running; NLL is explicitly teacher-
forced. Test examples never update weights or choose a checkpoint. The original
CPU engine independently replays all scores from saved checkpoint parameters.

## Learning and acceptance

No original recurrence, initialization, local/readout learning equation, rate,
clipping limit, or exact-gradient optimizer is changed. BPTT uses the existing
0.003 Adam reference and one update per episode. Local/readout update after each
answer token. Data and target budgets are matched, but optimizer steps and
training compute differ; a difference is not attributable solely to locality.
Each model retains its own optimizer state between its two checkpoints. Saved
checkpoints are for inference/evaluation, not optimizer-resume checkpoints.

An individual cell **fits its training examples** when free-running training
exact-answer accuracy is at least 0.95. With sixteen examples this requires all
sixteen answers correct. Report NLL separately rather than silently adding a
confidence gate. A separate **training memory control** passes when normal
minus reset training exact-answer accuracy is at least 0.50. Both facts and all
raw metrics are reported for each seed/task/training-delay/budget cell.

Fitting known examples is not held-out generalization. Held-out accuracy/NLL,
including cross-delay tests, is always published separately. There is no new
official-readiness gate or score award. Balanced eight-value guesses have
expected token accuracy 12.5% and copy-exact accuracy 1/512; these are descriptive
references, not significance tests. No inferential confidence or novelty claim
is made from this small matrix.

## Verification and resources

Freeze all source modules, the three relevant protocols/tests, dependency
metadata, executable/library version, every dataset and training schedule before
metrics. Rehash source/data/freeze inputs around checkpoint/result publication.
Use fresh output directories and immutable canonical JSON plus sidecars. Reopen
all reported artifacts. Record actual process working set/private commit and
per-cell total elapsed time; timings include checkpoint evaluation and are not
pure optimizer-throughput measurements.

Run sequentially on CPU with one BLAS thread, no CUDA context, no torch import,
and no worker pool. Retain the registered minimum 128 MiB available host RAM
before startup and at 32-example training boundaries/evaluation groups. The
separate official V3 8 GiB gate is unchanged. Tiny-set CPU fitting does not
establish a hardware-efficiency advantage over a Transformer.

A second complete fresh-process run must match every scientific row and every
checkpoint hash exactly. A third distinct process regenerates the data/order,
reopens both runs' reports/sidecars/checkpoints, recomputes fitting flags and
aggregate counts, and scores every primary checkpoint through the original
CpuSequence engine. Those complete scores must match exactly. Original memory
V1 and official V3 source/evidence remain unchanged.

## Predeclared interpretation and next decision

- If BPTT fits zero-delay training examples but local learning does not, the
  same architecture has demonstrated narrow trainability; investigate local
  credit/optimizer differences rather than declaring insufficient capacity.
- If a condition fits at zero delay but not at sixteen distractors, investigate
  retention/optimization across distractors, using the paired-example design.
- If training examples are fit but held-out scores remain low, the measured
  gap concerns generalization; do not call memorization a useful sequence model.
- If both training conditions fail even at zero delay, first diagnose gradients,
  optimization budget and representation capacity before scaling text tests.

A subsequent training-budget/state-width or learning-rule study requires a new
prospective specification and preserves every result here, including failures.
