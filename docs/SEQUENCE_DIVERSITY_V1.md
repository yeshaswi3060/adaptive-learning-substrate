# Sequence diversity V1 — prospective matched-budget generalization study

## Question and unchanged components
The previous fixed-feedback eligibility + Adam variant recalled known cues under
new noise, but novel and changed cues remained weak. This separate prospective
study tests training-cue diversity, not another learning-rule or width change.
Earlier results/protocols remain unchanged. The goal is still a competitive
Transformer replacement, not tiny-set memorization or a readiness percentage.

Use the exact previous 32-unit, four-parent, full-256-byte-output recurrence and
the existing CreditTrainer unchanged. Conditions are readout_adam, feedback_adam,
diagonal_adam, and bptt_adam. All use original episode-frozen weights, mean answer
loss, Adam 0.003, betas 0.9/0.999, epsilon 1e-8, gradient norm clip 1 and weight
clip +/-3. Their known feedback/temporal-credit differences are retained, not
rediscovered as new algorithms. No output mask, cue parser in the model, attention,
width change, tuning, early stopping, or default-learner promotion is added.

## Frozen matrix and coupled training input
- Model/data seeds: 19000, 19001. Tasks: copy3 and recall4.
- Training cue counts: 16 and 128. The sixteen cues are the first sixteen of the
  larger pool, not independently sampled. Both pools are balanced in blocks of
  eight over answer values; recall queries each key equally often.
- Budget checkpoints: 2,048 and 4,096 presentations, on the same trajectory.
  All 32 models run through 4,096 presentations regardless of earlier results.
- Every training presentation has sixteen newly sampled distractors. The same
  distractor vector is used at each presentation index across diversity/mode
  conditions. Distractors are distinct across presentation indices and absent
  from every evaluation noise set.
- Each common 128-presentation block is a seeded permutation of 0..127. The
  large condition uses that cue index; the small condition uses index modulo16.
  Thus each large cue appears once and each small cue eight times per block.
  Order is coupled, not identical in cue identity. Small-cue training is not
  claimed to use sixteen-presentation shuffled epochs. All answer positions
  are balanced over each common block in both conditions.
- Entire cue pools, noise arrays, order, composition recipe and SHA-256 of each
  complete canonical episode stream are frozen before model metrics. Models
  receive only the composed ordinary byte stream, never pool indices or labels
  before their causal prediction positions.

Counts: 32 trained models, 64 saved checkpoints per run. Presentation, input-token
and answer-target budgets match across diversity conditions; number of unique
cues and exposure frequency necessarily differ. Wall time need not match.

## Novel and counterfactual cues stay outside training
First generate 64 balanced novel cue examples for each seed/task. Also generate
their value interventions: increment all three copy values modulo eight, or only
the queried recall value. Exclude the signatures of BOTH the original and the
intervened examples from the entire 128-cue training pool, even after removing
distractors. This closes the possibility that an intervention was a training cue.
Exclusion and deterministic bounded rejection occur before any model metrics.

Five sets are evaluated at both checkpoints, each normal and state-reset:
1. common_seen: the same sixteen known cues with new noise, identical for both
   diversity conditions. These full prompts were never used for training.
2. all_seen: all sixteen or 128 known cues with new noise. Its populations differ
   across diversity conditions; do not mistake this for the common-cue contrast.
3. heldout0: 64 novel cues without distractors; cue/query markers remain.
4. heldout16: those same novel cues with independent sixteen-token distractors.
5. counterfactual16: the novel-cue interventions, paired with exactly the same
   distractors and query as heldout16, and correctly recomputed new targets.

common_seen is a subset of all_seen, not independent evidence. Interventions can
overlap another evaluation cue and are paired/correlated, but none may overlap
training. Both diversity conditions see identical novel/intervention evaluation
sets. Scoring never updates parameters or chooses a checkpoint. Accuracy is
free-running, NLL separately teacher-forced, and reset occurs before the final
query. Paired exact accuracy requires BOTH original and changed answers correct.

## Predicates and interpretation, fixed before results
- Common/all-seen fit: exact >=95%, reported separately (different population
  sizes imply different required correct counts).
- Held-out memory at delay16: token >=75%, normal-minus-reset token >=15 pp,
  and copy exact >=50%.
- Counterfactual response: changed exact >=75%, paired exact >=50%, and paired
  normal-minus-reset exact >=15 pp. Report all component measurements.
- All conditions/seeds/budgets remain in the report. No best-seed or post-result
  threshold changes; no official readiness score award.

The main comparison is 128 versus sixteen cues at a fixed budget, within each
learning mode and task. Also report the registered budget extension from 2,048
to 4,096, not as independent replication. More diverse cues helping held-out and
paired scores would support learned transfer, but not an unrestricted algorithm
or Transformer-replacement claim. Poor scores after broader training justify a
controlled representation/retention investigation; they do not prove insufficient
capacity or settle optimization. Two seeds provide limited development evidence,
not significance or novelty. Balanced-label and query-frequency controls do not
eliminate all finite-sample shortcuts. Keep original/readout/gradient evidence.

## Execution and verification
Run sequentially on CPU with one BLAS thread, no GPU, torch or worker pool, and
the unchanged 128 MiB available-host-RAM guard before startup, every 32 training
presentations and between evaluation sets. Official V3's separate 8 GiB gate is
unchanged. Record per-model elapsed time including evaluation/I/O, and observed
working set/private commit, not a Transformer efficiency comparison.

Freeze all current source modules, relevant protocols/tests, original learning
specification, executable hash/library version, and all data/order files. Recheck
freeze/data/source hashes at publication boundaries. Reopen immutable records,
sidecars and inference checkpoints. A second fresh training process must match
all scientific payload/checkpoint hashes exactly. A third distinct process must
regenerate data/stream digests and reproduce every primary checkpoint score,
paired score and predicate through the original CpuSequence engine. Preserve
failures. Saved checkpoints are inference artifacts, not optimizer-resume files.
