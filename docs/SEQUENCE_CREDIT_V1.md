# Sequence credit V1 — prospective learning-rule and cue controls, 2026-09-06

## Objective and hypotheses
The previous unchanged core fit 7/8 tiny conditions using BPTT but 0/8 using
local learning; held-out memory remained poor. This separate development study
implements an output-aligned local eligibility rule and tests whether feedback,
cross-unit temporal credit, or fixed distractor signatures explain the gap.
No previous result, protocol, model source or threshold is changed.

## Frozen matrix
- Seeds: 18000 and 18001; tasks: copy3 and recall4.
- Existing 32-unit, four-parent, 256-byte gated recurrence; same initialization
  within each seed across modes/regimes. No attention or architecture changes.
- Sixteen unique balanced cues per seed/task; 2,048 presentations in complete
  shuffled epochs; checkpoints at 512 and 2,048, with no early stopping.
- Training distractors: 16. Regimes: fixed noise per cue, or new independent
  noise on every presentation. Cue/order/target budgets are identical.
- Five modes: original local_sgd; readout_adam; feedback_adam; diagonal_adam;
  bptt_adam. Forty models and eighty checkpoints per complete run.
- All data, schedules and evaluation interventions are generated and frozen
  before any scientific metrics. There is no post-result tuning or seed selection.

## Actual learning-rule change and controlled contrasts
For each hidden unit, eligibility streams forward with its diagonal state
Jacobian g + (previous-candidate)*g*(1-g)*a. Direct parameter derivatives are
added for input, gate, recurrent edges and biases. Cross-unit Jacobian paths are
omitted. Eligibility/gradient state has fixed size with sequence length; the
diagonal rule builds no reverse-time tape. Parameters are frozen within an
episode, and supervised gradients are averaged over answer positions.

diagonal_adam uses the current output-weight transpose times the cross-entropy
error as its local learning signal. feedback_adam instead uses the original
fixed random feedback matrix; everything else in these two modes is identical.
The matrices differ in magnitude as well as direction (initial output weights
are 0.01 times feedback transpose), so this is not a scale-normalized geometry
ablation. Both omit the original eligibility/signal clips, while keeping a global
gradient norm clip. They are approximations, not exact BPTT for connected units.

bptt_adam retains all cross-unit temporal derivatives. Relative to diagonal_adam,
it changes the gradient calculation, not recurrence, optimizer, loss reduction,
initialization or update timing. The diagonal gradient is checked against exact
BPTT and finite differences when cross-unit Jacobians vanish, and against an
explicit missing cross-unit-path example when they do not.

All four Adam modes use rate 0.003, betas 0.9/0.999, epsilon 1e-8, global norm clip
1, weight clip +/-3, and one update per episode. readout_adam freezes all recurrent
parameters and updates only the output head. Shared BPTT Adam updates must match
the previous optimizer bitwise. local_sgd is the original clipped eligibility,
fixed-feedback and per-answer update rule with its original rates. Comparing it
to an Adam mode changes several training choices; do not attribute that contrast
solely to feedback or optimizer. Equal token budgets do not mean equal compute.
Output-weight transport and episode-batched updates are explicit tradeoffs, not
a claim of fully autonomous biologically local online learning.

## Paired data and counterfactual evaluation
Use the previous byte encodings and balanced cue generator with new seeds and
split names. Training cue signatures and 64 held-out cue signatures are disjoint
even when distractors are removed. Balance each answer position in blocks of
eight and recall key frequency; this does not remove finite-sample key/label
correlations. Every epoch contains the same sixteen cue indices, shuffled.

Noise vectors are unique across anchors, every fresh training presentation,
fresh-noise evaluation, and held-out evaluation. The intentional exception is
the changed-cue intervention, which reuses its paired anchor's noise. Five
evaluation sets are scored at both checkpoints, each normal and state-reset:
1. anchor_cues: 16 canonical prompts, seen during fixed-noise training but not
   during fresh-noise training. Do not call this training accuracy for fresh mode.
2. fresh_noise: 64 prompts, four new noise vectors per known training cue.
3. changed_cue: 16 paired anchors with identical noise/query and changed answer
   cues. Copy increments all three values modulo eight; recall increments only
   the queried key's value. Targets are recomputed, never reused incorrectly.
4. heldout0: 64 novel cues with zero distractors (markers still remain).
5. heldout16: the same 64 novel cues with new sixteen-token noise.

All inference uses the full output vocabulary and free-running answers; NLL is
separately teacher-forced. Evaluation never updates weights or selects a budget.
Reset immediately before the final query removes prior state. Changed-cue
accuracy tests response to a cue intervention, not universal causal sufficiency;
these transformations themselves may be out of the training distribution.

## Frozen descriptive predicates and interpretation
Per seed/task/regime/mode/checkpoint:
- Anchor fit: exact >=95% (16/16 required).
- Noise transfer: fresh_noise exact >=75% and normal-minus-reset exact >=50 pp.
- Changed-cue response: changed_cue exact >=75%.
- Held-out memory: heldout16 token >=75%, normal-minus-reset token >=15 pp,
  and copy exact >=50%. The full condition must pass, not a best seed.
These are development diagnostics with zero official readiness effect. Publish
all cells and both checkpoints. Two seeds are limited evidence; checkpoints on
one trajectory are correlated. No significance, novelty or replacement claim.

Fixed-noise fitting with poor fresh-noise transfer suggests nuisance dependence.
Diagonal-vs-feedback differences identify feedback-matrix choice under matched
training settings, not scale-free alignment. BPTT-vs-diagonal differences test
cross-unit temporal credit under the shared optimizer. Readout comparisons check
whether a gain needs learned recurrence. Fresh-noise improvement with poor novel
cue scores still leaves a generalization problem. Negative outcomes remain in
the report; later changes require a new prospective study.

## Reproducibility and resources
Run sequentially on CPU, one BLAS thread, no torch, GPU or worker pool. Require
at least 128 MiB available host RAM before startup, at 32-presentation boundaries,
and between evaluation sets. Original official V3's separate 8 GiB gate remains
unchanged. Record process peak working set/private commit and elapsed cell time
including evaluation/checkpoint I/O, not pure optimizer throughput.

Freeze sources, protocols/tests, model configuration, executable hash/library
version and every data/order file before metrics. Verify freeze/data/source
hashes at checkpoint/publication boundaries. Retain every raw prediction,
teacher probability, model checkpoint, loss trajectory and sidecar. A second
complete fresh process must reproduce the scientific payload/checkpoint hashes
exactly; a third process regenerates datasets and scores every primary checkpoint
through the original CpuSequence engine. It must reproduce all scores and flags.
Saved checkpoints support inference, not optimizer-resume. A future competitive
Transformer comparison still needs matched data, tuning, quality and resources.
