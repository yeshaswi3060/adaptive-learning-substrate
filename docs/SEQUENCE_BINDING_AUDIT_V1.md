# Sequence binding audit V1 — prospective checkpoint diagnostic

## Question and decision boundary
The completed diversity study improved held-out performance, but paired value
binding remained weak. Before another training/architecture intervention, locate
linearly accessible value information in its existing checkpoints. This protocol
is fixed before any new probe metric. Historical native scores are already known;
this is not a fresh confirmatory experiment or new training result.

Hypotheses: value information is weak immediately after encoding, is degraded by
distractors/query processing, or remains accessible to a different supervised
decoder. A failed linear probe cannot establish absence of information, and a
successful probe cannot identify the causal source of the native learner's errors.

## Locked inputs and matrix
- Reuse the verified diversity-V1 primary payload
  `b3b6ca62ebe784e50a51a4d8fde0f974c406d73910e535c4a5079cf30d97af2e`.
- Inspect every 128-cue checkpoint: seeds 19000/19001, copy3/recall4,
  readout_adam/feedback_adam/diagonal_adam/bptt_adam, and 2048/4096 presentations.
  All 32 checkpoints are reported; none is selected by probe performance.
- Keep all model parameters, original sources, datasets and learning budgets
  unchanged. No optimizer is resumed and no candidate-model update is performed.
- Reopen upstream freeze/report, source manifest, data, per-cell records,
  inference checkpoints and independent-verification receipt with SHA256
  sidecars. Check original frozen source entries (new audit files are additions,
  not amendments to the historical source freeze). Regenerate every dataset.

## Measurements and supervised probes
For each prompt, extract the 32-coordinate recurrent state at three fixed token
boundaries: after the cue terminator (cue_end), after all sixteen distractors but
before the query suffix (delay_end), and after the complete query (query_end).
Copy prefixes contain five bytes; recall prefixes contain ten. Query suffixes
contain one and two bytes, respectively. No answer token enters state extraction.

Probe training uses all 128 previously trained cue identities with the original
all_seen fresh distractors. These are training data for the NEW probe, not a
held-out probe evaluation. Fit separate stages from scratch with identical
128-example budgets. Evaluate the original 64 heldout16 cues and their 64 paired
counterfactual16 cues, excluded from the entire model/probe training pool even
after removing noise. Report probe-training scores explicitly as in-sample.

Labels encode all three copy values, or all four recall values in canonical key
order 32..35. The audit parser supplies these labels to probe fitting only; state
extraction accepts prompts, not labels. For recall, the diagnostic answer uses
the externally known queried key to select a head. Copy uses three simultaneous
heads instead of native autoregressive decoding. Each head predicts among eight
value bytes (64..71), unlike the model's 256-byte output. Therefore probe/native
gaps are descriptive upper diagnostic opportunities, NOT comparable model
performance, a deployable decoder improvement, or a causal optimization test.

For n training states X and concatenated one-hot labels Y, use float64 and:

    Xc = X - mean_train(X); Yc = Y - mean_train(Y)
    W = solve(Xc.T @ Xc / n + 0.001 * I, Xc.T @ Yc / n)
    b = mean_train(Y) - mean_train(X) @ W
    prediction_j = 64 + argmax((X @ W + b).reshape(n, heads, 8), axis=2)

The intercept is unpenalized; features are not normalized. Ties use NumPy's first
index. There is no hyperparameter search, held-out standardization or test-label
fitting. Scores are linear discriminants, not probabilities. Persist probe W/b,
state arrays, full-value labels, full predictions and task-answer predictions.

Controls at each stage use exactly the same budgets: (1) labels permuted across
training cue rows with one SHA256/PCG64-seeded permutation per seed/task, shared
across stages/modes/checkpoints; (2) zero states for BOTH fitting and evaluation,
so only training label frequencies remain. Permutation preserves row associations
between heads and marginal label counts; one realization is descriptive, not a
significance test. Zero control does not simulate the model's native query reset.
Separately replay and retain the historical native normal/query-reset scores for
heldout16 and counterfactual16 without any parameter updates.

## Predeclared diagnostic predicates, not advancement gates
Report token/full-vector exact accuracy, task-answer exact accuracy, and paired
task-answer exact (both original and changed correct). For the aligned probe:
- encoding_accessible: cue_end heldout full-value token accuracy >=75% AND
  >=15 percentage points above BOTH shuffled and zero controls;
- delay_drop: encoding_accessible AND cue_end-minus-delay_end heldout full-value
  token accuracy >=15 percentage points;
- query_drop: delay_end heldout full-value token accuracy >=75% AND
  delay_end-minus-query_end accuracy >=15 percentage points;
- diagnostic_decoder_gap: query_end heldout task-answer token accuracy >=75%,
  >=15 points above the native task-answer token score AND both probe controls,
  and query_end paired task-answer exact >=50%.

Predicates can coexist. Report every component, seed and checkpoint; differences
between checkpoints share a training trajectory. Four recall values and answer
selection are privileged diagnostic structure. No predicate is a readiness gate.
Failure of these predicates leaves encoding/retention/optimization unresolved.

## Execution, verification and next intervention
Run sequential CPU float64 with one BLAS thread, no GPU/torch/worker pool, and
the existing 128 MiB available-RAM guard at startup and each checkpoint. Freeze
new source/test/protocol hashes and all consumed evidence hashes before extracting
states. Recheck them and the written freeze at every publication boundary; write
new files exclusively with project SHA256 sidecars and reopen every record.

A second fresh process repeats the complete audit exactly. A third distinct
process regenerates datasets, re-extracts every state using CpuSequence.step
instead of the primary forward_one path, refits probes, and reproduces every
saved state, coefficient, prediction, metric and predicate exactly. Verify native
predictions against the historical checkpoint evidence as well. Record measured
runtime and process memory separately from the scientific payload.

If information survives but native decoding fails, prospectively compare a
decoder/training intervention at matched presentations. If information drops
across distractors, test one bounded retention change against the unchanged
fixed-feedback baseline. If cue-end probes are weak, prioritize encoding/binding
and keep optimization controls rather than asserting insufficient capacity.
Any intervention needs a new prospective protocol and locked held-out data.
This audit itself makes no architecture, novelty, Transformer-efficiency, or
official-readiness advance; the evidence-based readiness remains 21/100.
