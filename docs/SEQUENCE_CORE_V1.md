# Sequence core V1 — prospective implementation and development experiment

This implements the token-engine step of SEQUENCE_ARCHITECTURE_TARGET.md.
It is not a proven Transformer replacement. No attention, task-ID routing,
first-cue latch, autograd or end-to-end backward traversal is used. Vocabulary
is all 256 UTF-8 byte values. State has 32 units, four recurrent parents each
(no self edges), fixed topology. State persists until an explicit document reset.

## Model and local approximation (fixed before results)

    c_i = tanh(W_i[token] + sum_l R_il * s_old[parent_il] + bc_i)
    g_i = sigmoid(G_i[token] + a_i * s_old[i] + bg_i)
    s_i = g_i*s_old[i] + (1-g_i)*c_i
    p = softmax(O @ s + bo)

Token lookups are trainable columns, not a hard-coded cue decoder. Initial W has
normal std .2, G .05, R .04, a=bc=bo=0, bg=linspace(0,3). Fixed random feedback
B has std 1/sqrt(32); O starts at .01*B.T. PCG64 derives all values from the seed.

    D_c = (1-g)*(1-c*c)
    D_g = (s_old-c)*g*(1-g)
    J_local = g + D_g*a
    E_parameter <- clip(J_local*E_parameter + direct_derivative, -4, 4)

Direct derivatives are D_c*one_hot(token) for W, D_g*one_hot(token) for G,
D_c*s_old[parent] for R, D_g*s_old for a, D_c for bc, D_g for bg.
This is an explicitly **approximate** unit-local eligibility rule: cross-unit
temporal derivatives, higher-order derivatives of past weight changes, and
history removed by clipping are not tracked. It is not exact BPTT or a novelty
claim. Gates and recurrent/input weights are plastic in the candidate.

After prediction, target byte y is revealed exactly once:

    error = one_hot(y) - p
    local_signal = clip(B @ error, -1, 1)
    delta_core = clip(.001 * local_signal * E_parameter, -.02, .02)
    delta_O = clip(.05 * outer(error,s) / max(1,sum(s*s)), -.02, .02)
    delta_bo = clip(.05 * error, -.02, .02)

All changed weights are clipped to [-3,3]. In the readout-only baseline, all core
updates are disabled; O and bo learn identically. Both variants have the same
state/parameter allocation. Fixed B is not a learned parameter. Evaluation may
advance recurrent state/eligibilities, but never changes weights. No sequence
history, attention scores or growing cache is retained.

## Engineering acceptance

Independent NumPy and direct CUDA implementations must match probabilities,
state, eligibility buffers and parameter updates to atol=rtol=1e-10 on all 256
token values and 64 sequential updates. Both training modes are covered.
Discrete argmax must match whenever the reference top-two margin exceeds 1e-10;
near ties are reported, not treated as evidence of accuracy.
Test normalized finite probabilities; bounded state/traces/weights; prediction-
before-target; duplicate/invalid target rejection; invalid tokens before state
mutation; no evaluation weight changes; reset and explicit pending-target cancel;
chunked/unbroken equivalence; prefix causality; order sensitivity; cross-backend
checkpoint continuation including a pending target; corrupted/incompatible
checkpoints rejected without changing a live model; use-after-close rejection.

Device buffers must stay constant across 4096 repeated tokens and explicit
allocations below 16 MiB. Record physical process working set/private commit;
driver/compiler/context memory is excluded from the explicit device-buffer cap.
Require >=512 MiB available host RAM before CUDA startup, >=128 MiB at regular
boundaries, >=256 MiB initial GPU headroom. No torch import or worker pool.

## Tiny text development experiment — not a competitive LM benchmark

Use seeds 15000,15001,15002; train one pass on the first 2048 UTF-8 bytes each of
docs/CCF_V0.md and docs/EXPERIMENT_001.md, in that order. Reset sequence state,
not weights, at document boundaries. Predict t+1 from byte t before revealing it.
Evaluate on the first 1024 bytes of docs/SEQUENCE_ARCHITECTURE_TARGET.md, never
used for updates. Freeze exact byte files and their hashes before any metrics.
Report train/test byte n-gram overlap rather than asserting deduplication.

Compare full local learning, readout-only recurrence, a Laplace-smoothed bigram
model (alpha=1, same training pairs), and uniform bytes. Report held-out mean
negative log likelihood in nats/byte, byte perplexity, byte accuracy, training/
evaluation time, parameter changes, and memory. Tiny project-document text is
an engineering dataset; it is not representative general-language evaluation.

Predeclared descriptive learning gate: candidate held-out NLL below its own
pretraining NLL and uniform NLL for all three seeds. Report independently whether
it beats the bigram and readout-only controls; do not raise readiness on this gate.
No Transformer, gradient-trained recurrent or state-space comparison is claimed
until those baselines are implemented and actually executed. No seed selection,
threshold changes, corpus enlargement or tuning after valid results.

Two complete fresh-process runs must match the scientific payload. Freeze source,
protocol, environment/compiler identities and data before measuring; check drift
at seed boundaries and publication. Save exact weights, inputs, metrics, and
project-compatible hash sidecars. Independent verification recomputes held-out
predictions on CPU from saved weights. Canonical readiness remains 21/100.
