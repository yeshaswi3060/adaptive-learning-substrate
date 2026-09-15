# Experiment 000 TRACE-HEAD-L1 Exact-Alignment Protocol

**Protocol name:** `experiment_000_trace_head_alignment`  
**Protocol version:** `trace-head-l1-v1`  
**Scope:** exact local-gradient validation of a standalone trace readout head  
**Official alignment seeds:** custom seeds 75--79 only  
**Episodes:** 32 exactly cue-balanced episodes per seed  
**Scientific claim:** head-alignment evidence only; this is not recurrent temporal credit

## 1. Purpose and prerequisite

TRACE-HEAD-L1 is the first prerequisite after a valid A3 representation
selection. It asks whether the selected passive-trace representation supports an
exact, independently checked local update for a new 16-dimensional output head.
It does not update the recurrent graph and cannot validate recurrent temporal
credit.

Execution must stop before an official seed is evaluated unless all seven A3
phases exist in order and still validate:

`tests -> freeze -> persisted smoke and sidecar -> machine smoke verification -> full -> rerun -> determinism verification`

The sole upstream prerequisite is `readout-trace-v1a3` on registered custom
seeds `105, 106, 107, 108, 109`, accepted only through
`load_verified_a3_selection()`. The A3 primary report, rerun report, both
SHA-256 sidecars, phase sequence, freeze record, and
`DETERMINISM_VERIFICATION.json` must agree on a terminal
`SELECTED:<condition>` decision. `NO_SELECTION`, provisional, invalid, missing,
or inconsistent A3 evidence is a hard stop. `readout-trace-v1a2` is explicitly
resource-invalid and cannot satisfy this prerequisite because its experiment
partition did not complete the compact-resource recovery. The selected A3
condition determines the immutable trace retention used here.

## 2. Frozen data and representation

Official seeds are exactly `75, 76, 77, 78, 79`, in that order. Each seed uses
the deterministic Experiment-000 `train` namespace with exactly 32 episodes,
16 per cue. Seeds 0--74 and confirmatory seeds 1000--1019 are excluded. Tests
use only non-official scratch seeds.

For each seed, construct the frozen Experiment-000 recurrent graph and clone it
with the A3-selected passive trace retention. The graph topology, all graph
weights, and specifically all recurrent weights remain bitwise unchanged. After
each episode, read the 16 passive-trace coordinates corresponding to the frozen
output edges in lexical edge-ID order. The trace remains read-only at the head.

## 3. Head, objective, and analytic direction

The standalone head owns only

`theta in R^16`, initially the exact positive-zero vector.

For feature vector `z` and bipolar target `y in {-1,+1}`:

`p = theta^T z`

`q = tanh(p)`

`L = 0.5 * (y - q)^2`

The analytic **descent direction** is

`g = (y - q) * (1 - q^2) * z`.

This sign convention is the negative derivative of `L` with respect to
`theta`.

## 4. Independent numerical alignment

Before each update, compute a central finite-difference loss derivative for all
16 coordinates using `epsilon = 1e-6`:

`dL_j = (L(theta + epsilon*e_j) - L(theta - epsilon*e_j)) / (2*epsilon)`

and numerical descent direction `g_num,j = -dL_j`.

Every coordinate in every official case must satisfy either absolute error at
most `1e-8` or relative error at most `1e-6`. Every case must also satisfy:

- direction cosine between analytic and numerical descent directions at least
  `0.999999`;
- analytic/numerical dot product strictly greater than zero;
- all values finite.

There is no allowance for dropping zero, difficult, or failed cases.

## 5. Frozen update

Let `d = max(1, ||z||^2)`. Apply exactly:

`delta = clip(0.5 * g / d, -0.05, +0.05)`

`theta <- clip(theta + delta, -3, +3)`.

The persisted record must allow the update to be reconstructed exactly from the
stored pre-update head, feature, target, and analytic direction. The recurrent
graph must not contain or receive `theta` or `delta`.

## 6. Gates and interpretation

TRACE-HEAD-L1 passes only if 100% of the 160 official episode cases and 100% of
their 2,560 coordinate checks pass, all direction cosines and positive-dot
checks pass, every normalized/clipped update reconstructs exactly, all A3
bindings remain unchanged, and all topology/weight invariants hold.

A pass means only that the selected passive features and a standalone head have
an exactly aligned local gradient. It does not show recurrent learning, temporal
credit assignment, continual learning, or performance against learning
baselines. The next separately preregistered gate must integrate and test an
actual temporal-credit mechanism against compute-matched baselines.

## 7. Persistence and independent rerun

The two official executions are separate processes and persist to exactly:

- `artifacts/experiment_000/trace_head_alignment/alignment_seeds_75_79_episodes_32.json`
- `artifacts/experiment_000/trace_head_alignment/alignment_seeds_75_79_episodes_32.sha256`
- `artifacts/experiment_000/trace_head_alignment/alignment_seeds_75_79_episodes_32_rerun.json`
- `artifacts/experiment_000/trace_head_alignment/alignment_seeds_75_79_episodes_32_rerun.sha256`

The rerun is blocked unless the primary report and sidecar validate and the
process identity differs. A final verifier requires canonical deterministic
payload equality, equal deterministic payload SHA-256 values, fresh process
identities, valid sidecars, unchanged source manifests, identical A3 bindings,
and pass status in both reports. It persists:

- `artifacts/experiment_000/trace_head_alignment/DETERMINISM_VERIFICATION.json`

Only this final verification may issue terminal status
`TRACE_HEAD_ALIGNMENT_PASS`. Any failed scientific gate returns
`TRACE_HEAD_ALIGNMENT_FAIL`; any malformed prerequisite or persistence chain is
`INVALID_PROCEDURE_NONSELECTING`.

The command-line interface prints only status, paths, and hashes. It never
prints per-seed, per-episode, feature, loss, or alignment values.
