# Experiment 000 Alignment Diagnostic

**Status:** preregistered development diagnostic; write the result in the
development log, not in this protocol  
**Mechanism:** CCF-v0, implementation revision C000.1  
**Purpose:** determine whether the current failure comes from weak forward
memory, a misdirected CCF update, or both  
**Seed partition:** custom diagnostic seeds 42–46 only; confirmatory seeds are
forbidden

This diagnostic must run before changing the graph initialization, memory
dynamics, credit equation, or CCF defaults.

## 1. Question

For the exact timestamped event ancestry realized by one delayed-cue episode,
how closely does the total CCF weight change point toward the local
fixed-event-DAG descent direction?

The diagnostic does not use automatic differentiation and does not claim that
the exact direction is a training baseline. It is a measuring instrument.

> **Simple explanation:** We freeze which events happened, calculate the
> direction that would most directly improve that one prediction, and compare
> it with the direction CCF actually chose.

## 2. Data and execution

- Master seeds: `42, 43, 44, 45, 46`.
- These seeds have the `custom` role and are not development-gate or
  confirmatory seeds.
- Full diagnostic: 20 exactly balanced training episodes per seed.
- Smoke diagnostic: 2 episodes per seed.
- Each episode contains one bipolar cue, eight bipolar noise events, `QUERY`,
  and terminal feedback only after the forced output at graph tick 11.
- A seed uses one graph sequentially, so the diagnostic observes the update
  direction along the first 20 steps of the deterministic learning trajectory.
- Evaluation streams are generated and hashed for split integrity but are not
  used to fit, tune, or update the diagnostic.

## 3. Frozen graph and CCF settings

Use the Experiment-000 defaults: 64 hidden units, recurrent in-degree 8, input
fan-out 8, output fan-in 16, one-tick edges, initial scale 0.35, `tanh`, float64,
learning rate 0.01, trace decay 0.97, route gain 0.90, trace horizon 32, hop
limit 16, maximum update 0.05, and weight bound 3.0.

The structural topology must remain unchanged. The learner may modify only
plastic scalar weights through its ordinary `apply_supervised_credit` call.

## 4. Exact derivative on the realized event DAG

Let event (v) have activation (a_v), and let its incoming realized traces be
(r\in P_v). Trace (r) names shared structural weight (w_{e_r}) and exact
parent event (p_r). Source-event derivatives are zero.

For every structural parameter coordinate (k), process events in increasing
timestamp and deterministic event-ID order:

\[
\frac{\partial u_v}{\partial w_k}
=
\sum_{r\in P_v}
\left[
\mathbf{1}(e_r=k)a_{p_r}
+w_{e_r}\frac{\partial a_{p_r}}{\partial w_k}
\right],
\]

\[
\boxed{
\frac{\partial a_v}{\partial w_k}
=
(1-a_v^2)
\frac{\partial u_v}{\partial w_k}
}.
\]

This summation is essential because one recurrent parameter may occur many
times at different timestamps.

For bipolar target (y\in\{-1,+1\}), queried activation (a_o), and the same
bounded root adapter used by CCF,

\[
c=\operatorname{clip}(y-a_o,-1,1),
\qquad
\boxed{d^*=c\nabla_w a_o}.
\]

(d^*) is the exact descent direction under the realized gates and the frozen
root adapter. The observed CCF vector is

\[
\Delta w_{CCF}=w_{after}-w_{before}.
\]

## 5. Comparisons

Report globally and separately for `cue`, `noise`, `query`, `recurrent`, and
`output` parameter groups:

\[
\operatorname{cosine}=
\frac{\Delta w_{CCF}\cdot d^*}
{\|\Delta w_{CCF}\|_2\|d^*\|_2},
\]

- dot product;
- cosine similarity;
- sign agreement on coordinates nonzero in both vectors;
- CCF-to-exact norm ratio;
- each group's fraction of total squared update norm.

Construct an equal-norm measuring step, without training the live graph:

\[
\Delta w_{exact}=d^*
\frac{\|\Delta w_{CCF}\|_2}{\max(\|d^*\|_2,10^{-30})}.
\]

Replay the same episode from the same pre-update weights with the CCF weights
and with (w+\Delta w_{exact}). Report change in

\[
L=\tfrac12(y-a_o)^2
\]

and whether each replay preserved the original emission-event mask. Replay
loss is descriptive when a mask changes.

## 6. Finite-difference validation

For 32 deterministic, group-stratified coordinates per episode, use
(h=10^{-6}):

\[
g_k^{FD}=\frac{a_o(w+h\mathbf e_k)-a_o(w-h\mathbf e_k)}{2h}.
\]

Accept a coordinate only when both perturbed replays have the exact original
emission mask. For accepted coordinates require

\[
|g_k^{FD}-g_k|
\le 10^{-7}+10^{-4}\max(|g_k^{FD}|,|g_k|).
\]

The implementation is valid only if:

1. at least 80% of requested checks preserve the mask;
2. at least 99% of accepted checks meet the error tolerance;
3. every event trace names an earlier parent event;
4. the analytic root preactivation and activation reconstruct exactly to
   float64 tolerance;
5. topology is unchanged and all values are finite;
6. identical reruns produce the same deterministic payload hash.

## 7. Interpretation rules

These labels diagnose; none is an Experiment-000 pass.

- **FORWARD-MEMORY-LIMITED:** the earlier paired-cue probe remains weak and the
  exact direction itself assigns negligible useful sensitivity to cue paths.
- **CREDIT-ALLOCATION-LIMITED:** exact equal-norm steps usually reduce loss,
  while median CCF cosine is below 0.5 or CCF systematically allocates update
  norm to noise instead of the parameter groups favored by (d^*).
- **BOTH:** forward cue sensitivity is weak and CCF alignment is also poor.
- **DIRECTION-PLAUSIBLE:** median cosine is at least 0.5, positive-dot episodes
  are at least 80%, and CCF replay loss improves almost as consistently as the
  equal-norm exact step. This still does not demonstrate learning.

No threshold may be changed after viewing the full 20-episode result. A change
to the learning equation or its information-access contract requires a new CCF
version and a new protocol. Architecture-only changes receive a new experiment
version and must be separated from learning-rule changes.

## 8. Required output

Save the exact configuration, code/specification hashes, stream manifests,
graph/topology/initial-weight hashes, raw episode-level metric rows, aggregate
tables, finite-difference rows, deterministic payload hash, runtime/backend,
and an explicit count of confirmatory seeds used. The latter must be zero.

