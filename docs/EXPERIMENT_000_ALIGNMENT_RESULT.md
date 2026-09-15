# Experiment 000 fixed-event update-alignment result

**Status:** completed development diagnostic; not a mechanism pass  
**Date:** 2026-09-01  
**Mechanism:** CCF-v0, implementation C000.1  
**Seeds:** custom seeds 42–46; confirmatory seeds used: 0  
**Episodes:** 20 sequential balanced training episodes per seed, 100 total

## Question answered

For each episode, we froze the exact sparse event graph that had occurred and
computed the local direction that most directly reduced that episode's output
error. We compared this direction with the weight change actually made by CCF.

For output activation \(a_o\), bipolar target \(y\), and recorded event-DAG
output derivative \(\nabla_w a_o\):

\[
c=\operatorname{clip}(y-a_o,-1,1),
\qquad
d^*=c\nabla_w a_o.
\]

The actual CCF change was

\[
\Delta w_{CCF}=w_{after}-w_{before}.
\]

Their directional agreement was measured by

\[
\cos(\Delta w_{CCF},d^*)=
\frac{\Delta w_{CCF}\cdot d^*}
{\|\Delta w_{CCF}\|_2\|d^*\|_2}.
\]

We also gave the exact direction the same update length as CCF, so the loss
comparison tested direction rather than update size.

## Implementation validation

- 3,200 of 3,200 fixed-DAG finite differences passed tolerance.
- 3,200 of 3,200 full-simulator finite differences preserved the original
  emission mask and passed tolerance.
- Maximum derivative error: `7.0260255946585914e-12`.
- All topologies stayed fixed, all event DAGs stayed immutable, and every value
  remained finite.
- Both full runs produced deterministic payload SHA-256
  `17573dd402b252c82c3c22286f6cb0898bb9cd50fd8d37b6cdf57ea38fa7ec58`.
- Confirmatory seed count was exactly zero.

## Full result

| Measure | Result |
|---|---:|
| Mean global cosine | 0.46729765232402365 |
| Median global cosine | 0.516699176351386 |
| Minimum / maximum cosine | 0.21207301111765345 / 0.7133872551552264 |
| Episodes with positive CCF–exact dot product | 100% |
| Mean sign agreement on common active coordinates | 0.8682722849303844 |
| Episodes where CCF reduced fixed-DAG loss | 100% |
| Episodes where equal-norm exact step reduced loss | 100% |
| Mean CCF fixed-DAG loss change | -0.0005729503761999077 |
| Mean equal-norm exact loss change | -0.0011856806325182556 |
| CCF post-update full-simulator mask stability | 64% |
| Exact post-update full-simulator mask stability | 63% |

The equal-norm exact direction reduced the local loss about `2.06943` times as
much as CCF on average.

### Where the update energy went

Each entry is the mean fraction of squared update norm in that edge group.

| Parameter group | CCF change | Exact direction |
|---|---:|---:|
| Cue input | 0.0000000000030805 | 0.0000000007737365 |
| Noise input | 0.2997139678542214 | 0.021385137090708634 |
| Query input | 0.5844497129722295 | 0.11315236015604288 |
| Recurrent | 0.07544289832439464 | 0.07524380013747928 |
| Output | 0.04039342084607408 | 0.7902187018420328 |

CCF placed about 14.02 times the exact direction's fraction into noise inputs
and 5.17 times its fraction into query inputs. The exact direction placed about
19.56 times CCF's fraction into output edges. Both directions assigned nearly
zero energy to the original cue.

## Decision

**Primary diagnosis: BOTH forward-memory-limited and credit-allocation-limited.**

The earlier memory probe showed that the cue was already severely attenuated,
and this exact derivative assigns essentially no sensitivity to cue edges. That
is direct evidence of a forward-memory bottleneck. CCF nevertheless points
downhill in every measured episode and narrowly satisfies the preregistered
global `DIRECTION-PLAUSIBLE` threshold. However, it spends most of its update
energy on query and noise edges rather than the output-dominated allocation of
the exact direction, and its equal-norm loss reduction is only about half as
large. Thus its broad sign is useful, but its parameter allocation is
inefficient.

This result is solid evidence about why the current prototype stays near
chance. It is not evidence that the complete Adaptive Learning Substrate goal
has been achieved, and it is not a novelty or market claim.

## Next controlled step

Keep CCF-v0/C000.1 unchanged and preregister an **architecture-only memory
intervention**. Compare a small fixed set of recurrent-memory regimes on fresh
custom seeds, first with the no-learning paired-cue probe and then with the same
alignment instrument. Only a regime that materially raises cue sensitivity
without instability should advance to a learning run. This separates the
forward-memory repair from any later CCF-v1 credit-equation change.

## Artifacts

- Primary report:
  `artifacts/experiment_000/alignment/custom_seeds_42_46_episodes_20.json`
  (SHA-256 `60F7CEDBFB83E92AE2CB596FE2A9DEB12139BFCFF45BF464C03EA474E8561739`)
- Independent identical rerun:
  `artifacts/experiment_000/alignment/custom_seeds_42_46_episodes_20_rerun.json`
  (SHA-256 `36E7AE365B1492D169D05A59A3F7C254E9E6A04FC90601566A50E5BFD7DD9F48`)
- Frozen protocol:
  `docs/EXPERIMENT_000_ALIGNMENT_PROTOCOL.md`
  (SHA-256 `B9FF19EEF6517066F43FC644665CEF8927224A00C148284D65BE55D9227AC6CC`)
- Frozen configuration:
  `configs/experiment_000_alignment.toml`
  (SHA-256 `4B017D33C814440D6C895DD087EB926A887E4935768DB3BB7619AAFD3840D9F8`)
