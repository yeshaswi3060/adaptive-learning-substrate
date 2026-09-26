# Documentation index

Start with the illustrated [project guide](PROJECT_GUIDE.md). It explains the
idea, code layout, current failures, and next experiment in plain language.

| Read this for | Document |
| --- | --- |
| Official research gate and evidence state | [Canonical status](CANONICAL_STATUS.md) |
| Detailed sequence experiment timeline | [Sequence progress](SEQUENCE_PROGRESS.md) |
| Evidence-based readiness rubric | [Research readiness](RESEARCH_READINESS_90.md) |
| Implemented and pending Stage 1 work | [Implementation status](IMPLEMENTATION_STATUS.md) |
| Sequence model goal and evaluation ladder | [Sequence architecture target](SEQUENCE_ARCHITECTURE_TARGET.md) |
| Original local-credit equations | [CCF-v0](CCF_V0.md) |

## Experiment families

| Family | Main documents | Purpose |
| --- | --- | --- |
| Stage 1 causal credit | [Experiment 001](EXPERIMENT_001.md), [development log](EXPERIMENT_000_DEVELOPMENT_LOG.md), [alignment result](EXPERIMENT_000_ALIGNMENT_RESULT.md) | Build and test delayed local credit on a fixed graph. |
| Gate A memory | [Memory sweep result](EXPERIMENT_000_MEMORY_SWEEP_RESULT.md), [slow-state result](EXPERIMENT_000_SLOW_STATE_RESULT.md), [readout-trace result](EXPERIMENT_000_READOUT_TRACE_RESULT.md), [A3 protocol](EXPERIMENT_000_READOUT_TRACE_A3_PROTOCOL.md) | Check whether recurrent cue information survives delay without excess activity. |
| Sequence core | [Core protocol](SEQUENCE_CORE_V1.md), [sequence progress](SEQUENCE_PROGRESS.md) | Implement and assess a native token predictor. |
| Sequence memory and learning | [Memory](SEQUENCE_MEMORY_V1.md), [trainability](SEQUENCE_TRAINABILITY_V1.md), [credit](SEQUENCE_CREDIT_V1.md) | Separate memory capacity, training rule, and optimization effects. |
| Sequence generalization | [Diversity](SEQUENCE_DIVERSITY_V1.md), [binding audit](SEQUENCE_BINDING_AUDIT_V1.md) | Test new cues, changed values, and state information. |

Some documents are frozen protocols or historical proposals. Their original
wording records what was planned at that time. For current claims, consult
`CANONICAL_STATUS.md` for Gate A and `SEQUENCE_PROGRESS.md` for sequence work.
The local `artifacts/` directory contains experiment evidence but is Git ignored.
