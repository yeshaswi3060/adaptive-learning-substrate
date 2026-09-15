# Continual two-bit skills V1 — prospective development protocol

**Status: NOT EXECUTED; set aside following the user's clarification.**
The immediate target is a native sequence-model architecture that could compete
with Transformers, not an explicitly routed skill bank. See
`SEQUENCE_ARCHITECTURE_TARGET.md`. This document records an abandoned proposal,
not a result, active experiment, or readiness advance.

## Question and honest claim boundary

Can the tested low-RAM GPU memory/readout components support learning a new
primitive during use without losing old primitives? This tests a usable capability,
not a new CCF rule. Explicit skill identifiers, fixed preallocated context slots,
and an external composition controller are provided. No autonomous task discovery,
structural growth, learned routing, native single-episode composition, novelty,
or official readiness increase is claimed. CCF's closed gates remain unchanged.

## Fixed design before outcomes

- Five fresh development graph seeds: 14000..14004. No original V3 or confirmatory
  seeds. Skill order: COPY, NOT, SWAP, then FLIP0 as the new fourth skill.
- Two independent bounded recurrent lanes encode the two input bits. Both use the
  unchanged seed's sparse graph and TTL32 observer. The first lexical observer
  coordinate of each lane supplies two real features. No target, correct output,
  analytic cue decoder or rule identity enters those numerical memory features.
- Eight-coordinate local heads, one per output bit, zero initialized. The context
  identifier chooses one of four fixed two-coordinate feature blocks. All four
  slots exist before learning; only feature placement changes, not parameter count.
  Each head uses the unchanged local tanh delta rule and bounded updates.
- Baseline: same two eight-coordinate heads, same memory, examples, ordering,
  feature norm and update count. Context code is (one_hot + 0.5) / sqrt(3), yielding
  full-rank but overlapping features. Both models can represent all four tasks;
  this is a representation/interference comparison, not a general SOTA comparison.
- Eight independent-label replicate models, using isolated features. For each
  skill/output/replicate, exactly balanced bipolar labels use separate PCG64 streams.
  Controls do not alter learning examples or select a model. Their aggregate
  per-bit accuracy should be 45–55%, joint accuracy 20–30%; these are diagnostic
  controls, not a replacement for or repair of the earlier failed V1 control.
- Each skill gets 128 training examples (each of the four inputs occurs 32 times),
  exactly one online pass at D8. All predictions are made before target reveal;
  each output head gets exactly one update per example. No stored-example replay.
- After every stage, evaluate every acquired skill on 16 fresh D8 examples.
  At the end also evaluate every skill at D4 and D16. No evaluation updates.
  All inputs, noise, target reveals, predictions, update deltas and theta states
  are saved. Random streams are keyed by seed/stage/split/delay/purpose with SHA256.
- At each stage: newly acquired joint accuracy >=95%; previously acquired joint
  accuracy >=95%; forgetting from each skill's first post-training D8 score <=2
  percentage points. Final D4/D16 joint accuracy >=95% for every skill and seed.
- After training, test all 16 length-two and 64 length-three ordered compositions
  on all four inputs (320 cases/seed, D8). The controller passes the *predicted*
  bits to the next learned primitive. Training never contains compositions.
  Require joint accuracy >=90% per seed. This explicitly remains external chaining.
- Report dense-baseline results unconditionally. A >=10 percentage-point mean
  final retention advantage is a separate mechanism-evidence gate; parameter and
  logical update counts must match. A failure does not permit tuning or seed search.
- Check every main/baseline training update and all their evaluation outputs
  against separate CPU OnlineHead calculations on the same features. Check the
  first episode in each 16-example batch against native CPU recurrence. Negative
  controls get stream/label/update-ledger regeneration during separate verification.
- Run the complete experiment twice in distinct processes; scientific payloads
  must match exactly. Freeze source/config/protocol/compiler hashes before metrics,
  check them each seed and before publication, use exclusive JSON writes and
  project-compatible JSON SHA256 sidecars. Keep raw evidence and failures.
- Resource guard: >=512 MiB free host RAM at startup, >=128 MiB at each batch
  boundary; one CUDA process, no torch import, <=16 MiB explicit device buffers,
  >=256 MiB initial device headroom. Resource limits exclude compiler/context costs.

## Advancement

A pass is a small development capability: explicitly addressed primitive skills
can be learned sequentially, retained and externally chained on this exhaustive
two-bit domain. It does not finish the project's central novel-local-credit,
autonomous-growth or generalization hypothesis and does not change 21/100.
Publish failed gates and quantitative progress rather than changing a score label.
