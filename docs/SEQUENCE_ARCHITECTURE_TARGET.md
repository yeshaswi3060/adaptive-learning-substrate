# Actual target: a competitive non-Transformer sequence architecture

## User-aligned objective

Build a model that consumes a sequence of tokens, maintains useful internal
state, and predicts the next token without Transformer attention. Eventually it
should learn continuously and retain knowledge through a local learning rule.
The first useful comparison is a small language model, not an AGI claim and not
a perfect score on a two-bit exercise.

Success means comparable prediction quality with a measurable advantage in
memory, throughput, context handling or continual adaptation. Replacing attention
and replacing end-to-end backpropagation are distinct research questions. Test
them separately so a failure can be attributed to the architecture or its learning
rule instead of conflating both.

## What exists, and what does not

- Exists: deterministic local-credit simulator, negative research results, bounded
  GPU recurrence/readout execution and a measured low-RAM CUDA runtime.
- Exists: a narrow delayed binary-cue representation that a trained local readout
  decodes well. Its latest control/statistical gates failed reproducibly.
- Missing: a trainable general token-sequence core, persistent document-level
  memory interface, next-token language-model evaluations and matched Transformer
  comparisons. The current code is not a Transformer replacement.
- Explicit context-slot skill banks and externally chained primitive adapters are
  not the next milestone. They would not establish a competitive sequence model.
- Official readiness remains 21/100. This charter does not change any evidence gate.

## First candidate to formulate and falsify

Working description: an input-gated sparse recurrent state model with local
eligibility traces. This is a hypothesis, not an implemented or validated model.
Avoid a novelty claim: recurrent gating, selective state, feedback learning and
eligibility traces all have relevant existing literature.

Token -> byte/token representation -> sparse state update -> vocabulary logits
-> next-token probabilities. No attention matrix and no growing key/value cache.

An initial mathematical skeleton is:

    u_t = input_representation(token_t)
    g_t = sigmoid(W_g u_t + a_g * s_(t-1) + b_g)
    c_t = tanh(W_in u_t + R_sparse s_(t-1) + b_c)
    s_t = g_t * s_(t-1) + (1 - g_t) * c_t
    p_(t+1) = softmax(W_out s_t + b_out)

Here * denotes coordinatewise multiplication. Each unit has a bounded sparse
set of recurrent parents. The gate must learn what to retain or replace; a
hard-coded latch for the first cue is not sufficient. Keep topology fixed for
the initial tests; autonomous growth should not hide a failing learning rule.

The local-training hypothesis uses a unit-local eligibility approximation plus
an output-error feedback signal. Cross-unit recurrent derivative paths omitted
by that approximation must be documented. Do not describe this approximation as
exact backpropagation, or describe existing feedback/eligibility ideas as novel.
Specify equations, update chronology, clipping and trace lifetime before running
it. Compare against a frozen-encoder readout and a conventional gradient-trained
version of the same recurrence to separate credit assignment from representation.
Gradient training is a comparison condition, not a redefinition of the project's
eventual no-global-backprop objective.

## Small first build

- Start with byte tokens (vocabulary 256), a small state (e.g. 128 units) and sparse
  recurrence; parameter sizes and exact equations require a prospective freeze.
- Expose reset(), step(token), predict_next(), learn(revealed_target), save/load.
  Never give the target to a prediction call; state persists across sequence chunks.
- Store working state and learning buffers on the GPU through the measured low-RAM
  runtime. Stream data; avoid loading whole corpora or allocating worker pools.
- Keep available-RAM/VRAM checks and measure full process/context overhead. An 8 GB
  GPU permits small experiments; it does not establish a competitive large model.
- Create an independent small CPU reference for numerical and causal checks, not
  a multi-gigabyte host-memory training dependency.

## Evidence ladder

1. **Causal token engine:** normalized probabilities, reproducible streaming state,
   chunked/unbroken equivalence, correct reset/checkpoint behavior, no future-token
   access, CPU/GPU agreement and bounded memory as context grows.
2. **Actual sequence learning:** delayed multi-token copy and keyed associative
   recall, including longer unseen delays and distractors. Report failed cases;
   do not substitute a binary cue score.
3. **Small language model:** train on a legally usable, document-separated text
   corpus with frozen splits/tokenizer; report held-out negative log-likelihood
   and perplexity. Generated examples are illustrations, not the success metric.
4. **Fair baselines:** unigram/bigram, a conventional small recurrent model, a tiny
   Transformer, and a relevant state-space model when feasible. Match data/token
   budgets and report parameters, training compute, context and tuning budget.
   State explicitly when any comparison has not actually been run.
5. **Candidate advantage:** select the quality/efficiency trade-off and thresholds
   before results, then repeat with fresh seeds. Measure tokens/sec, actual RAM/
   VRAM and quality against context length; linear state size alone is insufficient.
6. **Continual learning:** learn a second text domain, then re-evaluate the first.
   Measure forgetting and adaptation, rather than asserting retention from isolated
   parameter slots. Address growth/rewiring only after fixed-topology learning works.

Each level needs frozen configurations, raw measurements, controls, reproducible
reruns and clearly bounded claims. No probability or date for replacing
Transformers is established by the project's present results.

## Prior work that must inform, not be relabeled as, the candidate

- Mamba demonstrates attention-free selective state-space sequence modeling:
  https://arxiv.org/abs/2312.00752
- Eligibility-propagation research studies local eligibility traces combined with
  approximate learning signals for recurrent networks:
  https://www.nature.com/articles/s41467-020-17236-y

The next implementation deliverable is the causal token engine and its independent
reference, followed by a frozen training comparison. This charter is a direction
and evaluation contract, not a record that those deliverables already exist.
