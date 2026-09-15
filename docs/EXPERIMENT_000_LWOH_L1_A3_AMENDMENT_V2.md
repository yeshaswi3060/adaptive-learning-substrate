# Experiment 000 LWOH-L1 A3 contingency amendment V2

**Amendment version:** `experiment-000-lwoh-l1-v2`  
**Status:** prospective blinded pre-terminal amendment; inactive until its
separate freeze record is created and independently validated  
**Base protocol:** `docs/EXPERIMENT_000_LWOH_L1_PROTOCOL.md`  
**Base protocol SHA-256:**
`42e237add4eac9ca24a6d5781efefce6367f8c9f2c4c4065b324a1703706dc7e`  
**Base configuration:** `configs/experiment_000_lwoh_l1.toml`  
**Base configuration SHA-256:**
`7ed14e7dc85f7178e6dce0fbdf1e1d05d93c41129eba8b3c3f491dbfeffc86b8`  
**Overlay configuration:** `configs/experiment_000_lwoh_l1_v2.toml`  
**Claim level:** unchanged custom-seed development evidence only

## 1. Purpose and narrow amendment boundary

The base LWOH-L1 protocol is a prospective contingency whose trigger names the
superseded A2 procedure. It cannot be activated by an A3 result. This amendment
preserves the already specified LWOH-L1 mechanism, data construction, admission
and learning gates, compute matching, sample sizes, seeds, statistics, controls,
integrity requirements, deterministic reruns, stop rules, and readiness-score
rule. It changes only:

1. the predecessor trigger and prerequisite bindings from A2 to A3;
2. the cancellation and invalid-predecessor labels needed for that trigger;
3. the explicit exclusion of A3 seeds `105--109`; and
4. the artifact namespace and source-freeze bindings needed to keep V2 evidence
   distinct from the unexecuted V1 contingency.

No other key or value in the base protocol or configuration may be overridden.
The base documents remain immutable historical inputs. A future implementation
must load both base files, verify the two hashes above, apply only the allowlisted
V2 overlay, and reject every unrecognized or scientific override.

## 2. Blinded pre-terminal constraint

This amendment must be frozen and independently verified before the A3 terminal
artifact is committed and while
`artifacts/experiment_000/readout_trace_a3/DETERMINISM_VERIFICATION.json`
is absent. Until that freeze is committed, no A3 scientific result field may be
read, logged, summarized, used for control flow, or used to revise this
amendment. In particular, the A3 status, selected condition, passing list,
condition rows, per-seed values, aggregates, gate values, effect sizes, and
near-miss information are prohibited inputs.

Before the V2 freeze, automation may inspect only path existence, byte length,
timestamps, whole-file SHA-256 values, A3 source-freeze integrity, and
non-scientific phase/artifact bindings. The primary and rerun reports and their
sidecars may be hashed as opaque bytes but must not be parsed. The future V2
freeze record must state that the A3 terminal artifact was absent, identify the
allowed observations, bind their hashes, and contain no A3 scientific field.

This amendment was derived by substituting the valid A3 terminal branch into
the already fixed V1 contingency; it does not select or tune a mechanism from
A3 measurements.

## 3. A3 prerequisite binding

The only eligible predecessor is A3 protocol `readout-trace-v1a3` with all of
the following immutable bindings:

- A3 freeze record:
  `artifacts/experiment_000/readout_trace_a3/FREEZE_RECORD.json`, SHA-256
  `50c8a2af24394a15ec8f93abb2f7c851d60ca385fd65764e2d22bda2757bc164`;
- A3 canonical source-manifest SHA-256:
  `492e3165eed4cb3e5f425f74b615f5a71ca3d281a1107bb6fb8c701d204ae4db`;
- A3 protocol SHA-256:
  `cb023552d50d568c3fe1e44c40038b064c275b513785c26d8693efa21c666963`;
- A3 configuration SHA-256:
  `f1ca8047c7f5d559a4b6ac3888278d867826227917436af6f5873b6f4cfe0282`;
- A3 runner SHA-256:
  `99403c6706c5e7cdb56f518a696ba9e9cf5101f048e810cee99b07eb828cfdc5`.

The A3 source manifest contains exactly these fourteen paths, each of which
must still match its entry in the A3 freeze record:

1. `configs/experiment_000_readout_trace_a3.toml`
2. `docs/CCF_V0.md`
3. `docs/EXPERIMENT_000_READOUT_TRACE_A3_PROTOCOL.md`
4. `pyproject.toml`
5. `src/adaptive_learning_substrate/__init__.py`
6. `src/adaptive_learning_substrate/events.py`
7. `src/adaptive_learning_substrate/experiment000_data.py`
8. `src/adaptive_learning_substrate/experiment_000.py`
9. `src/adaptive_learning_substrate/experiment_000_memory_probe.py`
10. `src/adaptive_learning_substrate/experiment_000_readout_trace_a3.py`
11. `src/adaptive_learning_substrate/experiment_000_slow_state.py`
12. `src/adaptive_learning_substrate/readout_trace_recurrent.py`
13. `src/adaptive_learning_substrate/recurrent.py`
14. `tests/test_experiment_000_readout_trace_a3.py`

The required A3 artifact chain is the frozen source record, pre-freeze
verification, persisted smoke and sidecar, smoke verification, phase sequence,
primary full report and sidecar, separate-process rerun report and sidecar, and
terminal determinism verification at their registered paths.

## 4. Exact V2 activation rule

LWOH-L1 V2 activates only when a read-only loader independently revalidates the
complete seven-phase A3 chain and returns terminal status exactly
`NO_SELECTION`. The loader must re-hash the fourteen frozen sources, validate
the A3 freeze record and phase sequence, validate both full reports and binding
sidecars, replay the registered evidence and deterministic payload, require
fresh distinct coordinator processes, and validate the terminal record against
both reports with the A3 validator's non-selection-capable semantics. It must
then repeat the source, freeze, phase, and artifact checks before returning a
minimal provenance binding.

The first valid activation must occur while both separately registered
post-activation source paths remain absent.  The controller then commits the
minimal binding, with no scientific fields, as the write-once pair
`artifacts/experiment_000/lwoh_l1_v2/activation/A3_NO_SELECTION_ACTIVATION.json`
and
`artifacts/experiment_000/lwoh_l1_v2/activation/A3_NO_SELECTION_ACTIVATION.sha256`.
Later read-only loads may coexist with the new execution sources only by
revalidating that activation pair, the current A3 terminal chain, and exact
equality of their provenance hashes.  A future source that exists before this
first activation invalidates the transition and creates no activation pair.

The selected-only A3 downstream loader is not an activation path because it
intentionally rejects `NO_SELECTION`. The V2 loader must not invoke or rewrite
the A3 terminal verifier after its terminal artifact exists. It may return only
the terminal status and provenance hashes; it must not expose A3 scientific
fields to LWOH-L1.

The branch outcomes are fixed:

- exact valid `NO_SELECTION` -> `ACTIVATED_A3_NO_SELECTION`;
- exact valid `SELECTED:<registered_condition>` ->
  `CANCELLED_A3_SELECTED`, with no LWOH-L1 seed generated;
- missing, incomplete, provisional, aborted, malformed, non-deterministic,
  source-drifted, sidecar-invalid, phase-invalid, replay-invalid, or otherwise
  invalid A3 evidence -> `INVALID_A3_DO_NOT_ACTIVATE_LWOH`; repair A3 instead.

No primary or rerun status alone can activate this contingency. Exit status
zero alone cannot activate it.

## 5. Unchanged scientific design and seeds

Every scientific section of `configs/experiment_000_lwoh_l1.toml` is inherited
byte-for-byte and semantically unchanged, including `graph`, `memory`,
`admission`, `admission.gates`, `learner`, `head_alignment_tests`, `conditions`,
`data`, every `data.eval_cells` table, `scratch`, every `learning_gates` table,
`control_integrity`, `compute`, `integrity`, and `decision`.

The fixed LWOH-L1 seeds remain:

- admission: `90,91,92,93,94`;
- learning: `95,96,97,98,99,100,101,102,103,104`;
- nonselecting scratch only: `9090,9091`.

The existing forbidden seed partitions remain forbidden. A3 seeds
`105,106,107,108,109` are additionally forbidden. Confirmatory seeds
`1000--1019` remain forbidden. No new tuning, candidate grid, threshold change,
sample-size change, or seed substitution is permitted.

## 6. Separate V2 artifacts and future implementation boundary

The V2 namespace is `artifacts/experiment_000/lwoh_l1_v2/`. The pre-terminal
amendment freeze will be
`PRE_TERMINAL_CONTINGENCY_FREEZE.json` with a binding `.sha256` sidecar. A V2
phase ledger and all later admission, learning, rerun, and verification evidence
must remain in the same V2 namespace. No V1 artifact path may be reused.

V2 uses a two-stage implementation boundary.  The existing
`src/adaptive_learning_substrate/experiment_000_lwoh_l1.py` and
`tests/test_experiment_000_lwoh_l1.py` are the **pre-terminal controller** and
its non-scientific tests.  They implement only the blinded freeze and the
read-only A3 terminal loader.  The pre-terminal freeze binds those files, this
amendment and overlay, the immutable V1 inputs, and their validation evidence.
After that freeze, every bound controller, test, document, configuration, and
dependency is immutable; none may be edited to implement LWOH-L1 execution.

The scientific execution runner and its tests are deliberately absent before
the A3 terminal decision.  Only after the immutable controller commits the
write-once minimal activation-provenance pair above with status exactly
`ACTIVATED_A3_NO_SELECTION` may these separately preregistered new files be
created:

- `src/adaptive_learning_substrate/experiment_000_lwoh_l1_execution_v2.py`;
- `tests/test_experiment_000_lwoh_l1_execution_v2.py`.

They are additions, never replacements or edits of the pre-terminal controller
or its tests.  Their implementation must encode the already immutable V1
scientific design plus this trigger-only overlay without changing a mechanism,
gate, threshold, sample size, compute match, seed, condition, analysis, stop
rule, or score rule.  A valid A3 selection, invalid A3 result, or incomplete A3
result does not permit LWOH-L1 execution.

Before any LWOH-L1 task metric, the new execution runner and tests, all of their
transitive runtime dependencies, and the immutable pre-terminal controller
bundle must be committed to
`artifacts/experiment_000/lwoh_l1_v2/IMPLEMENTATION_SOURCE_FREEZE.json` with a
binding `.sha256` sidecar.  A separate fresh process must validate that freeze,
the pre-terminal freeze, the immutable A3 activation-provenance pair, and exact
inheritance of every V1 scientific-section hash.  Focused tests, lint,
compilation, synthetic scratch verification, manifest validation, and rollback
verification must pass before official data generation.  This post-activation
implementation freeze makes the `NO_SELECTION` route usable without weakening
or mutating the blinded pre-terminal freeze.

## 7. Freeze and validation requirements

The pre-terminal V2 freeze record must use sorted-key compact ASCII JSON with no
NaN values and must bind:

- this amendment and overlay, plus both immutable V1 base hashes;
- the exact A3 freeze-record hash and fourteen-file source bundle above;
- opaque whole-file hashes for every A3 prerequisite artifact then present;
- the fact that the A3 terminal artifact was absent at freeze time;
- UTC and local timestamps, environment identity, exact commands, exit statuses,
  and strictly structured non-scientific result summaries used to validate the
  amendment; raw pytest output is reduced to its passing-test count and cannot
  carry arbitrary A3 content;
- an allowlisted semantic-delta result proving that no scientific field differs
  from V1;
- a canonical V2 source-manifest hash, record self-hash, and binding sidecar.

A separate fresh process must re-open the record and every bound file, recompute
the self-hash, sidecar, V1 hashes, V2 manifest, A3 source bundle, and allowed
semantic delta, and return an exact machine-readable pass before the A3 terminal
artifact may be committed.  The enforceable boundary is terminal-artifact
commit, not an unobservable process-invocation instant; the unchanged frozen A3
runner exposes no shared invocation lock. Any mismatch invalidates this
amendment and leaves the V1 contingency and readiness score unchanged.

The pre-terminal source manifest intentionally contains the controller and its
tests but not the two absent execution files.  Creation of the two exact
post-activation paths does not alter or invalidate that manifest.  Conversely,
editing any pre-terminal-bound path invalidates the controller freeze.  The
post-activation implementation freeze must bind the union of the immutable
pre-terminal sources and the two new execution files before any scientific run.

## 8. Terminal score rule

The V1 terminal and score rule is unchanged. Only a fully activated,
independently verified LWOH-L1 admission and learning result exactly
`LWOH_LEARNING_PASS` may move readiness from `21` to `25`. Every admission,
learning, source, phase, replay, sidecar, or determinism failure preserves
readiness at `21`; no partial score is permitted.
