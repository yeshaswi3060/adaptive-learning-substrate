# Experiment 000 development log

This log preserves development observations. It is not confirmatory evidence,
and seeds 1000–1019 have not been executed.

## Trial D000 — pre-correction routing audit

- Date: 2026-09-01
- Seeds: 0, 1, 2, 3, 4 (development only)
- Episodes: 100 training + 100 evaluation per seed
- Frozen-default settings: learning rate 0.01, trace decay 0.97, route gain
  0.90, credit minimum 1e-8, trace horizon 32, hop limit 16
- Mean evaluation accuracy: CCF-v0 0.482, CCF-NO-TRACE 0.482, RAND 0.488
- Full CCF weight writes: 1,546,215 across 500 training episodes
- Full CCF small-credit stops: 290,793
- Non-finite values: 0

The result was at chance and provided no evidence of learning. Audit found that
the implementation discarded each routed packet below `c_min` before packets
addressed to the same parent could reconverge. This contradicted the aggregated
event-credit rule in Sections 7.1 and 7.5 of `CCF_V0.md` and invalidated D000 as
a test of the specified mechanism. D000 is therefore retained as an engineering
bug-discovery record, not as evidence about CCF-v0.

## Correction C000.1

The early per-route threshold was removed. Routed packets now accumulate by
exact parent event ID in descending event-time order, and `c_min` is applied to
the signed aggregate when that parent event is processed. A constructed
two-branch reconvergence test verifies that two individually sub-threshold
packets can jointly update their shared ancestor.

The fixed timing contract is also explicit: `QUERY` occurs at event time 9, the
one-tick graph advances through two readout delays, output is forced at graph
tick 11, and only then is the terminal target revealed.

## Trial D001 — post-correction audit

- Date: 2026-09-01
- Mechanism: CCF-v0
- Implementation revision: C000.1
- Scope: reduced balanced diagnostic, not the canonical 2,000-episode run
- Seeds: 0, 1, 2, 3, 4 (development only)
- Episodes: 100 training + 100 evaluation per seed
- Effective configuration SHA-256:
  `b37c4dfb400e67d8f6aaad566ed43899f1f29ee1f6bbd6659ea0902cc8c52a0f`
- Report SHA-256:
  `EAD9B88CEE4B31CD7B085BD782977C3FEEF3D621043631DAF25664C6736D55DE`
- Runtime: 45.040066400004434 seconds on NumPy CPU float64; GPU not used

| Seed | CCF-v0 | CCF-NO-TRACE | RAND |
|---:|---:|---:|---:|
| 0 | 0.50 | 0.50 | 0.40 |
| 1 | 0.49 | 0.48 | 0.42 |
| 2 | 0.50 | 0.50 | 0.59 |
| 3 | 0.47 | 0.48 | 0.47 |
| 4 | 0.45 | 0.45 | 0.54 |
| **Mean** | **0.482** | **0.482** | **0.484** |

Mean CCF-v0 minus CCF-NO-TRACE was 0.000. Mean CCF-v0 minus RAND was
-0.002. Zero of five CCF-v0 seeds reached 0.85. Every topology hash stayed
fixed, evaluation changed no weight, and every run remained finite.

The aggregation correction therefore fixed a real invariant but did not recover
learning within this reduced budget. Full CCF performed about 1.55 million
weight writes across 500 training episodes while remaining at chance. This is a
valid negative **engineering diagnostic**, but it is not the preregistered
2,000-episode development gate and does not show that later learning is
impossible. It supports diagnosing representation and update direction before
spending the larger budget; confirmatory seeds remain closed.

Artifacts are stored at
`artifacts/experiment_000/development_100_corrected/`.

## Trial D001-A — provenance and metric audit rerun

The same CCF-v0/C000.1 reduced 100/100 diagnostic was rerun after correcting
report-only defects. Its behavior reproduced exactly: mean evaluation accuracy
was 0.482 for CCF-v0, 0.482 for no-trace, and 0.484 for RAND.

The new package binds source/specification hashes, full edge/initial-weight
manifests, hardware identity, artifact hashes, and a raw mechanism audit sample
for the first training episode of every learned method and seed. It also keeps
high-water gauges separate from additive ledger counters, distinguishes peak
from final weight magnitude, reports RAND activation as unavailable, and counts
silent roots separately for train and evaluation.

- Effective configuration SHA-256:
  `83655049388985c22cb42dfed213850fdcdf4a193825e5278f25c01cbc5e7b7d`
- Source/specification bundle SHA-256:
  `207d481b0ebf66c6346a06d13e88dddcb04d2211109f43c51dc668d5512da0cb`
- Report SHA-256:
  `9F24C6BE603EB4AF02BDAAC93FDCFF1FAA1CDE14261AB51D44B2127E7498C621`
- Artifact manifest SHA-256:
  `E40FEA529426CEA8C72D75AC025E4C426E165893B899E49BA6B311823D5C6F41`
- Runtime: 46.89198430000397 seconds
- Hardware record: Intel64 processor; NVIDIA GeForce RTX 5060 Laptop GPU
  detected; NumPy CPU float64 backend used; GPU not used

The package deliberately marks `protocol_archive_complete=false`: mechanism
logs are deterministic one-episode audit samples rather than complete bulk-run
logs. A compressed streaming logger is still required before a canonical or
confirmatory archive can meet the full raw-event requirement.

Artifacts are stored at
`artifacts/experiment_000/development_100_c000_1_audited/`.

## Positive control P001 — clean 11-edge delayed path

To distinguish a total routing failure from a distributed-memory failure, a
hand-built path carried the cue across the same eight event delays to output
tick 11. Exactly one internal edge, `chain:h04->h05`, was plastic and began at
-0.9; every other edge, including the readout, was frozen. Thus the no-trace
ablation could not solve the task by changing the final readout.

After 100 development training episodes, full CCF changed the earlier edge from
-0.9 to 0.12472518983448343 and reached 1.0 evaluation accuracy. No-trace made
zero weight writes, left the edge at -0.9, and remained at 0.0 accuracy. Both
topologies stayed fixed and both evaluations preserved their weights.

Result: **PASS**. Exact timestamped credit routing can repair an earlier edge on
a clean causal chain. D001 therefore narrows the failure to learning a useful,
distributed recurrent memory amid noise and many competing pathways; it does
not show that CCF works on the actual recurrent gate.

Artifact:
`artifacts/experiment_000/chain_positive_control.json`  
SHA-256:
`8696E877E16A51384B9D4A7737BA232BED9CDD7B5B6F9B7D1A73EDFF85B47CAF`

## Memory probe M001 — same noise, flipped cue

For each development noise stream, the untrained frozen graph was run twice:
once with cue 0 and once with cue 1. Because the noise was identical inside a
pair, their difference measures surviving cue information while their midpoint
measures noise-driven variation. No credit was applied and every weight hash
remained unchanged.

Across seeds 0–4:

- mean recurrent spectral radius: 0.36045983092162037;
- mean evaluation cue/noise feature ratio: 0.15163634325909459;
- mean output cue-delta/output-SD ratio: 0.1926050665950054;
- mean optimistic ridge-readout evaluation accuracy: 0.5740000000000001.

The spectral radius is the recurrent matrix's average per-step expansion or
contraction factor. A value near 0.36 means old signals contract strongly;
`0.36^9 ≈ 0.00010` is the rough nine-step linear attenuation before accounting
for `tanh`, emission gating, and eight new noise inputs. The paired ratios below
1 mean noise changes the query features much more than flipping the cue. Even a
separate trained linear decoder recovered the cue only slightly above the 0.5
chance level.

This supports a concrete diagnosis: C000.1 can route credit down a clean path,
but the current random recurrent initialization does not preserve a strong cue
representation long enough for that credit to select it reliably. The next
mechanism diagnostic should compare CCF's update vector with an exact
fixed-event-DAG descent direction before any architecture or equation change.

Artifact:
`artifacts/experiment_000/memory_probe/development_seeds_0_4_pairs_100.json`  
SHA-256:
`BE8EACCDCCE7BBC184D5166D92D6A3DD54E8C37CB204352E312F60D63E09E799`

## Diagnostic D002 — exact fixed-event update alignment

The preregistered update-alignment diagnostic ran on fresh custom seeds 42–46,
20 balanced sequential episodes per seed. It differentiated each realized
timestamped event DAG without autograd, accumulated every temporal use of a
shared structural weight, and compared CCF's actual change with an equal-norm
exact local descent direction. Confirmatory seeds remained closed.

Across 100 episodes, median global cosine was 0.516699176351386, mean cosine
was 0.46729765232402365, and every dot product was positive. CCF reduced the
fixed-DAG loss in every episode, but its mean loss change was only
-0.0005729503761999077 versus -0.0011856806325182556 for the equal-norm exact
direction. The exact direction was therefore about 2.06943 times more effective
per equal update norm.

The parameter allocation identified two coupled bottlenecks. Cue edges received
only 3.0805e-12 of CCF's squared update norm and 7.7374e-10 of the exact
direction's norm. Meanwhile CCF allocated 0.2997 to noise and 0.5844 to query,
whereas the exact direction allocated 0.7902 to output edges. This supports the
preregistered **BOTH** diagnosis: weak forward cue memory plus inefficient
credit allocation. CCF's broad direction still narrowly meets the diagnostic's
global `DIRECTION-PLAUSIBLE` threshold, so the update is not simply reversed.

All 3,200 fixed-DAG and all 3,200 mask-stable dynamic finite differences passed;
maximum error was 7.0261e-12. Two independent full runs produced the identical
deterministic payload SHA-256
`17573dd402b252c82c3c22286f6cb0898bb9cd50fd8d37b6cdf57ea38fa7ec58`.

Primary artifact SHA-256:
`60F7CEDBFB83E92AE2CB596FE2A9DEB12139BFCFF45BF464C03EA474E8561739`.
Full interpretation and rerun hashes are in
`docs/EXPERIMENT_000_ALIGNMENT_RESULT.md`.

## Diagnostic M002 -- architecture-only recurrent-radius sweep

The frozen A1 memory-strength protocol ran on fresh custom seeds 50--54 with
100 paired train noise streams and 100 paired evaluation streams per seed. The
native graph and targets `0.60`, `0.80`, `0.95`, and `1.05` were evaluated with
no credit event and no weight update. Each full run contained 10,000
counterfactual forward episodes across all seeds and conditions.

The official decision was **`NO_SELECTION`**. The native median evaluation
cue/noise ratio was 0.14621176936848163. Target 0.95 raised median absolute cue
RMS by 14.901744392807 times and ridge accuracy to 0.995, but its median
cue/noise ratio was only 0.317838049315261 and it exceeded the activity guards.
Target 1.05 raised median absolute cue RMS by 34.398928301751 times and reached
ridge accuracy 1.0, but its cue/noise ratio was 0.46844687493377773, only three
of five seeds reached the required threefold ratio improvement, activity was
too high, and four of five seeds failed late blank-tail quiescence.

Both full executions produced deterministic payload SHA-256
`f6ba680f91917521da5bbfb89d7d1dbd102fe59149049d0ed0464bb3ece7f8e6`.
The primary report SHA-256 is
`5DD993BA62770CE41EE84A8C8A42B80C642445A734FA4FDDD537D2500D641A4C`;
the rerun report SHA-256 is
`1BE4397C94DC28D25D70F0A3B722D4E4530A2C86581010DFBD1B171B95B245C5`.
All weights and explicit topologies stayed unchanged, every value remained
finite, and confirmatory seed count was zero. Uniform recurrent scaling is
rejected; no best-looking failure advances.
