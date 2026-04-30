# Matched-compute follow-up (#3): k=3 / h=512 saturation test

## Why

Sweep 1 (k=3) showed \texttt{oap} narrowly beating \texttt{random\_augmented}
at h=512 ($0.2431$ vs $0.2395$, +0.0033 AP) under the standard 200-epoch
protocol. The current explainer PDF hand-waves this as "likely also a
compute artefact" without direct evidence. The adversarial review
(2026-04-30) correctly noted this is the *weakest* cell for the
compute-artefact hypothesis: at k=3 the sign group is $\mathbb{Z}_2^3 = 8$
patterns — easy to learn invariance over in 200 epochs.

This sweep tests it directly: same matched protocol as Sweep 4 (500 ep /
patience 999 / no early stop), but at k=3 / h=512 across all three
methods so we have a direct head-to-head.

## Configuration

| Knob | Value |
|---|---|
| Canonicalizations | `map`, `oap`, `random_augmented` |
| k | 3 |
| cache-n-eigs | 15 (existing cache, no rebuild) |
| Hidden dim | 512 only |
| Seeds | 0, 1, 2 |
| Epochs | 500 |
| Patience | 999 (effectively disabled) |
| Eigval scale | off |
| Test-aug-samples | 16 (only affects random_augmented) |
| Base-dir | `results/matched_compute_k3_h512/` |

Total: 3 × 1 × 3 = **9 runs**, ~30–40 h wall-clock.

## Targets (200-ep numbers from Sweep 1)

| canon              | h=512 (200ep)        |
|--------------------|----------------------|
| map                | 0.2397 ± 0.0122      |
| **oap**            | **0.2431 ± 0.0022**  |
| random_augmented   | 0.2395 ± 0.0117      |

## Decision after this sweep

- **If random_augmented (500ep) > oap (500ep)** at h=512: the compute
  artefact hypothesis holds at k=3 too. Lone canonicalization edge
  closed; universal claim defensible.
- **If oap (500ep) ≥ random_augmented (500ep)** at h=512: the lone
  canonicalization edge is real. The headline "random augmentation
  beats canonicalization" needs the qualifier "at k≥8 or with
  matched compute" depending on outcomes.
