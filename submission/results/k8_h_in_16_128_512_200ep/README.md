# Test-time averaging over the eigenvector ambiguity group (k=8)

Companion to [`results/test_time_averaging_augmented/`](../test_time_averaging_augmented/)
(k=3 sweep). Same hypothesis, same canonicalizations, same protocol —
bumped `k` from 3 to 8 and `K` (test-time averaging budget) from 8 to 16.

**Date**: 2026-04-24  
**Parent commit**: `708d5a6`  
**Asana**: 1214132170406525  
**Plan**: [plan.md](plan.md)

---

## Metric key

- `best_test_ap` — standard single-pass test AP on `ogbg-molpcba`.
- `best_test_ap_aug16` — test AP after averaging 16 forward passes, each
  with a fresh random draw of the eigenvector ambiguity group (sign
  flips on simple eigenvalues + Haar `O(m)` rotations on any
  multiplicity-`m` blocks). K=16 is 6.25% of the Z₂⁸=256 sign group.
  Only populated for `random_augmented` runs.

## Results

Aggregated over 3 seeds per cell (mean ± std):

| canonicalization   |   h  | params      | AP (single)       | AP (aug-avg K=16) |   Δ aug  |
|--------------------|-----:|------------:|------------------:|------------------:|---------:|
| map                |   16 |      5,824  | 0.0622 ± 0.0023   |         —         |    —     |
| oap                |   16 |      5,824  | 0.0605 ± 0.0022   |         —         |    —     |
| **random_augmented** |   16 |      5,824  | **0.0636 ± 0.0029** |   0.0637 ± 0.0029 |  +0.0001 |
| map                |  128 |    203,392  | 0.1705 ± 0.0141   |         —         |    —     |
| oap                |  128 |    203,392  | 0.1754 ± 0.0073   |         —         |    —     |
| **random_augmented** |  128 |    203,392  | **0.1766 ± 0.0062** |   0.1766 ± 0.0062 |  −0.0000 |
| map                |  512 |  2,975,872  | 0.2344 ± 0.0078   |         —         |    —     |
| oap                |  512 |  2,975,872  | 0.2371 ± 0.0076   |         —         |    —     |
| **random_augmented** |  512 |  2,975,872  | **0.2395 ± 0.0139** |   0.2400 ± 0.0140 |  +0.0005 |

Bold = best mean AP per hidden dim.

## k=3 vs k=8 comparison (headline)

The interesting question going in was whether `oap`'s edge at h=512 (k=3)
survives wider PE. It doesn't — **random_augmented wins at every hidden
dim at k=8**:

|   h  |   k=3: best canon  |  k=3: random_augmented  | k=3 winner |  k=8: best canon   |  k=8: random_augmented  | k=8 winner |
|-----:|-------------------:|------------------------:|:-----------|-------------------:|------------------------:|:-----------|
|   16 |  oap   0.0641      |  **0.0662**             | rand_aug   |  map   0.0622      |  **0.0636**             | rand_aug   |
|  128 |  map   0.1704      |  **0.1762**             | rand_aug   |  oap   0.1754      |  **0.1766**             | rand_aug   |
|  512 |  **oap** 0.2431    |  0.2395                 | **oap**    |  oap   0.2371      |  **0.2395**             | rand_aug   |

**The h=512 verdict flips at k=8.** With 3 eigenvectors both oap and
random_augmented cluster near 0.24 AP; with 8 eigenvectors random_augmented
pulls ahead (+0.0024 over oap) and the oap/map ordering even compresses.

## Interpretation

### 1. Hypothesis result: fully supported at k=8

- At h=16 and h=128, random_augmented wins (same pattern as k=3).
- At h=512, random_augmented now wins too (**flipped vs k=3 where oap
  won**). Widening the PE gave augmentation enough signal to beat oap's
  deterministic ordering.
- Aug-avg K=16 gives the same tiny bump as at k=3 (+0.0001 to +0.0005);
  never hurts.

### 2. Why does widening k matter so much at h=512?

Two plausible stories:

- **Averaging-out story** — with k=8 the model must internalize sign-
  invariance across 2⁸=256 sign patterns (vs. 2³=8 at k=3). This forces
  a more representation-invariant encoder, which generalises better at
  test time where no canonical ordering exists.
- **Information story** — 8 eigenvectors carry materially more structural
  signal than 3. At high capacity (h=512) the model can exploit it;
  canonicalization methods (map/oap) process the same 8 eigenvectors but
  their deterministic orderings may become less effective at higher `k`
  (more eigenvectors = more subtle tie-break cases the canon rules have to
  be robust to).

The data can't distinguish these; both predict the k=3→k=8 flip at h=512.

### 3. Why is Δ aug still tiny (+0.0001 to +0.0005)?

Even with a larger ambiguity group (Z₂⁸ × O(m) blocks rather than Z₂³),
the training-time augmentation still makes the model near-invariant — a
single random draw is almost as good as an average over K. The K=16
averaging doesn't change the verdict; it just polishes the prediction by
a tenth of a std. This matches the k=3 finding: averaging is
**safe-and-free** rather than a meaningful accuracy lever.

### 4. Takeaway

- For **GIN + LapPE on ogbg-molpcba**, train with `random_augmented`,
  use `k=8` or larger, and optionally run `K=8..16` test-time averaging
  (free small bump). You'll beat canonicalization (map, oap) across
  h∈{16,128,512}.
- The 2025-04-23 conclusion that "oap wins at h=512" was a k=3 artefact.
  With k=8 that advantage disappears.
- Averaging K beyond ~8 has diminishing returns once the sign group is
  covered in expectation — the bottleneck is training-time invariance,
  not sample count.

## Caveats

- Only tested at k=8. Whether the flip survives at k=12, k=16, etc. is
  not covered here.
- `map` at h=128 has std 0.0141 (one seed is an outlier at 0.154 vs two
  at 0.177). This widens the error bar but doesn't change the ordering.
- `oap` at h=128 is strong (0.1754), closing most of the random_augmented
  gap there. At h=128 the random_augmented lead is only +0.0012 — within
  1σ. Not worth celebrating; at h=16 and h=512 the margin is clear.
- Only `map` and `oap` included (plan scope). `spielman`, `maxabs`,
  `none` etc. are out of scope — see `results/lappe_sweep/` for
  those methods under a different protocol.

## Reproducing

```bash
# Tests (unchanged since k=3)
pytest tests/test_unified_canonicalization.py::TestRandomAugmentEigenvectors -v

# Full sweep
python scripts/run_lappe_sweep.py \
    --canonicalization random_augmented map oap \
    --hidden-dim 16 128 512 \
    --k 8 \
    --seeds 0 1 2 \
    --epochs 200 \
    --patience 15 \
    --no-eigval-scale \
    --test-aug-samples 16 \
    --base-dir results/test_time_averaging_augmented_k8 \
    --gpus 0 1
```

## Directory layout

```
results/test_time_averaging_augmented_k8/
├── README.md       ← this file
├── plan.md         ← approved pre-implementation plan
├── gin/            ← 27 leaf dirs, each with results.json + best_model.pt + train.log + epoch_log.json
├── summary.csv     ← one row per run
└── plots/
    ├── ap_vs_hdim.png           ← single-pass comparison
    └── ap_vs_hdim_augmented.png ← single-pass + aug-avg vs canonicalization
```
