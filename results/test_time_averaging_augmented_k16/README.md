# Test-time averaging over the eigenvector ambiguity group (k=16)

Third sweep in the series after
[`results/test_time_averaging_augmented/`](../test_time_averaging_augmented/)
(k=3) and
[`results/test_time_averaging_augmented_k8/`](../test_time_averaging_augmented_k8/)
(k=8). Same protocol; bumped `--n-eigs 8 → 16` and rebuilt caches at
`--cache-n-eigs 16`. K=16 (same as the k=8 sweep) for clean
cross-sweep comparison.

**Date**: 2026-04-28  
**Parent commit**: `44ae99d`  
**Asana**: 1214132170406525  
**Plan**: [plan.md](plan.md)

---

## Headline

**The k=16 sweep does not extend the "random augmentation wins" story.**
At k=16 the verdict flips: \texttt{oap} wins at every hidden dim,
reversing the k=8 finding. Read together, the three sweeps show a
non-monotonic dependence on k:

| k    | h=16 winner       | h=128 winner      | h=512 winner      |
|-----:|-------------------|-------------------|-------------------|
|    3 | random_augmented  | random_augmented  | **oap**           |
|    8 | random_augmented  | random_augmented  | random_augmented  |
|   16 | **oap**           | **oap**           | **oap**           |

So the cross-k pattern is: canon wins narrowly at k=3 (h=512 only) →
random_augmented sweeps at k=8 → canon (specifically oap) sweeps at
k=16. The "use random_augmented" recommendation from the k=8 sweep
should be qualified by k.

## Results

| canonicalization   |   h  | params      | AP (single)         | AP (aug-avg K=16)  |   Δ aug   |
|--------------------|-----:|------------:|--------------------:|-------------------:|----------:|
| map                |   16 |      5,952  | 0.0610 ± 0.0009     |        —           |    —      |
| **oap**            |   16 |      5,952  | **0.0660 ± 0.0054** |        —           |    —      |
| random_augmented   |   16 |      5,952  | 0.0612 ± 0.0023     |  0.0614 ± 0.0023   |  +0.0002  |
| map                |  128 |    204,416  | 0.1779 ± 0.0065     |        —           |    —      |
| **oap**            |  128 |    204,416  | **0.1834 ± 0.0036** |        —           |    —      |
| random_augmented   |  128 |    204,416  | 0.1740 ± 0.0100     |  0.1748 ± 0.0097   |  +0.0008  |
| map                |  512 |  2,979,968  | 0.2381 ± 0.0147     |        —           |    —      |
| **oap**            |  512 |  2,979,968  | **0.2397 ± 0.0061** |        —           |    —      |
| random_augmented   |  512 |  2,979,968  | 0.2353 ± 0.0083     |  0.2359 ± 0.0077   |  +0.0006  |

Bold = best mean AP per hidden dim. \texttt{oap} wins at all three
hidden dims by margins of \mbox{$+0.0048$} (h=16), \mbox{$+0.0094$}
(h=128), and \mbox{$+0.0044$} (h=512) over \texttt{random\_augmented}'s
single-pass score. The aug-averaged \texttt{random\_augmented} is at
most $+0.0008$ AP above the single-pass version, far short of closing
any gap.

## Cross-sweep comparison

| h   |     k=3 (winner) |     k=8 (winner)  |    k=16 (winner)  |
|----:|------------------|-------------------|-------------------|
|  16 | rand_aug 0.0662  | rand_aug 0.0636   | **oap 0.0660**    |
| 128 | rand_aug 0.1762  | rand_aug 0.1766   | **oap 0.1834**    |
| 512 | **oap 0.2431**   | rand_aug 0.2395   | **oap 0.2397**    |

Two surprises:
1. At k=16, **\texttt{oap} h=128 beats every other test cell in the entire
   three-sweep grid for canonicalization** (0.1834 AP) — and even beats
   the corresponding random_augmented at the same setting by $+0.0094$.
2. At k=16, **\texttt{random\_augmented} h=512 (0.2353) is *lower* than at
   k=8 (0.2395)**. Adding more eigenvectors hurt the augmented model. This
   suggests the augmentation-trained network may be hitting an
   invariance-learning bottleneck: with $|\mathbb{Z}_2^{16}|=65{,}536$
   sign patterns to be invariant to under 200-epoch training, a 3M-param
   GIN may underfit the invariance.

## Aug-Δ trend

| k   | sign-group size  | Δ aug at h=512    |
|----:|-----------------:|------------------:|
|   3 | 8                | $+0.0002$         |
|   8 | 256              | $+0.0005$         |
|  16 | 65,536           | $+0.0006$         |

Aug-Δ does grow monotonically with $k$, but very slowly: from $+0.0002$
to $+0.0006$ AP over a $\sim$8000$\times$ growth in group size. Test-time
averaging with K=16 cannot rescue \texttt{random\_augmented} from
\texttt{oap} at k=16; its biggest impact is still in the noise floor.

## Interpretation

### 1. The "random_augmented wins" message from the k=8 sweep is k-specific

The k=8 sweep concluded that random augmentation dominates at every
hidden dim. The k=16 result complicates this:

- **k=8 was a sweet spot.** PE width is rich enough that augmentation
  can exploit it, while the ambiguity group (256 sign patterns) is
  small enough that 200-epoch training fully internalises invariance.
- **k=16 overshoots.** PE width grew but so did the invariance budget the
  network has to absorb (256× more sign patterns). At k=16, the
  augmentation signal is *too varied* — the network has only 200 epochs
  to learn invariance over 65k sign patterns vs. only 256 at k=8.
- **\texttt{oap} doesn't have this problem** — it presents a single
  deterministic ordering, so the model receives the full information
  content of the wider PE without paying an invariance cost.

### 2. Test-time averaging cannot rescue an under-trained augmented model

Aug-Δ at k=16 is $+0.0006$ AP at h=512 — same order of magnitude as at
k=8. If the random_augmented model under-fit invariance at training,
averaging over K=16 random draws at test time isn't enough samples (or
isn't the right operation) to compensate. To recover the k=8 win at k=16
would likely require more training epochs or larger K (potentially in the
hundreds), neither of which is free.

### 3. What this means for the overall recommendation

The recommendation needs to be qualified by k:

- **k=8 (or thereabouts)**: train with \texttt{random\_augmented}; this
  beats canonicalization at every $h$.
- **k=16+**: prefer \texttt{oap}; the larger ambiguity group makes
  random-augmentation training-time invariance harder to learn at fixed
  epoch budgets.
- **k=3**: \texttt{random\_augmented} is fine at small/medium $h$;
  \texttt{oap} edges it out at $h\!=\!512$ (the k=8 sweep result was
  what made us doubt this; the k=16 sweep confirms canonicalization can
  win at large model size when k is far from a sweet spot).

The earlier "use random_augmented across the board" recommendation
(commit `53d1f07`) holds at k=8, but is too strong as a general
prescription. The strongest defensible claim across all 81 runs is now:
**\texttt{random\_augmented} is the right choice at $k\!=\!8$; outside
that, \texttt{oap} is at least as competitive and often better.**

## Caveats

- 3 seeds per cell. Seed std for \texttt{oap} h=16 is 0.0054 — wide
  enough that the +0.0048 lead over random_augmented is approximately
  $1\sigma$, not significant on its own. The h=128 win (+0.0094 vs rand
  std 0.0100) is closer to $1\sigma$ as well.
- 200-epoch budget. The k=16 hypothesis above (random_augmented
  under-fitted invariance) implies that a larger budget could change the
  verdict. Not tested here.
- Same scope as before — only \texttt{map}, \texttt{oap},
  \texttt{random\_augmented}.

## Reproducing

```bash
python scripts/ablation_nadav_improvements.py \
    --canonicalization random_augmented map oap \
    --hidden-dim 16 128 512 \
    --k 16 \
    --cache-n-eigs 16 \
    --seeds 0 1 2 \
    --epochs 200 \
    --patience 15 \
    --no-eigval-scale \
    --test-aug-samples 16 \
    --base-dir results/test_time_averaging_augmented_k16 \
    --gpus 0 1
```

The new `--cache-n-eigs` flag (added in commit `44ae99d`) overrides the
previously-hardcoded `CACHE_K=15`, which would otherwise silently
truncate the PE to 15 columns at runtime even with `--n-eigs 16`.

## Directory layout

```
results/test_time_averaging_augmented_k16/
├── README.md               ← this file
├── plan.md                 ← pre-implementation plan
├── gin/                    ← 27 leaf dirs
├── summary.csv
└── plots/
    ├── ap_vs_hdim.png
    └── ap_vs_hdim_augmented.png
```
