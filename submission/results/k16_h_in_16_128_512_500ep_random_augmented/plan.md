# Plan: k=16 saturation test — random_augmented with no early stopping, 500 epochs

## Hypothesis being tested

The k=16 sweep (`results/test_time_averaging_augmented_k16/`, commit
`a922f59`) found that `random_augmented` lost to `oap` at every hidden
dim under the standard 200 epoch / patience 15 protocol. The README's
working hypothesis was:

> With $\mathbb{Z}_2^{16}=65{,}536$ sign patterns to be invariant to and
> only ~80 effective epochs (early stop), random_augmented under-fits
> invariance.

This sweep tests that hypothesis directly: **same k=16, same h grid, but
with the early-stopping budget lifted and many more epochs**. If the
hypothesis is correct, random_augmented should now beat the prior
(200-epoch) oap numbers; if it's wrong, val/test AP plateaus at the same
~80-epoch mark and oap genuinely beats augmentation at k=16.

## Targets to beat (from prior k=16 sweep)

| h    | oap test AP @ 200ep | random_augmented test AP @ 200ep |  gap to close  |
|-----:|--------------------:|---------------------------------:|---------------:|
|   16 |       0.0660        |             0.0612               |     +0.0048    |
|  128 |       0.1834        |             0.1740               |     +0.0094    |
|  512 |       0.2397        |             0.2353               |     +0.0044    |

## Hyperparameter changes vs. prior k=16 sweep

| Knob                | k=16 sweep | k=16 saturation test                                       |
|---------------------|-----------:|-----------------------------------------------------------:|
| canonicalizations   | 3 (rand, map, oap) | **1 (random_augmented only)**                       |
| epochs (cap)        | 200        | **500** (2.5×)                                              |
| patience            | 15         | **999** (effectively disabled — `patience > epochs`)        |
| Everything else     | (unchanged) — k=16, h∈{16,128,512}, seeds {0,1,2}, batch 32, lr 1e-3, no eigval scale, test-aug-samples 16 |

Total: $3\,h \times 3\,\text{seeds} = 9$ runs.

## Wall-clock estimate

Per-epoch time on the prior runs is ~5 min, roughly h-independent (the
data-loader is the bottleneck). $500$ epochs × $5$ min = $\sim$$42$ h
per run. $9$ runs in parallel on $12$ GPU slots ($2$ A40s × $6$ workers)
= $\sim$$42$ h wall-clock total.

## Launch command

```bash
python scripts/run_lappe_sweep.py \
    --canonicalization random_augmented \
    --hidden-dim 16 128 512 \
    --k 16 \
    --cache-n-eigs 16 \
    --seeds 0 1 2 \
    --epochs 500 \
    --patience 999 \
    --no-eigval-scale \
    --test-aug-samples 16 \
    --base-dir results/test_time_averaging_augmented_k16_long \
    --gpus 0 1
```

## Decision criteria

After the sweep:

- **If random_augmented beats oap at all 3 h values**: the under-fitting
  hypothesis is confirmed. The k=16 verdict from the prior sweep was an
  artefact of insufficient training. The PDF recommendation should be
  revised to "use random_augmented with sufficient epochs at all k tested".
- **If random_augmented matches oap at some h but not others**: the
  picture is more nuanced; partial confirmation.
- **If random_augmented plateaus at ~80 epochs and final test AP barely
  moves**: hypothesis refuted. Canonicalization (oap) genuinely beats
  augmentation at k=16. The non-monotonic-in-k recommendation in the
  current PDF stands.

We will know within ~42 h.

## Output

```
results/test_time_averaging_augmented_k16_long/
├── README.md               ← post-sweep writeup
├── plan.md                 ← this file
├── gin/
│   └── random_augmented_k16_h{16,128,512}_s{0,1,2}/results.json   (9 dirs)
├── summary.csv
└── plots/
    ├── ap_vs_hdim.png
    └── val_curves.png      ← per-seed val AP traces, to verify saturation
```
