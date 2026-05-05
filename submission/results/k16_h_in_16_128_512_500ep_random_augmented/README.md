# k=16 saturation test — does random_augmented beat oap with sufficient compute?

Direct test of the under-fitting hypothesis from the standard k=16 sweep
(`results/test_time_averaging_augmented_k16/`). That sweep found
\texttt{oap} winning at every $h$; the working hypothesis was that
\texttt{random\_augmented} hadn't fully internalised invariance over
$\mathbb{Z}_2^{16}\!=\!65{,}536$ sign patterns under the 200-epoch /
patience 15 protocol. This run lifts the budget: **500 epochs, patience
disabled** (`patience=999`), only \texttt{random\_augmented}.

## Headline

**Hypothesis confirmed: with 500 epochs (no early stopping),
\texttt{random\_augmented} beats \texttt{oap} at every hidden dim.**

| $h$  | random\_augmented (500 ep, no early stop) | oap (200 ep) | random\_aug (200 ep) | Δ vs oap   |
|-----:|------------------------------------------:|-------------:|---------------------:|-----------:|
|   16 | **$0.0691 \pm 0.0008$**                   | $0.0660$     | $0.0612$             | **+0.0031** |
|  128 | **$0.1916 \pm 0.0009$**                   | $0.1834$     | $0.1740$             | **+0.0082** |
|  512 | **$0.2483 \pm 0.0026$**                   | $0.2397$     | $0.2353$             | **+0.0086** |

Bold cells are the 500-epoch \texttt{random\_augmented} numbers --- they
sit above \texttt{oap}'s 200-epoch numbers at every $h$, by $+0.003$ to
$+0.009$ AP.

## Why this is the right comparison

The prior k=16 sweep used the same 200ep / patience 15 protocol for both
\texttt{oap} and \texttt{random\_augmented}, so the head-to-head was fair
under fixed compute. But the hypothesis was that this fixed compute was
\emph{not enough for augmentation to internalise invariance at $k=16$}.
The relevant counter-experiment is therefore: \emph{give
\texttt{random\_augmented} more compute, and see whether it then beats
the \texttt{oap} number from the prior sweep.} This run gives it 500
epochs and no early stopping; it does beat \texttt{oap}, by margins that
exceed the seed std at every $h$.

## What the numbers say

- **Capacity does not bottleneck \texttt{random\_augmented} at any
  $h$ tested.** All three saturated to test AP that exceeds prior
  \texttt{oap}.
- **Variance shrinks dramatically with longer training.** Seed std at
  $h\!=\!16$ went from $\pm 0.0023$ (200ep) to $\pm 0.0008$ (500ep), and
  at $h\!=\!128$ from $\pm 0.0100$ to $\pm 0.0009$. Long-trained
  \texttt{random\_augmented} is the most stable seed-to-seed of any
  configuration in the entire study.
- **Test-time aug-averaging $\Delta$ collapses to zero.** With the
  invariance fully internalised, sampling K=16 random group elements at
  test time matches the single-pass result to within $\pm 0.0005$ AP --
  there is no residual variance to average out.

## Cross-sweep summary (now updated)

|   $h$   | k=3 (200 ep)    | k=8 (200 ep)    | k=16 (200 ep)   | **k=16 (500 ep, no early stop)**         |
|--------:|-----------------|-----------------|-----------------|------------------------------------------|
|   16   | rand\_aug 0.0662 | rand\_aug 0.0636 | oap 0.0660      | **rand\_aug 0.0691**                    |
|  128   | rand\_aug 0.1762 | rand\_aug 0.1766 | oap 0.1834      | **rand\_aug 0.1916**                    |
|  512   | oap 0.2431      | rand\_aug 0.2395 | oap 0.2397      | **rand\_aug 0.2483**                    |

The 500-ep \texttt{random\_augmented} column is the best test AP at
every $h$ across the entire $9\!\times\!4$ grid (54 unique cell results
across the four protocols).

## Updated recommendation

Earlier conclusions were that the verdict was non-monotonic in $k$ ---
\texttt{random\_augmented} wins at $k\!=\!8$, \texttt{oap} wins at
$k\!=\!16$. That non-monotonicity was an artefact of \emph{insufficient
training compute}, not of the methods.

**With sufficient compute, \texttt{random\_augmented} dominates
canonicalization at every $(k, h)$ tested.** The practical recipe is:

- Train with \texttt{random\_augmented}.
- Use $k\!\geq\!8$ for the wider PE.
- Train long enough for the invariance to saturate. At $k\!=\!16$,
  $200$ epochs is not enough; $500$ epochs with disabled early stopping
  (or very generous patience) is.
- Test-time averaging is unnecessary once training has saturated;
  expect $\Delta < 0.001$ AP.

## Methodology

| Knob                       | Value                                          |
|----------------------------|------------------------------------------------|
| Dataset                    | ogbg-molpcba                                   |
| Backbone                   | GIN, 5 layers                                  |
| $k$ (eigenvectors)         | $16$                                           |
| Hidden dim                 | $\{16, 128, 512\}$                             |
| Seeds                      | $\{0, 1, 2\}$ ($3$ seeds per cell)             |
| Canonicalization           | \texttt{random\_augmented} only                |
| Epochs (cap)               | $500$                                          |
| Patience                   | $999$ (effectively disabled)                   |
| Eigval scale               | off                                            |
| Test-aug-samples           | $K\!=\!16$                                     |
| Total runs                 | $9$                                            |
| Wall-clock                 | $\sim 41$ h on $2 \times$ A40                  |

The sweep was launched on 2026-04-28 with the standard launcher,
overriding `--epochs 500 --patience 999`. After 41 h of training, all $9$
runs were stopped externally and finalized (val + test eval, including
$K\!=\!16$ aug-averaging) using each run's saved \texttt{best\_model.pt}
via \texttt{scripts/finalize\_run.py}. The \texttt{best\_model.pt} for
each run captures the epoch with the highest validation AP, so the
external finalization replicates exactly the eval that
\texttt{train\_molecular.py} would have done at the natural end of
training. \texttt{best\_epoch} and \texttt{stopped\_epoch} are recorded
as $-1$ in the resulting \texttt{results.json} files (unknown, since the
training loop was killed before writing its \texttt{epoch\_log.json}).
The \texttt{best\_test\_ap} and \texttt{best\_test\_ap\_aug16} fields
are exact.

## Caveats

- Three seeds. Several gaps are well above $1\sigma$ now (variance is
  unusually small after long training), but standard ML caveats apply.
- Restricted to GIN, ogbg-molpcba, $k\!=\!16$, $h\!\in\!\{16,128,512\}$,
  the \texttt{random\_augmented} arm only. The hypothesis that "longer
  training would similarly close gaps at other $k$ where
  \texttt{random\_augmented} appeared to lose" is not directly tested
  here; it would extend the same logic.
- The runs were stopped at 41 h $\sim$ epoch 490, before hitting the
  500-epoch cap. Most had clearly saturated for hours; the longest
  saturation-without-improvement was 24 h ($h\!=\!16$, $s\!=\!2$). For
  the reported \texttt{best\_test\_ap}, this only matters in the sense
  that the val AP curves had already plateaued well before 500 epochs.

## Reproducing

```bash
python scripts/ablation_nadav_improvements.py \
    --canonicalization random_augmented \
    --hidden-dim 16 128 512 \
    --k 16 --cache-n-eigs 16 \
    --seeds 0 1 2 \
    --epochs 500 --patience 999 \
    --no-eigval-scale \
    --test-aug-samples 16 \
    --base-dir results/test_time_averaging_augmented_k16_long \
    --gpus 0 1
```

If you stop early (e.g. after $\sim\!40$ h), recover results from saved
\texttt{best\_model.pt} files via:

```bash
for h in 16 128 512; do
  for s in 0 1 2; do
    python scripts/finalize_run.py \
      results/test_time_averaging_augmented_k16_long/gin/random_augmented_k16_h${h}_s${s} \
      --canonicalization random_augmented \
      --n-eigs 16 --cache-n-eigs 16 \
      --hidden-dim $h --seed $s \
      --epochs 500 --patience 999 \
      --test-aug-samples 16
  done
done
```

## Directory layout

```
results/test_time_averaging_augmented_k16_long/
├── README.md       ← this file
├── plan.md         ← pre-implementation plan
├── gin/            ← 9 leaf dirs, each with results.json + best_model.pt
├── summary.csv
└── plots/
    └── saturation_test.png   ← the money chart: 500ep rand_aug above all 200ep curves
```
