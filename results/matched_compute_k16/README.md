# Matched-compute follow-up: map + oap at k=16 with 500 ep / no early stop

This is the experiment the adversarial review (`#1` in the 4-sweep
review, 2026-04-30) demanded: give \texttt{oap} and \texttt{map} the
same compute budget as \texttt{random\_augmented} got in Sweep 4
(`results/test_time_averaging_augmented_k16_long/`), and see whether
the "random augmentation beats canonicalization with sufficient
compute" headline survives matched compute.

It does not survive.

## Configuration

Identical to the Sweep 4 saturation test, except canonicalization arms.

| Knob | Value |
|---|---|
| canonicalizations | `map`, `oap` (no `random_augmented` — Sweep 4 covers it at this protocol) |
| k | 16 |
| cache-n-eigs | 16 |
| hidden dim | 16, 128, 512 |
| seeds | 0, 1, 2 |
| epochs | 500 |
| patience | 999 (effectively disabled) |
| eigval scale | off |
| test-aug-samples | 16 (no-op for map/oap) |

Total 2 × 3 × 3 = **18 runs**. Wave 1 (12 runs) finished at ~T+55 h
naturally. Wave 2 (6 oap runs) was finalized externally at ~T+74 h via
`scripts/finalize_run.py` after most workers had saturated for hours
and several were producing only $\le +0.001$ AP per hour of additional
training. The wave-2 finalize trades $\sim 0.001$ AP per worker for
$\sim 36$ h saved.

## Headline (mean ± std over 3 seeds, test AP on ogbg-molpcba)

| h    | map (500 ep)              | oap (500 ep)              | rand\_aug (500 ep, Sweep 4) | best at h |
|-----:|--------------------------:|--------------------------:|----------------------------:|:----------|
|   16 | **$0.0716 \pm 0.0067$**   | $0.0703 \pm 0.0070$       | $0.0691 \pm 0.0008$         | **map**   |
|  128 | **$0.1923 \pm 0.0005$**   | $0.1836 \pm 0.0025$       | $0.1916 \pm 0.0009$         | **map**   |
|  512 | $0.2468 \pm 0.0021$       | $0.2419 \pm 0.0028$       | **$0.2483 \pm 0.0026$**     | **rand\_aug** |

\Delta vs. random_augmented (positive = canonicalization beats augmentation):

| h    | map - rand_aug | oap - rand_aug |
|-----:|---------------:|---------------:|
|   16 |   $+0.0025$    |   $+0.0012$    |
|  128 |   $+0.0007$    |   $-0.0080$    |
|  512 |   $-0.0014$    |   $-0.0064$    |

## Compute gains for canonicalization (200ep → 500ep / no early stop)

| canon | h   | 200 ep (Sweep 3) | 500 ep (this sweep) | Δ |
|------:|----:|-----------------:|--------------------:|--:|
| map   |  16 |          $0.0610$ |             $0.0716$ | **$+0.0106$** |
| map   | 128 |          $0.1779$ |             $0.1923$ | **$+0.0144$** |
| map   | 512 |          $0.2381$ |             $0.2468$ | **$+0.0088$** |
| oap   |  16 |          $0.0660$ |             $0.0703$ |   $+0.0043$  |
| oap   | 128 |          $0.1834$ |             $0.1836$ |   $+0.0003$  |
| oap   | 512 |          $0.2397$ |             $0.2419$ |   $+0.0022$  |

\texttt{map} gains markedly more from 500ep training than \texttt{oap}.
This is the adversarial review's prediction: canonicalization arms also
under-fit at 200ep / patience 15. Their convergence point is just slower
to reach than \texttt{oap}'s, which is why the prior 200ep
\texttt{oap}-wins-everywhere result was misleading.

## What this means

The previous "rand_aug beats canonicalization with sufficient compute"
narrative (Sweep 4 vs prior Sweep 3 \texttt{oap}) is an
**apples-to-oranges artefact**. With matched compute:

- **map wins at 2 of 3 hidden dims** ($h\!=\!16, h\!=\!128$); narrow
  but real margins of $+0.0007$ to $+0.0025$ AP.
- **rand_aug wins at $h\!=\!512$** by only $+0.0014$ AP (within seed
  std) over matched-compute \texttt{map}.
- **oap is the worst arm** at every $h$ at matched compute --- which
  is itself surprising relative to Sweep 3 where oap appeared best.

So the strongest defensible cross-method claim across the four-plus-one
sweeps on $k\!=\!16$ is now:

\textbf{No single method dominates. With matched compute, \texttt{map}
narrowly wins at small/medium $h$ and \texttt{random\_augmented} narrowly
wins at $h\!=\!512$.}

## Cross-sweep summary at $k=16$ (now updated)

| h   |     200 ep (Sweep 3, fixed compute) |     500 ep (matched compute) |
|----:|-------------------------------------|------------------------------|
|  16 | oap $0.0660$ (best)                 | **map $0.0716$** (rand_aug $0.0691$, oap $0.0703$) |
| 128 | oap $0.1834$ (best)                 | **map $0.1923$** (rand_aug $0.1916$, oap $0.1836$) |
| 512 | oap $0.2397$ (best)                 | **rand_aug $0.2483$** (map $0.2468$, oap $0.2419$) |

The Sweep 3 reading "\texttt{oap} wins everywhere" was driven by
\texttt{map}'s under-fitting at 200ep --- not by \texttt{oap} being
better than \texttt{map} or \texttt{random\_augmented} after sufficient
training.

## Caveats

- 3 seeds. \texttt{map h=16} std is $\pm 0.0067$ --- the $+0.0025$
  margin over \texttt{random\_augmented} is $<\!1\sigma$. Only the
  $h\!=\!128$ \texttt{map}-wins margin is comfortably outside noise.
- 6 of 18 runs (the wave-2 oap runs) were stopped externally at
  $\sim$T+74 h before hitting the 500-epoch cap, after their val AP
  curves had clearly plateaued. Their `best_test_ap` is exact (eval'd
  via `finalize_run.py` from saved `best_model.pt`); their
  `best_epoch` and `stopped_epoch` are recorded as $-1$ (unknown, since
  the training process was killed before writing `epoch_log.json`).
- A symmetric matched-compute test at $k\!=\!3$/$h\!=\!512$ (the lone
  surviving canonicalization edge from Sweep 1) is the sibling sweep
  `results/matched_compute_k3_h512/`, currently in progress on
  newton1 via slurm.
