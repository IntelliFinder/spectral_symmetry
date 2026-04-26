# Plan: k=16 sweep — third pass on the random-augmentation vs canonicalization story

## Context

Third sweep in the `test_time_averaging_augmented*` series. First was
k=3/K=8 (oap won at h=512), second was k=8/K=16 (random_augmented won
everywhere). This one bumps to k=16 to test whether the trend stabilises:

- Does random_augmented continue to dominate at k=16, h∈{16,128,512}?
- Does the test-time aug-averaging Δ stay negligible (≤+0.0005 AP from
  prior sweeps), or does the much larger ambiguity group (Z₂¹⁶ = 65,536
  vs. Z₂⁸ = 256 at k=8) start to matter?

## Hyperparameter choices — logic

| Knob | Value | Why |
|---|---|---|
| n-eigs (k) | **16** | Doubles k=8. Tests whether the random_augmented win is monotonic in k. |
| cache-n-eigs | **16** | Existing caches are at k=15. One-time rebuild of `ogbg-molpcba_{random_augmented,map,oap,raw}_k16` per the launcher's `prewarm_caches` step (~30–60 min). |
| test-aug-samples (K) | **16** | Same as k=8 sweep so the comparison is apples-to-apples. The ambiguity group at k=16 is 256× larger than at k=8, but we already established that aug-Δ is dominated by the model's training-time invariance (~+0.0001 to +0.0005 AP), not by K. K=16 is the minimum that lets us say "averaging didn't flip the verdict" with the same confidence as the k=8 sweep. |
| canonicalizations | random_augmented, map, oap | Same scope as the prior two sweeps. |
| hidden-dim | 16, 128, 512 | Same as before. |
| seeds | 0, 1, 2 | Same as before. |
| epochs / patience | 200 / 15 | Same protocol — most runs early-stop in the 40–80 epoch range. |
| eigval-scale | off | Same as before. |
| GPUs | 0, 1 | Same physical setup (2× A40). |
| base-dir | `results/test_time_averaging_augmented_k16` | Self-contained directory. |

Total runs: **3 canons × 3 hdims × 3 seeds = 27 runs**. Wall clock estimate
~30–40 h based on the k=8 sweep (~32 h). One extra hour for the k=16
cache rebuild.

## What changes from the k=8 plan

Only `--n-eigs 8 → 16`, `--cache-n-eigs 15 → 16`, and
`--base-dir → results/test_time_averaging_augmented_k16`. No code changes.
`random_augment_eigenvectors` already handles arbitrary k.

## Launch command

```bash
python scripts/ablation_nadav_improvements.py \
    --canonicalization random_augmented map oap \
    --hidden-dim 16 128 512 \
    --k 16 \
    --seeds 0 1 2 \
    --epochs 200 \
    --patience 15 \
    --no-eigval-scale \
    --test-aug-samples 16 \
    --base-dir results/test_time_averaging_augmented_k16 \
    --gpus 0 1
```

## Expected outcomes

Following the prior two sweeps, the most likely result is that
random_augmented wins at every hidden dim again, with aug-Δ remaining
≤+0.001 AP. The interesting alternative scenarios:

- map/oap recover at h=512 — would suggest k=8 was a localised flip,
  and the augmentation-vs-canonicalization picture is k-dependent.
- aug-Δ grows noticeably (≥+0.001 AP) — would suggest the larger
  ambiguity group does start to give test-time averaging more room to
  improve, despite training-time invariance.

Either way, this sweep extends the comparison to a third k value and
either strengthens or qualifies the "random augmentation is the better
default" message.

## Output

Same layout as the two prior sweeps:

```
results/test_time_averaging_augmented_k16/
├── README.md                  ← post-sweep writeup, same template
├── plan.md                    ← this file
├── gin/
│   ├── random_augmented_k16_h16_s0/results.json
│   └── …                                              (27 leaf dirs)
├── summary.csv
└── plots/
    ├── ap_vs_hdim.png
    └── ap_vs_hdim_augmented.png
```

Then a final addendum to the explainer PDF (`latex/report.pdf` in the
k=8 directory) — or an updated standalone PDF if the k=16 result is
qualitatively different from k=8 — to keep the cross-sweep narrative
in sync.
