# Plan: k=8 extension of test-time averaging for random-augmented LapPE models

## Context

Companion to `results/test_time_averaging_augmented/` (k=3 sweep, finished
2026-04-23). Same hypothesis, same three canonicalizations, same model, same
protocol — just bumping the PE width from `k=3` to `k=8` and the test-time
averaging budget from `K=8` to `K=16`.

**Why k=8.** At k=3 the per-node PE has 3 dimensions — small enough that
most of the LapPE signal on molecules (typical mol graph has 15–60 nodes,
eigengap structure fills the first ~6 eigenvalues) is compressed. k=8
roughly doubles the information content of the PE while staying well below
the cache-n-eigs=15 cap, and matches the canonicalization paper
conventions for mol benchmarks.

**Why K=16.** With k=8 simple eigenvalues the sign group is Z₂⁸ = 256.
K=16 is 6.25% of the group — not exhaustive (unlike K=8 at k=3 which was
exact), but enough that the Monte-Carlo error on the averaged prediction
scales as 1/√K = ¼. Compute cost: each train_molecular run does one K=16
test-eval pass at the end — negligible next to the 200-epoch training
loop.

**Hypothesis restated for k=8.** Random-augmented + test-time K=16 averaging
beats map/oap on ogbg-molpcba. From the k=3 result we already know this
holds at h=16 / h=128 and flips to oap at h=512. Two things to watch at
k=8:

1. Does the extra PE capacity close the h=512 gap (e.g. more PE info → more
   information for random_augmented to average over → might beat oap at
   h=512)?
2. Does the larger ambiguity group (Z₂⁸ × O(m) vs Z₂³) make test-time
   averaging matter more? At k=3 the Δ was +0.0001 … +0.0002 AP — "safe
   but tiny"; at k=8 the Δ could grow because a single random draw
   samples a thinner slice of the group.

## Shape of the change

**No code changes.** The k=3 implementation already supports arbitrary k
and arbitrary O(m) multiplicities:

- `random_augment_eigenvectors(V, eigenvalues, rng)` in
  `src/spectral_canonicalization.py` handles any k via
  `detect_eigenvalue_multiplicities`.
- `MolecularLapPEDataset.get_augmented_test_loader` in
  `src/experiments/molecular/dataset.py` is k-agnostic.
- `evaluate_with_aug_averaging` and `--test-aug-samples` in
  `scripts/train_molecular.py` take K as an int.
- `scripts/ablation_nadav_improvements.py` already threads `--k` and
  `--test-aug-samples` through.

This plan is a pure configuration change: `--k 8 --test-aug-samples 16
--base-dir results/test_time_averaging_augmented_k8`.

## Hyperparameters

Identical to the k=3 sweep except `k` and the test-aug budget.

| Knob                       | Value                                             |
|----------------------------|---------------------------------------------------|
| dataset                    | `ogbg-molpcba`                                    |
| model                      | `gin`, num-layers = 5                             |
| batch-size                 | 32                                                |
| lr                         | 1e-3                                              |
| epochs                     | 200 (cap)                                         |
| patience (early stop)      | 15                                                |
| **n-eigs (k)**             | **8**                                             |
| cache-n-eigs               | 15 (unchanged; caches already exist)              |
| hidden-dim                 | {16, 128, 512}                                    |
| seeds                      | {0, 1, 2}                                         |
| eigval-scale               | off (no scaling)                                  |
| canonicalizations          | `random_augmented`, `map`, `oap` (all retrained)  |
| **test-aug-samples (K)**   | **16** — only applied to random_augmented; silently ignored for map/oap |

Parameter counts (estimates, confirmed post-run via the `n_params` field
in each `results.json`):

| h   | params (k=3)  | params (k=8, +5×h) | Δ vs k=3 |
|----:|--------------:|-------------------:|---------:|
|  16 |        5,744  |            ~5,824  |     +80  |
| 128 |      202,752  |          ~203,392  |    +640  |
| 512 |    2,973,312  |        ~2,975,872  |  +2,560  |

Total runs: 3 canons × 3 hdims × 3 seeds = **27 runs**, same as k=3.
Expected wall clock similar too — roughly ~1.5–2 days on GPUs 0, 1.

## Launch command

```bash
python scripts/ablation_nadav_improvements.py \
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

Monitoring: same 1 h heartbeat loop as before.

## Cache path sanity check

- `data/lappe_cache/ogbg-molpcba_raw_k15/lappe.pkl` — raw eigendecomp,
  shared.
- `data/lappe_cache/ogbg-molpcba_{random_augmented,map,oap}_k15/lappe.pkl`
  — confirmed present (2026-04-23 `ls`). The loader at
  `dataset.py:_get_by_graph_idx` slices `pe[:, :k]` and `evals[:k]`, so
  k=8 reads the same cache; no new cache-build pass needed.

## Result directory layout

```
results/test_time_averaging_augmented_k8/
├── README.md              ← written post-sweep, following the style memory
├── plan.md                ← this file
├── gin/
│   ├── random_augmented_k8_h16_s0/results.json
│   ├── ...                                       (27 leaf dirs)
│   └── oap_k8_h512_s2/results.json
├── summary.csv            ← one row per run
└── plots/
    ├── ap_vs_hdim.png
    └── ap_vs_hdim_augmented.png
```

Writeup structure (per the feedback memory
`feedback_results_presentation.md`): aligned table with bolded winners,
explicit hypothesis framing, per-row interpretation (esp. any h where the
k=3 vs k=8 verdict flips), caveats, actionable takeaway.

## Verification

1. `pytest tests/test_unified_canonicalization.py::TestRandomAugmentEigenvectors -v`
   — already green from the k=3 PR; re-run as sanity.
2. Smoke on moltox21 (3 epochs, k=8, h=32, K=16):

   ```
   python scripts/train_molecular.py --dataset ogbg-moltox21 \
       --canonicalization random_augmented \
       --n-eigs 8 --cache-n-eigs 15 \
       --epochs 3 --hidden-dim 32 --seed 0 \
       --test-aug-samples 16 --save-dir /tmp/aug_smoke_k8
   ```

   Expect both `best_test_rocauc` and `best_test_rocauc_aug16` in
   `/tmp/aug_smoke_k8/results.json`. Re-run with same seed ⇒ identical
   aug-16 number (rng is seeded on `base_seed*1_000_003 + s`).
3. Full 27-run sweep. Expect `best_test_ap_aug16 ≥ best_test_ap` for every
   random_augmented row (aug-averaging should never hurt a converged
   model). Any row where it does is a training-instability signal and gets
   flagged in the writeup.
4. Commit + push results (README, summary.csv, gin/*, plots/*). Update
   Asana task 1214132170406525 with the results table and mark the new
   sub-item complete (or create a sibling task — check with the user if
   unclear).

## Directionally expected outcomes

Not predictions — things to look for when the results arrive, so the
interpretation section isn't written from scratch:

- **If oap still wins at h=512** (like k=3): the capacity-dependent
  ordering is robust to PE width. Canonicalization wins when the model
  has enough capacity; augmentation wins when it doesn't.
- **If random_augmented(aug16) beats oap at h=512**: the k=3 result was
  bottlenecked by PE width, not by augmentation. K=16 sampling over Z₂⁸
  reveals the real advantage of data augmentation.
- **If Δ aug grows noticeably** (say ≥ +0.001 AP) from the +0.0002
  observed at k=3: the ambiguity-group-size story matters — larger k
  means more diverse augmented predictions to average over.
- **If Δ aug stays tiny**: the model still becomes nearly
  ambiguity-invariant during training and averaging mainly polishes the
  residual.

## Out of scope

- Other canonicalizations (spielman, spielman_partition, maxabs, abs,
  none). Keeping apples-to-apples with the k=3 sweep.
- Other datasets, other models, other architectures.
- Other K values. If the k=8 run shows K=16 is bottlenecked
  (Δ still growing with K), a follow-up with K ∈ {32, 64} is trivial —
  same trained models, re-run only the final `evaluate_with_aug_averaging`
  call.

## Not doing

- Not rebuilding caches (they exist at k=15).
- Not adding new tests (the code is already covered by the k=3 PR tests).
- Not changing `_get_by_graph_idx`, the launcher, or the training loop.
- Not updating the old `results/test_time_averaging_augmented/` tree —
  the k=8 sweep gets its own directory.
