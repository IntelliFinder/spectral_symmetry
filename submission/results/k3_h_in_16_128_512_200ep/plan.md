# Plan: test-time averaging for random-augmented LapPE models

## Context

The weighted-frames repo (jwsiegel2510/Sn-invariant-weighted-frames) evaluates
models trained with Sn-permutation augmentation by **averaging predictions
across many random draws of the symmetry group at test time** (see the user's
`test_with_randomized_invariance` helper). That repo's symmetry is Sn
(point-cloud permutations); our symmetry is different — the Laplacian
eigenvector ambiguity group is

- a sign flip (±1) per simple eigenvalue, and
- a full orthogonal group O(m) per eigenvalue of multiplicity m.

Our `canonicalization="random_augmented"` models currently (a) only see sign
flips during training — never O(m) rotations — and (b) are evaluated once at
test time on the cached raw eigenvectors (`augment=False` in
`MolecularLapPEDataset._get_by_graph_idx`, L468). That mismatches how the
model was trained and wastes the variance-reduction that test-time averaging
buys. The hypothesis we want to validate: a random-augmented model evaluated
with K-fold averaging over (sign × O(m)) draws beats the canonicalization
baselines (`map`, `oap`). Scope is only the `random_augmented` training path;
`map`, `oap`, and `none` models keep their current single-pass evaluation —
Spielman and MaxAbs are intentionally excluded from the comparison.

## Shape of the change

```
                                ┌────────────────────────────────────────┐
                                │ BEFORE (train_molecular.py, today)     │
                                │                                        │
   cache(pe, evals) ──► _get_by_graph_idx(augment=split=="train")        │
                                │         │                              │
                                │         ▼  sign flips only             │
                                │   model.eval() once  ──► rocauc/ap     │
                                └────────────────────────────────────────┘

                                ┌────────────────────────────────────────┐
                                │ AFTER                                  │
                                │                                        │
   cache(pe, evals) ──► _get_by_graph_idx(augment=..., rng=seeded)      │
                             │                                          │
                             ▼   random_augment_eigenvectors(pe,evals,rng)│
                     (sign flip per mult-1 col; Haar O(m) per mult-m blk)│
                             │                                          │
   train epoch  ──► uses it with rng=None (fresh signs/rotations each   │
                             │        getitem, same as today's behaviour)│
   test (canon ≠ random_augmented) ──► single pass, no change           │
   test (canon == random_augmented) ──► for s in range(K=8):            │
                                          seed rng with base_seed+s     │
                                          collect sigmoid(logits)       │
                                        mean ──► OGB evaluator          │
                                        saved as best_test_<metric>_aug8│
                                └────────────────────────────────────────┘
```

## Files to change

### 1. `src/spectral_canonicalization.py`
- Add alongside `canonicalize_random_augmented` (L1058):
  ```python
  def random_augment_eigenvectors(eigenvectors, eigenvalues=None, rng=None):
      """Sample a random element of the eigenvector ambiguity group and apply
      it: ±1 per simple eigenvalue, Haar-distributed O(m) per multiplicity-m
      block. If ``eigenvalues`` is None, falls back to sign flips only.
      ``rng`` is a ``np.random.Generator``; None ⇒ legacy ``np.random``."""
  ```
  Implementation:
  - If `eigenvalues is None` or all multiplicities are 1, reuse the sign-flip
    loop from `canonicalize_random_augmented`.
  - Otherwise use `detect_eigenvalue_multiplicities` (`src/spectral_core.py`
    L287) to get `group_indices`.
  - For each group with size m==1, flip sign with p=0.5.
  - For each group with size m>1, sample a Haar O(m) matrix Q by
    `A = rng.standard_normal((m, m)); Q, R = np.linalg.qr(A); Q *= np.sign(np.diag(R))`
    and set `eigvecs[:, group_cols] = eigvecs[:, group_cols] @ Q`.
- Make `canonicalize_random_augmented` a thin wrapper — one line calling
  `random_augment_eigenvectors(V, eigenvalues=None, rng=None)` (preserves the
  existing test `test_only_sign_changes` in
  `tests/test_unified_canonicalization.py` L114).

### 2. `src/experiments/molecular/dataset.py`
- Import `random_augment_eigenvectors`.
- Extend `_get_by_graph_idx(self, graph_idx, augment=True, rng=None)` (L449).
  Replace the sign-only branch at L467–470 with
  `pe = random_augment_eigenvectors(pe, eigenvalues=evals, rng=rng)` — passing
  `evals` enables O(m). In training `rng=None` keeps today's behaviour.
- Extend `_SplitView`:
  - Add ctor args `augment_override: bool | None = None` and
    `base_seed: int | None = None`.
  - In `__getitem__`: `augment = augment_override if not None else (split == "train")`;
    `rng = None if base_seed is None else np.random.default_rng((base_seed << 32) ^ graph_idx)`.
- Add `MolecularLapPEDataset.get_augmented_test_loader(self, split, batch_size,
  base_seed, num_workers=0, **kw)` that wraps a `_SplitView` with
  `augment_override=True, base_seed=base_seed` in a non-shuffling `DataLoader`.
  Mirrors `get_dataloader` at L421.

### 3. `scripts/train_molecular.py`
- New CLI flag (near L199): `--test-aug-samples K` (int, default 0 ⇒ disabled;
  accepted range [0, 64]). **K=8 is the default we will run** because with
  `n_eigs = 3` and all simple eigenvalues, the sign group is Z₂³ of size 8 —
  K=8 independent draws effectively samples the full group (empirically
  saturates — any multiplicity groups are rare on molecular Laplacians).
- After the final single-pass test eval (L364–372), add:
  ```python
  if args.canonicalization == "random_augmented" and args.test_aug_samples > 0:
      aug_metric, y_true, y_pred_avg, graph_indices = evaluate_with_aug_averaging(
          model, mol_dataset, evaluator, device,
          dataset_name=args.dataset, pe_type=args.pe_type,
          num_samples=args.test_aug_samples, base_seed=args.seed,
          batch_size=args.batch_size, num_workers=args.num_workers,
      )
      results[f"best_test_{metric_name}_aug{args.test_aug_samples}"] = aug_metric
  ```
- Implement `evaluate_with_aug_averaging` next to `evaluate` (~L89): for each
  `s in range(num_samples)` rebuild a loader via
  `mol_dataset.get_augmented_test_loader("test", batch_size, base_seed=base_seed*1_000_003+s)`;
  run the model; accumulate `torch.sigmoid(logits)` into a running mean
  indexed by `graph_idx`; feed
  `{"y_true": y_true, "y_pred": y_pred_avg}` to the OGB evaluator.
- Silently skip when `canonicalization != "random_augmented"` even if the flag
  is set (so launcher scripts can pass it unconditionally), matching the
  user's rule: only for augmented models.

### 4. `scripts/ablation_nadav_improvements.py`
- Add `--test-aug-samples` (default 0) to argparse; thread it into
  `build_commands` (L218) as `--test-aug-samples <N>`.
- For this plan we launch with K=8.

### 5. Early stopping (no code change, just use it)
- `train_molecular.py` already has early stopping via `--patience` (L193, L357).
  We'll set `--patience 15` against `--epochs 200` so most runs terminate well
  below 200 epochs — the 300-epoch / cap-100-epoch inconsistency in current
  results is resolved by standardising on these values across the rerun.

### 6. Tests
- `tests/test_unified_canonicalization.py`:
  - `test_sign_only_when_no_multiplicities` — distinct eigvals ⇒ columns
    differ from input only by sign.
  - `test_preserves_subspace_for_multiplicity_block` — synthetic mult-2 block,
    check `V_out @ V_out.T ≈ V_in @ V_in.T` (projection invariant ⇒ result
    lies in O(m) orbit).
  - `test_rng_is_reproducible` — same seed ⇒ same output; different seed ⇒
    different output.
- `tests/test_molecular.py`:
  - Smoke: `get_augmented_test_loader("test", base_seed=0)` vs `base_seed=1`
    yield different `x_pe` for at least one graph (only when
    `canonicalization="random_augmented"`).

## Critical files

- `src/spectral_canonicalization.py` (L1058–1073, L1159 dispatcher)
- `src/spectral_core.py` (reuse `detect_eigenvalue_multiplicities` L287)
- `src/experiments/molecular/dataset.py` (L135 `_SplitView`, L412–484)
- `scripts/train_molecular.py` (L89 evaluate, L199 args, L362–417 final)
- `scripts/ablation_nadav_improvements.py` (L218 build_commands)

## Hyperparameters for the Asana rerun

Reduced from the earlier `launch_300ep_ablation.sh` sweep per the user's
feedback: single `n_eigs`, no eigval scaling, early stopping, K=8 (full sign
group for k=3). Everything compared apples-to-apples is retrained under these
settings — **including `map` and `oap`** — because existing results were
capped at 100 epochs and are not directly comparable.

| Knob                | Value                                              |
|---------------------|----------------------------------------------------|
| dataset             | `ogbg-molpcba`                                     |
| model               | `gin`, num-layers = 5                              |
| batch-size          | 32                                                 |
| lr                  | 1e-3                                               |
| epochs              | 200 (cap)                                          |
| patience (early stop) | 15                                              |
| n-eigs (k)          | **3 only**                                         |
| cache-n-eigs        | 3                                                  |
| hidden-dim          | {16, 128, 512}                                     |
| seeds               | {0, 1, 2}                                          |
| eigval-scale        | **off** (no scaling)                               |
| canonicalizations   | `random_augmented`, `map`, `oap` (all retrained)   |
| **test-aug-samples (K)** | **8** — applied to `random_augmented` only; K=8 exhausts the sign group `Z_2^3` that dominates the ambiguity group at k=3 |

Total runs: 3 canons × 3 hdims × 3 seeds = **27 runs**. Wall-clock: similar to
the prior ~100-epoch runs because early stopping at patience=15 typically
fires in the 40–80 epoch range on molpcba with k=3.

### Result directory layout

```
results/test_time_averaging_augmented/
├── README.md
├── plan.md
├── gin/
│   ├── random_augmented_k3_h16_s0/results.json
│   ├── random_augmented_k3_h16_s1/results.json
│   ├── ...
│   ├── map_k3_h16_s0/results.json
│   ├── oap_k3_h16_s0/results.json
│   └── ...                           (27 leaf dirs)
├── summary.csv
└── plots/
    ├── ap_vs_hdim.png
    └── ap_vs_hdim_augmented.png
```

### Launch command

```bash
python scripts/ablation_nadav_improvements.py \
    --canonicalization random_augmented map oap \
    --hidden-dim 16 128 512 \
    --k 3 \
    --seeds 0 1 2 \
    --epochs 200 \
    --patience 15 \
    --no-eigval-scale \
    --test-aug-samples 8 \
    --base-dir results/test_time_averaging_augmented \
    --gpus 0 1
```

## Verification

1. `pytest tests/test_unified_canonicalization.py tests/test_molecular.py`
2. Smoke on moltox21 (3 epochs, h=32)
3. Full 27-run sweep
4. Asana task description replaced with README content

## Out of scope

- DeepSets / ModelNet paths
- `none`, `maxabs`, `spielman`, `spielman_partition` training paths
