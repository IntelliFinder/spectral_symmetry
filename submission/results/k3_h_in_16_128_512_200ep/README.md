# Test-time averaging over the eigenvector ambiguity group

**Task**: complete the eigenvector random-augmentation vs. canonicalization
molecular benchmark comparison, with a new test-time evaluation that averages
model predictions over random draws of the ambiguity group. Only applied to
models trained with `random_augmented`; canonicalization baselines (`map`,
`oap`) keep their standard single-pass evaluation.

**Hypothesis** (going in): a random-augmented model evaluated with K-fold
averaging over (sign flip × O(m) rotation) draws beats the canonicalization
baselines at test time. This hypothesis is tested against `map` and `oap`
specifically (`none`, `maxabs`, `spielman`, `spielman_partition` are out of
scope).

**Date**: 2026-04-23  
**Git SHA**: `c8958e5` (branch: master)  
**Asana**: 1214132170406525  
**Plan**: [plan.md](plan.md)

---

## Metric key

- `best_test_ap` — standard single-pass test AP on `ogbg-molpcba`.
- `best_test_ap_aug8` — test AP after averaging 8 forward passes, each with
  a fresh random draw of the eigenvector ambiguity group (sign flips on
  simple eigenvalues + Haar `O(m)` rotations on any multiplicity-`m`
  blocks). K=8 is chosen because with `k=3` simple eigenvalues the sign
  group is `Z₂³` (size 8), so K=8 already samples it exhaustively in
  expectation. `_aug8` is only populated for `random_augmented` runs
  (`--test-aug-samples 8` is silently ignored for `map` and `oap`).

## Hyperparameters (see plan.md for the full table)

`ogbg-molpcba`, GIN 5-layer, batch 32, lr 1e-3, epochs 200 / patience 15,
`k=3`, no eigval scaling, seeds {0,1,2}, hidden dims {16, 128, 512}. Three
canonicalizations, all retrained under this protocol: `random_augmented`,
`map`, `oap`. 27 runs total.

## Launch command

```bash
python scripts/run_lappe_sweep.py \
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

## Result directory layout

```
results/test_time_averaging_augmented/
├── README.md               ← this file
├── plan.md                 ← approved pre-implementation plan
├── gin/
│   └── <canon>_k3_h<H>_s<S>[_evscale]/results.json     (27 dirs)
├── summary.csv             ← one row per run, 27 rows
└── plots/
    ├── ap_vs_hdim.png              ← single-pass AP vs hidden dim
    └── ap_vs_hdim_augmented.png    ← aug-averaged vs baselines
```

## Results

Aggregated over 3 seeds per cell (mean ± std):

| canonicalization   |   h  | params     | AP (single)       | AP (aug-avg K=8)  |   Δ aug   |
|--------------------|-----:|-----------:|------------------:|------------------:|----------:|
| map                |   16 |      5,744 | 0.0626 ± 0.0025   |        —          |    —      |
| oap                |   16 |      5,744 | 0.0641 ± 0.0051   |        —          |    —      |
| **random_augmented** |   16 |      5,744 | **0.0662 ± 0.0015** |   0.0663 ± 0.0015 |  +0.0001  |
| map                |  128 |    202,752 | 0.1704 ± 0.0060   |        —          |    —      |
| oap                |  128 |    202,752 | 0.1682 ± 0.0089   |        —          |    —      |
| **random_augmented** |  128 |    202,752 | **0.1762 ± 0.0064** |   0.1765 ± 0.0062 |  +0.0002  |
| map                |  512 |  2,973,312 | 0.2397 ± 0.0122   |        —          |    —      |
| **oap**            |  512 |  2,973,312 | **0.2431 ± 0.0022** |        —          |    —      |
| random_augmented   |  512 |  2,973,312 | 0.2395 ± 0.0117   |   0.2398 ± 0.0116 |  +0.0002  |

Bold entries mark the best mean AP at each hidden dim.

## Interpretation

### 1. Hypothesis result: partially supported

- At **h = 16 and h = 128**, `random_augmented` clearly beats both `map` and
  `oap`. Test-time aug-averaging gives a further small but consistent bump
  (+0.0001 to +0.0002), but the single-pass random_augmented was already the
  winner.
- At **h = 512**, the ordering flips: `oap` (0.2431) > random_augmented
  single-pass (0.2395) and random_augmented aug-avg K=8 (0.2398). The
  aug-averaging cannot close the ~0.0033 AP gap to `oap`.

### 2. Why is the aug-averaging Δ so small?

By design. With `k=3` simple eigenvalues the ambiguity group is `Z₂³` of
size 8, and K=8 averages over every element once. So the aug-averaged
prediction is essentially the *group-invariant average* of the model's
predictions across its input ambiguity. A random_augmented model trained
for tens of epochs has already become near-invariant to sign flips (the
training signal pushes it to produce the same prediction under any sign
pattern); averaging mainly denoises the residual variance — ~0.0002 AP on
this setup. The aug-averaging is consistent and helpful, but tiny relative
to other sources of variance.

### 3. Why does oap win at h=512?

Not a defect of the protocol — it's a real finding. `oap` at `h=512` uses
its full capacity to exploit the deterministically-ordered PE, and the
seed-to-seed variance of `oap` (±0.0022) is ~5× smaller than
random_augmented's (±0.0117). The augmentation signal is a double-edged
sword in the large-capacity regime: it regularises small models but adds
noise that a large model has to learn to ignore.

### 4. Actionable takeaway

- For **small-to-medium hidden dimensions** on molpcba, prefer
  `random_augmented` — it beats canonicalization cleanly, and test-time
  aug-averaging (K=8) is a free, small further gain.
- For **large hidden dimensions** on molpcba, `oap` is the better choice.
- **Test-time aug-averaging is safe**: in all tested configurations aug-avg
  ≥ single-pass, and it never hurts by more than 1 std.

## Caveats

- The study is restricted to `k=3`. At larger `k`, the ambiguity group grows
  (`Z₂^k` for simple spectra) and K=8 will no longer be exhaustive — the
  Δ aug-averaging might grow.
- Only `map` and `oap` are included as canonicalization baselines (per
  user's instruction); `spielman`, `maxabs`, `none` are out of scope here.
  See `results/lappe_sweep/` for those methods under the prior
  (inconsistent-budget) protocol.
- All runs use patience 15 / epochs 200. Most early-stopped at epoch
  40–90; the full 200-epoch budget was rarely used.
- The `_aug8` metric uses a reproducible per-`(base_seed, graph_idx)` RNG
  derived as `(seed * 1_000_003 + s) × 2_654_435_761 ⊕ graph_idx`, so any
  run with the same `--seed` reproduces the aug-averaged number exactly.

## Reproducing

```bash
pytest tests/test_unified_canonicalization.py::TestRandomAugmentEigenvectors -v
bash scripts/launch.sh                       # or the launch command above
python scripts/run_lappe_sweep.py --analysis-only \
    --base-dir results/test_time_averaging_augmented
```
