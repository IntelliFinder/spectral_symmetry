# Appendix B: Augmentation vs. Canonization on `ogbg-molpcba`

Reproducible code and raw results for the appendix experiments comparing
augmentation against deterministic canonization of Laplacian positional
encodings on the OGB `ogbg-molpcba` graph property prediction benchmark.

The appendix compares three strategies for resolving the eigenvector
ambiguity group of Laplacian positional encodings (LapPE) on the OGB
`ogbg-molpcba` benchmark: two deterministic projection-matrix
canonizations (`map`, `oap` from Ma et al., NeurIPS 2023) and a training-time
random sign-flip augmentation (`random_augmented`).

## Layout

```
submission/
├── README.md                        # this file
├── requirements.txt                 # pip-installable package list
├── environment.yml                  # full conda export
├── src/                             # python package
│   ├── spectral_canonicalization.py # map, oap, random_augmented, +abs+maxabs
│   ├── spectral_core.py             # Laplacian eigenpair computation
│   ├── training.py                  # seed_everything, worker_init_fn
│   └── experiments/molecular/
│       ├── dataset.py               # MolecularLapPEDataset, get_augmented_test_loader
│       └── model.py                 # GINLapPE, GCNLapPE
├── scripts/
│   ├── train_molecular.py           # main training entry point
│   ├── run_lappe_sweep.py # multi-cell sweep launcher
│   ├── orbit_stability_eval.py      # per-graph orbit-stability eval
│   └── finalize_run.py              # rerun eval from saved best_model.pt
└── results/                         # raw outputs (results.json, best_model.pt, orbit_stability.json, etc.)
    ├── k3_h_in_16_128_512_200ep/             # 27 runs: 3 canon × 3 hidden × 3 seeds at k=3
    ├── k8_h_in_16_128_512_200ep/             # 27 runs at k=8
    ├── k16_h_in_16_128_512_500ep_random_augmented/   # 9 random_augmented runs at k=16
    └── k16_h_in_16_128_512_500ep_map_oap/    # 18 map+oap runs at k=16 (matched compute)
```

The `results/` tree mirrors the per-cell directory naming the training
scripts produce: `<canon>_k<k>_h<h>_s<seed>/`. Each run directory holds:

- `results.json`     — best-validation-AP checkpoint metadata + final test AP (and aug-K when applicable)
- `best_model.pt`    — saved checkpoint (state_dict)
- `epoch_log.json`   — per-epoch train loss / val AP / test AP / wallclock time
- `train.log`        — stdout of the training run
- `orbit_stability.json` — present for `random_augmented` cells we measured (Table 6)

## Setup

This was developed with Python 3.10, PyTorch 2.10.0, CUDA 12.8.

```bash
conda create -n appendix_b python=3.10
conda activate appendix_b
pip install -r requirements.txt
# (or: conda env create -f environment.yml)
```

The OGB `ogbg-molpcba` raw data is not bundled. It will download automatically
on first use (~280 MB). Pin the OGB evaluator to version `1.3.6`.

## Reproducing the appendix's tables

All commands assume you are in the `submission/` directory. Set
`CUDA_VISIBLE_DEVICES` as appropriate.

### Tables 4, 5 — main test-AP comparison

The tables aggregate 81 training runs. The sweep launcher orchestrates them.

```bash
# k=3 row (Tables 4 row 1-3): 27 runs, 200 epochs, patience 15
python3 scripts/run_lappe_sweep.py \
    --canonicalization map oap random_augmented \
    --k 3 --hidden-dim 16 128 512 \
    --cache-n-eigs 15 --seeds 0 1 2 \
    --epochs 200 --patience 15 \
    --no-eigval-scale \
    --test-aug-samples 8 \
    --base-dir results/k3_h_in_16_128_512_200ep \
    --gpus 0 1 --workers-per-gpu 2

# k=8 row (Tables 4 row 4-6): 27 runs at 200 ep / patience 15
python3 scripts/run_lappe_sweep.py \
    --canonicalization map oap random_augmented \
    --k 8 --hidden-dim 16 128 512 \
    --cache-n-eigs 15 --seeds 0 1 2 \
    --epochs 200 --patience 15 \
    --no-eigval-scale \
    --test-aug-samples 16 \
    --base-dir results/k8_h_in_16_128_512_200ep \
    --gpus 0 1 --workers-per-gpu 2

# k=16 row (Tables 4 row 7-9 + Table 5): 27 runs at 500 ep / patience 999 (matched compute)
python3 scripts/run_lappe_sweep.py \
    --canonicalization random_augmented \
    --k 16 --hidden-dim 16 128 512 \
    --cache-n-eigs 16 --seeds 0 1 2 \
    --epochs 500 --patience 999 \
    --no-eigval-scale \
    --test-aug-samples 16 \
    --base-dir results/k16_h_in_16_128_512_500ep_random_augmented \
    --gpus 0 1 --workers-per-gpu 2

python3 scripts/run_lappe_sweep.py \
    --canonicalization map oap \
    --k 16 --hidden-dim 16 128 512 \
    --cache-n-eigs 16 --seeds 0 1 2 \
    --epochs 500 --patience 999 \
    --no-eigval-scale \
    --base-dir results/k16_h_in_16_128_512_500ep_map_oap \
    --gpus 0 1 --workers-per-gpu 2
```

After completion, each run's `results.json` contains `best_test_ap` and
(for `random_augmented` only) `best_test_ap_aug{K}`. Aggregate by hand or
with the loader code in `scripts/run_lappe_sweep.py`'s analysis
mode.

The `--cache-n-eigs` flag controls the eigenvector cache: at k=8 we use
`cache-n-eigs=15` so the same cache is shared across multiple k slices.

### Table 6 — per-graph orbit stability

For each `random_augmented` checkpoint we want to evaluate, run:

```bash
python3 scripts/orbit_stability_eval.py \
    results/k3_h_in_16_128_512_200ep/gin/random_augmented_k3_h128_s0 \
    --canonicalization random_augmented --n-eigs 3 --cache-n-eigs 15 \
    --hidden-dim 128 --seed 0 --K 64 --base-seed 1
```

Repeat for the other (k, h, seed) cells (substitute `--n-eigs`,
`--cache-n-eigs`, `--hidden-dim`, `--seed` and the save_dir path). Each
invocation writes `orbit_stability.json` next to the checkpoint, with
`mean_spread_overall`, `per_draw_metric.std_ddof1`, and
`reynolds_mean_metric` — the columns in Table 6.

Already-computed `orbit_stability.json` files are present in `results/`
for the 12 cells reported (k∈{3,8,16} × 3 seeds at h=128, plus h∈{16,512}
× 3 seeds at k=16).

### Recovering AP from a saved checkpoint without retraining

If you have `best_model.pt` but no `results.json`, use:

```bash
python3 scripts/finalize_run.py \
    results/<cell>/<run_dir> \
    --canonicalization <method> --n-eigs <k> --cache-n-eigs 15 \
    --hidden-dim <h> --seed <s> --test-aug-samples 16
```

This reloads `best_model.pt`, runs val + test eval, and writes a
`results.json` with the same schema as the training script.

## Hyperparameters (Appendix B, "Hyperparameters and protocol")

GIN (5 layers); hidden dim h ∈ {16, 128, 512}; dropout 0.5; batch size 32;
Adam + cosine LR @ 1e-3; eigval-scaling off; eigenvector cache size 15 (or
16 at k=16). Best-validation-AP checkpoint selection (ties broken by
earliest epoch).

`cudnn.deterministic` is **not** enforced, so runs are not bit-reproducible
across hardware; in practice the resulting variability is absorbed into the
seed std at the magnitudes shown in Table 5.

## Citation

This work builds on:

- Ma, Wang, Wang. *Laplacian Canonization: A Minimalist Approach to Sign and
  Basis Invariant Spectral Embedding.* NeurIPS 2023. (the `map` and `oap`
  algorithms; our code is a faithful numpy port of their reference torch
  implementation.)
- Hu et al. *Open Graph Benchmark.* NeurIPS 2020. (`ogbg-molpcba`.)

## License

Code: MIT. Data: per the OGB license.
