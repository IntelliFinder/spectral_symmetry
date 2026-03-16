#!/usr/bin/env python
"""Nadav improvements ablation on ogbg-molpcba.

Extended sweep with:
  - New canonicalizations: abs (SignNet-style), spielman_partition (partial Spielman)
  - Eigenvalue-scaled eigenvectors (--eigval-scale)
  - Wider hidden-dim sweep: h={16, 32, 64, 128, 256}
  - All 9 canonicalization methods

Produces summary CSV and comparison plots in results/nadav_improvements/plots/.

Usage:
    # Dry run
    python scripts/ablation_nadav_improvements.py --gpus 0 1 2 3 4 5 --dry-run

    # Full training
    python scripts/ablation_nadav_improvements.py --gpus 0 1 2 3 4 5

    # Analysis only (after training)
    python scripts/ablation_nadav_improvements.py --analysis-only

    # Filter
    python scripts/ablation_nadav_improvements.py --model gin \
        --canonicalization abs spielman_partition
"""

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────

CANONICALIZATIONS = [
    "spielman",
    "spielman_partition",
    "maxabs",
    "random_fixed",
    "random_augmented",
    "map",
    "oap",
    "abs",
    "none",
]

CANON_COLORS = {
    "spielman": "#9b59b6",
    "spielman_partition": "#8e44ad",
    "maxabs": "#e74c3c",
    "random_fixed": "#3498db",
    "random_augmented": "#2ecc71",
    "map": "#f39c12",
    "oap": "#e67e22",
    "abs": "#1abc9c",
    "none": "#95a5a6",
}

CANON_LABELS = {
    "spielman": "Spielman (full)",
    "spielman_partition": "Spielman (partition)",
    "maxabs": "MaxAbs",
    "random_fixed": "Random (fixed)",
    "random_augmented": "Random (aug)",
    "map": "MAP",
    "oap": "OAP",
    "abs": "Abs (SignNet)",
    "none": "None",
}

K_VALUES = [3, 6, 12]
HIDDEN_DIMS = [16, 32, 64, 128, 256]
MODELS = ["gin"]
SEEDS = [0, 1, 2]
EIGVAL_SCALE_OPTIONS = [False, True]
CACHE_K = 15
MAX_PARALLEL = 6

BASE_DIR = "results/nadav_improvements"
PLOT_DIR = os.path.join(BASE_DIR, "plots")

METRIC_KEY = "best_test_ap"
METRIC_LABEL = "Test AP"


# ── Helpers ──────────────────────────────────────────────────────────────────


def save_dir_for(model, canon, k, h, seed, eigval_scale=False):
    """Return the save directory for a given run."""
    scale_tag = "_evscale" if eigval_scale else ""
    return os.path.join(BASE_DIR, model, f"{canon}_k{k}_h{h}_s{seed}{scale_tag}")


def run_exists(model, canon, k, h, seed, eigval_scale=False):
    """Check if a run has already completed."""
    return os.path.exists(
        os.path.join(save_dir_for(model, canon, k, h, seed, eigval_scale), "results.json")
    )


def load_results():
    """Load all results.json files under BASE_DIR."""
    results = {}
    for model in MODELS:
        model_dir = os.path.join(BASE_DIR, model)
        if not os.path.isdir(model_dir):
            continue
        for run_name in sorted(os.listdir(model_dir)):
            json_path = os.path.join(model_dir, run_name, "results.json")
            if not os.path.exists(json_path):
                continue
            with open(json_path) as f:
                r = json.load(f)
            key = (
                r.get("model", model),
                r["canonicalization"],
                r.get("n_eigs", 8),
                r.get("hidden_dim", 256),
                r["seed"],
                r.get("eigval_scale", False),
            )
            results[key] = r
    return results


def aggregate(results, model, canon, k, h, eigval_scale=False):
    """Return (mean_ap, std_ap, n_seeds) for given params, or None."""
    vals = []
    for seed in SEEDS:
        key = (model, canon, k, h, seed, eigval_scale)
        if key in results:
            ap = results[key].get(METRIC_KEY)
            if ap is not None:
                vals.append(ap)
    if vals:
        return np.mean(vals), np.std(vals), len(vals)
    return None


# ── Cache Pre-warming ────────────────────────────────────────────────────────


def prewarm_caches(canons, k_values, dataset="ogbg-molpcba", data_dir="data"):
    """Pre-compute LapPE caches for all canonicalization methods."""
    print(f"\n{'=' * 70}")
    print(f"Pre-warming LapPE caches (base k={CACHE_K})")
    print(f"{'=' * 70}")

    per_k_methods = ("spielman", "spielman_partition")

    def _warm(canon, k):
        cache_k = k if canon in per_k_methods else CACHE_K
        cache_dir = os.path.join(data_dir, "lappe_cache", f"{dataset}_{canon}_k{cache_k}")
        cache_path = os.path.join(cache_dir, "lappe.pkl")

        if os.path.exists(cache_path):
            print(f"  {canon} (k={k}): cache exists, skipping")
            return

        print(f"  {canon} (k={k}): building cache...")
        cmd = [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, '.'); "
            f"from src.experiments.molecular.dataset import MolecularLapPEDataset; "
            f"ds = MolecularLapPEDataset("
            f"  dataset_name='{dataset}', canonicalization='{canon}', "
            f"  n_eigs={k}, data_dir='{data_dir}', "
            f"  cache_n_eigs={CACHE_K}); "
            f"print(f'  Cached {{len(ds)}} graphs')",
        ]
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"  WARNING: cache pre-warm failed for {canon} k={k} (rc={result.returncode})")

    # Build "none" first (base eigdecomp cache shared by most methods)
    _warm("none", CACHE_K)

    for canon in canons:
        if canon in per_k_methods:
            for k in k_values:
                _warm(canon, k)
        else:
            _warm(canon, CACHE_K)

    print("  Cache pre-warming complete.\n")


# ── Training Phase ───────────────────────────────────────────────────────────


def build_commands(models, canons, k_values, hdims, seeds, eigval_scale_opts):
    """Build list of (name, cmd) tuples for all combos, skipping existing."""
    commands = []
    skipped = 0
    for model, canon, k, h, seed, evs in itertools.product(
        models, canons, k_values, hdims, seeds, eigval_scale_opts
    ):
        if run_exists(model, canon, k, h, seed, evs):
            skipped += 1
            continue
        sd = save_dir_for(model, canon, k, h, seed, evs)
        cmd = [
            sys.executable,
            "scripts/train_molecular.py",
            "--dataset",
            "ogbg-molpcba",
            "--model",
            model,
            "--canonicalization",
            canon,
            "--n-eigs",
            str(k),
            "--cache-n-eigs",
            str(CACHE_K),
            "--hidden-dim",
            str(h),
            "--num-layers",
            "5",
            "--epochs",
            "50",
            "--batch-size",
            "32",
            "--lr",
            "1e-3",
            "--seed",
            str(seed),
            "--patience",
            "10",
            "--save-dir",
            sd,
        ]
        if evs:
            cmd.append("--eigval-scale")
        scale_tag = "_evscale" if evs else ""
        name = f"{model}/{canon}_k{k}_h{h}_s{seed}{scale_tag}"
        commands.append((name, cmd))
    return commands, skipped


def _parse_name(name):
    """Parse name back into (model, canon, k, h, seed, eigval_scale)."""
    model, rest = name.split("/", 1)
    eigval_scale = rest.endswith("_evscale")
    if eigval_scale:
        rest = rest[: -len("_evscale")]
    parts = rest.rsplit("_", 3)
    seed = int(parts[-1][1:])
    h = int(parts[-2][1:])
    k = int(parts[-3][1:])
    canon = rest[: rest.rfind(f"_k{k}_h{h}_s{seed}")]
    return model, canon, k, h, seed, eigval_scale


def run_training(commands, dry_run=False, max_parallel=MAX_PARALLEL, gpus=None):
    """Execute training commands with multi-GPU distribution."""
    n_gpus = len(gpus) if gpus else 1
    total_parallel = max_parallel * n_gpus
    print(f"\n{'=' * 70}")
    if gpus:
        print(
            f"Training Phase ({max_parallel} workers x {n_gpus} GPUs = {total_parallel} parallel)"
        )
    else:
        print(f"Training Phase ({max_parallel} parallel workers)")
    print(f"{'=' * 70}")
    print(f"  {len(commands)} runs to launch")

    if dry_run:
        print("\n  [DRY RUN] Would launch:")
        for name, cmd in commands:
            print(f"    {name}")
        print(f"\n  Total: {len(commands)} runs")
        return

    active = {}
    gpu_counts = {g: 0 for g in (gpus or [None])}
    queue = list(enumerate(commands))
    done = 0
    total = len(commands)

    def _next_gpu():
        return min(gpu_counts, key=gpu_counts.get)

    while queue or active:
        while queue:
            gpu = _next_gpu()
            if gpu_counts[gpu] >= max_parallel:
                break
            i, (name, cmd) = queue.pop(0)
            gpu_label = f"GPU {gpu}" if gpu is not None else "default"
            print(f"  [START {i + 1}/{total}] {name} [{gpu_label}]")
            parsed = _parse_name(name)
            log_path = os.path.join(save_dir_for(*parsed), "train.log")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            log_f = open(log_path, "w")
            env = os.environ.copy()
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
            active[proc.pid] = (name, proc, log_f, i, gpu)
            gpu_counts[gpu] += 1

        if active:
            pid, status = os.waitpid(-1, 0)
            if pid in active:
                name, proc, log_f, idx, gpu = active.pop(pid)
                log_f.close()
                gpu_counts[gpu] -= 1
                done += 1
                rc = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
                if rc != 0:
                    print(f"  [FAIL {done}/{total}] {name} (rc={rc})")
                else:
                    print(f"  [DONE {done}/{total}] {name}")

    print(f"\n  All {total} runs complete.")


# ── Analysis Phase ───────────────────────────────────────────────────────────


def write_summary_csv(results):
    """Write summary CSV."""
    csv_path = os.path.join(PLOT_DIR, "summary_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "canonicalization",
                "k",
                "hidden_dim",
                "eigval_scale",
                "mean_ap",
                "std_ap",
                "n_seeds",
            ]
        )
        for model in MODELS:
            for canon in CANONICALIZATIONS:
                for k in K_VALUES:
                    for h in HIDDEN_DIMS:
                        for evs in EIGVAL_SCALE_OPTIONS:
                            agg = aggregate(results, model, canon, k, h, evs)
                            if agg is not None:
                                mean_ap, std_ap, n = agg
                                writer.writerow(
                                    [
                                        model,
                                        canon,
                                        k,
                                        h,
                                        evs,
                                        f"{mean_ap:.6f}",
                                        f"{std_ap:.6f}",
                                        n,
                                    ]
                                )
    print(f"  Wrote {csv_path}")


def plot_ap_vs_hdim(results, model, k, eigval_scale=False):
    """Plot AP vs hidden dim for all canonicalizations (one plot per model×k×evs)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for canon in CANONICALIZATIONS:
        means, stds, hs = [], [], []
        for h in HIDDEN_DIMS:
            agg = aggregate(results, model, canon, k, h, eigval_scale)
            if agg is not None:
                means.append(agg[0])
                stds.append(agg[1])
                hs.append(h)
        if means:
            ax.errorbar(
                hs,
                means,
                yerr=stds,
                label=CANON_LABELS.get(canon, canon),
                color=CANON_COLORS.get(canon, "#333333"),
                marker="o",
                capsize=3,
            )
    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel(METRIC_LABEL)
    evs_tag = " (eigval-scaled)" if eigval_scale else ""
    ax.set_title(f"{model.upper()} — k={k}{evs_tag}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(HIDDEN_DIMS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    evs_suffix = "_evscale" if eigval_scale else ""
    path = os.path.join(PLOT_DIR, f"ap_vs_hdim_{model}_k{k}{evs_suffix}.pdf")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {path}")


def plot_ap_vs_k(results, model, h, eigval_scale=False):
    """Plot AP vs k for all canonicalizations (one plot per model×h×evs)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for canon in CANONICALIZATIONS:
        means, stds, ks = [], [], []
        for k in K_VALUES:
            agg = aggregate(results, model, canon, k, h, eigval_scale)
            if agg is not None:
                means.append(agg[0])
                stds.append(agg[1])
                ks.append(k)
        if means:
            ax.errorbar(
                ks,
                means,
                yerr=stds,
                label=CANON_LABELS.get(canon, canon),
                color=CANON_COLORS.get(canon, "#333333"),
                marker="o",
                capsize=3,
            )
    ax.set_xlabel("Number of Eigenvectors (k)")
    ax.set_ylabel(METRIC_LABEL)
    evs_tag = " (eigval-scaled)" if eigval_scale else ""
    ax.set_title(f"{model.upper()} — h={h}{evs_tag}")
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    evs_suffix = "_evscale" if eigval_scale else ""
    path = os.path.join(PLOT_DIR, f"ap_vs_k_{model}_h{h}{evs_suffix}.pdf")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {path}")


def plot_spielman_comparison(results, model, k):
    """Compare full Spielman vs partition-only Spielman."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for canon in ["spielman", "spielman_partition"]:
        means, stds, hs = [], [], []
        for h in HIDDEN_DIMS:
            agg = aggregate(results, model, canon, k, h, False)
            if agg is not None:
                means.append(agg[0])
                stds.append(agg[1])
                hs.append(h)
        if means:
            ax.errorbar(
                hs,
                means,
                yerr=stds,
                label=CANON_LABELS[canon],
                color=CANON_COLORS[canon],
                marker="o",
                capsize=3,
            )
    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel(METRIC_LABEL)
    ax.set_title(f"Spielman Full vs Partition — {model.upper()} k={k}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(HIDDEN_DIMS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"spielman_comparison_{model}_k{k}.pdf")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {path}")


def plot_eigval_scale_effect(results, model, k):
    """Compare with/without eigenvalue scaling for each canonicalization."""
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    x = np.arange(len(CANONICALIZATIONS))

    means_no, stds_no = [], []
    means_yes, stds_yes = [], []
    for canon in CANONICALIZATIONS:
        # Use h=128 as representative
        agg_no = aggregate(results, model, canon, k, 128, False)
        agg_yes = aggregate(results, model, canon, k, 128, True)
        means_no.append(agg_no[0] if agg_no else 0)
        stds_no.append(agg_no[1] if agg_no else 0)
        means_yes.append(agg_yes[0] if agg_yes else 0)
        stds_yes.append(agg_yes[1] if agg_yes else 0)

    ax.bar(x - width / 2, means_no, width, yerr=stds_no, label="Standard", capsize=3, alpha=0.8)
    ax.bar(
        x + width / 2,
        means_yes,
        width,
        yerr=stds_yes,
        label="Eigval-scaled (1/√λ)",
        capsize=3,
        alpha=0.8,
    )

    ax.set_xlabel("Canonicalization")
    ax.set_ylabel(METRIC_LABEL)
    ax.set_title(f"Eigenvalue Scaling Effect — {model.upper()} k={k} h=128")
    ax.set_xticks(x)
    ax.set_xticklabels([CANON_LABELS[c] for c in CANONICALIZATIONS], rotation=45, ha="right")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"eigval_scale_effect_{model}_k{k}.pdf")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {path}")


def run_analysis(results):
    """Generate all plots and summary."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"\n{'=' * 70}")
    print("Analysis Phase")
    print(f"{'=' * 70}")

    write_summary_csv(results)

    # AP vs hidden dim (per model × k × eigval_scale)
    for model in MODELS:
        for k in K_VALUES:
            for evs in EIGVAL_SCALE_OPTIONS:
                plot_ap_vs_hdim(results, model, k, evs)

    # AP vs k (per model × representative h × eigval_scale)
    for model in MODELS:
        for h in [64, 128]:
            for evs in EIGVAL_SCALE_OPTIONS:
                plot_ap_vs_k(results, model, h, evs)

    # Spielman comparison (per model × k)
    for model in MODELS:
        for k in K_VALUES:
            plot_spielman_comparison(results, model, k)

    # Eigenvalue scaling effect (per model × k)
    for model in MODELS:
        for k in K_VALUES:
            plot_eigval_scale_effect(results, model, k)

    print(f"\n  All plots saved to {PLOT_DIR}/")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Nadav improvements ablation on molpcba")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument(
        "--analysis-only", action="store_true", help="Skip training, run analysis only"
    )
    parser.add_argument("--gpus", type=int, nargs="+", default=None, help="GPU IDs to use")
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=MAX_PARALLEL,
        help="Max concurrent jobs per GPU",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific model(s)",
    )
    parser.add_argument(
        "--canonicalization",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific canonicalization(s)",
    )
    parser.add_argument(
        "--k", type=int, nargs="+", default=None, help="Filter to specific k value(s)"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        nargs="+",
        default=None,
        help="Filter to specific hidden dim(s)",
    )
    parser.add_argument(
        "--no-eigval-scale",
        action="store_true",
        help="Skip eigval-scale runs (only train without scaling)",
    )
    args = parser.parse_args()

    models = args.model or MODELS
    canons = args.canonicalization or CANONICALIZATIONS
    k_values = args.k or K_VALUES
    hdims = args.hidden_dim or HIDDEN_DIMS
    evs_opts = [False] if args.no_eigval_scale else EIGVAL_SCALE_OPTIONS

    if args.analysis_only:
        results = load_results()
        print(f"Loaded {len(results)} results")
        run_analysis(results)
        return

    # Pre-warm caches (skip for dry-run)
    if not args.dry_run:
        prewarm_caches(canons, k_values)

    # Build and run training commands
    commands, skipped = build_commands(models, canons, k_values, hdims, SEEDS, evs_opts)
    total = len(commands) + skipped
    print(f"\nTotal configurations: {total} ({skipped} already done, {len(commands)} to run)")

    run_training(commands, dry_run=args.dry_run, max_parallel=args.workers_per_gpu, gpus=args.gpus)

    if not args.dry_run:
        results = load_results()
        print(f"\nLoaded {len(results)} results")
        run_analysis(results)


if __name__ == "__main__":
    main()
