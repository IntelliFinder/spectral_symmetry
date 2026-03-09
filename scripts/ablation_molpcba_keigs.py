#!/usr/bin/env python
"""Eigenvector count (k) ablation study on ogbg-molpcba.

Sweeps over k (number of eigenvectors), hidden dims, canonicalization methods,
models (GIN/GCN), and seeds. Produces summary CSV and five plot families.

Configs:
  A  — h=32 only, all k x all methods x 3 seeds x both models (144 runs)
  B  — h={32,256} x all k x all methods x 3 seeds x both models (288 runs, includes A)

Usage:
    # Dry run (show what would be trained)
    python scripts/ablation_molpcba_keigs.py --config A --dry-run

    # Train Config A
    python scripts/ablation_molpcba_keigs.py --config A

    # Analysis only (after training)
    python scripts/ablation_molpcba_keigs.py --config B --analysis-only

    # Filter to a single model
    python scripts/ablation_molpcba_keigs.py --config A --model gin --dry-run

    # Filter by canonicalization or k
    python scripts/ablation_molpcba_keigs.py --config A --canonicalization maxabs --k 6 9
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

CANONICALIZATIONS = ["random_fixed", "random_augmented", "oap", "spielman"]

CANON_COLORS = {
    "random_fixed": "#3498db",
    "random_augmented": "#2ecc71",
    "oap": "#e67e22",
    "spielman": "#9b59b6",
}

CANON_LABELS = {
    "random_fixed": "Random (fixed)",
    "random_augmented": "Random (aug)",
    "oap": "OAP",
    "spielman": "Spielman",
}

K_VALUES = [3, 6, 12]
HIDDEN_DIMS = [32, 128]
MODELS = ["gin", "gcn"]
SEEDS = [0, 1, 2]
CACHE_K = 15  # precompute this many eigenvectors, slice at runtime
MAX_PARALLEL = 6  # concurrent training jobs per GPU (safe for h=128 on 48GB A40)

BASE_DIR = "results/molpcba_keigs_ablation"
PLOT_DIR = os.path.join(BASE_DIR, "plots")

METRIC_KEY = "best_test_ap"
METRIC_LABEL = "Test AP"


# ── Helpers ──────────────────────────────────────────────────────────────────


def save_dir_for(model, canon, k, h, seed):
    """Return the save directory for a given run."""
    return os.path.join(BASE_DIR, model, f"{canon}_k{k}_h{h}_s{seed}")


def run_exists(model, canon, k, h, seed):
    """Check if a run has already completed (results.json exists)."""
    return os.path.exists(os.path.join(save_dir_for(model, canon, k, h, seed), "results.json"))


def load_results():
    """Load all results.json files under BASE_DIR.

    Returns dict: (model, canon, k, h, seed) -> results_dict
    """
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
            )
            results[key] = r
    return results


def aggregate(results, model, canon, k, h):
    """Return (mean_ap, std_ap, n_seeds) for given params, or None."""
    vals = []
    for seed in SEEDS:
        key = (model, canon, k, h, seed)
        if key in results:
            ap = results[key].get(METRIC_KEY)
            if ap is not None:
                vals.append(ap)
    if vals:
        return np.mean(vals), np.std(vals), len(vals)
    return None


# ── Cache Pre-warming ────────────────────────────────────────────────────────


def prewarm_caches(canons, k_values, dataset="ogbg-molpcba", data_dir="data"):
    """Pre-compute LapPE caches for all canonicalization methods.

    Two-stage: eigdecomposition once (raw cache at k=CACHE_K), then derive
    each canon.  Spielman needs per-k caches (block structure depends on k).
    """
    print(f"\n{'=' * 70}")
    print(f"Pre-warming LapPE caches (base k={CACHE_K})")
    print(f"{'=' * 70}")

    def _warm(canon, k):
        cache_k = k if canon == "spielman" else CACHE_K
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

    for canon in canons:
        if canon == "spielman":
            # Spielman needs a separate cache per k
            for k in k_values:
                _warm(canon, k)
        else:
            # Other methods: one cache at CACHE_K, sliced at runtime
            _warm(canon, CACHE_K)

    print("  Cache pre-warming complete.\n")


# ── Training Phase ───────────────────────────────────────────────────────────


def build_commands(models, canons, k_values, hdims, seeds):
    """Build list of (name, cmd) tuples for all combos, skipping existing."""
    commands = []
    skipped = 0
    for model, canon, k, h, seed in itertools.product(models, canons, k_values, hdims, seeds):
        if run_exists(model, canon, k, h, seed):
            skipped += 1
            continue
        sd = save_dir_for(model, canon, k, h, seed)
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
            "100",
            "--batch-size",
            "32",
            "--lr",
            "1e-3",
            "--seed",
            str(seed),
            "--patience",
            "20",
            "--save-dir",
            sd,
        ]
        name = f"{model}/{canon}_k{k}_h{h}_s{seed}"
        commands.append((name, cmd))
    return commands, skipped


def run_training(commands, dry_run=False, max_parallel=MAX_PARALLEL, gpus=None):
    """Execute training commands with up to max_parallel concurrent jobs per GPU.

    Parameters
    ----------
    gpus : list of int or None
        GPU device IDs to distribute across (e.g. [0, 1, 2, 3, 4]).
        If None, no CUDA_VISIBLE_DEVICES is set (use default).
    """
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
            print(f"      {' '.join(cmd)}")
        return

    active = {}  # pid -> (name, Popen, log_f, idx, gpu_id)
    gpu_counts = {g: 0 for g in (gpus or [None])}  # track jobs per GPU
    queue = list(enumerate(commands))
    done = 0
    total = len(commands)

    def _next_gpu():
        """Return GPU with fewest active jobs, or None."""
        return min(gpu_counts, key=gpu_counts.get)

    while queue or active:
        # Launch new jobs up to max_parallel per GPU
        while queue:
            gpu = _next_gpu()
            if gpu_counts[gpu] >= max_parallel:
                break
            i, (name, cmd) = queue.pop(0)
            gpu_label = f"GPU {gpu}" if gpu is not None else "default"
            print(f"  [START {i + 1}/{total}] {name} [{gpu_label}]")
            log_path = os.path.join(save_dir_for(*_parse_name(name)), "train.log")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            log_f = open(log_path, "w")
            env = os.environ.copy()
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
            active[proc.pid] = (name, proc, log_f, i, gpu)
            gpu_counts[gpu] += 1

        # Wait for any child to finish
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


def _parse_name(name):
    """Parse 'model/canon_kK_hH_sS' back into (model, canon, k, h, seed)."""
    model, rest = name.split("/", 1)
    # rest = 'random_augmented_k6_h32_s0'
    parts = rest.rsplit("_", 3)  # [canon_prefix, kK, hH, sS]
    seed = int(parts[-1][1:])
    h = int(parts[-2][1:])
    k = int(parts[-3][1:])
    canon = rest[: rest.rfind(f"_k{k}_h{h}_s{seed}")]
    return model, canon, k, h, seed


# ── Analysis Phase ───────────────────────────────────────────────────────────


def write_summary_csv(results):
    """Write summary_results.csv with columns:
    model, canonicalization, k, hidden_dim, mean_ap, std_ap, n_seeds.
    """
    csv_path = os.path.join(PLOT_DIR, "summary_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["model", "canonicalization", "k", "hidden_dim", "mean_ap", "std_ap", "n_seeds"]
        )
        for model in MODELS:
            for canon in CANONICALIZATIONS:
                for k in K_VALUES:
                    for h in HIDDEN_DIMS:
                        agg = aggregate(results, model, canon, k, h)
                        if agg is not None:
                            mean_ap, std_ap, n = agg
                            writer.writerow(
                                [
                                    model,
                                    canon,
                                    k,
                                    h,
                                    f"{mean_ap:.6f}",
                                    f"{std_ap:.6f}",
                                    n,
                                ]
                            )
    print(f"  Saved {csv_path}")


def plot1_ap_vs_k_config_a(results):
    """Plot 1: Test AP vs k for Config A (h=32), one plot per model.

    4 lines (one per method), x=k, y=mean AP +/- std.
    """
    h = 32
    for model in MODELS:
        fig, ax = plt.subplots(figsize=(8, 5))
        for canon in CANONICALIZATIONS:
            means, stds, valid_k = [], [], []
            for k in K_VALUES:
                agg = aggregate(results, model, canon, k, h)
                if agg is not None:
                    means.append(agg[0])
                    stds.append(agg[1])
                    valid_k.append(k)
            if not valid_k:
                continue
            means = np.array(means)
            stds = np.array(stds)
            valid_k = np.array(valid_k)
            ax.plot(
                valid_k,
                means,
                label=CANON_LABELS[canon],
                color=CANON_COLORS[canon],
                linewidth=1.5,
                marker="o",
                markersize=5,
            )
            ax.fill_between(
                valid_k,
                means - stds,
                means + stds,
                alpha=0.15,
                color=CANON_COLORS[canon],
            )

        ax.set_xlabel("Number of Eigenvectors (k)")
        ax.set_ylabel(METRIC_LABEL)
        ax.set_title(
            f"Config A: {METRIC_LABEL} vs k\n"
            f"(ogbg-molpcba, {model.upper()}, h={h}, {len(SEEDS)} seeds)"
        )
        ax.set_xticks(K_VALUES)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf_path = os.path.join(PLOT_DIR, f"configA_{model}_ap_vs_k.pdf")
        fig.savefig(pdf_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {pdf_path}")


def plot2_ap_vs_k_grid(results):
    """Plot 2: Test AP vs k grid -- one subplot per h-dim, one figure per model (Config B).

    2x3 grid layout.
    """
    nrows, ncols = 2, 3
    for model in MODELS:
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True)
        axes_flat = axes.flatten()

        for idx, h in enumerate(HIDDEN_DIMS):
            ax = axes_flat[idx]
            for canon in CANONICALIZATIONS:
                means, stds, valid_k = [], [], []
                for k in K_VALUES:
                    agg = aggregate(results, model, canon, k, h)
                    if agg is not None:
                        means.append(agg[0])
                        stds.append(agg[1])
                        valid_k.append(k)
                if not valid_k:
                    continue
                means = np.array(means)
                stds = np.array(stds)
                valid_k = np.array(valid_k)
                ax.plot(
                    valid_k,
                    means,
                    label=CANON_LABELS[canon],
                    color=CANON_COLORS[canon],
                    linewidth=1.5,
                    marker="o",
                    markersize=4,
                )
                ax.fill_between(
                    valid_k,
                    means - stds,
                    means + stds,
                    alpha=0.15,
                    color=CANON_COLORS[canon],
                )

            ax.set_title(f"h={h}", fontsize=11)
            ax.set_xticks(K_VALUES)
            ax.set_xlabel("k")
            if idx % ncols == 0:
                ax.set_ylabel(METRIC_LABEL)
            ax.grid(True, alpha=0.3)

        # Hide extra subplot if HIDDEN_DIMS < nrows*ncols
        for idx in range(len(HIDDEN_DIMS), nrows * ncols):
            axes_flat[idx].set_visible(False)

        # Single legend
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower right", fontsize=9, ncol=2)

        fig.suptitle(
            f"Config B: {METRIC_LABEL} vs k by Hidden Dim\n"
            f"(ogbg-molpcba, {model.upper()}, {len(SEEDS)} seeds)",
            fontsize=13,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf_path = os.path.join(PLOT_DIR, f"configB_{model}_ap_vs_k_grid.pdf")
        fig.savefig(pdf_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {pdf_path}")


def plot3_ap_vs_hdim_grid(results):
    """Plot 3: Test AP vs h-dim grid -- one subplot per k, one figure per model (Config B).

    2x3 grid layout.
    """
    nrows, ncols = 2, 3
    for model in MODELS:
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True)
        axes_flat = axes.flatten()

        for idx, k in enumerate(K_VALUES):
            ax = axes_flat[idx]
            for canon in CANONICALIZATIONS:
                means, stds, valid_h = [], [], []
                for h in HIDDEN_DIMS:
                    agg = aggregate(results, model, canon, k, h)
                    if agg is not None:
                        means.append(agg[0])
                        stds.append(agg[1])
                        valid_h.append(h)
                if not valid_h:
                    continue
                means = np.array(means)
                stds = np.array(stds)
                valid_h = np.array(valid_h)
                ax.plot(
                    valid_h,
                    means,
                    label=CANON_LABELS[canon],
                    color=CANON_COLORS[canon],
                    linewidth=1.5,
                    marker="o",
                    markersize=4,
                )
                ax.fill_between(
                    valid_h,
                    means - stds,
                    means + stds,
                    alpha=0.15,
                    color=CANON_COLORS[canon],
                )

            ax.set_title(f"k={k}", fontsize=11)
            ax.set_xscale("log", base=2)
            ax.set_xticks(HIDDEN_DIMS)
            ax.set_xticklabels([str(h) for h in HIDDEN_DIMS])
            ax.set_xlabel("Hidden Dim")
            if idx % ncols == 0:
                ax.set_ylabel(METRIC_LABEL)
            ax.grid(True, alpha=0.3)

        # Hide extra subplot if K_VALUES < nrows*ncols
        for idx in range(len(K_VALUES), nrows * ncols):
            axes_flat[idx].set_visible(False)

        # Single legend
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower right", fontsize=9, ncol=2)

        fig.suptitle(
            f"Config B: {METRIC_LABEL} vs Hidden Dim by k\n"
            f"(ogbg-molpcba, {model.upper()}, {len(SEEDS)} seeds)",
            fontsize=13,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf_path = os.path.join(PLOT_DIR, f"configB_{model}_ap_vs_hdim_grid.pdf")
        fig.savefig(pdf_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {pdf_path}")


def plot4_heatmap(results):
    """Plot 4: Heatmap -- method x k, cell = mean AP, one per model at h=128."""
    h = 128
    for model in MODELS:
        n_methods = len(CANONICALIZATIONS)
        n_k = len(K_VALUES)
        matrix = np.full((n_methods, n_k), np.nan)

        for i, canon in enumerate(CANONICALIZATIONS):
            for j, k in enumerate(K_VALUES):
                agg = aggregate(results, model, canon, k, h)
                if agg is not None:
                    matrix[i, j] = agg[0]

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(METRIC_LABEL)

        # Annotate cells
        for i in range(n_methods):
            for j in range(n_k):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.4f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if val > np.nanmedian(matrix) else "black",
                    )

        ax.set_xticks(range(n_k))
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.set_yticks(range(n_methods))
        ax.set_yticklabels([CANON_LABELS[c] for c in CANONICALIZATIONS])
        ax.set_xlabel("Number of Eigenvectors (k)")
        ax.set_ylabel("Canonicalization Method")
        ax.set_title(
            f"Mean {METRIC_LABEL}: Method x k\n"
            f"(ogbg-molpcba, {model.upper()}, h={h}, {len(SEEDS)} seeds)"
        )
        fig.tight_layout()
        pdf_path = os.path.join(PLOT_DIR, f"heatmap_{model}_h{h}.pdf")
        fig.savefig(pdf_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {pdf_path}")


def plot5_gin_vs_gcn(results):
    """Plot 5: GIN vs GCN comparison -- for each method, overlay GIN and GCN
    lines on AP vs k plot. Single figure, 2x2 subplots (one per method).
    """
    h = 32  # Use Config A hidden dim for comparison
    n_methods = len(CANONICALIZATIONS)
    nrows, ncols = 2, 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True)
    axes_flat = axes.flatten()

    model_colors = {"gin": "#3498db", "gcn": "#e74c3c"}
    model_markers = {"gin": "o", "gcn": "s"}

    for idx, canon in enumerate(CANONICALIZATIONS):
        ax = axes_flat[idx]
        for model in MODELS:
            means, stds, valid_k = [], [], []
            for k in K_VALUES:
                agg = aggregate(results, model, canon, k, h)
                if agg is not None:
                    means.append(agg[0])
                    stds.append(agg[1])
                    valid_k.append(k)
            if not valid_k:
                continue
            means = np.array(means)
            stds = np.array(stds)
            valid_k = np.array(valid_k)
            ax.plot(
                valid_k,
                means,
                label=model.upper(),
                color=model_colors[model],
                linewidth=1.5,
                marker=model_markers[model],
                markersize=5,
            )
            ax.fill_between(
                valid_k,
                means - stds,
                means + stds,
                alpha=0.15,
                color=model_colors[model],
            )

        ax.set_title(CANON_LABELS[canon], fontsize=11)
        ax.set_xticks(K_VALUES)
        ax.set_xlabel("k")
        if idx % ncols == 0:
            ax.set_ylabel(METRIC_LABEL)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    # Hide extra subplots
    for idx in range(n_methods, nrows * ncols):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"GIN vs GCN: {METRIC_LABEL} vs k by Method\n(ogbg-molpcba, h={h}, {len(SEEDS)} seeds)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf_path = os.path.join(PLOT_DIR, "gin_vs_gcn_comparison.pdf")
    fig.savefig(pdf_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {pdf_path}")


def run_analysis(results):
    """Run all analysis: CSV + 5 plot families."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("Analysis Phase")
    print(f"{'=' * 70}")
    print(f"  Loaded {len(results)} completed runs")

    if not results:
        print("  No results found -- nothing to analyze.")
        return

    write_summary_csv(results)
    plot1_ap_vs_k_config_a(results)
    plot2_ap_vs_k_grid(results)
    plot3_ap_vs_hdim_grid(results)
    plot4_heatmap(results)
    plot5_gin_vs_gcn(results)

    print(f"\n  All plots saved to {PLOT_DIR}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Eigenvector count (k) ablation study on ogbg-molpcba"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="A",
        choices=["A", "B", "all"],
        help="Config A (h=32 only, 144 runs) or B (h=32+256, 288 runs). 'all' is an alias for B.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["gin", "gcn", "all"],
        help="Model backbone(s) to train/analyze (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands that would be run without executing",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Skip training, run analysis only",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Alias for --analysis-only",
    )

    # Filters
    parser.add_argument(
        "--canonicalization",
        type=str,
        nargs="+",
        default=None,
        choices=CANONICALIZATIONS,
        help="Filter to specific canonicalization method(s)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        nargs="+",
        default=None,
        help="Filter to specific hidden dim(s)",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=None,
        help="Filter to specific k value(s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=None,
        help="Filter to specific seed(s)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        help="GPU device IDs to distribute jobs across (e.g. --gpus 0 1 2 3 4)",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=MAX_PARALLEL,
        help=f"Max concurrent jobs per GPU (default: {MAX_PARALLEL})",
    )

    args = parser.parse_args()

    skip_training = args.analysis_only or args.skip_training

    # Resolve config
    if args.config in ("B", "all"):
        hdims = HIDDEN_DIMS
    else:
        hdims = [32]

    # Apply filters
    models = MODELS if args.model == "all" else [args.model]
    canons = args.canonicalization if args.canonicalization else CANONICALIZATIONS
    k_values = args.k if args.k else K_VALUES
    seeds = args.seed if args.seed else SEEDS

    if args.hidden_dim:
        hdims = [h for h in args.hidden_dim if h in hdims or args.config in ("B", "all")]
        if not hdims:
            hdims = args.hidden_dim

    total = len(models) * len(canons) * len(k_values) * len(hdims) * len(seeds)
    print(
        f"Config {args.config}: {len(models)} models x {len(canons)} methods "
        f"x {len(k_values)} k-values x {len(hdims)} h-dims x {len(seeds)} seeds "
        f"= {total} total runs"
    )

    # Training phase
    if not skip_training:
        # Pre-warm caches before launching parallel workers
        if not args.dry_run:
            prewarm_caches(canons, k_values)

        commands, skipped = build_commands(models, canons, k_values, hdims, seeds)
        print(f"  {skipped} already exist, {len(commands)} to run")
        run_training(
            commands,
            dry_run=args.dry_run,
            max_parallel=args.workers_per_gpu,
            gpus=args.gpus,
        )
        if args.dry_run:
            return

    # Analysis phase (always uses full results on disk)
    results = load_results()
    run_analysis(results)


if __name__ == "__main__":
    main()
