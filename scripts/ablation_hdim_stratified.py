#!/usr/bin/env python
"""Stratified h-dim ablation: training + analysis at low hidden dims.

Trains GIN and/or GCN models at low hidden dims (8, 16, 32, 64) across all
canonicalization methods and seeds, then runs stratified ROC-AUC analysis
by canonicalizability bin.

Usage:
    # Dry run (show what would be trained)
    python scripts/ablation_hdim_stratified.py --model gin --dry-run

    # Train GIN at low h-dims
    python scripts/ablation_hdim_stratified.py --model gin

    # Train GCN at low h-dims
    python scripts/ablation_hdim_stratified.py --model gcn

    # Analysis only (after training both)
    python scripts/ablation_hdim_stratified.py --model both --analysis-only

    # Custom h-dims
    python scripts/ablation_hdim_stratified.py --hdims 8 16 --model gin
"""

import argparse
import csv
import itertools
import os
import subprocess
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

# ── Constants (reused from ablation_moltox21.py) ─────────────────────────────

CANONICALIZATIONS = [
    "random_fixed",
    "random_augmented",
    "maxabs",
    "spielman",
    "map",
    "oap",
]

CANON_COLORS = {
    "random_fixed": "#888888",
    "random_augmented": "#bbbbbb",
    "maxabs": "#e67e22",
    "spielman": "#2980b9",
    "map": "#27ae60",
    "oap": "#8e44ad",
}

CANON_LABELS = {
    "random_fixed": "Random (fixed)",
    "random_augmented": "Random (aug)",
    "maxabs": "MaxAbs",
    "spielman": "Spielman",
    "map": "MAP",
    "oap": "OAP",
}

SEEDS = [0, 1, 2, 3, 4]

BINS = [
    ("<=1", 0, 1),
    ("2-3", 2, 3),
    ("4-5", 4, 5),
    ("6+", 6, 999),
]

MIN_SAMPLES = 6

# Training config matching existing experiments
TRAIN_CONFIG = {
    "dataset": "ogbg-moltox21",
    "num_layers": 5,
    "n_eigs": 8,
    "epochs": 200,
    "batch_size": 32,
    "lr": 1e-3,
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def compute_stratified_auc(y_true, y_pred):
    """Compute mean per-task ROC-AUC with NaN masking.

    Returns float or None if insufficient data.
    """
    task_aucs = []
    for t in range(y_true.shape[1]):
        valid = ~np.isnan(y_true[:, t])
        if valid.sum() >= MIN_SAMPLES and len(np.unique(y_true[valid, t])) > 1:
            task_aucs.append(roc_auc_score(y_true[valid, t], y_pred[valid, t]))
    return np.mean(task_aucs) if task_aucs else None


def load_audit(audit_dir):
    """Load audit_details.csv, return dict: graph_idx -> num_uncanonical."""
    csv_path = os.path.join(audit_dir, "audit_details.csv")
    uncanon_map = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["graph_idx"])
            n_eig = int(row.get("num_eigenvectors", 8))
            n_canon = int(row.get("num_individually_canonical", 0))
            uncanon_map[idx] = n_eig - n_canon
    return uncanon_map


def run_dir_for(base_dir, model, canon, hdim, seed):
    """Return the save directory for a given run."""
    return os.path.join(base_dir, model, f"{canon}_h{hdim}_s{seed}")


def run_exists(base_dir, model, canon, hdim, seed):
    """Check if a run has already completed."""
    rd = run_dir_for(base_dir, model, canon, hdim, seed)
    return os.path.exists(os.path.join(rd, "results.json"))


# ── Training Phase ───────────────────────────────────────────────────────────


def train_runs(base_dir, models, hdims, data_dir="data", dry_run=False):
    """Launch training runs for all model x canon x hdim x seed combos."""
    print("\n" + "=" * 70)
    print("Training Phase")
    print("=" * 70)

    commands = []
    for model, canon, hdim, seed in itertools.product(models, CANONICALIZATIONS, hdims, SEEDS):
        if run_exists(base_dir, model, canon, hdim, seed):
            continue
        save_dir = run_dir_for(base_dir, model, canon, hdim, seed)
        cmd = [
            sys.executable,
            "scripts/train_molecular.py",
            "--dataset",
            TRAIN_CONFIG["dataset"],
            "--data-dir",
            data_dir,
            "--model",
            model,
            "--canonicalization",
            canon,
            "--hidden-dim",
            str(hdim),
            "--num-layers",
            str(TRAIN_CONFIG["num_layers"]),
            "--n-eigs",
            str(TRAIN_CONFIG["n_eigs"]),
            "--epochs",
            str(TRAIN_CONFIG["epochs"]),
            "--batch-size",
            str(TRAIN_CONFIG["batch_size"]),
            "--lr",
            str(TRAIN_CONFIG["lr"]),
            "--seed",
            str(seed),
            "--save-dir",
            save_dir,
            "--save-predictions",
        ]
        commands.append((f"{model}/{canon}_h{hdim}_s{seed}", cmd))

    total_possible = len(models) * len(CANONICALIZATIONS) * len(hdims) * len(SEEDS)
    n_skipped = total_possible - len(commands)
    print(f"  {len(commands)} runs to launch ({n_skipped} already exist)")

    if dry_run:
        print("\n  [DRY RUN] Would launch:")
        for name, cmd in commands:
            print(f"    {name}")
            print(f"      {' '.join(cmd)}")
        return

    for i, (name, cmd) in enumerate(commands):
        print(f"\n{'=' * 70}")
        print(f"[{i + 1}/{len(commands)}] {name}")
        print(f"  {' '.join(cmd)}")
        print("=" * 70)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  WARNING: {name} failed with return code {result.returncode}")


# ── Analysis Phase ───────────────────────────────────────────────────────────


def load_predictions(base_dir, model, hdims):
    """Load test predictions for a model across all hdims.

    Returns dict: (canon, hdim, seed) -> {y_true, y_pred, graph_indices}.
    """
    predictions = {}
    for canon, hdim, seed in itertools.product(CANONICALIZATIONS, hdims, SEEDS):
        rd = run_dir_for(base_dir, model, canon, hdim, seed)
        pred_path = os.path.join(rd, "test_predictions.npz")
        if not os.path.exists(pred_path):
            continue
        pred = np.load(pred_path)
        predictions[(canon, hdim, seed)] = {
            "y_true": pred["y_true"],
            "y_pred": pred["y_pred"],
            "graph_indices": pred["graph_indices"],
        }
    return predictions


def analyze_single_model(base_dir, audit_dir, output_dir, model, hdims):
    """Stratified analysis for a single model across h-dims."""
    print(f"\n{'=' * 70}")
    print(f"Analysis: {model.upper()} — stratified by canonicalizability bin")
    print("=" * 70)

    uncanon_map = load_audit(audit_dir)
    predictions = load_predictions(base_dir, model, hdims)
    print(f"  Loaded {len(predictions)} prediction runs for {model}")

    # Build bin membership
    bin_indices = {}
    for label, lo, hi in BINS:
        bin_indices[label] = {idx for idx, n in uncanon_map.items() if lo <= n <= hi}

    model_dir = os.path.join(output_dir, model)
    os.makedirs(model_dir, exist_ok=True)

    # Per h-dim analysis
    all_results = {}  # hdim -> {bin_label -> {canon -> (mean, std)}}

    for hdim in hdims:
        results_table = defaultdict(dict)

        for bin_label, b_indices in bin_indices.items():
            for canon in CANONICALIZATIONS:
                seed_aucs = []
                for seed in SEEDS:
                    key = (canon, hdim, seed)
                    if key not in predictions:
                        continue
                    pred = predictions[key]
                    mask = np.array([i in b_indices for i in pred["graph_indices"]])
                    if mask.sum() == 0:
                        continue
                    auc = compute_stratified_auc(pred["y_true"][mask], pred["y_pred"][mask])
                    if auc is not None:
                        seed_aucs.append(auc)
                if seed_aucs:
                    results_table[bin_label][canon] = (np.mean(seed_aucs), np.std(seed_aucs))

        all_results[hdim] = results_table

        # Save CSV for this h-dim
        csv_path = os.path.join(model_dir, f"stratified_h{hdim}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["bin"] + [CANON_LABELS[c] for c in CANONICALIZATIONS]
            writer.writerow(header)
            for bin_label, _, _ in BINS:
                row = [bin_label]
                for canon in CANONICALIZATIONS:
                    if canon in results_table[bin_label]:
                        m, s = results_table[bin_label][canon]
                        row.append(f"{m:.4f} +/- {s:.4f}")
                    else:
                        row.append("N/A")
                writer.writerow(row)
        print(f"  Saved {csv_path}")

        # Bar chart for this h-dim
        _plot_stratified_bar(
            results_table,
            bin_indices,
            title=(
                f"Stratified ROC-AUC — {model.upper()}, h={hdim}\n"
                f"(ogbg-moltox21, {len(SEEDS)} seeds)"
            ),
            save_path=os.path.join(model_dir, f"stratified_h{hdim}.pdf"),
        )

    # Cross h-dim summary
    _cross_hdim_summary(all_results, hdims, model, model_dir)

    return all_results


def _plot_stratified_bar(results_table, bin_indices, title, save_path):
    """Grouped bar chart of stratified ROC-AUC."""
    bin_labels = [b[0] for b in BINS]
    n_bins = len(bin_labels)
    n_methods = len(CANONICALIZATIONS)
    width = 0.8 / n_methods
    x = np.arange(n_bins)

    fig, ax = plt.subplots(figsize=(12, 6))
    for j, canon in enumerate(CANONICALIZATIONS):
        means = []
        stds = []
        for bin_label in bin_labels:
            if canon in results_table[bin_label]:
                m, s = results_table[bin_label][canon]
                means.append(m)
                stds.append(s)
            else:
                means.append(0)
                stds.append(0)
        offset = (j - n_methods / 2 + 0.5) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=CANON_LABELS[canon],
            color=CANON_COLORS[canon],
            capsize=2,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_ylabel("Test ROC-AUC")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{bl}\n({len(bin_indices[bl])} graphs)" for bl in bin_labels])
    ax.set_xlabel("Number of uncanonical eigenvectors")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


def _cross_hdim_summary(all_results, hdims, model, output_dir):
    """Summary of how canonicalization benefit scales with hidden dim."""
    csv_path = os.path.join(output_dir, "cross_hdim_summary.csv")

    # For each hdim and bin, compute delta = best_deterministic - random_fixed
    deterministic = ["maxabs", "spielman", "map", "oap"]
    rows = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "hdim",
                "bin",
                "random_fixed_auc",
                "best_determ_method",
                "best_determ_auc",
                "delta",
            ]
        )
        for hdim in hdims:
            rt = all_results.get(hdim, {})
            for bin_label, _, _ in BINS:
                rf_auc = rt.get(bin_label, {}).get("random_fixed", (None, None))[0]
                best_method = None
                best_auc = None
                for c in deterministic:
                    val = rt.get(bin_label, {}).get(c, (None, None))[0]
                    if val is not None and (best_auc is None or val > best_auc):
                        best_auc = val
                        best_method = CANON_LABELS[c]
                delta = None
                if rf_auc is not None and best_auc is not None:
                    delta = best_auc - rf_auc
                writer.writerow(
                    [
                        hdim,
                        bin_label,
                        f"{rf_auc:.4f}" if rf_auc is not None else "N/A",
                        best_method or "N/A",
                        f"{best_auc:.4f}" if best_auc is not None else "N/A",
                        f"{delta:.4f}" if delta is not None else "N/A",
                    ]
                )
                rows.append(
                    {
                        "hdim": hdim,
                        "bin": bin_label,
                        "delta": delta,
                    }
                )

    print(f"  Saved {csv_path}")

    # Cross-hdim line plot: delta (best deterministic - random) per bin
    fig, ax = plt.subplots(figsize=(10, 6))
    bin_labels = [b[0] for b in BINS]
    bin_colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    for i, bin_label in enumerate(bin_labels):
        deltas = []
        valid_hdims = []
        for hdim in hdims:
            d = next(
                (r["delta"] for r in rows if r["hdim"] == hdim and r["bin"] == bin_label),
                None,
            )
            if d is not None:
                deltas.append(d)
                valid_hdims.append(hdim)
        if valid_hdims:
            ax.plot(
                valid_hdims,
                deltas,
                label=f"Bin {bin_label}",
                color=bin_colors[i],
                marker="o",
                linewidth=1.5,
                markersize=5,
            )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(hdims)
    ax.set_xticklabels([str(h) for h in hdims])
    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel("Delta ROC-AUC (best deterministic - random fixed)")
    ax.set_title(
        f"Canonicalization Benefit vs Model Capacity — {model.upper()}\n"
        f"(ogbg-moltox21, {len(SEEDS)} seeds)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pdf_path = os.path.join(output_dir, "cross_hdim_delta.pdf")
    fig.savefig(pdf_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {pdf_path}")


def compare_gin_gcn(base_dir, audit_dir, output_dir, hdims):
    """Compare GIN vs GCN: per-bin delta showing whether weaker backbone benefits more."""
    print(f"\n{'=' * 70}")
    print("GIN vs GCN Comparison")
    print("=" * 70)

    uncanon_map = load_audit(audit_dir)
    bin_indices = {}
    for label, lo, hi in BINS:
        bin_indices[label] = {idx for idx, n in uncanon_map.items() if lo <= n <= hi}

    # Load predictions for both models
    gin_preds = load_predictions(base_dir, "gin", hdims)
    gcn_preds = load_predictions(base_dir, "gcn", hdims)

    if not gin_preds or not gcn_preds:
        print("  Skipping comparison: need predictions for both gin and gcn")
        return

    comp_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comp_dir, exist_ok=True)

    # Compute per-model, per-hdim, per-bin: mean over seeds of (best_determ - random_fixed)
    deterministic = ["maxabs", "spielman", "map", "oap"]

    def compute_deltas(predictions, hdims_list):
        """Return {(hdim, bin_label): delta}."""
        deltas = {}
        for hdim in hdims_list:
            for bin_label, b_indices in bin_indices.items():
                rf_aucs = []
                best_det_aucs = []
                for seed in SEEDS:
                    # Random fixed
                    key_rf = ("random_fixed", hdim, seed)
                    if key_rf in predictions:
                        pred = predictions[key_rf]
                        mask = np.array([i in b_indices for i in pred["graph_indices"]])
                        if mask.sum() > 0:
                            a = compute_stratified_auc(pred["y_true"][mask], pred["y_pred"][mask])
                            if a is not None:
                                rf_aucs.append(a)

                    # Best deterministic
                    seed_best = None
                    for c in deterministic:
                        key_c = (c, hdim, seed)
                        if key_c in predictions:
                            pred = predictions[key_c]
                            mask = np.array([i in b_indices for i in pred["graph_indices"]])
                            if mask.sum() > 0:
                                a = compute_stratified_auc(
                                    pred["y_true"][mask], pred["y_pred"][mask]
                                )
                                if a is not None and (seed_best is None or a > seed_best):
                                    seed_best = a
                    if seed_best is not None:
                        best_det_aucs.append(seed_best)

                if rf_aucs and best_det_aucs:
                    deltas[(hdim, bin_label)] = np.mean(best_det_aucs) - np.mean(rf_aucs)
        return deltas

    gin_deltas = compute_deltas(gin_preds, hdims)
    gcn_deltas = compute_deltas(gcn_preds, hdims)

    # CSV
    csv_path = os.path.join(comp_dir, "gin_vs_gcn_deltas.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hdim", "bin", "gin_delta", "gcn_delta", "gcn_minus_gin"])
        for hdim in hdims:
            for bin_label, _, _ in BINS:
                gd = gin_deltas.get((hdim, bin_label))
                cd = gcn_deltas.get((hdim, bin_label))
                diff = None
                if gd is not None and cd is not None:
                    diff = cd - gd
                writer.writerow(
                    [
                        hdim,
                        bin_label,
                        f"{gd:.4f}" if gd is not None else "N/A",
                        f"{cd:.4f}" if cd is not None else "N/A",
                        f"{diff:.4f}" if diff is not None else "N/A",
                    ]
                )
    print(f"  Saved {csv_path}")

    # Plot: grouped bars per hdim, one group per bin, bars = GIN vs GCN delta
    bin_labels = [b[0] for b in BINS]
    fig, axes = plt.subplots(1, len(hdims), figsize=(4 * len(hdims), 5), sharey=True)
    if len(hdims) == 1:
        axes = [axes]

    for ax, hdim in zip(axes, hdims):
        x = np.arange(len(bin_labels))
        width = 0.35
        gin_vals = [gin_deltas.get((hdim, bl), 0) or 0 for bl in bin_labels]
        gcn_vals = [gcn_deltas.get((hdim, bl), 0) or 0 for bl in bin_labels]
        ax.bar(x - width / 2, gin_vals, width, label="GIN", color="#3498db", edgecolor="white")
        ax.bar(x + width / 2, gcn_vals, width, label="GCN", color="#e74c3c", edgecolor="white")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"h={hdim}")
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, fontsize=8)
        ax.set_xlabel("Uncanonical eigvecs")
        if ax == axes[0]:
            ax.set_ylabel("Delta ROC-AUC\n(best determ. - random)")
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Canonicalization Benefit: GIN vs GCN by Capacity\n(ogbg-moltox21)",
        fontsize=12,
    )
    fig.tight_layout()
    pdf_path = os.path.join(comp_dir, "gin_vs_gcn_delta.pdf")
    fig.savefig(pdf_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {pdf_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Stratified h-dim ablation: training + analysis at low hidden dims"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="results/canonicalization_experiments/exp_hdim_stratified",
        help="Base directory for saving runs ({model}/{canon}_h{hdim}_s{seed})",
    )
    parser.add_argument(
        "--audit-dir",
        type=str,
        default="results/canonicalization_experiments/test_split_audit",
        help="Directory containing audit_details.csv",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gin",
        choices=["gin", "gcn", "both"],
        help="Model backbone(s) to train/analyze",
    )
    parser.add_argument(
        "--hdims",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64],
        help="Hidden dimensions to train at",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory for training",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, run analysis only (alias for --analysis-only)",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Skip training, run analysis only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands that would be run without executing",
    )
    args = parser.parse_args()

    models = ["gin", "gcn"] if args.model == "both" else [args.model]
    skip_training = args.skip_training or args.analysis_only

    output_dir = os.path.join(args.base_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Training phase
    if not skip_training:
        train_runs(args.base_dir, models, args.hdims, args.data_dir, args.dry_run)
        if args.dry_run:
            return

    # Analysis phase
    for model in models:
        analyze_single_model(args.base_dir, args.audit_dir, output_dir, model, args.hdims)

    # Comparison if both models available
    if len(models) == 2 or args.model == "both":
        compare_gin_gcn(args.base_dir, args.audit_dir, output_dir, args.hdims)

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
