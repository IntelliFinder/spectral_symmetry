#!/usr/bin/env python
"""Ablation analysis for ogbg-moltox21 canonicalization experiments.

Three parts:
  Part 1: Finer stratification by uncanonical eigenvector count (from exp23 data)
  Part 2: Hidden-dim behavior investigation (from exp1 data)
  Part 3: Low hidden-dim training runs + extended analysis

Usage:
    # Parts 1+2 only (no training)
    python scripts/ablation_moltox21.py --skip-training

    # Part 3 only (training + extended analysis)
    python scripts/ablation_moltox21.py --part3-only

    # All parts
    python scripts/ablation_moltox21.py
"""

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

# ── Constants ────────────────────────────────────────────────────────────────

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

# Bins: (label, min_uncanon, max_uncanon_inclusive)
BINS = [
    ("<=1", 0, 1),
    ("2-3", 2, 3),
    ("4-5", 4, 5),
    ("6+", 6, 999),
]

EXISTING_HDIMS = [32, 64, 128, 256, 512]
LOW_HDIMS = [4, 8, 16]
ALL_HDIMS = sorted(LOW_HDIMS + EXISTING_HDIMS)

MIN_SAMPLES = 6  # Minimum samples per task for AUC computation


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


def load_exp23_predictions(exp23_dir):
    """Load test predictions from exp23 runs.

    Returns dict: (canonicalization, seed) -> {y_true, y_pred, graph_indices}.
    """
    predictions = {}
    for run_dir in os.listdir(exp23_dir):
        result_path = os.path.join(exp23_dir, run_dir, "results.json")
        pred_path = os.path.join(exp23_dir, run_dir, "test_predictions.npz")
        if not os.path.exists(pred_path):
            continue
        with open(result_path) as f:
            r = json.load(f)
        pred = np.load(pred_path)
        predictions[(r["canonicalization"], r["seed"])] = {
            "y_true": pred["y_true"],
            "y_pred": pred["y_pred"],
            "graph_indices": pred["graph_indices"],
        }
    return predictions


def load_exp1_results(exp1_dir):
    """Load results.json from all exp1 runs.

    Returns dict: (canonicalization, hidden_dim) -> [test_rocauc_values].
    """
    results = defaultdict(list)
    for run_dir in os.listdir(exp1_dir):
        result_path = os.path.join(exp1_dir, run_dir, "results.json")
        if not os.path.exists(result_path):
            continue
        with open(result_path) as f:
            r = json.load(f)
        canon = r["canonicalization"]
        hdim = r["hidden_dim"]
        test_auc = r.get("best_test_rocauc")
        if test_auc is not None:
            results[(canon, hdim)].append(test_auc)
    return results


# ── Part 1: Finer Stratification ────────────────────────────────────────────


def _exp1_dir_name(model):
    return "exp1_param_efficiency" if model == "gin" else f"exp1_param_efficiency_{model}"


def _exp23_dir_name(model):
    return "exp23_subset_convergence" if model == "gin" else f"exp23_subset_convergence_{model}"


def part1_stratification(base_dir, output_dir, model="gin"):
    """Stratified analysis by uncanonical eigenvector count."""
    print("\n" + "=" * 70)
    print(f"Part 1: Finer Stratification (4-bin) [{model.upper()}]")
    print("=" * 70)

    audit_dir = os.path.join(base_dir, "test_split_audit")
    exp23_dir = os.path.join(base_dir, _exp23_dir_name(model))

    uncanon_map = load_audit(audit_dir)
    predictions = load_exp23_predictions(exp23_dir)

    print(f"Loaded {len(uncanon_map)} audit entries, {len(predictions)} prediction runs")

    # Build bin membership: bin_label -> set of graph indices
    bin_indices = {}
    for label, lo, hi in BINS:
        bin_indices[label] = {idx for idx, n in uncanon_map.items() if lo <= n <= hi}
        print(f"  Bin '{label}': {len(bin_indices[label])} graphs")

    # Compute per-bin, per-method AUC
    # results_table[bin_label][canon] = (mean, std)
    results_table = defaultdict(dict)

    for bin_label, b_indices in bin_indices.items():
        for canon in CANONICALIZATIONS:
            seed_aucs = []
            for seed in SEEDS:
                key = (canon, seed)
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

    # ── Save CSV ──
    csv_path = os.path.join(output_dir, "stratified_4bin.csv")
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

    # ── Bar chart ──
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
    ax.set_title(
        f"Stratified Test ROC-AUC by Uncanonical Eigenvector Count\n"
        f"(ogbg-moltox21, {model.upper()}, h=256, 5 seeds)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{bl}\n({len(bin_indices[bl])} graphs)" for bl in bin_labels])
    ax.set_xlabel("Number of uncanonical eigenvectors")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    pdf_path = os.path.join(output_dir, "stratified_4bin.pdf")
    fig.savefig(pdf_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {pdf_path}")


# ── Part 2: H-dim Behavior ──────────────────────────────────────────────────


def part2_hdim_analysis(base_dir, output_dir, hdims=None, suffix="", model="gin"):
    """Hidden-dim vs ROC-AUC analysis from exp1 data."""
    label = "Extended " if suffix else ""
    print(f"\n{'=' * 70}")
    print(f"Part 2: {label}H-dim Behavior Investigation [{model.upper()}]")
    print("=" * 70)

    exp1_dir = os.path.join(base_dir, _exp1_dir_name(model))
    results = load_exp1_results(exp1_dir)

    if hdims is None:
        hdims = EXISTING_HDIMS

    # ── Line plot ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for canon in CANONICALIZATIONS:
        means = []
        stds = []
        valid_hdims = []
        for h in hdims:
            vals = results.get((canon, h), [])
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                valid_hdims.append(h)
        if not valid_hdims:
            continue
        means = np.array(means)
        stds = np.array(stds)
        valid_hdims = np.array(valid_hdims)
        ax.plot(
            valid_hdims,
            means,
            label=CANON_LABELS[canon],
            color=CANON_COLORS[canon],
            linewidth=1.5,
            marker="o",
            markersize=4,
        )
        ax.fill_between(
            valid_hdims,
            means - stds,
            means + stds,
            alpha=0.15,
            color=CANON_COLORS[canon],
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(hdims)
    ax.set_xticklabels([str(h) for h in hdims])
    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title(f"Hidden Dimension vs Test ROC-AUC\n(ogbg-moltox21, {model.upper()}, 5 seeds)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pdf_path = os.path.join(output_dir, f"hdim_vs_rocauc{suffix}.pdf")
    fig.savefig(pdf_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {pdf_path}")

    # ── Rank table ──
    rank_data = {}  # hdim -> [(canon, mean_auc)]
    for h in hdims:
        entries = []
        for canon in CANONICALIZATIONS:
            vals = results.get((canon, h), [])
            if vals:
                entries.append((canon, np.mean(vals)))
        entries.sort(key=lambda x: -x[1])
        rank_data[h] = entries

    csv_path = os.path.join(output_dir, f"hdim_rank_table{suffix}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hidden_dim", "rank", "method", "mean_rocauc"])
        for h in hdims:
            for rank, (canon, auc) in enumerate(rank_data[h], 1):
                writer.writerow([h, rank, CANON_LABELS[canon], f"{auc:.4f}"])
    print(f"  Saved {csv_path}")

    # ── Textual analysis ──
    txt_path = os.path.join(output_dir, f"hdim_analysis{suffix}.txt")
    with open(txt_path, "w") as f:
        f.write(f"H-dim Behavior Analysis{' (Extended)' if suffix else ''}\n")
        f.write("=" * 60 + "\n\n")

        # Rank summary
        f.write("Method Rankings by Hidden Dimension\n")
        f.write("-" * 40 + "\n")
        for h in hdims:
            ranking_str = ", ".join(
                f"{rank}. {CANON_LABELS[c]} ({auc:.4f})"
                for rank, (c, auc) in enumerate(rank_data[h], 1)
            )
            f.write(f"  h={h:>3d}: {ranking_str}\n")
        f.write("\n")

        # MAP vs OAP analysis
        f.write("MAP vs OAP Crossover Analysis\n")
        f.write("-" * 40 + "\n")
        map_vals = {h: np.mean(results.get(("map", h), [0])) for h in hdims}
        oap_vals = {h: np.mean(results.get(("oap", h), [0])) for h in hdims}
        for h in hdims:
            diff = map_vals[h] - oap_vals[h]
            leader = "MAP" if diff > 0 else "OAP"
            f.write(
                f"  h={h:>3d}: MAP={map_vals[h]:.4f}, OAP={oap_vals[h]:.4f}"
                f" -> {leader} by {abs(diff):.4f}\n"
            )
        f.write(
            "\nHypothesis: MAP and OAP differ only for eigenspaces with multiplicity > 1.\n"
            "At low hidden dims, the model may exploit MAP's axis-aligned projections\n"
            "which provide a simpler inductive bias. At higher dims, OAP's orthogonal\n"
            "frame approach gains an advantage as the model has enough capacity to\n"
            "leverage the richer representation.\n\n"
        )

        # Random augmented analysis
        f.write("Random (augmented) Crossover Analysis\n")
        f.write("-" * 40 + "\n")
        ra_vals = {h: np.mean(results.get(("random_augmented", h), [0])) for h in hdims}
        for h in hdims:
            rank = next(
                (r for r, (c, _) in enumerate(rank_data[h], 1) if c == "random_augmented"),
                None,
            )
            f.write(f"  h={h:>3d}: ROC-AUC={ra_vals[h]:.4f}, rank={rank}\n")
        f.write(
            "\nHypothesis: Random augmentation forces the model to learn sign-invariant\n"
            "representations from data. At low capacity, this is a disadvantage (too\n"
            "many parameters spent on invariance). Above a capacity threshold, the\n"
            "learned invariance generalizes better than fixed heuristic canonicalization.\n\n"
        )

        # Spielman profile
        f.write("Spielman Performance Profile\n")
        f.write("-" * 40 + "\n")
        sp_vals = {h: np.mean(results.get(("spielman", h), [0])) for h in hdims}
        for h in hdims:
            rank = next(
                (r for r, (c, _) in enumerate(rank_data[h], 1) if c == "spielman"),
                None,
            )
            f.write(f"  h={h:>3d}: ROC-AUC={sp_vals[h]:.4f}, rank={rank}\n")
        f.write(
            "\nObservation: Spielman consistently underperforms other deterministic methods.\n"
            "This may be because Spielman canonicalization is designed for graph-level\n"
            "isomorphism testing, not for preserving spectral locality that GNN message\n"
            "passing relies on.\n"
        )

    print(f"  Saved {txt_path}")

    return results


# ── Part 3: Low H-dim Training ──────────────────────────────────────────────


def part3_low_hdim_training(base_dir, output_dir, data_dir="data", model="gin"):
    """Train at low hidden dims and re-run extended analysis."""
    print(f"\n{'=' * 70}")
    print(f"Part 3: Low H-dim Training Runs [{model.upper()}]")
    print("=" * 70)

    exp1_dir = os.path.join(base_dir, _exp1_dir_name(model))

    # Training config matching existing exp1 runs
    config = {
        "dataset": "ogbg-moltox21",
        "num_layers": 5,
        "n_eigs": 8,
        "epochs": 200,
        "batch_size": 32,
        "lr": 1e-3,
    }

    commands = []
    for canon, hdim, seed in itertools.product(CANONICALIZATIONS, LOW_HDIMS, SEEDS):
        save_dir = os.path.join(exp1_dir, f"{canon}_h{hdim}_s{seed}")
        # Skip if already completed
        if os.path.exists(os.path.join(save_dir, "results.json")):
            print(f"  Skipping {canon}_h{hdim}_s{seed} (already exists)")
            continue
        cmd = [
            sys.executable,
            "scripts/train_molecular.py",
            "--dataset",
            config["dataset"],
            "--data-dir",
            data_dir,
            "--canonicalization",
            canon,
            "--hidden-dim",
            str(hdim),
            "--num-layers",
            str(config["num_layers"]),
            "--n-eigs",
            str(config["n_eigs"]),
            "--epochs",
            str(config["epochs"]),
            "--batch-size",
            str(config["batch_size"]),
            "--lr",
            str(config["lr"]),
            "--seed",
            str(seed),
            "--save-dir",
            save_dir,
            "--model",
            model,
        ]
        commands.append((f"{canon}_h{hdim}_s{seed}", cmd))

    n_skipped = 6 * len(LOW_HDIMS) * 5 - len(commands)
    print(f"  {len(commands)} runs to launch ({n_skipped} already exist)")

    for i, (name, cmd) in enumerate(commands):
        cmd_str = " ".join(cmd)
        print(f"\n{'=' * 70}")
        print(f"[{i + 1}/{len(commands)}] {name}")
        print(f"  {cmd_str}")
        print(f"{'=' * 70}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  WARNING: {name} failed with return code {result.returncode}")

    # Re-run Part 2 with extended h-dims
    print("\nRunning extended h-dim analysis...")
    part2_hdim_analysis(base_dir, output_dir, hdims=ALL_HDIMS, suffix="_extended", model=model)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Ablation analysis for ogbg-moltox21 canonicalization experiments"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="results/canonicalization_experiments",
        help="Base directory for experiment results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/canonicalization_experiments/moltox21_ablation",
        help="Output directory for ablation results",
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
        help="Run Parts 1+2 only (no new training)",
    )
    parser.add_argument(
        "--part3-only",
        action="store_true",
        help="Run Part 3 only (training + extended analysis)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gin",
        choices=["gin", "gcn"],
        help="Model backbone (default: gin)",
    )
    args = parser.parse_args()

    # Adjust output dir for non-gin models
    if (
        args.model != "gin"
        and args.output_dir == "results/canonicalization_experiments/moltox21_ablation"
    ):
        args.output_dir = f"results/canonicalization_experiments/moltox21_ablation_{args.model}"

    os.makedirs(args.output_dir, exist_ok=True)

    if args.skip_training and args.part3_only:
        print("ERROR: --skip-training and --part3-only are mutually exclusive")
        sys.exit(1)

    if not args.part3_only:
        part1_stratification(args.base_dir, args.output_dir, model=args.model)
        part2_hdim_analysis(args.base_dir, args.output_dir, model=args.model)

    if not args.skip_training:
        part3_low_hdim_training(args.base_dir, args.output_dir, args.data_dir, model=args.model)


if __name__ == "__main__":
    main()
