#!/usr/bin/env python
"""Eigenvalue vs eigenvector ablation on ogbg-moltox21.

Compares three PE types (eigvec, eigval, both) across different numbers of
eigenvalues (k) and hidden dimensions. Tests whether eigenvalues alone carry
useful positional information and how they interact with canonicalization.

Usage:
    # Dry run
    python scripts/ablation_eigval_moltox21.py --dry-run

    # Run training
    python scripts/ablation_eigval_moltox21.py

    # Analysis only (after training is complete)
    python scripts/ablation_eigval_moltox21.py --analysis-only

    # Specific model
    python scripts/ablation_eigval_moltox21.py --model gcn
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PE_TYPES = ["eigvec", "eigval", "both"]
K_VALUES = [1, 2, 4, 8, 12, 16]
CANONICALIZATIONS = ["maxabs", "none"]  # eigval is canon-independent; test both for eigvec
HIDDEN_DIMS = [32, 256]
SEEDS = [0, 1, 2]

PE_COLORS = {"eigvec": "#1f77b4", "eigval": "#ff7f0e", "both": "#2ca02c"}
PE_LABELS = {"eigvec": "Eigenvectors", "eigval": "Eigenvalues", "both": "Both"}

BASE_DIR = "results/canonicalization_experiments"


def save_dir_for(model, pe_type, canon, k, h, seed):
    return os.path.join(
        BASE_DIR, f"eigval_ablation_{model}", f"{pe_type}_{canon}_k{k}_h{h}_s{seed}"
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_commands(model, dry_run=False):
    """Build all training commands."""
    commands = []
    skipped = 0

    for pe_type in PE_TYPES:
        # For eigval, canonicalization doesn't matter (eigenvalues are the same).
        # Use "none" to avoid unnecessary preprocessing.
        canons = ["none"] if pe_type == "eigval" else CANONICALIZATIONS
        for canon in canons:
            for k in K_VALUES:
                for h in HIDDEN_DIMS:
                    for seed in SEEDS:
                        sd = save_dir_for(model, pe_type, canon, k, h, seed)
                        if os.path.exists(os.path.join(sd, "results.json")):
                            skipped += 1
                            continue
                        cmd = [
                            sys.executable,
                            "scripts/train_molecular.py",
                            "--dataset",
                            "ogbg-moltox21",
                            "--model",
                            model,
                            "--canonicalization",
                            canon,
                            "--n-eigs",
                            str(k),
                            "--pe-type",
                            pe_type,
                            "--hidden-dim",
                            str(h),
                            "--num-layers",
                            "5",
                            "--epochs",
                            "200",
                            "--batch-size",
                            "32",
                            "--lr",
                            "1e-3",
                            "--seed",
                            str(seed),
                            "--save-dir",
                            sd,
                        ]
                        name = f"{pe_type}/{canon}_k{k}_h{h}_s{seed}"
                        commands.append((name, cmd))

    print(f"Commands: {len(commands)} to run, {skipped} skipped (already done)")
    return commands


def run_training(commands, dry_run=False):
    """Run training commands sequentially."""
    for i, (name, cmd) in enumerate(commands):
        print(f"[{i + 1}/{len(commands)}] {name}")
        if dry_run:
            print(f"  CMD: {' '.join(cmd)}")
            continue

        save_dir = cmd[cmd.index("--save-dir") + 1]
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, "train.log")
        with open(log_path, "w") as log_f:
            result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            print(f"  WARNING: {name} failed (rc={result.returncode})")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def load_results(model):
    """Load all completed results."""
    results = []
    ablation_dir = os.path.join(BASE_DIR, f"eigval_ablation_{model}")
    if not os.path.isdir(ablation_dir):
        return results

    for entry in os.listdir(ablation_dir):
        rpath = os.path.join(ablation_dir, entry, "results.json")
        if os.path.exists(rpath):
            with open(rpath) as f:
                r = json.load(f)
            # Parse directory name: {pe_type}_{canon}_k{k}_h{h}_s{seed}
            parts = entry.split("_")
            pe_type = parts[0]
            r["pe_type"] = pe_type
            results.append(r)
    return results


def plot_rocauc_vs_k(results, model, out_dir):
    """Plot ROC-AUC vs number of eigenvalues for each PE type."""
    os.makedirs(out_dir, exist_ok=True)

    for h in HIDDEN_DIMS:
        fig, ax = plt.subplots(figsize=(8, 5))

        for pe_type in PE_TYPES:
            # For eigval, canonicalization doesn't matter; for eigvec, use maxabs
            if pe_type == "eigval":
                subset = [
                    r for r in results if r.get("pe_type") == pe_type and r["hidden_dim"] == h
                ]
            else:
                subset = [
                    r
                    for r in results
                    if r.get("pe_type") == pe_type
                    and r["hidden_dim"] == h
                    and r["canonicalization"] == "maxabs"
                ]

            if not subset:
                continue

            k_to_scores = defaultdict(list)
            for r in subset:
                metric = r.get("best_test_rocauc", r.get("best_test_ap", 0))
                k_to_scores[r["n_eigs"]].append(metric)

            ks = sorted(k_to_scores.keys())
            means = [np.mean(k_to_scores[k]) for k in ks]
            stds = [np.std(k_to_scores[k]) for k in ks]

            color = PE_COLORS[pe_type]
            label = PE_LABELS[pe_type]
            ax.plot(ks, means, "o-", color=color, label=label, linewidth=2, markersize=6)
            ax.fill_between(
                ks,
                np.array(means) - np.array(stds),
                np.array(means) + np.array(stds),
                alpha=0.15,
                color=color,
            )

        ax.set_xlabel("Number of Eigenvalues/Eigenvectors (k)", fontsize=12)
        ax.set_ylabel("Test ROC-AUC", fontsize=12)
        ax.set_title(f"PE Type Comparison (ogbg-moltox21, {model.upper()}, h={h})", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"pe_type_vs_k_h{h}.pdf"), dpi=150)
        plt.close(fig)
        print(f"  Saved pe_type_vs_k_h{h}.pdf")


def plot_canon_effect_on_eigvec(results, model, out_dir):
    """Compare canonicalization effect: maxabs vs none for eigvec PE type."""
    os.makedirs(out_dir, exist_ok=True)

    for h in HIDDEN_DIMS:
        fig, ax = plt.subplots(figsize=(8, 5))

        for canon in CANONICALIZATIONS:
            subset = [
                r
                for r in results
                if r.get("pe_type") == "eigvec"
                and r["hidden_dim"] == h
                and r["canonicalization"] == canon
            ]
            if not subset:
                continue

            k_to_scores = defaultdict(list)
            for r in subset:
                metric = r.get("best_test_rocauc", r.get("best_test_ap", 0))
                k_to_scores[r["n_eigs"]].append(metric)

            ks = sorted(k_to_scores.keys())
            means = [np.mean(k_to_scores[k]) for k in ks]
            stds = [np.std(k_to_scores[k]) for k in ks]

            label = f"Eigvec ({canon})"
            ax.plot(ks, means, "o-", label=label, linewidth=2, markersize=6)
            ax.fill_between(
                ks,
                np.array(means) - np.array(stds),
                np.array(means) + np.array(stds),
                alpha=0.15,
            )

        ax.set_xlabel("Number of Eigenvectors (k)", fontsize=12)
        ax.set_ylabel("Test ROC-AUC", fontsize=12)
        ax.set_title(
            f"Canonicalization Effect on Eigenvectors ({model.upper()}, h={h})", fontsize=13
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"canon_effect_h{h}.pdf"), dpi=150)
        plt.close(fig)
        print(f"  Saved canon_effect_h{h}.pdf")


def write_summary_csv(results, model, out_dir):
    """Write summary CSV with mean/std per configuration."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    for pe_type in PE_TYPES:
        canons = ["none"] if pe_type == "eigval" else CANONICALIZATIONS
        for canon in canons:
            for k in K_VALUES:
                for h in HIDDEN_DIMS:
                    subset = [
                        r
                        for r in results
                        if r.get("pe_type") == pe_type
                        and r["canonicalization"] == canon
                        and r["n_eigs"] == k
                        and r["hidden_dim"] == h
                    ]
                    if not subset:
                        continue
                    scores = [r.get("best_test_rocauc", r.get("best_test_ap", 0)) for r in subset]
                    rows.append(
                        {
                            "pe_type": pe_type,
                            "canon": canon,
                            "k": k,
                            "h": h,
                            "n_runs": len(scores),
                            "mean": f"{np.mean(scores):.4f}",
                            "std": f"{np.std(scores):.4f}",
                        }
                    )

    csv_path = os.path.join(out_dir, "eigval_ablation_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["pe_type", "canon", "k", "h", "n_runs", "mean", "std"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved eigval_ablation_summary.csv ({len(rows)} rows)")


def run_analysis(model, out_dir):
    """Run all analysis plots and tables."""
    results = load_results(model)
    print(f"Loaded {len(results)} results for {model}")

    if not results:
        print("No results found. Run training first.")
        return

    plot_rocauc_vs_k(results, model, out_dir)
    plot_canon_effect_on_eigvec(results, model, out_dir)
    write_summary_csv(results, model, out_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Eigenvalue ablation on moltox21")
    parser.add_argument("--model", type=str, default="gin", choices=["gin", "gcn"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.join(BASE_DIR, f"eigval_ablation_{args.model}")

    if args.analysis_only:
        run_analysis(args.model, out_dir)
        return

    # Count total runs
    total = 0
    for pe_type in PE_TYPES:
        canons = 1 if pe_type == "eigval" else len(CANONICALIZATIONS)
        total += canons * len(K_VALUES) * len(HIDDEN_DIMS) * len(SEEDS)
    print(f"Eigenvalue ablation: {total} total runs for {args.model.upper()}")

    commands = build_commands(args.model, args.dry_run)
    run_training(commands, args.dry_run)

    if not args.dry_run:
        print("\n=== Running analysis ===")
        run_analysis(args.model, out_dir)


if __name__ == "__main__":
    main()
