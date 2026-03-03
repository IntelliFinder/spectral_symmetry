#!/usr/bin/env python
"""Analyze canonicalization experiment results.

Reads experiment results directories and generates:
  - Experiment 1: parameter efficiency curves (test metric vs #params)
  - Experiment 2: subset accuracy tables (metric per canonicalization x subset)
  - Experiment 3: convergence learning curves (metric vs epoch)

Usage
-----
    python scripts/analyze_canonicalization_experiments.py \
        --results-dir results/canonicalization_experiments
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

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


def load_experiment_results(results_dir, experiment_subdir):
    """Load all results.json files from experiment subdirectory.

    Returns dict: {(canonicalization, hidden_dim, seed): results_dict}
    """
    exp_dir = os.path.join(results_dir, experiment_subdir)
    if not os.path.isdir(exp_dir):
        print(f"  Directory not found: {exp_dir}")
        return {}

    results = {}
    for run_dir in sorted(os.listdir(exp_dir)):
        json_path = os.path.join(exp_dir, run_dir, "results.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path) as f:
            r = json.load(f)
        key = (r["canonicalization"], r.get("hidden_dim", 256), r["seed"])
        r["_save_dir"] = os.path.join(exp_dir, run_dir)
        results[key] = r
        # Also load epoch log if present
        log_path = os.path.join(exp_dir, run_dir, "epoch_log.json")
        if os.path.exists(log_path):
            with open(log_path) as f:
                r["_epoch_log"] = json.load(f)
    return results


def load_audit_csv(results_dir):
    """Load test-split audit CSV.

    Returns dict: {graph_idx: row_dict}
    """
    audit_dir = os.path.join(results_dir, "test_split_audit")
    csv_path = os.path.join(audit_dir, "audit_details.csv")
    if not os.path.exists(csv_path):
        print(f"  Audit CSV not found: {csv_path}")
        return {}

    audit = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["graph_idx"])
            audit[idx] = row
    return audit


# ---------------------------------------------------------------------------
# Experiment 1: Parameter Efficiency
# ---------------------------------------------------------------------------


def analyze_param_efficiency(results, metric_key, output_dir):
    """Parameter efficiency analysis: metric vs #params by canonicalization."""
    os.makedirs(output_dir, exist_ok=True)

    # Group by (canon, hidden_dim) -> list of metric values
    grouped = defaultdict(list)
    param_counts = {}
    for (canon, hdim, seed), r in results.items():
        metric = r.get(metric_key)
        if metric is None:
            continue
        grouped[(canon, hdim)].append(metric)
        param_counts[(canon, hdim)] = r.get("n_params", 0)

    if not grouped:
        print("  No Experiment 1 results found.")
        return

    # Plot: metric vs #params
    fig, ax = plt.subplots(figsize=(10, 6))
    hidden_dims = sorted(set(h for (_, h) in grouped.keys()))

    for canon in CANONICALIZATIONS:
        x_vals, y_means, y_stds = [], [], []
        for hdim in hidden_dims:
            key = (canon, hdim)
            if key not in grouped:
                continue
            metrics = grouped[key]
            n_params = param_counts.get(key, hdim * 10)
            x_vals.append(n_params)
            y_means.append(np.mean(metrics))
            y_stds.append(np.std(metrics))

        if not x_vals:
            continue

        x_vals = np.array(x_vals)
        y_means = np.array(y_means)
        y_stds = np.array(y_stds)

        ax.plot(
            x_vals,
            y_means,
            "o-",
            label=CANON_LABELS.get(canon, canon),
            color=CANON_COLORS.get(canon, "gray"),
        )
        ax.fill_between(
            x_vals,
            y_means - y_stds,
            y_means + y_stds,
            alpha=0.15,
            color=CANON_COLORS.get(canon, "gray"),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel(metric_key.replace("_", " ").title())
    ax.set_title("Parameter Efficiency: Canonicalization Impact")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "param_efficiency.pdf"))
    plt.close(fig)

    # Table
    table_path = os.path.join(output_dir, "param_efficiency_table.csv")
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["canonicalization"] + [f"h={h}" for h in hidden_dims]
        writer.writerow(header)
        for canon in CANONICALIZATIONS:
            row = [canon]
            for hdim in hidden_dims:
                key = (canon, hdim)
                if key in grouped:
                    m = np.mean(grouped[key])
                    s = np.std(grouped[key])
                    row.append(f"{m:.4f} +/- {s:.4f}")
                else:
                    row.append("")
            writer.writerow(row)

    # Efficiency threshold: smallest hdim achieving >=95% of best
    print("\n  Efficiency thresholds (95% of best):")
    for canon in CANONICALIZATIONS:
        all_metrics = []
        for hdim in hidden_dims:
            key = (canon, hdim)
            if key in grouped:
                all_metrics.extend(grouped[key])
        if not all_metrics:
            continue
        best = max(np.mean(grouped[(canon, h)]) for h in hidden_dims if (canon, h) in grouped)
        threshold = 0.95 * best
        for hdim in hidden_dims:
            key = (canon, hdim)
            if key in grouped and np.mean(grouped[key]) >= threshold:
                print(f"    {canon:20s}: h={hdim} ({np.mean(grouped[key]):.4f} >= {threshold:.4f})")
                break

    # Statistical tests: paired t-test Spielman vs each, per hidden_dim
    print("\n  Paired t-tests (Spielman vs others, per hidden_dim):")
    for hdim in hidden_dims:
        spielman_key = ("spielman", hdim)
        if spielman_key not in grouped:
            continue
        spielman_vals = grouped[spielman_key]
        for canon in CANONICALIZATIONS:
            if canon == "spielman":
                continue
            other_key = (canon, hdim)
            if other_key not in grouped:
                continue
            other_vals = grouped[other_key]
            n_compare = min(len(spielman_vals), len(other_vals))
            if n_compare < 2:
                continue
            t_stat, p_val = stats.ttest_ind(spielman_vals[:n_compare], other_vals[:n_compare])
            sig = "*" if p_val < 0.05 else ""
            print(f"    h={hdim:3d} Spielman vs {canon:20s}: t={t_stat:+.3f} p={p_val:.4f} {sig}")


# ---------------------------------------------------------------------------
# Experiment 2: Subset Test Accuracy
# ---------------------------------------------------------------------------


def analyze_subset_accuracy(results, audit, metric_key, output_dir):
    """Subset accuracy analysis: metric per canonicalization x taxonomy subset."""
    os.makedirs(output_dir, exist_ok=True)

    if not audit:
        print("  No audit data — skipping subset analysis.")
        return

    # Define subsets
    subsets = {
        "S1: Easy-canonical": lambda row: row["taxonomy"] == "easy-canonical",
        "S2: Joint-canonical": lambda row: row["taxonomy"] == "joint-canonical",
        "S3: Non-simple": lambda row: row["taxonomy"] == "non-simple",
        "S4: High-uncanon": lambda row: (
            int(row.get("num_spielman_rescued", 0)) + int(row.get("num_degenerate_uncanon", 0)) >= 4
        ),
        "S5: Zero-uncanon": lambda row: (
            (
                int(row.get("num_spielman_rescued", 0))
                + int(row.get("num_degenerate_uncanon", 0))
                + int(row.get("num_truly_hard", 0))
            )
            == 0
        ),
    }

    # Classify audit graphs into subsets
    subset_indices = {}
    for name, condition in subsets.items():
        subset_indices[name] = set()
        for idx, row in audit.items():
            if condition(row):
                subset_indices[name].add(idx)

    print("\n  Subset sizes:")
    for name, indices in subset_indices.items():
        print(f"    {name}: {len(indices)} graphs")

    # For each (canon, seed), load test predictions
    per_run_predictions = {}
    for (canon, hdim, seed), r in results.items():
        # Find predictions file
        pred_path = None
        save_dir = r.get("_save_dir")
        if save_dir:
            pred_path = os.path.join(save_dir, "test_predictions.npz")

        # Try to find it in the results dir structure
        if pred_path is None or not os.path.exists(pred_path):
            continue

        pred_data = np.load(pred_path)
        per_run_predictions[(canon, seed)] = {
            "y_true": pred_data["y_true"],
            "y_pred": pred_data["y_pred"],
            "graph_indices": pred_data["graph_indices"],
        }

    if not per_run_predictions:
        print("  No prediction files found — skipping per-graph subset analysis.")
        print("  (Run experiments with --save-predictions to enable this.)")
        return

    # Compute per-subset metrics
    print("\n  Subset metrics:")
    for subset_name in subsets:
        print(f"\n    {subset_name}:")
        for canon in CANONICALIZATIONS:
            metrics = []
            for seed in range(5):
                key = (canon, seed)
                if key not in per_run_predictions:
                    continue
                pred = per_run_predictions[key]
                # Filter to subset
                mask = np.array([i in subset_indices[subset_name] for i in pred["graph_indices"]])
                if mask.sum() == 0:
                    continue
                # Compute metric on subset (simplified — use per-task AUC average)
                y_t = pred["y_true"][mask]
                y_p = pred["y_pred"][mask]
                from sklearn.metrics import roc_auc_score

                try:
                    valid = ~np.isnan(y_t)
                    if valid.sum() > 0:
                        auc = roc_auc_score(y_t[valid].flatten(), y_p[valid].flatten())
                        metrics.append(auc)
                except Exception:
                    pass
            if metrics:
                print(f"      {canon:20s}: {np.mean(metrics):.4f} +/- {np.std(metrics):.4f}")


# ---------------------------------------------------------------------------
# Experiment 3: Convergence Speed
# ---------------------------------------------------------------------------


def analyze_convergence(results, metric_key, output_dir):
    """Convergence analysis: learning curves and derived metrics."""
    os.makedirs(output_dir, exist_ok=True)

    # Collect epoch logs by canonicalization
    logs_by_canon = defaultdict(list)
    for (canon, hdim, seed), r in results.items():
        if "_epoch_log" not in r:
            continue
        logs_by_canon[canon].append(r["_epoch_log"])

    if not logs_by_canon:
        print("  No epoch logs found — skipping convergence analysis.")
        return

    # Plot: learning curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for canon in CANONICALIZATIONS:
        if canon not in logs_by_canon:
            continue
        logs = logs_by_canon[canon]

        # Align by epoch
        max_epochs = max(len(log) for log in logs)
        test_curves = np.full((len(logs), max_epochs), np.nan)
        loss_curves = np.full((len(logs), max_epochs), np.nan)

        for i, log in enumerate(logs):
            for entry in log:
                ep = entry["epoch"] - 1
                if ep < max_epochs:
                    test_curves[i, ep] = entry["test_metric"]
                    loss_curves[i, ep] = entry["train_loss"]

        epochs = np.arange(1, max_epochs + 1)
        mean_test = np.nanmean(test_curves, axis=0)
        std_test = np.nanstd(test_curves, axis=0)
        mean_loss = np.nanmean(loss_curves, axis=0)
        std_loss = np.nanstd(loss_curves, axis=0)

        color = CANON_COLORS.get(canon, "gray")
        label = CANON_LABELS.get(canon, canon)

        axes[0].plot(epochs, mean_test, label=label, color=color)
        axes[0].fill_between(
            epochs, mean_test - std_test, mean_test + std_test, alpha=0.15, color=color
        )

        axes[1].plot(epochs, mean_loss, label=label, color=color)
        axes[1].fill_between(
            epochs, mean_loss - std_loss, mean_loss + std_loss, alpha=0.15, color=color
        )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Test Metric")
    axes[0].set_title("Convergence: Test Metric vs Epoch")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Train Loss")
    axes[1].set_title("Convergence: Train Loss vs Epoch")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "convergence_curves.pdf"))
    plt.close(fig)

    # Derived metrics: AuLC, epochs-to-X%
    print("\n  Convergence metrics:")
    print(f"    {'Method':20s} | {'AuLC':>8s} | {'Ep→90%':>6s} | {'Ep→95%':>6s} | {'Ep→99%':>6s}")
    print(f"    {'-' * 60}")

    aulc_by_canon = {}
    for canon in CANONICALIZATIONS:
        if canon not in logs_by_canon:
            continue
        logs = logs_by_canon[canon]
        aulcs = []
        ep90s, ep95s, ep99s = [], [], []

        for log in logs:
            test_vals = [e["test_metric"] for e in log]
            if not test_vals:
                continue
            best = max(test_vals)
            n_ep = len(test_vals)
            # AuLC: normalized area under learning curve
            aulc = sum(test_vals) / (best * n_ep) if best > 0 else 0
            aulcs.append(aulc)

            # Epochs to X%
            for threshold, dest in [(0.90, ep90s), (0.95, ep95s), (0.99, ep99s)]:
                target = threshold * best
                found = False
                for i, v in enumerate(test_vals):
                    if v >= target:
                        dest.append(i + 1)
                        found = True
                        break
                if not found:
                    dest.append(n_ep)

        aulc_by_canon[canon] = aulcs
        label = CANON_LABELS.get(canon, canon)
        print(
            f"    {label:20s} | {np.mean(aulcs):.4f}  "
            f"| {np.mean(ep90s):5.1f}  "
            f"| {np.mean(ep95s):5.1f}  "
            f"| {np.mean(ep99s):5.1f}"
        )

    # AuLC bar chart
    if aulc_by_canon:
        fig, ax = plt.subplots(figsize=(8, 4))
        canons = [c for c in CANONICALIZATIONS if c in aulc_by_canon]
        means = [np.mean(aulc_by_canon[c]) for c in canons]
        stds = [np.std(aulc_by_canon[c]) for c in canons]
        colors = [CANON_COLORS.get(c, "gray") for c in canons]
        labels = [CANON_LABELS.get(c, c) for c in canons]

        ax.bar(range(len(canons)), means, yerr=stds, color=colors, capsize=4)
        ax.set_xticks(range(len(canons)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("AuLC (higher = faster convergence)")
        ax.set_title("Area under Learning Curve")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "aulc_bar.pdf"))
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Analyze canonicalization experiment results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/canonicalization_experiments",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbg-moltox21",
        choices=["ogbg-moltox21", "ogbg-molpcba"],
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots/tables (default: results_dir/analysis)",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "analysis")
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine metric key
    if "moltox21" in args.dataset:
        metric_key = "best_test_rocauc"
    else:
        metric_key = "best_test_ap"

    print(f"Analyzing results in {args.results_dir}")
    print(f"Dataset: {args.dataset}, metric: {metric_key}")

    # Load results
    print("\nExperiment 1: Parameter Efficiency")
    exp1_results = load_experiment_results(args.results_dir, "exp1_param_efficiency")
    print(f"  Loaded {len(exp1_results)} runs")
    if exp1_results:
        analyze_param_efficiency(
            exp1_results,
            metric_key,
            os.path.join(args.output_dir, "exp1"),
        )

    print("\nExperiments 2+3: Subset Accuracy & Convergence")
    exp23_results = load_experiment_results(args.results_dir, "exp23_subset_convergence")
    print(f"  Loaded {len(exp23_results)} runs")

    audit = load_audit_csv(args.results_dir)
    print(f"  Audit data: {len(audit)} graphs")

    if exp23_results:
        analyze_subset_accuracy(
            exp23_results,
            audit,
            metric_key,
            os.path.join(args.output_dir, "exp2"),
        )
        analyze_convergence(
            exp23_results,
            metric_key,
            os.path.join(args.output_dir, "exp3"),
        )

    print(f"\nAnalysis saved to {args.output_dir}")


if __name__ == "__main__":
    main()
