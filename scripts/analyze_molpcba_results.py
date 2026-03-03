#!/usr/bin/env python
"""Analyze ogbg-molpcba canonicalization experiment results.

Reads experiment results directories and generates:
  - Experiment 1: parameter efficiency curves (AP vs hidden_dim)
  - Experiment 2: stratified AP by canonicalizability bin
  - Experiment 3: convergence learning curves (AP vs epoch)
  - Cross-dataset comparison with moltox21 (if results available)

Usage
-----
    python scripts/analyze_molpcba_results.py \\
        --results-dir results/molpcba_canonicalization

    # With cross-dataset comparison
    python scripts/analyze_molpcba_results.py \\
        --results-dir results/molpcba_canonicalization \\
        --moltox21-dir results/canonicalization_experiments
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
from sklearn.metrics import average_precision_score  # noqa: E402

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

HIDDEN_DIMS = [32, 64, 128, 256, 512]
SEEDS = [0, 1, 2, 3, 4]
METRIC_KEY = "best_test_ap"
METRIC_LABEL = "Test AP"

# Bins: (label, min_uncanon, max_uncanon_inclusive)
BINS = [
    ("<=1", 0, 1),
    ("2-3", 2, 3),
    ("4-5", 4, 5),
    ("6+", 6, 999),
]

MIN_SAMPLES = 6  # Minimum samples per task for AP computation


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


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


def load_audit(audit_dir):
    """Load audit_details.csv, return dict: graph_idx -> num_uncanonical."""
    csv_path = os.path.join(audit_dir, "audit_details.csv")
    if not os.path.exists(csv_path):
        print(f"  Audit CSV not found: {csv_path}")
        return {}
    uncanon_map = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["graph_idx"])
            n_eig = int(row.get("num_eigenvectors", 8))
            n_canon = int(row.get("num_individually_canonical", 0))
            uncanon_map[idx] = n_eig - n_canon
    return uncanon_map


def compute_stratified_ap(y_true, y_pred):
    """Compute mean per-task AP with NaN masking.

    Returns float or None if insufficient data.
    """
    task_aps = []
    for t in range(y_true.shape[1]):
        valid = ~np.isnan(y_true[:, t])
        if valid.sum() >= MIN_SAMPLES and len(np.unique(y_true[valid, t])) > 1:
            task_aps.append(average_precision_score(y_true[valid, t], y_pred[valid, t]))
    return np.mean(task_aps) if task_aps else None


def load_predictions(exp_dir):
    """Load test predictions from experiment runs.

    Returns dict: (canonicalization, seed) -> {y_true, y_pred, graph_indices}.
    """
    predictions = {}
    if not os.path.isdir(exp_dir):
        return predictions
    for run_dir in sorted(os.listdir(exp_dir)):
        pred_path = os.path.join(exp_dir, run_dir, "test_predictions.npz")
        result_path = os.path.join(exp_dir, run_dir, "results.json")
        if not os.path.exists(pred_path) or not os.path.exists(result_path):
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


# ---------------------------------------------------------------------------
# Experiment 1: Parameter Efficiency
# ---------------------------------------------------------------------------


def analyze_param_efficiency(results, output_dir):
    """Parameter efficiency analysis: AP vs hidden_dim by canonicalization."""
    os.makedirs(output_dir, exist_ok=True)

    # Group by (canon, hidden_dim) -> list of AP values
    grouped = defaultdict(list)
    param_counts = {}
    for (canon, hdim, _seed), r in results.items():
        metric = r.get(METRIC_KEY)
        if metric is None:
            continue
        grouped[(canon, hdim)].append(metric)
        param_counts[(canon, hdim)] = r.get("n_params", 0)

    if not grouped:
        print("  No Experiment 1 results found.")
        return

    hidden_dims = sorted(set(h for (_, h) in grouped.keys()))

    # Plot: AP vs hidden_dim (log2 scale)
    fig, ax = plt.subplots(figsize=(10, 6))

    for canon in CANONICALIZATIONS:
        x_vals, y_means, y_stds = [], [], []
        for hdim in hidden_dims:
            key = (canon, hdim)
            if key not in grouped:
                continue
            metrics = grouped[key]
            x_vals.append(hdim)
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
            linewidth=1.5,
            markersize=4,
        )
        ax.fill_between(
            x_vals,
            y_means - y_stds,
            y_means + y_stds,
            alpha=0.15,
            color=CANON_COLORS.get(canon, "gray"),
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(hidden_dims)
    ax.set_xticklabels([str(h) for h in hidden_dims])
    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel(METRIC_LABEL)
    ax.set_title(f"Parameter Efficiency: {METRIC_LABEL} vs Hidden Dim\n(ogbg-molpcba, 5 seeds)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "param_efficiency.pdf"), dpi=150)
    plt.close(fig)
    print(f"  Saved {os.path.join(output_dir, 'param_efficiency.pdf')}")

    # CSV summary table
    csv_path = os.path.join(output_dir, "param_efficiency.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["canonicalization"] + [f"h={h}" for h in hidden_dims]
        writer.writerow(header)
        for canon in CANONICALIZATIONS:
            row = [CANON_LABELS[canon]]
            for hdim in hidden_dims:
                key = (canon, hdim)
                if key in grouped:
                    m = np.mean(grouped[key])
                    s = np.std(grouped[key])
                    row.append(f"{m:.4f} +/- {s:.4f}")
                else:
                    row.append("")
            writer.writerow(row)
    print(f"  Saved {csv_path}")

    # Rank table per hidden_dim
    csv_path = os.path.join(output_dir, "rank_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hidden_dim", "rank", "method", "mean_ap"])
        for hdim in hidden_dims:
            entries = []
            for canon in CANONICALIZATIONS:
                key = (canon, hdim)
                if key in grouped:
                    entries.append((canon, np.mean(grouped[key])))
            entries.sort(key=lambda x: -x[1])
            for rank, (canon, ap) in enumerate(entries, 1):
                writer.writerow([hdim, rank, CANON_LABELS[canon], f"{ap:.4f}"])
    print(f"  Saved {csv_path}")

    # Paired t-tests: best method vs each other, per hidden_dim
    print("\n  Paired t-tests (per hidden_dim):")
    for hdim in hidden_dims:
        # Find best method at this hdim
        best_canon = None
        best_mean = -1
        for canon in CANONICALIZATIONS:
            key = (canon, hdim)
            if key in grouped and np.mean(grouped[key]) > best_mean:
                best_mean = np.mean(grouped[key])
                best_canon = canon
        if best_canon is None:
            continue

        for canon in CANONICALIZATIONS:
            if canon == best_canon:
                continue
            other_key = (canon, hdim)
            best_key = (best_canon, hdim)
            if other_key not in grouped or best_key not in grouped:
                continue
            n_compare = min(len(grouped[best_key]), len(grouped[other_key]))
            if n_compare < 2:
                continue
            t_stat, p_val = stats.ttest_ind(
                grouped[best_key][:n_compare], grouped[other_key][:n_compare]
            )
            sig = "*" if p_val < 0.05 else ""
            print(
                f"    h={hdim:3d} {CANON_LABELS[best_canon]} vs "
                f"{CANON_LABELS[canon]:15s}: t={t_stat:+.3f} p={p_val:.4f} {sig}"
            )


# ---------------------------------------------------------------------------
# Experiment 2: Stratified Scores by Canonicalizability
# ---------------------------------------------------------------------------


def analyze_stratified_scores(results_dir, output_dir):
    """Stratified AP by uncanonical eigenvector count (4-bin scheme)."""
    os.makedirs(output_dir, exist_ok=True)

    audit_dir = os.path.join(results_dir, "test_split_audit")
    exp23_dir = os.path.join(results_dir, "exp23_subset_convergence")

    uncanon_map = load_audit(audit_dir)
    if not uncanon_map:
        print("  No audit data — skipping stratified analysis.")
        return

    predictions = load_predictions(exp23_dir)
    if not predictions:
        print("  No prediction files found — skipping stratified analysis.")
        print("  (Run experiments with --save-predictions to enable this.)")
        return

    print(f"  Loaded {len(uncanon_map)} audit entries, {len(predictions)} prediction runs")

    # Build bin membership: bin_label -> set of graph indices
    bin_indices = {}
    for label, lo, hi in BINS:
        bin_indices[label] = {idx for idx, n in uncanon_map.items() if lo <= n <= hi}
        print(f"  Bin '{label}': {len(bin_indices[label])} graphs")

    # Compute per-bin, per-method AP
    results_table = defaultdict(dict)

    for bin_label, b_indices in bin_indices.items():
        for canon in CANONICALIZATIONS:
            seed_aps = []
            for seed in SEEDS:
                key = (canon, seed)
                if key not in predictions:
                    continue
                pred = predictions[key]
                mask = np.array([i in b_indices for i in pred["graph_indices"]])
                if mask.sum() == 0:
                    continue
                ap = compute_stratified_ap(pred["y_true"][mask], pred["y_pred"][mask])
                if ap is not None:
                    seed_aps.append(ap)
            if seed_aps:
                results_table[bin_label][canon] = (np.mean(seed_aps), np.std(seed_aps))

    # Save CSV
    csv_path = os.path.join(output_dir, "stratified_scores.csv")
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

    # Bar chart
    bin_labels = [b[0] for b in BINS]
    n_bins = len(bin_labels)
    n_methods = len(CANONICALIZATIONS)
    width = 0.8 / n_methods
    x = np.arange(n_bins)

    fig, ax = plt.subplots(figsize=(12, 6))
    for j, canon in enumerate(CANONICALIZATIONS):
        means = []
        stds_arr = []
        for bin_label in bin_labels:
            if canon in results_table[bin_label]:
                m, s = results_table[bin_label][canon]
                means.append(m)
                stds_arr.append(s)
            else:
                means.append(0)
                stds_arr.append(0)
        offset = (j - n_methods / 2 + 0.5) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds_arr,
            label=CANON_LABELS[canon],
            color=CANON_COLORS[canon],
            capsize=2,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_ylabel(METRIC_LABEL)
    ax.set_title(
        f"Stratified {METRIC_LABEL} by Uncanonical Eigenvector Count\n"
        f"(ogbg-molpcba, h=256, 5 seeds)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{bl}\n({len(bin_indices[bl])} graphs)" for bl in bin_labels])
    ax.set_xlabel("Number of uncanonical eigenvectors")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    pdf_path = os.path.join(output_dir, "stratified_scores.pdf")
    fig.savefig(pdf_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {pdf_path}")


# ---------------------------------------------------------------------------
# Experiment 3: Convergence Speed
# ---------------------------------------------------------------------------


def analyze_convergence(results, output_dir):
    """Convergence analysis: learning curves and derived metrics."""
    os.makedirs(output_dir, exist_ok=True)

    # Collect epoch logs by canonicalization
    logs_by_canon = defaultdict(list)
    for (canon, _hdim, _seed), r in results.items():
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

        max_epochs = max(len(log) for log in logs)
        test_curves = np.full((len(logs), max_epochs), np.nan)
        loss_curves = np.full((len(logs), max_epochs), np.nan)

        for i, log in enumerate(logs):
            for entry in log:
                ep = entry["epoch"] - 1
                if ep < max_epochs:
                    test_curves[i, ep] = entry.get("test_metric", entry.get("val_metric", np.nan))
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
    axes[0].set_ylabel(METRIC_LABEL)
    axes[0].set_title(f"Convergence: {METRIC_LABEL} vs Epoch (ogbg-molpcba)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Train Loss")
    axes[1].set_title("Convergence: Train Loss vs Epoch (ogbg-molpcba)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "convergence.pdf"), dpi=150)
    plt.close(fig)
    print(f"  Saved {os.path.join(output_dir, 'convergence.pdf')}")

    # Derived metrics: AuLC, epochs-to-X%
    print(
        f"\n  {'Method':20s} | {'AuLC':>8s} | {'Ep->90%':>7s} | {'Ep->95%':>7s} | {'Ep->99%':>7s}"
    )
    print(f"  {'-' * 65}")

    for canon in CANONICALIZATIONS:
        if canon not in logs_by_canon:
            continue
        logs = logs_by_canon[canon]
        aulcs = []
        ep90s, ep95s, ep99s = [], [], []

        for log in logs:
            test_vals = [e.get("test_metric", e.get("val_metric", 0)) for e in log]
            if not test_vals:
                continue
            best = max(test_vals)
            n_ep = len(test_vals)
            aulc = sum(test_vals) / (best * n_ep) if best > 0 else 0
            aulcs.append(aulc)

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

        label = CANON_LABELS.get(canon, canon)
        print(
            f"  {label:20s} | {np.mean(aulcs):.4f}  "
            f"| {np.mean(ep90s):6.1f}  "
            f"| {np.mean(ep95s):6.1f}  "
            f"| {np.mean(ep99s):6.1f}"
        )


# ---------------------------------------------------------------------------
# Cross-dataset comparison
# ---------------------------------------------------------------------------


def cross_dataset_comparison(molpcba_results, moltox21_dir, output_dir):
    """Compare method rankings between molpcba and moltox21."""
    os.makedirs(output_dir, exist_ok=True)

    # Load moltox21 exp1 results
    moltox21_exp1_dir = os.path.join(moltox21_dir, "exp1_param_efficiency")
    if not os.path.isdir(moltox21_exp1_dir):
        print(f"  moltox21 exp1 dir not found: {moltox21_exp1_dir}")
        return

    moltox21_results = defaultdict(list)
    for run_dir in sorted(os.listdir(moltox21_exp1_dir)):
        json_path = os.path.join(moltox21_exp1_dir, run_dir, "results.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path) as f:
            r = json.load(f)
        test_auc = r.get("best_test_rocauc")
        if test_auc is not None:
            moltox21_results[r["canonicalization"]].append(test_auc)

    if not moltox21_results:
        print("  No moltox21 results found.")
        return

    # molpcba: aggregate across all hidden dims
    molpcba_by_canon = defaultdict(list)
    for (canon, _hdim, _seed), r in molpcba_results.items():
        ap = r.get(METRIC_KEY)
        if ap is not None:
            molpcba_by_canon[canon].append(ap)

    # Build ranking table
    moltox21_ranking = sorted(
        [(c, np.mean(vals)) for c, vals in moltox21_results.items()],
        key=lambda x: -x[1],
    )
    molpcba_ranking = sorted(
        [(c, np.mean(vals)) for c, vals in molpcba_by_canon.items()],
        key=lambda x: -x[1],
    )

    csv_path = os.path.join(output_dir, "cross_dataset_ranks.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "moltox21_rank",
                "moltox21_rocauc",
                "molpcba_rank",
                "molpcba_ap",
            ]
        )

        moltox21_rank_map = {c: i + 1 for i, (c, _) in enumerate(moltox21_ranking)}
        moltox21_val_map = {c: v for c, v in moltox21_ranking}
        molpcba_rank_map = {c: i + 1 for i, (c, _) in enumerate(molpcba_ranking)}
        molpcba_val_map = {c: v for c, v in molpcba_ranking}

        for canon in CANONICALIZATIONS:
            writer.writerow(
                [
                    CANON_LABELS[canon],
                    moltox21_rank_map.get(canon, "N/A"),
                    f"{moltox21_val_map.get(canon, 0):.4f}" if canon in moltox21_val_map else "N/A",
                    molpcba_rank_map.get(canon, "N/A"),
                    f"{molpcba_val_map.get(canon, 0):.4f}" if canon in molpcba_val_map else "N/A",
                ]
            )

    print(f"  Saved {csv_path}")

    # Print table
    print(f"\n  {'Method':<20s} | {'moltox21':>12s} | {'molpcba':>12s} | {'rank shift':>10s}")
    print(f"  {'-' * 65}")
    for canon in CANONICALIZATIONS:
        label = CANON_LABELS[canon]
        r1 = moltox21_rank_map.get(canon)
        r2 = molpcba_rank_map.get(canon)
        v1 = moltox21_val_map.get(canon, 0)
        v2 = molpcba_val_map.get(canon, 0)
        shift = f"{r1 - r2:+d}" if r1 is not None and r2 is not None else "N/A"
        print(f"  {label:<20s} | #{r1} ({v1:.4f}) | #{r2} ({v2:.4f}) | {shift:>10s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ogbg-molpcba canonicalization experiment results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/molpcba_canonicalization",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots/tables (default: results_dir/analysis)",
    )
    parser.add_argument(
        "--moltox21-dir",
        type=str,
        default=None,
        help="moltox21 results dir for cross-dataset comparison",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "analysis")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Analyzing ogbg-molpcba results in {args.results_dir}")
    print(f"Metric: {METRIC_KEY} ({METRIC_LABEL})")

    # Experiment 1: Parameter Efficiency
    print("\nExperiment 1: Parameter Efficiency")
    exp1_results = load_experiment_results(args.results_dir, "exp1_param_efficiency")
    print(f"  Loaded {len(exp1_results)} runs")
    if exp1_results:
        analyze_param_efficiency(exp1_results, os.path.join(args.output_dir, "exp1"))

    # Experiments 2+3: Stratified Scores & Convergence
    print("\nExperiments 2+3: Stratified Scores & Convergence")
    exp23_results = load_experiment_results(args.results_dir, "exp23_subset_convergence")
    print(f"  Loaded {len(exp23_results)} runs")

    if exp23_results:
        analyze_stratified_scores(args.results_dir, os.path.join(args.output_dir, "exp2"))
        analyze_convergence(exp23_results, os.path.join(args.output_dir, "exp3"))

    # Cross-dataset comparison
    if args.moltox21_dir:
        print("\nCross-dataset Comparison (molpcba vs moltox21)")
        all_results = {}
        all_results.update(exp1_results)
        all_results.update(exp23_results)
        if all_results:
            cross_dataset_comparison(all_results, args.moltox21_dir, args.output_dir)

    print(f"\nAnalysis saved to {args.output_dir}")


if __name__ == "__main__":
    main()
