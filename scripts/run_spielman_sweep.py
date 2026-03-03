"""Sweep script comparing eigenvector counts and canonicalization methods.

Systematically runs train_spielman_node.py and train_spielman_spectral.py
across all combinations of:
  - Models: node (DistanceTransformer + eigvec features), spectral (Standard Transformer)
  - Canonicalization: spielman, maxabs, random, none
  - n_eigs: 4, 8, 16

Experiments run sequentially (GPU memory constraint). After all runs finish,
a summary table sorted by test accuracy is printed and saved to CSV.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

# Map model short names to training script filenames
SCRIPT_MAP = {
    "node": "train_spielman_node.py",
    "spectral": "train_spielman_spectral.py",
}

DEFAULT_MODELS = ["node", "spectral"]
DEFAULT_CANONICALIZATIONS = ["spielman", "maxabs", "random", "none"]
DEFAULT_N_EIGS = [4, 8, 16]


def build_experiment_grid(models, canonicalizations, n_eigs_list, variant, epochs, weighted=False):
    """Build a list of experiment configs from the Cartesian product of parameters.

    Returns
    -------
    list[dict] : Each dict has keys: model, canonicalization, n_eigs, variant, epochs, weighted.
    """
    experiments = []
    for model in models:
        for canon in canonicalizations:
            for k in n_eigs_list:
                experiments.append(
                    {
                        "model": model,
                        "canonicalization": canon,
                        "n_eigs": k,
                        "variant": variant,
                        "epochs": epochs,
                        "weighted": weighted,
                    }
                )
    return experiments


def experiment_save_dir(base_dir, exp):
    """Return the save directory for an experiment config."""
    return Path(base_dir) / f"{exp['model']}_{exp['canonicalization']}_k{exp['n_eigs']}"


def experiment_results_path(base_dir, exp):
    """Return the path to the results JSON for an experiment config."""
    return experiment_save_dir(base_dir, exp) / "results.json"


def build_command(exp, data_dir, save_dir, scripts_dir):
    """Build the subprocess command list for one experiment.

    Parameters
    ----------
    exp : dict
        Experiment config with model, canonicalization, n_eigs, variant, epochs,
        and optionally weighted.
    data_dir : str
        Root data directory.
    save_dir : Path
        Save directory for this experiment.
    scripts_dir : Path
        Directory containing the training scripts.

    Returns
    -------
    list[str] : Command suitable for subprocess.run.
    """
    script_path = scripts_dir / SCRIPT_MAP[exp["model"]]
    cmd = [
        sys.executable,
        str(script_path),
        "--data-dir",
        str(data_dir),
        "--variant",
        str(exp["variant"]),
        "--n-eigs",
        str(exp["n_eigs"]),
        "--canonicalization",
        exp["canonicalization"],
        "--epochs",
        str(exp["epochs"]),
        "--save-dir",
        str(save_dir),
    ]
    if exp.get("weighted"):
        cmd.append("--weighted")
    return cmd


def format_duration(seconds):
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h{mins:02d}m{secs:02d}s"


def run_experiments(experiments, data_dir, base_save_dir, scripts_dir, resume=False):
    """Run all experiments sequentially.

    Parameters
    ----------
    experiments : list[dict]
    data_dir : str
    base_save_dir : str
    scripts_dir : Path
    resume : bool
        If True, skip experiments whose results JSON already exists.

    Returns
    -------
    list[dict] : Results summaries for completed experiments (including skipped).
    """
    total = len(experiments)
    results = []
    sweep_start = time.time()

    for i, exp in enumerate(experiments, 1):
        label = f"{exp['model']} {exp['canonicalization']} k={exp['n_eigs']}"
        save_dir = experiment_save_dir(base_save_dir, exp)
        results_path = experiment_results_path(base_save_dir, exp)

        # Resume: skip if results already exist
        if resume and results_path.exists():
            print(f"[{i}/{total}] SKIP (resume): {label}  — {results_path} exists")
            try:
                with open(results_path) as f:
                    data = json.load(f)
                results.append(
                    {
                        "model": exp["model"],
                        "canonicalization": data.get("canonicalization", exp["canonicalization"]),
                        "n_eigs": data.get("n_eigs", exp["n_eigs"]),
                        "best_val_acc": data.get("best_val_acc"),
                        "test_acc": data.get("test_acc"),
                        "status": "skipped",
                    }
                )
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  WARNING: could not parse {results_path}: {e}")
            continue

        print(f"\n{'=' * 80}")
        print(f"Running experiment {i}/{total}: {label}")
        print(f"{'=' * 80}")

        cmd = build_command(exp, data_dir, save_dir, scripts_dir)
        print(f"Command: {' '.join(cmd)}")

        # Ensure save dir exists for log file
        save_dir.mkdir(parents=True, exist_ok=True)
        log_path = save_dir / "train.log"

        exp_start = time.time()
        try:
            with open(log_path, "w") as log_file:
                proc = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=7200,  # 2 hour timeout per experiment
                )
            elapsed = time.time() - exp_start
            duration_str = format_duration(elapsed)

            if proc.returncode != 0:
                print(f"  FAILED (exit code {proc.returncode}) after {duration_str}")
                print(f"  See log: {log_path}")
                results.append(
                    {
                        "model": exp["model"],
                        "canonicalization": exp["canonicalization"],
                        "n_eigs": exp["n_eigs"],
                        "best_val_acc": None,
                        "test_acc": None,
                        "status": f"failed (rc={proc.returncode})",
                    }
                )
                continue

            print(f"  DONE in {duration_str}")

            # Read results
            if results_path.exists():
                with open(results_path) as f:
                    data = json.load(f)
                val_acc = data.get("best_val_acc")
                test_acc = data.get("test_acc")
                print(f"  Val acc: {val_acc:.4f}  Test acc: {test_acc:.4f}")
                results.append(
                    {
                        "model": exp["model"],
                        "canonicalization": data.get("canonicalization", exp["canonicalization"]),
                        "n_eigs": data.get("n_eigs", exp["n_eigs"]),
                        "best_val_acc": val_acc,
                        "test_acc": test_acc,
                        "status": "ok",
                    }
                )
            else:
                print(f"  WARNING: results.json not found at {results_path}")
                results.append(
                    {
                        "model": exp["model"],
                        "canonicalization": exp["canonicalization"],
                        "n_eigs": exp["n_eigs"],
                        "best_val_acc": None,
                        "test_acc": None,
                        "status": "no results file",
                    }
                )

        except subprocess.TimeoutExpired:
            elapsed = time.time() - exp_start
            print(f"  TIMEOUT after {format_duration(elapsed)}")
            results.append(
                {
                    "model": exp["model"],
                    "canonicalization": exp["canonicalization"],
                    "n_eigs": exp["n_eigs"],
                    "best_val_acc": None,
                    "test_acc": None,
                    "status": "timeout",
                }
            )
        except Exception as e:
            elapsed = time.time() - exp_start
            print(f"  ERROR after {format_duration(elapsed)}: {e}")
            results.append(
                {
                    "model": exp["model"],
                    "canonicalization": exp["canonicalization"],
                    "n_eigs": exp["n_eigs"],
                    "best_val_acc": None,
                    "test_acc": None,
                    "status": f"error: {e}",
                }
            )

    total_elapsed = time.time() - sweep_start
    print(f"\nAll experiments completed in {format_duration(total_elapsed)}")
    return results


def print_summary_table(results):
    """Print a formatted summary table sorted by test accuracy descending."""

    # Sort: successful results first (by test_acc desc), then failed
    def sort_key(r):
        acc = r.get("test_acc")
        if acc is not None:
            return (0, -acc)
        return (1, 0)

    sorted_results = sorted(results, key=sort_key)

    header = f"{'Model':<10} {'Canon':<12} {'k':>3}  {'Val Acc':>8}  {'Test Acc':>9}  {'Status'}"
    sep = "-" * len(header)

    print(f"\n{sep}")
    print("SPIELMAN SWEEP SUMMARY")
    print(sep)
    print(header)
    print(sep)

    for r in sorted_results:
        val_str = f"{r['best_val_acc']:.4f}" if r["best_val_acc"] is not None else "   N/A"
        test_str = f"{r['test_acc']:.4f}" if r["test_acc"] is not None else "    N/A"
        status = r.get("status", "")
        print(
            f"{r['model']:<10} {r['canonicalization']:<12} {r['n_eigs']:>3}"
            f"  {val_str:>8}  {test_str:>9}  {status}"
        )

    print(sep)


def save_summary_csv(results, csv_path):
    """Save the summary table to a CSV file sorted by test accuracy descending."""

    def sort_key(r):
        acc = r.get("test_acc")
        if acc is not None:
            return (0, -acc)
        return (1, 0)

    sorted_results = sorted(results, key=sort_key)

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["model", "canonicalization", "n_eigs", "best_val_acc", "test_acc", "status"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted_results:
            writer.writerow(
                {
                    "model": r["model"],
                    "canonicalization": r["canonicalization"],
                    "n_eigs": r["n_eigs"],
                    "best_val_acc": r.get("best_val_acc", ""),
                    "test_acc": r.get("test_acc", ""),
                    "status": r.get("status", ""),
                }
            )

    print(f"Summary saved to {csv_path}")


def collect_existing_results(experiments, base_save_dir):
    """Collect results from already-completed experiments (for summary-only mode).

    Returns
    -------
    list[dict] : Results summaries for experiments with existing results files.
    """
    results = []
    for exp in experiments:
        results_path = experiment_results_path(base_save_dir, exp)
        if results_path.exists():
            try:
                with open(results_path) as f:
                    data = json.load(f)
                results.append(
                    {
                        "model": exp["model"],
                        "canonicalization": data.get("canonicalization", exp["canonicalization"]),
                        "n_eigs": data.get("n_eigs", exp["n_eigs"]),
                        "best_val_acc": data.get("best_val_acc"),
                        "test_acc": data.get("test_acc"),
                        "status": "ok",
                    }
                )
            except (json.JSONDecodeError, KeyError):
                pass
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Sweep over eigenvector counts and canonicalization methods"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root data directory (default: data)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs per experiment (default: 100)",
    )
    parser.add_argument(
        "--variant",
        type=int,
        default=10,
        choices=[10, 40],
        help="ModelNet variant (default: 10)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=DEFAULT_MODELS,
        help="Models to sweep (default: node spectral)",
    )
    parser.add_argument(
        "--canonicalizations",
        nargs="+",
        default=DEFAULT_CANONICALIZATIONS,
        choices=DEFAULT_CANONICALIZATIONS,
        help="Canonicalization methods to sweep (default: all four)",
    )
    parser.add_argument(
        "--n-eigs-list",
        nargs="+",
        type=int,
        default=DEFAULT_N_EIGS,
        help="Eigenvector counts to sweep (default: 4 8 16)",
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Use Gaussian kernel weighted graph Laplacian for all experiments",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Base save directory (default: results/spielman_sweep[_weighted])",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all commands without executing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip experiments whose results JSON already exists",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print/save summary from existing results (no training)",
    )
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    if args.save_dir is not None:
        base_save_dir = Path(args.save_dir)
    elif args.weighted:
        base_save_dir = Path("results") / "spielman_sweep_weighted"
    else:
        base_save_dir = Path("results") / "spielman_sweep"

    # Build experiment grid
    experiments = build_experiment_grid(
        models=args.models,
        canonicalizations=args.canonicalizations,
        n_eigs_list=args.n_eigs_list,
        variant=args.variant,
        epochs=args.epochs,
        weighted=args.weighted,
    )

    total = len(experiments)
    print(f"Experiment grid: {total} runs")
    print(f"  Models: {args.models}")
    print(f"  Canonicalizations: {args.canonicalizations}")
    print(f"  n_eigs: {args.n_eigs_list}")
    print(f"  Variant: ModelNet{args.variant}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Weighted: {args.weighted}")
    print(f"  Save dir: {base_save_dir}")
    print()

    # Dry run: print commands and exit
    if args.dry_run:
        print("DRY RUN — commands that would be executed:\n")
        for i, exp in enumerate(experiments, 1):
            label = f"{exp['model']} {exp['canonicalization']} k={exp['n_eigs']}"
            save_dir = experiment_save_dir(base_save_dir, exp)
            cmd = build_command(exp, args.data_dir, save_dir, scripts_dir)
            print(f"  [{i}/{total}] {label}")
            print(f"    {' '.join(cmd)}")
            print()
        return

    # Summary-only mode: collect existing results and display
    if args.summary_only:
        print("Summary-only mode — collecting existing results...\n")
        results = collect_existing_results(experiments, base_save_dir)
        if not results:
            print("No results found.")
            return
        print_summary_table(results)
        csv_path = base_save_dir / "summary.csv"
        save_summary_csv(results, csv_path)
        return

    # Run all experiments
    results = run_experiments(
        experiments=experiments,
        data_dir=args.data_dir,
        base_save_dir=base_save_dir,
        scripts_dir=scripts_dir,
        resume=args.resume,
    )

    # Print and save summary
    if results:
        print_summary_table(results)
        csv_path = base_save_dir / "summary.csv"
        save_summary_csv(results, csv_path)


if __name__ == "__main__":
    main()
