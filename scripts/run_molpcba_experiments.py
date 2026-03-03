#!/usr/bin/env python
"""Launcher for ogbg-molpcba canonicalization impact experiments.

Generates and optionally runs all experiment configurations:

  Phase 0: Test-split spectral audit
  Phase 1: Parameter Efficiency  (6 methods x 5 sizes x 5 seeds = 150 runs)
  Phase 2+3: Subset Accuracy & Convergence  (6 methods x 5 seeds = 30 runs)

Usage
-----
    # Dry run — print all 181 commands
    python scripts/run_molpcba_experiments.py --dry-run

    # Run all experiments sequentially
    python scripts/run_molpcba_experiments.py

    # Run experiment 1 only
    python scripts/run_molpcba_experiments.py --experiment 1

    # Single smoke test
    python scripts/run_molpcba_experiments.py --experiment 1 \\
        --hidden-dim 32 --seed 0 --canonicalization maxabs

    # Skip audit phase
    python scripts/run_molpcba_experiments.py --skip-audit

    # Run analysis after training
    python scripts/run_molpcba_experiments.py --analyze-only
"""

import argparse
import itertools
import os
import subprocess
import sys

CANONICALIZATIONS = [
    "random_fixed",
    "random_augmented",
    "maxabs",
    "spielman",
    "map",
    "oap",
]

HIDDEN_DIMS = [32, 64, 128, 256, 512]
SEEDS = [0, 1, 2, 3, 4]

DATASET = "ogbg-molpcba"
DEFAULT_RESULTS_DIR = "results/molpcba_canonicalization"


def build_audit_command(args):
    """Phase 0: Test-split spectral audit command."""
    output_dir = os.path.join(args.results_dir, "test_split_audit")
    cmd = [
        sys.executable,
        "scripts/spectral_audit.py",
        "--dataset",
        DATASET,
        "--data-dir",
        args.data_dir,
        "--split",
        "test",
        "--n-eigs",
        str(args.n_eigs),
        "--output",
        output_dir,
    ]
    return cmd


def build_exp1_commands(args):
    """Experiment 1: Parameter Efficiency.

    6 methods x 5 hidden_dims x 5 seeds = 150 runs.
    """
    commands = []
    for canon, hdim, seed in itertools.product(CANONICALIZATIONS, HIDDEN_DIMS, SEEDS):
        save_dir = os.path.join(
            args.results_dir,
            "exp1_param_efficiency",
            f"{canon}_h{hdim}_s{seed}",
        )
        # Skip if already completed
        if os.path.exists(os.path.join(save_dir, "results.json")) and not args.force:
            continue
        cmd = [
            sys.executable,
            "scripts/train_molecular.py",
            "--dataset",
            DATASET,
            "--data-dir",
            args.data_dir,
            "--canonicalization",
            canon,
            "--hidden-dim",
            str(hdim),
            "--num-layers",
            str(args.num_layers),
            "--n-eigs",
            str(args.n_eigs),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--seed",
            str(seed),
            "--save-dir",
            save_dir,
        ]
        commands.append(cmd)
    return commands


def build_exp23_commands(args):
    """Experiments 2+3: Subset Accuracy & Convergence.

    6 methods x 5 seeds = 30 runs, with predictions saved.
    """
    commands = []
    for canon, seed in itertools.product(CANONICALIZATIONS, SEEDS):
        save_dir = os.path.join(
            args.results_dir,
            "exp23_subset_convergence",
            f"{canon}_s{seed}",
        )
        # Skip if already completed
        if os.path.exists(os.path.join(save_dir, "results.json")) and not args.force:
            continue
        cmd = [
            sys.executable,
            "scripts/train_molecular.py",
            "--dataset",
            DATASET,
            "--data-dir",
            args.data_dir,
            "--canonicalization",
            canon,
            "--hidden-dim",
            str(args.fixed_hidden_dim),
            "--num-layers",
            str(args.num_layers),
            "--n-eigs",
            str(args.n_eigs),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--seed",
            str(seed),
            "--save-dir",
            save_dir,
            "--save-predictions",
        ]
        commands.append(cmd)
    return commands


def run_analysis(args):
    """Run the analysis script after training."""
    cmd = [
        sys.executable,
        "scripts/analyze_molpcba_results.py",
        "--results-dir",
        args.results_dir,
    ]
    if args.moltox21_dir:
        cmd.extend(["--moltox21-dir", args.moltox21_dir])
    print(f"\nRunning analysis: {' '.join(cmd)}")
    subprocess.run(cmd)


def _cmd_arg(cmd, flag):
    """Extract the value following a CLI flag in a command list."""
    try:
        idx = cmd.index(flag)
        return cmd[idx + 1]
    except (ValueError, IndexError):
        return None


def apply_filters(commands, args):
    """Filter commands by canonicalization, hidden-dim, and seed."""
    filtered = commands
    if args.canonicalization:
        filtered = [
            c
            for c in filtered
            if _cmd_arg(c, "--canonicalization") in (None, args.canonicalization)
        ]
    if args.hidden_dim is not None:
        filtered = [
            c
            for c in filtered
            if _cmd_arg(c, "--hidden-dim") in (None, str(args.hidden_dim))
        ]
    if args.seed is not None:
        filtered = [
            c for c in filtered if _cmd_arg(c, "--seed") in (None, str(args.seed))
        ]
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Launch ogbg-molpcba canonicalization impact experiments"
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument("--n-eigs", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fixed-hidden-dim", type=int, default=256, help="Hidden dim for Exp 2+3")
    parser.add_argument(
        "--experiment",
        type=int,
        default=0,
        choices=[0, 1, 23],
        help="Which experiment to run (0=all, 1=param efficiency, 23=subset+convergence)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--skip-audit", action="store_true", help="Skip Phase 0 spectral audit")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only run analysis (skip audit + training)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if results.json already exists",
    )
    # Filter options
    parser.add_argument(
        "--canonicalization",
        type=str,
        default=None,
        help="Run only this canonicalization method",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Run only this hidden dim (exp1)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Run only this seed")
    # Cross-dataset comparison
    parser.add_argument(
        "--moltox21-dir",
        type=str,
        default=None,
        help="moltox21 results dir for cross-dataset comparison in analysis",
    )
    args = parser.parse_args()

    if args.analyze_only:
        run_analysis(args)
        return

    commands = []

    # Phase 0: Spectral audit
    if not args.skip_audit:
        audit_csv = os.path.join(args.results_dir, "test_split_audit", "audit_details.csv")
        if os.path.exists(audit_csv) and not args.force:
            print(f"Audit already exists: {audit_csv} (use --force to re-run)")
        else:
            commands.append(build_audit_command(args))

    # Phase 1+2: Training runs
    if args.experiment in (0, 1):
        commands.extend(build_exp1_commands(args))
    if args.experiment in (0, 23):
        commands.extend(build_exp23_commands(args))

    # Apply filters
    commands = apply_filters(commands, args)

    print(f"Dataset: {DATASET}")
    print(f"Results dir: {args.results_dir}")
    print(f"Total commands: {len(commands)}")

    for i, cmd in enumerate(commands):
        cmd_str = " ".join(cmd)
        if args.dry_run:
            print(f"[{i + 1}/{len(commands)}] {cmd_str}")
        else:
            print(f"\n{'=' * 70}")
            print(f"[{i + 1}/{len(commands)}] {cmd_str}")
            print(f"{'=' * 70}")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"WARNING: Command failed with return code {result.returncode}")

    if not args.dry_run and commands:
        print(f"\nAll {len(commands)} commands finished.")
        print("Run analysis with: python scripts/run_molpcba_experiments.py --analyze-only")


if __name__ == "__main__":
    main()
