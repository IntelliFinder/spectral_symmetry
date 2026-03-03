#!/usr/bin/env python
"""Launcher for canonicalization impact experiments.

Generates and optionally runs all experiment configurations for the three
experiments described in the spec:

  Experiment 1: Parameter Efficiency  (6 methods x 5 sizes x 5 seeds = 150 runs)
  Experiment 2+3: Subset Accuracy & Convergence  (6 methods x 5 seeds = 30 runs)

Usage
-----
    # Generate shell commands (dry run)
    python scripts/run_canonicalization_experiments.py --dry-run

    # Run all experiments sequentially
    python scripts/run_canonicalization_experiments.py --dataset ogbg-moltox21

    # Run experiment 1 only
    python scripts/run_canonicalization_experiments.py --experiment 1

    # Run a specific configuration
    python scripts/run_canonicalization_experiments.py --canonicalization spielman --hidden-dim 256
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
        cmd = [
            sys.executable,
            "scripts/train_molecular.py",
            "--dataset",
            args.dataset,
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
        cmd = [
            sys.executable,
            "scripts/train_molecular.py",
            "--dataset",
            args.dataset,
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


def build_audit_command(args):
    """Test-split spectral audit command."""
    output_dir = os.path.join(args.results_dir, "test_split_audit")
    cmd = [
        sys.executable,
        "scripts/spectral_audit.py",
        "--dataset",
        args.dataset,
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


def main():
    parser = argparse.ArgumentParser(description="Launch canonicalization impact experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbg-moltox21",
        choices=["ogbg-moltox21", "ogbg-molpcba"],
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/canonicalization_experiments",
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
    parser.add_argument("--audit-only", action="store_true", help="Only run test-split audit")
    # Filter options
    parser.add_argument(
        "--canonicalization", type=str, default=None, help="Run only this canonicalization method"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=None, help="Run only this hidden dim (exp1)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Run only this seed")
    args = parser.parse_args()

    commands = []

    if args.audit_only:
        commands.append(build_audit_command(args))
    else:
        # Audit
        commands.append(build_audit_command(args))

        # Experiment commands
        if args.experiment in (0, 1):
            commands.extend(build_exp1_commands(args))
        if args.experiment in (0, 23):
            commands.extend(build_exp23_commands(args))

    # Apply filters
    if args.canonicalization:
        commands = [
            c for c in commands if "--canonicalization" not in c or args.canonicalization in c
        ]
    if args.hidden_dim is not None:
        commands = [c for c in commands if "--hidden-dim" not in c or str(args.hidden_dim) in c]
    if args.seed is not None:
        commands = [c for c in commands if "--seed" not in c or str(args.seed) in c]

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


if __name__ == "__main__":
    main()
