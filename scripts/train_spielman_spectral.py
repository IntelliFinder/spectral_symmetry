"""Train standard transformer with Spielman-canonicalized spectral positional encodings.

Uses SpectralTransformerClassifier + SpielmanSpectralModelNet. Supports multiple
eigenvector canonicalization strategies: spielman, maxabs, random, none.

Includes proper train/val/test split:
- The official "train" split is divided 80/20 into train/val (seed=42).
- Model selection uses best validation accuracy.
- Test set is evaluated only once after training completes.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.spectral_transformer.dataset_spielman import (  # noqa: E402
    CANONICALIZATION_CHOICES,
    SpielmanSpectralModelNet,
)
from src.experiments.spectral_transformer.model import (  # noqa: E402
    SpectralTransformerClassifier,
)
from src.experiments.spectral_transformer.train import (  # noqa: E402
    evaluate,
    train_one_epoch,
)
from src.training import seed_everything, worker_init_fn  # noqa: E402


def make_train_val_split(dataset, val_fraction=0.2, seed=42):
    """Split a dataset into train/val subsets using deterministic indices.

    Parameters
    ----------
    dataset : Dataset
    val_fraction : float
    seed : int

    Returns
    -------
    train_subset, val_subset : Subset, Subset
    """
    n = len(dataset)
    indices = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    n_val = int(n * val_fraction)
    val_indices = indices[:n_val].tolist()
    train_indices = indices[n_val:].tolist()

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train standard transformer with Spielman-canonicalized spectral positional encodings"
        )
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument(
        "--variant",
        type=int,
        default=10,
        choices=[10, 40],
        help="ModelNet variant",
    )
    parser.add_argument("--n-points", type=int, default=512, help="Points per shape")
    parser.add_argument("--n-eigs", type=int, default=16, help="Number of eigenvectors")
    parser.add_argument("--n-neighbors", type=int, default=12, help="k-NN neighbors")
    parser.add_argument(
        "--canonicalization",
        type=str,
        default="spielman",
        choices=CANONICALIZATION_CHOICES,
        help="Eigenvector canonicalization method",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--d-model", type=int, default=128, help="Transformer hidden dim")
    parser.add_argument("--nhead", type=int, default=8, help="Attention heads")
    parser.add_argument("--num-layers", type=int, default=6, help="Transformer layers")
    parser.add_argument("--dim-feedforward", type=int, default=512, help="FFN dim")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download ModelNet if missing",
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Use Gaussian kernel weighted graph Laplacian",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Save directory (default: results/spielman_spectral/{canon}_k{n_eigs})",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Default save directory
    if args.save_dir is None:
        save_dir = Path("results") / "spielman_spectral" / f"{args.canonicalization}_k{args.n_eigs}"
    else:
        save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")

    # --- Datasets ---
    print(
        f"Loading ModelNet{args.variant} training set "
        f"(canonicalization={args.canonicalization}, k={args.n_eigs})..."
    )
    full_train_ds = SpielmanSpectralModelNet(
        args.data_dir,
        split="train",
        n_points=args.n_points,
        n_eigs=args.n_eigs,
        n_neighbors=args.n_neighbors,
        download=args.download,
        variant=args.variant,
        canonicalization=args.canonicalization,
        weighted=args.weighted,
    )

    # 80/20 train/val split with fixed seed
    train_ds, val_ds = make_train_val_split(full_train_ds, val_fraction=0.2, seed=42)
    print(f"Train: {len(train_ds)} shapes, Val: {len(val_ds)} shapes")

    print(f"Loading ModelNet{args.variant} test set...")
    test_ds = SpielmanSpectralModelNet(
        args.data_dir,
        split="test",
        n_points=args.n_points,
        n_eigs=args.n_eigs,
        n_neighbors=args.n_neighbors,
        download=False,
        variant=args.variant,
        canonicalization=args.canonicalization,
        weighted=args.weighted,
    )
    print(f"Test: {len(test_ds)} shapes")
    print(f"Classes: {full_train_ds.classes}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    # --- Model ---
    input_dim = 3 + args.n_eigs
    model = SpectralTransformerClassifier(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        n_classes=len(full_train_ds.classes),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training loop with val-based model selection ---
    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best_model.pt")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # --- Final test evaluation (once) ---
    model.load_state_dict(torch.load(save_dir / "best_model.pt", weights_only=True))
    test_acc = evaluate(model, test_loader, device)
    print(f"Final test accuracy: {test_acc:.4f}")

    # --- Save results ---
    results = {
        "canonicalization": args.canonicalization,
        "weighted": args.weighted,
        "n_eigs": args.n_eigs,
        "variant": args.variant,
        "n_points": args.n_points,
        "n_neighbors": args.n_neighbors,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "n_params": n_params,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "history": history,
    }

    results_path = save_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
