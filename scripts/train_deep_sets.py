#!/usr/bin/env python
"""Train Deep Sets with spectral features on ModelNet.

Uses DeepSetsClassifier with per-point features (xyz + eigenvectors) and
eigenvalue conditioning. Supports multiple eigenvector canonicalization
strategies and optional Gaussian-weighted graph Laplacian.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.deep_sets.dataset import DeepSetsModelNet  # noqa: E402
from src.experiments.deep_sets.model import DeepSetsClassifier, SpectrumClassifier  # noqa: E402


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    n_samples = 0
    for features, eigenvalues, mask, labels in loader:
        features = features.to(device)
        eigenvalues = eigenvalues.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        logits = model(features, eigenvalues, mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        n_samples += labels.size(0)
    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate and return accuracy."""
    model.eval()
    correct = 0
    total = 0
    for features, eigenvalues, mask, labels in loader:
        features = features.to(device)
        eigenvalues = eigenvalues.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        logits = model(features, eigenvalues, mask)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Train Deep Sets with spectral features on ModelNet"
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ModelNet10",
        choices=["ModelNet10", "ModelNet40"],
        help="Dataset name",
    )
    parser.add_argument("--max-points", type=int, default=1024, help="Max points per shape")
    parser.add_argument("--n-eigs", type=int, default=8, help="Number of eigenvectors")
    parser.add_argument("--n-neighbors", type=int, default=12, help="k-NN neighbors")
    parser.add_argument(
        "--canonicalization",
        type=str,
        default="spielman",
        choices=["spielman", "maxabs", "random", "none"],
        help="Eigenvector canonicalization method",
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Use Gaussian kernel weighted graph Laplacian",
    )
    parser.add_argument(
        "--no-xyz",
        action="store_true",
        help="Exclude xyz coords; use only eigenvectors + eigenvalues",
    )
    parser.add_argument(
        "--scaling-mode",
        type=str,
        default="fixed",
        choices=["fixed", "learnable"],
        help="Eigenvalue scaling mode",
    )
    parser.add_argument(
        "--use-spectrum",
        action="store_true",
        help="Concatenate eigenvalue spectrum descriptor to pooled features",
    )
    parser.add_argument(
        "--spectrum-only",
        action="store_true",
        help="Use SpectrumClassifier (eigenvalue spectrum only, no per-point features)",
    )
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/deep_sets",
        help="Save directory",
    )
    args = parser.parse_args()

    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Datasets
    print(
        f"Loading {args.dataset} training set "
        f"(canonicalization={args.canonicalization}, k={args.n_eigs})..."
    )
    include_xyz = not args.no_xyz
    train_ds = DeepSetsModelNet(
        args.data_dir,
        dataset=args.dataset,
        split="train",
        max_points=args.max_points,
        n_eigs=args.n_eigs,
        n_neighbors=args.n_neighbors,
        canonicalization=args.canonicalization,
        weighted=args.weighted,
        include_xyz=include_xyz,
    )
    print(f"Loading {args.dataset} test set...")
    test_ds = DeepSetsModelNet(
        args.data_dir,
        dataset=args.dataset,
        split="test",
        max_points=args.max_points,
        n_eigs=args.n_eigs,
        n_neighbors=args.n_neighbors,
        canonicalization=args.canonicalization,
        weighted=args.weighted,
        include_xyz=include_xyz,
    )
    print(f"Train: {len(train_ds)} shapes, Test: {len(test_ds)} shapes")
    print(f"Classes: {train_ds.classes}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    n_classes = len(train_ds.classes)
    in_channels = 3 if include_xyz else 0
    if args.spectrum_only:
        model = SpectrumClassifier(
            n_eigs=args.n_eigs,
            n_classes=n_classes,
            hidden_dim=args.hidden_dim,
        ).to(device)
    else:
        model = DeepSetsClassifier(
            in_channels=in_channels,
            n_eigs=args.n_eigs,
            n_classes=n_classes,
            hidden_dim=args.hidden_dim,
            scaling_mode=args.scaling_mode,
            use_spectrum=args.use_spectrum,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  test_acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_dir / "best_model.pt")

    print(f"Best test accuracy: {best_acc:.4f}")
    print(f"Model saved to {save_dir / 'best_model.pt'}")

    # Save results
    results = {
        "best_test_acc": best_acc,
        "dataset": args.dataset,
        "scaling_mode": args.scaling_mode,
        "n_eigs": args.n_eigs,
        "canonicalization": args.canonicalization,
        "weighted": args.weighted,
        "include_xyz": include_xyz,
        "use_spectrum": args.use_spectrum,
        "spectrum_only": args.spectrum_only,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {save_dir / 'results.json'}")


if __name__ == "__main__":
    main()
