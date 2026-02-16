#!/usr/bin/env python
"""Train Deep Sets with HKS features on ModelNet.

Uses HKSDeepSetsClassifier with per-point features (optionally xyz + HKS).
HKS features are sign-invariant by construction, so no eigenvector
canonicalization is needed.
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

from src.experiments.deep_sets.dataset_hks import HKSModelNet  # noqa: E402
from src.experiments.deep_sets.model_hks import (  # noqa: E402
    HKSDeepSetsClassifier,
)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    n_samples = 0
    for features, mask, labels in loader:
        features = features.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        logits = model(features, mask)
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
    for features, mask, labels in loader:
        features = features.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        logits = model(features, mask)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Train Deep Sets with HKS features on ModelNet")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root data directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ModelNet10",
        choices=["ModelNet10", "ModelNet40"],
        help="Dataset name",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=1024,
        help="Max points per shape",
    )
    parser.add_argument(
        "--n-eigs",
        type=int,
        default=32,
        help="Number of eigenpairs for HKS computation",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=30,
        help="k-NN neighbors for graph construction",
    )
    parser.add_argument(
        "--n-times",
        type=int,
        default=16,
        help="Number of HKS time samples",
    )
    parser.add_argument(
        "--no-weighted",
        action="store_true",
        help="Disable Gaussian kernel weighted graph Laplacian",
    )
    parser.add_argument(
        "--no-normalized",
        action="store_true",
        help="Use combinatorial instead of normalized Laplacian",
    )
    parser.add_argument(
        "--no-xyz",
        action="store_true",
        help="Exclude xyz coords; use only HKS features",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (cpu/cuda/auto)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/deep_sets_hks",
        help="Save directory",
    )
    args = parser.parse_args()

    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    weighted = not args.no_weighted
    normalized = not args.no_normalized
    include_xyz = not args.no_xyz

    # Datasets
    print(
        f"Loading {args.dataset} training set "
        f"(n_eigs={args.n_eigs}, n_times={args.n_times}, "
        f"weighted={weighted}, normalized={normalized})..."
    )
    train_ds = HKSModelNet(
        args.data_dir,
        dataset=args.dataset,
        split="train",
        max_points=args.max_points,
        n_eigs=args.n_eigs,
        n_neighbors=args.n_neighbors,
        n_times=args.n_times,
        weighted=weighted,
        normalized=normalized,
        include_xyz=include_xyz,
    )
    print(f"Loading {args.dataset} test set...")
    test_ds = HKSModelNet(
        args.data_dir,
        dataset=args.dataset,
        split="test",
        max_points=args.max_points,
        n_eigs=args.n_eigs,
        n_neighbors=args.n_neighbors,
        n_times=args.n_times,
        weighted=weighted,
        normalized=normalized,
        include_xyz=include_xyz,
    )
    print(f"Train: {len(train_ds)} shapes, Test: {len(test_ds)} shapes")
    print(f"Classes: {train_ds.classes}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # Model
    n_classes = len(train_ds.classes)
    in_channels = 3 if include_xyz else 0
    model = HKSDeepSetsClassifier(
        in_channels=in_channels,
        n_times=args.n_times,
        n_classes=n_classes,
        hidden_dim=args.hidden_dim,
        include_xyz=include_xyz,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
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
        "n_eigs": args.n_eigs,
        "n_neighbors": args.n_neighbors,
        "n_times": args.n_times,
        "weighted": weighted,
        "normalized": normalized,
        "include_xyz": include_xyz,
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
