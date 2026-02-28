#!/usr/bin/env python
"""Train Deep Sets with PCA-canonicalized xyz on ModelNet.

Uses HKSDeepSetsClassifier with n_times=0 (xyz-only features).
PCA canonicalization aligns point clouds to principal axes before
classification, providing rotation invariance without spectral features.
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

from src.experiments.deep_sets.dataset_pca import PCAModelNet  # noqa: E402
from src.experiments.deep_sets.model_hks import (  # noqa: E402
    HKSDeepSetsClassifier,
)
from src.training import make_train_val_split  # noqa: E402


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
    parser = argparse.ArgumentParser(
        description="Train Deep Sets with PCA-canonicalized xyz on ModelNet"
    )
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
        "--sign-method",
        type=str,
        default="majority",
        choices=["majority", "maxabs", "random", "spielman"],
        help="PCA sign canonicalization method",
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
        default="results/pca_xyz_mn10",
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
        f"Loading {args.dataset} training set (PCA-canonicalized xyz, "
        f"sign_method={args.sign_method})..."
    )
    full_train_ds = PCAModelNet(
        args.data_dir,
        dataset=args.dataset,
        split="train",
        max_points=args.max_points,
        sign_method=args.sign_method,
    )
    train_ds, val_ds = make_train_val_split(full_train_ds, val_fraction=0.2, seed=42)
    print(f"Loading {args.dataset} test set...")
    test_ds = PCAModelNet(
        args.data_dir,
        dataset=args.dataset,
        split="test",
        max_points=args.max_points,
        sign_method=args.sign_method,
    )
    print(f"Train: {len(train_ds)} shapes, Val: {len(val_ds)} shapes, Test: {len(test_ds)} shapes")
    print(f"Classes: {full_train_ds.classes}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # Model: HKSDeepSetsClassifier with n_times=0 (xyz only)
    n_classes = len(full_train_ds.classes)
    model = HKSDeepSetsClassifier(
        in_channels=3,
        n_times=0,
        n_classes=n_classes,
        hidden_dim=args.hidden_dim,
        include_xyz=True,
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
    best_val_acc = 0.0

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best_model.pt")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # Final test evaluation (once)
    model.load_state_dict(torch.load(save_dir / "best_model.pt", weights_only=True))
    test_acc = evaluate(model, test_loader, device)
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Model saved to {save_dir / 'best_model.pt'}")

    # Save results
    results = {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "dataset": args.dataset,
        "sign_method": args.sign_method,
        "max_points": args.max_points,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {save_dir / 'results.json'}")


if __name__ == "__main__":
    main()
