#!/usr/bin/env python
"""Train Deep Sets with concatenated heat kernel features (no summation).

Uses HKSDeepSetsClassifier with per-point features where each eigenvector's
contribution is kept separate: (N, K*T) features instead of (N, T).
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

from src.experiments.deep_sets.dataset_concat_hks import ConcatHKSModelNet  # noqa: E402
from src.experiments.deep_sets.model_hks import HKSDeepSetsClassifier  # noqa: E402
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
        description="Train Deep Sets with concatenated heat kernel features"
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
    parser.add_argument("--n-eigs", type=int, default=8, help="Number of eigenpairs (K)")
    parser.add_argument(
        "--n-neighbors", type=int, default=30, help="k-NN neighbors for graph construction"
    )
    parser.add_argument(
        "--n-times", type=int, default=32, help="Number of time samples per eigenvector (T)"
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
        "--no-xyz", action="store_true", help="Exclude xyz coords; use only spectral features"
    )
    parser.add_argument(
        "--no-squared",
        action="store_true",
        help="Use v_k(i) instead of v_k(i)^2 (requires canonicalization)",
    )
    parser.add_argument(
        "--canonicalization",
        type=str,
        default="none",
        choices=["spielman", "maxabs", "random", "none"],
        help="Eigenvector canonicalization method (only used with --no-squared)",
    )
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0=safe)")
    parser.add_argument("--save-dir", type=str, default="results/concat_hks", help="Save directory")
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
    use_squared = not args.no_squared

    spectral_dim = args.n_eigs * args.n_times

    # Datasets
    print(
        f"Loading {args.dataset} training set "
        f"(n_eigs={args.n_eigs}, n_times={args.n_times}, "
        f"spectral_dim={spectral_dim}, "
        f"weighted={weighted}, normalized={normalized}, "
        f"use_squared={use_squared}, canonicalization={args.canonicalization})..."
    )
    full_train_ds = ConcatHKSModelNet(
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
        use_squared=use_squared,
        canonicalization=args.canonicalization,
    )
    train_ds, val_ds = make_train_val_split(full_train_ds, val_fraction=0.2, seed=42)
    print(f"Loading {args.dataset} test set...")
    test_ds = ConcatHKSModelNet(
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
        use_squared=use_squared,
        canonicalization=args.canonicalization,
    )
    print(f"Train: {len(train_ds)} shapes, Val: {len(val_ds)} shapes, Test: {len(test_ds)} shapes")
    print(f"Classes: {full_train_ds.classes}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Model -- pass K*T as n_times so the model sees all concatenated features
    n_classes = len(full_train_ds.classes)
    in_channels = 3 if include_xyz else 0
    model = HKSDeepSetsClassifier(
        in_channels=in_channels,
        n_times=spectral_dim,
        n_classes=n_classes,
        hidden_dim=args.hidden_dim,
        include_xyz=include_xyz,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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

    # Save results
    results = {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "dataset": args.dataset,
        "n_eigs": args.n_eigs,
        "n_times": args.n_times,
        "spectral_dim": spectral_dim,
        "weighted": weighted,
        "normalized": normalized,
        "include_xyz": include_xyz,
        "use_squared": use_squared,
        "canonicalization": args.canonicalization,
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
