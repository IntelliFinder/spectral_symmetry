#!/usr/bin/env python
"""Train PTv3 classifier on ModelNet with xyz-only features (baseline)."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.experiments.ptv3.classifier import PTv3Classifier
from src.experiments.ptv3.dataset import PTv3ModelNet, ptv3_collate_fn
from src.experiments.ptv3.train import evaluate_ptv3, train_one_epoch_ptv3
from src.training import make_train_val_split  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train PTv3 on ModelNet (xyz baseline)")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--variant", type=int, default=10, choices=[10, 40])
    parser.add_argument("--n-points", type=int, default=512)
    parser.add_argument("--grid-size", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--save-dir", type=str, default="results/ptv3_xyz")
    args = parser.parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Datasets
    print(f"Loading ModelNet{args.variant} training set...")
    full_train_ds = PTv3ModelNet(
        args.data_dir,
        split="train",
        n_points=args.n_points,
        download=args.download,
        variant=args.variant,
    )
    train_ds, val_ds = make_train_val_split(full_train_ds, val_fraction=0.2, seed=42)
    print(f"Loading ModelNet{args.variant} test set...")
    test_ds = PTv3ModelNet(
        args.data_dir,
        split="test",
        n_points=args.n_points,
        download=False,
        variant=args.variant,
    )
    print(f"Train: {len(train_ds)} shapes, Val: {len(val_ds)} shapes, Test: {len(test_ds)} shapes")
    print(f"Classes: {full_train_ds.classes}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=ptv3_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=ptv3_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=ptv3_collate_fn,
    )

    # Model
    n_classes = len(full_train_ds.classes)
    model = PTv3Classifier(
        in_channels=3,
        n_classes=n_classes,
        grid_size=args.grid_size,
        drop_path=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch_ptv3(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate_ptv3(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best_model.pt")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # Final test evaluation (once)
    model.load_state_dict(torch.load(save_dir / "best_model.pt", weights_only=True))
    test_acc = evaluate_ptv3(model, test_loader, device)
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Model saved to {save_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
