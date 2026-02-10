#!/usr/bin/env python
"""Train PTv3 classifier on ModelNet with spectral features (xyz + eigenvectors)."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.experiments.ptv3.classifier import PTv3Classifier
from src.experiments.ptv3.dataset import PTv3SpectralModelNet, ptv3_collate_fn
from src.experiments.ptv3.train import train_one_epoch_ptv3, evaluate_ptv3


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train PTv3 on ModelNet (spectral features)")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--variant", type=int, default=10, choices=[10, 40])
    parser.add_argument("--n-points", type=int, default=512)
    parser.add_argument("--n-eigs", type=int, default=16)
    parser.add_argument("--n-neighbors", type=int, default=12)
    parser.add_argument("--grid-size", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--canonicalize", action="store_true", default=True)
    parser.add_argument("--no-canonicalize", dest="canonicalize", action="store_false")
    parser.add_argument("--save-dir", type=str, default="results/ptv3_spectral")
    args = parser.parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Datasets
    print(f"Loading ModelNet{args.variant} training set (spectral, canon={args.canonicalize})...")
    train_ds = PTv3SpectralModelNet(
        args.data_dir, split="train", n_points=args.n_points,
        n_eigs=args.n_eigs, n_neighbors=args.n_neighbors,
        download=args.download, variant=args.variant,
        canonicalize=args.canonicalize,
    )
    print(f"Loading ModelNet{args.variant} test set (spectral, canon={args.canonicalize})...")
    test_ds = PTv3SpectralModelNet(
        args.data_dir, split="test", n_points=args.n_points,
        n_eigs=args.n_eigs, n_neighbors=args.n_neighbors,
        download=False, variant=args.variant,
        canonicalize=args.canonicalize,
    )
    print(f"Train: {len(train_ds)} shapes, Test: {len(test_ds)} shapes")
    print(f"Classes: {train_ds.classes}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=ptv3_collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=ptv3_collate_fn,
    )

    # Model
    in_channels = 3 + args.n_eigs
    n_classes = len(train_ds.classes)
    model = PTv3Classifier(
        in_channels=in_channels,
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
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch_ptv3(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate_ptv3(model, test_loader, device)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  test_acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_dir / "best_model.pt")

    print(f"Best test accuracy: {best_acc:.4f}")
    print(f"Model saved to {save_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
