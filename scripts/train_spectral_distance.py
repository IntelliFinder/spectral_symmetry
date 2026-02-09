"""Train spectral distance model on ModelNet.

Combines k-NN distance attention (multiplicative) with invariant spectral
distance bias (additive). Spectral distances are computed from Laplacian
eigenvectors, grouped by eigenvalue multiplicity to ensure sign- and
rotation-invariance.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.spectral_transformer.dataset import SpectralDistanceModelNet
from src.experiments.spectral_transformer.model import SpectralDistanceTransformerClassifier
from src.experiments.spectral_transformer.train import (
    evaluate_spectral_dist,
    train_one_epoch_spectral_dist,
)


def main():
    parser = argparse.ArgumentParser(description="Train spectral distance model on ModelNet")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--variant", type=int, default=10, choices=[10, 40], help="ModelNet variant")
    parser.add_argument("--n-points", type=int, default=512, help="Points per shape")
    parser.add_argument("--n-eigs", type=int, default=16, help="Number of eigenvectors")
    parser.add_argument("--n-neighbors", type=int, default=12, help="k-NN neighbors")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--d-model", type=int, default=128, help="Transformer hidden dim")
    parser.add_argument("--nhead", type=int, default=8, help="Attention heads")
    parser.add_argument("--num-layers", type=int, default=4, help="Transformer layers")
    parser.add_argument("--dim-feedforward", type=int, default=256, help="FFN dim")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--download", action="store_true", help="Download ModelNet if missing")
    parser.add_argument("--save-dir", type=str, default="results/spectral_distance", help="Save directory")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Datasets
    print(f"Loading ModelNet{args.variant} training set (spectral distances)...")
    train_ds = SpectralDistanceModelNet(
        args.data_dir,
        split="train",
        n_points=args.n_points,
        n_eigs=args.n_eigs,
        n_neighbors=args.n_neighbors,
        download=args.download,
        variant=args.variant,
    )
    print(f"Loading ModelNet{args.variant} test set (spectral distances)...")
    test_ds = SpectralDistanceModelNet(
        args.data_dir,
        split="test",
        n_points=args.n_points,
        n_eigs=args.n_eigs,
        n_neighbors=args.n_neighbors,
        download=False,
        variant=args.variant,
    )
    print(f"Train: {len(train_ds)} shapes, Test: {len(test_ds)} shapes")
    print(f"Classes: {train_ds.classes}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = SpectralDistanceTransformerClassifier(
        input_dim=3,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        n_spectral_channels=args.n_eigs,
        dropout=args.dropout,
        n_classes=len(train_ds.classes),
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
        train_loss = train_one_epoch_spectral_dist(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate_spectral_dist(model, test_loader, device)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  test_acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_dir / "best_model.pt")

    print(f"Best test accuracy: {best_acc:.4f}")
    print(f"Model saved to {save_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
