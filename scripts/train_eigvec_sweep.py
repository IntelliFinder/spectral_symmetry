"""Eigenvector-only sweep: compare canonicalized vs non-canonicalized across k values.

Uses ONLY eigenvector features (no xyz) with a standard transformer to isolate
the effect of sign canonicalization on spectral feature quality.
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

from src.experiments.spectral_transformer.dataset import (
    SpectralModelNet,
    TruncatedSpectralDataset,
)
from src.experiments.spectral_transformer.model import SpectralTransformerClassifier
from src.experiments.spectral_transformer.train import evaluate, train_one_epoch


def main():
    parser = argparse.ArgumentParser(description="Eigenvector-only sweep on ModelNet")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--variant", type=int, default=10, choices=[10, 40], help="ModelNet variant")
    parser.add_argument("--n-points", type=int, default=512, help="Points per shape")
    parser.add_argument("--k-eigs", type=int, default=3, help="Number of eigenvector dims to use")
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
    parser.add_argument("--canonicalize", action="store_true", help="Canonicalize eigenvector signs")
    parser.add_argument("--save-dir", type=str, default=None, help="Save directory (auto-generated if not set)")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    canon_str = "canon" if args.canonicalize else "nocanon"
    if args.save_dir is None:
        args.save_dir = f"results/eigvec_sweep/{canon_str}_k{args.k_eigs}"

    print(f"Config: k_eigs={args.k_eigs}, canonicalize={args.canonicalize}")
    print(f"Save dir: {args.save_dir}")

    # Load base dataset with n_eigs=8 (max needed)
    n_eigs_base = 8
    print(f"Loading ModelNet{args.variant} training set (n_eigs={n_eigs_base}, canon={args.canonicalize})...")
    train_base = SpectralModelNet(
        args.data_dir,
        split="train",
        n_points=args.n_points,
        n_eigs=n_eigs_base,
        n_neighbors=args.n_neighbors,
        download=args.download,
        canonicalize=args.canonicalize,
        variant=args.variant,
    )
    print(f"Loading ModelNet{args.variant} test set...")
    test_base = SpectralModelNet(
        args.data_dir,
        split="test",
        n_points=args.n_points,
        n_eigs=n_eigs_base,
        n_neighbors=args.n_neighbors,
        download=False,
        canonicalize=args.canonicalize,
        variant=args.variant,
    )

    # Wrap with TruncatedSpectralDataset: eigenvectors only, no xyz
    train_ds = TruncatedSpectralDataset(train_base, k=args.k_eigs, use_xyz=False)
    test_ds = TruncatedSpectralDataset(test_base, k=args.k_eigs, use_xyz=False)

    print(f"Train: {len(train_ds)} shapes, Test: {len(test_ds)} shapes")
    print(f"Classes: {train_ds.classes}")
    print(f"Feature dim: {args.k_eigs} (eigenvectors only, no xyz)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model: standard transformer with input_dim = k_eigs
    model = SpectralTransformerClassifier(
        input_dim=args.k_eigs,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
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
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  test_acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_dir / "best_model.pt")

    print(f"Best test accuracy: {best_acc:.4f}")
    print(f"Model saved to {save_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
