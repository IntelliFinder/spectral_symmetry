"""Training loop for the Spectral Transformer classifier on ModelNet10."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .dataset import SpectralModelNet10
from .model import SpectralTransformerClassifier


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    n_samples = 0
    for features, mask, labels in loader:
        features = features.to(device)
        mask = mask.to(device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        logits = model(features, mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        n_samples += features.size(0)
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
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        logits = model(features, mask)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train Spectral Transformer on ModelNet10")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--n-points", type=int, default=512, help="Points per shape")
    parser.add_argument("--n-eigs", type=int, default=16, help="Number of eigenvectors")
    parser.add_argument("--n-neighbors", type=int, default=12, help="k-NN neighbors for graph")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--d-model", type=int, default=64, help="Transformer hidden dim")
    parser.add_argument("--nhead", type=int, default=4, help="Attention heads")
    parser.add_argument("--num-layers", type=int, default=3, help="Transformer layers")
    parser.add_argument("--dim-feedforward", type=int, default=128, help="FFN dim")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--download", action="store_true", help="Download ModelNet10 if missing")
    parser.add_argument(
        "--canonicalize",
        action="store_true",
        help="Canonicalize eigenvector signs",
    )
    parser.add_argument("--save-dir", type=str, default="results/classifier", help="Save directory")
    args = parser.parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Datasets
    print("Loading training set...")
    train_ds = SpectralModelNet10(
        args.data_dir,
        split="train",
        n_points=args.n_points,
        n_eigs=args.n_eigs,
        n_neighbors=args.n_neighbors,
        download=args.download,
        canonicalize=args.canonicalize,
    )
    print("Loading test set...")
    test_ds = SpectralModelNet10(
        args.data_dir,
        split="test",
        n_points=args.n_points,
        n_eigs=args.n_eigs,
        n_neighbors=args.n_neighbors,
        download=False,
        canonicalize=args.canonicalize,
    )
    print(f"Train: {len(train_ds)} shapes, Test: {len(test_ds)} shapes")
    print(f"Classes: {train_ds.classes}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    input_dim = 3 + args.n_eigs
    model = SpectralTransformerClassifier(
        input_dim=input_dim,
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
