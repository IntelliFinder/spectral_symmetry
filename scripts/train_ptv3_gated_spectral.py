#!/usr/bin/env python
"""Train PTv3 with gated spectral fusion on ModelNet40.

Same recipe as train_ptv3_pointcept.py but uses GatedSpectralPTv3Classifier
which learns per-channel gates between geometric and spectral features.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from src.experiments.ptv3.augmentations import (
    Compose,
    GridSample,
    NormalizeCoord,
    RandomScale,
    RandomShift,
    ShufflePoint,
)
from src.experiments.ptv3.dataset_spectral_pointcept import (
    ModelNet40SpectralNormals,
    spectral_pointcept_collate_fn,
)
from src.experiments.ptv3.gated_classifier import GatedSpectralPTv3Classifier
from src.experiments.ptv3.lovasz_loss import LovaszLoss

GRID_SAMPLE_KEYS = ("coord", "normal", "feat", "eigvec")


def build_train_transform(grid_size=0.01):
    return Compose(
        [
            NormalizeCoord(),
            RandomScale(scale=(0.7, 1.5), anisotropic=True),
            RandomShift(shift=[(-0.2, 0.2), (-0.2, 0.2), (0, 0)]),
            GridSample(grid_size=grid_size, mode="train", keys=GRID_SAMPLE_KEYS),
            ShufflePoint(),
        ]
    )


def build_test_transform(grid_size=0.01):
    return Compose(
        [
            NormalizeCoord(),
            GridSample(grid_size=grid_size, mode="test", keys=GRID_SAMPLE_KEYS),
        ]
    )


def build_optimizer(model, lr=1e-3, weight_decay=0.01, block_lr_mult=0.1):
    """Two param groups: default lr for most params, reduced lr for 'block' params."""
    block_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "block" in name:
            block_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": other_params, "lr": lr, "weight_decay": weight_decay},
        {"params": block_params, "lr": lr * block_lr_mult, "weight_decay": weight_decay},
    ]
    return AdamW(param_groups)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate and return accuracy."""
    model.eval()
    correct = 0
    total = 0
    for data_dict, labels in loader:
        data_dict = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()
        }
        labels = labels.to(device)
        logits = model(data_dict)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_with_voting(
    model,
    test_dataset,
    device,
    num_augments=10,
    grid_size=0.01,
):
    """Test-time voting: augment each sample, sum softmax scores, take argmax."""
    model.eval()
    correct = 0
    total = len(test_dataset)

    voting_transform = Compose(
        [
            NormalizeCoord(),
            RandomScale(scale=(0.8, 1.2), anisotropic=True),
            GridSample(grid_size=grid_size, mode="test", keys=GRID_SAMPLE_KEYS),
        ]
    )

    for idx in range(total):
        raw_data = test_dataset._get_raw(idx)
        label = raw_data["label"]
        scores_sum = None

        for _ in range(num_augments):
            data = {
                "coord": raw_data["coord"].copy(),
                "normal": raw_data["normal"].copy(),
                "eigvec": raw_data["eigvec"].copy(),
                "label": label,
            }
            data["feat"] = np.concatenate([data["coord"], data["normal"]], axis=1)
            data = voting_transform(data)

            coord = torch.from_numpy(data["coord"]).float().to(device)
            feat = torch.from_numpy(data["feat"]).float().to(device)
            eigvec = torch.from_numpy(data["eigvec"]).float().to(device)
            n = coord.shape[0]

            data_dict = {
                "coord": coord,
                "feat": feat,
                "eigvec": eigvec,
                "offset": torch.tensor([n], dtype=torch.long).to(device),
            }
            if "grid_coord" in data:
                data_dict["grid_coord"] = torch.from_numpy(data["grid_coord"]).int().to(device)

            logits = model(data_dict)
            probs = softmax(logits, dim=1)
            if scores_sum is None:
                scores_sum = probs
            else:
                scores_sum = scores_sum + probs

        pred = scores_sum.argmax(dim=1).item()
        if pred == label:
            correct += 1

        if (idx + 1) % 200 == 0:
            print(f"  Voting progress: {idx + 1}/{total}, running acc={correct / (idx + 1):.4f}")

    return correct / total


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train PTv3 with gated spectral fusion on ModelNet40"
    )
    parser.add_argument("--data-dir", type=str, default="data/modelnet")
    parser.add_argument("--grid-size", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--block-lr-mult", type=float, default=0.1)
    parser.add_argument("--pct-start", type=float, default=0.05)
    parser.add_argument("--drop-path", type=float, default=0.3)
    parser.add_argument("--lovasz-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="results/ptv3_gated_spectral")
    parser.add_argument("--voting", action="store_true", help="Run voting eval at end")
    parser.add_argument("--num-augments", type=int, default=10)
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)",
    )
    # Spectral args
    parser.add_argument("--n-eigs", type=int, default=8)
    parser.add_argument(
        "--canonicalization",
        type=str,
        default="spielman",
        choices=("spielman", "maxabs", "random", "none"),
    )
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--n-neighbors", type=int, default=12)
    args = parser.parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Transforms
    train_transform = build_train_transform(grid_size=args.grid_size)
    test_transform = build_test_transform(grid_size=args.grid_size)

    # Datasets
    print("Loading training set...")
    train_ds = ModelNet40SpectralNormals(
        args.data_dir,
        split="train",
        transform=train_transform,
        n_eigs=args.n_eigs,
        n_neighbors=args.n_neighbors,
        canonicalization=args.canonicalization,
        weighted=args.weighted,
    )
    print("Loading test set...")
    test_ds = ModelNet40SpectralNormals(
        args.data_dir,
        split="test",
        transform=test_transform,
        n_eigs=args.n_eigs,
        n_neighbors=args.n_neighbors,
        canonicalization=args.canonicalization,
        weighted=args.weighted,
    )
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}, Classes: {len(train_ds.classes)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=spectral_pointcept_collate_fn,
        drop_last=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=spectral_pointcept_collate_fn,
        pin_memory=True,
    )

    # Model
    n_classes = len(train_ds.classes)
    model = GatedSpectralPTv3Classifier(
        n_eigs=args.n_eigs,
        in_channels=6,
        n_classes=n_classes,
        grid_size=args.grid_size,
        drop_path=args.drop_path,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer with two param groups
    optimizer = build_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        block_lr_mult=args.block_lr_mult,
    )

    # Scheduler: OneCycleLR stepped per iteration
    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = args.epochs * steps_per_epoch
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[args.lr, args.lr * args.block_lr_mult],
        total_steps=total_steps,
        pct_start=args.pct_start,
        anneal_strategy="cos",
    )

    # Loss
    ce_loss_fn = nn.CrossEntropyLoss()
    lovasz_loss_fn = LovaszLoss(mode="multiclass")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0

        optimizer.zero_grad()
        for step, (data_dict, labels) in enumerate(train_loader):
            data_dict = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()
            }
            labels = labels.to(device)

            logits = model(data_dict)
            loss_ce = ce_loss_fn(logits, labels)
            loss_lovasz = lovasz_loss_fn(logits, labels)
            loss = args.ce_weight * loss_ce + args.lovasz_weight * loss_lovasz

            if args.grad_accum > 1:
                loss = loss / args.grad_accum

            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size * args.grad_accum
            n_samples += batch_size

        avg_loss = total_loss / max(n_samples, 1)
        test_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  test_acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_dir / "best_model.pt")

    print(f"Best test accuracy (no voting): {best_acc:.4f}")
    print(f"Model saved to {save_dir / 'best_model.pt'}")

    # Save results
    results = {
        "best_test_acc": best_acc,
        "epochs": args.epochs,
        "n_eigs": args.n_eigs,
        "canonicalization": args.canonicalization,
        "weighted": args.weighted,
        "n_neighbors": args.n_neighbors,
        "grid_size": args.grid_size,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "drop_path": args.drop_path,
    }

    # Optional voting evaluation
    if args.voting:
        print(f"\nRunning voting evaluation ({args.num_augments} augments)...")
        model.load_state_dict(torch.load(save_dir / "best_model.pt", weights_only=True))
        test_ds_raw = ModelNet40SpectralNormals(
            args.data_dir,
            split="test",
            transform=None,
            n_eigs=args.n_eigs,
            n_neighbors=args.n_neighbors,
            canonicalization=args.canonicalization,
            weighted=args.weighted,
        )
        voting_acc = evaluate_with_voting(
            model,
            test_ds_raw,
            device,
            num_augments=args.num_augments,
            grid_size=args.grid_size,
        )
        print(f"Voting test accuracy: {voting_acc:.4f}")
        results["voting_acc"] = voting_acc

    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {save_dir / 'results.json'}")


if __name__ == "__main__":
    main()
