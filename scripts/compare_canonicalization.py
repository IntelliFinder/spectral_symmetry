#!/usr/bin/env python
"""Sweep k eigenvectors with canon vs no-canon on ModelNet40.

Trains a spectral transformer using only eigenvector features (no xyz)
for k in {5, 10, 15, 20} with and without sign canonicalization.
Produces a plot and JSON of best test accuracy vs k.
"""

import json
import sys
import time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.experiments.spectral_transformer.dataset import (  # noqa: E402
    SpectralModelNet,
    TruncatedSpectralDataset,
)
from src.experiments.spectral_transformer.model import SpectralTransformerClassifier  # noqa: E402
from src.experiments.spectral_transformer.train import evaluate, train_one_epoch  # noqa: E402

# ── Hyperparameters ──────────────────────────────────────────────────────────
DATA_DIR = "data"
VARIANT = 40
N_POINTS = 512
N_EIGS = 20  # load max eigenvectors; truncate per run
N_NEIGHBORS = 12
EPOCHS = 100
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
SEED = 42
SAVE_DIR = Path("results/classifier")

K_VALUES = [5, 10, 15, 20]


def run_training(train_ds, test_ds, input_dim, n_classes, label, device):
    """Train a fresh model and return best test accuracy."""
    torch.manual_seed(SEED)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SpectralTransformerClassifier(
        input_dim=input_dim,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        n_classes=n_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, test_loader, device)
        scheduler.step()

        best_acc = max(best_acc, acc)
        elapsed = time.time() - t0

        print(
            f"  [{label}] Epoch {epoch:3d}/{EPOCHS}  loss={loss:.4f}  "
            f"test_acc={acc:.4f}  best={best_acc:.4f}  time={elapsed:.1f}s"
        )

    return best_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Load full datasets once (n_eigs=20) ──────────────────────────────────
    print(f"=== Loading ModelNet{VARIANT} datasets WITHOUT canonicalization (n_eigs={N_EIGS}) ===")
    train_no = SpectralModelNet(
        DATA_DIR, split="train", n_points=N_POINTS, n_eigs=N_EIGS,
        n_neighbors=N_NEIGHBORS, download=True, canonicalize=False, variant=VARIANT,
    )
    test_no = SpectralModelNet(
        DATA_DIR, split="test", n_points=N_POINTS, n_eigs=N_EIGS,
        n_neighbors=N_NEIGHBORS, download=False, canonicalize=False, variant=VARIANT,
    )
    print(f"  Train: {len(train_no)}, Test: {len(test_no)}\n")

    print(f"=== Loading ModelNet{VARIANT} datasets WITH canonicalization (n_eigs={N_EIGS}) ===")
    train_canon = SpectralModelNet(
        DATA_DIR, split="train", n_points=N_POINTS, n_eigs=N_EIGS,
        n_neighbors=N_NEIGHBORS, download=False, canonicalize=True, variant=VARIANT,
    )
    test_canon = SpectralModelNet(
        DATA_DIR, split="test", n_points=N_POINTS, n_eigs=N_EIGS,
        n_neighbors=N_NEIGHBORS, download=False, canonicalize=True, variant=VARIANT,
    )
    print(f"  Train: {len(train_canon)}, Test: {len(test_canon)}\n")

    n_classes = len(train_no.classes)

    # ── Sweep k ──────────────────────────────────────────────────────────────
    results = {}
    for k in K_VALUES:
        print(f"\n{'='*60}")
        print(f"  k = {k} eigenvectors (input_dim = {k})")
        print(f"{'='*60}")

        # Truncated wrappers (eigenvectors only, no xyz)
        tr_no = TruncatedSpectralDataset(train_no, k=k, use_xyz=False)
        te_no = TruncatedSpectralDataset(test_no, k=k, use_xyz=False)
        tr_ca = TruncatedSpectralDataset(train_canon, k=k, use_xyz=False)
        te_ca = TruncatedSpectralDataset(test_canon, k=k, use_xyz=False)

        print(f"\n--- k={k}, no-canon ---")
        acc_no = run_training(tr_no, te_no, input_dim=k, n_classes=n_classes,
                              label=f"k={k} no-canon", device=device)

        print(f"\n--- k={k}, canon ---")
        acc_ca = run_training(tr_ca, te_ca, input_dim=k, n_classes=n_classes,
                              label=f"k={k} canon", device=device)

        results[k] = {"no_canon": acc_no, "canon": acc_ca}
        print(f"\n  k={k} => no-canon={acc_no:.4f}  canon={acc_ca:.4f}")

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'k':>4s}  {'No-Canon':>10s}  {'Canon':>10s}  {'Diff':>10s}")
    print("-" * 40)
    for k in K_VALUES:
        nc = results[k]["no_canon"]
        ca = results[k]["canon"]
        diff = ca - nc
        print(f"{k:4d}  {nc:10.4f}  {ca:10.4f}  {diff:+10.4f}")

    # ── Save JSON ────────────────────────────────────────────────────────────
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    json_data = {
        "k_values": K_VALUES,
        "results": {str(k): v for k, v in results.items()},
        "config": {
            "variant": VARIANT,
            "n_points": N_POINTS,
            "n_eigs": N_EIGS,
            "n_neighbors": N_NEIGHBORS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_layers": NUM_LAYERS,
            "dim_feedforward": DIM_FEEDFORWARD,
            "seed": SEED,
        },
    }
    json_path = SAVE_DIR / "canonicalization_vs_k.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    no_canon_accs = [results[k]["no_canon"] for k in K_VALUES]
    canon_accs = [results[k]["canon"] for k in K_VALUES]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_VALUES, no_canon_accs, "o-", label="No canonicalization", linewidth=2, markersize=8)
    ax.plot(K_VALUES, canon_accs, "s-", label="With canonicalization", linewidth=2, markersize=8)
    ax.set_xlabel("Number of eigenvectors (k)", fontsize=12)
    ax.set_ylabel("Best test accuracy", fontsize=12)
    ax.set_title(f"Spectral Transformer on ModelNet{VARIANT}: Canon vs No-Canon", fontsize=13)
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        plot_path = SAVE_DIR / f"canonicalization_vs_k.{ext}"
        fig.savefig(plot_path, dpi=150)
        print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
