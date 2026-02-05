#!/usr/bin/env python
"""Compare training with and without eigenvector sign canonicalization."""

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

from src.experiments.spectral_transformer.dataset import SpectralModelNet10  # noqa: E402
from src.experiments.spectral_transformer.model import SpectralTransformerClassifier  # noqa: E402
from src.experiments.spectral_transformer.train import evaluate, train_one_epoch  # noqa: E402

# ── Hyperparameters ──────────────────────────────────────────────────────────
DATA_DIR = "data"
N_POINTS = 512
N_EIGS = 16
N_NEIGHBORS = 12
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 3
DIM_FEEDFORWARD = 128
DROPOUT = 0.1
SEED = 42
SAVE_DIR = Path("results/classifier")


def run_training(train_ds, test_ds, label, device):
    """Train a fresh model and return per-epoch logs."""
    torch.manual_seed(SEED)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    input_dim = 3 + N_EIGS
    model = SpectralTransformerClassifier(
        input_dim=input_dim,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        n_classes=len(train_ds.classes),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    losses = []
    accuracies = []
    wall_times = []
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, test_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        losses.append(loss)
        accuracies.append(acc)
        wall_times.append(elapsed)
        best_acc = max(best_acc, acc)

        print(
            f"  [{label}] Epoch {epoch:3d}/{EPOCHS}  loss={loss:.4f}  "
            f"test_acc={acc:.4f}  time={elapsed:.1f}s"
        )

    return {
        "losses": losses,
        "accuracies": accuracies,
        "wall_times": wall_times,
        "best_acc": best_acc,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Load datasets (both variants share the same raw data) ────────────────
    print("=== Loading datasets WITHOUT canonicalization ===")
    train_no = SpectralModelNet10(
        DATA_DIR,
        split="train",
        n_points=N_POINTS,
        n_eigs=N_EIGS,
        n_neighbors=N_NEIGHBORS,
        download=True,
        canonicalize=False,
    )
    test_no = SpectralModelNet10(
        DATA_DIR,
        split="test",
        n_points=N_POINTS,
        n_eigs=N_EIGS,
        n_neighbors=N_NEIGHBORS,
        download=False,
        canonicalize=False,
    )
    print(f"  Train: {len(train_no)}, Test: {len(test_no)}, Classes: {train_no.classes}\n")

    print("=== Loading datasets WITH canonicalization ===")
    train_canon = SpectralModelNet10(
        DATA_DIR,
        split="train",
        n_points=N_POINTS,
        n_eigs=N_EIGS,
        n_neighbors=N_NEIGHBORS,
        download=False,
        canonicalize=True,
    )
    test_canon = SpectralModelNet10(
        DATA_DIR,
        split="test",
        n_points=N_POINTS,
        n_eigs=N_EIGS,
        n_neighbors=N_NEIGHBORS,
        download=False,
        canonicalize=True,
    )
    print(f"  Train: {len(train_canon)}, Test: {len(test_canon)}\n")

    # ── Train without canonicalization ───────────────────────────────────────
    print("=== Training WITHOUT canonicalization ===")
    results_no = run_training(train_no, test_no, "no-canon", device)

    # ── Train with canonicalization ──────────────────────────────────────────
    print("\n=== Training WITH canonicalization ===")
    results_canon = run_training(train_canon, test_canon, "canon", device)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  No canonicalization  — best test acc: {results_no['best_acc']:.4f}")
    print(f"  With canonicalization — best test acc: {results_canon['best_acc']:.4f}")
    diff = results_canon["best_acc"] - results_no["best_acc"]
    print(f"  Difference (canon - no-canon): {diff:+.4f}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss vs wall-clock time
    ax = axes[0]
    ax.plot(results_no["wall_times"], results_no["losses"], label="No canonicalization")
    ax.plot(results_canon["wall_times"], results_canon["losses"], label="With canonicalization")
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Training loss")
    ax.set_title("Training Loss vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Test accuracy vs wall-clock time
    ax = axes[1]
    ax.plot(results_no["wall_times"], results_no["accuracies"], label="No canonicalization")
    ax.plot(results_canon["wall_times"], results_canon["accuracies"], label="With canonicalization")
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Test Accuracy vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Spectral Transformer on ModelNet10: Canonicalization Comparison", fontsize=13)
    fig.tight_layout()

    plot_path = SAVE_DIR / "canonicalization_comparison.png"
    fig.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    main()
