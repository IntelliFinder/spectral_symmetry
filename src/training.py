"""Shared training utilities for classification experiments.

Provides deterministic train/val splitting and a reusable training loop
that selects the best model by validation accuracy and reports test
accuracy once after training completes.
"""

import numpy as np
import torch
from torch.utils.data import Subset


def make_train_val_split(dataset, val_fraction=0.2, seed=42):
    """Split a dataset into train/val subsets using deterministic indices.

    Parameters
    ----------
    dataset : Dataset
        Full training dataset.
    val_fraction : float
        Fraction of data to use for validation (default 0.2).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_subset, val_subset : Subset, Subset
    """
    n = len(dataset)
    indices = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    n_val = int(n * val_fraction)
    val_indices = indices[:n_val].tolist()
    train_indices = indices[n_val:].tolist()

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def run_classification_loop(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    epochs,
    save_dir,
    device,
    train_fn,
    eval_fn,
):
    """Shared training loop with val-based model selection.

    Selects best model by validation accuracy, then evaluates once on test.

    Parameters
    ----------
    model : nn.Module
    train_loader, val_loader, test_loader : DataLoader
    criterion : loss function
    optimizer : Optimizer
    scheduler : LR scheduler (or None)
    epochs : int
    save_dir : Path
        Directory to save best_model.pt.
    device : torch.device
    train_fn : callable(model, loader, criterion, optimizer, device) -> float
        Returns average training loss for one epoch.
    eval_fn : callable(model, loader, device) -> float
        Returns accuracy (0-1) on the given loader.

    Returns
    -------
    dict with keys: best_val_acc, test_acc, history
    """
    best_val_acc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_fn(model, train_loader, criterion, optimizer, device)
        val_acc = eval_fn(model, val_loader, device)
        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch:3d}/{epochs}  loss={train_loss:.4f}  val_acc={val_acc:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best_model.pt")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # Final test evaluation (once)
    model.load_state_dict(torch.load(save_dir / "best_model.pt", weights_only=True))
    test_acc = eval_fn(model, test_loader, device)
    print(f"Final test accuracy: {test_acc:.4f}")

    return {"best_val_acc": best_val_acc, "test_acc": test_acc, "history": history}
