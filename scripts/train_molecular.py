#!/usr/bin/env python
"""Train GIN + LapPE on OGB molecular property prediction.

Supports configurable canonicalization, hidden_dim, num_layers, and seed.
Logs per-epoch metrics (train loss, val/test metric, wall-clock time) and
optionally saves per-graph test predictions for subset analysis.

Usage
-----
    python scripts/train_molecular.py --dataset ogbg-moltox21 --canonicalization spielman
    python scripts/train_molecular.py --dataset ogbg-molpcba --hidden-dim 256 --seed 42
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.molecular.dataset import (  # noqa: E402
    CANONICALIZATION_METHODS,
    MolecularLapPEDataset,
)
from src.experiments.molecular.model import GINLapPE  # noqa: E402
from src.training import seed_everything, worker_init_fn  # noqa: E402


def get_evaluator(dataset_name):
    """Return the OGB evaluator for the dataset."""
    from ogb.graphproppred import Evaluator

    return Evaluator(name=dataset_name)


def train_one_epoch(model, loader, optimizer, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        x_atom = batch.x
        x_pe = batch.x_pe
        logits = model(x_atom, x_pe, batch.edge_index, batch.batch)

        # BCEWithLogitsLoss with NaN masking (OGB has missing labels)
        y = batch.y.float()
        is_valid = ~torch.isnan(y)
        loss = nn.functional.binary_cross_entropy_with_logits(logits[is_valid], y[is_valid])

        if torch.isnan(loss):
            raise RuntimeError("NaN loss detected")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * int(batch.y.shape[0])
        n_graphs += int(batch.y.shape[0])

    return total_loss / max(n_graphs, 1)


@torch.no_grad()
def evaluate(model, loader, evaluator, device, dataset_name):
    """Evaluate and return metric (ROC-AUC or AP).

    Returns
    -------
    metric : float
    y_true : ndarray (N, num_tasks)
    y_pred : ndarray (N, num_tasks)
    graph_indices : ndarray (N,)
    """
    model.eval()
    y_true_list = []
    y_pred_list = []
    graph_idx_list = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.x_pe, batch.edge_index, batch.batch)
        y_true_list.append(batch.y.cpu().numpy())
        y_pred_list.append(logits.cpu().numpy())
        if hasattr(batch, "graph_idx"):
            graph_idx_list.append(batch.graph_idx.cpu().numpy().reshape(-1))

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    if graph_idx_list:
        graph_indices = np.concatenate(graph_idx_list, axis=0)
    else:
        graph_indices = np.arange(len(y_true))

    eval_result = evaluator.eval({"y_true": y_true, "y_pred": y_pred})

    if "ogbg-moltox21" in dataset_name:
        metric = eval_result["rocauc"]
    elif "ogbg-molpcba" in dataset_name:
        metric = eval_result["ap"]
    else:
        metric = list(eval_result.values())[0]

    return metric, y_true, y_pred, graph_indices


def main():
    parser = argparse.ArgumentParser(
        description="Train GIN + LapPE on OGB molecular property prediction"
    )
    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbg-moltox21",
        choices=["ogbg-moltox21", "ogbg-molpcba"],
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument(
        "--canonicalization",
        type=str,
        default="spielman",
        choices=list(CANONICALIZATION_METHODS),
    )
    parser.add_argument("--n-eigs", type=int, default=8)

    # Model
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--jumping-knowledge", action="store_true")

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)

    # Output
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save per-graph test predictions for subset analysis",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    seed_everything(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Save directory
    if args.save_dir is None:
        args.save_dir = (
            f"results/molecular_{args.dataset}_{args.canonicalization}"
            f"_h{args.hidden_dim}_s{args.seed}"
        )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    print(
        f"Loading {args.dataset} with {args.canonicalization} canonicalization (k={args.n_eigs})..."
    )
    mol_dataset = MolecularLapPEDataset(
        dataset_name=args.dataset,
        canonicalization=args.canonicalization,
        n_eigs=args.n_eigs,
        data_dir=args.data_dir,
    )

    split_idx = mol_dataset.get_split_indices()
    num_tasks = mol_dataset.num_tasks

    # Build dataloaders from split indices
    train_loader = mol_dataset.get_dataloader(
        "train",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = mol_dataset.get_dataloader(
        "valid",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    test_loader = mol_dataset.get_dataloader(
        "test",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    print(f"  Train: {len(split_idx['train'])} graphs")
    print(f"  Valid: {len(split_idx['valid'])} graphs")
    print(f"  Test:  {len(split_idx['test'])} graphs")
    print(f"  Tasks: {num_tasks}")

    # Determine atom feature dimension from first graph
    sample = mol_dataset.ogb_dataset[0]
    atom_dim = sample.x.shape[1] if sample.x is not None else 9

    # Model
    model = GINLapPE(
        atom_dim=atom_dim,
        pe_dim=args.n_eigs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_tasks=num_tasks,
        dropout=args.dropout,
        jumping_knowledge=args.jumping_knowledge,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Evaluator
    evaluator = get_evaluator(args.dataset)

    # Training loop
    best_val_metric = -float("inf")
    best_epoch = 0
    epoch_log = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metric, _, _, _ = evaluate(model, val_loader, evaluator, device, args.dataset)

        scheduler.step()
        epoch_time = time.time() - epoch_start
        wall_time = time.time() - start_time

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_metric": val_metric,
            "epoch_time_s": epoch_time,
            "wall_time_s": wall_time,
            "lr": optimizer.param_groups[0]["lr"],
        }
        epoch_log.append(entry)

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_epoch = epoch
            torch.save(model.state_dict(), save_dir / "best_model.pt")

        if epoch % 10 == 0 or epoch == 1:
            metric_name = "ROC-AUC" if "moltox21" in args.dataset else "AP"
            print(
                f"Epoch {epoch:3d}/{args.epochs}  "
                f"loss={train_loss:.4f}  "
                f"val_{metric_name}={val_metric:.4f}  "
                f"({epoch_time:.1f}s)"
            )

    # Final evaluation with best model
    model.load_state_dict(torch.load(save_dir / "best_model.pt", weights_only=True))
    final_val, _, _, _ = evaluate(model, val_loader, evaluator, device, args.dataset)
    final_test, y_true, y_pred, graph_indices = evaluate(
        model,
        test_loader,
        evaluator,
        device,
        args.dataset,
    )

    metric_name = "rocauc" if "moltox21" in args.dataset else "ap"
    print(f"\nBest epoch: {best_epoch}")
    print(f"Final val {metric_name}: {final_val:.4f}")
    print(f"Final test {metric_name}: {final_test:.4f}")

    # Save results
    results = {
        "dataset": args.dataset,
        "canonicalization": args.canonicalization,
        "n_eigs": args.n_eigs,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "jumping_knowledge": args.jumping_knowledge,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "n_params": n_params,
        "best_epoch": best_epoch,
        f"best_val_{metric_name}": final_val,
        f"best_test_{metric_name}": final_test,
        "device": str(device),
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save epoch log
    with open(save_dir / "epoch_log.json", "w") as f:
        json.dump(epoch_log, f, indent=2)

    # Save per-graph predictions if requested
    if args.save_predictions:
        np.savez(
            save_dir / "test_predictions.npz",
            y_true=y_true,
            y_pred=y_pred,
            graph_indices=graph_indices,
        )
        print(f"Saved test predictions to {save_dir / 'test_predictions.npz'}")

    print(f"Results saved to {save_dir}")


if __name__ == "__main__":
    main()
