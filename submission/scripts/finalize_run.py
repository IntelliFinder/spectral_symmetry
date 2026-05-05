"""Finalize a training run from its saved best_model.pt without retraining.

Loads the model, runs val + test eval, optionally aug-averaged eval for
random_augmented runs, and writes results.json with the same schema that
train_molecular.py would have written.

Usage:
    python scripts/finalize_run.py <save_dir> --hidden-dim 16 --n-eigs 16 \
        --canonicalization random_augmented --seed 0 --test-aug-samples 16
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

# Ensure local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.molecular.dataset import MolecularLapPEDataset
from src.experiments.molecular.model import GINLapPE, GCNLapPE
from scripts.train_molecular import (
    evaluate,
    evaluate_with_aug_averaging,
    get_evaluator,
    transform_pe,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("save_dir", type=str)
    p.add_argument("--dataset", default="ogbg-molpcba")
    p.add_argument("--canonicalization", required=True)
    p.add_argument("--n-eigs", type=int, required=True)
    p.add_argument("--cache-n-eigs", type=int, default=None)
    p.add_argument("--pe-type", default="eigvec")
    p.add_argument("--eigval-scale", action="store_true")
    p.add_argument("--model", default="gin", choices=["gin", "gcn"])
    p.add_argument("--hidden-dim", type=int, required=True)
    p.add_argument("--num-layers", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--jumping-knowledge", action="store_true")
    p.add_argument("--epochs", type=int, default=500, help="cap that was used")
    p.add_argument("--patience", type=int, default=999)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--test-aug-samples", type=int, default=0)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default="auto")
    p.add_argument("--save-predictions", action="store_true")
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    if not (save_dir / "best_model.pt").exists():
        print(f"FATAL: {save_dir}/best_model.pt not found", file=sys.stderr)
        sys.exit(1)

    if (save_dir / "results.json").exists():
        print(f"results.json already exists in {save_dir} — skipping")
        return

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    print(f"Loading {args.dataset} with {args.canonicalization} (k={args.n_eigs})...")
    mol_dataset = MolecularLapPEDataset(
        dataset_name=args.dataset,
        canonicalization=args.canonicalization,
        n_eigs=args.n_eigs,
        data_dir=args.data_dir,
        cache_n_eigs=args.cache_n_eigs,
        eigval_scale=args.eigval_scale,
    )

    val_loader = mol_dataset.get_dataloader(
        "valid", batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = mol_dataset.get_dataloader(
        "test", batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    sample = mol_dataset.ogb_dataset[0]
    atom_dim = sample.x.shape[1] if sample.x is not None else 9
    pe_dim = args.n_eigs * (2 if args.pe_type == "both" else 1)

    model_cls = GINLapPE if args.model == "gin" else GCNLapPE
    model = model_cls(
        atom_dim=atom_dim,
        pe_dim=pe_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_tasks=mol_dataset.num_tasks,
        dropout=args.dropout,
        jumping_knowledge=args.jumping_knowledge,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    state = torch.load(save_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    evaluator = get_evaluator(args.dataset)

    final_val, _, _, _ = evaluate(model, val_loader, evaluator, device, args.dataset, args.pe_type)
    final_test, y_true, y_pred, graph_indices = evaluate(
        model, test_loader, evaluator, device, args.dataset, args.pe_type,
    )

    metric_name = "rocauc" if "moltox21" in args.dataset else "ap"
    print(f"Final val {metric_name}: {final_val:.4f}")
    print(f"Final test {metric_name}: {final_test:.4f}")

    results = {
        "dataset": args.dataset,
        "model": args.model,
        "canonicalization": args.canonicalization,
        "n_eigs": args.n_eigs,
        "pe_type": args.pe_type,
        "eigval_scale": args.eigval_scale,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "jumping_knowledge": args.jumping_knowledge,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "n_params": n_params,
        "best_epoch": -1,  # unknown — finalized externally
        "stopped_epoch": -1,  # unknown
        "patience": args.patience,
        f"best_val_{metric_name}": final_val,
        f"best_test_{metric_name}": final_test,
        "device": str(device),
        "finalized_externally": True,
    }

    if (
        args.canonicalization == "random_augmented"
        and args.test_aug_samples is not None
        and args.test_aug_samples > 0
    ):
        aug_metric, _, y_pred_avg, _ = evaluate_with_aug_averaging(
            model, mol_dataset, evaluator, device,
            dataset_name=args.dataset, pe_type=args.pe_type,
            num_samples=args.test_aug_samples, base_seed=args.seed,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        key = f"best_test_{metric_name}_aug{args.test_aug_samples}"
        results[key] = aug_metric
        results["test_aug_samples"] = args.test_aug_samples
        print(f"Final test {metric_name} (aug-avg K={args.test_aug_samples}): {aug_metric:.4f}")

    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {save_dir / 'results.json'}")


if __name__ == "__main__":
    main()
