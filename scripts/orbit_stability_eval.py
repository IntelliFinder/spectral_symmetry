"""Per-graph orbit-stability evaluation for a random_augmented LapPE model.

Loads a saved best_model.pt and, for each test graph, runs K independent
forward passes under random elements of the eigenvector ambiguity group
(sign flips on simple eigenvalues, Haar O(m) on multiplicity blocks). For
each graph and each task we record the K predicted probabilities, then
report per-task spread (max - min) and standard deviation, aggregated
across the test set.

Output: orbit_stability.json next to the loaded model, with fields
  - K, num_graphs, num_tasks
  - mean_spread_per_task : mean across graphs of (max prob - min prob)
  - mean_spread_overall : float, mean over graphs and tasks
  - std_per_task : mean across graphs of per-graph std of probabilities
  - mean_spread_top1pct : mean spread on the most variable 1% of graphs
  - per_graph_spread_summary : mean, p50, p95, p99 of per-graph mean spread

Usage:
  python3 scripts/orbit_stability_eval.py <save_dir> \\
      --canonicalization random_augmented --n-eigs 16 --hidden-dim 128 \\
      --seed 0 --K 64 --base-seed 1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.molecular.dataset import MolecularLapPEDataset  # noqa: E402
from src.experiments.molecular.model import GCNLapPE, GINLapPE  # noqa: E402
from scripts.train_molecular import transform_pe  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("save_dir", type=str)
    p.add_argument("--dataset", default="ogbg-molpcba")
    p.add_argument("--canonicalization", default="random_augmented")
    p.add_argument("--n-eigs", type=int, required=True)
    p.add_argument("--cache-n-eigs", type=int, default=None)
    p.add_argument("--pe-type", default="eigvec")
    p.add_argument("--eigval-scale", action="store_true")
    p.add_argument("--model", default="gin", choices=["gin", "gcn"])
    p.add_argument("--hidden-dim", type=int, required=True)
    p.add_argument("--num-layers", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--jumping-knowledge", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, required=True, help="Run seed (for ckpt naming)")
    p.add_argument("--K", type=int, default=64, help="Number of orbit draws")
    p.add_argument("--base-seed", type=int, default=1, help="RNG seed offset for orbit draws")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default="auto")
    p.add_argument("--out-name", default="orbit_stability.json")
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    if not (save_dir / "best_model.pt").exists():
        print(f"FATAL: {save_dir}/best_model.pt not found", file=sys.stderr)
        sys.exit(1)

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

    state = torch.load(save_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Collect per-graph predictions across K orbit draws, ordered identically.
    # mol_dataset.get_augmented_test_loader gives a deterministic ordering
    # regardless of the orbit-draw seed.
    K = args.K
    all_probs = None  # (K, N, T)
    y_true = None

    with torch.no_grad():
        for s in range(K):
            seed = args.base_seed * 1_000_003 + s
            loader = mol_dataset.get_augmented_test_loader(
                "test", batch_size=args.batch_size, base_seed=seed, num_workers=0,
            )
            ys = []
            ps = []
            for batch in loader:
                batch = batch.to(device)
                logits = model(
                    batch.x, transform_pe(batch, args.pe_type), batch.edge_index, batch.batch
                )
                probs = torch.sigmoid(logits)
                ys.append(batch.y.cpu().numpy())
                ps.append(probs.cpu().numpy())
            y_s = np.concatenate(ys, axis=0)
            p_s = np.concatenate(ps, axis=0)

            if all_probs is None:
                N, T = p_s.shape
                all_probs = np.empty((K, N, T), dtype=np.float32)
                y_true = y_s
            all_probs[s] = p_s
            if (s + 1) % 8 == 0 or s == K - 1:
                print(f"  draw {s+1}/{K} done")

    # Per-graph, per-task spread
    spread = all_probs.max(axis=0) - all_probs.min(axis=0)  # (N, T)
    std_per = all_probs.std(axis=0, ddof=1)  # (N, T)

    # Aggregate
    mean_spread_per_task = spread.mean(axis=0).tolist()  # (T,)
    mean_std_per_task = std_per.mean(axis=0).tolist()
    mean_spread_overall = float(spread.mean())
    mean_std_overall = float(std_per.mean())

    # Per-graph mean spread (across tasks), then percentiles
    per_graph_mean_spread = spread.mean(axis=1)  # (N,)
    p50 = float(np.percentile(per_graph_mean_spread, 50))
    p95 = float(np.percentile(per_graph_mean_spread, 95))
    p99 = float(np.percentile(per_graph_mean_spread, 99))

    # Top 1% most-variable graphs
    top1pct_threshold = np.percentile(per_graph_mean_spread, 99)
    top1pct_mask = per_graph_mean_spread >= top1pct_threshold
    mean_spread_top1pct = float(per_graph_mean_spread[top1pct_mask].mean())

    # AP-by-orbit-element spread: for each orbit draw, score the model on
    # that single draw's predictions and report mean / std of resulting AP.
    from ogb.graphproppred import Evaluator

    evaluator = Evaluator(name=args.dataset)
    metric_name = "ap" if "molpcba" in args.dataset else "rocauc"
    per_draw_metric = []
    for s in range(K):
        eval_result = evaluator.eval({"y_true": y_true, "y_pred": all_probs[s]})
        per_draw_metric.append(float(eval_result[metric_name]))
    per_draw_metric = np.array(per_draw_metric)
    mean_metric = float(per_draw_metric.mean())
    std_metric = float(per_draw_metric.std(ddof=1))
    min_metric = float(per_draw_metric.min())
    max_metric = float(per_draw_metric.max())

    # Mean-prob (Reynolds) AP
    mean_probs = all_probs.mean(axis=0)
    mean_eval = evaluator.eval({"y_true": y_true, "y_pred": mean_probs.astype(np.float32)})
    mean_ap = float(mean_eval[metric_name])

    out = {
        "save_dir": str(save_dir),
        "canonicalization": args.canonicalization,
        "n_eigs": args.n_eigs,
        "hidden_dim": args.hidden_dim,
        "seed": args.seed,
        "K": K,
        "base_seed": args.base_seed,
        "num_graphs": int(spread.shape[0]),
        "num_tasks": int(spread.shape[1]),
        "mean_spread_overall": mean_spread_overall,
        "mean_std_overall": mean_std_overall,
        "mean_spread_per_task_summary": {
            "mean": float(np.mean(mean_spread_per_task)),
            "min": float(np.min(mean_spread_per_task)),
            "max": float(np.max(mean_spread_per_task)),
        },
        "per_graph_mean_spread": {
            "mean": float(per_graph_mean_spread.mean()),
            "p50": p50,
            "p95": p95,
            "p99": p99,
        },
        "mean_spread_top1pct_graphs": mean_spread_top1pct,
        "per_draw_metric": {
            "mean": mean_metric,
            "std_ddof1": std_metric,
            "min": min_metric,
            "max": max_metric,
            "spread": max_metric - min_metric,
        },
        "reynolds_mean_metric": mean_ap,
        "metric_name": metric_name,
    }

    out_path = save_dir / args.out_name
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"\nSummary:")
    print(f"  Per-graph spread (max-min prob, mean over graphs/tasks): {mean_spread_overall:.4f}")
    print(f"  Per-graph std (ddof=1, mean over graphs/tasks):           {mean_std_overall:.4f}")
    print(f"  Per-draw {metric_name}: {mean_metric:.4f} ± {std_metric:.4f} "
          f"[{min_metric:.4f}, {max_metric:.4f}]")
    print(f"  Reynolds (mean-prob) {metric_name}: {mean_ap:.4f}")


if __name__ == "__main__":
    main()
