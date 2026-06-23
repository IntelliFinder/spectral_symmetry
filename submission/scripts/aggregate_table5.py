#!/usr/bin/env python3
"""Aggregate per-run ``results.json`` files into Appendix B Table 5.

Table 5 reports the mean test AP on ``ogbg-molpcba`` for three strategies that
resolve the eigenvector ambiguity group of Laplacian positional encodings
(``map``, ``oap``, ``random_augmented``) across the eigenvector count
``k in {3, 8, 16}`` and the hidden dimension ``h in {16, 128, 512}``, averaged
over 3 seeds per cell. The AUG-K column reports test-time averaging over ``K``
ambiguity-group draws applied to the ``random_augmented``-trained model
(``K = 8`` at ``k = 3``, ``K = 16`` at ``k in {8, 16}``).

The script walks one or more results roots, reads every ``results.json`` it
finds, groups runs by ``(k, h, canonicalization)`` taken from the JSON contents
(so it is robust to the ``<base_dir>/gin/<run>/`` nesting the sweep produces),
and prints the 9-cell table. Bolding / the ``*`` marker denotes the nominal
row-best mean among the three arms -- it is not a significance claim (see the
Table 5 caption).

Only the Python standard library is used, so it runs without the training
dependencies installed.

Usage (from the ``submission/`` directory)::

    python3 scripts/aggregate_table5.py                       # text table
    python3 scripts/aggregate_table5.py --format markdown     # markdown table
    python3 scripts/aggregate_table5.py --csv table5.csv      # also write CSV
    python3 scripts/aggregate_table5.py --results-root results another_root
"""

import argparse
import csv as csv_module
import glob
import json
import os
import statistics
from collections import defaultdict

ARMS = ["map", "oap", "random_augmented"]
KS = [3, 8, 16]
HS = [16, 128, 512]


def find_results(roots):
    """Return every ``results.json`` path under the given roots (deduped)."""
    files = []
    for root in roots:
        files.extend(
            glob.glob(os.path.join(root, "**", "results.json"), recursive=True)
        )
    return sorted(set(files))


def aug_key_and_k(record, metric):
    """Return ``(json_key, K)`` for the AUG-K test metric if present.

    The training script writes the test-time-averaging result under the key
    ``best_test_<metric>_aug<K>`` (e.g. ``best_test_ap_aug8``). Returns
    ``(None, None)`` when no such key exists (i.e. non-augmented runs).
    """
    prefix = "best_test_{}_aug".format(metric)
    for key in record:
        if key.startswith(prefix):
            suffix = key[len(prefix):]
            if suffix.isdigit():
                return key, int(suffix)
    return None, None


def collect(files, metric):
    """Group runs into per-cell, per-arm lists of the test metric.

    Returns ``(cells, augk, skipped)`` where:
      * ``cells[(k, h)][arm]`` -> list of ``best_test_<metric>`` floats
      * ``augk[(k, h)]``       -> dict with ``K`` and the list of AUG-K floats
      * ``skipped``            -> count of unreadable ``results.json`` files
    """
    metric_key = "best_test_{}".format(metric)
    cells = defaultdict(lambda: defaultdict(list))
    augk = defaultdict(lambda: {"K": None, "values": []})
    skipped = 0
    for path in files:
        try:
            with open(path) as handle:
                record = json.load(handle)
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue
        arm = record.get("canonicalization")
        k = record.get("n_eigs")
        h = record.get("hidden_dim")
        if arm not in ARMS or k is None or h is None or metric_key not in record:
            continue
        cells[(k, h)][arm].append(float(record[metric_key]))
        if arm == "random_augmented":
            key, K = aug_key_and_k(record, metric)
            if key is not None:
                augk[(k, h)]["K"] = K
                augk[(k, h)]["values"].append(float(record[key]))
    return cells, augk, skipped


def summarize(values):
    """Return ``(mean, sample_std, n)`` or ``None`` when no values."""
    if not values:
        return None
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std, len(values)


def row_best_arm(cell):
    """Return the arm with the highest mean among present arms, else ``None``."""
    best_arm, best_mean = None, float("-inf")
    for arm in ARMS:
        stat = summarize(cell.get(arm, []))
        if stat is not None and stat[0] > best_mean:
            best_mean, best_arm = stat[0], arm
    return best_arm


def iter_cells(cells, augk, ks, hs):
    """Yield ``(k, h, per_arm_stats, best_arm, augk_mean, K, warnings)``."""
    for k in ks:
        for h in hs:
            cell = cells.get((k, h), {})
            per_arm = {arm: summarize(cell.get(arm, [])) for arm in ARMS}
            best = row_best_arm(cell)
            aug = augk.get((k, h), {"K": None, "values": []})
            aug_stat = summarize(aug["values"])
            warnings = []
            for arm in ARMS:
                stat = per_arm[arm]
                if stat is None:
                    warnings.append("missing {}".format(arm))
                elif stat[2] != 3:
                    warnings.append("{} has {} seed(s)".format(arm, stat[2]))
            yield k, h, per_arm, best, aug_stat, aug["K"], warnings


def mean_or_dash(stat):
    return "--" if stat is None else "{:.4f}".format(stat[0])


def print_text(rows):
    header = " k    h      map        oap        random_aug   AUG-K"
    print("Table 5 - mean test AP on ogbg-molpcba (3 seeds/cell).")
    print("'*' marks the nominal row-best arm mean (not a significance claim).")
    print(header)
    print("-" * len(header))
    for k, h, per_arm, best, aug_stat, K, warnings in rows:
        cols = []
        for arm in ARMS:
            text = mean_or_dash(per_arm[arm])
            text += "*" if arm == best and per_arm[arm] is not None else " "
            cols.append(text)
        if aug_stat is None:
            aug_text = "--"
        else:
            aug_text = "{:.4f} (K={})".format(aug_stat[0], K)
        print(
            "{:>2}  {:>4}   {:>9}  {:>9}  {:>9}   {}".format(
                k, h, cols[0], cols[1], cols[2], aug_text
            )
        )
        if warnings:
            print("        ! {}".format("; ".join(warnings)))
    print()
    print("Per-arm mean +/- sample std (ddof=1) over seeds, n in brackets:")
    for k, h, per_arm, best, aug_stat, K, warnings in rows:
        parts = []
        for arm in ARMS:
            stat = per_arm[arm]
            if stat is None:
                parts.append("{}: --".format(arm))
            else:
                parts.append(
                    "{}: {:.4f}+/-{:.4f} (n={})".format(arm, stat[0], stat[1], stat[2])
                )
        if aug_stat is not None:
            parts.append("aug{}: {:.4f}+/-{:.4f}".format(K, aug_stat[0], aug_stat[1]))
        print("  k={:<2} h={:<4} {}".format(k, h, " | ".join(parts)))


def print_markdown(rows):
    print("**Table 5 - mean test AP on `ogbg-molpcba` (3 seeds/cell).** "
          "Bold = nominal row-best arm mean (not a significance claim).")
    print()
    print("| k | h | map | oap | random_augmented | AUG-K |")
    print("|---|---|-----|-----|------------------|-------|")
    for k, h, per_arm, best, aug_stat, K, _warnings in rows:
        cols = []
        for arm in ARMS:
            text = mean_or_dash(per_arm[arm])
            if arm == best and per_arm[arm] is not None:
                text = "**{}**".format(text)
            cols.append(text)
        if aug_stat is None:
            aug_text = "--"
        else:
            aug_text = "{:.4f} (K={})".format(aug_stat[0], K)
        print("| {} | {} | {} | {} | {} | {} |".format(
            k, h, cols[0], cols[1], cols[2], aug_text))


def write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv_module.writer(handle)
        writer.writerow([
            "k", "h",
            "map_mean", "map_std", "map_n",
            "oap_mean", "oap_std", "oap_n",
            "random_augmented_mean", "random_augmented_std", "random_augmented_n",
            "row_best_arm",
            "aug_K", "aug_mean", "aug_std", "aug_n",
        ])
        for k, h, per_arm, best, aug_stat, K, _warnings in rows:
            row = [k, h]
            for arm in ARMS:
                stat = per_arm[arm]
                if stat is None:
                    row += ["", "", 0]
                else:
                    row += ["{:.6f}".format(stat[0]), "{:.6f}".format(stat[1]), stat[2]]
            row.append(best or "")
            if aug_stat is None:
                row += ["", "", "", 0]
            else:
                row += [K, "{:.6f}".format(aug_stat[0]), "{:.6f}".format(aug_stat[1]),
                        aug_stat[2]]
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results-root", nargs="+", default=["results"],
        help="One or more directories searched recursively for results.json "
             "(default: results).",
    )
    parser.add_argument(
        "--metric", default="ap",
        help="Metric stem; reads best_test_<metric> (default: ap for molpcba).",
    )
    parser.add_argument(
        "--format", choices=["text", "markdown"], default="text",
        help="Console output format (default: text).",
    )
    parser.add_argument(
        "--csv", default=None, metavar="PATH",
        help="Also write the aggregated table (with per-seed std) to this CSV.",
    )
    args = parser.parse_args()

    files = find_results(args.results_root)
    if not files:
        raise SystemExit(
            "No results.json found under: {}".format(", ".join(args.results_root))
        )

    cells, augk, skipped = collect(files, args.metric)
    rows = list(iter_cells(cells, augk, KS, HS))

    if args.format == "markdown":
        print_markdown(rows)
    else:
        print_text(rows)

    if skipped:
        print("\n[warn] skipped {} unreadable results.json file(s).".format(skipped))

    if args.csv:
        write_csv(args.csv, rows)
        print("\nWrote {}".format(args.csv))


if __name__ == "__main__":
    main()
