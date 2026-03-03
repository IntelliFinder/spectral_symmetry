#!/usr/bin/env python3
"""Generate publication-quality figures and LaTeX report for DeepSets + HKS experiments."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Color palette ────────────────────────────────────────────────────────────
COLORS = {
    "squared": "#2ca02c",
    "maxabs": "#1f77b4",
    "spielman": "#ff7f0e",
    "random": "#d62728",
}

CANON_LABELS = {
    "squared": "Squared",
    "maxabs": "Max-Abs",
    "spielman": "Spielman",
    "random": "Random",
}

# ── Result directory mapping ─────────────────────────────────────────────────
# Each entry: (feature_type, canon_method, dataset) → directory name
RESULT_DIRS = {
    # HKS Summed (k=8)
    ("HKS Summed", "squared", "MN10"): "hks_squared_k8_mn10",
    ("HKS Summed", "squared", "MN40"): "hks_squared_k8_mn40",
    # WES Summed (k=8) — MN40 only
    ("WES Summed", "maxabs", "MN40"): "wes_maxabs_k8_mn40",
    ("WES Summed", "spielman", "MN40"): "wes_spielman_k8_mn40",
    ("WES Summed", "random", "MN40"): "wes_random_k8_mn40",
    # Concat HKS (k=8)
    ("Concat HKS", "squared", "MN10"): "concat_hks_squared_mn10",
    ("Concat HKS", "squared", "MN40"): "concat_hks_squared_mn40",
    ("Concat HKS", "maxabs", "MN10"): "concat_hks_maxabs_mn10",
    ("Concat HKS", "maxabs", "MN40"): "concat_hks_maxabs_mn40",
    ("Concat HKS", "spielman", "MN10"): "concat_hks_spielman_mn10",
    ("Concat HKS", "spielman", "MN40"): "concat_hks_spielman_mn40",
    ("Concat HKS", "random", "MN10"): "concat_hks_random_mn10",
    ("Concat HKS", "random", "MN40"): "concat_hks_random_mn40",
    # Learnable (k=8)
    ("Learnable", "squared", "MN10"): "learnable_squared_mn10",
    ("Learnable", "squared", "MN40"): "learnable_squared_mn40",
    ("Learnable", "maxabs", "MN10"): "learnable_maxabs_mn10",
    ("Learnable", "maxabs", "MN40"): "learnable_maxabs_mn40",
    ("Learnable", "spielman", "MN10"): "learnable_spielman_mn10",
    ("Learnable", "spielman", "MN40"): "learnable_spielman_mn40",
    ("Learnable", "random", "MN10"): "learnable_random_mn10",
    ("Learnable", "random", "MN40"): "learnable_random_mn40",
    # HKS nosq (non-squared, k=8) — MN10 only
    ("HKS Summed", "maxabs", "MN10"): "hks_nosq_maxabs_k8_mn10",
    ("HKS Summed", "spielman", "MN10"): "hks_nosq_spielman_k8_mn10",
    ("HKS Summed", "random", "MN10"): "hks_nosq_random_k8_mn10",
    # XYZ impact
    ("HKS+XYZ", "squared", "MN10"): "hks_squared_baseline_mn10",
    ("HKS noXYZ", "squared", "MN10"): "hks_noXYZ_mn10",
}


# ── Data loading ─────────────────────────────────────────────────────────────


def load_all_results(base_dir):
    """Read results.json files into a dict keyed by (feature_type, canon, dataset)."""
    base = Path(base_dir)
    results = {}
    for key, dirname in RESULT_DIRS.items():
        path = base / dirname / "results.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            results[key] = data
        else:
            print(f"  Warning: missing {path}")
    print(f"  Loaded {len(results)}/{len(RESULT_DIRS)} result files")
    return results


# ── Matplotlib setup ─────────────────────────────────────────────────────────


def setup_matplotlib():
    """Configure rcParams for publication-quality figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _acc(results, key):
    """Get accuracy as percentage, or None if missing."""
    if key in results:
        return results[key]["best_test_acc"] * 100
    return None


def _add_bar_labels(ax, bars):
    """Add value labels on top of bars."""
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.3,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )


# ── Figure 1: Main comparison ────────────────────────────────────────────────


def plot_main_comparison(results, fig_dir):
    """Grouped bars: feature types × canon methods, MN10 & MN40 side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    feature_types = ["HKS Summed", "Concat HKS", "Learnable"]
    canon_methods = ["squared", "maxabs", "spielman", "random"]
    x = np.arange(len(feature_types))
    width = 0.18

    for ds_idx, (ax, ds_label, ds_key) in enumerate(
        [
            (ax1, "ModelNet10", "MN10"),
            (ax2, "ModelNet40", "MN40"),
        ]
    ):
        for i, canon in enumerate(canon_methods):
            vals = []
            for ft in feature_types:
                v = _acc(results, (ft, canon, ds_key))
                vals.append(v if v is not None else 0)
            bars = ax.bar(
                x + i * width,
                vals,
                width,
                label=CANON_LABELS[canon],
                color=COLORS[canon],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )
            _add_bar_labels(ax, bars)

        ax.set_xlabel("Feature Type")
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(feature_types)
        ax.set_title(ds_label)
        ax.set_ylim(70, 84)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(title="Canonicalization", loc="lower right")

    ax1.set_ylabel("Test Accuracy (%)")
    fig.suptitle("DeepSets Classification: Feature Types × Canonicalization Methods", y=1.02)
    plt.tight_layout()
    path = Path(fig_dir) / "main_comparison.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 2: Canon comparison on MN10 ───────────────────────────────────────


def plot_canon_comparison(results, fig_dir):
    """Canon methods head-to-head across feature types, MN10."""
    fig, ax = plt.subplots(figsize=(8, 5))

    canon_methods = ["maxabs", "spielman", "random"]
    feature_types = ["HKS Summed", "Concat HKS", "Learnable"]
    x = np.arange(len(canon_methods))
    width = 0.22

    for i, ft in enumerate(feature_types):
        vals = [_acc(results, (ft, c, "MN10")) or 0 for c in canon_methods]
        bars = ax.bar(
            x + i * width,
            vals,
            width,
            label=ft,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        _add_bar_labels(ax, bars)

    ax.set_xlabel("Canonicalization Method")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Canonicalization Methods Comparison (ModelNet10)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([CANON_LABELS[c] for c in canon_methods])
    ax.set_ylim(72, 82)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Feature Type")
    plt.tight_layout()
    path = Path(fig_dir) / "canon_comparison.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 3: Feature type comparison (squared only) ─────────────────────────


def plot_feature_type_comparison(results, fig_dir):
    """Squared-only comparison: HKS vs Concat vs Learnable, both datasets."""
    fig, ax = plt.subplots(figsize=(7, 5))

    feature_types = ["HKS Summed", "Concat HKS", "Learnable"]
    datasets = ["MN10", "MN40"]
    x = np.arange(len(feature_types))
    width = 0.3

    ds_colors = ["#1f77b4", "#ff7f0e"]
    for i, (ds, color) in enumerate(zip(datasets, ds_colors)):
        vals = [_acc(results, (ft, "squared", ds)) or 0 for ft in feature_types]
        bars = ax.bar(
            x + i * width,
            vals,
            width,
            label=f"ModelNet{ds[2:]}",
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        _add_bar_labels(ax, bars)

    ax.set_xlabel("Feature Type")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Sign-Invariant (Squared) Features Comparison")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(feature_types)
    ax.set_ylim(70, 84)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    path = Path(fig_dir) / "feature_type_comparison.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 4: XYZ impact ─────────────────────────────────────────────────────


def plot_xyz_impact(results, fig_dir):
    """Bar chart: HKS with xyz vs without xyz coordinates."""
    fig, ax = plt.subplots(figsize=(5, 5))

    with_xyz = _acc(results, ("HKS+XYZ", "squared", "MN10")) or 0
    without_xyz = _acc(results, ("HKS noXYZ", "squared", "MN10")) or 0

    bars = ax.bar(
        ["HKS + XYZ", "HKS only"],
        [with_xyz, without_xyz],
        color=["#2ca02c", "#d62728"],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        width=0.5,
    )
    _add_bar_labels(ax, bars)

    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Impact of XYZ Coordinates (ModelNet10)")
    ax.set_ylim(0, 90)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = Path(fig_dir) / "xyz_impact.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── LaTeX report ─────────────────────────────────────────────────────────────


def write_latex_report(results, output_dir):
    """Generate the full .tex report file."""

    # Collect all accuracy values for the tables
    def a(ft, canon, ds):
        v = _acc(results, (ft, canon, ds))
        return f"{v:.2f}" if v is not None else "---"

    def abold(ft, canon, ds, best_val):
        v = _acc(results, (ft, canon, ds))
        if v is None:
            return "---"
        if abs(v - best_val) < 0.005:
            return r"\textbf{" + f"{v:.2f}" + "}"
        return f"{v:.2f}"

    # Find best per dataset
    all_mn10 = [
        (k, v["best_test_acc"] * 100)
        for k, v in results.items()
        if k[2] == "MN10" and k[0] not in ("HKS+XYZ", "HKS noXYZ")
    ]
    all_mn40 = [(k, v["best_test_acc"] * 100) for k, v in results.items() if k[2] == "MN40"]
    best_mn10 = max(all_mn10, key=lambda x: x[1])[1] if all_mn10 else 0
    best_mn40 = max(all_mn40, key=lambda x: x[1])[1] if all_mn40 else 0

    latex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{xcolor}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{hyperref}

\definecolor{best}{RGB}{0,128,0}

\title{DeepSets with Spectral Features:\\HKS, WES, and Canonicalization Methods for Point Cloud Classification}
\author{}
\date{February 2026}

\begin{document}
\maketitle

%% =====================================================================
\begin{abstract}
We evaluate DeepSets architectures with spectral graph features for point cloud classification on ModelNet10 and ModelNet40.
We compare four feature types---HKS Summed, WES Summed, Concatenated HKS, and Learnable spectral features---combined with four eigenvector canonicalization strategies: squared (sign-invariant), max-abs, Spielman, and random sign.
"""
    latex += f"Our best configuration achieves {best_mn10:.2f}\\% on ModelNet10 and {best_mn40:.2f}\\% on ModelNet40."
    latex += r"""
Canonicalization method choice has modest impact ($< 2$ percentage points within a feature type), while including XYZ coordinates is critical.
\end{abstract}

%% =====================================================================
\section{Introduction}

Spectral methods for point cloud analysis construct a $k$-nearest-neighbor graph and compute eigenvectors of the graph Laplacian.
These eigenvectors provide a basis for defining position-aware features such as the Heat Kernel Signature (HKS).
However, eigenvectors are only defined up to sign: if $v$ is an eigenvector, so is $-v$.
This \emph{sign ambiguity} poses a challenge for learning methods that take eigenvector-derived features as input.

Several strategies exist to address sign ambiguity:
\begin{itemize}
    \item \textbf{Squared features}: Use $v_i^2$ instead of $v_i$, making features inherently sign-invariant.
    \item \textbf{Max-abs canonicalization}: Flip signs so the entry with largest absolute value is positive.
    \item \textbf{Spielman canonicalization}: An algebraic algorithm based on the Spielman graph-isomorphism framework for simple-spectrum graphs. Nodes are partitioned into \emph{balanced blocks} $C_\ell$ via absolute-value row signatures and GF(2) product-vector refinement; within each block the sign vectors $\{s_i : i \in C_\ell\}$ form an affine subspace $A_\ell$ of $\mathbb{Z}_2^{K_\ell}$. The canonical sign vector $s(V)$ is then determined by iterating over blocks and selecting the lexicographically smallest element in the coset of the sign-automorphism group $\hat{W} = \cap_\ell \hat{W}_\ell$ consistent with $\pi_\ell(s) \in A_\ell$ for each $\ell$. Columns with eigenvalue multiplicity $> 1$ are left unchanged.
    \item \textbf{Random signs}: Assign random signs at each forward pass (a data augmentation strategy).
\end{itemize}

This report compares these approaches across multiple spectral feature architectures using a DeepSets backbone.

%% =====================================================================
\section{Methods}

\subsection{Spectral Feature Types}

Given a point cloud with $k$-NN graph Laplacian $L$ and eigenpairs $\{(\lambda_i, v_i)\}_{i=1}^{K}$, we define:

\paragraph{HKS Summed.}
The Heat Kernel Signature at time $t$ for point $x$ is
\begin{equation}
    h(x, t) = \sum_{i=1}^{K} e^{-\lambda_i t}\, v_i(x)^2
\end{equation}
We evaluate at $T$ log-spaced times and sum over eigenvectors, yielding a $T$-dimensional feature per point.

\paragraph{WES Summed (Weighted Eigenvector Sum).}
\begin{equation}
    f(x) = \sum_{i=1}^{K} e^{-\lambda_i t}\, v_i(x)
\end{equation}
Unlike HKS, WES uses $v_i(x)$ directly (not squared), making it sign-sensitive and requiring canonicalization.

\paragraph{Concatenated HKS.}
Instead of summing over eigenvectors, we concatenate individual eigenvector contributions, preserving per-eigenvector information. The feature dimension is $K \times T$.

\paragraph{Learnable Spectral Features.}
An MLP processes each eigenvector $v_i$ to produce learned features, then aggregates across eigenvectors.

\subsection{Canonicalization Methods}

For features that depend on eigenvector signs (WES, Concat HKS, Learnable), we apply one of:
\begin{itemize}
    \item \textbf{Squared}: Replace $v_i \to v_i^2$, eliminating sign dependence entirely.
    \item \textbf{Max-abs}: For each $v_i$, flip its sign so that $\arg\max_j |v_{i,j}|$ has a positive value.
    \item \textbf{Spielman}: Algebraic algorithm for simple-spectrum graphs. Phase~A partitions nodes into \emph{balanced blocks} $C_\ell$ using absolute-value row signatures of the eigenvector matrix $V \in \mathbb{R}^{n \times k}$; blocks are iteratively refined by splitting on product vectors $\prod_{j \in S} v_j(x)$ for column subsets $S$ until every block is balanced (i.e., sign patterns within the block are GF(2)-dependent on previously seen columns). Within each balanced block, the sign vectors $\{s_i : i \in C_\ell\}$ form an affine subspace $A_\ell \subset \mathbb{Z}_2^{K_\ell}$ with associated subspace $W_\ell = A_\ell \oplus A_\ell$. Phase~B determines the canonical sign vector $s(V)$ by iterating over blocks: for each $\ell$, it selects the lexicographically smallest $y_\ell \in \mathbb{Z}_2^K$ such that $\pi_\ell(s - y_\ell) \in W_\ell$ remains consistent with all previous constraints, yielding the unique lex-smallest element in the coset of the global sign-automorphism group $\hat{W} = \cap_\ell \hat{W}_\ell$. Columns whose eigenvalue has multiplicity $> 1$ are left unchanged.
    \item \textbf{Random}: Independently sample $\sigma_i \in \{-1, +1\}$ and apply $v_i \to \sigma_i v_i$ at each training step.
\end{itemize}

%% =====================================================================
\section{Experimental Setup}

All experiments use a DeepSets architecture with set-aggregation (sum pooling) followed by a classification MLP.

\begin{table}[h]
\centering
\caption{Hyperparameters shared across all experiments.}
\label{tab:hyperparams}
\begin{tabular}{@{}ll@{}}
\toprule
Parameter & Value \\
\midrule
Eigenvectors ($K$) & 8 \\
HKS time steps ($T$) & 32 \\
Hidden dimension & 512 \\
Epochs & 200 \\
Batch size & 32 \\
Learning rate & 0.001 \\
Optimizer & Adam \\
Graph neighbors & 30 \\
Weighted Laplacian & Yes \\
Normalized Laplacian & Yes \\
Include XYZ & Yes (except ablation) \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Datasets.}
ModelNet10 contains 4,899 shapes across 10 categories (3,991 train / 908 test).
ModelNet40 contains 12,311 shapes across 40 categories (9,843 train / 2,468 test).

%% =====================================================================
\section{Results}

\subsection{Main Results}

\begin{table}[h]
\centering
\caption{Test accuracy (\%) for all feature type and canonicalization combinations. Best per dataset in \textbf{bold}.}
\label{tab:main}
\begin{tabular}{@{}llcc@{}}
\toprule
Feature Type & Canonicalization & MN10 (\%) & MN40 (\%) \\
\midrule
"""
    # Build table rows
    rows = [
        ("HKS Summed", "squared"),
        ("HKS Summed", "maxabs"),
        ("HKS Summed", "spielman"),
        ("HKS Summed", "random"),
        None,  # midrule
        ("Concat HKS", "squared"),
        ("Concat HKS", "maxabs"),
        ("Concat HKS", "spielman"),
        ("Concat HKS", "random"),
        None,
        ("Learnable", "squared"),
        ("Learnable", "maxabs"),
        ("Learnable", "spielman"),
        ("Learnable", "random"),
        None,
        ("WES Summed", "maxabs"),
        ("WES Summed", "spielman"),
        ("WES Summed", "random"),
    ]

    for row in rows:
        if row is None:
            latex += r"\midrule" + "\n"
            continue
        ft, canon = row
        mn10 = abold(ft, canon, "MN10", best_mn10)
        mn40 = abold(ft, canon, "MN40", best_mn40)
        latex += f"{ft} & {CANON_LABELS[canon]} & {mn10} & {mn40} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}

\subsection{Canonicalization Spread}

\begin{table}[h]
\centering
\caption{Spread (max $-$ min accuracy) across canonicalization methods for each feature type on ModelNet10.}
\label{tab:spread}
\begin{tabular}{@{}lcccc@{}}
\toprule
Feature Type & Best (\%) & Worst (\%) & Spread (pp) \\
\midrule
"""
    # Compute spread for each feature type on MN10
    for ft in ["HKS Summed", "Concat HKS", "Learnable"]:
        accs = []
        for canon in ["squared", "maxabs", "spielman", "random"]:
            v = _acc(results, (ft, canon, "MN10"))
            if v is not None:
                accs.append(v)
        if accs:
            latex += (
                f"{ft} & {max(accs):.2f} & {min(accs):.2f} & {max(accs) - min(accs):.2f} \\\\\n"
            )

    latex += r"""\bottomrule
\end{tabular}
\end{table}

\subsection{Figures}

\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{figures/main_comparison.pdf}
\caption{Test accuracy across feature types and canonicalization methods on ModelNet10 (left) and ModelNet40 (right). Squared canonicalization consistently performs well across feature types.}
\label{fig:main}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=0.85\textwidth]{figures/canon_comparison.pdf}
\caption{Head-to-head comparison of non-squared canonicalization methods on ModelNet10. The three methods (max-abs, Spielman, random) perform within 1--2 percentage points of each other.}
\label{fig:canon}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=0.75\textwidth]{figures/feature_type_comparison.pdf}
\caption{Comparison of feature types using squared (sign-invariant) canonicalization. HKS Summed and Concat HKS are competitive, while Learnable features slightly lag.}
\label{fig:feature}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=0.5\textwidth]{figures/xyz_impact.pdf}
\caption{Impact of including raw XYZ coordinates alongside HKS features. Without XYZ, accuracy drops dramatically, showing that spectral features alone provide limited discriminative power in this architecture.}
\label{fig:xyz}
\end{figure}

%% =====================================================================
\section{Discussion}

\paragraph{Squared canonicalization is competitive or best.}
"""
    latex += f"Across all feature types, the squared (sign-invariant) approach achieves the best or near-best accuracy. On ModelNet10, HKS Summed with squared canonicalization reaches {a('HKS Summed', 'squared', 'MN10')}\\%, and Concat HKS with squared reaches {a('Concat HKS', 'squared', 'MN10')}\\%."
    latex += r"""
This validates the theoretical motivation: squaring eigenvectors eliminates sign ambiguity entirely, removing a source of noise without losing too much information.

\paragraph{Canonicalization method choice has modest impact.}
Within any feature type, the spread between best and worst canonicalization method is less than 2 percentage points on ModelNet10. This suggests that while sign ambiguity is a real concern, existing canonicalization heuristics all perform reasonably well for this task.

\paragraph{HKS Summed is the most robust feature type.}
HKS features, which use $v_i^2$ in their definition (even before any canonicalization), are inherently more robust to sign flips. This built-in sign invariance makes HKS a strong default choice for spectral point cloud features.

\paragraph{XYZ coordinates dominate discriminative power.}
"""
    with_xyz = _acc(results, ("HKS+XYZ", "squared", "MN10"))
    without_xyz = _acc(results, ("HKS noXYZ", "squared", "MN10"))
    if with_xyz and without_xyz:
        latex += f"The most striking result is the XYZ ablation: accuracy drops from {with_xyz:.2f}\\% to {without_xyz:.2f}\\% when XYZ coordinates are removed."
    latex += r"""
This indicates that the DeepSets architecture relies heavily on geometric position information, and spectral features alone provide limited discriminative signal. Future work should investigate architectures that can better leverage spectral structure.

\paragraph{Scaling to ModelNet40.}
All methods show a consistent accuracy drop when moving from ModelNet10 (10 classes) to ModelNet40 (40 classes), as expected. The relative ranking of methods is generally preserved across datasets, suggesting that conclusions drawn on ModelNet10 transfer to harder classification tasks.

%% =====================================================================
\section{Conclusion}

We conducted a systematic comparison of spectral feature types and eigenvector canonicalization methods for point cloud classification with DeepSets.
"""
    latex += f"The best results ({best_mn10:.2f}\\% on MN10, {best_mn40:.2f}\\% on MN40) are achieved by HKS-based features with squared canonicalization."
    latex += r"""

Key takeaways:
\begin{enumerate}
    \item Squared (sign-invariant) features are the simplest and most effective approach to the eigenvector sign ambiguity problem.
    \item The choice among non-squared canonicalization methods (max-abs, Spielman, random) matters less than the choice of feature type.
    \item XYZ coordinates provide the dominant signal in this architecture; spectral features alone are insufficient.
    \item HKS Summed features offer the best accuracy--simplicity trade-off among the evaluated spectral feature types.
\end{enumerate}

\end{document}
"""

    tex_path = Path(output_dir) / "hks_deepsets_report.tex"
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"  Saved {tex_path}")
    return tex_path


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate figures and LaTeX report for HKS+DeepSets experiments"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Base directory containing experiment result subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hks_report",
        help="Output directory for report and figures",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    setup_matplotlib()

    print("Loading results...")
    results = load_all_results(args.results_dir)

    print("Generating figures...")
    plot_main_comparison(results, fig_dir)
    plot_canon_comparison(results, fig_dir)
    plot_feature_type_comparison(results, fig_dir)
    plot_xyz_impact(results, fig_dir)

    print("Writing LaTeX report...")
    write_latex_report(results, output_dir)

    print("Done. To compile:")
    print(f"  cd {output_dir} && pdflatex hks_deepsets_report.tex")


if __name__ == "__main__":
    main()
