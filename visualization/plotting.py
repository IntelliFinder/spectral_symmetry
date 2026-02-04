"""Consolidated visualization code for spectral symmetry analysis."""

from pathlib import Path

import numpy as np

from src.datasets import SymmetriaDataset
from src.spectral_core import analyze_spectrum

# Module-level constants
SHAPE_ORDER = ['vase', 'human', 'table', 'chair', 'airplane']
SHAPE_LABELS = {
    'vase': 'Vase (rotational)',
    'human': 'Human (bilateral)',
    'table': 'Table (four-fold)',
    'chair': 'Chair (bilateral)',
    'airplane': 'Airplane (bilateral)',
}


def _setup_matplotlib():
    """Shared matplotlib setup. Returns (matplotlib, plt) tuple."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update({
        'font.size': 9,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'figure.dpi': 150,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })
    return matplotlib, plt


def _get_representatives(n_eigs=20, n_neighbors=12):
    """Load one representative instance per shape type at noise=0."""
    ds = SymmetriaDataset(n_instances=10, n_points=1024, noise_levels=(0.0,), seed=42)
    seen = set()
    representatives = {}
    for name, pts in ds:
        shape_type = name.split('/')[1]
        if shape_type in seen:
            continue
        seen.add(shape_type)
        result = analyze_spectrum(pts, n_eigs=n_eigs, n_neighbors=n_neighbors)
        result['name'] = name
        representatives[shape_type] = (pts, result)
        if len(seen) == 5:
            break
    return representatives


def make_visualizations(results, output_dir, max_plots=10):
    """Save eigenvector distribution plots for shapes with uncanonicalizable eigenvectors."""
    try:
        _, plt = _setup_matplotlib()
    except ImportError:
        print("matplotlib not installed, skipping visualizations")
        return

    vis_dir = Path(output_dir) / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)

    plotted = 0
    for r in results:
        if plotted >= max_plots:
            break
        if not any(r['uncanonicalizable'][1:]):  # skip if only trivial eigvec
            continue

        n_eigs = min(6, len(r['scores']))
        fig, axes = plt.subplots(1, n_eigs, figsize=(3 * n_eigs, 3))
        if n_eigs == 1:
            axes = [axes]

        for i in range(n_eigs):
            ax = axes[i]
            vec = r['eigenvectors'][:, i]
            ax.hist(vec, bins=40, density=True, alpha=0.7, color='steelblue')
            ax.hist(-vec, bins=40, density=True, alpha=0.5, color='salmon')
            label = "U" if r['uncanonicalizable'][i] else ""
            ax.set_title(
                f"\u03bb{i}={r['eigenvalues'][i]:.4f}\nscore={r['scores'][i]:.3f} {label}",
                fontsize=8,
            )
            ax.tick_params(labelsize=6)

        safe_name = r['name'].replace('/', '_').replace('\\', '_')
        fig.suptitle(safe_name, fontsize=9)
        fig.tight_layout()
        fig.savefig(vis_dir / f"{safe_name}.png", dpi=120)
        plt.close(fig)
        plotted += 1


def plot_eigenvector_histograms(representatives, fig_dir):
    """Generate eigenvector histogram grid: 5 shapes x 6 eigenvectors."""
    _, plt = _setup_matplotlib()
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_eigs_show = 6
    fig, axes = plt.subplots(5, n_eigs_show, figsize=(7.0, 7.5))
    for row, shape in enumerate(SHAPE_ORDER):
        pts, r = representatives[shape]
        for col in range(n_eigs_show):
            ax = axes[row, col]
            vec = r['eigenvectors'][:, col]
            ax.hist(
                vec, bins=40, density=True, alpha=0.7, color='steelblue',
                edgecolor='none',
                label=r'$\mathrm{sort}(v)$' if col == 0 and row == 0 else None,
            )
            ax.hist(
                -vec, bins=40, density=True, alpha=0.5, color='salmon',
                edgecolor='none',
                label=r'$\mathrm{sort}(-v)$' if col == 0 and row == 0 else None,
            )
            tag = r'$\checkmark$' if r['uncanonicalizable'][col] else ''
            ax.set_title(
                f"$\\lambda_{{{col}}}$={r['eigenvalues'][col]:.3f}\n"
                f"s={r['scores'][col]:.3f} {tag}",
                fontsize=7,
            )
            if col == 0:
                ax.set_ylabel(SHAPE_LABELS[shape], fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.legend(loc='upper center', ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fig_dir / "eigenvector_histograms.pdf")
    fig.savefig(fig_dir / "eigenvector_histograms.png", dpi=200)
    plt.close(fig)
    print("  Saved eigenvector_histograms.pdf")


def plot_score_heatmap(representatives, fig_dir):
    """Generate score heatmap across shapes and eigenvector indices."""
    _, plt = _setup_matplotlib()
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_eigs_show = 6
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    score_matrix = []
    for shape in SHAPE_ORDER:
        _, r = representatives[shape]
        score_matrix.append(r['scores'][:n_eigs_show * 3])  # show up to 18
    score_matrix = np.array(score_matrix)
    n_cols = score_matrix.shape[1]

    im = ax.imshow(score_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=0.5)
    ax.set_yticks(range(5))
    ax.set_yticklabels([SHAPE_LABELS[s] for s in SHAPE_ORDER], fontsize=7)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([str(i) for i in range(n_cols)], fontsize=6)
    ax.set_xlabel('Eigenvector index')

    threshold = 5.0 / np.sqrt(1024)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Uncanonicalizability score', fontsize=8)
    cbar.ax.axhline(y=threshold, color='black', linewidth=1.5, linestyle='--')
    cbar.ax.text(1.5, threshold, f'  \u03c4={threshold:.3f}', fontsize=6, va='center')

    fig.tight_layout()
    fig.savefig(fig_dir / "score_heatmap.pdf")
    fig.savefig(fig_dir / "score_heatmap.png", dpi=200)
    plt.close(fig)
    print("  Saved score_heatmap.pdf")


def plot_fiedler_coloring(representatives, fig_dir):
    """Generate 3D point cloud colored by Fiedler vector."""
    _, plt = _setup_matplotlib()
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.0, 3.0))
    for idx, shape in enumerate(SHAPE_ORDER):
        pts, r = representatives[shape]
        comp = r['component_indices']
        pts_c = pts[comp]
        fiedler = r['eigenvectors'][:, 1]

        ax = fig.add_subplot(1, 5, idx + 1, projection='3d')
        ax.scatter(
            pts_c[:, 0], pts_c[:, 1], pts_c[:, 2],
            c=fiedler, cmap='coolwarm', s=1.0,
            vmin=-np.max(np.abs(fiedler)),
            vmax=np.max(np.abs(fiedler)),
        )
        ax.set_title(shape.capitalize(), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    fig.tight_layout()
    fig.savefig(fig_dir / "fiedler_coloring.pdf")
    fig.savefig(fig_dir / "fiedler_coloring.png", dpi=200)
    plt.close(fig)
    print("  Saved fiedler_coloring.pdf")


def generate_representative_figures(output_dir="results"):
    """Generate all representative figures. Returns representatives dict."""
    fig_dir = Path(output_dir) / "figures"
    representatives = _get_representatives()
    plot_eigenvector_histograms(representatives, fig_dir)
    plot_score_heatmap(representatives, fig_dir)
    plot_fiedler_coloring(representatives, fig_dir)
    return representatives
