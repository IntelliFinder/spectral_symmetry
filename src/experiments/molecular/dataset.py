"""OGB molecular graph dataset with configurable Laplacian PE canonicalization.

Loads OGB graph property prediction datasets, computes Laplacian eigenpairs on
each graph's largest connected component, applies a chosen canonicalization
method, and caches the resulting LapPE features to disk.

Supported canonicalization methods:
  - ``spielman``: Spielman balanced-block sign canonicalization
  - ``maxabs``: max-absolute-value sign convention
  - ``random_fixed``: deterministic random signs (seeded by graph index)
  - ``random_augmented``: random signs each call (data augmentation)
  - ``map``: Maximal Axis Projection (Ma et al., NeurIPS 2023)
  - ``oap``: Orthogonalized Axis Projection (Ma et al., NeurIPS 2024)
  - ``none``: raw eigsh output
"""

import os
import pickle

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.spectral_canonicalization import (
    CANONICALIZATION_METHODS,
    canonicalize,
)
from src.spectral_core import (
    _largest_connected_component,
    compute_eigenpairs,
)


def _patch_torch_load():
    """Monkey-patch torch.load for PyTorch 2.6+ compatibility with OGB."""
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
    return _orig


def _restore_torch_load(orig):
    torch.load = orig


def _edge_index_to_laplacian(edge_index_np, num_nodes):
    """Build combinatorial Laplacian on LCC from edge_index numpy array.

    Returns
    -------
    L : sparse CSR matrix
    lcc_node_mask : ndarray of bool (num_nodes,), True for nodes in LCC
    lcc_indices : ndarray of int, original node indices in LCC
    """
    row, col = edge_index_np[0], edge_index_np[1]
    mask = row != col
    row, col = row[mask], col[mask]

    data = np.ones(len(row), dtype=np.float64)
    A = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    A = A.maximum(A.T)
    A.data[:] = 1.0

    A, lcc_indices = _largest_connected_component(A)
    A = sp.csr_matrix(A)
    degrees = np.array(A.sum(axis=1)).flatten()
    L = sp.diags(degrees) - A

    lcc_mask = np.zeros(num_nodes, dtype=bool)
    lcc_mask[lcc_indices] = True

    return L, lcc_mask, lcc_indices


def _apply_canonicalization(eigenvectors, eigenvalues, method, graph_idx):
    """Apply canonicalization to eigenvectors.

    Thin wrapper around ``src.spectral_canonicalization.canonicalize``.
    """
    return canonicalize(eigenvectors, eigenvalues=eigenvalues, method=method, sample_idx=graph_idx)


def _compute_lappe_for_graph(edge_index_np, num_nodes, n_eigs, method, graph_idx):
    """Compute LapPE for a single graph.

    Returns
    -------
    pe : ndarray (num_nodes, n_eigs), LapPE values (zero for non-LCC nodes)
    eigenvalues : ndarray (n_eigs,), padded with zeros
    success : bool
    """
    pe = np.zeros((num_nodes, n_eigs), dtype=np.float32)
    evals_out = np.zeros(n_eigs, dtype=np.float32)

    if num_nodes < 3:
        return pe, evals_out, False

    try:
        L, lcc_mask, lcc_indices = _edge_index_to_laplacian(edge_index_np, num_nodes)
    except Exception:
        return pe, evals_out, False

    n_lcc = L.shape[0]
    if n_lcc < 3:
        return pe, evals_out, False

    k = min(n_eigs, n_lcc - 2)
    if k < 1:
        return pe, evals_out, False

    try:
        eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=k)
    except Exception:
        return pe, evals_out, False

    if len(eigenvalues) == 0:
        return pe, evals_out, False

    # Apply canonicalization (random_augmented is deferred to __getitem__)
    if method != "random_augmented":
        eigenvectors = _apply_canonicalization(eigenvectors, eigenvalues, method, graph_idx)

    n_actual = eigenvectors.shape[1]
    pe[lcc_indices, :n_actual] = eigenvectors.astype(np.float32)
    evals_out[:n_actual] = eigenvalues[:n_actual].astype(np.float32)

    return pe, evals_out, True


class _SplitView:
    """Lightweight view into a MolecularLapPEDataset for a specific split."""

    def __init__(self, parent, indices, split="train"):
        self.parent = parent
        self.indices = indices
        self.split = split

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        graph_idx = self.indices[idx]
        augment = self.split == "train"
        return self.parent._get_by_graph_idx(graph_idx, augment=augment)


class MolecularLapPEDataset:
    """OGB molecular dataset with precomputed Laplacian PE.

    Wraps an OGB ``PygGraphPropPredDataset``, precomputes (or loads from cache)
    LapPE features for each graph, and returns PyG ``Data`` objects augmented
    with ``x_pe`` (LapPE node features).

    Parameters
    ----------
    dataset_name : str
        OGB dataset name (e.g. ``"ogbg-moltox21"``).
    canonicalization : str
        One of :data:`CANONICALIZATION_METHODS`.
    n_eigs : int
        Number of Laplacian eigenvectors per graph.
    data_dir : str
        Root directory for OGB data.
    cache_dir : str or None
        Directory for caching LapPE features. If None, uses
        ``data_dir/lappe_cache/<dataset>_<method>_k<n_eigs>/``.
    split : str or None
        If set, restrict to this split (``"train"``, ``"valid"``, ``"test"``).
    """

    def __init__(
        self,
        dataset_name="ogbg-moltox21",
        canonicalization="spielman",
        n_eigs=8,
        data_dir="data",
        cache_dir=None,
        split=None,
    ):
        if canonicalization not in CANONICALIZATION_METHODS:
            raise ValueError(
                f"canonicalization must be one of {CANONICALIZATION_METHODS}, "
                f"got {canonicalization!r}"
            )

        self.dataset_name = dataset_name
        self.canonicalization = canonicalization
        self.n_eigs = n_eigs
        self.split = split

        # Load OGB dataset
        orig = _patch_torch_load()
        try:
            from ogb.graphproppred import PygGraphPropPredDataset

            self.ogb_dataset = PygGraphPropPredDataset(name=dataset_name, root=data_dir)
        finally:
            _restore_torch_load(orig)

        self.split_dict = self.ogb_dataset.get_idx_split()
        self.num_tasks = self.ogb_dataset.num_tasks

        # Determine which indices to use
        if split is not None:
            self.indices = self.split_dict[split].numpy().tolist()
        else:
            self.indices = list(range(len(self.ogb_dataset)))

        # Cache directory
        if cache_dir is None:
            cache_dir = os.path.join(
                data_dir, "lappe_cache", f"{dataset_name}_{canonicalization}_k{n_eigs}"
            )
        self.cache_dir = cache_dir

        # Precompute or load LapPE
        self._pe_data = {}  # graph_idx -> (pe, eigenvalues)
        self._precompute_lappe()

    def _cache_path(self):
        return os.path.join(self.cache_dir, "lappe.pkl")

    def _precompute_lappe(self):
        """Compute LapPE for all graphs, using disk cache if available."""
        cache_path = self._cache_path()

        if os.path.exists(cache_path) and self.canonicalization != "random_augmented":
            with open(cache_path, "rb") as f:
                self._pe_data = pickle.load(f)
            return

        os.makedirs(self.cache_dir, exist_ok=True)
        all_indices = list(range(len(self.ogb_dataset)))

        n_failed = 0
        for idx in tqdm(all_indices, desc=f"LapPE ({self.canonicalization})"):
            data = self.ogb_dataset[idx]
            edge_index_np = data.edge_index.numpy()
            num_nodes = int(data.num_nodes)

            pe, evals, ok = _compute_lappe_for_graph(
                edge_index_np,
                num_nodes,
                self.n_eigs,
                self.canonicalization,
                idx,
            )
            if not ok:
                n_failed += 1
            self._pe_data[idx] = (pe, evals)

        if n_failed > 0:
            print(f"  LapPE: {n_failed} graphs failed (using zero PE)")

        # Save cache (except for random_augmented which is non-deterministic)
        if self.canonicalization != "random_augmented":
            with open(cache_path, "wb") as f:
                pickle.dump(self._pe_data, f, protocol=4)

    def get_split_indices(self):
        """Return OGB split indices."""
        return self.split_dict

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Return a PyG Data object augmented with LapPE.

        Returns
        -------
        data : torch_geometric.data.Data with:
            - x : Tensor (N, atom_dim), atom features
            - x_pe : Tensor (N, n_eigs), LapPE features
            - edge_index : LongTensor (2, E)
            - y : Tensor (1, num_tasks), labels
        """
        graph_idx = self.indices[idx]
        data = self.ogb_dataset[graph_idx]

        pe, evals = self._pe_data[graph_idx]

        # For random_augmented, recompute with fresh random signs each time
        if self.canonicalization == "random_augmented":
            edge_index_np = data.edge_index.numpy()
            num_nodes = int(data.num_nodes)
            pe, evals, _ = _compute_lappe_for_graph(
                edge_index_np,
                num_nodes,
                self.n_eigs,
                "random_augmented",
                graph_idx,
            )

        # Build output Data object
        out = Data(
            x=data.x.float() if data.x is not None else torch.zeros(data.num_nodes, 9),
            x_pe=torch.from_numpy(pe).float(),
            edge_index=data.edge_index,
            y=data.y,
            graph_idx=torch.tensor([graph_idx], dtype=torch.long),
        )
        return out

    def get_dataloader(self, split, batch_size=32, shuffle=True, num_workers=0):
        """Create a DataLoader for the given split.

        Parameters
        ----------
        split : str
            ``"train"``, ``"valid"``, or ``"test"``.
        batch_size : int
        shuffle : bool
        num_workers : int

        Returns
        -------
        DataLoader
        """
        split_indices = self.split_dict[split].numpy().tolist()
        ds = _SplitView(self, split_indices, split=split)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def _get_by_graph_idx(self, graph_idx, augment=True):
        """Get a single graph by its OGB index.

        Parameters
        ----------
        graph_idx : int
        augment : bool
            If True and method is ``random_augmented``, recompute PE with
            fresh random signs. If False, return the cached (pre-computed) PE.
        """
        data = self.ogb_dataset[graph_idx]
        pe, evals = self._pe_data[graph_idx]

        if self.canonicalization == "random_augmented" and augment:
            edge_index_np = data.edge_index.numpy()
            num_nodes = int(data.num_nodes)
            pe, evals, _ = _compute_lappe_for_graph(
                edge_index_np,
                num_nodes,
                self.n_eigs,
                "random_augmented",
                graph_idx,
            )

        out = Data(
            x=data.x.float() if data.x is not None else torch.zeros(data.num_nodes, 9),
            x_pe=torch.from_numpy(pe).float(),
            edge_index=data.edge_index,
            y=data.y,
            graph_idx=torch.tensor([graph_idx], dtype=torch.long),
        )
        return out
