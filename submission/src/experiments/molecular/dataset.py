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
import tempfile

import filelock
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.spectral_canonicalization import (
    CANONICALIZATION_METHODS,
    canonicalize,
    scale_eigenvectors_by_eigenvalues,
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

    # Apply canonicalization (random_augmented stores raw eigvecs; signs applied at runtime)
    if method != "random_augmented":
        eigenvectors = _apply_canonicalization(eigenvectors, eigenvalues, method, graph_idx)

    n_actual = eigenvectors.shape[1]
    pe[lcc_indices, :n_actual] = eigenvectors.astype(np.float32)
    evals_out[:n_actual] = eigenvalues[:n_actual].astype(np.float32)

    return pe, evals_out, True


class _SplitView:
    """Lightweight view into a MolecularLapPEDataset for a specific split."""

    def __init__(self, parent, indices, split="train", augment_override=None, base_seed=None):
        self.parent = parent
        self.indices = indices
        self.split = split
        # augment_override: if set, override the split-based default. Used at
        # test time for aug-averaged evaluation of random_augmented models.
        self.augment_override = augment_override
        # base_seed: if set, each __getitem__ call derives a seeded rng from
        # base_seed and the graph_idx (reproducible augmentation draws).
        self.base_seed = base_seed

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        graph_idx = self.indices[idx]
        if self.augment_override is not None:
            augment = self.augment_override
        else:
            augment = self.split == "train"
        if self.base_seed is None:
            rng = None
        else:
            # Mix base_seed with graph_idx so different graphs see different
            # random draws while the combination (base_seed, graph_idx) is
            # fully reproducible.
            mixed = (int(self.base_seed) * 2_654_435_761 + int(graph_idx)) & 0xFFFFFFFF
            rng = np.random.default_rng(mixed)
        return self.parent._get_by_graph_idx(graph_idx, augment=augment, rng=rng)


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
        cache_n_eigs=None,
        eigval_scale=False,
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
        self.eigval_scale = eigval_scale

        # cache_n_eigs: compute/cache this many eigenvectors, then slice to n_eigs
        # at runtime. Avoids redundant preprocessing when sweeping k values.
        self._cache_k = cache_n_eigs if cache_n_eigs is not None else n_eigs

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

        # Cache directory — keyed on cache_k so all k<=cache_k share the same cache.
        # Exception: spielman needs per-k caches because its block structure depends
        # on which k eigenvectors are considered together.
        if cache_dir is None:
            cache_k_for_path = (
                n_eigs if canonicalization in ("spielman", "spielman_partition") else self._cache_k
            )
            cache_dir = os.path.join(
                data_dir, "lappe_cache", f"{dataset_name}_{canonicalization}_k{cache_k_for_path}"
            )
        self.cache_dir = cache_dir

        # Precompute or load LapPE
        self._pe_data = {}  # graph_idx -> (pe, eigenvalues)
        self._precompute_lappe()

    def _cache_path(self):
        return os.path.join(self.cache_dir, "lappe.pkl")

    def _base_cache_dir(self):
        """Return the directory for the raw (uncanonicalized) eigenvector cache."""
        lappe_dir = os.path.dirname(self.cache_dir)
        return os.path.join(lappe_dir, f"{self.dataset_name}_raw_k{self._cache_k}")

    def _base_cache_path(self):
        return os.path.join(self._base_cache_dir(), "lappe.pkl")

    def _ensure_base_cache(self):
        """Ensure the raw eigenvector base cache exists, computing it if needed.

        Returns the base cache data dict: {graph_idx: (pe, evals)}.
        """
        base_dir = self._base_cache_dir()
        base_path = self._base_cache_path()

        # Fast path: base cache already on disk
        if os.path.exists(base_path):
            with open(base_path, "rb") as f:
                return pickle.load(f)

        os.makedirs(base_dir, exist_ok=True)
        lock_path = os.path.join(base_dir, "lappe.lock")

        with filelock.FileLock(lock_path, timeout=7200):
            # Re-check inside lock — another process may have computed it
            if os.path.exists(base_path):
                with open(base_path, "rb") as f:
                    return pickle.load(f)

            all_indices = list(range(len(self.ogb_dataset)))
            base_data = {}

            n_failed = 0
            for idx in tqdm(all_indices, desc="LapPE (raw eigdecomp)"):
                data = self.ogb_dataset[idx]
                edge_index_np = data.edge_index.numpy()
                num_nodes = int(data.num_nodes)

                pe, evals, ok = _compute_lappe_for_graph(
                    edge_index_np,
                    num_nodes,
                    self._cache_k,
                    "none",
                    idx,
                )
                if not ok:
                    n_failed += 1
                base_data[idx] = (pe, evals)

            if n_failed > 0:
                print(f"  LapPE (raw): {n_failed} graphs failed (using zero PE)")

            # Atomic write
            fd, tmp_path = tempfile.mkstemp(dir=base_dir, suffix=".pkl")
            try:
                with os.fdopen(fd, "wb") as f:
                    pickle.dump(base_data, f, protocol=4)
                os.replace(tmp_path, base_path)
            except BaseException:
                os.unlink(tmp_path)
                raise

            return base_data

    def _precompute_lappe(self):
        """Compute LapPE for all graphs using two-stage caching.

        Stage 1 (base cache): Raw eigenvectors computed once and shared across
        all canonicalization methods.

        Stage 2 (canon cache): Canonicalized eigenvectors derived from the base
        cache.  For ``"none"`` and ``"random_augmented"`` the canon cache IS the
        base cache (no extra work).

        Uses file locking to prevent race conditions when multiple processes
        initialize the same cache concurrently (e.g. multi-GPU ablations).
        """
        cache_path = self._cache_path()

        # Step 1: Fast path — canon cache already exists
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self._pe_data = pickle.load(f)
            return

        # Step 2: Ensure the base (raw) cache exists
        base_data = self._ensure_base_cache()

        # Step 3: For "none" and "random_augmented", the canon cache is the
        # base cache itself (random_augmented applies signs at runtime).
        if self.canonicalization in ("none", "random_augmented"):
            self._pe_data = base_data
            # Symlink or copy the canon cache so future loads hit Step 1
            os.makedirs(self.cache_dir, exist_ok=True)
            lock_path = os.path.join(self.cache_dir, "lappe.lock")
            with filelock.FileLock(lock_path, timeout=7200):
                if not os.path.exists(cache_path):
                    fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, suffix=".pkl")
                    try:
                        with os.fdopen(fd, "wb") as f:
                            pickle.dump(base_data, f, protocol=4)
                        os.replace(tmp_path, cache_path)
                    except BaseException:
                        os.unlink(tmp_path)
                        raise
            return

        # Step 4: Apply canonicalization to each graph's eigvecs from base cache
        os.makedirs(self.cache_dir, exist_ok=True)
        lock_path = os.path.join(self.cache_dir, "lappe.lock")

        with filelock.FileLock(lock_path, timeout=7200):
            # Re-check inside lock
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    self._pe_data = pickle.load(f)
                return

            canon_data = {}
            method = self.canonicalization

            for idx in tqdm(base_data, desc=f"LapPE canonicalize ({method})"):
                pe, evals = base_data[idx]

                # Find LCC nodes (non-zero rows in pe)
                row_nonzero = np.any(pe != 0, axis=1)
                n_actual_eigs = int(np.sum(evals != 0))

                if not np.any(row_nonzero) or n_actual_eigs == 0:
                    # No valid eigenvectors — keep zeros
                    canon_data[idx] = (pe, evals)
                    continue

                lcc_indices = np.where(row_nonzero)[0]

                # For spielman, slice to n_eigs BEFORE canonicalizing —
                # block structure depends on which k eigenvectors are present.
                if method == "spielman":
                    n_use = min(n_actual_eigs, self.n_eigs)
                else:
                    n_use = n_actual_eigs
                raw_eigvecs = pe[lcc_indices, :n_use]
                raw_evals = evals[:n_use]

                # Apply canonicalization
                canon_eigvecs = _apply_canonicalization(raw_eigvecs, raw_evals, method, idx)

                # Write back into full pe array
                pe_canon = pe.copy()
                pe_canon[lcc_indices, :n_use] = canon_eigvecs.astype(np.float32)
                canon_data[idx] = (pe_canon, evals)

            self._pe_data = canon_data

            # Atomic write
            fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, suffix=".pkl")
            try:
                with os.fdopen(fd, "wb") as f:
                    pickle.dump(canon_data, f, protocol=4)
                os.replace(tmp_path, cache_path)
            except BaseException:
                os.unlink(tmp_path)
                raise

    def get_split_indices(self):
        """Return OGB split indices."""
        return self.split_dict

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Return a PyG Data object augmented with LapPE.

        Delegates to :meth:`_get_by_graph_idx`. When accessed directly
        (not via :class:`_SplitView`), augmentation is enabled by default.
        """
        graph_idx = self.indices[idx]
        return self._get_by_graph_idx(graph_idx, augment=True)

    def get_dataloader(self, split, batch_size=32, shuffle=True, num_workers=0, **kwargs):
        """Create a DataLoader for the given split.

        Parameters
        ----------
        split : str
            ``"train"``, ``"valid"``, or ``"test"``.
        batch_size : int
        shuffle : bool
        num_workers : int
        **kwargs
            Additional keyword arguments passed to ``DataLoader``
            (e.g. ``pin_memory``, ``worker_init_fn``).

        Returns
        -------
        DataLoader
        """
        split_indices = self.split_dict[split].numpy().tolist()
        ds = _SplitView(self, split_indices, split=split)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs,
        )

    def get_augmented_test_loader(
        self, split, batch_size=32, base_seed=0, num_workers=0, **kwargs
    ):
        """Create a non-shuffling DataLoader that applies the random_augmented
        ambiguity group to every sample (both signs and O(m) rotations),
        seeded reproducibly by ``base_seed``.

        Intended for test-time averaged evaluation of random_augmented models.
        Always returns deterministic output for a given ``base_seed``.

        Parameters
        ----------
        split : str
            ``"train"``, ``"valid"``, or ``"test"``.
        batch_size : int
        base_seed : int
            Seed that mixes with each graph_idx to produce a reproducible
            augmentation draw.
        num_workers : int
        **kwargs
            Forwarded to ``DataLoader``.

        Returns
        -------
        DataLoader
        """
        split_indices = self.split_dict[split].numpy().tolist()
        ds = _SplitView(
            self,
            split_indices,
            split=split,
            augment_override=True,
            base_seed=base_seed,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            **kwargs,
        )

    def _get_by_graph_idx(self, graph_idx, augment=True, rng=None):
        """Get a single graph by its OGB index.

        Parameters
        ----------
        graph_idx : int
        augment : bool
            If True and method is ``random_augmented``, recompute PE with
            fresh random signs + Haar O(m) rotation per multiplicity block.
            If False, return the cached (pre-computed) PE.
        rng : numpy.random.Generator or None
            Reproducible random generator for the augmentation draw. ``None``
            falls back to ``np.random`` for the legacy training loop.
        """
        data = self.ogb_dataset[graph_idx]
        pe, evals = self._pe_data[graph_idx]

        # Truncate to n_eigs if cache has more columns
        k = self.n_eigs
        pe = pe[:, :k]
        evals = evals[:k]

        # For random_augmented, apply a random element of the eigenvector
        # ambiguity group: sign flips on simple eigenvalues + Haar O(m)
        # rotations on multiplicity-m blocks.
        if self.canonicalization == "random_augmented" and augment:
            from ...spectral_canonicalization import random_augment_eigenvectors

            pe = random_augment_eigenvectors(pe, eigenvalues=evals, rng=rng).astype(np.float32)

        # Scale eigenvectors by 1/sqrt(eigenvalue) if requested
        if self.eigval_scale:
            pe = scale_eigenvectors_by_eigenvalues(pe, evals).astype(np.float32)

        out = Data(
            x=data.x.float() if data.x is not None else torch.zeros(data.num_nodes, 9),
            x_pe=torch.from_numpy(pe).float(),
            x_evals=torch.from_numpy(evals).float().unsqueeze(0).expand(data.num_nodes, -1),
            edge_index=data.edge_index,
            y=data.y,
            graph_idx=torch.tensor([graph_idx], dtype=torch.long),
        )
        return out
