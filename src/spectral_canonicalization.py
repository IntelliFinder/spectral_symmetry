"""Spielman-style spectral eigenvector canonicalization (simple spectrum).

Implements a principled algebraic approach to resolving eigenvector sign
ambiguity for graphs with distinct eigenvalues (multiplicity 1). The algorithm:

1. Partitions nodes into balanced blocks via absolute-value row signatures
   and GF(2) product-vector refinement (Phase A).
2. Uses the balanced blocks to define a canonical node ordering, then for
   each column picks the sign that makes the first significant entry (in
   block-canonical order) positive (Phase B).
3. Applies the sign vector to flip eigenvector columns (Phase C).

Columns with multiplicity > 1 are left unchanged (their ambiguity requires
orthogonal group handling, which is out of scope).

Reference: Spielman's partitioning algorithm, adapted from
``theory/simple_isomorphism.py``.
"""

import collections
from itertools import chain, combinations

import numpy as np

from src.spectral_core import detect_eigenvalue_multiplicities

# ---------------------------------------------------------------------------
# GF(2) linear algebra
# ---------------------------------------------------------------------------


def solve_z2_system(A_in, b_in):
    """Solve a linear system Ax = b over GF(2).

    Parameters
    ----------
    A_in : ndarray of shape (m, n)
        Coefficient matrix (entries 0/1).
    b_in : ndarray of shape (m,)
        Right-hand side vector (entries 0/1).

    Returns
    -------
    x : ndarray of shape (n,) or None
        A particular solution, or None if inconsistent.
    consistent : bool
        True if the system has at least one solution.
    """
    A = (np.asarray(A_in).copy() % 2).astype(int)
    b = (np.asarray(b_in).copy() % 2).astype(int)
    m, n = A.shape

    # Augmented matrix [A | b]
    Ab = np.hstack([A, b.reshape(-1, 1)])

    pivot_row = 0
    for j in range(n):
        if pivot_row >= m:
            break
        # Find pivot in column j
        i = pivot_row
        while i < m and Ab[i, j] == 0:
            i += 1
        if i < m:
            if i != pivot_row:
                Ab[[i, pivot_row], :] = Ab[[pivot_row, i], :]
            for k in range(m):
                if k != pivot_row and Ab[k, j] == 1:
                    Ab[k, :] = (Ab[k, :] + Ab[pivot_row, :]) % 2
            pivot_row += 1

    # Back-substitution
    x = np.zeros(n, dtype=int)
    for i in range(m - 1, -1, -1):
        if np.all(Ab[i, :n] == 0) and Ab[i, n] == 1:
            return None, False
        pivot_col = -1
        for j in range(n):
            if Ab[i, j] == 1:
                pivot_col = j
                break
        if pivot_col != -1:
            x[pivot_col] = Ab[i, n]
            for k in range(i):
                if Ab[k, pivot_col] == 1:
                    Ab[k, n] = (Ab[k, n] + x[pivot_col]) % 2
                    Ab[k, pivot_col] = 0

    # Consistency check
    rank_A = np.sum(np.any(Ab[:, :n], axis=1))
    rank_Ab = np.sum(np.any(Ab, axis=1))
    if rank_A < rank_Ab:
        return None, False

    return x, True


def _gf2_row_reduce(M):
    """Compute the reduced row echelon form of M over GF(2).

    Parameters
    ----------
    M : ndarray of shape (m, n), entries in {0, 1}.

    Returns
    -------
    R : ndarray of shape (m, n), RREF over GF(2).
    """
    R = (np.asarray(M).copy() % 2).astype(int)
    m, n = R.shape
    pivot_row = 0
    for j in range(n):
        if pivot_row >= m:
            break
        i = pivot_row
        while i < m and R[i, j] == 0:
            i += 1
        if i < m:
            if i != pivot_row:
                R[[i, pivot_row], :] = R[[pivot_row, i], :]
            for k in range(m):
                if k != pivot_row and R[k, j] == 1:
                    R[k, :] = (R[k, :] + R[pivot_row, :]) % 2
            pivot_row += 1
    return R


def _gf2_null_space(A, n_cols):
    """Compute a basis for the null space of A over GF(2).

    Parameters
    ----------
    A : ndarray of shape (m, n_cols), entries in {0, 1}.
    n_cols : int
        Number of columns (variables).

    Returns
    -------
    basis : ndarray of shape (dim_kernel, n_cols), rows are basis vectors.
    """
    R = _gf2_row_reduce(A)
    m, n = R.shape

    # Identify pivot columns
    pivot_cols = []
    for i in range(m):
        for j in range(n):
            if R[i, j] == 1:
                pivot_cols.append(j)
                break

    free_cols = sorted(set(range(n)) - set(pivot_cols))
    if not free_cols:
        return np.zeros((0, n_cols), dtype=int)

    # Build basis vectors: for each free column, set it to 1 and solve
    basis = []
    for fc in free_cols:
        vec = np.zeros(n, dtype=int)
        vec[fc] = 1
        # For each pivot row, determine value from the free columns
        for i, pc in enumerate(pivot_cols):
            if i < m:
                # Row i has pivot at pc: R[i, pc] = 1
                # x[pc] = R[i, rhs_bits] but we use the free columns
                val = 0
                for fj in free_cols:
                    if R[i, fj] == 1 and fj == fc:
                        val = (val + 1) % 2
                vec[pc] = val
        basis.append(vec)

    return np.array(basis, dtype=int) if basis else np.zeros((0, n_cols), dtype=int)


def _lex_smallest_in_coset(s0, ker_basis, k):
    """Find the lexicographically smallest vector in the coset s0 + ker(A) over GF(2).

    Parameters
    ----------
    s0 : ndarray of shape (k,), a particular solution.
    ker_basis : ndarray of shape (d, k), basis for the kernel.
    k : int
        Vector dimension.

    Returns
    -------
    s_min : ndarray of shape (k,), lex-smallest element.
    """
    s = s0.copy() % 2
    if ker_basis.shape[0] == 0:
        return s

    # Put kernel basis in RREF
    rref = _gf2_row_reduce(ker_basis)
    # Remove zero rows
    nonzero = np.any(rref, axis=1)
    rref = rref[nonzero]

    # Greedily XOR basis vectors to minimize leading bits
    for row in rref:
        # Find the leading 1 position of this basis vector
        lead = -1
        for j in range(k):
            if row[j] == 1:
                lead = j
                break
        if lead == -1:
            continue
        # If s has a 1 at the leading position, XOR with this basis vector
        if s[lead] == 1:
            s = (s + row) % 2

    return s


# ---------------------------------------------------------------------------
# Phase A: Find Balanced Blocks
# ---------------------------------------------------------------------------


def _get_abs_row_signature(row, precision):
    """Get a hashable absolute-value signature for a row vector.

    Parameters
    ----------
    row : ndarray of shape (k,)
    precision : int

    Returns
    -------
    tuple of floats, rounded absolute values.
    """
    return tuple(np.round(np.abs(row), precision))


def _powerset(iterable):
    """Return all subsets of iterable as a list of tuples."""
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))


def _get_product_balance(product_vector, precision):
    """Check if a product vector is unbalanced.

    A product vector is unbalanced if for some non-zero value x,
    the count of +x differs from the count of -x.

    Returns
    -------
    bool : True if unbalanced.
    """
    counts = collections.Counter(np.round(product_vector, precision))
    abs_values = set(np.abs(list(counts.keys())))

    if len(abs_values) <= 1:
        return False

    for x in abs_values:
        if x == 0.0:
            continue
        if counts.get(x, 0) != counts.get(-x, 0):
            return True
    return False


def _find_dependency(eigenvectors, psi_i_on_C, Q, C_list):
    """Check if eigenvector column psi_i is GF(2)-dependent on base Q for class C.

    Parameters
    ----------
    eigenvectors : ndarray of shape (n, k)
    psi_i_on_C : ndarray of shape (|C|,)
    Q : list of int, column indices forming the current base.
    C_list : list of int, node indices in the class.

    Returns
    -------
    tuple (gamma, y) if dependent, None otherwise.
    """
    B_Q = (eigenvectors[np.ix_(C_list, Q)] < 0).astype(int)
    t_i = (psi_i_on_C < 0).astype(int)

    # Try gamma = 0 (no global flip)
    y_0, ok_0 = solve_z2_system(B_Q, t_i)
    if ok_0:
        pred = (B_Q @ y_0) % 2
        if np.all(pred == t_i):
            return (0, y_0)

    # Try gamma = 1 (global flip)
    target_1 = (t_i + 1) % 2
    y_1, ok_1 = solve_z2_system(B_Q, target_1)
    if ok_1:
        pred = (B_Q @ y_1) % 2
        if np.all(pred == target_1):
            return (1, y_1)

    return None


def _process_class(eigenvectors, C, precision):
    """Process a single class to check if it's balanced.

    Parameters
    ----------
    eigenvectors : ndarray of shape (n, k)
    C : frozenset of int, node indices.
    precision : int

    Returns
    -------
    status : str, "balanced" or "split"
    payload : ndarray or None, the splitting product vector if "split".
    """
    C_list = sorted(C)
    n_cols = eigenvectors.shape[1]

    Q = []  # GF(2)-independent base column indices

    for col_idx in range(n_cols):
        psi_i_on_C = eigenvectors[C_list, col_idx]

        # Check dependency on current base
        if Q:
            dep = _find_dependency(eigenvectors, psi_i_on_C, Q, C_list)
            if dep is not None:
                continue

        # Independent — add to base
        Q.append(col_idx)

        # Check all subsets involving the new column for unbalanced products
        subsets_without_new = _powerset(Q[:-1])
        for R in subsets_without_new:
            S = list(R) + [col_idx]
            product_vector = np.prod(eigenvectors[np.ix_(C_list, S)], axis=1)

            if _get_product_balance(product_vector, precision):
                return "split", product_vector

    return "balanced", None


def find_balanced_blocks(eigenvectors, precision=8):
    """Partition nodes into balanced blocks (Phase A).

    Parameters
    ----------
    eigenvectors : ndarray of shape (n, k)
    precision : int

    Returns
    -------
    list of frozenset, each containing node indices of a balanced block.
    """
    n = eigenvectors.shape[0]

    # Initial partition by absolute-value row signature
    initial_partition = {}
    for i in range(n):
        sig = _get_abs_row_signature(eigenvectors[i, :], precision)
        if sig not in initial_partition:
            initial_partition[sig] = []
        initial_partition[sig].append(i)

    work_queue = collections.deque([frozenset(indices) for indices in initial_partition.values()])
    balanced_blocks = []

    while work_queue:
        C = work_queue.popleft()

        status, payload = _process_class(eigenvectors, C, precision)

        if status == "balanced":
            balanced_blocks.append(C)
        elif status == "split":
            product_vector = payload
            C_list = sorted(C)
            new_classes = {}
            for idx, node in enumerate(C_list):
                val = np.round(product_vector[idx], precision)
                if val not in new_classes:
                    new_classes[val] = []
                new_classes[val].append(node)
            for vertices in new_classes.values():
                if vertices:
                    work_queue.append(frozenset(vertices))

    return balanced_blocks


# ---------------------------------------------------------------------------
# Phase B: Canonical Sign Vector
# ---------------------------------------------------------------------------


def compute_canonical_signs(eigenvectors, blocks, precision=8):
    """Compute the canonical sign vector from balanced blocks (Phase B).

    Uses the Spielman balanced blocks to define a canonical node ordering,
    then for each column determines the sign by finding the first node
    (in block-canonical order) with a significant non-zero entry and
    ensuring that entry is positive.

    Block-canonical order: blocks sorted lexicographically by absolute-value
    signature (then by node indices for sub-blocks with identical signatures);
    nodes within each block sorted by index.

    This is sign-invariant because:
    - Blocks depend only on absolute values (invariant under column sign flips)
    - The first-non-zero-entry rule flips consistently with the column sign

    Parameters
    ----------
    eigenvectors : ndarray of shape (n, k)
    blocks : list of frozenset, balanced blocks from Phase A.
    precision : int

    Returns
    -------
    signs : ndarray of shape (k,), entries in {-1, +1}.
    """
    k = eigenvectors.shape[1]
    threshold = 10 ** (-precision)

    # Build canonical node ordering from blocks
    def block_sort_key(block):
        nodes = sorted(block)
        representative = nodes[0]
        abs_sig = tuple(np.round(np.abs(eigenvectors[representative, :]), precision))
        return (abs_sig, tuple(nodes))

    sorted_blocks = sorted(blocks, key=block_sort_key)

    canonical_order = []
    for block in sorted_blocks:
        canonical_order.extend(sorted(block))

    # For each column, find first non-zero entry in canonical order
    signs = np.ones(k, dtype=int)
    for j in range(k):
        for node in canonical_order:
            if abs(eigenvectors[node, j]) > threshold:
                if eigenvectors[node, j] < 0:
                    signs[j] = -1
                break

    return signs


# ---------------------------------------------------------------------------
# Phase C: Apply + Main Entry Point
# ---------------------------------------------------------------------------


def spectral_canonicalize(eigenvectors, eigenvalues, precision=8):
    """Canonicalize eigenvector signs using Spielman-style partitioning.

    For eigenvalues with multiplicity 1, applies the algebraic
    canonicalization algorithm. Columns corresponding to repeated
    eigenvalues are left unchanged.

    Parameters
    ----------
    eigenvectors : ndarray of shape (n, k)
    eigenvalues : ndarray of shape (k,)
    precision : int
        Decimal precision for rounding comparisons.

    Returns
    -------
    canonicalized : ndarray of shape (n, k), sign-canonicalized copy.
    """
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    eigenvalues = np.asarray(eigenvalues, dtype=float)

    if eigenvectors.size == 0:
        return eigenvectors.copy()

    n, k = eigenvectors.shape

    # Detect multiplicities
    mult_info = detect_eigenvalue_multiplicities(eigenvalues)
    multiplicity = mult_info["multiplicity"]

    # Find columns with multiplicity == 1
    simple_cols = [j for j in range(k) if multiplicity[j] == 1]

    if not simple_cols:
        # All columns have multiplicity > 1 — nothing to canonicalize
        return eigenvectors.copy()

    # Extract the simple-spectrum sub-matrix
    V_simple = eigenvectors[:, simple_cols].copy()

    # Phase A: find balanced blocks
    blocks = find_balanced_blocks(V_simple, precision=precision)

    # Phase B: compute canonical signs
    signs = compute_canonical_signs(V_simple, blocks, precision=precision)

    # Phase C: apply signs
    result = eigenvectors.copy()
    for idx, col in enumerate(simple_cols):
        result[:, col] = eigenvectors[:, col] * signs[idx]

    return result


# ---------------------------------------------------------------------------
# MAP / OAP shared helpers (faithful numpy ports of the reference torch code)
# ---------------------------------------------------------------------------


def _group_eigenvalues_by_rounding(eigenvalues, decimals):
    """Group eigenvalues by rounding to a fixed number of decimal places.

    Matches the reference MAP/OAP eigenvalue grouping exactly:
    ``E.round(decimals=decimals)`` then ``unique(..., return_counts=True)``.

    Parameters
    ----------
    eigenvalues : ndarray of shape (k,)
    decimals : int

    Returns
    -------
    ind : ndarray of shape (g+1,), boundary indices for each group.
    mult : ndarray of shape (g,), multiplicity of each group.
    """
    rounded = np.round(eigenvalues, decimals=decimals)
    _, mult = np.unique(rounded, return_counts=True)
    ind = np.concatenate([[0], np.cumsum(mult)])
    return ind, mult


def _hash_column(col, i):
    """Hash a column vector with a distinguished index (OAP grouping).

    Faithful port of ``hash_tensor`` from the reference ``oap.py``:
    ``float('%.5g' % (sum(col**3) + col[i]))``.
    """
    return float("%.5g" % (np.sum(col**3) + col[i]))


def _orthogonalize_qr(U):
    """Orthogonalize via plain QR (MAP style, no sign correction)."""
    Q, _R = np.linalg.qr(U)
    return Q


def _orthogonalize_qr_signed(U):
    """Orthogonalize via QR with sign correction (OAP style).

    Returns ``Q * sign(diag(R))``.
    """
    Q, R = np.linalg.qr(U)
    S = np.sign(np.diag(R))
    return Q * S


def _find_complementary_space(U, u_span, orthogonalize_fn):
    """Find the orthogonal complementary space of u_span in span(U).

    Faithful port of ``find_complementary_space`` from the reference.

    Parameters
    ----------
    U : ndarray of shape (n, d)
    u_span : ndarray of shape (n, s), s <= d
    orthogonalize_fn : callable
        ``_orthogonalize_qr`` for MAP, ``_orthogonalize_qr_signed`` for OAP.

    Returns
    -------
    u_perp : ndarray of shape (n, d - s)
    """
    n, d = U.shape
    s = u_span.shape[1]
    u_base = u_span.copy()
    for j in range(d):
        i = u_base.shape[1]
        u_j = U[:, j : j + 1]  # (n, 1)
        u_temp = np.concatenate([u_base, u_j], axis=1)
        if np.linalg.matrix_rank(u_temp) == i + 1:
            u_base = u_temp
        if u_base.shape[1] == d:
            break
    u_base = orthogonalize_fn(u_base)
    return u_base[:, s:d]


# ---------------------------------------------------------------------------
# MAP sign disambiguation (faithful port of unique_sign from map.py)
# ---------------------------------------------------------------------------


def _map_sign_disambiguate(U):
    """Eliminate sign ambiguity using projection-based grouping (MAP).

    For each column u, forms P = u u^T, groups coordinates by
    ||P[:, i]|| (rounded to 14 decimals), builds test vectors from those
    groups, and picks the sign from the first non-zero projection.

    Parameters
    ----------
    U : ndarray of shape (n, d)

    Returns
    -------
    U_out : ndarray of shape (n, d)
    """
    n, d = U.shape
    U = U.copy()
    for i in range(d):
        u = U[:, i].reshape(n, 1)
        P = u @ u.T  # (n, n)
        E = np.eye(n)
        J = np.ones(n)
        Pe = [np.round(np.linalg.norm(P[:, j]), decimals=14) for j in range(n)]
        Pe = list(enumerate(Pe))
        Pe.sort(key=lambda x: x[1])
        indices = [x[0] for x in Pe]
        lengths = [x[1] for x in Pe]
        _, counts = np.unique(lengths, return_counts=True)
        step = 0
        X = np.zeros((len(counts), n))
        for j in range(len(counts)):
            for _ in range(counts[j]):
                X[j] += E[indices[step]]
                step += 1
            X[j] += 10 * J
        flag = True
        u_0 = np.zeros(n)
        for j in range(len(counts)):
            u_0 = P @ X[j]
            if np.round(np.linalg.norm(u_0), decimals=12) != 0:
                x = X[j]
                flag = False
                break
        if flag:
            continue
        u_flat = u.ravel()
        u_0 /= np.abs(u_flat @ x)
        U[:, i] = u_0
    return U


# ---------------------------------------------------------------------------
# OAP sign disambiguation (faithful port of oap_sign from oap.py)
# ---------------------------------------------------------------------------


def _oap_sign_disambiguate(U):
    """Eliminate sign ambiguity using projection-based grouping (OAP).

    Same algorithm as MAP sign but rounds norms to 6 decimals and
    zero-check to 6 decimals.

    Parameters
    ----------
    U : ndarray of shape (n, d)

    Returns
    -------
    U_out : ndarray of shape (n, d)
    """
    n, d = U.shape
    U = U.copy()
    for i in range(d):
        u = U[:, i].reshape(n, 1)
        P = u @ u.T
        E = np.eye(n)
        J = np.ones(n)
        Pe = [np.round(np.linalg.norm(P[:, j]), decimals=6) for j in range(n)]
        Pe = list(enumerate(Pe))
        Pe.sort(key=lambda x: x[1])
        indices = [x[0] for x in Pe]
        lengths = [x[1] for x in Pe]
        _, counts = np.unique(lengths, return_counts=True)
        step = 0
        X = np.zeros((len(counts), n))
        for j in range(len(counts)):
            for _ in range(counts[j]):
                X[j] += E[indices[step]]
                step += 1
            X[j] += 10 * J
        flag = True
        u_0 = np.zeros(n)
        for j in range(len(counts)):
            u_0 = P @ X[j]
            if np.round(np.linalg.norm(u_0), decimals=6) != 0:
                x = X[j]
                flag = False
                break
        if flag:
            continue
        u_flat = u.ravel()
        u_0 /= np.abs(u_flat @ x)
        U[:, i] = u_0
    return U


# ---------------------------------------------------------------------------
# MAP basis disambiguation (faithful port of unique_basis from map.py)
# ---------------------------------------------------------------------------


def _map_canonicalize_eigenspace(U_i):
    """Eliminate basis ambiguity for a degenerate eigenspace (MAP).

    Uses projection matrix P = U U^T, groups coordinates by ||P[:, i]||
    (rounded to 14 decimals), selects d groups from largest norm down,
    and incrementally builds a canonical basis via complementary space
    projection.

    Parameters
    ----------
    U_i : ndarray of shape (n, d)

    Returns
    -------
    U_0 : ndarray of shape (n, d), canonical orthonormal basis.

    Raises
    ------
    AssertionError
        If basis assumptions are violated.
    """
    n, d = U_i.shape
    E = np.eye(n)
    J = np.ones(n)
    P = U_i @ U_i.T
    Pe = [np.round(np.linalg.norm(P[:, i]), decimals=14) for i in range(n)]
    Pe = list(enumerate(Pe))
    Pe.sort(key=lambda x: x[1])
    indices = [x[0] for x in Pe]
    lengths = [x[1] for x in Pe]
    _, counts = np.unique(lengths, return_counts=True)
    assert len(counts) >= d  # basis assumption 1
    X = np.zeros((d, n))
    step = -1
    for i in range(1, d + 1):
        x = np.zeros(n)
        for _ in range(counts[-i]):
            x += E[indices[step]]
            step -= 1
        X[i - 1] = x + 10 * J
    U_0 = np.zeros((n, d))
    u_span = np.empty((n, 0))
    u_perp = U_i.copy()
    for i in range(d):
        P_perp = u_perp @ u_perp.T
        u_i = P_perp @ X[i]
        assert np.linalg.norm(u_i) != 0  # basis assumption 2
        u_i = u_i / np.linalg.norm(u_i)
        U_0[:, i] = u_i
        u_span = np.concatenate([u_span, u_i.reshape(n, 1)], axis=1)
        u_perp = _find_complementary_space(U_i, u_span, _orthogonalize_qr)
    return U_0


# ---------------------------------------------------------------------------
# OAP basis disambiguation (faithful port of oap_basis from oap.py)
# ---------------------------------------------------------------------------


def _oap_canonicalize_eigenspace(U_i):
    """Eliminate basis ambiguity for a degenerate eigenspace (OAP).

    Uses projection matrix P = U U^T, groups coordinates by hash
    (``_hash_column``), greedily collects d linearly independent
    projections, and orthogonalizes with QR + sign correction.

    Parameters
    ----------
    U_i : ndarray of shape (n, d)

    Returns
    -------
    U_0 : ndarray of shape (n, d), canonical orthonormal basis.

    Raises
    ------
    AssertionError
        If basis assumptions are violated.
    """
    n, d = U_i.shape
    E = np.eye(n)
    J = np.ones(n)
    P = U_i @ U_i.T
    Pe = [_hash_column(P[:, i], i) for i in range(n)]
    Pe = list(enumerate(Pe))
    Pe.sort(key=lambda x: x[1])
    indices = [x[0] for x in Pe]
    lengths = [x[1] for x in Pe]
    _, counts = np.unique(lengths, return_counts=True)
    k = len(counts)
    assert k >= d
    X = np.zeros((k, n))
    step = -1
    for i in range(1, k + 1):
        x = np.zeros(n)
        for _ in range(counts[-i]):
            x += E[indices[step]]
            step -= 1
        X[i - 1] = x + 10 * J
    u_span = np.empty((n, 0))
    current_rank = 0
    for i in range(k):
        u_i = P @ X[i]
        norm_u = np.linalg.norm(u_i)
        if norm_u < 1e-15:
            continue
        u_i = u_i / norm_u
        u_span_tmp = np.concatenate([u_span, u_i.reshape(n, 1)], axis=1)
        if np.linalg.matrix_rank(u_span_tmp) == current_rank + 1:
            u_span = u_span_tmp
            current_rank += 1
            if current_rank == d:
                break
    assert current_rank == d
    U_0 = _orthogonalize_qr_signed(u_span)
    return U_0


# ---------------------------------------------------------------------------
# MAP: Maximal Axis Projection (Ma et al., NeurIPS 2023)
# ---------------------------------------------------------------------------


def spectral_canonicalize_map(eigenvectors, eigenvalues):
    """Canonicalize eigenvectors via Maximal Axis Projection (MAP).

    Faithful numpy port of the reference implementation from
    PKU-ML/canonicalization (map.py). Uses projection-matrix-based sign
    disambiguation (mult-1) and complementary-space basis construction
    (mult > 1) for permutation-equivariant canonicalization.

    Eigenvalue grouping uses ``round(decimals=14)`` matching the reference.

    Reference: Ma et al., "Laplacian Canonization: A Minimalist Approach to
    Sign and Basis Invariance", NeurIPS 2023.

    Parameters
    ----------
    eigenvectors : ndarray of shape (n, k)
    eigenvalues : ndarray of shape (k,)

    Returns
    -------
    canonicalized : ndarray of shape (n, k)
    """
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    eigenvalues = np.asarray(eigenvalues, dtype=float)

    if eigenvectors.size == 0:
        return eigenvectors.copy()

    n, k = eigenvectors.shape
    result = eigenvectors.copy()

    ind, mult = _group_eigenvalues_by_rounding(eigenvalues, decimals=14)

    for i in range(len(mult)):
        cols = slice(int(ind[i]), int(ind[i + 1]))
        U = result[:, cols].copy()
        if mult[i] == 1:
            result[:, cols] = _map_sign_disambiguate(U)
        else:
            try:
                result[:, cols] = _map_canonicalize_eigenspace(U)
            except AssertionError:
                continue

    return result


# ---------------------------------------------------------------------------
# OAP: Orthogonalized Axis Projection (Ma et al., NeurIPS 2024)
# ---------------------------------------------------------------------------


def spectral_canonicalize_oap(eigenvectors, eigenvalues):
    """Canonicalize eigenvectors via Orthogonalized Axis Projection (OAP).

    Faithful numpy port of the reference implementation from
    PKU-ML/canonicalization (oap.py). Uses projection-matrix-based sign
    disambiguation (mult-1) and hash-based greedy basis construction
    with QR+sign orthogonalization (mult > 1).

    Eigenvalue grouping uses ``round(decimals=6)`` matching the reference.

    Reference: Ma et al., NeurIPS 2024.

    Parameters
    ----------
    eigenvectors : ndarray of shape (n, k)
    eigenvalues : ndarray of shape (k,)

    Returns
    -------
    canonicalized : ndarray of shape (n, k)
    """
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    eigenvalues = np.asarray(eigenvalues, dtype=float)

    if eigenvectors.size == 0:
        return eigenvectors.copy()

    n, k = eigenvectors.shape
    result = eigenvectors.copy()

    ind, mult = _group_eigenvalues_by_rounding(eigenvalues, decimals=6)

    for i in range(len(mult)):
        cols = slice(int(ind[i]), int(ind[i + 1]))
        U = result[:, cols].copy()
        if mult[i] == 1:
            result[:, cols] = _oap_sign_disambiguate(U)
        else:
            try:
                result[:, cols] = _oap_canonicalize_eigenspace(U)
            except AssertionError:
                continue

    return result


# ---------------------------------------------------------------------------
# Unified canonicalization dispatcher
# ---------------------------------------------------------------------------

CANONICALIZATION_METHODS = (
    "spielman",
    "maxabs",
    "random_fixed",
    "random_augmented",
    "map",
    "oap",
    "none",
)


def canonicalize_maxabs(eigenvectors):
    """Max-absolute-value sign convention with unique-tie check.

    For each eigenvector column, find the entry with the largest absolute
    value. If that entry is unique (no other entry has the same absolute
    value), flip the eigenvector so that entry is positive. If the max
    absolute value is not unique, leave the eigenvector unchanged.

    Parameters
    ----------
    eigenvectors : ndarray of shape (N, k)

    Returns
    -------
    ndarray of shape (N, k), sign-canonicalized copy.
    """
    eigenvectors = np.asarray(eigenvectors, dtype=float).copy()
    for j in range(eigenvectors.shape[1]):
        col = eigenvectors[:, j]
        abs_col = np.abs(col)
        max_abs = abs_col.max()
        mask = np.isclose(abs_col, max_abs, rtol=1e-8, atol=1e-12)
        if mask.sum() == 1:
            idx = np.argmax(abs_col)
            if col[idx] < 0:
                eigenvectors[:, j] = -col
    return eigenvectors


def canonicalize_random_fixed(eigenvectors, sample_idx):
    """Deterministic random sign flips seeded by sample index.

    Parameters
    ----------
    eigenvectors : ndarray of shape (N, k)
    sample_idx : int
        Index used for seeding (e.g. graph/shape index in dataset).

    Returns
    -------
    ndarray of shape (N, k)
    """
    result = np.asarray(eigenvectors, dtype=float).copy()
    for j in range(result.shape[1]):
        rng = np.random.RandomState(seed=sample_idx * 1000 + j)
        if rng.random() < 0.5:
            result[:, j] = -result[:, j]
    return result


def canonicalize_random_augmented(eigenvectors):
    """Non-deterministic random sign flips (data augmentation).

    Parameters
    ----------
    eigenvectors : ndarray of shape (N, k)

    Returns
    -------
    ndarray of shape (N, k)
    """
    result = np.asarray(eigenvectors, dtype=float).copy()
    for j in range(result.shape[1]):
        if np.random.random() < 0.5:
            result[:, j] = -result[:, j]
    return result


def canonicalize(eigenvectors, eigenvalues=None, method="maxabs", sample_idx=0):
    """Unified dispatcher for eigenvector canonicalization.

    Routes to the appropriate canonicalization implementation based on
    ``method``.

    Parameters
    ----------
    eigenvectors : ndarray of shape (N, k)
    eigenvalues : ndarray of shape (k,) or None
        Required for ``spielman``, ``map``, and ``oap`` methods.
    method : str
        One of :data:`CANONICALIZATION_METHODS`.
    sample_idx : int
        Index used for seeding ``random_fixed`` method.

    Returns
    -------
    ndarray of shape (N, k), canonicalized copy.

    Raises
    ------
    ValueError
        If ``method`` is unknown or ``eigenvalues`` is missing for methods
        that require it.
    """
    if method not in CANONICALIZATION_METHODS:
        raise ValueError(
            f"Unknown canonicalization method: {method!r}. Choose from {CANONICALIZATION_METHODS}"
        )

    needs_eigenvalues = ("spielman", "map", "oap")
    if method in needs_eigenvalues and eigenvalues is None:
        raise ValueError(f"eigenvalues required for method={method!r}")

    if method == "spielman":
        return spectral_canonicalize(eigenvectors, eigenvalues)
    elif method == "maxabs":
        return canonicalize_maxabs(eigenvectors)
    elif method == "random_fixed":
        return canonicalize_random_fixed(eigenvectors, sample_idx)
    elif method == "random_augmented":
        return canonicalize_random_augmented(eigenvectors)
    elif method == "map":
        return spectral_canonicalize_map(eigenvectors, eigenvalues)
    elif method == "oap":
        return spectral_canonicalize_oap(eigenvectors, eigenvalues)
    elif method == "none":
        return np.asarray(eigenvectors, dtype=float).copy()
