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
# MAP: Maximal Axis Projection (Ma et al., NeurIPS 2023)
# ---------------------------------------------------------------------------


def spectral_canonicalize_map(eigenvectors, eigenvalues, rtol=1e-3):
    """Canonicalize eigenvectors via Maximal Axis Projection (MAP).

    For each eigenspace (group of eigenvectors sharing an eigenvalue):
    - **Multiplicity 1**: flip sign so the entry with largest absolute value
      is positive (with deterministic tie-breaking by smallest index).
    - **Multiplicity m > 1**: greedily select m coordinate axes that have
      maximal projection onto the eigenspace, then build a canonical basis
      by projecting those axes into the eigenspace and orthonormalizing.

    Reference: Ma et al., "Laplacian Canonization: A Minimalist Approach to
    Sign and Basis Invariance", NeurIPS 2023.

    Parameters
    ----------
    eigenvectors : ndarray of shape (n, k)
    eigenvalues : ndarray of shape (k,)
    rtol : float
        Relative tolerance for eigenvalue grouping.

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

    mult_info = detect_eigenvalue_multiplicities(eigenvalues, rtol=rtol)
    group_indices = mult_info["group_indices"]

    # Process each eigenspace group
    groups = {}
    for j in range(k):
        g = group_indices[j]
        if g not in groups:
            groups[g] = []
        groups[g].append(j)

    for cols in groups.values():
        m = len(cols)
        U = eigenvectors[:, cols].copy()  # (n, m)

        if m == 1:
            # Multiplicity 1: maxabs sign flip
            v = U[:, 0]
            abs_v = np.abs(v)
            i_max = int(np.argmax(abs_v))
            if v[i_max] < 0:
                result[:, cols[0]] = -v
            else:
                result[:, cols[0]] = v
        else:
            # Multiplicity > 1: axis projection canonicalization
            result[:, cols] = _map_canonicalize_eigenspace(U)

    return result


def _map_canonicalize_eigenspace(U):
    """Canonicalize a degenerate eigenspace via maximal axis projection.

    Given U (n x m) whose columns span an eigenspace of multiplicity m,
    greedily selects m coordinate axes e_{i_1}, ..., e_{i_m} that have
    maximal projection onto the eigenspace, then builds a canonical ONB by
    projecting each axis into the remaining subspace and orthonormalizing.

    All operations use an ONB Q_U for the eigenspace, ensuring canonical
    vectors stay exactly within span(U).

    Parameters
    ----------
    U : ndarray of shape (n, m)

    Returns
    -------
    V_canon : ndarray of shape (n, m), canonical orthonormal basis.
    """
    n, m = U.shape

    # ONB for the eigenspace — all projections go through this
    Q_U, _ = np.linalg.qr(U, mode="reduced")  # (n, m)

    # We'll work in the m-dimensional coefficient space
    # A row of Q_U gives the coefficients of e_i projected into the eigenspace
    # ||proj_{eigenspace}(e_i)||^2 = sum_j Q_U[i,j]^2

    result = np.zeros((n, m))
    used_dirs = np.zeros((m, 0))  # directions already used (in coeff space)

    for col_idx in range(m):
        # Compute projection norms onto the remaining subspace
        if used_dirs.shape[1] > 0:
            # Remove used direction components from each row's coefficient vector
            Q_U_coeffs = Q_U.copy()
            for j in range(used_dirs.shape[1]):
                d = used_dirs[:, j]
                # For each row i: remove component along d
                dots = Q_U_coeffs @ d  # (n,)
                Q_U_coeffs -= np.outer(dots, d)
            proj_norms_sq = np.sum(Q_U_coeffs**2, axis=1)
        else:
            proj_norms_sq = np.sum(Q_U**2, axis=1)

        # Select axis with maximal remaining projection
        i_star = int(np.argmax(proj_norms_sq))

        # Get the coefficient vector for this axis in the eigenspace
        coeff = Q_U[i_star, :].copy()  # (m,)

        # Remove components along already-used directions (Gram-Schmidt in coeff space)
        for j in range(used_dirs.shape[1]):
            d = used_dirs[:, j]
            coeff -= np.dot(coeff, d) * d

        norm = np.linalg.norm(coeff)
        if norm < 1e-12:
            # Degenerate — shouldn't happen for correct eigenspace dimension
            break
        coeff /= norm

        # Convert back to n-dimensional space: v = Q_U @ coeff
        v = Q_U @ coeff

        # Sign convention: largest absolute entry positive
        i_max = int(np.argmax(np.abs(v)))
        if v[i_max] < 0:
            coeff = -coeff
            v = -v

        result[:, col_idx] = v
        if used_dirs.shape[1] > 0:
            used_dirs = np.column_stack([used_dirs, coeff])
        else:
            used_dirs = coeff.reshape(-1, 1)

    return result


# ---------------------------------------------------------------------------
# OAP: Orthogonalized Axis Projection (Ma et al., NeurIPS 2024)
# ---------------------------------------------------------------------------


def spectral_canonicalize_oap(eigenvectors, eigenvalues, rtol=1e-3):
    """Canonicalize eigenvectors via Orthogonalized Axis Projection (OAP).

    Improvement over MAP: after selecting axes and projecting into the
    eigenspace, applies Gram-Schmidt orthogonalization for a cleaner
    orthonormal basis. Also uses a more robust axis selection criterion.

    Reference: Ma et al., NeurIPS 2024.

    Parameters
    ----------
    eigenvectors : ndarray of shape (n, k)
    eigenvalues : ndarray of shape (k,)
    rtol : float
        Relative tolerance for eigenvalue grouping.

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

    mult_info = detect_eigenvalue_multiplicities(eigenvalues, rtol=rtol)
    group_indices = mult_info["group_indices"]

    # Process each eigenspace group
    groups = {}
    for j in range(k):
        g = group_indices[j]
        if g not in groups:
            groups[g] = []
        groups[g].append(j)

    for cols in groups.values():
        m = len(cols)
        U = eigenvectors[:, cols].copy()  # (n, m)

        if m == 1:
            # Multiplicity 1: maxabs sign flip (same as MAP)
            v = U[:, 0]
            abs_v = np.abs(v)
            i_max = int(np.argmax(abs_v))
            if v[i_max] < 0:
                result[:, cols[0]] = -v
            else:
                result[:, cols[0]] = v
        else:
            # Multiplicity > 1: OAP with Gram-Schmidt
            result[:, cols] = _oap_canonicalize_eigenspace(U)

    return result


def _oap_canonicalize_eigenspace(U):
    """Canonicalize a degenerate eigenspace via orthogonalized axis projection.

    Like MAP, selects m coordinate axes greedily by maximal projection onto
    the remaining eigenspace. Then performs Gram-Schmidt orthogonalization
    on the raw projections (in coefficient space) for improved numerical
    stability. All operations are done in the eigenspace coefficient space
    to guarantee the result spans exactly span(U).

    Parameters
    ----------
    U : ndarray of shape (n, m)

    Returns
    -------
    V_canon : ndarray of shape (n, m), canonical orthonormal basis.
    """
    n, m = U.shape

    # ONB for the eigenspace
    Q_U, _ = np.linalg.qr(U, mode="reduced")  # (n, m)

    # Step 1: Select m axes greedily and collect raw coefficient vectors
    raw_coeffs = []  # raw Q_U[i_star, :] before orthogonalization
    used_coeff_dirs = np.zeros((m, 0))

    for _ in range(m):
        # Compute remaining projection norms
        if used_coeff_dirs.shape[1] > 0:
            Q_U_rem = Q_U.copy()
            for j in range(used_coeff_dirs.shape[1]):
                d = used_coeff_dirs[:, j]
                dots = Q_U_rem @ d
                Q_U_rem -= np.outer(dots, d)
            proj_norms_sq = np.sum(Q_U_rem**2, axis=1)
        else:
            proj_norms_sq = np.sum(Q_U**2, axis=1)

        i_star = int(np.argmax(proj_norms_sq))

        # Raw projection coefficient: Q_U[i_star, :]
        coeff = Q_U[i_star, :].copy()
        raw_coeffs.append(coeff)

        # Track which direction we used (normalized, for next iteration)
        c_orth = coeff.copy()
        for j in range(used_coeff_dirs.shape[1]):
            d = used_coeff_dirs[:, j]
            c_orth -= np.dot(c_orth, d) * d
        norm_c = np.linalg.norm(c_orth)
        if norm_c > 1e-12:
            c_orth /= norm_c
            used_coeff_dirs = (
                np.column_stack([used_coeff_dirs, c_orth])
                if used_coeff_dirs.shape[1] > 0
                else c_orth.reshape(-1, 1)
            )

    # Step 2: Gram-Schmidt on raw coefficients
    result = np.zeros((n, m))
    orth_coeffs = np.zeros((m, 0))

    for j, raw_c in enumerate(raw_coeffs):
        c = raw_c.copy()
        # Subtract components along already-built directions
        for i in range(orth_coeffs.shape[1]):
            d = orth_coeffs[:, i]
            c -= np.dot(c, d) * d
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            # Fallback: find unused direction in eigenspace
            for k in range(m):
                e_k = np.zeros(m)
                e_k[k] = 1.0
                for i in range(orth_coeffs.shape[1]):
                    d = orth_coeffs[:, i]
                    e_k -= np.dot(e_k, d) * d
                if np.linalg.norm(e_k) > 1e-10:
                    c = e_k
                    norm_c = np.linalg.norm(c)
                    break
        c /= norm_c

        # Convert to n-dimensional space
        v = Q_U @ c

        # Sign convention: largest absolute entry positive
        i_max = int(np.argmax(np.abs(v)))
        if v[i_max] < 0:
            c = -c
            v = -v

        result[:, j] = v
        orth_coeffs = (
            np.column_stack([orth_coeffs, c]) if orth_coeffs.shape[1] > 0 else c.reshape(-1, 1)
        )

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
