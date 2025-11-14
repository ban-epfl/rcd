"""Shared utility functions used across the Recursive Causal Discovery package."""

from __future__ import annotations

from collections.abc import Callable

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

REMOVABLE_NOT_FOUND = -1

CiTest = Callable[[int, int, list[int], np.ndarray], bool]


def sanitize_data(data: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Convert ``data`` into a 2D NumPy array.

    Parameters
    ----------
    data : pandas.DataFrame or np.ndarray
        Input dataset with variables in columns.

    Returns
    -------
    np.ndarray
        Matrix representation of ``data``.

    Raises
    ------
    TypeError
        If ``data`` is neither a ``DataFrame`` nor ``np.ndarray``.
    ValueError
        If ``data`` is an ``np.ndarray`` but not two-dimensional.
    """

    if isinstance(data, pd.DataFrame):
        # Convert DataFrame to numpy matrix
        return data.values
    if isinstance(data, np.ndarray):
        # Check if the numpy array is 2-dimensional (matrix)
        if data.ndim == 2:
            return data
        raise ValueError("Input numpy array is not a matrix (2-dimensional).")
    raise TypeError("Input must be a pandas DataFrame or a numpy array.")


def get_clique_number(graph: nx.Graph) -> int:
    """Return the clique number of ``graph``."""

    maximum_clique = max(nx.find_cliques(graph), key=len)
    clique_number = len(maximum_clique)
    return clique_number


def f1_score_edges(
    true_graph: nx.Graph,
    est_graph: nx.Graph,
    return_only_f1: bool = True,
) -> float | tuple[float, float, float]:
    """Compute edge-level F1 score between ``true_graph`` and ``est_graph``."""

    edges = set(true_graph.edges())
    edges_est = set(est_graph.edges())

    # compute F1 score
    precision = len(edges.intersection(edges_est)) / len(edges_est) if len(edges_est) > 0 else 0
    recall = len(edges.intersection(edges_est)) / len(edges) if len(edges) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    if return_only_f1:
        return f1_score
    return precision, recall, f1_score


def sort_vars_by_mkb_size(markov_boundary_matrix: np.ndarray, var_idx_arr: np.ndarray) -> np.ndarray:
    """Return ``var_idx_arr`` sorted by ascending Markov-boundary size."""

    # sort variables by the size of their Markov boundary
    mb_size = np.sum(markov_boundary_matrix, axis=1)
    sort_indices = np.argsort(mb_size)
    sorted_var_idx = np.asarray(var_idx_arr, dtype=int)[sort_indices]
    return sorted_var_idx


def compute_mb(data: np.ndarray, ci_test: CiTest) -> np.ndarray:
    """Compute the symmetric Markov-boundary matrix via exhaustive CI tests."""

    num_vars = data.shape[1]
    markov_boundary_matrix = np.zeros((num_vars, num_vars), dtype=bool)

    for i in range(num_vars - 1):  # -1 because no need to check last variable
        for j in range(i + 1, num_vars):
            # Check whether variable i and j are independent of each other given the rest of the variables
            cond_set = list(set(range(num_vars)) - {i, j})
            if not ci_test(i, j, cond_set, data):
                markov_boundary_matrix[i, j] = True
                markov_boundary_matrix[j, i] = True

    return markov_boundary_matrix


def compute_mb_gaussian(data: np.ndarray, sig_level: float | None = None) -> np.ndarray:
    """Estimate Markov boundaries using Gaussian partial correlations."""

    num_samples, n = data.shape
    crr = np.corrcoef(data, rowvar=False)
    prec = np.linalg.pinv(crr)

    norm_vec = np.sqrt(np.diag(prec))
    mb_mat = np.abs(prec / norm_vec[:, None] / norm_vec[None, :])

    sig_level = 1 / n**2 if sig_level is None else sig_level

    thresh = np.tanh(stats.norm.ppf(1 - sig_level / 2) / np.sqrt(num_samples - n - 1))

    mb_mat = mb_mat > thresh

    # set diagonal to 0
    np.fill_diagonal(mb_mat, 0)

    return mb_mat


def update_markov_boundary_matrix(
    markov_boundary_matrix: np.ndarray,
    data_included_ci_test: Callable[[int, int, list[int]], bool],
    var: int,
    var_neighbors: np.ndarray,
    is_diamond_free: bool = False,
    skip_check: np.ndarray | None = None,
) -> None:
    """Update ``markov_boundary_matrix`` after removing ``var``.

    Parameters
    ----------
    markov_boundary_matrix : np.ndarray
        Symmetric boolean Markov-boundary matrix.
    data_included_ci_test : Callable[[int, int, list[int]], bool]
        Wrapper around the CI oracle with data bound already.
    var : int
        Variable being removed.
    var_neighbors : np.ndarray
        Indices of the neighbors of ``var``.
    is_diamond_free : bool, default False
        Set to ``True`` when applying the diamond-free sufficient condition.
    skip_check : np.ndarray, optional
        Boolean array tracking which variables' removability checks can be skipped.
    """

    var_markov_boundary = np.flatnonzero(markov_boundary_matrix[var])

    # Remove var from Markov boundaries
    markov_boundary_matrix[var_markov_boundary, var] = False
    markov_boundary_matrix[var, var_markov_boundary] = False

    if skip_check is not None:
        skip_check[var_markov_boundary] = False

    if is_diamond_free:
        if len(var_markov_boundary) > len(var_neighbors):
            # Sufficient condition for diamond-free graphs
            return

    # Find nodes whose co-parent status changes after removing var
    # Only remove Y from Markov boundary of Z iff var is their ONLY common child and they are NOT neighbors
    for ne_idx_y in range(len(var_neighbors) - 1):  # -1 because no need to check last variable and also symmetry
        for ne_idx_z in range(ne_idx_y + 1, len(var_neighbors)):
            var_y = var_neighbors[ne_idx_y]
            var_z = var_neighbors[ne_idx_z]

            # determine whether the markov boundary of var_y or var_z is smaller, and use the smaller one as cond_set
            var_y_markov_boundary = np.flatnonzero(markov_boundary_matrix[var_y])
            var_z_markov_boundary = np.flatnonzero(markov_boundary_matrix[var_z])
            if np.sum(markov_boundary_matrix[var_y]) < np.sum(markov_boundary_matrix[var_z]):
                cond_set = list(set(var_y_markov_boundary) - {var_z})
            else:
                cond_set = list(set(var_z_markov_boundary) - {var_y})

            if data_included_ci_test(var_y, var_z, cond_set):
                # Y and Z are co-parents and thus NOT neighbors
                markov_boundary_matrix[var_y, var_z] = False
                markov_boundary_matrix[var_z, var_y] = False

                if skip_check is not None:
                    skip_check[var_y] = False
                    skip_check[var_z] = False

