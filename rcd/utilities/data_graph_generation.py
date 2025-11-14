"""Synthetic DAG and data generators used in the RCD experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def gen_er_dag_adj_mat(num_vars: int, edge_prob: float) -> np.ndarray:
    """Sample an Erdős–Rényi DAG adjacency matrix.

    Parameters
    ----------
    num_vars : int
        Number of vertices in the graph.
    edge_prob : float
        Probability of including a directed edge ``i -> j`` when ``i < j`` in
        the topological order.

    Returns
    -------
    np.ndarray
        Boolean adjacency matrix with shape ``(num_vars, num_vars)``.
    """

    # Generate a random upper triangular matrix
    arr = np.triu(np.random.rand(num_vars, num_vars), k=1)

    # Convert to adjacency matrix with probability p
    adj_mat = (arr > 1 - edge_prob).astype(int)

    # Generate a random permutation
    perm = np.random.permutation(num_vars)

    # Apply the permutation to rows and the corresponding columns
    adj_mat_perm = adj_mat[perm, :][:, perm]

    return adj_mat_perm.astype(bool)


def gen_gaussian_data(
    dag_adj_mat: np.ndarray,
    num_samples: int,
    return_numpy: bool = False,
) -> pd.DataFrame | np.ndarray:
    """Generate linear-Gaussian samples compatible with ``dag_adj_mat``.

    Parameters
    ----------
    dag_adj_mat : np.ndarray
        Boolean adjacency matrix of the DAG (parents along columns).
    num_samples : int
        Number of IID observations to draw.
    return_numpy : bool, default False
        When ``True`` return a NumPy array; otherwise return a ``DataFrame``.

    Returns
    -------
    pandas.DataFrame or np.ndarray
        Synthetic dataset whose columns align with the DAG variables.
    """

    n = dag_adj_mat.shape[1]
    noise = np.random.normal(size=(num_samples, n)) @ np.diag(0.7 + 0.5 * np.random.rand(n))
    B = dag_adj_mat.T * ((1 + 0.5 * np.random.rand(n)) * ((-1) ** (np.random.rand(n) > 0.5)))
    D = noise @ np.linalg.pinv(np.eye(n) - B.T)

    if return_numpy:
        return D
    return pd.DataFrame(D)

